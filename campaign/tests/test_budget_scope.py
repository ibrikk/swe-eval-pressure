#!/usr/bin/env python3
"""Offline tests for the SHARD-SCOPED budget gate.

NO Harbor job, NO model call, NO network, NO gateway probe. Every fixture is a
temporary directory holding synthetic JSON; `lib.probe_budget` is mocked.

The situation these tests freeze is the real one on 2026-09-03. FULL shard 1 is
complete (1,200/1,200 accepted). The gateway has $5,999.14 left. The whole
remaining campaign needs more than that, so the old whole-campaign gate refused
EVERY command -- including `run-shard full 2`, which is comfortably affordable
on its own and which the cell-level controller can resume if it is interrupted.

  1  whole remaining campaign does not fit      -> standalone preflight REFUSES
  2  selected FULL shard 2 does fit             -> scoped gate PASSES
  3  a passing scoped gate still WARNS that the whole campaign does not fit
  4  repair-shard gates on its repair plan ONLY, not the shard's 300 cells
  5  a scope prices only its own cells (no leakage from other shards/modes)
  6  a shard the budget CANNOT cover is still refused (exhaustion unchanged)
  7  both safety factors survive scoping: 20% contingency, then 10% margin
  8  campaign.sh scopes run-shard/repair-shard and NOTHING else
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from campaign import lib, preflight  # noqa: E402

# One flat price for every cell, so the arithmetic in each assertion is
# checkable by hand and a wrong SCOPE cannot hide behind a per-profile mean.
MEAN_USD_PER_TRIAL = 2.00
REMAINING_BUDGET = 5_999.14      # the gateway's real remaining balance

# Whole remaining campaign with FULL shard 1 complete:
#   full s2 1,200 + full s3 400 + resource 840 = 2,440 trials
WHOLE_TRIALS = 2_440
FULL_S2_TRIALS = 1_200
REPAIR_SUBSET_PER_PROFILE = 6    # 24 cells across the four profiles


def planning(trials: int) -> float:
    return trials * MEAN_USD_PER_TRIAL * (1 + preflight.PLANNING_CONTINGENCY)


def required(trials: int) -> float:
    return planning(trials) * (1 + preflight.SAFETY_MARGIN)


def fake_estimate() -> dict:
    return {
        "cells": {f"{m}/{p}": {"mean_usd_per_trial": MEAN_USD_PER_TRIAL}
                  for m in lib.MODES for p in lib.PROFILES},
        # Below any remaining balance used here, so the pre-existing p90 warning
        # never fires and cannot be mistaken for the new whole-campaign one.
        "p90_total_usd": 1.0,
    }


def fake_budget(remaining: float = REMAINING_BUDGET):
    return lib.BudgetStatus(
        ok=True, spend=4_000.86, max_budget=remaining + 4_000.86,
        remaining=remaining, tpm_limit="2000000", rpm_limit="10000",
        http_status=200, key_fingerprint="sha256:test")


class BudgetFixture:
    """A provenance directory holding only what `remaining_trials` reads."""

    def __init__(self):
        self._td = tempfile.mkdtemp(prefix="campaign-budget-scope-")
        self.root = Path(self._td)
        (self.root / "provenance").mkdir(parents=True)
        self.paths = {"root": self.root, "provenance": self.root / "provenance"}

    def write_plan(self, mode: str, shard: int, per_profile: int) -> Path:
        """A repair plan naming `per_profile` outstanding cells per profile."""
        plan = {
            "campaign_id": lib.CAMPAIGN_ID,
            "mode": mode,
            "shard": shard,
            "by_profile": {
                p: [{"cell_key": f"{lib.CAMPAIGN_ID}/{mode}/{p}/s{shard}/task-{i}/eval-src"}
                    for i in range(per_profile)]
                for p in lib.PROFILES
            },
        }
        out = self.paths["provenance"] / f"{mode}-shard{shard}-repair-plan.json"
        out.write_text(json.dumps(plan, indent=1), encoding="utf-8")
        return out

    def cleanup(self):
        import shutil
        shutil.rmtree(self._td, ignore_errors=True)


class TestShardScopedBudgetGate(unittest.TestCase):
    """FULL shard 1 complete; the campaign does not fit but shard 2 does."""

    def setUp(self):
        self.fx = BudgetFixture()
        self.addCleanup(self.fx.cleanup)
        # FULL shard 1 is finished: its repair plan is empty, so it is priced at
        # zero outstanding trials. Nothing else has a plan, so every other
        # profile/shard falls through to its full "not_started" trial count.
        self.fx.write_plan("full", 1, 0)
        self.est = fake_estimate()

    def check(self, scope=None, remaining=REMAINING_BUDGET):
        with mock.patch.object(lib, "probe_budget", return_value=fake_budget(remaining)):
            return preflight.check_budget(self.fx.paths, self.est, "k", scope=scope)

    def scope_full_2(self):
        return preflight.BudgetScope("FULL shard 2 outstanding inference work",
                                     mode="full", shard=2)

    # -- 1 ------------------------------------------------------------------
    def test_whole_remaining_campaign_does_not_fit(self):
        """The premise. Unscoped, this is a refusal -- as it always was."""
        with self.assertRaises(preflight.Fail) as cm:
            self.check(scope=None)
        msg = str(cm.exception)
        self.assertIn("INSUFFICIENT BUDGET", msg)
        self.assertIn("whole remaining campaign", msg)
        self.assertIn(f"${required(WHOLE_TRIALS):,.2f}", msg)
        self.assertGreater(required(WHOLE_TRIALS), REMAINING_BUDGET)

    # -- 2 ------------------------------------------------------------------
    def test_selected_full_shard_2_fits_and_passes(self):
        notes = self.check(scope=self.scope_full_2())
        blob = "\n".join(notes)
        self.assertIn(f"{FULL_S2_TRIALS:,} trials still requiring inference", blob)
        self.assertIn(f"planning cost   : ${planning(FULL_S2_TRIALS):,.2f}", blob)
        self.assertIn(f"required        : ${required(FULL_S2_TRIALS):,.2f}", blob)
        self.assertIn("gating on       : FULL shard 2", blob)
        self.assertLess(required(FULL_S2_TRIALS), REMAINING_BUDGET)

    # -- 3 ------------------------------------------------------------------
    def test_passing_scoped_gate_still_warns_about_the_whole_campaign(self):
        """The operator must not lose sight of the number that did not gate."""
        blob = "\n".join(self.check(scope=self.scope_full_2()))
        self.assertIn("WARNING: the WHOLE remaining campaign does NOT fit", blob)
        self.assertIn(f"${required(WHOLE_TRIALS):,.2f}", blob)
        short = required(WHOLE_TRIALS) - REMAINING_BUDGET
        self.assertIn(f"short by ${short:,.2f}", blob)
        self.assertIn(f"whole campaign  : ${planning(WHOLE_TRIALS):,.2f} planning", blob)

    # -- 4 ------------------------------------------------------------------
    def test_repair_shard_gates_only_on_its_repair_plan_subset(self):
        """A 24-cell repair costs 24 trials, not the shard's 1,200."""
        self.fx.write_plan("full", 2, REPAIR_SUBSET_PER_PROFILE)
        subset = REPAIR_SUBSET_PER_PROFILE * len(lib.PROFILES)
        notes = self.check(scope=self.scope_full_2())
        blob = "\n".join(notes)
        self.assertIn(f"cells pending   : {len(lib.PROFILES)}  ({subset} trials", blob)
        self.assertIn(f"required        : ${required(subset):,.2f}", blob)
        # And emphatically NOT the unrepaired shard's full price.
        self.assertNotIn(f"${required(FULL_S2_TRIALS):,.2f}", blob)

    # -- 5 ------------------------------------------------------------------
    def test_a_scope_prices_only_its_own_cells(self):
        _, pending, trials = preflight.remaining_campaign_cost(
            self.fx.paths, self.est, scope=self.scope_full_2())
        self.assertEqual(trials, FULL_S2_TRIALS)
        self.assertEqual(len(pending), len(lib.PROFILES))
        for line in pending:
            self.assertTrue(line.startswith("full/"), line)
            self.assertIn("chunk-2-", line)
        # The unscoped call is unchanged and still sees the whole campaign.
        _, whole_pending, whole_trials = preflight.remaining_campaign_cost(
            self.fx.paths, self.est)
        self.assertEqual(whole_trials, WHOLE_TRIALS)
        self.assertGreater(len(whole_pending), len(pending))

    # -- 6 ------------------------------------------------------------------
    def test_a_shard_the_budget_cannot_cover_is_still_refused(self):
        """Scoping narrows WHAT is priced. It never makes the gate optional."""
        thin = required(FULL_S2_TRIALS) - 0.01
        with self.assertRaises(preflight.Fail) as cm:
            self.check(scope=self.scope_full_2(), remaining=thin)
        self.assertIn("INSUFFICIENT BUDGET", str(cm.exception))
        self.assertIn("short by $0.01", str(cm.exception))

    # -- 7 ------------------------------------------------------------------
    def test_safety_factors_both_survive_a_scoped_gate(self):
        self.assertEqual(preflight.PLANNING_CONTINGENCY, 0.20)
        self.assertEqual(preflight.SAFETY_MARGIN, 0.10)
        need, _, trials = preflight.remaining_campaign_cost(
            self.fx.paths, self.est, scope=self.scope_full_2())
        # 20% contingency is inside the planning cost...
        self.assertAlmostEqual(need, trials * MEAN_USD_PER_TRIAL * 1.20, places=6)
        # ...and the 10% margin is charged on top of it, not instead of it.
        self.assertAlmostEqual(need * 1.10, required(trials), places=6)


class TestOperatorWiring(unittest.TestCase):
    """Section 8: only the two single-shard commands scope the budget."""

    SH = PROJECT_ROOT / "campaign.sh"

    def setUp(self):
        self.body = self.SH.read_text()

    def segment(self, start: str, end: str) -> str:
        seg = self.body[self.body.index(start):]
        return seg[:seg.index(end)] if end in seg else seg

    def test_campaign_sh_is_syntactically_valid(self):
        rc = subprocess.run(["bash", "-n", str(self.SH)], capture_output=True, text=True)
        self.assertEqual(rc.returncode, 0, rc.stderr)

    def test_run_shard_scopes_the_budget_to_its_own_shard(self):
        seg = self.segment("run_shard() {", "\n# ---")
        self.assertIn('PREFLIGHT_BUDGET_SCOPE=(--budget-scope-mode "$mode" '
                      '--budget-scope-shard "$shard")', seg)
        self.assertIn("do_preflight", seg)

    def test_repair_shard_scopes_the_budget_to_its_own_shard(self):
        seg = self.segment("repair_shard() {", "\nrun_mode() {")
        self.assertIn('PREFLIGHT_BUDGET_SCOPE=(--budget-scope-mode "$mode" '
                      '--budget-scope-shard "$shard")', seg)

    def test_run_mode_and_standalone_preflight_stay_whole_campaign(self):
        """run-full / run-resource launch three shards without stopping, so
        they keep the whole-campaign gate; standalone preflight never scopes."""
        seg = self.segment("run_mode() {", "\n# ---")
        self.assertNotIn("PREFLIGHT_BUDGET_SCOPE=(", seg)
        self.assertIn("do_preflight standalone ;;", self.body)
        self.assertEqual(self.body.count("PREFLIGHT_BUDGET_SCOPE=(--budget-scope-mode"), 2)

    def test_structural_checks_are_never_scoped(self):
        """The budget scope must not narrow integrity/amendment hashing."""
        self.assertNotIn('PREFLIGHT_ARGS=(--mode', self.body)
        self.assertIn('PREFLIGHT_ARGS=("$@")', self.body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
