#!/usr/bin/env python3
"""Offline tests for REPAIR-ATTEMPT provenance accounting.

NO Harbor job, NO model call, NO network, NO trial is launched anywhere in this
file. Every fixture is synthetic JSON in a temp directory.

THE BUG THESE TESTS PIN
-----------------------
`campaign.sh record_attempts()` sends both run-shard and repair-shard through
`campaign.provenance record --status auto`. The old "auto" rule completed an
attempt only when it produced the WHOLE profile/shard cell:

    complete iff accepted == cell.expected_trials      # 300 for FULL shard 1

A repair invocation only ever produces the cells its plan listed. So the
2026-09-02 FULL shard-1 repair, which executed every one of its 214 planned
cells (claude 103/103, fable 3/3, codex 5/5, llama 103/103) and left the shard
at 1,200/1,200 accepted observations, was recorded as four FAILED attempts with
`observed=103 expected=300` -- and campaign.sh died FATAL on a repair that had
succeeded.

Coverage:
  1  the whole-cell rule reproduces the bug (regression guard on the old rule)
  2  END-TO-END: 197 valid + a 103-cell repair plan -> repair complete,
     300/300 claude cells, wrapper exits 0
  3  a repair that misses a planned cell is still FAILED
  4  a repair attempt never supersedes the full-cell attempt it complements
  5  a complete repair attempt is not promoted to a whole-cell acceptance
  6  reconcile corrects mislabelled records WITHOUT deleting history
  7  reconcile is a dry run unless --apply is passed
  8  cell closure states original + blocked + repair = expected
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from campaign import cells, lib, provenance
from campaign.tests.test_repair import Fixture, relocate_campaign_root

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The real FULL shard-1 claude cell: 30 base tasks x 10 arms = 300.
FULL_ARMS = tuple(lib.ARMS["full"])
CELL_TRIALS = 300
ORIGINAL_VALID = 197
REPAIR_CELLS = 103


class RepairProvenanceCase(unittest.TestCase):
    """A real-shape FULL shard-1 claude cell, split 197 original / 103 repair."""

    maxDiff = None

    def setUp(self):
        # .resolve(): on macOS /var is a symlink to /private/var, and the
        # ledger stores run dirs relative to PROJECT_ROOT after resolving.
        self._td = Path(tempfile.mkdtemp(prefix="campaign-repair-prov-")).resolve()
        self.addCleanup(shutil.rmtree, self._td, ignore_errors=True)
        # The campaign namespace guard is NOT weakened: it is pointed at the
        # fixture root and still refuses everything outside it.
        self.root = self._td / "results" / "campaigns" / lib.CAMPAIGN_ID
        self.root.mkdir(parents=True, exist_ok=True)
        relocate_campaign_root(self, self.root)
        pp = mock.patch.object(lib, "PROJECT_ROOT", self._td)
        pp.start()
        self.addCleanup(pp.stop)

        self.fx = Fixture(self.root, mode="full", shard=1, profile="claude",
                          base_tasks=30, arms=FULL_ARMS)
        self.cell = lib.Cell("full", "claude", 1)
        self.assertEqual(self.cell.expected_trials, CELL_TRIALS)
        # `planned` is the canonical (base_index, arm) order the fixture lays
        # cells out in; the first ORIGINAL_VALID ran, the rest did not.
        self.planned = [(i, a) for i in range(30) for a in FULL_ARMS]
        self.assertEqual(len(self.planned), CELL_TRIALS)

    # -- fixture construction ---------------------------------------------- #
    def original_run(self, n=ORIGINAL_VALID):
        """The interrupted original attempt: n valid trajectories, then nothing."""
        run = self.fx.run_dir("job-original")
        for i, arm in self.planned[:n]:
            self.fx.add_trial(run, i, arm)
        return run.parent

    def repair_run(self, cells_to_run, *, name="rep01"):
        run = self.fx.run_dir(name)
        for i, arm in cells_to_run:
            self.fx.add_trial(run, i, arm, suffix="REP1")
        return run.parent

    def cell_key(self, base_index, arm):
        return cells.CellKey(lib.CAMPAIGN_ID, "full", "claude", 1,
                             self.fx.base_ids[base_index], arm).key

    def write_repair_plan(self, outstanding):
        """A frozen repair plan naming exactly the outstanding cells."""
        plan = {
            "campaign_id": lib.CAMPAIGN_ID,
            "mode": "full", "shard": 1,
            "generated_at": lib.now_iso(),
            "expected": CELL_TRIALS,
            "repair_required": len(outstanding),
            "by_profile": {"claude": [
                {"cell_key": self.cell_key(i, arm),
                 "base_task_id": self.fx.base_ids[i], "arm": arm,
                 "status": cells.MISSING, "reason": "no observation on disk"}
                for i, arm in outstanding]},
        }
        path = self.root / "provenance" / "full-shard1-repair-plan.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(plan, indent=2) + "\n")
        return path

    def record(self, run_dirs, *, repair_plan=None, status="auto", note=""):
        args = mock.Mock(mode="full", profile="claude", shard=1,
                         run_dir=[str(d) for d in run_dirs], status=status,
                         repair_plan=(str(repair_plan) if repair_plan else None),
                         started_at=None, finished_at=None, note=note)
        rc = provenance.cmd_record(args)
        return rc, self.attempts()[-1]

    def attempts(self):
        path = self.root / "provenance" / "attempts.jsonl"
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

    def audit(self):
        return cells.audit("full", 1, profiles=["claude"])


# --------------------------------------------------------------------------- #
# 1 - the old rule, pinned as the bug it was
# --------------------------------------------------------------------------- #
class TestWholeCellRuleIsWrongForRepair(RepairProvenanceCase):

    def test_01_whole_cell_rule_fails_a_perfect_repair(self):
        """Without a repair plan, a 103/103 repair is judged against 300.

        This is the exact shape of the four bad records: the controller did
        everything it was asked to do and the ledger says `failed`.
        """
        self.original_run()
        outstanding = self.planned[ORIGINAL_VALID:]
        self.assertEqual(len(outstanding), REPAIR_CELLS)
        repair = self.repair_run(outstanding)

        rc, entry = self.record([repair])   # no --repair-plan: the old path
        self.assertEqual(rc, 1)
        self.assertEqual(entry["status"], "failed")
        self.assertEqual(entry["observed_trials"], REPAIR_CELLS)
        self.assertEqual(entry["expected_trials"], CELL_TRIALS,
                         "the old rule measures a repair against the whole cell")
        self.assertEqual(entry["attempt_kind"], provenance.ATTEMPT_KIND_FULL)

        # ...while the shard itself is in fact complete.
        v = cells.validate_shard_complete(self.audit())
        self.assertEqual(v["accepted_observations"], CELL_TRIALS)


# --------------------------------------------------------------------------- #
# 2 - the integration regression test
# --------------------------------------------------------------------------- #
class TestRepairAttemptAccounting(RepairProvenanceCase):

    def test_02_end_to_end_197_original_plus_103_repair(self):
        """The whole contract, in the shape the FULL shard-1 repair had.

        original shard has 197 valid claude cells
        repair plan has 103 claude cells
        repair executes 103/103
        provenance marks the repair attempt complete
        final cell audit has 300/300 claude observations
        the wrapper exits 0
        """
        original = self.original_run()
        outstanding = self.planned[ORIGINAL_VALID:]

        # -- the original attempt: valid work, but not a complete cell -------
        rc_orig, orig = self.record([original])
        self.assertEqual(rc_orig, 1, "an interrupted cell is not complete")
        self.assertEqual(orig["status"], "failed")
        self.assertEqual(orig["observed_trials"], ORIGINAL_VALID)

        pre = self.audit()
        self.assertEqual(pre["counts"][cells.COMPLETE_VALID], ORIGINAL_VALID,
                         "original shard has 197 valid claude cells")
        self.assertEqual(pre["repair_required"], REPAIR_CELLS)

        # -- the plan: exactly the 103 outstanding cells ---------------------
        plan_path = self.write_repair_plan(outstanding)
        plan = provenance.load_repair_plan(plan_path)
        planned = provenance.planned_repair_cells(plan, "claude")
        self.assertEqual(len(planned), REPAIR_CELLS,
                         "repair plan has 103 claude cells")
        frozen = set(cells.frozen_cells(pre))
        self.assertFalse(set(planned) & frozen,
                         "a COMPLETE_VALID cell must never enter a repair plan")

        # -- the repair executes all 103 -------------------------------------
        repair = self.repair_run(outstanding)
        rc_rep, rep = self.record([repair], repair_plan=plan_path,
                                  note="campaign.sh repair-shard full shard 1")

        self.assertEqual(rc_rep, 0, "a complete repair must return 0")
        self.assertEqual(rep["status"], "complete",
                         "provenance marks the repair attempt complete")
        self.assertEqual(rep["attempt_kind"], provenance.ATTEMPT_KIND_REPAIR)

        # Requirement 3: the explicit provenance a repair attempt must carry.
        self.assertEqual(rep["planned_repair_cells"], REPAIR_CELLS)
        self.assertEqual(rep["observed_repair_cells"], REPAIR_CELLS)
        self.assertEqual(rep["expected_repair_cells"], REPAIR_CELLS)
        self.assertEqual(rep["repair_plan_basis"], provenance.PLAN_BASIS_FILE)
        self.assertEqual(Path(rep["repair_plan"]).name, plan_path.name)
        self.assertEqual(rep["repair_plan_sha256"], lib.sha256_file(plan_path))
        self.assertEqual(sorted(rep["planned_repair_cell_keys"]), sorted(planned))
        self.assertEqual(rep["outstanding_repair_cell_keys"], [])
        # ...and it does NOT demand the 300 cells of the profile/shard.
        self.assertEqual(rep["expected_trials"], REPAIR_CELLS)
        self.assertEqual(rep["profile_shard_expected_trials"], CELL_TRIALS)
        self.assertIn(orig["attempt_id"], rep["complements_attempt_ids"])

        # -- the shard, decided at CELL level --------------------------------
        post = self.audit()
        v = cells.validate_shard_complete(post)
        self.assertEqual(post["counts"][cells.COMPLETE_VALID], CELL_TRIALS,
                         "final cell audit has 300/300 claude observations")
        self.assertEqual(v["accepted_observations"], CELL_TRIALS)
        self.assertEqual(v["missing"], 0)
        self.assertEqual(v["outstanding_total"], 0)
        self.assertTrue(v["ok"], v["problems"])

        # -- the wrapper's own gate ------------------------------------------
        self.assertEqual(self.record_attempts_rc(plan_path), 0,
                         "wrapper exits 0")

    def record_attempts_rc(self, plan_path):
        """Run campaign.sh's record_attempts() body for the repair profile.

        This is the exact call the wrapper makes -- the one whose non-zero
        return produced the FATAL. Executed in-process against the fixture, so
        no Harbor and no campaign.sh preflight is involved.
        """
        repair = self.root / "full"
        dirs = [d for d in sorted(repair.iterdir()) if "rep01" in d.name]
        args = mock.Mock(mode="full", profile="claude", shard=1,
                         run_dir=[str(d) for d in dirs], status="auto",
                         repair_plan=str(plan_path), started_at=None,
                         finished_at=None,
                         note="campaign.sh repair-shard full shard 1")
        return provenance.cmd_record(args)

    def test_03_repair_that_misses_a_planned_cell_is_failed(self):
        """The rule must still be able to say no."""
        self.original_run()
        outstanding = self.planned[ORIGINAL_VALID:]
        plan_path = self.write_repair_plan(outstanding)
        repair = self.repair_run(outstanding[:-1])   # one planned cell skipped

        rc, rep = self.record([repair], repair_plan=plan_path)
        self.assertEqual(rc, 1)
        self.assertEqual(rep["status"], "failed")
        self.assertEqual(rep["planned_repair_cells"], REPAIR_CELLS)
        self.assertEqual(rep["observed_repair_cells"], REPAIR_CELLS - 1)
        self.assertEqual(len(rep["outstanding_repair_cell_keys"]), 1)
        self.assertEqual(cells.validate_shard_complete(self.audit())["missing"], 1)

    def test_04_repair_never_supersedes_the_attempt_it_complements(self):
        """The original holds 197 cells nothing else is evidence for."""
        original = self.original_run()
        outstanding = self.planned[ORIGINAL_VALID:]
        plan_path = self.write_repair_plan(outstanding)
        _rc, orig = self.record([original])
        self.repair_run(outstanding)
        rc, rep = self.record([self.root / "full" / f"rep01-full-claude-{self.cell.shard_label}-b01"],
                              repair_plan=plan_path)
        self.assertEqual(rc, 0)

        after = {a["attempt_id"]: a for a in self.attempts()}
        self.assertIsNone(after[orig["attempt_id"]]["superseded_by"],
                          "a partial repair must not supersede the original")
        self.assertEqual(after[orig["attempt_id"]]["status"], "failed")
        self.assertEqual(len(after), 2, "history is preserved, not replaced")

    def test_05_complete_repair_is_not_a_whole_cell_acceptance(self):
        """`accepted` means one attempt covers all 300. A repair covers 103."""
        self.original_run()
        outstanding = self.planned[ORIGINAL_VALID:]
        plan_path = self.write_repair_plan(outstanding)
        self.repair_run(outstanding)
        self.record([self.root / "full" / f"rep01-full-claude-{self.cell.shard_label}-b01"],
                    repair_plan=plan_path)

        doc = json.loads((self.root / "provenance" / "accepted_runs.json").read_text())
        self.assertEqual(doc["accepted"], [],
                         "a 103-cell repair must not be sold as a 300-cell cell")
        self.assertEqual(len(doc["accepted_repair_attempts"]), 1)
        self.assertEqual(doc["repair_cells_completed"], REPAIR_CELLS)
        self.assertNotIn(self.cell.key, cells.accepted_cell_keys())


# --------------------------------------------------------------------------- #
# 6-8 - reconcile
# --------------------------------------------------------------------------- #
class TestReconcile(RepairProvenanceCase):

    def mislabelled(self):
        """Reproduce a repair recorded by the OLD whole-cell rule."""
        self.original_run()
        outstanding = self.planned[ORIGINAL_VALID:]
        self.repair_run(outstanding)
        _rc, bad = self.record(
            [self.root / "full" / f"rep01-full-claude-{self.cell.shard_label}-b01"],
            note="campaign.sh repair-shard full shard 1")
        self.assertEqual(bad["status"], "failed")
        return bad

    def reconcile(self, apply=False, plan=None):
        args = mock.Mock(mode="full", shard=1, attempt_id=None,
                         profile=["claude"],
                         source_repair_plan=(str(plan) if plan else None),
                         apply=apply)
        return provenance.cmd_reconcile(args)

    def test_06_reconcile_corrects_without_deleting_history(self):
        bad = self.mislabelled()
        self.assertEqual(self.reconcile(apply=True), 0)

        by_id = {a["attempt_id"]: a for a in self.attempts()}
        old = by_id[bad["attempt_id"]]
        # Preserved, with its original numbers intact...
        self.assertEqual(old["observed_trials"], REPAIR_CELLS)
        self.assertEqual(old["expected_trials"], CELL_TRIALS)
        # ...and explicitly superseded, never silently rewritten.
        self.assertEqual(old["status"], "superseded")
        self.assertIsNotNone(old["superseded_by"])
        self.assertIn("expected_trials=300", old["superseded_reason"])

        new = by_id[old["superseded_by"]]
        self.assertEqual(new["record_kind"], "repair_attempt_correction")
        self.assertEqual(new["corrects_attempt_id"], bad["attempt_id"])
        self.assertEqual(new["corrects_bookkeeping"]["status"], "failed")
        self.assertEqual(new["corrects_bookkeeping"]["expected_trials"], CELL_TRIALS)
        self.assertEqual(new["status"], "complete")
        self.assertEqual(new["attempt_kind"], provenance.ATTEMPT_KIND_REPAIR)
        self.assertEqual(new["observed_repair_cells"], REPAIR_CELLS)
        self.assertEqual(new["expected_repair_cells"], REPAIR_CELLS)
        self.assertEqual(new["profile_shard_expected_trials"], CELL_TRIALS)
        # The plan file is gone, so the scope is reconstructed -- and SAYS so.
        self.assertEqual(new["repair_plan_basis"],
                         provenance.PLAN_BASIS_RECONSTRUCTED)

    def test_07_reconcile_is_a_dry_run_by_default(self):
        bad = self.mislabelled()
        before = (self.root / "provenance" / "attempts.jsonl").read_text()
        self.assertEqual(self.reconcile(apply=False), 0)
        self.assertEqual((self.root / "provenance" / "attempts.jsonl").read_text(),
                         before, "a dry run must write nothing")
        self.assertFalse((self.root / "provenance" / "cell_closures.jsonl").exists())
        self.assertEqual(
            [a for a in self.attempts() if a["attempt_id"] == bad["attempt_id"]][0]["status"],
            "failed")

    def test_08_closure_states_the_cell_level_sum(self):
        self.mislabelled()
        self.assertEqual(self.reconcile(apply=True), 0)
        closures = [json.loads(l) for l in
                    (self.root / "provenance" / "cell_closures.jsonl").read_text().splitlines()
                    if l.strip()]
        claude = [c for c in closures if c["profile"] == "claude"]
        self.assertEqual(len(claude), 1)
        c = claude[0]
        self.assertEqual(c["original_valid_observations"], ORIGINAL_VALID)
        self.assertEqual(c["accepted_provider_blocked"], 0)
        self.assertEqual(c["valid_repair_observations"], REPAIR_CELLS)
        self.assertEqual(c["accounted_cells"], CELL_TRIALS)
        self.assertEqual(c["expected_cells"], CELL_TRIALS)
        self.assertEqual(c["outstanding_cells"], 0)
        self.assertTrue(c["complete"])
        self.assertEqual(
            c["original_valid_observations"] + c["accepted_provider_blocked"]
            + c["valid_repair_observations"], c["expected_cells"])

    def test_09_reconcile_uses_the_frozen_plan_when_it_still_exists(self):
        """The executed-plan snapshot is preferred over reconstruction."""
        self.original_run()
        outstanding = self.planned[ORIGINAL_VALID:]
        plan_path = self.write_repair_plan(outstanding)
        self.repair_run(outstanding)
        self.record([self.root / "full" / f"rep01-full-claude-{self.cell.shard_label}-b01"],
                    note="campaign.sh repair-shard full shard 1")
        self.assertEqual(self.reconcile(apply=True, plan=plan_path), 0)
        new = [a for a in self.attempts()
               if a.get("record_kind") == "repair_attempt_correction"][0]
        self.assertEqual(new["repair_plan_basis"], provenance.PLAN_BASIS_FILE)
        self.assertEqual(new["repair_plan_sha256"], lib.sha256_file(plan_path))


# --------------------------------------------------------------------------- #
# 10 - the wrapper's own plumbing
# --------------------------------------------------------------------------- #
class TestWrapperWiring(unittest.TestCase):
    """campaign.sh must actually pass the plan through, and snapshot it."""

    def setUp(self):
        self.sh = (PROJECT_ROOT / "campaign.sh").read_text()

    def test_10_campaign_sh_is_syntactically_valid(self):
        subprocess.run(["bash", "-n", str(PROJECT_ROOT / "campaign.sh")], check=True)

    def test_11_repair_shard_passes_its_executed_plan_to_the_ledger(self):
        self.assertIn('record_attempts() {  # mode shard status started_at '
                      'result_file note [repair_plan]', self.sh)
        self.assertIn('local repair_plan="${7:-}"', self.sh)
        self.assertIn('plan_args=(--repair-plan "$repair_plan")', self.sh)
        # repair-shard passes the snapshot; run-shard passes nothing and keeps
        # the whole-cell rule.
        self.assertIn('"campaign.sh repair-shard $mode shard $shard" "$executed_plan"',
                      self.sh)
        self.assertIn('cp "$plan_file" "$executed_plan"', self.sh)
        self.assertIn("campaign.provenance reconcile --mode", self.sh)


if __name__ == "__main__":
    unittest.main(verbosity=2)
