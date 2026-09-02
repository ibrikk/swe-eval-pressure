#!/usr/bin/env python3
"""Local, offline tests that prove the campaign tooling DETECTS bad corpora.

Every test builds a synthetic campaign namespace on a temp filesystem, injects
exactly one defect, and asserts the validator FAILS with the expected check id.
A control test asserts a clean corpus PASSES, so the failures are meaningful.

No model call, no Harbor call, no network, no historical result directory.

Covered defects (the ones that actually killed the Aug 2026 study, plus the
ones the new design must be immune to):
  1  incomplete shard            -> X1 / <mode>/<profile>:1 / :4
  2  duplicate cell              -> provenance build errors + :4
  3  wrong attempt admitted      -> two accepted attempts for one cell
  4  synthetic trial             -> :5
  5  budget-censored trial       -> :6
  6  version mismatch / drift    -> :7
  7  missing task (arm hole)     -> :3 / :4
  8  historical result dir       -> assert_campaign_path refuses it outright

Single-shard execution planning (./campaign.sh run-shard) is covered too:
  9  valid slice                 -> plan lists all four profiles, right trial count
 10  invalid shard index         -> refused before anything is launched
 11  invalid mode                -> refused before anything is launched
 12  already-accepted shard      -> refused unless --new-attempt is explicit
"""
from __future__ import annotations

import importlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from campaign import lib  # noqa: E402


# --------------------------------------------------------------------------- #
# synthetic campaign fixture
# --------------------------------------------------------------------------- #
class Fixture:
    """A throwaway campaign namespace with a fabricated, structurally valid corpus.

    The trials contain no model output - only the metadata files the tooling
    reads (agent/trajectory.json, result.json). Nothing here costs money.
    """

    MODEL_WIRE = {
        "claude": "anthropic/claude-opus-4-8",
        "fable": "anthropic/claude-fable-5",
        "codex": "openai/gpt-5.6",
        "llama": "openai/llmengine/llama-3-3-70b-instruct",
    }

    def __init__(self, tmp: Path):
        tmp = Path(tmp).resolve()
        self.tmp = tmp
        self.root = tmp / "results" / "campaigns" / lib.CAMPAIGN_ID
        # Redirect the whole library at the temp namespace.
        self._orig = (lib.PROJECT_ROOT, lib.CAMPAIGN_ROOT)
        lib.PROJECT_ROOT = tmp
        lib.CAMPAIGN_ROOT = self.root
        for mod in ("campaign.provenance", "campaign.validate",
                    "campaign.analyze", "campaign.shard"):
            if mod in sys.modules:
                importlib.reload(sys.modules[mod])
        for k, p in lib.campaign_paths().items():
            if k not in ("manifest", "readme"):
                p.mkdir(parents=True, exist_ok=True)
        (lib.campaign_paths()["manifests"] / "cells").mkdir(parents=True, exist_ok=True)
        self.base_ids = [f"{i:08x}" for i in range(1, lib.BASE_TASK_COUNT + 1)]

    def restore(self):
        lib.PROJECT_ROOT, lib.CAMPAIGN_ROOT = self._orig

    # -- dataset side ------------------------------------------------------- #
    def write_cell_manifest(self, mode, profile, shard_index):
        cell = lib.Cell(mode, profile, shard_index)
        ds = self.root / "datasets" / "_shards" / mode / profile / cell.shard_label
        ds.mkdir(parents=True, exist_ok=True)
        (ds / "manifest.json").write_text(json.dumps({"mode": mode, "profile": profile}) + "\n")
        trials = []
        for b in self.shard_bases(shard_index):
            for arm in lib.ARMS[mode]:
                cond, chan, pressure = lib.ARMS[mode][arm]
                trials.append({
                    "base_task_id": b, "arm": arm, "condition": cond,
                    "delivery_channel": chan, "pressure_kind": pressure,
                    "resource_derivation_parent":
                        f"full/{profile}/ea-{b}-{arm}"
                        if (mode == "resource" and arm in ("clean-n", "eval-scaf")) else None,
                })
        mf = lib.campaign_paths()["manifests"] / "cells" / f"{mode}__{profile}__{cell.shard_label}.json"
        mf.write_text(json.dumps({
            "campaign_id": lib.CAMPAIGN_ID, "mode": mode, "profile": profile,
            "shard": cell.shard_label,
            "dataset_path": str(ds.relative_to(self.tmp)),
            "dataset_manifest_sha256": lib.sha256_file(ds / "manifest.json"),
            "expected_trials": cell.expected_trials, "trials": trials,
        }, indent=2) + "\n")

    def shard_bases(self, shard_index):
        start = (shard_index - 1) * lib.SHARD_SIZE
        return self.base_ids[start:min(start + lib.SHARD_SIZE, lib.BASE_TASK_COUNT)]

    # -- run side ----------------------------------------------------------- #
    def write_trial(self, parent: Path, base_id, arm, profile, *,
                    version=None, model=None, synthetic=False, censored=False):
        td = parent / f"ea-{base_id}-{arm}__x{abs(hash((base_id, arm))) % 9999:04d}"
        (td / "agent").mkdir(parents=True, exist_ok=True)
        (td / "agent" / "trajectory.json").write_text(json.dumps({
            "agent": {
                "name": lib.VERSION_PINS[profile]["agent"],
                "version": version or lib.VERSION_PINS[profile]["version"],
                "model_name": "<synthetic>" if synthetic else (model or self.MODEL_WIRE[profile]),
            },
            "final_metrics": {"total_prompt_tokens": 10, "total_completion_tokens": 5,
                              "total_cached_tokens": 0, "total_cost_usd": 0.0 if synthetic else 1.25,
                              "total_steps": 3},
        }))
        (td / "result.json").write_text(json.dumps({
            "task_name": f"ea-{base_id}-{arm}",
            "verifier_result": {"rewards": {"reward": 1.0, "overall_pass": 1.0}},
        }))
        if censored:
            (td / "exception.txt").write_text(
                "Budget has been exceeded! Key=pub-sprl-eval-awareness (sk-...XXXX)\n")
        return td

    def make_run(self, mode, profile, shard_index, *, drop=0, dup=False,
                 synthetic=0, censored=0, version=None, drop_arm=None, suffix=""):
        cell = lib.Cell(mode, profile, shard_index)
        run_dir = self.root / mode / f"swe-{mode}-{profile}-{cell.shard_label}{suffix}"
        parent = run_dir / run_dir.name
        parent.mkdir(parents=True, exist_ok=True)
        made = 0
        for b in self.shard_bases(shard_index):
            for arm in lib.ARMS[mode]:
                if drop_arm and b == self.shard_bases(shard_index)[0] and arm == drop_arm:
                    continue
                if drop and made >= cell.expected_trials - drop:
                    break
                self.write_trial(parent, b, arm, profile, version=version,
                                 synthetic=(made < synthetic),
                                 censored=(synthetic <= made < synthetic + censored))
                made += 1
            if drop and made >= cell.expected_trials - drop:
                break
        if dup:
            b, arm = self.shard_bases(shard_index)[0], list(lib.ARMS[mode])[0]
            td = parent / f"ea-{b}-{arm}__DUPLICATE"
            (td / "agent").mkdir(parents=True, exist_ok=True)
            shutil.copy(parent / f"ea-{b}-{arm}__x{abs(hash((b, arm))) % 9999:04d}" / "agent" / "trajectory.json",
                        td / "agent" / "trajectory.json")
        (run_dir / "run_metadata.json").write_text(json.dumps({
            "model": lib.MODEL_PINS[profile], "agent": lib.VERSION_PINS[profile]["agent"],
            "agent_version_requested": version or lib.VERSION_PINS[profile]["version"],
            "created_at": lib.now_iso(),
        }))
        return run_dir

    def build(self, defects=None):
        """Materialise every cell, then run record + build."""
        from campaign import provenance
        for mode in lib.MODES:
            for profile in lib.PROFILES:
                for i in lib.SHARD_INDICES:
                    self.write_cell_manifest(mode, profile, i)
                    kw = dict((defects or {}).get((mode, profile, i), {}))
                    # `force` admits a defective attempt into the corpus on
                    # purpose, so the test exercises the VALIDATOR rather than
                    # stopping at the earlier attempt-status gate.
                    status = kw.pop("force", None) and "complete" or "auto"
                    run_dir = self.make_run(mode, profile, i, **kw)
                    self.record(provenance, mode, profile, i, run_dir, status=status)
        return self.build_corpus(provenance)

    def record(self, provenance, mode, profile, shard, run_dir, status="auto"):
        ns = _Args(mode=mode, profile=profile, shard=shard, run_dir=str(run_dir),
                   status=status, started_at=None, finished_at=None, note="")
        return _quiet(provenance.cmd_record, ns)

    def build_corpus(self, provenance):
        return _quiet(provenance.cmd_build, _Args())


class _Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _quiet(fn, args):
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        rc = fn(args)
    return rc


def _validate():
    from campaign import validate
    importlib.reload(validate)
    return validate.validate()


class CampaignToolingTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="campv2-test-")
        self.fx = Fixture(Path(self.tmpdir))

    def tearDown(self):
        self.fx.restore()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _failed_ids(self, report):
        return set(report["failed"])

    # -- control ------------------------------------------------------------ #
    def test_00_clean_corpus_passes(self):
        self.fx.build()
        rep = _validate()
        self.assertTrue(rep["ok"], f"clean corpus should validate; failed={rep['failed']}")
        self.assertEqual(rep["trials"], 3640)

    def test_00b_expected_totals(self):
        exp = lib.expected_totals()
        self.assertEqual(exp["full"]["total"], 2800)
        self.assertEqual(exp["resource"]["total"], 840)
        self.assertEqual(exp["campaign_total"], 3640)
        self.assertEqual(exp["cells"], 24)

    # -- 1. incomplete shard ------------------------------------------------ #
    def test_01_incomplete_shard_detected(self):
        self.fx.build({("full", "claude", 2): {"drop": 17}})
        rep = _validate()
        self.assertFalse(rep["ok"])
        self.assertIn("X1", self._failed_ids(rep))
        self.assertTrue(any(f.startswith("F:full/claude") for f in rep["failed"]))

    # -- 2. duplicate cell -------------------------------------------------- #
    def test_02_duplicate_cell_detected(self):
        from campaign import provenance
        for mode in lib.MODES:
            for profile in lib.PROFILES:
                for i in lib.SHARD_INDICES:
                    self.fx.write_cell_manifest(mode, profile, i)
                    dup = (mode, profile, i) == ("resource", "codex", 1)
                    rd = self.fx.make_run(mode, profile, i, dup=dup)
                    self.fx.record(provenance, mode, profile, i, rd,
                                   status="complete" if dup else "auto")
        rc = self.fx.build_corpus(provenance)
        self.assertEqual(rc, 1, "provenance build must FAIL on a duplicate, not dedupe it")
        rep = _validate()
        self.assertFalse(rep["ok"], "a corpus the builder rejected must never validate")
        self.assertIn("X4", self._failed_ids(rep))
        self.assertIn("X0", self._failed_ids(rep))

    # -- 3. wrong attempt admitted ------------------------------------------ #
    def test_03_two_accepted_attempts_for_one_cell_detected(self):
        from campaign import provenance
        self.fx.build()
        # A second, also-complete attempt for a cell that already has one.
        rd2 = self.fx.make_run("full", "fable", 1, suffix="-RETRY")
        self.fx.record(provenance, "full", "fable", 1, rd2)
        acc = json.loads((lib.campaign_paths()["provenance"] / "accepted_runs.json").read_text())
        # The newest complete attempt supersedes the old one - exactly one winner.
        self.assertEqual(acc["conflicting_cells"], [])
        self.assertEqual(acc["cells_complete"], 24)
        superseded = [a for a in acc["rejected"] if a["status"] == "superseded"]
        self.assertEqual(len(superseded), 1, "the earlier attempt must be preserved as superseded")
        self.assertIsNotNone(superseded[0]["superseded_by"])
        # And forcing both to be accepted is a hard error, never a silent dedupe.
        att = lib.campaign_paths()["provenance"] / "attempts.jsonl"
        lines = [json.loads(l) for l in att.read_text().splitlines() if l.strip()]
        for e in lines:
            if e["status"] == "superseded":
                e["status"], e["superseded_by"] = "complete", None
        att.write_text("".join(json.dumps(e) + "\n" for e in lines))
        rc = self.fx.build_corpus(provenance)
        self.assertEqual(rc, 1, "two accepted attempts for one cell must be a hard error")

    # -- 4. synthetic trial ------------------------------------------------- #
    def test_04_synthetic_trial_detected(self):
        self.fx.build({("full", "fable", 3): {"synthetic": 12, "force": True}})
        rep = _validate()
        self.assertFalse(rep["ok"])
        self.assertIn("F:full/fable:5", self._failed_ids(rep))

    # -- 5. budget-censored trial ------------------------------------------- #
    def test_05_budget_censored_trial_detected(self):
        self.fx.build({("resource", "claude", 2): {"censored": 9, "force": True}})
        rep = _validate()
        self.assertFalse(rep["ok"])
        self.assertIn("R:resource/claude:6", self._failed_ids(rep))

    # -- 6. version mismatch ------------------------------------------------ #
    def test_06_version_drift_detected(self):
        self.fx.build({("full", "claude", 2): {"version": "2.1.241"}})
        rep = _validate()
        self.assertFalse(rep["ok"])
        self.assertIn("F:full/claude:7", self._failed_ids(rep))
        detail = next(c["detail"] for c in rep["checks"] if c["id"] == "F:full/claude:7")
        self.assertIn("2.1.241", detail)

    # -- 7. missing task / arm hole ----------------------------------------- #
    def test_07_missing_arm_detected(self):
        self.fx.build({("resource", "llama", 1): {"drop_arm": "eval-resource-scaf", "force": True}})
        rep = _validate()
        self.assertFalse(rep["ok"])
        failed = self._failed_ids(rep)
        self.assertIn("R:resource/llama:3", failed)
        self.assertIn("R:resource/llama:4", failed)

    # -- 8. historical result directory ------------------------------------- #
    def test_08_historical_run_dir_refused(self):
        from campaign import provenance
        hist = self.fx.tmp / "results" / "full" / "ea-full-claude-20260816-153633"
        (hist / hist.name).mkdir(parents=True, exist_ok=True)
        self.fx.write_trial(hist / hist.name, self.fx.base_ids[0], "clean-n", "claude")
        with self.assertRaises(ValueError) as cm:
            lib.assert_campaign_path(hist, "run directory")
        self.assertIn("outside the campaign namespace", str(cm.exception))
        with self.assertRaises(ValueError):
            self.fx.record(provenance, "full", "claude", 1, hist)

    # -- 9. analysis refuses an unvalidated campaign ------------------------ #
    def test_09_analyze_refuses_unvalidated(self):
        from campaign import analyze
        importlib.reload(analyze)
        self.fx.build({("full", "codex", 1): {"drop": 5}})
        _validate()
        with self.assertRaises(SystemExit) as cm:
            analyze.require_validated(lib.campaign_paths())
        self.assertIn("validation did not pass", str(cm.exception))

    # -- 10. corpus never mixes modes --------------------------------------- #
    def test_10_no_cross_mode_substitution(self):
        from campaign import analyze
        importlib.reload(analyze)
        self.fx.build()
        rows = analyze.load_corpus(lib.campaign_paths())
        leaks = analyze.cross_mode_leak_check(rows)
        self.assertTrue(leaks["ok"], leaks)
        full = [r for r in rows if r["mode"] == "full"]
        res = [r for r in rows if r["mode"] == "resource"]
        self.assertEqual(len(full), 2800)
        self.assertEqual(len(res), 840)
        self.assertEqual(set(r["run_id"] for r in full) & set(r["run_id"] for r in res), set())
        for arm in ("clean-n", "eval-scaf"):
            self.assertEqual(len([r for r in res if r["arm"] == arm]), 280,
                             f"resource must own its {arm} executions")

    # -- single-shard execution planning (run-shard) ------------------------ #
    def _prepare_slice(self, mode, shard):
        """Materialise only the dataset + frozen manifest for one slice."""
        for profile in lib.PROFILES:
            self.fx.write_cell_manifest(mode, profile, shard)

    def test_11_run_shard_plan_valid_slice(self):
        from campaign import shard
        self._prepare_slice("full", 1)
        plan = shard.plan_shard("full", 1)

        # all four profiles, never a subset
        self.assertEqual([c["profile"] for c in plan["cells"]], list(lib.PROFILES))
        self.assertEqual(plan["profiles"], list(lib.PROFILES))
        # 30 base tasks x 10 full arms x 4 profiles
        self.assertEqual(plan["expected_trials"], 1200)
        self.assertEqual(plan["shard_label"], "chunk-1-size-30")
        self.assertFalse(plan["new_attempt"])
        self.assertEqual(plan["supersedes"], [])
        # every planned path stays inside the campaign namespace
        for c in plan["cells"]:
            lib.assert_campaign_path(Path(c["dataset"]), "planned dataset")
            self.assertFalse(c["already_accepted"])

        # shard 3 is the short shard: 10 base tasks, not 30
        self._prepare_slice("resource", 3)
        p3 = shard.plan_shard("resource", 3)
        self.assertEqual(p3["base_tasks"], 10)
        self.assertEqual(p3["expected_trials"], 10 * 3 * 4)

    def test_12_run_shard_invalid_shard_index_refused(self):
        from campaign import shard
        self._prepare_slice("full", 1)
        for bad in (0, 4, -1, 99):
            with self.assertRaises(shard.ShardPlanError) as cm:
                shard.plan_shard("full", bad)
            self.assertIn("out of range", str(cm.exception))
        with self.assertRaises(shard.ShardPlanError) as cm:
            shard.plan_shard("full", "two")
        self.assertIn("not an integer", str(cm.exception))

    def test_13_run_shard_invalid_mode_refused(self):
        from campaign import shard
        self._prepare_slice("full", 1)
        for bad in ("FULL", "resources", "pilot", "", "full/claude"):
            with self.assertRaises(shard.ShardPlanError) as cm:
                shard.plan_shard(bad, 1)
            self.assertIn("unknown mode", str(cm.exception))

    def test_14_run_shard_refuses_rerun_of_accepted_shard(self):
        from campaign import shard
        self.fx.build()                       # every cell accepted
        rep = _validate()
        self.assertTrue(rep["ok"], f"precondition: clean corpus; failed={rep['failed']}")

        # a plain rerun of an accepted shard is refused
        with self.assertRaises(shard.ShardPlanError) as cm:
            shard.plan_shard("full", 1)
        msg = str(cm.exception) + " " + " ".join(cm.exception.detail)
        self.assertIn("already has accepted attempts", msg)
        self.assertIn("--new-attempt", msg)
        # it names every accepted cell it is protecting
        for profile in lib.PROFILES:
            self.assertIn(f"full/{profile}/chunk-1-size-30", msg)

        # the explicit new-attempt workflow is allowed, and says what it supersedes
        plan = shard.plan_shard("full", 1, new_attempt=True)
        self.assertTrue(plan["new_attempt"])
        self.assertEqual(len(plan["supersedes"]), len(lib.PROFILES))
        self.assertTrue(all(c["already_accepted"] for c in plan["cells"]))
        self.assertTrue(all(c["prior_attempt_id"] for c in plan["cells"]))

        # an unaffected shard is unaffected by the refusal
        with self.assertRaises(shard.ShardPlanError):
            shard.plan_shard("resource", 2)

    def test_15_run_shard_unprepared_slice_refused(self):
        from campaign import shard
        with self.assertRaises(shard.ShardPlanError) as cm:
            shard.plan_shard("full", 2)       # nothing prepared in this fixture
        self.assertIn("not prepared", str(cm.exception))

    def test_16_campaign_sh_accepts_run_shard_command(self):
        """The shell whitelist knows run-shard. Exits at the id gate, before
        sourcing .env, so this launches nothing and needs no credentials."""
        import subprocess
        proc = subprocess.run(
            ["bash", str(PROJECT_ROOT / "campaign.sh"), "run-shard", "wrong-id", "full", "1"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT))
        self.assertEqual(proc.returncode, 2)
        self.assertIn("does not match", proc.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
