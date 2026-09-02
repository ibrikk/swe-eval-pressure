#!/usr/bin/env python3
"""Offline tests for cell-level Campaign V2 repair and live telemetry.

NO Harbor job, NO model call, NO benchmark trajectory is launched anywhere in
this file. Every fixture is a temporary directory holding synthetic JSON.

Coverage maps 1:1 onto the behaviours the 2026-09-02 FULL shard-1 crash forced:
   1 an interrupted cell keeps its valid completed trajectories
   2 a trajectory that FAILS the SWE verifier is still COMPLETE_VALID
   3 a partial trajectory is NOT COMPLETE_VALID
   4 the repair manifest contains only missing/invalid cells
   5 the exact individual repair dataset contains no sibling arms
   6 a retry can never pass base_count=0 into the base-task slicer
   7 retry lineage closes: accepted / failed / abandoned
   8 a duplicate valid trajectory for one cell is a hard refusal
   9 timestamp-based TPM avoids the batch-collapse spike
  10 budget polling happens during long in-flight jobs
  11 a fully repaired shard validates to exactly 1,200 cells
  12 no historical Aug-2026 trajectory can enter Campaign V2
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from campaign import cells, controller, failures, lib, retry_dataset, tokens

ARMS = ("clean-n", "eval-src", "eval-fin-src")


# --------------------------------------------------------------------------- #
# fixture helpers
# --------------------------------------------------------------------------- #
def write_json(path: Path, blob) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(blob, indent=1), encoding="utf-8")
    return path


def trajectory(*, steps=3, completion=500, prompt=4000, cached=3000,
               model="anthropic/claude-opus-4-8", t0="2026-09-02T06:00:00Z"):
    """A trajectory whose per-step metrics sum exactly to final_metrics."""
    per_c, per_p, per_k = completion // steps, prompt // steps, cached // steps
    out = []
    for i in range(steps):
        out.append({
            "step_id": i,
            "timestamp": f"2026-09-02T06:{i:02d}:00Z" if t0 else None,
            "source": "assistant",
            "model_name": model,
            "metrics": {"prompt_tokens": per_p, "completion_tokens": per_c,
                        "cached_tokens": per_k},
        })
    return {
        "schema_version": "atif-v1.7",
        "session_id": "s",
        "agent": {"name": "claude-code", "version": "1.0", "model_name": model},
        "steps": out,
        "final_metrics": {"total_prompt_tokens": per_p * steps,
                          "total_completion_tokens": per_c * steps,
                          "total_cached_tokens": per_k * steps,
                          "total_cost_usd": 1.5, "total_steps": steps},
    }


def result_json(*, reward=1.0, agent_finished=True, verifier_finished=True,
                task_path="/campaign/task", exception_type=None):
    blob = {
        "task_id": {"path": task_path},
        "started_at": "2026-09-02T06:00:00Z",
        "finished_at": "2026-09-02T06:30:00Z",
        "agent_execution": ({"finished_at": "2026-09-02T06:25:00Z"}
                            if agent_finished else {}),
        "verifier": ({"finished_at": "2026-09-02T06:30:00Z"}
                     if verifier_finished else {}),
        "verifier_result": {"rewards": {"reward": reward, "tests_reward": reward}},
    }
    if exception_type:
        blob["exception_info"] = {"exception_type": exception_type}
    return blob


def make_trial(root: Path, name: str, *, traj=None, result=None) -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    if traj is not None:
        write_json(d / "agent" / "trajectory.json", traj)
    if result is not None:
        write_json(d / "result.json", result)
    return d


class Fixture:
    """A miniature but structurally faithful campaign tree.

    3 base tasks x 3 arms x 1 profile = 9 cells, laid out exactly as the real
    campaign is so the production code paths run unmodified.
    """

    def __init__(self, td: Path, *, mode="full", shard=1, profile="claude",
                 base_tasks=3, arms=ARMS):
        self.root = Path(td)
        self.mode, self.shard, self.profile = mode, shard, profile
        self.arms = list(arms)
        self.label = lib.Cell(mode, profile, shard).shard_label
        self.paths = {
            "root": self.root, "full": self.root / "full",
            "resource": self.root / "resource", "logs": self.root / "logs",
            "provenance": self.root / "provenance",
            "datasets": self.root / "datasets",
            "manifests": self.root / "manifests",
        }
        for p in self.paths.values():
            Path(p).mkdir(parents=True, exist_ok=True)
        self.base_ids = [f"task-{i:024x}" for i in range(1, base_tasks + 1)]
        self.dataset = (self.paths["datasets"] / "_shards" / mode / profile
                        / self.label)
        tasks = []
        for bid in self.base_ids:
            for arm in self.arms:
                directory = f"ea-{bid[-8:]}-{arm}"
                tdir = self.dataset / directory
                tdir.mkdir(parents=True, exist_ok=True)
                (tdir / "task.yaml").write_text(f"id: {directory}\n", encoding="utf-8")
                (tdir / "PROMPT.md").write_text(f"do {directory}\n", encoding="utf-8")
                tasks.append({"directory": directory, "base_task_id": bid,
                              "condition": arm, "content_hash": lib.sha256_tree(tdir)})
        write_json(self.dataset / "manifest.json", {
            "schema_version": "2.1", "mode": mode, "profile": profile,
            "campaign_id": lib.CAMPAIGN_ID, "base_task_count": base_tasks,
            "variants_per_task": len(self.arms), "tasks": tasks,
            "shard": {"type": "fixed_size", "index": shard,
                      "base_task_size": base_tasks},
            "adaptive_batch": {"note": "must be stripped from a repair dataset"},
        })

    def task_dir(self, base_index: int, arm: str) -> str:
        return f"ea-{self.base_ids[base_index][-8:]}-{arm}"

    def run_dir(self, name="job-1") -> Path:
        d = (self.paths[self.mode]
             / f"{name}-{self.mode}-{self.profile}-{self.label}-b01" / "harbor-job")
        d.mkdir(parents=True, exist_ok=True)
        return d

    def add_trial(self, run: Path, base_index: int, arm: str, *, suffix="AAA1",
                  traj=..., result=...) -> Path:
        name = f"{self.task_dir(base_index, arm)}__{suffix}"
        return make_trial(
            run, name,
            traj=(trajectory() if traj is ... else traj),
            result=(result_json(task_path=str(self.dataset / self.task_dir(base_index, arm)))
                    if result is ... else result))

    def fill_all_valid(self, run: Path) -> None:
        for i in range(len(self.base_ids)):
            for arm in self.arms:
                self.add_trial(run, i, arm)

    def audit(self):
        return cells.audit(self.mode, self.shard, profiles=[self.profile],
                           paths=self.paths)


def relocate_campaign_root(case: unittest.TestCase, root: Path) -> None:
    """Point lib.CAMPAIGN_ROOT at a temp tree for the duration of one test.

    The namespace guard is deliberately NOT weakened for tests -- it is the
    structural defence against a historical run directory being handed to
    campaign tooling. It is simply pointed at the fixture's own root so it
    still runs, and still refuses anything outside it.
    """
    patcher = mock.patch.object(lib, "CAMPAIGN_ROOT", Path(root))
    patcher.start()
    case.addCleanup(patcher.stop)


class FixtureCase(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.mkdtemp(prefix="campaign-repair-test-")
        self.addCleanup(shutil.rmtree, self._td, ignore_errors=True)
        relocate_campaign_root(self, Path(self._td))
        self.fx = Fixture(Path(self._td))


# --------------------------------------------------------------------------- #
# 1-3, 12: what counts as a valid observation
# --------------------------------------------------------------------------- #
class TestCellValidity(FixtureCase):

    def test_01_interrupted_cell_keeps_its_valid_completed_trajectories(self):
        """The core of the recovery policy: finished work is never discarded.

        Nine cells were planned; the run was interrupted after seven. The two
        that never ran are MISSING, and the seven that finished stay valid --
        including the ones from a batch nobody reaped.
        """
        run = self.fx.run_dir()
        planned = [(i, a) for i in range(3) for a in ARMS]
        for i, arm in planned[:7]:
            self.fx.add_trial(run, i, arm)
        res = self.fx.audit()
        self.assertEqual(res["expected"], 9)
        self.assertEqual(res["counts"][cells.COMPLETE_VALID], 7)
        self.assertEqual(res["counts"][cells.MISSING], 2)
        self.assertEqual(res["repair_required"], 2,
                         "only the two cells that never ran need work")
        frozen = cells.frozen_cells(res)
        self.assertEqual(len(frozen), 7)
        plan = cells.repair_plan(res)
        for key in frozen:
            self.assertNotIn(key, [c["cell_key"] for c in plan["by_profile"][self.fx.profile]],
                             "a COMPLETE_VALID cell must never appear in a repair plan")

    def test_01b_unreaped_batch_trajectories_are_counted(self):
        """The controller's own completed counter is not the authority.

        At the crash it displayed 584; disk held 976. Trials in a run directory
        the controller never reaped are ordinary trajectories on disk.
        """
        reaped = self.fx.run_dir("job-reaped")
        unreaped = self.fx.run_dir("job-never-reaped")
        for arm in ARMS:
            self.fx.add_trial(reaped, 0, arm)
        for arm in ARMS:
            self.fx.add_trial(unreaped, 1, arm)
        res = self.fx.audit()
        self.assertEqual(res["counts"][cells.COMPLETE_VALID], 6)
        self.assertEqual(len(res["run_dirs"][self.fx.profile]), 2)

    def test_02_failed_swe_verifier_is_still_complete_valid(self):
        """Outcome must not affect validity, in either direction."""
        run = self.fx.run_dir()
        self.fx.add_trial(run, 0, "clean-n",
                          result=result_json(reward=0.0,
                                             task_path=str(self.fx.dataset)))
        self.fx.add_trial(run, 1, "clean-n",
                          result=result_json(reward=1.0,
                                             task_path=str(self.fx.dataset)))
        res = self.fx.audit()
        got = {k: r.status for k, r in res["records"].items()
               if r.observations}
        self.assertEqual(set(got.values()), {cells.COMPLETE_VALID},
                         "reward 0.0 is a RESULT, not a broken trial")
        self.assertEqual(res["counts"][cells.COMPLETE_VALID], 2)

    def test_02b_step_cap_exit_after_a_verdict_is_still_valid(self):
        """The real llama case: 222 steps, non-zero exit, full rubric verdict."""
        run = self.fx.run_dir()
        p = self.fx.add_trial(
            run, 0, "clean-n",
            traj=trajectory(steps=3, completion=120_000),
            result=result_json(reward=0.0, exception_type="NonZeroAgentExitCodeError",
                               task_path=str(self.fx.dataset)))
        obs = cells.classify_observation(p, campaign_root=self.fx.root)
        self.assertEqual(obs.status, cells.COMPLETE_VALID)
        self.assertEqual(obs.agent_exit_exception, "NonZeroAgentExitCodeError",
                         "the exception is recorded as provenance...")
        self.assertIn("NonZeroAgentExitCodeError", obs.reason,
                      "...and surfaced in the reason, but is not disqualifying")

    def test_03_partial_trajectory_is_not_complete_valid(self):
        run = self.fx.run_dir()
        truncated = self.fx.add_trial(
            run, 0, "clean-n",
            result=result_json(agent_finished=False,
                               task_path=str(self.fx.dataset)))
        no_verdict = self.fx.add_trial(
            run, 1, "clean-n",
            result=result_json(verifier_finished=False,
                               task_path=str(self.fx.dataset)))
        no_output = self.fx.add_trial(
            run, 2, "clean-n", traj=trajectory(completion=0),
            result=result_json(task_path=str(self.fx.dataset)))
        self.assertEqual(
            cells.classify_observation(truncated, campaign_root=self.fx.root).status,
            cells.PARTIAL_MODEL_FAILURE)
        self.assertEqual(
            cells.classify_observation(no_verdict, campaign_root=self.fx.root).status,
            cells.PARTIAL_MODEL_FAILURE)
        self.assertEqual(
            cells.classify_observation(no_output, campaign_root=self.fx.root).status,
            cells.PRE_MODEL_FAILURE)
        res = self.fx.audit()
        self.assertEqual(res["counts"][cells.COMPLETE_VALID], 0)
        self.assertEqual(res["repair_required"], 9)

    def test_12_historical_trajectory_cannot_enter_campaign_v2(self):
        """An Aug-2026 trajectory is rejected on task-path provenance alone."""
        run = self.fx.run_dir()
        p = self.fx.add_trial(
            run, 0, "clean-n",
            result=result_json(reward=1.0,
                               task_path="/repo/results/2026-08-27-matrix/ea-x/task"))
        obs = cells.classify_observation(p, campaign_root=self.fx.root)
        self.assertEqual(obs.status, cells.OTHER_INVALID)
        self.assertIn("outside the campaign namespace", obs.reason)
        res = self.fx.audit()
        self.assertEqual(res["counts"][cells.COMPLETE_VALID], 0,
                         "a historical trajectory must never satisfy a campaign cell")
        self.assertEqual(res["counts"][cells.OTHER_INVALID], 1)
        # And the structural guard refuses the path outright.
        with self.assertRaises(ValueError):
            lib.assert_campaign_path(Path("/repo/results/2026-08-27-matrix"),
                                     "historical run")


# --------------------------------------------------------------------------- #
# 4, 8: repair planning
# --------------------------------------------------------------------------- #
class TestRepairPlan(FixtureCase):

    def test_04_repair_manifest_holds_only_missing_or_invalid_cells(self):
        run = self.fx.run_dir()
        self.fx.fill_all_valid(run)
        # break exactly two cells, in two different ways
        shutil.rmtree(run / f"{self.fx.task_dir(0, 'eval-src')}__AAA1")
        make_trial(run, f"{self.fx.task_dir(1, 'eval-fin-src')}__AAA1",
                   traj=trajectory(),
                   result=result_json(agent_finished=False,
                                      task_path=str(self.fx.dataset)))
        res = self.fx.audit()
        plan = cells.repair_plan(res)
        listed = {c["cell_key"] for c in plan["by_profile"][self.fx.profile]}
        self.assertEqual(len(listed), 2)
        self.assertEqual(plan["complete_valid"], 7)
        self.assertEqual(plan["repair_required"], 2)
        want_dirs = {self.fx.task_dir(0, "eval-src"), self.fx.task_dir(1, "eval-fin-src")}
        self.assertEqual({c["task_dir"] for c in plan["by_profile"][self.fx.profile]},
                         want_dirs)
        for key in cells.frozen_cells(res):
            self.assertNotIn(key, listed)

    def test_08_duplicate_valid_trajectory_is_a_hard_refusal(self):
        """Two valid trajectories for one cell is a provenance decision.

        Not a dedupe by task_name, not "keep the newest", and emphatically not
        "keep the one that scored better".
        """
        run = self.fx.run_dir()
        self.fx.fill_all_valid(run)
        self.fx.add_trial(run, 0, "clean-n", suffix="BBB2",
                          result=result_json(reward=0.0,
                                             task_path=str(self.fx.dataset)))
        res = self.fx.audit()
        self.assertEqual(len(res["duplicates"]), 1)
        dup = res["records"][res["duplicates"][0]]
        self.assertEqual(dup.status, cells.DUPLICATE)
        self.assertIn("human", dup.reason)
        with self.assertRaises(SystemExit) as cm:
            cells.repair_plan(res)
        self.assertIn("provenance decision", str(cm.exception))

    def test_08b_every_corpus_row_exposes_its_source_attempt(self):
        run = self.fx.run_dir("job-7")
        self.fx.add_trial(run, 0, "clean-n")
        res = self.fx.audit()
        rec = next(r for r in res["records"].values() if r.observations)
        row = rec.as_dict()
        obs = row["observations"][0]
        self.assertTrue(obs["run_dir"].startswith("job-7-"),
                        "the source Harbor run must be recoverable from the row")
        self.assertTrue(obs["trial_dir"])
        self.assertTrue(obs["path"])
        self.assertEqual(row["campaign_id"], lib.CAMPAIGN_ID)


# --------------------------------------------------------------------------- #
# 5, 6: the repair work unit
# --------------------------------------------------------------------------- #
class TestRepairWorkUnit(FixtureCase):

    def test_05_exact_repair_dataset_has_no_sibling_arms(self):
        out = Path(self._td) / "retry-out"
        want = self.fx.task_dir(0, "eval-src")
        retry_dataset.build_retry_dataset(
            source_dataset=self.fx.dataset, output=out,
            original_trial_id=f"{want}__AAA1",
            retry_trial_id=f"{want}__AAA1--retry1", retry_number=1,
            failure_class=failures.PRE_MODEL_INFRA, failure_reason="image build",
            original_run_dir="job-1", cell="full/claude/chunk-1-size-30")
        mf_checked = retry_dataset.assert_single_trial_dataset(out)
        self.assertEqual(mf_checked["retry_provenance"]["original_task_directory"], want)
        on_disk = sorted(d.name for d in out.iterdir()
                         if d.is_dir() and d.name.startswith("ea-"))
        self.assertEqual(on_disk, [want], "no sibling arm may be materialised")
        for arm in ("clean-n", "eval-fin-src"):
            self.assertFalse((out / self.fx.task_dir(0, arm)).exists())
        mf = json.loads((out / "manifest.json").read_text())
        self.assertEqual(mf["base_task_count"], 1)
        self.assertEqual(len(mf["tasks"]), 1)
        self.assertNotIn("adaptive_batch", mf)
        prov = mf["retry_provenance"]
        self.assertEqual(prov["original_trial_id"], f"{want}__AAA1")
        self.assertEqual(prov["retry_number"], 1)
        self.assertEqual(prov["failure_class"], failures.PRE_MODEL_INFRA)

    def test_05b_bulk_repair_dataset_holds_exactly_the_listed_cells(self):
        out = Path(self._td) / "repair-out"
        want = [self.fx.task_dir(0, "eval-src"), self.fx.task_dir(2, "clean-n")]
        retry_dataset.build_repair_dataset(
            source_dataset=self.fx.dataset, output=out, task_dirs=want,
            label="repair-01", provenance={"shard_index": 1})
        retry_dataset.assert_exact_cells(out, want)
        on_disk = sorted(d.name for d in out.iterdir()
                         if d.is_dir() and d.name.startswith("ea-"))
        self.assertEqual(on_disk, sorted(want))
        self.assertEqual(len(json.loads((out / "manifest.json").read_text())["tasks"]), 2)
        with self.assertRaises(SystemExit) as cm:
            retry_dataset.assert_exact_cells(out, want + [self.fx.task_dir(1, "clean-n")])
        self.assertIn("do not match the requested cells", str(cm.exception))

    def test_05c_repair_dataset_preserves_task_content_byte_for_byte(self):
        out = Path(self._td) / "retry-digest"
        want = self.fx.task_dir(1, "clean-n")
        retry_dataset.build_retry_dataset(
            source_dataset=self.fx.dataset, output=out,
            original_trial_id=f"{want}__ZZZ9", retry_trial_id=f"{want}__ZZZ9--retry1",
            retry_number=1, failure_class=failures.TRANSIENT_PROVIDER,
            failure_reason="429", original_run_dir="job-1", cell="c")
        self.assertEqual(lib.sha256_tree(self.fx.dataset / want),
                         lib.sha256_tree(out / want),
                         "a repair must re-run the SAME task, not a re-generated one")

    def test_06_retry_can_never_pass_base_count_zero_to_the_slicer(self):
        """The exact 2026-09-02 fatal path, pinned shut two ways."""
        c = controller.Controller("full", 1, dry_run=True, paths=self.fx.paths)
        unit = controller.RepairUnit(profile="claude", index=1001, expected_trials=1,
                                     cells=[self.fx.task_dir(0, "eval-src")],
                                     original_trial_id="x", retry_trial_id="x--retry1")
        with self.assertRaises(TypeError) as cm:
            c._slice_dataset(unit)
        self.assertIn("base-task batches only", str(cm.exception))
        # ...and even a Batch cannot smuggle the sentinel through.
        with self.assertRaises(ValueError) as cm2:
            c._slice_dataset(controller.Batch(profile="claude", index=1,
                                              start_base=0, base_count=0,
                                              expected_trials=1))
        self.assertIn("base_count=0", str(cm2.exception))
        # A repair unit is a different TYPE, so the sentinel is unconstructible.
        self.assertFalse(hasattr(unit, "base_count"))
        self.assertNotIsInstance(unit, controller.Batch)

    def test_06b_a_queued_retry_is_a_repair_unit_carrying_one_cell(self):
        c = controller.Controller("full", 1, dry_run=True, paths=self.fx.paths,
                                  profiles=["claude"])
        c.units["claude"] = []
        batch = controller.Batch(profile="claude", index=1, start_base=0,
                                 base_count=3, expected_trials=9,
                                 run_dir=str(self.fx.run_dir()))
        trial = lib.Trial(run_dir="job", trial_dir=f"{self.fx.task_dir(0,'eval-src')}__AAA1",
                          base_task_id="b", arm="eval-src", agent_name="", agent_version="",
                          model_name="", cost_usd=None, prompt_tokens=None,
                          completion_tokens=None, steps=None, status="error",
                          reward=None, resolved=None)
        tp = make_trial(Path(self._td) / "failing", "t")
        # A CORROBORATED 429: the classifier rejects Harbor's class name alone.
        (tp / "exception.txt").write_text(
            'ApiRateLimitError: HTTP/1.1 429 Too Many Requests\n'
            '{"error": {"type": "rate_limit_error", "message": "slow down"}}\n',
            encoding="utf-8")
        c.budget.enabled = False
        c._handle_failure(batch, trial, tp)
        queued = c.units["claude"]
        self.assertEqual(len(queued), 1)
        u = queued[0]
        self.assertIsInstance(u, controller.RepairUnit)
        self.assertEqual(u.expected_trials, 1)
        self.assertEqual(u.cells, [self.fx.task_dir(0, "eval-src")],
                         "exactly the failed cell, no siblings")


# --------------------------------------------------------------------------- #
# 7: retry lineage
# --------------------------------------------------------------------------- #
class TestRetryLineage(FixtureCase):

    def ledger(self):
        return failures.RetryLedger(Path(self._td) / "retries.jsonl")

    def unit(self, retry_id="orig--retry1"):
        return controller.RepairUnit(profile="claude", index=1001, expected_trials=1,
                                     cells=["ea-aaaaaaaa-eval-src"],
                                     original_trial_id="orig",
                                     retry_trial_id=retry_id, retry_number=1)

    def test_07_retry_lineage_closes_accepted_failed_abandoned(self):
        cls = failures.Classification(failures.TRANSIENT_PROVIDER, "429", True, True)
        for status in ("accepted", "failed", "abandoned"):
            led = failures.RetryLedger(Path(self._td) / f"{status}.jsonl")
            rec = led.open_retry("orig", f"orig--retry-{status}", cls, cell="c")
            self.assertEqual(rec.accepted_status, "pending")
            self.assertEqual(len(led.open_records()), 1)
            closed = led.close_retry(f"orig--retry-{status}", status,
                                     retry_run_dir="/runs/job-9")
            self.assertIsNotNone(closed)
            self.assertEqual(closed.accepted_status, status)
            self.assertTrue(closed.finished_at, "a closed record must be timestamped")
            self.assertEqual(closed.retry_run_dir, "/runs/job-9")
            self.assertEqual(led.open_records(), [],
                             "no record may stay pending after close")
        with self.assertRaises(ValueError):
            self.ledger().close_retry("x", "maybe")

    def test_07b_controller_closes_a_retry_on_reap(self):
        c = controller.Controller("full", 1, dry_run=True, paths=self.fx.paths,
                                  profiles=["claude"])
        c.units["claude"] = []
        cls = failures.Classification(failures.TRANSIENT_PROVIDER, "429", True, True)
        c.ledger.open_retry("orig", "orig--retry1", cls, cell="c")
        u = self.unit()
        u.run_dir = ""                       # the retry produced no run directory
        c.reap(u)
        recs = c.ledger.records()
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0].accepted_status, "failed")
        self.assertEqual(c.ledger.open_records(), [])

    def test_07c_a_hard_stop_abandons_open_retries_instead_of_leaving_them_pending(self):
        """The 2026-09-02 ledger left six records pending forever."""
        c = controller.Controller("full", 1, dry_run=True, paths=self.fx.paths,
                                  profiles=["claude"])
        cls = failures.Classification(failures.TRANSIENT_PROVIDER, "429", True, True)
        c.ledger.open_retry("orig", "orig--retry1", cls, cell="c")
        u = self.unit()
        c.inflight["claude"] = u
        c.units["claude"] = [u]
        c._stop("budget exhausted")
        recs = c.ledger.records()
        self.assertEqual(recs[0].accepted_status, "abandoned")
        self.assertTrue(recs[0].finished_at)

    def test_07d_a_failing_retry_does_not_start_a_fresh_retry_budget(self):
        led = self.ledger()
        cls = failures.Classification(failures.TRANSIENT_PROVIDER, "429", True, True)
        for i in range(failures.MAX_RETRIES):
            led.open_retry("orig", f"orig--retry{i+1}", cls, cell="c")
            led.close_retry(f"orig--retry{i+1}", "failed")
        ok, why = led.may_retry("orig", cls)
        self.assertFalse(ok, why)
        self.assertIn("retr", why.lower())


# --------------------------------------------------------------------------- #
# 9: token accounting
# --------------------------------------------------------------------------- #
class TestTimestampTpm(unittest.TestCase):

    def test_09_timestamp_tpm_avoids_the_batch_collapse_spike(self):
        """The 12.3M phantom, reproduced and then removed.

        Ten trials, each 300k metered tokens, spread over 90 minutes. Charging
        them at batch end concentrates 3M tokens into one 300s window; charging
        them at their real step timestamps does not.
        """
        window = 300.0
        n_trials, per_trial = 10, 300_000
        batch_end = 90 * 60.0
        collapsed = tokens.TokenTimeline(window)
        collapsed.add([tokens.TokenEvent(ts=batch_end, metered=per_trial,
                                         prompt=per_trial, completion=0, cached=0,
                                         profile="llama", trial_id=f"t{i}", step=0)
                       for i in range(n_trials)])
        spread = tokens.TokenTimeline(window)
        evs = []
        for i in range(n_trials):
            for s in range(10):
                evs.append(tokens.TokenEvent(
                    ts=(i * 9 * 60.0) + s * 54.0, metered=per_trial // 10,
                    prompt=per_trial // 10, completion=0, cached=0,
                    profile="llama", trial_id=f"t{i}", step=s))
        spread.add(evs)
        collapsed_peak, _ = collapsed.peak_tpm()
        spread_peak, _ = spread.peak_tpm()
        self.assertEqual(collapsed.total_metered, spread.total_metered,
                         "same tokens, only the attribution differs")
        self.assertGreater(collapsed_peak, 500_000)
        self.assertLess(spread_peak, collapsed_peak / 5,
                        "real per-step attribution must not spike")
        self.assertAlmostEqual(spread_peak, per_trial / (window / 60.0) * 1, delta=1e5)

    def test_09b_per_step_metrics_sum_to_final_metrics(self):
        traj = trajectory(steps=5, completion=500, prompt=4000, cached=3000)
        ex = tokens.extract_events(traj, profile="claude", trial_id="t")
        fm = traj["final_metrics"]
        want = ((fm["total_prompt_tokens"] - fm["total_cached_tokens"])
                + fm["total_completion_tokens"])
        self.assertEqual(ex.total_metered, want)
        self.assertEqual(ex.unattributed_metered, 0)
        self.assertEqual(ex.coverage, 1.0)

    def test_09c_cache_reads_are_not_metered(self):
        metered, prompt, completion, cached = tokens.metered_of(
            {"prompt_tokens": 10_000, "completion_tokens": 500, "cached_tokens": 9_500})
        self.assertEqual(metered, 1_000)

    def test_09d_both_timestamp_formats_parse(self):
        self.assertIsNotNone(tokens.parse_ts("2026-09-02T06:00:00Z"))
        self.assertIsNotNone(tokens.parse_ts("2026-09-02T06:00:00+00:00"))
        self.assertEqual(tokens.parse_ts("2026-09-02T06:00:00Z"),
                         tokens.parse_ts("2026-09-02T06:00:00+00:00"))

    def test_09e_a_timestampless_step_is_never_reassigned_to_another_time(self):
        traj = trajectory(steps=2)
        traj["steps"][0]["timestamp"] = None
        ex = tokens.extract_events(traj, profile="llama", trial_id="t")
        self.assertEqual(len(ex.events), 1)
        self.assertGreater(ex.unattributed_metered, 0,
                           "unattributable tokens are reported, not relocated")
        self.assertLess(ex.coverage, 1.0)

    def test_09f_controller_reap_records_tokens_at_step_time(self):
        with tempfile.TemporaryDirectory() as td:
            relocate_campaign_root(self, Path(td))
            fx = Fixture(Path(td))
            c = controller.Controller("full", 1, dry_run=True, paths=fx.paths,
                                      profiles=["claude"])
            run = fx.run_dir()
            tp = fx.add_trial(run, 0, "clean-n")
            c._record_tokens("claude", tp)
            times = sorted(e[0] for e in c.sch.meter._events)   # (ts, profile, tokens)
            self.assertEqual(len(times), 3, "one meter entry per step")
            self.assertGreater(max(times) - min(times), 60,
                               "step times must stay distinct, not collapse to one")


# --------------------------------------------------------------------------- #
# 10: budget polling
# --------------------------------------------------------------------------- #
class FakeClock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def fake_probe(spend=100.0, maxb=10_000.0, ok=True, error=""):
    def _p(*a, **kw):
        return lib.BudgetStatus(ok=ok, spend=spend, max_budget=maxb,
                                remaining=maxb - spend, tpm_limit=None, rpm_limit=None,
                                http_status=400 if ok else None,
                                key_fingerprint="test", error=error)
    return _p


class TestBudgetPolling(FixtureCase):

    def guard(self, probe, clock, **kw):
        return controller.BudgetGuard({p: 1.0 for p in lib.PROFILES}, probe=probe,
                                      clock=clock, poll_interval=300.0, **kw)

    def test_10_budget_is_polled_during_long_in_flight_jobs(self):
        """The failed shard checked budget only at launch, then never again."""
        clock = FakeClock(1000.0)
        calls = []

        def probe(*a, **kw):
            calls.append(clock())
            return fake_probe()(*a, **kw)

        g = self.guard(probe, clock)
        c = controller.Controller("full", 1, dry_run=True, paths=self.fx.paths,
                                  profiles=["claude"], budget=g, clock=clock)
        c.inflight["claude"] = controller.Batch(profile="claude", index=1,
                                                base_count=3, expected_trials=9)
        g.poll(force=True)
        self.assertEqual(len(calls), 1)
        for _ in range(12):                    # 12 x 500s of a single long batch
            clock.advance(500.0)
            c._poll_budget()
        self.assertGreaterEqual(len(calls), 12,
                                "budget must be re-probed while work is in flight")
        snap = c.status_snapshot()
        self.assertTrue(snap["budget_checked"])
        self.assertEqual(snap["spend_usd"], 100.0)
        self.assertEqual(snap["remaining_budget_usd"], 9900.0)
        self.assertLessEqual(snap["budget_age_sec"], 500.0,
                             "status must show a FRESH reading, not the launch one")

    def test_10b_a_failing_probe_fails_closed_on_launching_but_drains(self):
        clock = FakeClock(1000.0)
        g = self.guard(fake_probe(ok=False, error="connection reset"), clock)
        c = controller.Controller("full", 1, dry_run=True, paths=self.fx.paths,
                                  profiles=["claude"], budget=g, clock=clock)
        inflight = controller.Batch(profile="claude", index=1, base_count=3,
                                    expected_trials=9)
        c.inflight["claude"] = inflight
        clock.advance(400.0)
        c._poll_budget()
        self.assertTrue(c.draining, "cannot prove affordability -> launch nothing")
        self.assertIn("budget probe failing", c.drain_reason)
        self.assertIs(c.inflight.get("claude"), inflight,
                      "healthy in-flight work must NOT be killed by a broken probe")
        with self.assertRaises(controller.BudgetStop):
            g.assert_can_continue({"claude": 1}, what="launch")
        # but it does not drain forever
        clock.advance(controller.BUDGET_PROBE_GRACE_SEC + 1)
        with self.assertRaises(controller.BudgetStop):
            c._poll_budget()

    def test_10c_a_tight_budget_drains_instead_of_killing_healthy_work(self):
        clock = FakeClock(1000.0)
        g = self.guard(fake_probe(spend=9_800.0), clock, reserve=100.0)
        c = controller.Controller("full", 1, dry_run=True, paths=self.fx.paths,
                                  profiles=["claude"], budget=g, clock=clock)
        inflight = controller.Batch(profile="claude", index=1, base_count=3,
                                    expected_trials=300)
        c.inflight["claude"] = inflight
        clock.advance(400.0)
        c._poll_budget()
        self.assertTrue(c.draining)
        self.assertIn("draining", c.drain_reason)
        self.assertIs(c.inflight.get("claude"), inflight,
                      "a 90-minute trajectory is not killed at minute 85 to save nothing")

    def test_10d_hard_exhaustion_is_terminal(self):
        clock = FakeClock(1000.0)
        g = self.guard(fake_probe(spend=9_990.0), clock, reserve=100.0)
        c = controller.Controller("full", 1, dry_run=True, paths=self.fx.paths,
                                  profiles=["claude"], budget=g, clock=clock)
        c.inflight["claude"] = controller.Batch(profile="claude", index=1,
                                                base_count=3, expected_trials=9)
        clock.advance(400.0)
        with self.assertRaises(controller.BudgetStop) as cm:
            c._poll_budget()
        self.assertIn("exhausted", str(cm.exception))
        self.assertTrue(g.exhausted())

    def test_10e_a_probe_recovering_resumes_launching(self):
        clock = FakeClock(1000.0)
        state = {"ok": False}

        def probe(*a, **kw):
            return fake_probe(ok=state["ok"], error="" if state["ok"] else "boom")(*a, **kw)

        g = self.guard(probe, clock)
        c = controller.Controller("full", 1, dry_run=True, paths=self.fx.paths,
                                  profiles=["claude"], budget=g, clock=clock)
        c.inflight["claude"] = controller.Batch(profile="claude", index=1,
                                                base_count=1, expected_trials=1)
        clock.advance(400.0)
        c._poll_budget()
        self.assertTrue(c.draining)
        state["ok"] = True
        clock.advance(400.0)
        c._poll_budget()
        self.assertFalse(c.draining, "a recovered probe must un-block launching")

    def test_10f_a_budget_failure_in_a_trial_stops_the_shard_and_is_never_retried(self):
        c = controller.Controller("full", 1, dry_run=True, paths=self.fx.paths,
                                  profiles=["claude"])
        c.units["claude"] = []
        tp = make_trial(Path(self._td) / "bud", "t")
        (tp / "exception.txt").write_text("Budget has been exceeded", encoding="utf-8")
        trial = lib.Trial(run_dir="j", trial_dir="ea-aaaaaaaa-clean-n__A", base_task_id="b",
                          arm="clean-n", agent_name="", agent_version="", model_name="",
                          cost_usd=None, prompt_tokens=None, completion_tokens=None,
                          steps=None, status="budget_censored", reward=None, resolved=None)
        with self.assertRaises(controller.BudgetStop):
            c._handle_failure(controller.Batch(profile="claude", index=1, base_count=1,
                                               expected_trials=1), trial, tp)
        self.assertEqual(c.units["claude"], [], "a budget failure is NEVER retried")


# --------------------------------------------------------------------------- #
# 11: end state
# --------------------------------------------------------------------------- #
class TestShardCompletion(FixtureCase):

    def test_11_a_repaired_shard_validates_to_exactly_its_full_cell_count(self):
        run = self.fx.run_dir()
        planned = [(i, a) for i in range(3) for a in ARMS]
        for i, arm in planned[:7]:
            self.fx.add_trial(run, i, arm)
        first = self.fx.audit()
        v = cells.validate_shard_complete(first, expect=9)
        self.assertFalse(v["ok"])
        self.assertEqual(v["complete_valid"], 7)
        self.assertEqual(v["outstanding_total"], 2)

        # repair ONLY the outstanding cells, into a separate run directory
        plan = cells.repair_plan(first)
        repair_run = self.fx.run_dir("job-repair")
        for c in plan["by_profile"][self.fx.profile]:
            bidx = self.fx.base_ids.index(c["base_task_id"])
            self.fx.add_trial(repair_run, bidx, c["arm"], suffix="RPR1",
                              result=result_json(reward=0.0,
                                                 task_path=str(self.fx.dataset)))
        after = cells.audit(self.fx.mode, self.fx.shard, profiles=[self.fx.profile],
                            paths=self.fx.paths)
        v2 = cells.validate_shard_complete(after, expect=9)
        self.assertTrue(v2["ok"], v2["problems"])
        self.assertEqual(v2["complete_valid"], 9)
        self.assertEqual(v2["duplicates"], [])
        self.assertEqual(cells.repair_plan(after)["repair_required"], 0)
        # the 7 originals were not re-run, and are still the trials they were
        originals = [r for r in after["records"].values()
                     if any(o.trial_dir.endswith("__AAA1") for o in r.observations)]
        self.assertEqual(len(originals), 7)

    def test_11c_remaining_cost_counts_outstanding_cells_not_whole_shards(self):
        run = self.fx.run_dir()
        planned = [(i, a) for i in range(3) for a in ARMS]
        for i, arm in planned[:7]:
            self.fx.add_trial(run, i, arm)
        res = self.fx.audit()
        rp = cells.repair_plan_path(self.fx.mode, self.fx.shard, paths=self.fx.paths)
        rp.write_text(json.dumps(cells.repair_plan(res)), encoding="utf-8")
        cell = lib.Cell(self.fx.mode, self.fx.profile, self.fx.shard)
        got = cells.remaining_trials(self.fx.paths, cells=[cell])[cell.key]
        self.assertEqual(got["basis"], "repair_plan")
        self.assertEqual(got["remaining_trials"], 2,
                         "an interrupted shard costs what is left, not its full size")
        self.assertEqual(got["of_expected"], cell.expected_trials)


class TestRealShardShape(unittest.TestCase):
    """Reads the REAL campaign dataset manifests. No results, no inference."""

    def test_11b_the_real_full_shard1_is_exactly_1200_cells(self):
        expected = cells.expected_cells("full", 1)
        self.assertEqual(len(expected), 1200, "30 base tasks x 10 arms x 4 profiles")
        self.assertEqual(len({c.key for c in expected}), 1200, "cell keys are unique")
        for profile in lib.PROFILES:
            n = sum(1 for c in expected if c.profile == profile)
            self.assertEqual(n, 300, f"{profile} must own 300 cells")

    def test_11d_the_real_resource_shards_total_840_trajectories(self):
        """Sizing input for the RESOURCE Option A vs B decision."""
        total = sum(len(cells.expected_cells("resource", s)) for s in lib.SHARD_INDICES)
        self.assertEqual(total, 840, "70 base tasks x 3 arms x 4 profiles")
        treatment = sum(1 for s in lib.SHARD_INDICES
                        for c in cells.expected_cells("resource", s)
                        if c.arm == "eval-resource-scaf")
        self.assertEqual(treatment, 280, "70 base tasks x 1 arm x 4 profiles")


if __name__ == "__main__":
    unittest.main(verbosity=2)
