#!/usr/bin/env python3
"""Provider blocks are explicit stack outcomes, not missing cells.

WHAT IS UNDER TEST
------------------
On 2026-09-02 all ten FULL arms of one base task came back from fable's API
with the safety layer's answer instead of the model's: `<synthetic>` model name,
zero completion tokens, `stop_reason "refusal"`, `api_refusal_category "cyber"`.
The same task ran normally for claude, codex and llama, and the blocked set
includes `clean-n` -- no cue, no injected content -- so the block is a property
of the TASK, not of the pressure treatment.

Option A, implemented here: the cell keeps its place in the design and carries
an explicit outcome, so the corpus is rectangular with zero missing cells, and
that outcome is kept rigorously distinct from a model observation:

    accepted_observations = model_observations + provider_blocked
    missing               = 0

The three risks these tests exist to prevent, in order of severity:

  1. A blocked cell being reported, anywhere, as a model-generated refusal.
     No model saw the task. Every blocked row must say so in its own fields.
  2. A blocked cell leaking into a behavioural statistic, where its 0 tokens,
     0 steps and reward 0 would be averaged in as if a model had produced them.
  3. A blocked cell being retried until it complies, which would condition
     acceptance on the behaviour under study.

NOTHING HERE TOUCHES THE NETWORK. No Harbor job, no model call, no image build
and no benchmark trajectory is launched anywhere in this file. Every corpus in
it is fabricated in a temp directory.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from campaign import analyze, cells, lib, provenance, validate
from campaign.tests.test_repair import (
    Fixture, FixtureCase, relocate_campaign_root, result_json, trajectory,
    write_json)


BLOCKED_BASE = "task-694b4b99829f00e24fd11885"


def blocked_trajectory(*, model="<synthetic>", completion=0):
    """A trajectory shaped exactly like the 2026-09-02 fable blocks."""
    traj = trajectory(steps=1, completion=max(completion, 1), model=model)
    traj["steps"][0]["extra"] = {"stop_reason": "refusal"}
    traj["final_metrics"]["total_completion_tokens"] = completion
    traj["final_metrics"]["total_cost_usd"] = 0.02   # the prompt WAS billed
    return traj


def block_cell(fx: Fixture, run: Path, base_index: int, arm: str, *,
               suffix="BLK1", category="cyber", exception_type=None,
               model="<synthetic>", completion=0) -> Path:
    """Replace one cell's trial with the exact shape fable's blocks had."""
    existing = run / f"{fx.task_dir(base_index, arm)}__AAA1"
    if existing.exists():
        shutil.rmtree(existing)
    t = fx.add_trial(
        run, base_index, arm, suffix=suffix,
        traj=blocked_trajectory(model=model, completion=completion),
        result=result_json(
            reward=0.0, exception_type=exception_type,
            task_path=str(fx.dataset / fx.task_dir(base_index, arm))))
    if category:
        refusal_log(t, category=category)
    return t


def refusal_log(trial: Path, category="cyber") -> None:
    """The agent-side log the safety layer leaves behind."""
    (trial / "agent").mkdir(parents=True, exist_ok=True)
    (trial / "agent" / "claude-code.txt").write_text(
        '{"type":"system","subtype":"init"}\n'
        '{"type":"result","subtype":"model_refusal_no_fallback",'
        f'"api_refusal_category":"{category}"}}\n',
        encoding="utf-8")


# --------------------------------------------------------------------------- #
# 1. classification: the block is its own outcome, in lib and in cells alike
# --------------------------------------------------------------------------- #
class TestScanRunDirStatus(FixtureCase):
    """`lib.scan_run_dir` builds the corpus rows, so it must agree with cells."""

    def test_blocked_trial_is_not_synthetic(self):
        """`synthetic` means no real request; a block means the vendor killed one.

        Reusing STATUS_SYNTHETIC here would make a genuine stack outcome look
        like a harness defect -- and the only way to admit it would be to relax
        the validator's zero-synthetic check for every row.
        """
        run = self.fx.run_dir()
        t = block_cell(self.fx, run, 0, "clean-n")
        got = {x.trial_dir: x for x in lib.scan_run_dir(run)}[t.name]
        self.assertEqual(got.status, lib.STATUS_PROVIDER_BLOCKED)
        self.assertNotEqual(got.status, lib.STATUS_SYNTHETIC)
        self.assertFalse(got.model_started)
        self.assertTrue(got.provider_refusal)
        self.assertEqual(got.provider_refusal_category, "cyber")

    def test_category_is_read_from_the_vendor_log_never_invented(self):
        """An invented category would be a fabricated experimental field."""
        run = self.fx.run_dir()
        t = block_cell(self.fx, run, 0, "clean-n", category="self_harm")
        got = {x.trial_dir: x for x in lib.scan_run_dir(run)}[t.name]
        self.assertEqual(got.provider_refusal_category, "self_harm")

        # No category recorded anywhere: say so, do not guess one.
        t2 = block_cell(self.fx, run, 1, "clean-n", suffix="BBB2", category="",
                        exception_type="AgentSafetyRefusalError")
        got2 = {x.trial_dir: x for x in lib.scan_run_dir(run)}[t2.name]
        self.assertEqual(got2.status, lib.STATUS_PROVIDER_BLOCKED)
        self.assertEqual(got2.provider_refusal_category,
                         lib.REFUSAL_CATEGORY_UNKNOWN)

    def test_a_model_that_refuses_in_its_own_words_is_a_complete_observation(self):
        """The experiment's central distinction.

        Real model, real tokens, its own decision to decline. That is a RESULT.
        Re-running it until it complies would select on the behaviour under
        study, so it must never be marked blocked and never enter a repair plan.
        """
        run = self.fx.run_dir()
        t = block_cell(self.fx, run, 0, "clean-n",
                       model="anthropic/claude-fable-5", completion=900)
        got = {x.trial_dir: x for x in lib.scan_run_dir(run)}[t.name]
        self.assertEqual(got.status, lib.STATUS_COMPLETE)
        self.assertTrue(got.model_started)
        self.assertFalse(got.provider_refusal)

    def test_cells_and_lib_never_disagree(self):
        """Two classifiers, one definition. Divergence would split the corpus."""
        run = self.fx.run_dir()
        t = block_cell(self.fx, run, 0, "clean-n")
        obs = cells.classify_observation(t, campaign_root=self.fx.root)
        trial = {x.trial_dir: x for x in lib.scan_run_dir(run)}[t.name]
        self.assertEqual(obs.status, cells.PROVIDER_BLOCKED)
        self.assertEqual(trial.status, lib.STATUS_PROVIDER_BLOCKED)
        self.assertEqual(obs.model_started, trial.model_started)
        self.assertEqual(obs.provider_refusal_category,
                         trial.provider_refusal_category)


# --------------------------------------------------------------------------- #
# 2. cell accounting: accepted != model observations
# --------------------------------------------------------------------------- #
class TestAcceptedObservationAccounting(FixtureCase):

    def _all_valid_with_one_block(self):
        run = self.fx.run_dir()
        self.fx.fill_all_valid(run)
        block_cell(self.fx, run, 0, "clean-n")
        return self.fx.audit()

    def test_accepted_is_model_plus_blocked_and_nothing_is_missing(self):
        res = self._all_valid_with_one_block()
        n = res["expected"]
        self.assertEqual(res["accepted_observations"], n)
        self.assertEqual(res["model_observations"], n - 1)
        self.assertEqual(res["provider_blocked"], 1)
        self.assertEqual(res["missing"], 0)
        self.assertEqual(
            res["accepted_observations"],
            res["model_observations"] + res["provider_blocked"],
            "accepted must decompose exactly; any slack hides a cell")

    def test_the_shard_gate_reports_the_three_numbers_separately(self):
        v = cells.validate_shard_complete(self._all_valid_with_one_block())
        self.assertTrue(v["ok"])
        self.assertEqual(v["accepted_observations"], v["expected"])
        self.assertEqual(v["model_observations"], v["expected"] - 1)
        self.assertEqual(v["provider_blocked"], 1)
        self.assertEqual(v["missing"], 0)
        # `full_corpus` stays False: not every cell is a model observation, and
        # the gate must keep saying so rather than rounding up to "complete".
        self.assertFalse(v["full_corpus"])

    def test_a_missing_cell_can_never_pass_the_gate(self):
        """The rectangularity requirement, stated as a test.

        A hole is a hole whether or not blocked cells exist elsewhere; the gate
        must never report a closeable shard while any cell is unobserved.
        """
        run = self.fx.run_dir()
        self.fx.fill_all_valid(run)
        shutil.rmtree(run / f"{self.fx.task_dir(2, 'eval-src')}__AAA1")
        res = self.fx.audit()
        v = cells.validate_shard_complete(res)
        self.assertFalse(v["ok"])
        self.assertEqual(v["missing"], 1)
        self.assertLess(v["accepted_observations"], v["expected"])
        self.assertTrue(any("no observation at all" in p for p in v["problems"]))

    def test_blocked_cells_are_never_queued_for_repair(self):
        res = self._all_valid_with_one_block()
        self.assertEqual(res["repair_required"], 0)
        plan = cells.repair_plan(res)
        self.assertEqual(plan["repair_required"], 0)
        rec = res["records"][cells.CellKey(
            lib.CAMPAIGN_ID, self.fx.mode, self.fx.profile, self.fx.shard,
            self.fx.base_ids[0], "clean-n").key]
        self.assertEqual(rec.repair_outlook, cells.REPAIR_EXCLUDED)

    def test_the_audit_row_carries_the_full_explicit_contract(self):
        """Every claim about a blocked cell is written down, not inferred."""
        res = self._all_valid_with_one_block()
        key = cells.CellKey(lib.CAMPAIGN_ID, self.fx.mode, self.fx.profile,
                            self.fx.shard, self.fx.base_ids[0], "clean-n").key
        row = res["records"][key].as_dict()
        self.assertEqual(row["status"], cells.PROVIDER_BLOCKED)
        self.assertTrue(row["accepted"])
        self.assertFalse(row["model_started"])
        self.assertTrue(row["provider_refusal"])
        self.assertEqual(row["provider_refusal_category"], "cyber")
        # Design coordinates and provenance survive the block.
        self.assertEqual(row["base_task_id"], self.fx.base_ids[0])
        self.assertEqual(row["arm"], "clean-n")
        self.assertEqual(row["condition"], "clean")
        self.assertEqual(row["delivery_channel"], "none")
        self.assertEqual(row["pressure_kind"], "none")
        self.assertTrue(row["observations"][0]["run_dir"])

    def test_a_valid_cell_is_never_marked_as_a_refusal(self):
        res = self._all_valid_with_one_block()
        for rec in res["records"].values():
            row = rec.as_dict()
            if row["status"] == cells.COMPLETE_VALID:
                self.assertTrue(row["model_started"])
                self.assertFalse(row["provider_refusal"])
                self.assertEqual(row["provider_refusal_category"], "")


# --------------------------------------------------------------------------- #
# 3. the corpus builder
# --------------------------------------------------------------------------- #
class TestCorpusRows(FixtureCase):

    def test_a_blocked_attempt_is_accepted_and_labelled(self):
        """The attempt closes; the row says exactly what it is.

        If a blocked cell kept an attempt open, the only way to close the shard
        would be to re-run the block until it complied.
        """
        run = self.fx.run_dir()
        self.fx.fill_all_valid(run)
        block_cell(self.fx, run, 0, "clean-n")

        trials = {x.trial_dir: x for x in lib.scan_run_dir(run)}
        counts = {}
        for x in trials.values():
            counts[x.status] = counts.get(x.status, 0) + 1
        self.assertEqual(counts[lib.STATUS_PROVIDER_BLOCKED], 1)
        self.assertEqual(counts[lib.STATUS_COMPLETE], len(trials) - 1)
        # Which is precisely what makes `accepted == expected` in cmd_record.
        self.assertEqual(
            counts[lib.STATUS_COMPLETE] + counts[lib.STATUS_PROVIDER_BLOCKED],
            len(trials))

    def test_model_id_is_taken_only_from_trials_where_a_model_ran(self):
        """`<synthetic>` is the safety layer's placeholder, not a model id."""
        run = self.fx.run_dir()
        self.fx.fill_all_valid(run)
        block_cell(self.fx, run, 0, "clean-n")
        trials = lib.scan_run_dir(run)
        models = sorted({x.model_name for x in trials
                         if x.model_name and x.model_started})
        self.assertEqual(models, ["anthropic/claude-opus-4-8"])
        self.assertIn("<synthetic>", {x.model_name for x in trials})


# --------------------------------------------------------------------------- #
# 4. the validator gate
# --------------------------------------------------------------------------- #
class CorpusBuilder:
    """A complete, structurally faithful 3,640-row campaign corpus in a temp tree.

    Built row by row from the real design constants, so the validator runs its
    production code path unmodified against it.
    """

    def __init__(self, root: Path):
        self.root = root
        self.paths = lib.campaign_paths()
        for key in ("provenance", "manifests", "validation"):
            self.paths[key].mkdir(parents=True, exist_ok=True)
        self.rows: list[dict] = []
        self.accepted: list[dict] = []
        self._build()

    def _row(self, mode, profile, shard, base, arm, run_id):
        return {
            "campaign_id": lib.CAMPAIGN_ID,
            "cell": f"{mode}/{profile}/chunk-{shard}-size-{lib.SHARD_SIZE}",
            "mode": mode, "profile": profile,
            "shard": f"chunk-{shard}-size-{lib.SHARD_SIZE}",
            "attempt_id": f"{mode}-{profile}-s{shard}-a01",
            "run_id": run_id,
            "base_task_id": base, "arm": arm,
            "trial_dir": f"ea-{base[-8:]}-{arm}__{mode[0]}{profile[0]}{shard}",
            "status": lib.STATUS_COMPLETE,
            "agent_name": lib.VERSION_PINS[profile]["agent"],
            "agent_version": lib.VERSION_PINS[profile]["version"],
            "model_name": lib.MODEL_PINS[profile],
            "cost_usd": 1.0, "prompt_tokens": 4000, "completion_tokens": 500,
            "steps": 3, "reward": 1.0, "resolved": True,
            "shard_index": shard,
            **lib.arm_factors(mode, arm),
            "model_started": True,
            "provider_refusal": False,
            "provider_refusal_category": "",
        }

    def _build(self):
        bases = [f"task-{i:024x}" for i in range(lib.BASE_TASK_COUNT)]
        idx = 0
        for mode in lib.MODES:
            for profile in lib.PROFILES:
                for shard in lib.SHARD_INDICES:
                    n = lib.BASE_TASKS_PER_SHARD[shard]
                    run_id = f"job-{mode}-{profile}-{shard}"
                    label = f"chunk-{shard}-size-{lib.SHARD_SIZE}"
                    for base in bases[idx:idx + n]:
                        for arm in lib.ARMS[mode]:
                            self.rows.append(
                                self._row(mode, profile, shard, base, arm, run_id))
                    self.accepted.append({
                        "cell": f"{lib.CAMPAIGN_ID}/{mode}/{profile}/s{shard}",
                        "mode": mode, "profile": profile, "shard": label,
                        "shard_index": shard, "attempt_id": f"{mode}-{profile}-s{shard}-a01",
                        "run_id": run_id, "status": "complete", "superseded_by": None,
                        "run_dir": self._run_dir(mode, profile, run_id),
                    })
                    self._cell_manifest(mode, profile, label)
                    idx = (idx + n) % lib.BASE_TASK_COUNT

    def _run_dir(self, mode, profile, run_id) -> str:
        d = self.paths[mode] / run_id
        d.mkdir(parents=True, exist_ok=True)
        return str(d.relative_to(lib.PROJECT_ROOT)) if str(d).startswith(
            str(lib.PROJECT_ROOT)) else str(d)

    def _cell_manifest(self, mode, profile, label):
        ds = self.root / "datasets" / "_shards" / mode / profile / label
        ds.mkdir(parents=True, exist_ok=True)
        trials = [{"arm": a, "resource_derivation_parent": f"full/{profile}/{a}"}
                  for a in lib.ARMS[mode]]
        write_json(ds / "manifest.json", {"tasks": trials})
        out = self.paths["manifests"] / "cells" / f"{mode}__{profile}__{label}.json"
        write_json(out, {
            "dataset_path": str(ds.relative_to(lib.PROJECT_ROOT))
                            if str(ds).startswith(str(lib.PROJECT_ROOT)) else str(ds),
            "dataset_manifest_sha256": lib.sha256_file(ds / "manifest.json"),
            "trials": trials,
        })

    def block(self, profile="fable", base=None, mode="full") -> list[dict]:
        """Turn every arm of one base task into a provider block, as fable saw."""
        base = base or self.rows[0]["base_task_id"]
        hit = [r for r in self.rows
               if r["profile"] == profile and r["mode"] == mode
               and r["base_task_id"] == base]
        for r in hit:
            r.update({
                "status": lib.STATUS_PROVIDER_BLOCKED,
                "model_name": "<synthetic>",
                "completion_tokens": 0, "steps": 1, "cost_usd": 0.02,
                "reward": 0.0, "resolved": False,
                "model_started": False,
                "provider_refusal": True,
                "provider_refusal_category": "cyber",
            })
        return hit

    def write(self):
        p = self.paths["provenance"]
        (p / "corpus.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in self.rows))
        write_json(p / "accepted_runs.json", {"accepted": self.accepted})
        write_json(p / "build_report.json", {"errors": [], "ok": True})


class TestValidatorAcceptsBlockedCells(unittest.TestCase):

    def setUp(self):
        self._td = tempfile.mkdtemp(prefix="campaign-blocked-validate-")
        self.addCleanup(shutil.rmtree, self._td, ignore_errors=True)
        relocate_campaign_root(self, Path(self._td))
        self.b = CorpusBuilder(Path(self._td))

    def _run(self):
        self.b.write()
        return validate.validate()

    def _check(self, doc, cid):
        return next(c for c in doc["checks"] if c["id"] == cid)

    def test_a_clean_corpus_passes(self):
        doc = self._run()
        self.assertTrue(doc["ok"], [c for c in doc["checks"] if not c["ok"]])
        self.assertEqual(doc["accepted_observations"], doc["expected_cells"])
        self.assertEqual(doc["model_observations"], doc["expected_cells"])
        self.assertEqual(doc["provider_blocked"], 0)
        self.assertEqual(doc["missing"], 0)

    def test_ten_blocked_cells_validate_and_are_counted_apart(self):
        """Option A, end to end: rectangular corpus, two distinct populations."""
        blocked = self.b.block()
        self.assertEqual(len(blocked), lib.VARIANTS_PER_TASK["full"])
        doc = self._run()
        self.assertTrue(doc["ok"], [c for c in doc["checks"] if not c["ok"]])
        total = doc["expected_cells"]
        self.assertEqual(doc["accepted_observations"], total)
        self.assertEqual(doc["model_observations"], total - 10)
        self.assertEqual(doc["provider_blocked"], 10)
        self.assertEqual(doc["missing"], 0)
        self.assertEqual(doc["provider_blocked_categories"], ["cyber"])
        # F5 must still be a real check, not one relaxed to let these through.
        self.assertTrue(self._check(doc, "F:full/fable:5")["ok"])
        self.assertIn("0 synthetic", self._check(doc, "F:full/fable:5")["detail"])

    def test_a_missing_cell_fails_even_when_the_rest_is_perfect(self):
        """The validator must never report complete with a hole in the design."""
        self.b.rows.pop()
        doc = self._run()
        self.assertFalse(doc["ok"])
        self.assertIn("X5", doc["failed"])
        self.assertEqual(doc["missing"], 1)
        self.assertLess(doc["accepted_observations"], doc["expected_cells"])

    def test_a_blocked_row_without_its_category_is_rejected(self):
        for r in self.b.block():
            r["provider_refusal_category"] = ""
        doc = self._run()
        self.assertFalse(doc["ok"])
        self.assertIn("F:full/fable:12", doc["failed"])
        self.assertIn("no vendor refusal category",
                      self._check(doc, "F:full/fable:12")["detail"])

    def test_a_blocked_row_claiming_model_output_is_rejected(self):
        """The one lie that would corrupt every downstream behavioural number."""
        for r in self.b.block():
            r["model_started"] = True
            r["completion_tokens"] = 500
        doc = self._run()
        self.assertFalse(doc["ok"])
        self.assertIn("F:full/fable:11", doc["failed"])
        self.assertIn("F:full/fable:12", doc["failed"])

    def test_a_blocked_row_scored_as_a_solve_is_rejected(self):
        for r in self.b.block():
            r["resolved"] = True
            r["reward"] = 1.0
        doc = self._run()
        self.assertFalse(doc["ok"])
        self.assertIn("scored as if a model had run",
                      self._check(doc, "F:full/fable:12")["detail"])

    def test_a_blocked_row_stripped_of_its_design_coordinates_is_rejected(self):
        for r in self.b.block():
            r["pressure_kind"] = ""
        doc = self._run()
        self.assertFalse(doc["ok"])
        self.assertIn("lost design coordinates",
                      self._check(doc, "F:full/fable:12")["detail"])

    def test_a_genuinely_synthetic_row_still_fails(self):
        """The zero-synthetic gate is not weakened by admitting blocks."""
        self.b.rows[0]["status"] = lib.STATUS_SYNTHETIC
        self.b.rows[0]["model_name"] = "<synthetic>"
        doc = self._run()
        self.assertFalse(doc["ok"])
        prof = self.b.rows[0]["profile"]
        self.assertIn(f"F:full/{prof}:5", doc["failed"])


# --------------------------------------------------------------------------- #
# 5. the analysis gate and the complete-case sensitivity analysis
# --------------------------------------------------------------------------- #
def corpus_rows(root: Path) -> list[dict]:
    b = CorpusBuilder(root)
    b.block()
    return b.rows


class TestAnalysisGate(unittest.TestCase):

    def setUp(self):
        self._td = tempfile.mkdtemp(prefix="campaign-blocked-analyze-")
        self.addCleanup(shutil.rmtree, self._td, ignore_errors=True)
        relocate_campaign_root(self, Path(self._td))
        self.rows = corpus_rows(Path(self._td))

    def test_behavioural_analysis_refuses_ungated_rows(self):
        """Fail loud rather than filter: a silent filter hides the exclusion."""
        with self.assertRaises(analyze.NonModelRowsInAnalysis) as cm:
            analyze.summarise(self.rows)
        self.assertIn("model_started", str(cm.exception))
        self.assertIn("model_rows", str(cm.exception))

    def test_the_gate_removes_exactly_the_blocked_cells(self):
        gated = analyze.model_rows(self.rows)
        self.assertEqual(len(self.rows) - len(gated), 10)
        self.assertTrue(all(r["status"] == lib.STATUS_COMPLETE for r in gated))
        summary = analyze.summarise(gated)
        self.assertEqual(summary["population"], "model_observations")
        self.assertEqual(summary["n_trials"], len(self.rows) - 10)

    def test_no_arm_statistic_is_computed_over_a_blocked_cell(self):
        """Ten arms lose one observation each -- and only one each."""
        gated = analyze.model_rows(self.rows)
        per_arm = {(a["profile"], a["arm"]): a["n"]
                   for a in analyze.summarise(gated)["arms"]
                   if a["mode"] == "full"}
        for arm in lib.ARMS["full"]:
            self.assertEqual(per_arm[("fable", arm)] + 1,
                             per_arm[("claude", arm)],
                             f"arm {arm} should be short exactly one fable cell")

    def test_stack_view_keeps_the_blocked_cells_as_observed_failures(self):
        """The stack DID fail to deliver a solution. That is real, and counted."""
        st = analyze.stack_outcomes(self.rows)
        fable = next(r for r in st["by_mode_profile"]
                     if r["mode"] == "full" and r["profile"] == "fable")
        claude = next(r for r in st["by_mode_profile"]
                      if r["mode"] == "full" and r["profile"] == "claude")
        self.assertEqual(fable["accepted_observations"],
                         claude["accepted_observations"])
        self.assertEqual(fable["model_observations"],
                         claude["model_observations"] - 10)
        self.assertEqual(fable["provider_blocked"], 10)
        self.assertEqual(fable["provider_blocked_categories"], ["cyber"])
        self.assertLess(fable["stack_resolved_rate"], claude["stack_resolved_rate"])

    def test_sensitivity_drops_the_blocked_base_task_for_every_profile(self):
        """The point of the complete-case analysis: one task set, all models."""
        sens = analyze.sensitivity_complete_cases(self.rows)
        self.assertEqual(sens["base_tasks_total"], lib.BASE_TASK_COUNT)
        self.assertEqual(sens["base_tasks_included"], lib.BASE_TASK_COUNT - 1)
        dropped = sens["base_tasks_dropped"]
        self.assertEqual(len(dropped), 1)
        self.assertEqual(len(sens["drop_reason"][dropped[0]]), 10)

        kept = analyze.complete_case_base_tasks(self.rows)
        by_profile = {}
        for r in self.rows:
            if r["mode"] == "full" and r["base_task_id"] in kept:
                by_profile.setdefault(r["profile"], set()).add(
                    (r["base_task_id"], r["arm"]))
        self.assertEqual(len({frozenset(v) for v in by_profile.values()}), 1,
                         "every profile must be summarised over the identical "
                         "base-task/arm set, or the models are not comparable")

    def test_sensitivity_is_balanced_where_the_headline_is_not(self):
        sens = analyze.sensitivity_complete_cases(self.rows)
        ns = {(a["mode"], a["profile"], a["arm"]): a["n"] for a in sens["arms"]}
        full_ns = {v for k, v in ns.items() if k[0] == "full"}
        self.assertEqual(len(full_ns), 1,
                         f"complete-case FULL arms must all have equal n, got {full_ns}")

    def test_an_unblocked_corpus_drops_nothing(self):
        """The sensitivity analysis is a no-op when the design is already full."""
        clean = CorpusBuilder(Path(self._td)).rows
        sens = analyze.sensitivity_complete_cases(clean)
        self.assertEqual(sens["base_tasks_dropped"], [])
        self.assertEqual(sens["base_tasks_included"], lib.BASE_TASK_COUNT)


if __name__ == "__main__":
    unittest.main()
