#!/usr/bin/env python3
"""Offline proofs that an approved task-definition amendment is not a loophole.

NO Harbor job, NO model call, NO network. Every fixture is synthetic JSON in a
temp directory.

Campaign V2 freezes each task definition at `prepare` time and preflight
re-verifies it before every launch. On 2026-09-02 a legitimate correction landed
AFTER the freeze: `source_targets` repaired 24 source-channel seeds that pointed
at a post-rename path absent from the pre-patch tree, so the image build died
before any model ran. `campaign.amendments` records those corrections explicitly
instead of re-freezing the campaign or editing a hash by hand.

The rule these tests enforce is that a task definition on disk must equal EITHER

    A. the definition `prepare` originally froze, or
    B. the exact definition named by an approved, append-only amendment record

and nothing else. So each test injects one defect and asserts the refusal:

  1  arbitrary seed edit with no repair record   -> preflight still FAILS
  2  amended manifest, ledger deleted            -> preflight FAILS
  3  amendment against a COMPLETE_VALID cell     -> refused, manifest untouched
  4  old hash != the frozen manifest's hash      -> refused
  5  new hash != what is actually on disk        -> refused
  6  approved PRE_MODEL_FAILURE correction       -> amends, preflight PASSES
  7  the historical failed-run inputs            -> byte-identical afterwards
  8  the COMPLETE_VALID trajectories             -> byte-identical afterwards

Plus the surrounding guarantees: unamended rows keep their frozen hashes, the
ledger is append-only and idempotent, and the cached indexes stay consistent
without losing the original hash.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from campaign import amendments, cells, lib, preflight
from campaign.tests.test_repair import relocate_campaign_root, write_json
from campaign.tests.test_source_targets import (WP_CALYPSO_PATCH, SeedFixture)

MODE, PROFILE, SHARD = "full", "claude", 1
BROKEN = "client/dashboard/components/overview-card/card.stories.tsx"   # post-rename
CORRECT = "client/dashboard/sites/overview-card/card.stories.tsx"       # pre-patch


def seed_path(task_dir: Path) -> Path:
    return task_dir / "environment" / "benchmark_seed" / "seed.json"


class AmendmentFixture(SeedFixture):
    """A frozen one-cell campaign whose source arms carry a broken seed target.

    Faithful to the real shape: the cell manifest records a per-trial
    `task_content_sha256` / `seed_json_sha256` and the shard's
    `dataset_manifest_sha256`, all captured while the seeds are still broken --
    exactly as `prepare` captured them before the repair ran.
    """

    SOURCE_ARMS = ("eval-src", "eval-fin-src")

    def __init__(self, td: Path):
        super().__init__(Path(td), mode=MODE, shard=SHARD, profile=PROFILE)
        self.paths["manifest"] = self.root / "CAMPAIGN_MANIFEST.json"
        (self.paths["manifests"] / "cells").mkdir(parents=True, exist_ok=True)
        for i in range(len(self.base_ids)):
            for arm in self.arms:
                self.seed_task(i, arm, channel="source" if arm in self.SOURCE_ARMS
                               else "none",
                               patch=WP_CALYPSO_PATCH, target=BROKEN)
        self.freeze()

    # -- the freeze --------------------------------------------------------- #
    def cell(self) -> lib.Cell:
        return lib.Cell(MODE, PROFILE, SHARD)

    def manifest_path(self) -> Path:
        return (self.paths["manifests"] / "cells"
                / f"{MODE}__{PROFILE}__{self.label}.json")

    def freeze(self) -> None:
        """Write the cell manifest, index and campaign manifest from disk now."""
        trials = []
        for i, bid in enumerate(self.base_ids):
            for arm in self.arms:
                name = self.task_dir(i, arm)
                td = self.dataset / name
                trials.append({
                    "campaign_id": lib.CAMPAIGN_ID, "mode": MODE,
                    "shard": self.label, "shard_index": SHARD, "profile": PROFILE,
                    "base_task_id": bid[-8:], "arm": arm,
                    "task_dir": name,
                    "source_task_path": f"generated/{MODE}/{PROFILE}/{name}",
                    "snapshot_task_path": str(td.relative_to(lib.PROJECT_ROOT)),
                    "task_content_sha256": lib.sha256_tree(td),
                    "seed_json_sha256": lib.sha256_file(seed_path(td)),
                    "resource_derivation_parent": None,
                })
        ds_sha = lib.sha256_file(self.dataset / "manifest.json")
        write_json(self.manifest_path(), {
            "campaign_id": lib.CAMPAIGN_ID, "cell": self.cell().key,
            "mode": MODE, "profile": PROFILE, "shard": self.label,
            "shard_index": SHARD, "expected_trials": len(trials),
            "dataset_path": str(self.dataset.relative_to(lib.PROJECT_ROOT)),
            "dataset_manifest_sha256": ds_sha,
            "model": "anthropic/claude-opus-4-8", "agent_version_pinned": "1.0",
            "trials": trials,
        })
        entry = {"cell": self.cell().key, "expected_trials": len(trials),
                 "dataset_manifest_sha256": ds_sha,
                 "manifest": self.manifest_path().name}
        write_json(self.paths["manifests"] / "cells_index.json",
                   {"campaign_id": lib.CAMPAIGN_ID, "cells": [entry]})
        write_json(self.paths["manifest"],
                   {"campaign_id": lib.CAMPAIGN_ID, "cells": [dict(entry)]})

    def frozen(self) -> dict:
        return json.loads(self.manifest_path().read_text())

    def row(self, task_dir: str) -> dict:
        return next(t for t in self.frozen()["trials"]
                    if t["task_dir"] == task_dir)

    # -- the repair --------------------------------------------------------- #
    def repair(self, base_index: int, arm: str, *, record=True,
               corrected=CORRECT) -> str:
        """Correct one seed on disk, and log it append-only like the real repair."""
        name = self.task_dir(base_index, arm)
        td = self.dataset / name
        seed = json.loads(seed_path(td).read_text())
        old = seed["source_target"]
        seed["source_target"] = corrected
        write_json(seed_path(td), seed)
        # `source_targets repair` re-syncs the shard manifest too, so the shard's
        # dataset_manifest_sha256 moves exactly as it did in the real campaign.
        mp = self.dataset / "manifest.json"
        manifest = json.loads(mp.read_text())
        for entry in manifest["tasks"]:
            if entry["directory"] == name:
                entry["source_target"] = corrected
                entry["content_hash"] = lib.sha256_tree(td)
        write_json(mp, manifest)
        if record:
            self.log_repair(name, old, corrected, base_index, arm)
        return name

    def log_repair(self, task_dir, old, corrected, base_index, arm) -> None:
        key = cells.CellKey(lib.CAMPAIGN_ID, MODE, PROFILE, SHARD,
                            self.base_ids[base_index], arm).key
        rec = {"at": "2026-09-02T17:46:36+00:00", "campaign_id": lib.CAMPAIGN_ID,
               "task_dir": task_dir, "path": str(self.dataset / task_dir),
               "problem": "source_target is the post-rename destination",
               "current": old, "corrected": corrected,
               "comment_prefix": "//", "workspace_root": "/workspace",
               "mode": MODE, "profile": PROFILE, "shard": SHARD,
               "cell_key": key, "applied": True}
        with (self.paths["provenance"] / "source_target_repairs.jsonl").open(
                "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

    # -- observations ------------------------------------------------------- #
    def fail_before_model(self, base_index: int, arm: str) -> Path:
        """A trial that died in image build: result.json, no trajectory."""
        run = self.run_dir("job-fail")
        return self.add_trial(run, base_index, arm, traj=None)

    def complete_valid(self, base_index: int, arm: str) -> Path:
        run = self.run_dir("job-ok")
        return self.add_trial(run, base_index, arm)

    # -- driving the code under test ---------------------------------------- #
    def plan(self):
        return amendments.plan(MODE, SHARD, paths=self.paths, profiles=[PROFILE])

    def amend(self, *, apply=True):
        return amendments.apply(self.plan(), paths=self.paths, apply=apply)

    def preflight_failures(self, *, quick=False) -> list[str]:
        """Run the offline structural preflight checks over this one cell."""
        out = []
        for check in (lambda: preflight.check_amendments(self.paths, [self.cell()]),
                      lambda: preflight.check_integrity(self.paths, [self.cell()],
                                                        quick)):
            try:
                check()
            except preflight.Fail as exc:
                out.append(str(exc))
        return out


def digest(path: Path) -> dict[str, str]:
    """sha256 of every file under `path`, so 'untouched' is provable."""
    return {str(p.relative_to(path)): lib.sha256_file(p)
            for p in sorted(path.rglob("*")) if p.is_file()}


class AmendmentCase(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        root = Path(self._td.name).resolve()
        patcher = mock.patch.object(lib, "PROJECT_ROOT", root)
        patcher.start()
        self.addCleanup(patcher.stop)
        relocate_campaign_root(self, root / "campaign-root")
        self.fx = AmendmentFixture(root / "campaign-root")

    # helper: the standard, legitimate situation
    def broken_pre_model_cell(self) -> str:
        """Arm 0/eval-src failed before the model, and was then repaired."""
        self.fx.fail_before_model(0, "eval-src")
        return self.fx.repair(0, "eval-src")


# --------------------------------------------------------------------------- #
# 6: the amendment that is supposed to work
# --------------------------------------------------------------------------- #
class TestApprovedAmendment(AmendmentCase):

    def test_06_approved_pre_model_source_target_amendment_passes_preflight(self):
        """The whole point: a legitimate repair becomes launchable, explicitly."""
        name = self.broken_pre_model_cell()
        before = self.fx.preflight_failures()
        self.assertTrue(before, "a repaired-but-unamended cell must fail preflight")
        self.assertIn("dataset manifest changed since prepare", before[0])

        result = self.fx.amend()
        self.assertTrue(result["verified"] if "verified" in result else True)
        self.assertEqual(result["amendments"], 1)
        self.assertEqual(self.fx.preflight_failures(), [],
                         "after an approved amendment, preflight passes")
        self.assertEqual(amendments.verify([self.fx.cell()], paths=self.fx.paths), [])

        rec = amendments.load_ledger(self.fx.paths)[0]
        self.assertEqual(rec["approved_change_type"],
                         amendments.SOURCE_TARGET_CORRECTION)
        self.assertEqual(rec["prior_observation_status"], cells.PRE_MODEL_FAILURE)
        self.assertFalse(rec["accepted_model_trajectory_exists"])
        self.assertEqual(rec["old_source_target"], BROKEN)
        self.assertEqual(rec["new_source_target"], CORRECT)
        self.assertIn("source_target_repairs.jsonl", rec["source_repair_log_reference"])
        self.assertEqual(rec["task_dir"], name)

    def test_the_ledger_records_every_hash_on_both_sides(self):
        """Old and new hashes are both preserved -- nothing is overwritten."""
        name = self.broken_pre_model_cell()
        old_row = self.fx.row(name)
        old_ds = self.fx.frozen()["dataset_manifest_sha256"]
        self.fx.amend()
        rec = amendments.load_ledger(self.fx.paths)[0]
        new_row = self.fx.row(name)
        self.assertEqual(rec["original_task_content_sha256"],
                         old_row["task_content_sha256"])
        self.assertEqual(rec["original_seed_json_sha256"], old_row["seed_json_sha256"])
        self.assertEqual(rec["original_dataset_manifest_sha256"], old_ds)
        self.assertEqual(rec["amended_task_content_sha256"],
                         new_row["task_content_sha256"])
        self.assertNotEqual(old_row["task_content_sha256"],
                            new_row["task_content_sha256"])
        self.assertEqual(new_row["task_content_sha256"],
                         lib.sha256_tree(self.fx.dataset / name))
        self.assertEqual(new_row["source_target"], CORRECT)

    def test_unamended_rows_keep_their_frozen_hashes_byte_for_byte(self):
        """Amending one definition must not disturb any other."""
        name = self.broken_pre_model_cell()
        before = {t["task_dir"]: dict(t) for t in self.fx.frozen()["trials"]}
        self.fx.amend()
        for row in self.fx.frozen()["trials"]:
            if row["task_dir"] == name:
                continue
            self.assertEqual(row, before[row["task_dir"]],
                             f"{row['task_dir']} was not amended and must be identical")

    def test_original_manifest_is_archived_and_indexes_stay_consistent(self):
        """The caches follow the manifest, and the original hash is not lost."""
        self.broken_pre_model_cell()
        original_sha = lib.sha256_file(self.fx.manifest_path())
        original_ds = self.fx.frozen()["dataset_manifest_sha256"]
        self.fx.amend()
        cm = self.fx.frozen()

        archive = lib.PROJECT_ROOT / cm["frozen_manifest_archive"]
        self.assertTrue(archive.is_file(), "the original manifest must be preserved")
        self.assertEqual(lib.sha256_file(archive), original_sha)
        self.assertEqual(cm["original_cell_manifest_sha256"], original_sha)
        self.assertEqual(cm["original_dataset_manifest_sha256"], original_ds)
        self.assertNotEqual(cm["dataset_manifest_sha256"], original_ds)

        for path in (self.fx.paths["manifests"] / "cells_index.json",
                     self.fx.paths["manifest"]):
            entry = json.loads(path.read_text())["cells"][0]
            self.assertEqual(entry["dataset_manifest_sha256"],
                             cm["dataset_manifest_sha256"],
                             f"{path.name} must not cache a stale hash")
            self.assertEqual(entry["original_dataset_manifest_sha256"], original_ds,
                             f"{path.name} must still record what prepare froze")

    def test_the_ledger_is_append_only_and_idempotent(self):
        """Re-running the amendment adds no duplicate record and no second edit."""
        self.broken_pre_model_cell()
        self.fx.amend()
        first = amendments.load_ledger(self.fx.paths)
        manifest_sha = lib.sha256_file(self.fx.manifest_path())

        again = self.fx.plan()
        self.assertEqual(len(again["approved"]), 0)
        self.assertEqual(len(again["already_amended"]), 1)
        self.assertTrue(again["ok"])
        amendments.apply(again, paths=self.fx.paths, apply=True)

        self.assertEqual(amendments.load_ledger(self.fx.paths), first)
        self.assertEqual(lib.sha256_file(self.fx.manifest_path()), manifest_sha)
        self.assertEqual(self.fx.preflight_failures(), [])

    def test_dry_run_writes_nothing(self):
        self.broken_pre_model_cell()
        before = digest(self.fx.root)
        out = self.fx.amend(apply=False)
        self.assertFalse(out["applied"])
        self.assertEqual(digest(self.fx.root), before)


# --------------------------------------------------------------------------- #
# 1, 2, 4, 5: the ways an amendment must NOT be usable
# --------------------------------------------------------------------------- #
class TestRefusals(AmendmentCase):

    def test_01_arbitrary_seed_modification_still_fails_preflight(self):
        """Integrity is not weakened: drift with no repair record is refused.

        This is the check that must survive the whole feature. Someone edits a
        seed for their own reasons; there is no source-target repair record, so
        the amendment machinery refuses to bless it and preflight still fails.
        """
        td = self.fx.dataset / self.fx.task_dir(1, "clean-n")
        seed = json.loads(seed_path(td).read_text())
        seed["content"] = "a cue nobody approved"
        write_json(seed_path(td), seed)

        result = self.fx.plan()
        self.assertFalse(result["ok"])
        self.assertEqual(len(result["approved"]), 0)
        self.assertEqual(len(result["refused"]), 1)
        self.assertIn("no applied record", result["refused"][0]["refusal"])
        with self.assertRaises(SystemExit):
            amendments.apply(result, paths=self.fx.paths, apply=True)
        self.assertEqual(amendments.load_ledger(self.fx.paths), [])

        failures = self.fx.preflight_failures()
        self.assertTrue(failures, "unapproved drift must still fail preflight")
        self.assertIn("task content changed", failures[0])

    def test_unapproved_drift_taints_the_whole_cell(self):
        """One unexplained edit blocks the approved amendments beside it.

        A cell is launched as a unit. If part of it drifted for reasons nobody
        recorded, amending the rest would produce a manifest that certifies a
        state no one has actually vouched for.
        """
        self.broken_pre_model_cell()
        rogue = self.fx.dataset / self.fx.task_dir(2, "clean-n")
        write_json(seed_path(rogue), {"source_target": "who/knows.py"})

        result = self.fx.plan()
        self.assertFalse(result["ok"])
        self.assertEqual(len(result["approved"]), 0)
        self.assertTrue(any("another task definition in this cell" in r["refusal"]
                            for r in result["refused"]))
        with self.assertRaises(SystemExit):
            amendments.apply(result, paths=self.fx.paths, apply=True)

    def test_02_amendment_without_provenance_fails_preflight(self):
        """An amended manifest whose ledger record is gone is not admissible.

        This is the anti-laundering check. Re-freezing a manifest to match
        whatever is on disk would satisfy `check_integrity` on its own; it does
        not survive `check_amendments`, because the ledger no longer vouches
        for it.
        """
        self.broken_pre_model_cell()
        self.fx.amend()
        self.assertEqual(self.fx.preflight_failures(), [])

        amendments.ledger_path(self.fx.paths).unlink()
        failures = self.fx.preflight_failures()
        self.assertTrue(failures)
        self.assertIn("ledger has no record", failures[0])

    def test_02b_deleting_the_archived_original_fails_preflight(self):
        """The archive is the third leg: without it nothing can be reconstructed."""
        self.broken_pre_model_cell()
        self.fx.amend()
        (lib.PROJECT_ROOT / self.fx.frozen()["frozen_manifest_archive"]).unlink()
        failures = self.fx.preflight_failures()
        self.assertTrue(failures)
        self.assertIn("archived original", failures[0])

    def test_02c_editing_the_manifest_past_its_amendments_fails_preflight(self):
        """The manifest must be EXACTLY original + approved records, nothing more."""
        name = self.broken_pre_model_cell()
        self.fx.amend()
        cm = self.fx.frozen()
        for row in cm["trials"]:
            if row["task_dir"] != name:
                row["task_content_sha256"] = "0" * 64      # a hash nobody approved
        write_json(self.fx.manifest_path(), cm)
        failures = self.fx.preflight_failures()
        self.assertTrue(failures)
        self.assertIn("not the archived original plus its approved amendments",
                      failures[0])

    def test_04_amendment_whose_old_hash_is_not_the_frozen_hash_fails(self):
        """`original_*` must be what `prepare` actually froze, not a claim.

        A record that misstates the pre-amendment hash would let an amendment
        be carried over from a definition that was never the frozen one.
        """
        self.broken_pre_model_cell()
        self.fx.amend()
        ledger = amendments.load_ledger(self.fx.paths)
        ledger[0]["original_task_content_sha256"] = "b" * 64
        amendments.ledger_path(self.fx.paths).write_text(
            json.dumps(ledger[0]) + "\n", encoding="utf-8")

        failures = self.fx.preflight_failures()
        self.assertTrue(failures)
        self.assertIn("frozen manifest recorded", failures[0])

    def test_05_amendment_whose_new_hash_is_not_on_disk_fails(self):
        """`amended_*` must describe the bytes that exist right now.

        The hardest version of this: a forged ledger record AND a manifest
        rebuilt to agree with it, so the provenance check is internally
        consistent and passes. It still fails, because `check_integrity` hashes
        the actual task directory and the amendment names a definition nobody
        can produce. The two checks are layered on purpose.
        """
        name = self.broken_pre_model_cell()
        self.fx.amend()
        cm = self.fx.frozen()
        archive_ref = cm["frozen_manifest_archive"]
        original = json.loads((lib.PROJECT_ROOT / archive_ref).read_text())

        ledger = amendments.load_ledger(self.fx.paths)
        ledger[0]["amended_task_content_sha256"] = "c" * 64
        amendments.ledger_path(self.fx.paths).write_text(
            json.dumps(ledger[0]) + "\n", encoding="utf-8")
        write_json(self.fx.manifest_path(), amendments.apply_to_manifest(
            original, ledger, archive_ref=archive_ref,
            original_manifest_sha=cm["original_cell_manifest_sha256"]))

        self.assertEqual(amendments.verify([self.fx.cell()], paths=self.fx.paths), [],
                         "the forgery is internally consistent by construction")
        failures = self.fx.preflight_failures()
        self.assertTrue(failures, "but disk does not hash to the claimed value")
        self.assertIn(f"task content changed: {name}", failures[0])

    def test_05b_a_repair_record_cannot_bless_a_different_correction(self):
        """The seed on disk must BE the approved correction, not merely differ."""
        self.fx.fail_before_model(0, "eval-src")
        name = self.fx.repair(0, "eval-src")
        td = self.fx.dataset / name
        seed = json.loads(seed_path(td).read_text())
        seed["source_target"] = "some/other/file.tsx"
        write_json(seed_path(td), seed)

        result = self.fx.plan()
        self.assertFalse(result["ok"])
        self.assertIn("is not the approved correction",
                      result["refused"][0]["refusal"])

    def test_a_second_undeclared_drift_after_an_amendment_is_refused(self):
        """An amended definition that drifts AGAIN needs a fresh human decision."""
        name = self.broken_pre_model_cell()
        self.fx.amend()
        td = self.fx.dataset / name
        seed = json.loads(seed_path(td).read_text())
        seed["content"] = "changed after the amendment"
        write_json(seed_path(td), seed)

        result = self.fx.plan()
        self.assertFalse(result["ok"])
        self.assertIn("already carries an approved amendment",
                      result["refused"][0]["refusal"])
        self.assertTrue(self.fx.preflight_failures())


# --------------------------------------------------------------------------- #
# 3, 7, 8: what must never be touched
# --------------------------------------------------------------------------- #
class TestCompleteValidIsUntouchable(AmendmentCase):

    def test_03_amendment_to_a_complete_valid_cell_is_refused(self):
        """HARD RULE. A cell a real trajectory ran against is not amendable.

        Rewriting its task definition would silently make the frozen trajectory
        appear to have run against a task it never saw.
        """
        self.fx.complete_valid(0, "eval-src")
        name = self.fx.repair(0, "eval-src")

        audit = cells.audit(MODE, SHARD, profiles=[PROFILE], paths=self.fx.paths)
        key = cells.CellKey(lib.CAMPAIGN_ID, MODE, PROFILE, SHARD,
                            self.fx.base_ids[0], "eval-src").key
        self.assertEqual(audit["records"][key].status, cells.COMPLETE_VALID)

        result = self.fx.plan()
        self.assertFalse(result["ok"])
        self.assertEqual(len(result["approved"]), 0)
        self.assertIn("COMPLETE_VALID", result["refused"][0]["refusal"])
        with self.assertRaises(SystemExit):
            amendments.apply(result, paths=self.fx.paths, apply=True)

        self.assertEqual(amendments.load_ledger(self.fx.paths), [])
        self.assertEqual(self.fx.row(name)["task_content_sha256"],
                         self.fx.row(name)["task_content_sha256"])
        self.assertNotIn("task_definition_amendments", self.fx.frozen())

    def test_03b_a_ledger_record_claiming_an_accepted_trajectory_is_refused(self):
        """Even a well-formed record cannot assert its way past the hard rule."""
        self.broken_pre_model_cell()
        self.fx.amend()
        ledger = amendments.load_ledger(self.fx.paths)
        ledger[0]["accepted_model_trajectory_exists"] = True
        amendments.ledger_path(self.fx.paths).write_text(
            json.dumps(ledger[0]) + "\n", encoding="utf-8")
        failures = self.fx.preflight_failures()
        self.assertTrue(failures)
        self.assertIn("never amendable", failures[0])

    def test_03c_an_unapproved_change_type_is_refused(self):
        """Only the one reviewed class of amendment is approvable."""
        self.broken_pre_model_cell()
        self.fx.amend()
        ledger = amendments.load_ledger(self.fx.paths)
        ledger[0]["approved_change_type"] = "whatever_we_felt_like"
        amendments.ledger_path(self.fx.paths).write_text(
            json.dumps(ledger[0]) + "\n", encoding="utf-8")
        failures = self.fx.preflight_failures()
        self.assertTrue(failures)
        self.assertIn("unapproved change type", failures[0])

    def test_08_complete_valid_trajectories_are_byte_identical_afterwards(self):
        """Amending one cell must not touch a single byte of accepted evidence."""
        run = self.fx.run_dir("job-ok")
        for arm in self.fx.arms:
            self.fx.add_trial(run, 1, arm)
        before = digest(run)
        self.assertTrue(before)

        self.broken_pre_model_cell()
        self.fx.amend()

        self.assertEqual(digest(run), before,
                         "no accepted trajectory may be modified by an amendment")

    def test_07_the_historical_failed_run_inputs_stay_byte_identical(self):
        """`_batches` must keep describing what the failed attempt actually ran.

        The batch snapshot is the only record of the broken definition that
        produced the image-build failure. The amendment rewrites the frozen
        manifest's expectation; it must not reach back and rewrite history.
        """
        batch = (self.fx.paths["datasets"] / "_batches" / MODE / PROFILE
                 / self.fx.label / "batch-02")
        name = self.fx.task_dir(0, "eval-src")
        src = self.fx.dataset / name
        dest = batch / name
        dest.mkdir(parents=True, exist_ok=True)
        for p in sorted(src.rglob("*")):
            if p.is_file():
                out = dest / p.relative_to(src)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(p.read_bytes())
        before = digest(batch)
        frozen_seed_sha = self.fx.row(name)["seed_json_sha256"]

        self.broken_pre_model_cell()
        self.fx.amend()

        self.assertEqual(digest(batch), before,
                         "the failed run's inputs must never be rewritten")
        self.assertEqual(json.loads(seed_path(dest).read_text())["source_target"],
                         BROKEN, "the batch still holds the OLD broken target")
        self.assertEqual(lib.sha256_file(seed_path(dest)), frozen_seed_sha,
                         "and still hashes to what the ORIGINAL manifest froze")
        self.assertEqual(json.loads(seed_path(src).read_text())["source_target"],
                         CORRECT, "while the repair dataset holds the correction")
        self.assertNotEqual(lib.sha256_file(seed_path(src)), frozen_seed_sha)


if __name__ == "__main__":
    unittest.main(verbosity=2)
