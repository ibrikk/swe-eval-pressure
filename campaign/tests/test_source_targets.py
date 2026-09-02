#!/usr/bin/env python3
"""Offline tests for source-channel seed targets and provider-blocked cells.

NO Harbor job, NO model call, NO image build, NO benchmark trajectory is
launched anywhere in this file. Every fixture is a temporary directory holding
synthetic JSON and synthetic diffs.

These cover the two failure families that took 22 cells out of FULL shard 1 on
2026-09-02:

  A. 12 cells died at image build because the source-channel seed pointed at a
     path the gold patch RENAMES INTO, which does not exist in the pre-patch
     tree the image is built from.
       1  a rename section resolves to its pre-patch path
       2  a created file is not a candidate at all, however long the header
       3  a plain modification is unchanged (no regression on the other 68)
       4  the real wp-calypso patch shape selects the pre-rename path
       5  a prepared task dir with a rename-destination target is flagged
       6  repair rewrites seed AND manifest together
       7  repair REFUSES a cell that already holds a valid trajectory
       8  manifest sync never touches the frozen record of a past attempt

  B. 10 cells came back as API-layer refusals with no model execution.
       9  synthetic model + 0 tokens + refusal evidence -> PROVIDER_BLOCKED
      10  a MODEL's own refusal with real output stays COMPLETE_VALID
      11  PROVIDER_BLOCKED is excluded from the repair plan, not queued
      12  a corrected task definition reopens a cell written off as futile
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from campaign import cells, lib, source_targets
from campaign.tests.test_repair import (Fixture, relocate_campaign_root,
                                        result_json, trajectory, write_json)

# The shape that actually broke: a 100%-similarity rename, then a modification.
WP_CALYPSO_PATCH = """\
diff --git a/client/dashboard/sites/overview-card/card.stories.tsx \
b/client/dashboard/components/overview-card/card.stories.tsx
similarity index 100%
rename from client/dashboard/sites/overview-card/card.stories.tsx
rename to client/dashboard/components/overview-card/card.stories.tsx
diff --git a/client/dashboard/sites/overview-card/index.tsx \
b/client/dashboard/components/overview-card/index.tsx
similarity index 92%
rename from client/dashboard/sites/overview-card/index.tsx
rename to client/dashboard/components/overview-card/index.tsx
--- a/client/dashboard/sites/overview-card/index.tsx
+++ b/client/dashboard/components/overview-card/index.tsx
@@ -1,3 +1,3 @@
-const a = 1;
+const a = 2;
"""

MODIFY_PATCH = """\
diff --git a/src/lib/handler.py b/src/lib/handler.py
index 1111111..2222222 100644
--- a/src/lib/handler.py
+++ b/src/lib/handler.py
@@ -1,2 +1,2 @@
-x = 1
+x = 2
"""

# A created file whose header is long enough to fall outside the old 500-byte
# window that `new file mode` used to be searched in.
_LONG = "a/" + "deeply/" * 40 + "nested/generated_module.py"
CREATE_PATCH = f"""\
diff --git {_LONG} b/{_LONG[2:]}
new file mode 100644
index 0000000..3333333
--- /dev/null
+++ b/{_LONG[2:]}
@@ -0,0 +1 @@
+created = True
"""


def refusal_trial(model="<synthetic>", completion=0):
    """Trial artifacts matching the 2026-09-02 fable refusals."""
    traj = trajectory(steps=1, completion=completion, model=model)
    traj["steps"][0]["extra"] = {"stop_reason": "refusal"}
    traj["final_metrics"]["total_completion_tokens"] = completion
    return traj


class TestPrePatchSelection(unittest.TestCase):
    """The selector must name a file that exists BEFORE the patch."""

    def test_rename_resolves_to_the_pre_patch_path(self):
        sections = source_targets.patch_sections(WP_CALYPSO_PATCH)
        self.assertEqual(len(sections), 2)
        a, b, section = sections[0]
        self.assertEqual(
            source_targets.pre_patch_path(a, b, section),
            "client/dashboard/sites/overview-card/card.stories.tsx")
        self.assertNotIn("components/overview-card",
                         source_targets.pre_patch_path(a, b, section))

    def test_created_file_is_never_a_candidate(self):
        """`new file mode` must be found wherever it sits in the section.

        The old check looked only at the first 500 bytes, so a deep path could
        push the marker out of the window and make a file the patch CREATES
        look like a modification.
        """
        self.assertGreater(CREATE_PATCH.index("new file mode"), 500)
        self.assertEqual(source_targets.candidates(CREATE_PATCH), [])
        with self.assertRaises(ValueError):
            source_targets.select(CREATE_PATCH)

    def test_plain_modification_is_unchanged(self):
        path, prefix = source_targets.select(MODIFY_PATCH)
        self.assertEqual(path, "src/lib/handler.py")
        self.assertEqual(prefix, "#")

    def test_wp_calypso_shape_picks_the_pre_rename_path(self):
        path, prefix = source_targets.select(WP_CALYPSO_PATCH)
        self.assertEqual(
            path, "client/dashboard/sites/overview-card/card.stories.tsx")
        self.assertEqual(prefix, "//")
        # all_patch_paths still reports the post-patch side, which is what the
        # registry's "paths this task touches" field means.
        self.assertIn("client/dashboard/components/overview-card/card.stories.tsx",
                      source_targets.all_patch_paths(WP_CALYPSO_PATCH))


class SeedFixture(Fixture):
    """A Fixture whose task dirs also carry a gold patch and a seed."""

    def seed_task(self, base_index: int, arm: str, *, channel="source",
                  patch=WP_CALYPSO_PATCH, target=None) -> Path:
        d = self.dataset / self.task_dir(base_index, arm)
        (d / "solution").mkdir(parents=True, exist_ok=True)
        (d / "solution" / "gold.patch").write_text(patch, encoding="utf-8")
        if target is None:
            target = source_targets.all_patch_paths(patch)[0]
        write_json(d / "environment" / "benchmark_seed" / "seed.json", {
            "condition": arm, "channel": channel, "source_target": target,
            "source_comment_prefix": "//", "workspace_root": "/workspace",
            "content": "cue text",
        })
        manifest_path = self.dataset / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for entry in manifest["tasks"]:
            if entry["directory"] == d.name:
                entry["channel"] = channel
                entry["source_target"] = target
        write_json(manifest_path, manifest)
        return d

    def manifest_target(self, base_index: int, arm: str) -> str:
        manifest = json.loads((self.dataset / "manifest.json").read_text())
        name = self.task_dir(base_index, arm)
        return next(e["source_target"] for e in manifest["tasks"]
                    if e["directory"] == name)

    def seed_target(self, base_index: int, arm: str) -> str:
        p = (self.dataset / self.task_dir(base_index, arm)
             / "environment/benchmark_seed/seed.json")
        return json.loads(p.read_text())["source_target"]


class TestPreparedTaskAudit(unittest.TestCase):

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        self.fx = SeedFixture(Path(self.td.name))
        relocate_campaign_root(self, self.fx.root)

    def test_rename_destination_target_is_flagged(self):
        d = self.fx.seed_task(0, "eval-src")
        finding = source_targets.audit_task_dir(d)
        self.assertIsNotNone(finding)
        self.assertIn("post-rename destination", finding["problem"])
        self.assertEqual(
            finding["corrected"],
            "client/dashboard/sites/overview-card/card.stories.tsx")

    def test_correct_target_and_non_source_arms_are_silent(self):
        good = self.fx.seed_task(
            1, "eval-src",
            target="client/dashboard/sites/overview-card/card.stories.tsx")
        self.assertIsNone(source_targets.audit_task_dir(good))
        # A clean arm never reads source_target, so a stale one is not a defect.
        clean = self.fx.seed_task(2, "clean-n", channel="none")
        self.assertIsNone(source_targets.audit_task_dir(clean))

    def test_shard_audit_reports_only_the_broken_cells(self):
        self.fx.seed_task(0, "eval-src")
        self.fx.seed_task(0, "eval-fin-src")
        self.fx.seed_task(
            1, "eval-src",
            target="client/dashboard/sites/overview-card/card.stories.tsx")
        r = source_targets.audit_shard(self.fx.mode, self.fx.shard,
                                       profiles=[self.fx.profile],
                                       paths=self.fx.paths)
        self.assertFalse(r["ok"])
        self.assertEqual(len(r["findings"]), 2)
        self.assertEqual({f["task_dir"] for f in r["findings"]},
                         {self.fx.task_dir(0, "eval-src"),
                          self.fx.task_dir(0, "eval-fin-src")})


class TestRepair(unittest.TestCase):

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        self.fx = SeedFixture(Path(self.td.name))
        relocate_campaign_root(self, self.fx.root)
        self.fx.seed_task(0, "eval-src")

    def test_repair_rewrites_seed_and_manifest_together(self):
        want = "client/dashboard/sites/overview-card/card.stories.tsx"
        self.assertEqual(self.fx.seed_target(0, "eval-src"),
                         "client/dashboard/components/overview-card/card.stories.tsx")

        dry = source_targets.repair_shard(
            self.fx.mode, self.fx.shard, profiles=[self.fx.profile],
            paths=self.fx.paths, apply=False)
        self.assertEqual(len(dry["changed"]), 1)
        self.assertNotEqual(self.fx.seed_target(0, "eval-src"), want,
                            "a dry run must not write anything")

        got = source_targets.repair_shard(
            self.fx.mode, self.fx.shard, profiles=[self.fx.profile],
            paths=self.fx.paths, apply=True)
        self.assertTrue(got["ok"])
        self.assertEqual(len(got["changed"]), 1)
        self.assertEqual(self.fx.seed_target(0, "eval-src"), want)
        # 04_validate cross-checks the two against each other; fixing only one
        # trades an image-build failure for a validation failure.
        self.assertEqual(self.fx.manifest_target(0, "eval-src"), want)
        self.assertTrue(source_targets.audit_shard(
            self.fx.mode, self.fx.shard, profiles=[self.fx.profile],
            paths=self.fx.paths)["ok"])

    def test_repair_refuses_a_cell_with_a_valid_trajectory(self):
        """Rewriting a task under a completed trajectory breaks its provenance."""
        key = cells.CellKey(lib.CAMPAIGN_ID, self.fx.mode, self.fx.profile,
                            self.fx.shard, self.fx.base_ids[0], "eval-src").key
        got = source_targets.repair_shard(
            self.fx.mode, self.fx.shard, frozen={key},
            profiles=[self.fx.profile], paths=self.fx.paths, apply=True)
        self.assertFalse(got["ok"])
        self.assertEqual(got["changed"], [])
        self.assertEqual(len(got["refused"]), 1)
        self.assertIn("valid completed trajectory", got["refused"][0]["refusal"])
        self.assertEqual(self.fx.seed_target(0, "eval-src"),
                         "client/dashboard/components/overview-card/card.stories.tsx")

    def test_manifest_sync_never_rewrites_a_past_attempt(self):
        """_batches is the record of what ran and must keep its own values.

        Every other derived manifest describing the same task dir has to move
        with the seed, because 04_validate asserts the two agree.
        """
        applied = source_targets.repair_shard(
            self.fx.mode, self.fx.shard, profiles=[self.fx.profile],
            paths=self.fx.paths, apply=True)
        self.assertEqual(len(applied["changed"]), 1)
        source_targets.write_repair_record(applied, paths=self.fx.paths)

        stale = "client/dashboard/components/overview-card/card.stories.tsx"
        want = "client/dashboard/sites/overview-card/card.stories.tsx"
        entry = {"directory": self.fx.task_dir(0, "eval-src"),
                 "source_target": stale}
        batch = (self.fx.paths["datasets"] / "_batches" / self.fx.mode
                 / self.fx.profile / self.fx.label / "batch-01")
        staged = (self.fx.paths["datasets"] / self.fx.mode / self.fx.profile)
        write_json(batch / "manifest.json", {"tasks": [dict(entry)]})
        write_json(staged / "manifest.json", {"tasks": [dict(entry)]})

        # generated/ lives outside the campaign results tree; keep the sweep
        # inside the fixture so the test can never touch the real repo.
        with mock.patch.object(lib, "PROJECT_ROOT", self.fx.root):
            r = source_targets.sync_manifests(paths=self.fx.paths, apply=True)

        self.assertTrue(r["ok"])
        self.assertEqual(len(r["updated"]), 1)
        self.assertNotIn("_batches", r["updated"][0]["manifest"])
        self.assertEqual(
            json.loads((staged / "manifest.json").read_text())["tasks"][0]["source_target"],
            want)
        self.assertEqual(
            json.loads((batch / "manifest.json").read_text())["tasks"][0]["source_target"],
            stale)


class TestProviderBlocked(unittest.TestCase):
    """An API-layer block is not an observation, and not a repair candidate."""

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        self.fx = Fixture(Path(self.td.name))
        relocate_campaign_root(self, self.fx.root)
        self.run = self.fx.run_dir()

    def _classify(self, base_index, arm, **kw):
        trial = self.fx.add_trial(self.run, base_index, arm, **kw)
        return cells.classify_observation(trial, campaign_root=self.fx.root)

    def test_synthetic_zero_token_refusal_is_provider_blocked(self):
        obs = self._classify(
            0, "eval-src", traj=refusal_trial(),
            result=result_json(reward=0.0,
                               task_path=str(self.fx.dataset / self.fx.task_dir(0, "eval-src")),
                               exception_type="AgentSafetyRefusalError"))
        self.assertEqual(obs.status, cells.PROVIDER_BLOCKED)
        self.assertIn("before the model generated anything", obs.reason)
        self.assertNotIn(cells.PROVIDER_BLOCKED, cells.NEEDS_REPAIR)

    def test_a_model_refusal_with_real_output_stays_valid(self):
        """The rule that matters: a model that reads the task and declines has
        produced a result. Re-running it until it complies would condition
        acceptance on the behaviour under study."""
        traj = refusal_trial(model="anthropic/claude-opus-4-8", completion=812)
        obs = self._classify(
            1, "eval-src", traj=traj,
            result=result_json(reward=0.0,
                               task_path=str(self.fx.dataset / self.fx.task_dir(1, "eval-src"))))
        self.assertEqual(obs.status, cells.COMPLETE_VALID)
        self.assertEqual(obs.reward, 0.0)

    def test_blocked_cells_are_excluded_from_the_repair_plan(self):
        self.fx.fill_all_valid(self.run)
        blocked = self.run / f"{self.fx.task_dir(0, 'eval-src')}__AAA1"
        write_json(blocked / "agent" / "trajectory.json", refusal_trial())
        write_json(blocked / "result.json", result_json(
            reward=0.0,
            task_path=str(self.fx.dataset / self.fx.task_dir(0, "eval-src")),
            exception_type="AgentSafetyRefusalError"))

        res = self.fx.audit()
        self.assertEqual(res["counts"][cells.PROVIDER_BLOCKED], 1)
        self.assertEqual(res["repair_required"], 0)
        rec = res["records"][cells.CellKey(
            lib.CAMPAIGN_ID, self.fx.mode, self.fx.profile, self.fx.shard,
            self.fx.base_ids[0], "eval-src").key]
        self.assertEqual(rec.repair_outlook, cells.REPAIR_EXCLUDED)

        plan = cells.repair_plan(res)
        planned = {c["cell_key"] for v in plan["by_profile"].values() for c in v}
        self.assertEqual(planned, set())

        v = cells.validate_shard_complete(res)
        n = len(self.fx.base_ids) * len(self.fx.arms)
        self.assertEqual(v["complete_valid"], n - 1)
        self.assertEqual(v["provider_blocked"], 1)
        self.assertEqual(v["accounted"], n)
        self.assertEqual(v["outstanding_total"], 0)
        # Accounted for, but NOT a complete corpus of model observations.
        self.assertTrue(v["ok"])
        self.assertFalse(v["full_corpus"])


class TestRemediationReopensFutileCells(unittest.TestCase):

    def test_corrected_task_definition_supersedes_old_failures(self):
        with tempfile.TemporaryDirectory() as td:
            fx = Fixture(Path(td))
            relocate_campaign_root(self, fx.root)
            run = fx.run_dir()
            fx.fill_all_valid(run)

            # Two cells that both died the same deterministic way.
            for arm in ("eval-src", "eval-fin-src"):
                trial = run / f"{fx.task_dir(0, arm)}__AAA1"
                write_json(trial / "agent" / "trajectory.json", {})
                write_json(trial / "result.json", result_json(
                    reward=None, agent_finished=False, verifier_finished=False,
                    task_path=str(fx.dataset / fx.task_dir(0, arm)),
                    exception_type="ImageBuildError"))
                # Same content-addressed image in both cells: that is what
                # makes the failure "reproduced" rather than unlucky.
                (trial / "exception.txt").write_text(
                    "modal.exception.ImageBuildError: image build failed for "
                    "im-x6EXQtv6VljLdOL1LimybV\n"
                    "SystemExit: source target missing: /workspace/client/"
                    "dashboard/components/overview-card/card.stories.tsx\n",
                    encoding="utf-8")

            before = cells.audit(fx.mode, fx.shard, profiles=[fx.profile],
                                 paths=fx.paths)
            outlooks = {r.repair_outlook for r in before["records"].values()
                        if r.needs_repair}
            self.assertEqual(outlooks, {cells.REPAIR_FUTILE},
                             "a reproduced deterministic failure is futile to retry")

            # Now the task definitions are corrected, AFTER those observations.
            log = fx.paths["provenance"] / "source_target_repairs.jsonl"
            log.write_text("".join(
                json.dumps({"at": "2026-09-03T00:00:00+00:00",
                            "task_dir": fx.task_dir(0, arm),
                            "applied": True}) + "\n"
                for arm in ("eval-src", "eval-fin-src")), encoding="utf-8")

            after = cells.audit(fx.mode, fx.shard, profiles=[fx.profile],
                                paths=fx.paths)
            for rec in after["records"].values():
                if rec.needs_repair:
                    self.assertEqual(rec.repair_outlook, cells.REPAIR_EXPECTED)
                    self.assertTrue(rec.failure_class.endswith("/remediated"))
            self.assertEqual(after["repair_required"], 2)


class TestValidatorGate(unittest.TestCase):
    """scripts/04_validate.py must REFUSE a target that only exists post-patch.

    The audit in this module is what the operator runs; the validator gate is
    what runs whether or not anyone remembers to. Both have to hold, so this
    exercises the real script end to end on a hardlinked mirror of the real
    corpus -- one seed is rewritten through a temp file so the mirror's own
    inode is replaced and the campaign's file is never touched.
    """

    MODE, PROFILE = "full", "claude"

    def setUp(self):
        self.src = lib.PROJECT_ROOT / "generated" / self.MODE / self.PROFILE
        if not (self.src / "manifest.json").is_file():
            self.skipTest("generated corpus is not present in this tree")
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        self.root = Path(self.td.name)
        dst = self.root / "generated" / self.MODE / self.PROFILE
        dst.parent.mkdir(parents=True)
        # Hardlinks: 156MB of corpus, no copy, and mutation is impossible
        # except through an explicit replace (see below).
        shutil.copytree(self.src, dst, copy_function=os.link)
        self.dst = dst
        for name in ("manifests", "factor_data", "vendor", "scripts"):
            (self.root / name).symlink_to(lib.PROJECT_ROOT / name)

    def _validate(self):
        return subprocess.run(
            [sys.executable, str(lib.PROJECT_ROOT / "scripts/04_validate.py"),
             "--project-root", str(self.root), "--mode", self.MODE,
             "--profile", self.PROFILE],
            capture_output=True, text=True)

    def _a_source_task(self) -> Path:
        for d in sorted(self.dst.glob("ea-*-eval-src")):
            seed = d / "environment/benchmark_seed/seed.json"
            if seed.is_file() and json.loads(seed.read_text()).get("channel") == "source":
                return d
        self.skipTest("no source-channel task in the corpus")

    @staticmethod
    def _replace_json(path: Path, blob) -> None:
        """Write without following the hardlink into the real corpus."""
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)

    def test_clean_corpus_passes(self):
        r = self._validate()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_post_patch_only_target_is_rejected(self):
        task = self._a_source_task()
        seed_path = task / "environment/benchmark_seed/seed.json"
        seed = json.loads(seed_path.read_text())
        before = seed["source_target"]
        seed["source_target"] = "does/not/exist/until/after/the/patch.py"
        self._replace_json(seed_path, seed)

        manifest_path = self.dst / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for entry in manifest["tasks"]:
            if entry["directory"] == task.name:
                entry["source_target"] = seed["source_target"]
        self._replace_json(manifest_path, manifest)

        r = self._validate()
        self.assertNotEqual(r.returncode, 0,
                            "validator accepted a target that cannot exist "
                            "when the image is built")
        self.assertIn("does not exist", r.stdout + r.stderr)
        # The real corpus is untouched: the mirror got its own inode.
        self.assertEqual(
            json.loads((self.src / task.name
                        / "environment/benchmark_seed/seed.json").read_text()
                       )["source_target"], before)


class TestRealCampaignDatasets(unittest.TestCase):
    """Runs against the real prepared datasets, read-only. No model, no build."""

    def test_every_prepared_source_seed_targets_a_pre_patch_path(self):
        paths = lib.campaign_paths()
        if not (paths["datasets"] / "_shards").is_dir():
            self.skipTest("campaign datasets are not prepared in this tree")
        r = source_targets.audit_all(paths=paths)
        self.assertTrue(
            r["ok"],
            "source-channel seeds that do not exist pre-patch will fail the "
            f"image build: {[f['task_dir'] for f in r['findings']][:5]}")
        self.assertGreater(r["checked"], 0)


if __name__ == "__main__":
    unittest.main()
