#!/usr/bin/env python3
"""Offline tests for how the corpus resolves TWO observations of ONE cell.

NO Harbor job, NO model call, NO network. Every fixture is a temporary
directory holding synthetic JSON.

THE QUESTION THESE TESTS SETTLE
-------------------------------
After a repair, one experimental cell can have two trajectories on disk: the
failed original and the repair that replaced it. The Aug 2026 study answered
that by deduping on `task_name`, so whichever directory the loop happened to
see last silently won. The corpus must instead answer it from explicit
provenance lineage -- never task_name order, timestamp, newest run, or reward.

The policy, one cell at a time:

  1 failed original (PRE_MODEL_FAILURE / PARTIAL_MODEL_FAILURE) + an accepted
    repair linked through the repair ledger -> the corpus holds ONLY the repair
    row, and the failed original stays on disk and in the resolution ledger.
  2 two COMPLETE_VALID observations -> HARD ERROR unless a human recorded which
    one is authoritative. Never silently choose one.
  3 PROVIDER_BLOCKED is an accepted stack outcome and is never replaced by a
    repair.
  4 a failed original with no valid accepted repair leaves the cell incomplete,
    and the build fails closed rather than writing a short corpus.
  5 a repair observation is eligible only if its cell was in the repair plan,
    its attempt completed, model/runtime/task-definition provenance match, and
    the cell-level validator accepted it.

The fixture is the real shape FULL shard 1 ended in on 2026-09-02: 4 profiles x
300 cells, 976 valid originals, 14 failed originals, 10 provider blocks and 200
cells that never ran. One repair attempt per profile covers those 214 unclosed
cells (claude 103, fable 3, codex 5, llama 103), and exactly 14 of them
supersede an original that actually produced a failed observation.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from campaign import cells, lib  # noqa: E402
from campaign.tests.test_tooling import Fixture, _Args, _quiet  # noqa: E402

MODE = "full"
SHARD = 1

# 2026-09-02 FULL shard 1, per profile. valid + failed + blocked + missing = 300.
SHAPE = {
    "claude": {"valid": 197, "failed": 3, "blocked": 0},
    "fable":  {"valid": 287, "failed": 3, "blocked": 10},
    "codex":  {"valid": 295, "failed": 5, "blocked": 0},
    "llama":  {"valid": 197, "failed": 3, "blocked": 0},
}
TOTAL_CELLS = 1200
VALID_ORIGINALS = 976
FAILED_ORIGINALS = 14          # failed-original + successful-repair pairs
PROVIDER_BLOCKED = 10
NEVER_RAN = 200
REPAIRED = FAILED_ORIGINALS + NEVER_RAN     # 214

REFUSAL_LOG = ('{"event": "model_refusal_no_fallback", '
               '"api_refusal_category": "cyber"}\n')


def _trial(parent: Path, base_id: str, arm: str, profile: str, *,
           kind="valid", tag="x", reward=1.0) -> Path:
    """One trial directory, shaped so the CELL-level validator classifies it.

    kind: valid | pre_model | partial | blocked
    """
    td = parent / f"ea-{base_id}-{arm}__{tag}"
    (td / "agent").mkdir(parents=True, exist_ok=True)
    completion = 0 if kind == "blocked" else 5
    model = "<synthetic>" if kind == "blocked" else Fixture.MODEL_WIRE[profile]
    # PRE_MODEL_FAILURE is the infra death: the trial directory exists and
    # nothing was ever written into it.
    if kind != "pre_model":
        (td / "agent" / "trajectory.json").write_text(json.dumps({
            "agent": {"name": lib.VERSION_PINS[profile]["agent"],
                      "version": lib.VERSION_PINS[profile]["version"],
                      "model_name": model},
            "steps": [],
            "final_metrics": {"total_prompt_tokens": 10,
                              "total_completion_tokens": completion,
                              "total_cached_tokens": 0,
                              "total_cost_usd": 0.0 if kind == "blocked" else 1.25,
                              "total_steps": 3},
        }))
    (td / "result.json").write_text(json.dumps({
        "task_name": f"ea-{base_id}-{arm}",
        "started_at": "2026-09-02T06:00:00Z",
        "finished_at": "2026-09-02T06:30:00Z",
        # A PARTIAL trial is one the agent never finished.
        "agent_execution": {} if kind == "partial" else {"finished_at": "2026-09-02T06:25:00Z"},
        "verifier": {"finished_at": "2026-09-02T06:30:00Z"},
        "verifier_result": {"rewards": {"reward": reward, "overall_pass": reward}},
    }))
    if kind == "blocked":
        # The safety layer answered instead of the model; the vendor's own
        # category is read verbatim from its log.
        (td / "agent" / "run.txt").write_text(REFUSAL_LOG)
    return td


class ShardOne:
    """FULL shard 1 as it actually ended, plus the repair that finished it."""

    def __init__(self, tmp: Path):
        self.fx = Fixture(tmp)
        for profile in lib.PROFILES:
            self.fx.write_cell_manifest(MODE, profile, SHARD)
        self.origin: dict[str, Path] = {}
        self.repair: dict[str, Path] = {}
        self.failed: dict[str, list] = {}
        self.blocked: dict[str, list] = {}
        self.never_ran: dict[str, list] = {}
        self.plan_path = cells.repair_plan_path(MODE, SHARD)

    # -- layout ------------------------------------------------------------- #
    def order(self):
        return [(b, arm) for b in self.fx.shard_bases(SHARD) for arm in lib.ARMS[MODE]]

    def key(self, profile, base_id, arm):
        return cells.CellKey(lib.CAMPAIGN_ID, MODE, profile, SHARD, base_id, arm).key

    def run_parent(self, profile: str, tag: str) -> Path:
        label = lib.Cell(MODE, profile, SHARD).shard_label
        rd = self.fx.root / MODE / f"swe-{MODE}-{profile}-{label}-{tag}"
        (rd / rd.name).mkdir(parents=True, exist_ok=True)
        (rd / "run_metadata.json").write_text(json.dumps({
            "model": lib.MODEL_PINS[profile],
            "agent": lib.VERSION_PINS[profile]["agent"],
            "agent_version_requested": lib.VERSION_PINS[profile]["version"],
            "created_at": lib.now_iso(),
        }))
        return rd

    def build_original_runs(self):
        """The interrupted run: what it finished, what it broke on, what it never reached."""
        for profile in lib.PROFILES:
            shape = SHAPE[profile]
            rd = self.run_parent(profile, "b01")
            parent = rd / rd.name
            order, i = self.order(), 0
            for _ in range(shape["valid"]):
                b, arm = order[i]
                _trial(parent, b, arm, profile, kind="valid", tag="orig")
                i += 1
            self.failed[profile] = []
            for n in range(shape["failed"]):
                b, arm = order[i]
                # Both failure classes the policy names, not just one.
                _trial(parent, b, arm, profile,
                       kind="pre_model" if n % 2 == 0 else "partial", tag="orig")
                self.failed[profile].append((b, arm))
                i += 1
            self.blocked[profile] = []
            for _ in range(shape["blocked"]):
                b, arm = order[i]
                _trial(parent, b, arm, profile, kind="blocked", tag="orig")
                self.blocked[profile].append((b, arm))
                i += 1
            self.never_ran[profile] = order[i:]
            self.origin[profile] = rd

    def write_repair_plan(self, *, drop_cells=()):
        plan = cells.repair_plan(cells.audit(MODE, SHARD))
        drop = set(drop_cells)
        for profile, entries in plan["by_profile"].items():
            plan["by_profile"][profile] = [c for c in entries if c["cell_key"] not in drop]
        self.plan_path.write_text(json.dumps(plan, indent=1))
        return plan

    def build_repair_runs(self, plan, *, skip_cells=(), extra=()):
        """One repair run per profile, producing exactly the planned cells."""
        skip = set(skip_cells)
        for profile in lib.PROFILES:
            rd = self.run_parent(profile, "repair-b01")
            parent = rd / rd.name
            for entry in plan["by_profile"].get(profile, []):
                if entry["cell_key"] in skip:
                    continue
                _trial(parent, entry["base_task_id"], entry["arm"], profile,
                       kind="valid", tag="rep")
            for prof, base_id, arm in extra:
                if prof == profile:
                    _trial(parent, base_id, arm, profile, kind="valid", tag="rep")
            self.repair[profile] = rd

    # -- ledger ------------------------------------------------------------- #
    def record_originals(self, provenance):
        out = {}
        for profile in lib.PROFILES:
            out[profile] = self.fx.record(provenance, MODE, profile, SHARD,
                                          self.origin[profile])
        return out

    def record_repairs(self, provenance, *, failed_profiles=()):
        out = {}
        for profile in lib.PROFILES:
            status = "failed" if profile in failed_profiles else "auto"
            ns = _Args(mode=MODE, profile=profile, shard=SHARD,
                       run_dir=str(self.repair[profile]), status=status,
                       started_at=None, finished_at=None, note="",
                       repair_plan=str(self.plan_path))
            out[profile] = _quiet(provenance.cmd_record, ns)
        return out

    def approve(self, cell_key, winner_trial_dir, *, who="ops", reason="reviewed"):
        p = lib.campaign_paths()["provenance"] / "observation_supersessions.jsonl"
        with p.open("a") as fh:
            fh.write(json.dumps({"cell_key": cell_key,
                                 "winner_trial_dir": winner_trial_dir,
                                 "approved_by": who, "reason": reason}) + "\n")

    # -- results ------------------------------------------------------------ #
    def report(self):
        return json.loads(
            (lib.campaign_paths()["provenance"] / "build_report.json").read_text())

    def corpus(self):
        p = lib.campaign_paths()["provenance"] / "corpus.jsonl"
        if not p.is_file():
            return []
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

    def resolutions(self):
        p = lib.campaign_paths()["provenance"] / "corpus_resolution.jsonl"
        if not p.is_file():
            return []
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


class ShardOneCase(unittest.TestCase):
    """Base: the repaired shard, built once per test."""

    SKIP_CELLS = ()
    DROP_FROM_PLAN = ()
    EXTRA_REPAIR_TRIALS = ()
    FAILED_REPAIR_PROFILES = ()

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="campv2-resolution-")
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        self.s1 = ShardOne(Path(self.tmpdir))
        self.addCleanup(self.s1.fx.restore)
        from campaign import provenance
        self.provenance = provenance
        self.s1.build_original_runs()
        self.s1.record_originals(provenance)
        self.plan = self.s1.write_repair_plan(drop_cells=self.DROP_FROM_PLAN)
        self.s1.build_repair_runs(self.plan, skip_cells=self.SKIP_CELLS,
                                  extra=self.EXTRA_REPAIR_TRIALS)
        self.s1.record_repairs(
            provenance, failed_profiles=self.FAILED_REPAIR_PROFILES)

    def build(self):
        return _quiet(self.provenance.cmd_build, _Args())


# --------------------------------------------------------------------------- #
# the fixture itself is the claim, so assert its shape first
# --------------------------------------------------------------------------- #
class TestFixtureShape(ShardOneCase):

    def test_00_fixture_is_the_real_shard_one_shape(self):
        res = cells.audit(MODE, SHARD)
        self.assertEqual(res["expected"], TOTAL_CELLS)
        self.assertEqual(sum(len(v) for v in self.s1.failed.values()), FAILED_ORIGINALS)
        self.assertEqual(sum(len(v) for v in self.s1.blocked.values()), PROVIDER_BLOCKED)
        self.assertEqual(sum(len(v) for v in self.s1.never_ran.values()), NEVER_RAN)
        planned = sum(len(v) for v in self.plan["by_profile"].values())
        self.assertEqual(planned, REPAIRED,
                         "the repair plan covers the failed and the never-run cells")

    def test_00b_originals_are_failed_attempts_repairs_are_complete(self):
        """The repair stands in for a run that never closed. That is the premise."""
        att = [json.loads(l) for l in
               (lib.campaign_paths()["provenance"] / "attempts.jsonl").read_text().splitlines()
               if l.strip()]
        originals = [a for a in att if a["attempt_kind"] == "full_cell"]
        repairs = [a for a in att if a["attempt_kind"] == "repair"]
        self.assertEqual(len(originals), 4)
        self.assertEqual(len(repairs), 4)
        self.assertTrue(all(a["status"] == "failed" for a in originals))
        self.assertTrue(all(a["status"] == "complete" for a in repairs))
        # A repair never promotes itself into the whole-shard accepted set.
        acc = json.loads(
            (lib.campaign_paths()["provenance"] / "accepted_runs.json").read_text())
        self.assertEqual(acc["accepted"], [])
        self.assertEqual(len(acc["accepted_repair_attempts"]), 4)


# --------------------------------------------------------------------------- #
# 1: the repaired corpus
# --------------------------------------------------------------------------- #
class TestRepairedCorpus(ShardOneCase):

    def setUp(self):
        super().setUp()
        self.rc = self.build()
        self.rep = self.s1.report()
        self.rows = self.s1.corpus()

    def test_01_corpus_is_exactly_1200_rows(self):
        self.assertEqual(self.rc, 0, f"build should succeed; errors={self.rep['errors'][:3]}")
        self.assertEqual(self.rep["rows"], TOTAL_CELLS)
        self.assertEqual(self.rep["unique_cells"], TOTAL_CELLS)
        self.assertEqual(self.rep["expected_cells"], TOTAL_CELLS)
        self.assertEqual(self.rep["duplicates"], 0)
        self.assertEqual(self.rep["missing"], 0)
        self.assertEqual(self.rep["repair_resolved_cells"], FAILED_ORIGINALS)
        self.assertEqual(self.rep["repair_sourced_rows"], REPAIRED)
        self.assertEqual(self.rep["original_rows"], VALID_ORIGINALS)
        self.assertEqual(self.rep["provider_blocked_rows"], PROVIDER_BLOCKED)
        self.assertEqual(len(self.rows), TOTAL_CELLS)

    def test_02_exactly_one_row_per_experimental_cell(self):
        keys = [r["cell_key"] for r in self.rows]
        self.assertEqual(len(keys), len(set(keys)))
        expected = {ck.key for ck in cells.expected_cells(MODE, SHARD)}
        self.assertEqual(set(keys), expected)

    def test_03_the_repaired_cells_carry_the_repair_row(self):
        by_key = {r["cell_key"]: r for r in self.rows}
        for profile, pairs in self.s1.failed.items():
            for base_id, arm in pairs:
                row = by_key[self.s1.key(profile, base_id, arm)]
                self.assertEqual(row["resolution"], "repair")
                self.assertEqual(row["source_run_dir"], self.s1.repair[profile].name)
                self.assertTrue(row["trial_dir"].endswith("__rep"))

    def test_04_failed_originals_are_preserved_not_deleted(self):
        res = {r["cell_key"]: r for r in self.s1.resolutions()}
        # Every repair-resolved cell is logged, so the ledger explains all 214
        # rows that did not come from a plain original. Only the 14 whose
        # original actually produced a failed observation supersede anything --
        # the other 200 cells never ran at all, so there is nothing to preserve.
        self.assertEqual(len(res), REPAIRED)
        superseding = [r for r in res.values() if r["superseded_observations"]]
        self.assertEqual(len(superseding), FAILED_ORIGINALS)
        for profile, pairs in self.s1.failed.items():
            for base_id, arm in pairs:
                entry = res[self.s1.key(profile, base_id, arm)]
                self.assertEqual(len(entry["superseded_observations"]), 1)
                sup = entry["superseded_observations"][0]
                self.assertIn(sup["status"], ("PRE_MODEL_FAILURE", "PARTIAL_MODEL_FAILURE"))
                self.assertEqual(sup["run_dir"], self.s1.origin[profile].name)
                self.assertTrue(Path(sup["preserved_at"]).is_dir(),
                                "the failed original is still on disk; the path is the proof")
                self.assertEqual(entry["winner"]["attempt_kind"], "repair")

    def test_05_provider_blocked_rows_stay_explicit(self):
        blocked = [r for r in self.rows if r["status"] == lib.STATUS_PROVIDER_BLOCKED]
        self.assertEqual(len(blocked), PROVIDER_BLOCKED)
        for r in blocked:
            self.assertEqual(r["resolution"], "provider_blocked")
            self.assertFalse(r["model_started"], "a blocked row carries no model behaviour")
            self.assertTrue(r["provider_refusal"])
            self.assertEqual(r["provider_refusal_category"], "cyber")
        planned = {c["cell_key"] for v in self.plan["by_profile"].values() for c in v}
        for profile, pairs in self.s1.blocked.items():
            for base_id, arm in pairs:
                self.assertNotIn(self.s1.key(profile, base_id, arm), planned,
                                 "a blocked cell is an outcome, never a repair target")

    def test_06_no_dedupe_by_task_name(self):
        """The Aug 2026 bug: 1200 cells collapsing to 300 task names.

        Every base task/arm pair occurs once per profile, and a repaired cell's
        repair trial shares its task name with the failed original it replaced.
        Deduping on that name is what silently promoted a failed trial before.
        """
        task_names = {r["trial_dir"].split("__", 1)[0] for r in self.rows}
        self.assertEqual(len(task_names), TOTAL_CELLS // len(lib.PROFILES))
        self.assertEqual(len(self.rows), TOTAL_CELLS)
        for profile, pairs in self.s1.failed.items():
            for base_id, arm in pairs:
                origin = self.s1.origin[profile] / self.s1.origin[profile].name
                self.assertTrue((origin / f"ea-{base_id}-{arm}__orig").is_dir(),
                                "the identically named failed original is still there")


# --------------------------------------------------------------------------- #
# 2: two COMPLETE_VALID observations
# --------------------------------------------------------------------------- #
class TestTwoValidObservations(ShardOneCase):

    def _second_valid(self, *, reward=1.0, tag="second"):
        """A second, better-rewarded, later-written valid trial for one cell."""
        profile = "codex"
        base_id, arm = self.s1.order()[0]
        parent = self.s1.origin[profile] / self.s1.origin[profile].name
        _trial(parent, base_id, arm, profile, kind="valid", tag=tag, reward=reward)
        return profile, base_id, arm

    def test_07_two_unexplained_valid_observations_fail_closed(self):
        profile, base_id, arm = self._second_valid(reward=1.0)
        rc = self.build()
        rep = self.s1.report()
        self.assertEqual(rc, 1, "two valid observations must be a hard error")
        self.assertEqual(self.s1.corpus(), [], "no corpus at all, not a deduped one")
        key = self.s1.key(profile, base_id, arm)
        hit = [e for e in rep["errors"] if e.startswith(key)]
        self.assertTrue(hit, f"the conflicted cell must be named: {rep['errors'][:3]}")
        self.assertIn("NOT deduping", hit[0])
        self.assertIn("observation_supersessions.jsonl", hit[0])
        self.assertEqual(rep["unresolved_cells"], 1)

    def test_08_neither_reward_nor_write_order_may_decide(self):
        """The original scored 1.0 and the newcomer 0.0 -- still a hard error.

        Reward, timestamp and directory order are all available here and none of
        them is allowed to resolve the conflict.
        """
        self._second_valid(reward=0.0, tag="zzz-latest")
        self.assertEqual(self.build(), 1)
        self.assertEqual(self.s1.corpus(), [])

    def test_09_an_approved_supersession_resolves_it(self):
        profile, base_id, arm = self._second_valid()
        key = self.s1.key(profile, base_id, arm)
        self.s1.approve(key, f"ea-{base_id}-{arm}__second", who="lead@example")
        rc = self.build()
        rep = self.s1.report()
        self.assertEqual(rc, 0, f"errors={rep['errors'][:3]}")
        self.assertEqual(rep["rows"], TOTAL_CELLS)
        self.assertEqual(rep["human_approved_supersessions"], 1)
        row = next(r for r in self.s1.corpus() if r["cell_key"] == key)
        self.assertEqual(row["resolution"], "human_approved_supersession")
        self.assertEqual(row["trial_dir"], f"ea-{base_id}-{arm}__second")
        entry = next(e for e in self.s1.resolutions() if e["cell_key"] == key)
        self.assertEqual(entry["approved_by"], "lead@example")
        self.assertEqual(len(entry["superseded_observations"]), 1)


# --------------------------------------------------------------------------- #
# 3: a block is an outcome
# --------------------------------------------------------------------------- #
class TestProviderBlockedIsNeverReplaced(ShardOneCase):

    @property
    def EXTRA_REPAIR_TRIALS(self):
        return ()

    def test_10_a_repair_may_not_overwrite_a_block(self):
        """Re-running a refusal until it complies is outcome-conditioned acceptance."""
        profile = "fable"
        base_id, arm = self.s1.blocked[profile][0]
        parent = self.s1.repair[profile] / self.s1.repair[profile].name
        _trial(parent, base_id, arm, profile, kind="valid", tag="rep")
        rc = self.build()
        rep = self.s1.report()
        self.assertEqual(rc, 1)
        key = self.s1.key(profile, base_id, arm)
        hit = [e for e in rep["errors"] if e.startswith(key)]
        self.assertTrue(hit, f"errors={rep['errors'][:3]}")
        self.assertIn("accepted stack outcome", hit[0])
        self.assertEqual(self.s1.corpus(), [])


# --------------------------------------------------------------------------- #
# 4: fail closed on an unrepaired cell
# --------------------------------------------------------------------------- #
class TestUnrepairedCellFailsClosed(ShardOneCase):

    def test_11_failed_original_without_a_repair_leaves_the_cell_incomplete(self):
        profile = "claude"
        base_id, arm = self.s1.failed[profile][0]
        key = self.s1.key(profile, base_id, arm)
        shutil.rmtree(self.s1.repair[profile] / self.s1.repair[profile].name
                      / f"ea-{base_id}-{arm}__rep")
        rc = self.build()
        rep = self.s1.report()
        self.assertEqual(rc, 1)
        self.assertEqual(self.s1.corpus(), [], "a short corpus is never written")
        hit = [e for e in rep["errors"] if e.startswith(key)]
        self.assertTrue(hit, f"errors={rep['errors'][:3]}")
        self.assertIn("cell incomplete", hit[0])
        self.assertIn("refusing to write a short corpus", hit[0])
        self.assertEqual(rep["missing"], 1)


# --------------------------------------------------------------------------- #
# 5: what makes a repair observation eligible
# --------------------------------------------------------------------------- #
class TestRepairEligibility(ShardOneCase):

    def test_12_a_repair_for_a_cell_outside_the_plan_is_not_eligible(self):
        """Provenance runs through the PLAN, not through the directory listing."""
        profile = "codex"
        base_id, arm = self.s1.failed[profile][0]
        key = self.s1.key(profile, base_id, arm)
        # Rewrite the plan without this cell, then re-record the repair against
        # it. The trajectory on disk does not change; only its lineage does.
        self.s1.write_repair_plan(drop_cells=[key])
        (lib.campaign_paths()["provenance"] / "attempts.jsonl").write_text("".join(
            json.dumps(a) + "\n" for a in
            [json.loads(l) for l in
             (lib.campaign_paths()["provenance"] / "attempts.jsonl").read_text().splitlines()
             if l.strip()]
            if a["attempt_kind"] != "repair" or a["profile"] != profile))
        ns = _Args(mode=MODE, profile=profile, shard=SHARD,
                   run_dir=str(self.s1.repair[profile]), status="auto",
                   started_at=None, finished_at=None, note="",
                   repair_plan=str(self.s1.plan_path))
        _quiet(self.provenance.cmd_record, ns)
        rc = self.build()
        rep = self.s1.report()
        self.assertEqual(rc, 1)
        hit = [e for e in rep["errors"] if e.startswith(key)]
        self.assertTrue(hit, f"errors={rep['errors'][:3]}")
        self.assertIn("was not in the plan", hit[0])
        self.assertEqual(self.s1.corpus(), [])


class TestFailedRepairAttempt(ShardOneCase):
    """One profile's repair attempt died; the other three closed normally."""

    FAILED_REPAIR_PROFILES = ("llama",)

    def test_13_a_repair_attempt_that_failed_is_not_eligible(self):
        """And it is NOT quietly reclassified as an ordinary observation either.

        The shard is still in scope -- three profiles produced live attempts --
        so llama is audited rather than dropped, and every cell that depended on
        its dead repair goes unresolved by name.
        """
        rc = self.build()
        rep = self.s1.report()
        self.assertEqual(rc, 1)
        self.assertEqual(self.s1.corpus(), [],
                         "one dead repair attempt fails the whole build closed")
        prefix = f"{lib.CAMPAIGN_ID}/{MODE}/llama/s{SHARD}/"
        mine = [e for e in rep["errors"] if e.startswith(prefix)]
        self.assertEqual(len(mine), len(self.plan["by_profile"]["llama"]),
                         "every llama cell the dead attempt was meant to repair "
                         f"is named: {rep['errors'][:3]}")
        self.assertTrue(all("is failed" in e for e in mine), mine[:3])
        # The three healthy profiles are untouched: their failure is llama's, not
        # a reclassification of their own repairs into originals.
        for other in ("claude", "fable", "codex"):
            self.assertFalse(
                [e for e in rep["errors"]
                 if e.startswith(f"{lib.CAMPAIGN_ID}/{MODE}/{other}/s{SHARD}/")],
                f"{other} should have resolved cleanly")


class TestModelPinMismatch(ShardOneCase):

    def test_14_a_repair_run_on_the_wrong_model_is_not_eligible(self):
        profile = "llama"
        base_id, arm = self.s1.failed[profile][0]
        key = self.s1.key(profile, base_id, arm)
        td = (self.s1.repair[profile] / self.s1.repair[profile].name
              / f"ea-{base_id}-{arm}__rep" / "agent" / "trajectory.json")
        traj = json.loads(td.read_text())
        traj["agent"]["model_name"] = Fixture.MODEL_WIRE["codex"]
        td.write_text(json.dumps(traj))
        rc = self.build()
        rep = self.s1.report()
        self.assertEqual(rc, 1)
        hit = [e for e in rep["errors"] if e.startswith(key)]
        self.assertTrue(hit, f"errors={rep['errors'][:3]}")
        self.assertIn("does not match the pin", hit[0])
        self.assertEqual(self.s1.corpus(), [])


if __name__ == "__main__":
    unittest.main()
