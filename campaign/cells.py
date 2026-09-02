#!/usr/bin/env python3
"""Cell-level completion tracking, audit and repair planning for Campaign V2.

THE UNIT OF WORK IS ONE EXPERIMENTAL CELL
-----------------------------------------
    (campaign_id, mode, profile, shard, base_task_id, arm)

not a batch, not a base task, and not a profile/shard. The 2026-09-02 FULL
shard-1 controller crash motivated this module: 976 trajectories were already
on disk and valid, but the only recovery primitive available was "re-run the
whole 300-trial profile/shard", which would have discarded ~$2k of good data and
bought nothing. Repair operates per cell so a valid observation is never
re-purchased.

VALIDITY IS INDEPENDENT OF OUTCOME
----------------------------------
`COMPLETE_VALID` means the experiment RAN: a real model produced output and the
verifier reached a verdict. Whether that verdict is pass or fail is the result
being studied. A trajectory whose SWE reward is 0.0 is a perfectly valid
observation -- treating it as damaged and re-running it would silently select
for successful outcomes and bias the whole corpus. `classify_observation` never
reads the reward value, only that a verdict exists.

CLASSES
-------
  COMPLETE_VALID          real model output + verifier verdict, no exception
  PRE_MODEL_FAILURE       died before the model produced anything (infra, setup)
  PARTIAL_MODEL_FAILURE   model produced output, then the trial did not finish
  MISSING                 expected cell has no observation at all
  DUPLICATE               >1 COMPLETE_VALID observation -- fails closed
  OTHER_INVALID           synthetic, budget-censored, or unreadable

FAIL CLOSED ON DUPLICATES
-------------------------
Two valid complete trajectories for one cell is a provenance question a human
must answer (which attempt is authoritative?), not something to resolve by
picking the newest, the best-scoring, or the first seen. `audit` reports it and
`repair_plan` refuses to emit a plan while any duplicate is unresolved.

NO HISTORICAL CONTAMINATION
---------------------------
Every observation must live under the campaign root AND name a task path inside
the campaign's own dataset namespace. An Aug-2026 trajectory sitting in an
older results directory can never be counted, even if its task name matches.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path

from campaign import lib, failures

COMPLETE_VALID = "COMPLETE_VALID"
PRE_MODEL_FAILURE = "PRE_MODEL_FAILURE"
PARTIAL_MODEL_FAILURE = "PARTIAL_MODEL_FAILURE"
MISSING = "MISSING"
DUPLICATE = "DUPLICATE"
OTHER_INVALID = "OTHER_INVALID"

CLASSES = (COMPLETE_VALID, PRE_MODEL_FAILURE, PARTIAL_MODEL_FAILURE,
           MISSING, DUPLICATE, OTHER_INVALID)
# Everything that is not COMPLETE_VALID needs the cell to be re-run.
NEEDS_REPAIR = (PRE_MODEL_FAILURE, PARTIAL_MODEL_FAILURE, MISSING, OTHER_INVALID)


@dataclass(frozen=True)
class CellKey:
    campaign_id: str
    mode: str
    profile: str
    shard: int
    base_task_id: str
    arm: str

    @property
    def key(self) -> str:
        return (f"{self.campaign_id}/{self.mode}/{self.profile}/s{self.shard}/"
                f"{self.base_task_id}/{self.arm}")

    @property
    def task_dir(self) -> str:
        """Dataset directory name for this cell, e.g. 'ea-10d4b434-eval-src'."""
        return f"ea-{self.base_task_id[-8:]}-{self.arm}"

    def as_dict(self):
        return asdict(self)


@dataclass
class Observation:
    """One trial directory found on disk, and what it is worth."""
    trial_dir: str
    run_dir: str
    status: str
    reason: str = ""
    model_name: str = ""
    reward: float | None = None
    completion_tokens: int = 0
    steps: int = 0
    cost_usd: float | None = None
    agent_exit_exception: str = ""
    started_at: str = ""
    finished_at: str = ""
    path: str = ""

    def as_dict(self):
        return asdict(self)


@dataclass
class CellRecord:
    cell: CellKey
    status: str
    observations: list[Observation] = field(default_factory=list)
    reason: str = ""
    # Set for failing cells by `annotate_failures`.
    failure_class: str = ""
    repair_outlook: str = ""

    @property
    def needs_repair(self) -> bool:
        return self.status in NEEDS_REPAIR

    def as_dict(self):
        d = self.cell.as_dict()
        d.update({
            "cell_key": self.cell.key,
            "task_dir": self.cell.task_dir,
            "status": self.status,
            "reason": self.reason,
            "needs_repair": self.needs_repair,
            "failure_class": self.failure_class,
            "repair_outlook": self.repair_outlook,
            "observation_count": len(self.observations),
            "observations": [o.as_dict() for o in self.observations],
        })
        return d


# --------------------------------------------------------------------------- #
# expected cells -- the dataset manifest is the only authority
# --------------------------------------------------------------------------- #
def expected_cells(mode: str, shard: int, *, profiles=None, paths=None) -> list[CellKey]:
    """Enumerate every cell the shard is supposed to produce.

    Read from the shard DATASET manifests, never from what happens to be on
    disk in the results tree -- otherwise a missing cell could never be
    detected, because the evidence of its absence is its absence.
    """
    paths = paths or lib.campaign_paths()
    profiles = list(profiles or lib.PROFILES)
    out: list[CellKey] = []
    for profile in profiles:
        label = lib.Cell(mode, profile, shard).shard_label
        mf = paths["datasets"] / "_shards" / mode / profile / label / "manifest.json"
        if not mf.is_file():
            raise SystemExit(f"missing shard dataset manifest: {mf}")
        manifest = json.loads(mf.read_text(encoding="utf-8"))
        for t in manifest.get("tasks", []):
            directory = str(t["directory"])
            arm = directory.split("-", 2)[2]
            out.append(CellKey(lib.CAMPAIGN_ID, mode, profile, int(shard),
                               str(t["base_task_id"]), arm))
    return out


def task_dir_index(mode: str, shard: int, profile: str, *, paths=None) -> dict[str, str]:
    """{task directory -> base_task_id} for one profile's shard dataset."""
    paths = paths or lib.campaign_paths()
    label = lib.Cell(mode, profile, shard).shard_label
    mf = paths["datasets"] / "_shards" / mode / profile / label / "manifest.json"
    manifest = json.loads(mf.read_text(encoding="utf-8"))
    return {str(t["directory"]): str(t["base_task_id"]) for t in manifest.get("tasks", [])}


# --------------------------------------------------------------------------- #
# observation classification
# --------------------------------------------------------------------------- #
def _budget_censored(trial_path: Path) -> bool:
    for name in lib.BUDGET_MARKER_FILES if hasattr(lib, "BUDGET_MARKER_FILES") else ():
        if (trial_path / name).exists():
            return True
    text = ""
    for name in ("exception.txt", "trial.log"):
        p = trial_path / name
        if p.is_file():
            try:
                text += p.read_text(errors="replace")
            except OSError:
                pass
    low = text.lower()
    return any(m.lower() in low for m in lib.BUDGET_MARKERS)


def classify_observation(trial_path: Path, *, campaign_root: Path | None = None
                         ) -> Observation:
    """Classify ONE trial directory. Never reads the reward VALUE for validity."""
    trial_path = Path(trial_path)
    obs = Observation(trial_dir=trial_path.name, run_dir="", status=OTHER_INVALID,
                      path=str(trial_path))

    result = lib.jload(trial_path / "result.json") or {}
    traj = lib.jload(trial_path / "agent" / "trajectory.json") or {}

    agent = traj.get("agent") or {}
    fm = traj.get("final_metrics") or {}
    obs.model_name = str(agent.get("model_name") or "")
    obs.completion_tokens = int(fm.get("total_completion_tokens") or 0)
    obs.steps = int(fm.get("total_steps") or len(traj.get("steps") or []))
    cost = fm.get("total_cost_usd")
    obs.cost_usd = float(cost) if isinstance(cost, (int, float)) else None
    obs.started_at = str(result.get("started_at") or "")
    obs.finished_at = str(result.get("finished_at") or "")
    exc = result.get("exception_info") or {}
    obs.agent_exit_exception = str(exc.get("exception_type") or "") if exc else ""
    rewards = ((result.get("verifier_result") or {}).get("rewards") or {})
    r = rewards.get("reward")
    obs.reward = float(r) if isinstance(r, (int, float)) else None

    # Contamination guard: the task this trial ran must belong to THIS campaign.
    if campaign_root is not None:
        task_path = str(((result.get("task_id") or {}).get("path")) or "")
        if task_path and not str(Path(task_path)).startswith(str(Path(campaign_root))):
            obs.status, obs.reason = OTHER_INVALID, (
                f"task path {task_path} is outside the campaign namespace")
            return obs

    if not traj:
        obs.status, obs.reason = PRE_MODEL_FAILURE, "no trajectory written"
        return obs
    if "synthetic" in obs.model_name.lower() or not obs.model_name:
        obs.status, obs.reason = OTHER_INVALID, (
            f"synthetic/absent model {obs.model_name!r} - no real inference occurred")
        return obs
    if _budget_censored(trial_path):
        obs.status, obs.reason = OTHER_INVALID, "budget-censored trial"
        return obs

    produced_output = obs.completion_tokens > 0
    agent_exec = result.get("agent_execution") or {}
    verifier = result.get("verifier") or {}
    verdict_reached = bool(verifier.get("finished_at")) and obs.reward is not None
    agent_finished = bool(agent_exec.get("finished_at"))

    if not produced_output:
        obs.status, obs.reason = PRE_MODEL_FAILURE, (
            "model produced no completion tokens")
        return obs
    if not agent_finished:
        obs.status, obs.reason = PARTIAL_MODEL_FAILURE, (
            "agent execution has no finished_at - trial was truncated mid-run")
        return obs
    if not verdict_reached:
        obs.status, obs.reason = PARTIAL_MODEL_FAILURE, (
            "agent finished but the verifier reached no verdict - nothing to observe")
        return obs

    # The trial completed its full lifecycle: the model produced output, the
    # agent execution ended, and the verifier scored the resulting workspace.
    #
    # `exception_info` is recorded but is deliberately NOT disqualifying here.
    # The 2026-09-02 llama run is the case that settles it: three trials worked
    # for 1h54m, hit mini-swe-agent's step cap at 222 steps, and exited non-zero
    # (`NonZeroAgentExitCodeError`) -- and the verifier then scored the workspace
    # and returned a full rubric verdict (reward 0.0, rubrics_agg_score 0.375).
    # An agent exhausting its step budget is a RESULT, not a broken trial.
    # Discarding those would quietly drop exactly the cases where the model
    # floundered, which is outcome-dependent validity by the back door -- the
    # same error as re-running a trajectory because its reward was 0.
    note = (f" (agent exited via {obs.agent_exit_exception} after the verifier "
            f"verdict)" if obs.agent_exit_exception else "")
    obs.status, obs.reason = COMPLETE_VALID, (
        f"real model output ({obs.completion_tokens:,} completion tokens) and a "
        f"verifier verdict (reward={obs.reward}){note}")
    return obs


# --------------------------------------------------------------------------- #
# audit
# --------------------------------------------------------------------------- #
def shard_run_dirs(mode: str, shard: int, *, profiles=None, paths=None
                   ) -> dict[str, list[Path]]:
    """Every Harbor output directory for this shard, per profile.

    Includes run directories the controller never reaped -- the crash left four
    batches in flight, and their trajectories are on disk and perfectly valid.
    """
    paths = paths or lib.campaign_paths()
    profiles = list(profiles or lib.PROFILES)
    root = paths[mode] if mode in paths else paths["root"] / mode
    out: dict[str, list[Path]] = {p: [] for p in profiles}
    if not Path(root).is_dir():
        return out
    for d in sorted(Path(root).iterdir()):
        if not d.is_dir():
            continue
        for p in profiles:
            label = lib.Cell(mode, p, shard).shard_label
            if f"-{mode}-{p}-{label}-" in d.name:
                out[p].append(d)
                break
    return out


def audit(mode: str, shard: int, *, profiles=None, paths=None) -> dict:
    """Classify all expected cells against everything actually on disk."""
    paths = paths or lib.campaign_paths()
    profiles = list(profiles or lib.PROFILES)
    campaign_root = paths["root"]

    records: dict[str, CellRecord] = {}
    for ck in expected_cells(mode, shard, profiles=profiles, paths=paths):
        records[ck.key] = CellRecord(ck, MISSING, [], "no observation on disk")

    runs = shard_run_dirs(mode, shard, profiles=profiles, paths=paths)
    unmapped: list[str] = []
    for profile, dirs in runs.items():
        index = task_dir_index(mode, shard, profile, paths=paths)
        for rd in dirs:
            for trial in lib.scan_run_dir(rd):
                matches = [p for p in rd.rglob(trial.trial_dir) if p.is_dir()]
                if not matches:
                    continue
                tp = matches[0]
                task_dir = trial.trial_dir.split("__", 1)[0]
                base_task_id = index.get(task_dir)
                if base_task_id is None:
                    unmapped.append(f"{rd.name}/{trial.trial_dir}")
                    continue
                arm = task_dir.split("-", 2)[2]
                ck = CellKey(lib.CAMPAIGN_ID, mode, profile, int(shard),
                             base_task_id, arm)
                rec = records.get(ck.key)
                if rec is None:
                    unmapped.append(f"{rd.name}/{trial.trial_dir}")
                    continue
                obs = classify_observation(tp, campaign_root=campaign_root)
                obs.run_dir = rd.name
                rec.observations.append(obs)

    duplicates: list[str] = []
    for rec in records.values():
        if not rec.observations:
            continue
        valid = [o for o in rec.observations if o.status == COMPLETE_VALID]
        if len(valid) > 1:
            rec.status = DUPLICATE
            rec.reason = (f"{len(valid)} COMPLETE_VALID observations for one cell "
                          f"({', '.join(o.trial_dir for o in valid)}) - a human must "
                          "declare which attempt is authoritative")
            duplicates.append(rec.cell.key)
        elif valid:
            rec.status = COMPLETE_VALID
            rec.reason = valid[0].reason
        else:
            # Worst-but-most-informative of the failures we did see.
            order = [PARTIAL_MODEL_FAILURE, PRE_MODEL_FAILURE, OTHER_INVALID]
            best = min(rec.observations,
                       key=lambda o: order.index(o.status) if o.status in order else 99)
            rec.status, rec.reason = best.status, best.reason

    annotate_failures(records)

    counts = Counter(r.status for r in records.values())
    by_profile: dict[str, Counter] = defaultdict(Counter)
    for r in records.values():
        by_profile[r.cell.profile][r.status] += 1

    return {
        "campaign_id": lib.CAMPAIGN_ID,
        "mode": mode,
        "shard": int(shard),
        "generated_at": lib.now_iso(),
        "expected": len(records),
        "counts": {c: int(counts.get(c, 0)) for c in CLASSES},
        "counts_by_profile": {p: {c: int(by_profile[p].get(c, 0)) for c in CLASSES}
                              for p in profiles},
        "run_dirs": {p: [d.name for d in ds] for p, ds in runs.items()},
        "duplicates": duplicates,
        "unmapped_trials": unmapped,
        "repair_required": sum(1 for r in records.values() if r.needs_repair),
        "records": records,
    }


# Repair outlooks.
REPAIR_EXPECTED = "expected_to_succeed"      # nothing says it will fail again
REPAIR_FUTILE = "expected_to_reproduce"      # deterministic; a re-run buys the same failure
REPAIR_UNKNOWN = "unknown"


def annotate_failures(records: dict[str, CellRecord]) -> None:
    """Attach a failure class and a repair outlook to every failing cell.

    This is where the audit stops being optimistic. A cell whose failure is a
    provider content refusal, or an image build that has already reproduced
    across profiles, will fail again in exactly the same way -- listing it as
    "to repair" without saying so would promise a 1,200-cell shard that the
    evidence says is not reachable by re-running.
    """
    failing = [r for r in records.values() if r.needs_repair and r.observations]
    order = sorted(failing, key=lambda r: (r.cell.profile, r.cell.base_task_id,
                                           r.cell.arm))

    # PASS 1: classify everything and count how often each signature appears.
    # The audit is retrospective, so unlike the live controller -- which must
    # retry once to EARN reproduction evidence -- we already hold the whole
    # record. A signature seen in two independent cells has demonstrably
    # reproduced, and calling the alphabetically-first one "expected to
    # succeed" would just be an artifact of iteration order.
    first: dict[str, object] = {}
    seen = Counter()
    for rec in order:
        paths = [Path(o.path) for o in rec.observations if o.path]
        if not paths:
            continue
        cls = failures.classify_trial_dir(paths[0])
        first[rec.cell.key] = cls
        if cls.signature:
            seen[cls.signature] += 1

    reproduced = {sig for sig, n in seen.items() if n >= 2}

    # PASS 2: apply the reproduction evidence uniformly.
    for rec in order:
        cls = first.get(rec.cell.key)
        if cls is None:
            continue
        if cls.signature in reproduced:
            cls = failures.apply_reproduction_evidence(cls, reproduced)
        rec.failure_class = cls.failure_class
        if cls.failure_class in (failures.PROVIDER_REFUSAL,
                                 failures.DETERMINISTIC_INFRA):
            rec.repair_outlook = REPAIR_FUTILE
        elif cls.retryable:
            rec.repair_outlook = REPAIR_EXPECTED
        else:
            rec.repair_outlook = REPAIR_UNKNOWN
    for rec in records.values():
        if rec.needs_repair and not rec.observations:
            rec.failure_class = "never_ran"
            rec.repair_outlook = REPAIR_EXPECTED


def write_audit(result: dict, out_path: Path) -> Path:
    """Write the per-cell JSONL. One line per expected cell, always all 1,200."""
    out_path = Path(out_path)
    lib.assert_campaign_path(out_path.parent, "cell audit directory")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for rec in result["records"].values():
            fh.write(json.dumps(rec.as_dict()) + "\n")
    return out_path


def summary_of(result: dict) -> dict:
    return {k: v for k, v in result.items() if k != "records"}


# --------------------------------------------------------------------------- #
# repair planning
# --------------------------------------------------------------------------- #
def repair_plan(result: dict) -> dict:
    """Cells that must be re-run. COMPLETE_VALID cells are frozen, never listed.

    Refuses to plan while any DUPLICATE is unresolved: repairing around an
    unresolved provenance conflict would bake the conflict into the corpus.
    """
    if result["duplicates"]:
        raise SystemExit(
            f"refusing to plan repair: {len(result['duplicates'])} cell(s) have "
            f"more than one COMPLETE_VALID trajectory and need an explicit "
            f"provenance decision:\n  " + "\n  ".join(result["duplicates"][:20]))

    cells = [r for r in result["records"].values() if r.needs_repair]
    by_profile: dict[str, list[dict]] = defaultdict(list)
    for r in cells:
        by_profile[r.cell.profile].append({
            "cell_key": r.cell.key,
            "base_task_id": r.cell.base_task_id,
            "arm": r.cell.arm,
            "task_dir": r.cell.task_dir,
            "status": r.status,
            "reason": r.reason,
            "failure_class": r.failure_class,
            "repair_outlook": r.repair_outlook,
            "prior_trial_dirs": [o.trial_dir for o in r.observations],
            "prior_run_dirs": sorted({o.run_dir for o in r.observations if o.run_dir}),
        })
    return {
        "campaign_id": lib.CAMPAIGN_ID,
        "mode": result["mode"],
        "shard": result["shard"],
        "generated_at": lib.now_iso(),
        "expected": result["expected"],
        "complete_valid": result["counts"][COMPLETE_VALID],
        "repair_required": len(cells),
        "counts_by_status": {c: sum(1 for r in cells if r.status == c)
                             for c in NEEDS_REPAIR},
        "counts_by_outlook": {
            o: sum(1 for r in cells if r.repair_outlook == o)
            for o in (REPAIR_EXPECTED, REPAIR_FUTILE, REPAIR_UNKNOWN)},
        "expected_to_reproduce": sorted(
            r.cell.key for r in cells if r.repair_outlook == REPAIR_FUTILE),
        "by_profile": {p: sorted(v, key=lambda d: (d["base_task_id"], d["arm"]))
                       for p, v in sorted(by_profile.items())},
    }


def frozen_cells(result: dict) -> list[str]:
    """Cell keys that are COMPLETE_VALID and must never be re-run."""
    return sorted(r.cell.key for r in result["records"].values()
                  if r.status == COMPLETE_VALID)


def validate_shard_complete(result: dict, *, expect: int | None = None) -> dict:
    """Would this shard validate as a complete N-cell corpus right now?

    Deliberately reports rather than raises, so it can be used both as a gate
    and as a progress readout during repair. `ok` requires EVERY expected cell
    to hold exactly one COMPLETE_VALID observation -- no duplicates, no
    unmapped trials, no substitutions.
    """
    expect = int(expect if expect is not None else result["expected"])
    recs = result["records"]
    complete = [k for k, r in recs.items() if r.status == COMPLETE_VALID]
    outstanding = sorted(k for k, r in recs.items() if r.status != COMPLETE_VALID)
    multi = sorted(k for k, r in recs.items()
                   if sum(1 for o in r.observations if o.status == COMPLETE_VALID) > 1)
    problems = []
    if len(recs) != expect:
        problems.append(f"expected {expect} cells, the plan enumerates {len(recs)}")
    if outstanding:
        problems.append(f"{len(outstanding)} cell(s) are not COMPLETE_VALID")
    if multi:
        problems.append(f"{len(multi)} cell(s) have more than one valid trajectory")
    if result["unmapped_trials"]:
        problems.append(f"{len(result['unmapped_trials'])} trial(s) map to no expected cell")
    return {
        "ok": not problems,
        "expected": expect,
        "complete_valid": len(complete),
        "outstanding": outstanding[:50],
        "outstanding_total": len(outstanding),
        "duplicates": multi,
        "unmapped_trials": len(result["unmapped_trials"]),
        "problems": problems,
    }


# --------------------------------------------------------------------------- #
# remaining-work accounting (used by the cost model / preflight)
# --------------------------------------------------------------------------- #
def repair_plan_path(mode: str, shard: int, *, paths=None) -> Path:
    paths = paths or lib.campaign_paths()
    return paths["provenance"] / f"{mode}-shard{shard}-repair-plan.json"


def accepted_cell_keys(paths=None) -> set:
    """Profile/shard keys already accepted as complete by campaign.provenance."""
    paths = paths or lib.campaign_paths()
    state = paths["provenance"] / "accepted_runs.json"
    if not state.is_file():
        return set()
    try:
        blob = json.loads(state.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    return {e["cell"] for e in blob.get("accepted", [])
            if e.get("status") == "complete" and e.get("cell")}


def remaining_trials(paths=None, cells=None) -> dict:
    """Trials that still require INFERENCE, per profile/shard cell.

    Three sources, in order of authority:

      1. An accepted profile/shard needs nothing.
      2. A written repair plan says exactly how many experimental cells are
         still outstanding -- so an interrupted 300-trial profile/shard with
         276 valid trajectories on disk costs 24 trials to finish, not 300.
      3. Otherwise the cell has not started and needs its full trial count.

    Source (2) is the fix for the planning error the interrupted shard exposed:
    charging a resumed shard at full price both overstates remaining cost by
    thousands of dollars and, through preflight's budget gate, can refuse to
    launch work the budget comfortably covers.
    """
    paths = paths or lib.campaign_paths()
    cells = list(cells if cells is not None else lib.all_cells())
    accepted = accepted_cell_keys(paths)
    plans: dict[tuple, dict] = {}
    out: dict[str, dict] = {}
    for cell in cells:
        if cell.key in accepted:
            out[cell.key] = {"remaining_trials": 0, "basis": "accepted"}
            continue
        pk = (cell.mode, cell.shard_index)
        if pk not in plans:
            p = repair_plan_path(cell.mode, cell.shard_index, paths=paths)
            try:
                plans[pk] = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                plans[pk] = {}
        plan = plans[pk]
        if plan.get("campaign_id") == lib.CAMPAIGN_ID and "by_profile" in plan:
            n = len(plan["by_profile"].get(cell.profile) or [])
            out[cell.key] = {"remaining_trials": n, "basis": "repair_plan",
                             "of_expected": cell.expected_trials}
        else:
            out[cell.key] = {"remaining_trials": cell.expected_trials,
                             "basis": "not_started"}
    return out


def render_summary(result: dict) -> str:
    c = result["counts"]
    lines = [
        f"Campaign {result['campaign_id']}  {result['mode']} shard {result['shard']}",
        f"generated {result['generated_at']}",
        "",
        f"expected               {result['expected']:>6}",
    ]
    for name in CLASSES:
        lines.append(f"{name:<22} {c[name]:>6}")
    lines += ["", f"repair required        {result['repair_required']:>6}", "",
              f"{'profile':<9} " + " ".join(f"{n[:9]:>9}" for n in CLASSES)]
    for p, cc in result["counts_by_profile"].items():
        lines.append(f"{p:<9} " + " ".join(f"{cc[n]:>9}" for n in CLASSES))
    outlook = Counter(r.repair_outlook for r in result["records"].values()
                      if r.needs_repair)
    if outlook:
        lines += ["", "repair outlook:"]
        for k, v in sorted(outlook.items()):
            lines.append(f"  {k:<24} {v:>6}")
    if result["duplicates"]:
        lines += ["", f"DUPLICATES ({len(result['duplicates'])}) - repair is blocked:"]
        lines += [f"  {d}" for d in result["duplicates"][:20]]
    if result["unmapped_trials"]:
        lines += ["", f"unmapped trials: {len(result['unmapped_trials'])}"]
    return "\n".join(lines)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Cell-level Campaign V2 audit and repair plan.")
    ap.add_argument("--mode", choices=lib.MODES, required=True)
    ap.add_argument("--shard", type=int, choices=lib.SHARD_INDICES, required=True)
    ap.add_argument("--write", action="store_true",
                    help="write the audit JSONL and repair plan under provenance/")
    a = ap.parse_args()

    res = audit(a.mode, a.shard)
    print(render_summary(res))
    if a.write:
        paths = lib.campaign_paths()
        jl = write_audit(res, paths["provenance"] / f"{a.mode}-shard{a.shard}-cell-audit.jsonl")
        sm = paths["provenance"] / f"{a.mode}-shard{a.shard}-cell-audit-summary.json"
        sm.write_text(json.dumps(summary_of(res), indent=2) + "\n")
        print(f"\nwrote {jl}\nwrote {sm}")
        try:
            plan = repair_plan(res)
        except SystemExit as exc:
            print(f"\n{exc}")
            return 2
        rp = paths["provenance"] / f"{a.mode}-shard{a.shard}-repair-plan.json"
        rp.write_text(json.dumps(plan, indent=2) + "\n")
        print(f"wrote {rp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
