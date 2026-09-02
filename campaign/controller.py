#!/usr/bin/env python3
"""Adaptive execution controller for one Campaign V2 shard.

Replaces the sequential `for profile in PROFILES; do run_one; done` loop with a
concurrent, allocator-driven one. The audit measured that 47.0% of historically
active minutes had only ONE profile running; running all four concurrently and
re-planning at batch boundaries is where the wall-clock saving comes from.

ARCHITECTURAL CONSTRAINT (see audit S4): `harbor run -n <concurrency>` is a
launch-time flag, so concurrency cannot be retuned on an in-flight job. The
controller therefore splits each cell into BATCHES; the allocator is consulted
before each batch launch, and capacity freed by a finishing profile is handed
to whoever still has a queue at that boundary. A batch already in flight keeps
the concurrency it was launched with -- that is a property of Harbor, not a
choice made here, and it bounds how fast redistribution can take effect.

Budget outranks TPM everywhere: the gateway budget is checked before the shard,
ON A TIMER WHILE WORK IS IN FLIGHT, and before every repair. The Aug-27 corpus
shows what happens otherwise -- 300 trial directories and 1,065 budget-marker
files after the key hit $10,001.94 of a $10,000 ceiling, all of it worthless.

THREE FIXES FROM THE 2026-09-02 SHARD-1 CRASH
---------------------------------------------
1. WORK-UNIT IDENTITY. A repair used to be queued as `Batch(base_count=0)`, a
   sentinel meaning "one trial, not one base task". That reached the base-task
   slicer, which correctly refused with `--base-task-count must be >= 1`, and
   the uncaught SystemExit killed the whole shard with four batches in flight.
   Repairs are now a DIFFERENT TYPE (`RepairUnit`) carrying explicit cells, and
   `_slice_dataset` refuses to accept anything that is not a real `Batch`. The
   sentinel is gone, and so is the path that produced it.

2. TOKEN ACCOUNTING. `reap` used to record every trial's lifetime token total
   against `batch.finished_at`, collapsing ninety minutes of traffic onto one
   instant. That produced a 12,298,656 TPM reading -- 2.46x the hard ceiling --
   when real traffic that minute was 31,521, and throttled llama from 21 workers
   to 15 for nothing. Tokens now enter the meter at the per-step timestamps the
   trajectories already carry (`campaign.tokens`), and the resulting series is
   labelled RETROSPECTIVE and never used to throttle.

3. CRASH ACCOUNTING. An uncaught exception meant `main()` never wrote its
   result file, so `campaign.sh` recorded `attempt_no_output` and the run dirs
   that DID exist went unrecorded. `run()` now converts any unexpected exception
   into a failed result that still names every run directory produced.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path

from campaign import lib, failures, retry_dataset, tokens, scheduler as sched
from campaign.scheduler import Scheduler, PROFILE_LIMITS

BATCH_BASE_TASKS = int(os.environ.get("CAMPAIGN_BATCH_BASE_TASKS", "10"))
STATUS_INTERVAL_SEC = int(os.environ.get("CAMPAIGN_STATUS_INTERVAL", "45"))
BUDGET_POLL_SEC = int(os.environ.get("CAMPAIGN_BUDGET_POLL", "300"))
# Stop launching if remaining budget would not cover the work still in flight
# plus this reserve. Sized above the worst single observed trial ($24.90).
BUDGET_RESERVE_USD = float(os.environ.get("CAMPAIGN_BUDGET_RESERVE", "150"))
# How long a broken budget probe may persist before we stop rather than keep
# draining blind. We fail closed on LAUNCHING immediately; this bounds the wait.
BUDGET_PROBE_GRACE_SEC = float(os.environ.get("CAMPAIGN_BUDGET_PROBE_GRACE", "1800"))
# Cells per repair Harbor invocation. Repair cells are scattered across base
# tasks, so this is purely a batching-efficiency knob; the dataset always holds
# exactly the listed cells and never a sibling arm that was not asked for.
REPAIR_CELLS_PER_UNIT = int(os.environ.get("CAMPAIGN_REPAIR_CELLS_PER_UNIT", "25"))


# --------------------------------------------------------------------------- #
# budget guard
# --------------------------------------------------------------------------- #
class BudgetStop(Exception):
    """Raised to stop the shard cleanly. Never retried, never suppressed."""


@dataclass
class BudgetView:
    spend: float = 0.0
    max_budget: float = 0.0
    remaining: float = 0.0
    checked_at: float = 0.0
    ok: bool = False
    error: str = ""


class BudgetGuard:
    """Non-billable gateway budget monitoring.

    Uses lib.probe_budget: a pinned allowed model with an EMPTY message list,
    which upstream rejects at HTTP 400 before inference, so the probe costs
    nothing (`x-litellm-response-cost: 0`) while still returning the budget
    headers.
    """

    def __init__(self, cost_per_trial: dict[str, float], *, enabled: bool = True,
                 reserve: float = BUDGET_RESERVE_USD,
                 poll_interval: float = BUDGET_POLL_SEC,
                 probe=None, clock=time.time):
        self.cost_per_trial = cost_per_trial
        self.enabled = enabled
        self.reserve = reserve
        self.poll_interval = float(poll_interval)
        self.probe = probe or lib.probe_budget
        self.clock = clock
        self.view = BudgetView()
        self._last_poll = 0.0
        self.poll_count = 0            # observable so tests can prove polling
        self.first_failure_at = 0.0

    def poll(self, *, force: bool = False, now: float | None = None) -> BudgetView:
        now = self.clock() if now is None else now
        if not self.enabled:
            self.view = BudgetView(ok=True, error="budget guard disabled")
            return self.view
        if not force and self._last_poll and (now - self._last_poll) < self.poll_interval:
            return self.view
        self._last_poll = now
        self.poll_count += 1
        st = self.probe()
        if not st.ok:
            self.view = BudgetView(ok=False, error=st.error or "probe failed",
                                   checked_at=now)
            if not self.first_failure_at:
                self.first_failure_at = now
            return self.view
        self.first_failure_at = 0.0
        self.view = BudgetView(spend=st.spend or 0.0, max_budget=st.max_budget or 0.0,
                               remaining=st.remaining or 0.0, checked_at=now, ok=True)
        return self.view

    def due(self, now: float | None = None) -> bool:
        now = self.clock() if now is None else now
        return (now - self._last_poll) >= self.poll_interval

    def estimated_cost(self, remaining_trials: dict[str, int]) -> float:
        return sum(self.cost_per_trial.get(p, 0.0) * n for p, n in remaining_trials.items())

    def exhausted(self) -> bool:
        """Hard exhaustion. Terminal: in-flight work is not worth preserving."""
        v = self.view
        return bool(self.enabled and v.ok and v.remaining <= self.reserve)

    def assert_can_continue(self, remaining_trials: dict[str, int], *,
                            what: str = "continue") -> None:
        """Raise BudgetStop rather than let the gateway cut work off mid-flight."""
        v = self.poll()
        if not self.enabled:
            return
        if not v.ok:
            raise BudgetStop(f"cannot verify gateway budget before {what}: {v.error}")
        need = self.estimated_cost(remaining_trials) + self.reserve
        if v.remaining <= self.reserve:
            raise BudgetStop(
                f"gateway budget effectively exhausted before {what}: "
                f"remaining ${v.remaining:,.2f} <= reserve ${self.reserve:,.2f}")
        if v.remaining < need:
            raise BudgetStop(
                f"insufficient budget to {what}: remaining ${v.remaining:,.2f} < "
                f"${need:,.2f} (est. ${need - self.reserve:,.2f} of work + "
                f"${self.reserve:,.2f} reserve). Stopping cleanly BEFORE starting "
                "more trials rather than producing budget-censored failures.")


# --------------------------------------------------------------------------- #
# work units -- a repair is a DIFFERENT TYPE, not a Batch with a magic field
# --------------------------------------------------------------------------- #
@dataclass
class WorkUnit:
    profile: str
    index: int = 0
    expected_trials: int = 1
    dataset: Path | None = None
    concurrency: int = 0
    run_dir: str = ""
    rc: int | None = None
    started_at: float = 0.0
    finished_at: float = 0.0
    proc: object = None
    outfile: str = ""

    kind = "work"

    @property
    def done(self) -> bool:
        return self.rc is not None


@dataclass
class Batch(WorkUnit):
    """N contiguous base tasks x ALL their arms. Sliced by base-task index."""
    start_base: int = 0
    base_count: int = 0

    kind = "batch"


@dataclass
class RepairUnit(WorkUnit):
    """An explicit list of task/arm cells. NEVER sliced by base-task index.

    `cells` holds dataset task directory names ('ea-10d4b434-eval-src'). A
    retry queued from a mid-run failure carries exactly one; a bulk repair from
    the cell audit carries the cells that audit found invalid, and nothing else.
    """
    cells: list[str] = field(default_factory=list)
    label: str = ""
    # Set only for a single-cell retry with retry-ledger lineage.
    original_trial_id: str = ""
    retry_trial_id: str = ""
    retry_number: int = 0
    failure_class: str = ""
    failure_reason: str = ""
    original_run_dir: str = ""

    kind = "repair"

    @property
    def is_retry(self) -> bool:
        return bool(self.retry_trial_id)


@dataclass
class ControllerResult:
    ok: bool
    stopped_reason: str = ""
    run_dirs: dict[str, list[str]] = field(default_factory=dict)
    retries: dict[str, int] = field(default_factory=dict)
    elapsed_sec: float = 0.0
    crashed: bool = False
    budget_polls: int = 0
    spend_usd: float | None = None
    remaining_budget_usd: float | None = None
    retrospective_peak_tpm: float = 0.0
    unattributed_metered_tokens: int = 0

    def as_dict(self):
        return asdict(self)


class Controller:
    def __init__(self, mode: str, shard: int, *, paths=None, dry_run: bool = False,
                 budget: BudgetGuard | None = None, profiles=lib.PROFILES,
                 clock=time.time, repair_plan: dict | None = None,
                 sleep=time.sleep):
        self.mode = mode
        self.shard = shard
        self.paths = paths or lib.campaign_paths()
        self.dry_run = dry_run
        self.profiles = list(profiles)
        self.clock = clock
        self.sleep = sleep
        self.sch = Scheduler(self.profiles)
        self.budget = budget or BudgetGuard(self._cost_model(), enabled=not dry_run,
                                            clock=clock)
        self.repair_plan = repair_plan
        self.units: dict[str, list[WorkUnit]] = {}
        self.inflight: dict[str, WorkUnit] = {}
        self.completed_trials: dict[str, int] = {p: 0 for p in self.profiles}
        self.failed_trials: dict[str, int] = {p: 0 for p in self.profiles}
        self.retry_stats = {"attempted": 0, "recovered": 0, "permanent": 0}
        self.real_429 = 0
        self.started = clock()
        self.ledger = failures.RetryLedger(self.paths["provenance"] / "retries.jsonl")
        self.log_path = self.paths["logs"] / f"controller-{mode}-shard{shard}.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_status = 0.0
        self._retry_seq = 0
        self.draining = False
        self.drain_reason = ""
        # Retrospective token timeline, for reporting only. See campaign.tokens.
        self.timeline = tokens.TokenTimeline(self.sch.meter.window_sec)
        self._seen_signatures: set[str] = set(self.ledger.signatures())

    # -- setup --------------------------------------------------------------
    @staticmethod
    def _cost_model() -> dict[str, float]:
        try:
            from campaign import cost_model
            est = cost_model.estimate()
            return {p: (est["cells"].get(f"full/{p}", {}).get("mean_usd_per_trial") or 0.0)
                    for p in lib.PROFILES}
        except Exception:
            return {"claude": 2.64, "fable": 3.65, "codex": 1.22, "llama": 0.0}

    def plan_batches(self) -> None:
        """Full-shard plan: every base task, all arms."""
        cell0 = lib.Cell(self.mode, self.profiles[0], self.shard)
        base_total = cell0.base_tasks
        variants = lib.VARIANTS_PER_TASK[self.mode]
        for p in self.profiles:
            out: list[WorkUnit] = []
            idx, start = 0, 0
            while start < base_total:
                n = min(BATCH_BASE_TASKS, base_total - start)
                idx += 1
                out.append(Batch(profile=p, index=idx, start_base=start,
                                 base_count=n, expected_trials=n * variants))
                start += n
            self.units[p] = out
            self.sch.states[p].queued = sum(b.expected_trials for b in out)

    def plan_repair(self) -> None:
        """Repair plan: ONLY the cells the audit found invalid or missing.

        A COMPLETE_VALID trajectory from an earlier attempt is frozen -- it is
        not in the plan, so it is never re-run and never re-purchased.
        """
        plan = self.repair_plan or {}
        by_profile = plan.get("by_profile") or {}
        for p in self.profiles:
            cells = [str(c["task_dir"]) for c in by_profile.get(p, [])]
            out: list[WorkUnit] = []
            for i in range(0, len(cells), REPAIR_CELLS_PER_UNIT):
                chunk = cells[i:i + REPAIR_CELLS_PER_UNIT]
                out.append(RepairUnit(
                    profile=p, index=len(out) + 1, expected_trials=len(chunk),
                    cells=chunk,
                    label=f"repair-{self.mode}-{p}-s{self.shard}-{len(out) + 1:02d}"))
            self.units[p] = out
            self.sch.states[p].queued = sum(u.expected_trials for u in out)

    def plan(self) -> None:
        if self.repair_plan is not None:
            self.plan_repair()
        else:
            self.plan_batches()

    # -- status -------------------------------------------------------------
    def _remaining_trials(self) -> dict[str, int]:
        return {p: sum(u.expected_trials for u in self.units.get(p, []) if not u.done)
                for p in self.profiles}

    def status_snapshot(self, decision=None) -> dict:
        now = self.clock()
        elapsed = now - self.started
        done = sum(self.completed_trials.values())
        remaining = sum(self._remaining_trials().values())
        # A rate computed over a few seconds of wall clock is noise, not a
        # measurement; report nothing rather than an indefensible ETA.
        rate = (done / (elapsed / 3600.0)) if (elapsed >= 300.0 and done) else None
        v = self.budget.view
        retro = self.sch.meter.tpm(now)
        fresh = (now - v.checked_at) if (v.ok and v.checked_at) else None
        return {
            "ts": lib.now_iso(),
            "campaign_id": lib.CAMPAIGN_ID,
            "mode": self.mode,
            "shard": self.shard,
            "plan": "repair" if self.repair_plan is not None else "full",
            "elapsed_sec": round(now - self.started, 1),
            "active_workers": {p: (self.inflight[p].concurrency if p in self.inflight else 0)
                               for p in self.profiles},
            "queued_trials": self._remaining_trials(),
            "completed_trials": dict(self.completed_trials),
            "failed_trials": dict(self.failed_trials),
            # NAMED retrospective on purpose. Token events only exist once a
            # trajectory is written, so this series lags reality and must never
            # be read as live gateway pressure -- that misreading is what
            # produced the 12.3M phantom and two false throttles.
            "retrospective_metered_tpm": round(retro, 1),
            "retrospective_tpm_by_profile": {
                k: round(x, 1) for k, x in self.sch.meter.tpm_by_profile(now).items()},
            "tpm_is_retrospective": True,
            "live_tpm_source": "none (no per-request metering available)",
            "projected_tpm": round(decision.projected_tpm, 1) if decision else None,
            "tpm_soft_ceiling": self.sch.soft_tpm,
            "tpm_hard_ceiling": self.sch.hard_tpm,
            "tpm_unattributed_tokens": self.timeline.unattributed_metered,
            "real_429_total": self.real_429,
            "retries": dict(self.retry_stats),
            "draining": self.draining,
            "drain_reason": self.drain_reason,
            "budget_checked": bool(v.ok and v.checked_at),
            "budget_polls": self.budget.poll_count,
            "budget_age_sec": round(fresh, 1) if fresh is not None else None,
            "budget_probe_error": v.error if not v.ok else "",
            "spend_usd": round(v.spend, 2) if (v.ok and v.checked_at) else None,
            "remaining_budget_usd": round(v.remaining, 2) if (v.ok and v.checked_at) else None,
            "trials_per_hour": round(rate, 1) if rate else None,
            "eta_hours": round(remaining / rate, 2) if (rate and remaining) else None,
            "decision": decision.as_dict() if decision else None,
        }

    def emit_status(self, decision=None, *, force: bool = False) -> None:
        now = self.clock()
        if not force and (now - self._last_status) < STATUS_INTERVAL_SEC:
            return
        self._last_status = now
        snap = self.status_snapshot(decision)
        with open(self.log_path, "a") as fh:
            fh.write(json.dumps(snap) + "\n")
        aw = snap["active_workers"]
        q = snap["queued_trials"]
        c = snap["completed_trials"]
        eta = f"{snap['eta_hours']:.1f}h" if snap["eta_hours"] is not None else "n/a"
        rate_s = f"{snap['trials_per_hour']:.0f} trials/h" if snap["trials_per_hour"] else "rate n/a"
        if snap["budget_checked"]:
            age = snap["budget_age_sec"]
            bud = (f"spend ${snap['spend_usd']:,.0f} rem ${snap['remaining_budget_usd']:,.0f}"
                   f" ({age:.0f}s ago)" if age is not None else "")
        else:
            bud = f"budget UNVERIFIED ({snap['budget_probe_error'] or 'not yet polled'})"
        proj = f"{snap['projected_tpm']:,.0f}" if snap["projected_tpm"] else "n/a"
        print(
            f"[{lib.CAMPAIGN_ID} {self.mode} shard {self.shard} {snap['plan']}] "
            f"t+{snap['elapsed_sec']/60:.0f}m  "
            f"projTPM {proj}/{self.sch.soft_tpm:,}  "
            f"retroTPM {snap['retrospective_metered_tpm']:,.0f}  "
            f"429s {self.real_429}  "
            f"repairs {self.retry_stats['attempted']}a/{self.retry_stats['recovered']}r/"
            f"{self.retry_stats['permanent']}p  "
            f"{bud}  {rate_s}  ETA {eta}"
            + ("  DRAINING" if self.draining else ""),
            flush=True)
        print("    " + "  ".join(
            f"{p}: {aw[p]}w {c[p]}done/{q[p]}queued" for p in self.profiles), flush=True)

    # -- datasets -----------------------------------------------------------
    def _slice_dataset(self, b: WorkUnit) -> Path:
        """Base-task slicer. Accepts ONLY a real Batch with >= 1 base task.

        The guards are the fix for the 2026-09-02 crash: a repair can no longer
        reach this code path at all, so `--base-task-count 0` cannot be
        constructed even by a future caller that reinvents the sentinel.
        """
        if not isinstance(b, Batch) or b.kind != "batch":
            raise TypeError(
                f"_slice_dataset accepts base-task batches only, got "
                f"{type(b).__name__}(kind={getattr(b, 'kind', '?')}). Repairs "
                "must go through _repair_dataset -- they are individual cells, "
                "not a contiguous run of base tasks.")
        if b.base_count < 1:
            raise ValueError(
                f"batch {b.index} for {b.profile} has base_count={b.base_count}; "
                "the base-task slicer requires >= 1 and there is no 'zero base "
                "tasks' batch. A single-cell unit is a RepairUnit.")
        label = lib.Cell(self.mode, b.profile, self.shard).shard_label
        src = self.paths["datasets"] / "_shards" / self.mode / b.profile / label
        dst = (self.paths["datasets"] / "_batches" / self.mode / b.profile / label
               / f"batch-{b.index:02d}")
        lib.assert_campaign_path(dst, "batch dataset")
        if self.dry_run:
            return dst
        if not dst.is_dir():
            dst.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [sys.executable, str(lib.PROJECT_ROOT / "scripts" / "05_slice_dataset.py"),
                 "--project-root", str(lib.PROJECT_ROOT), "--source", str(src),
                 "--output", str(dst), "--start-index", str(b.start_base),
                 "--base-task-count", str(b.base_count),
                 "--label", f"campaign-{self.mode}-{b.profile}-s{self.shard}-b{b.index}"],
                check=True)
        return dst

    def _repair_dataset(self, u: RepairUnit) -> Path:
        """Exact-cell dataset. Never slices, never adds a sibling arm."""
        label = lib.Cell(self.mode, u.profile, self.shard).shard_label
        src = self.paths["datasets"] / "_shards" / self.mode / u.profile / label
        name = u.retry_trial_id or u.label or f"repair-{u.index:02d}"
        dst = (self.paths["datasets"] / "_repairs" / self.mode / u.profile / label
               / name)
        lib.assert_campaign_path(dst, "repair dataset")
        if self.dry_run:
            return dst
        dst.parent.mkdir(parents=True, exist_ok=True)
        if u.is_retry:
            if len(u.cells) != 1:
                raise ValueError(f"a retry must carry exactly one cell, got {u.cells}")
            return retry_dataset.build_retry_dataset(
                source_dataset=src, output=dst,
                original_trial_id=u.original_trial_id,
                retry_trial_id=u.retry_trial_id, retry_number=u.retry_number,
                failure_class=u.failure_class, failure_reason=u.failure_reason,
                original_run_dir=u.original_run_dir,
                cell=lib.Cell(self.mode, u.profile, self.shard).key)
        return retry_dataset.build_repair_dataset(
            source_dataset=src, output=dst, task_dirs=u.cells, label=u.label,
            provenance={"shard_index": self.shard, "controller_plan": "repair"})

    # -- execution ----------------------------------------------------------
    def launch(self, u: WorkUnit, concurrency: int) -> None:
        u.concurrency = concurrency
        u.started_at = self.clock()
        u.dataset = (self._repair_dataset(u) if isinstance(u, RepairUnit)
                     else self._slice_dataset(u))
        if self.dry_run:
            u.proc = None
            return
        import tempfile
        fd, u.outfile = tempfile.mkstemp(prefix="swe-run-")
        os.close(fd)
        env = dict(os.environ)
        env["HARBOR_CONCURRENCY"] = str(concurrency)
        env["SWE_RUN_OUTPUT_FILE"] = u.outfile
        env["RESULTS_ROOT"] = str(self.paths["root"])
        label = lib.Cell(self.mode, u.profile, self.shard).shard_label
        if isinstance(u, RepairUnit):
            suffix = f"-{label}-rep{u.index:02d}"
        else:
            suffix = f"-{label}-b{u.index:02d}"
        u.proc = subprocess.Popen(
            [str(lib.PROJECT_ROOT / "scripts" / "05_run_profile.sh"), self.mode, u.profile,
             "--dataset-override", str(u.dataset),
             "--shard-label", suffix],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    def _record_tokens(self, profile: str, trial_path: Path) -> None:
        """Feed token events at the times they were ACTUALLY spent.

        The alternative -- one lump at reap time -- is what manufactured the
        12.3M-TPM spike. `campaign.tokens` reads the per-step timestamps that
        every profile already emits.
        """
        ex = tokens.extract_trial(trial_path, profile=profile)
        for e in ex.events:
            self.sch.meter.record(profile, e.metered, e.ts)
        self.timeline.add_extraction(ex)

    def reap(self, u: WorkUnit) -> None:
        """Collect a finished unit: tokens into the meter, failures classified."""
        u.finished_at = self.clock()
        if u.outfile:
            try:
                u.run_dir = Path(u.outfile).read_text().strip()
                os.unlink(u.outfile)
            except OSError:
                pass
        if not u.run_dir or not Path(u.run_dir).is_dir():
            self.failed_trials[u.profile] += u.expected_trials
            if isinstance(u, RepairUnit) and u.is_retry:
                self._close_retry(u, "failed", reason="no run directory produced")
            return

        run_dir = Path(u.run_dir)
        observed_complete = 0
        for t in lib.scan_run_dir(run_dir):
            matches = [p for p in run_dir.rglob(t.trial_dir) if p.is_dir()]
            trial_path = matches[0] if matches else None
            if trial_path is not None:
                self._record_tokens(u.profile, trial_path)
            if t.status == lib.STATUS_COMPLETE:
                self.completed_trials[u.profile] += 1
                observed_complete += 1
            else:
                self.failed_trials[u.profile] += 1
                self._handle_failure(u, t, trial_path)

        if isinstance(u, RepairUnit) and u.is_retry:
            # A retry is accepted only if its single trial actually completed.
            ok = observed_complete >= 1
            self._close_retry(u, "accepted" if ok else "failed",
                              reason=("retry produced a complete trajectory" if ok
                                      else "retry did not produce a complete trajectory"))
            if ok:
                self.retry_stats["recovered"] += 1
        self.sch.states[u.profile].queued = self._remaining_trials()[u.profile]

    def _close_retry(self, u: RepairUnit, status: str, *, reason: str = "") -> None:
        """Close the ledger lineage. Every opened retry MUST reach here.

        The 2026-09-02 ledger has six records all stuck at `pending` with no
        `finished_at`, because `close_retry` existed but nothing ever called it.
        A permanently-pending record is indistinguishable from a retry that was
        queued and silently dropped.
        """
        try:
            self.ledger.close_retry(u.retry_trial_id, status,
                                    retry_run_dir=u.run_dir)
        except Exception as exc:                       # never lose the shard to bookkeeping
            print(f"WARNING: could not close retry {u.retry_trial_id}: {exc}",
                  file=sys.stderr, flush=True)

    def _handle_failure(self, u: WorkUnit, trial, trial_path: Path | None) -> None:
        cls = (failures.classify_trial_dir(trial_path) if trial_path is not None
               else failures.Classification(failures.PERMANENT,
                                            "trial directory not found", False, False))
        # Reproduction evidence: an infra fault that has already failed in this
        # campaign is deterministic, and retrying it just buys it again.
        cls = failures.apply_reproduction_evidence(cls, self._seen_signatures)
        if cls.signature:
            self._seen_signatures.add(cls.signature)

        if cls.failure_class == failures.BUDGET:
            raise BudgetStop(
                f"budget failure observed in {trial.trial_dir}: {cls.reason}. "
                "Stopping the shard immediately; budget failures are NEVER retried.")
        if cls.failure_class == failures.TRANSIENT_PROVIDER and "429" in cls.evidence:
            self.real_429 += 1
            self.sch.note_rate_limit(u.profile)

        # Lineage is anchored to the ORIGINAL trial, so a failing retry cannot
        # start a fresh retry budget under its own id.
        origin = (u.original_trial_id if isinstance(u, RepairUnit) and u.is_retry
                  else trial.trial_dir)
        ok, why = self.ledger.may_retry(origin, cls)
        if not ok:
            self.retry_stats["permanent"] += 1
            return
        if self.draining:
            self.retry_stats["permanent"] += 1
            return
        # Budget is re-checked before every retry -- a retry is new spend.
        try:
            self.budget.assert_can_continue({u.profile: 1}, what="retry")
        except BudgetStop:
            self.retry_stats["permanent"] += 1
            raise

        self._retry_seq += 1
        retry_number = self.ledger.retry_count(origin) + 1
        retry_trial_id = f"{origin}--retry{retry_number}"
        task_dir = retry_dataset.task_dir_for_trial(origin)
        rec = self.ledger.open_retry(
            origin, retry_trial_id, cls,
            cell=lib.Cell(self.mode, u.profile, self.shard).key,
            original_run_dir=u.run_dir)
        self.retry_stats["attempted"] += 1
        # Queued as a RepairUnit -- a distinct type, carrying the exact cell.
        # The failed original stays on disk untouched; lineage is the ledger's.
        self.units[u.profile].append(RepairUnit(
            profile=u.profile, index=1000 + self._retry_seq, expected_trials=1,
            cells=[task_dir], label=retry_trial_id,
            original_trial_id=origin, retry_trial_id=retry_trial_id,
            retry_number=rec.retry_number, failure_class=cls.failure_class,
            failure_reason=cls.reason, original_run_dir=u.run_dir))

    # -- budget during flight ------------------------------------------------
    def _poll_budget(self) -> None:
        """Poll on a timer while work is in flight, and act on the result.

        Three distinct outcomes, deliberately different:
          * HARD EXHAUSTION -> terminal, immediately. Continuing produces
            budget-censored garbage, which is worse than stopping.
          * TIGHT BUDGET -> drain. Stop launching, let healthy trajectories
            finish, then stop. Killing a 90-minute trajectory at minute 85 to
            save nothing is pure loss.
          * PROBE BROKEN -> fail closed on LAUNCHING (we cannot prove we can
            afford it), but again do not kill healthy work. If the probe stays
            broken past the grace period, stop.
        """
        if not self.budget.enabled or not self.budget.due():
            return
        v = self.budget.poll()

        if not v.ok:
            broken_for = self.clock() - (self.budget.first_failure_at or self.clock())
            if broken_for >= BUDGET_PROBE_GRACE_SEC:
                raise BudgetStop(
                    f"gateway budget probe has been failing for {broken_for/60:.0f} "
                    f"minutes ({v.error}); stopping rather than spending blind")
            self._enter_drain(f"budget probe failing ({v.error}) - "
                              "not launching new work until it recovers")
            return

        if v.remaining <= self.budget.reserve:
            raise BudgetStop(
                f"gateway budget exhausted mid-flight: remaining ${v.remaining:,.2f} "
                f"<= reserve ${self.budget.reserve:,.2f}")

        inflight_cost = self.budget.estimated_cost(
            {p: b.expected_trials for p, b in self.inflight.items()})
        if v.remaining < inflight_cost + self.budget.reserve:
            self._enter_drain(
                f"remaining ${v.remaining:,.2f} no longer covers in-flight work "
                f"(est. ${inflight_cost:,.2f}) plus ${self.budget.reserve:,.2f} "
                "reserve - draining in-flight work, launching nothing new")
        elif self.draining and self.drain_reason.startswith("budget probe failing"):
            self.draining = False
            self.drain_reason = ""
            print("budget probe recovered; resuming launches", flush=True)

    def _enter_drain(self, reason: str) -> None:
        if not self.draining:
            self.draining = True
            self.drain_reason = reason
            print(f"\nDRAINING: {reason}", file=sys.stderr, flush=True)

    # -- main loop -----------------------------------------------------------
    def run(self) -> ControllerResult:
        try:
            return self._run()
        except BudgetStop as exc:
            return self._stop(str(exc))
        except Exception as exc:                        # noqa: BLE001 - see below
            # An uncaught exception used to mean NO result file, so campaign.sh
            # recorded `attempt_no_output` and every run directory the shard had
            # already produced went unrecorded. A crash is still a FAILED
            # attempt, and it must name its outputs.
            traceback.print_exc()
            res = self._stop(f"controller crashed: {type(exc).__name__}: {exc}")
            res.crashed = True
            return res

    def _run(self) -> ControllerResult:
        self.plan()
        if not any(self.units.get(p) for p in self.profiles):
            return self._result(True, "nothing to do: no work units planned")
        try:
            self.budget.assert_can_continue(self._remaining_trials(),
                                            what="start this shard")
        except BudgetStop as exc:
            return self._stop(str(exc))

        while True:
            for p, u in list(self.inflight.items()):
                finished = u.proc is None or u.proc.poll() is not None
                if finished:
                    u.rc = 0 if u.proc is None else u.proc.returncode
                    del self.inflight[p]
                    self.reap(u)

            # Budget is polled on its own timer from the MAIN LOOP, so it is
            # checked while long batches are in flight, not only at launches.
            self._poll_budget()

            pending = {p: [u for u in self.units[p]
                           if not u.done and u is not self.inflight.get(p)]
                       for p in self.profiles}
            for p in self.profiles:
                st = self.sch.states[p]
                st.queued = sum(u.expected_trials for u in pending[p])
                st.active = self.inflight[p].concurrency if p in self.inflight else 0
            if not self.inflight and (self.draining or not any(pending.values())):
                break

            self.sch.note_healthy_window()
            decision = self.sch.allocate()
            if not self.draining:
                for p in self.profiles:
                    if p in self.inflight or not pending[p]:
                        continue
                    n = decision.allocation.get(p, 0)
                    if n <= 0:
                        continue
                    nxt = pending[p][0]
                    # A repair unit runs exactly its own cells; never give it
                    # more workers than it has trials.
                    n = min(n, nxt.expected_trials)
                    try:
                        self.budget.assert_can_continue({p: nxt.expected_trials},
                                                        what=f"launch {p} {nxt.kind}")
                    except BudgetStop as exc:
                        if self.budget.exhausted():
                            return self._stop(str(exc))
                        self._enter_drain(str(exc))
                        break
                    self.launch(nxt, n)
                    self.inflight[p] = nxt
            self.emit_status(decision)
            if self.dry_run:
                for p, u in list(self.inflight.items()):
                    u.rc = 0
                    self.completed_trials[p] += u.expected_trials
                    del self.inflight[p]
                continue
            self.sleep(5)

        self.emit_status(force=True)
        if self.draining:
            return self._result(False, f"stopped after draining: {self.drain_reason}")
        return self._result(True, "")

    def _result(self, ok: bool, reason: str) -> ControllerResult:
        peak, _ = self.timeline.peak_tpm()
        v = self.budget.view
        return ControllerResult(
            ok, reason,
            {p: [u.run_dir for u in self.units.get(p, []) if u.run_dir]
             for p in self.profiles},
            dict(self.retry_stats), self.clock() - self.started,
            budget_polls=self.budget.poll_count,
            spend_usd=round(v.spend, 2) if (v.ok and v.checked_at) else None,
            remaining_budget_usd=(round(v.remaining, 2) if (v.ok and v.checked_at)
                                  else None),
            retrospective_peak_tpm=round(peak, 1),
            unattributed_metered_tokens=self.timeline.unattributed_metered)

    def _stop(self, reason: str) -> ControllerResult:
        for p, u in list(self.inflight.items()):
            if u.proc is not None:
                try:
                    u.proc.terminate()
                except Exception:
                    pass
        # Any retry still open at a hard stop is abandoned, not left pending.
        for p, u in list(self.inflight.items()):
            if isinstance(u, RepairUnit) and u.is_retry:
                self._close_retry(u, "abandoned", reason=reason)
        self.emit_status(force=True)
        print(f"\nSTOPPING CLEANLY: {reason}", file=sys.stderr, flush=True)
        return self._result(False, reason)


def load_repair_plan(path: Path) -> dict:
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    if plan.get("campaign_id") != lib.CAMPAIGN_ID:
        raise SystemExit(f"repair plan is for campaign {plan.get('campaign_id')!r}, "
                         f"not {lib.CAMPAIGN_ID!r}")
    return plan


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=lib.MODES)
    ap.add_argument("--shard", type=int, choices=lib.SHARD_INDICES)
    ap.add_argument("--dry-run", action="store_true",
                    help="plan and log only; never launches Harbor")
    ap.add_argument("--limits", action="store_true", help="print limits and exit")
    ap.add_argument("--repair-plan", type=Path,
                    help="run ONLY the cells in this repair plan (from campaign.cells)")
    ap.add_argument("--result-file", help="write the result JSON here (stdout stays live status)")
    args = ap.parse_args()
    if args.limits:
        print(sched.limits_table())
        return 0
    if not args.mode or not args.shard:
        ap.error("--mode and --shard are required unless --limits is given")

    plan = load_repair_plan(args.repair_plan) if args.repair_plan else None
    c = Controller(args.mode, args.shard, dry_run=args.dry_run, repair_plan=plan)
    try:
        res = c.run()
    except BaseException as exc:            # last-resort: still leave a record
        res = ControllerResult(False, f"controller aborted: {type(exc).__name__}: {exc}",
                               crashed=True)
        if args.result_file:
            Path(args.result_file).write_text(json.dumps(res.as_dict(), indent=2) + "\n")
        raise
    blob = json.dumps(res.as_dict(), indent=2)
    if args.result_file:
        Path(args.result_file).write_text(blob + "\n")
    else:
        print(blob)
    return 0 if res.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
