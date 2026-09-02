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
periodically during it, and before every retry. The Aug-27 corpus shows what
happens otherwise -- 300 trial directories and 1,065 budget-marker files after
the key hit $10,001.94 of a $10,000 ceiling, all of it worthless.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from campaign import lib, failures, scheduler as sched
from campaign.scheduler import Scheduler, PROFILE_LIMITS

BATCH_BASE_TASKS = int(os.environ.get("CAMPAIGN_BATCH_BASE_TASKS", "10"))
STATUS_INTERVAL_SEC = int(os.environ.get("CAMPAIGN_STATUS_INTERVAL", "45"))
BUDGET_POLL_SEC = int(os.environ.get("CAMPAIGN_BUDGET_POLL", "300"))
# Stop launching if remaining budget would not cover the work still in flight
# plus this reserve. Sized above the worst single observed trial ($24.90).
BUDGET_RESERVE_USD = float(os.environ.get("CAMPAIGN_BUDGET_RESERVE", "150"))


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
                 reserve: float = BUDGET_RESERVE_USD):
        self.cost_per_trial = cost_per_trial
        self.enabled = enabled
        self.reserve = reserve
        self.view = BudgetView()
        self._last_poll = 0.0

    def poll(self, *, force: bool = False, now: float | None = None) -> BudgetView:
        now = time.time() if now is None else now
        if not self.enabled:
            self.view = BudgetView(ok=True, error="budget guard disabled")
            return self.view
        if not force and (now - self._last_poll) < BUDGET_POLL_SEC:
            return self.view
        self._last_poll = now
        st = lib.probe_budget()
        if not st.ok:
            self.view = BudgetView(ok=False, error=st.error or "probe failed", checked_at=now)
            return self.view
        self.view = BudgetView(spend=st.spend or 0.0, max_budget=st.max_budget or 0.0,
                               remaining=st.remaining or 0.0, checked_at=now, ok=True)
        return self.view

    def estimated_cost(self, remaining_trials: dict[str, int]) -> float:
        return sum(self.cost_per_trial.get(p, 0.0) * n for p, n in remaining_trials.items())

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
# work units
# --------------------------------------------------------------------------- #
@dataclass
class Batch:
    profile: str
    index: int
    start_base: int
    base_count: int
    expected_trials: int
    dataset: Path | None = None
    concurrency: int = 0
    run_dir: str = ""
    rc: int | None = None
    started_at: float = 0.0
    finished_at: float = 0.0
    proc: object = None
    outfile: str = ""

    @property
    def done(self) -> bool:
        return self.rc is not None


@dataclass
class ControllerResult:
    ok: bool
    stopped_reason: str = ""
    run_dirs: dict[str, list[str]] = field(default_factory=dict)
    retries: dict[str, int] = field(default_factory=dict)
    elapsed_sec: float = 0.0

    def as_dict(self):
        return asdict(self)


class Controller:
    def __init__(self, mode: str, shard: int, *, paths=None, dry_run: bool = False,
                 budget: BudgetGuard | None = None, profiles=lib.PROFILES,
                 clock=time.time):
        self.mode = mode
        self.shard = shard
        self.paths = paths or lib.campaign_paths()
        self.dry_run = dry_run
        self.profiles = list(profiles)
        self.clock = clock
        self.sch = Scheduler(self.profiles)
        self.budget = budget or BudgetGuard(self._cost_model(), enabled=not dry_run)
        self.batches: dict[str, list[Batch]] = {}
        self.inflight: dict[str, Batch] = {}
        self.completed_trials: dict[str, int] = {p: 0 for p in self.profiles}
        self.failed_trials: dict[str, int] = {p: 0 for p in self.profiles}
        self.retry_stats = {"attempted": 0, "recovered": 0, "permanent": 0}
        self.real_429 = 0
        self.started = clock()
        self.ledger = failures.RetryLedger(self.paths["provenance"] / "retries.jsonl")
        self.log_path = self.paths["logs"] / f"controller-{mode}-shard{shard}.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_status = 0.0

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
        cell0 = lib.Cell(self.mode, self.profiles[0], self.shard)
        base_total = cell0.base_tasks
        variants = lib.VARIANTS_PER_TASK[self.mode]
        for p in self.profiles:
            out, idx, start = [], 0, 0
            while start < base_total:
                n = min(BATCH_BASE_TASKS, base_total - start)
                idx += 1
                out.append(Batch(profile=p, index=idx, start_base=start,
                                 base_count=n, expected_trials=n * variants))
                start += n
            self.batches[p] = out
            self.sch.states[p].queued = sum(b.expected_trials for b in out)

    # -- status -------------------------------------------------------------
    def _remaining_trials(self) -> dict[str, int]:
        return {p: sum(b.expected_trials for b in self.batches.get(p, []) if not b.done)
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
        return {
            "ts": lib.now_iso(),
            "campaign_id": lib.CAMPAIGN_ID,
            "mode": self.mode,
            "shard": self.shard,
            "elapsed_sec": round(now - self.started, 1),
            "active_workers": {p: (self.inflight[p].concurrency if p in self.inflight else 0)
                               for p in self.profiles},
            "queued_trials": self._remaining_trials(),
            "completed_trials": dict(self.completed_trials),
            "failed_trials": dict(self.failed_trials),
            "aggregate_metered_tpm": round(self.sch.meter.tpm(now), 1),
            "tpm_by_profile": {k: round(x, 1) for k, x in self.sch.meter.tpm_by_profile(now).items()},
            "tpm_soft_ceiling": self.sch.soft_tpm,
            "tpm_hard_ceiling": self.sch.hard_tpm,
            "tpm_utilisation_pct": round(100.0 * self.sch.meter.tpm(now) / self.sch.hard_tpm, 3),
            "real_429_total": self.real_429,
            "retries": dict(self.retry_stats),
            "budget_checked": bool(v.ok and v.checked_at),
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
        bud = (f"spend ${snap['spend_usd']:,.0f} rem ${snap['remaining_budget_usd']:,.0f}"
               if snap["budget_checked"] else "budget unchecked")
        print(
            f"[{lib.CAMPAIGN_ID} {self.mode} shard {self.shard}] "
            f"t+{snap['elapsed_sec']/60:.0f}m  "
            f"TPM {snap['aggregate_metered_tpm']:,.0f}/{self.sch.hard_tpm:,} "
            f"({snap['tpm_utilisation_pct']:.2f}%)  429s {self.real_429}  "
            f"retries {self.retry_stats['attempted']}a/{self.retry_stats['recovered']}r/"
            f"{self.retry_stats['permanent']}p  "
            f"{bud}  {rate_s}  ETA {eta}",
            flush=True)
        print("    " + "  ".join(
            f"{p}: {aw[p]}w {c[p]}done/{q[p]}queued" for p in self.profiles), flush=True)

    # -- execution ----------------------------------------------------------
    def _slice_dataset(self, b: Batch) -> Path:
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

    def launch(self, b: Batch, concurrency: int) -> None:
        b.concurrency = concurrency
        b.started_at = self.clock()
        b.dataset = self._slice_dataset(b)
        if self.dry_run:
            b.proc = None
            return
        import tempfile
        fd, b.outfile = tempfile.mkstemp(prefix="swe-run-")
        os.close(fd)
        env = dict(os.environ)
        env["HARBOR_CONCURRENCY"] = str(concurrency)
        env["SWE_RUN_OUTPUT_FILE"] = b.outfile
        env["RESULTS_ROOT"] = str(self.paths["root"])
        label = lib.Cell(self.mode, b.profile, self.shard).shard_label
        b.proc = subprocess.Popen(
            [str(lib.PROJECT_ROOT / "scripts" / "05_run_profile.sh"), self.mode, b.profile,
             "--dataset-override", str(b.dataset),
             "--shard-label", f"-{label}-b{b.index:02d}"],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    def reap(self, b: Batch) -> None:
        """Collect a finished batch: tokens into the meter, failures classified."""
        b.finished_at = self.clock()
        if b.outfile:
            try:
                b.run_dir = Path(b.outfile).read_text().strip()
                os.unlink(b.outfile)
            except OSError:
                pass
        if not b.run_dir or not Path(b.run_dir).is_dir():
            self.failed_trials[b.profile] += b.expected_trials
            return
        for t in lib.scan_run_dir(Path(b.run_dir)):
            # metered = (prompt - cache_reads) + completion. Counting cache
            # reads here would inflate observed TPM and throttle us for tokens
            # the gateway never meters.
            tokens = sched.TpmMeter.metered_tokens({
                "total_prompt_tokens": t.prompt_tokens,
                "total_completion_tokens": t.completion_tokens,
                "total_cached_tokens": t.cached_tokens,
            })
            self.sch.meter.record(b.profile, tokens, b.finished_at)
            if t.status == lib.STATUS_COMPLETE:
                self.completed_trials[b.profile] += 1
            else:
                self.failed_trials[b.profile] += 1
                self._handle_failure(b, t)
        self.sch.states[b.profile].queued = self._remaining_trials()[b.profile]

    def _handle_failure(self, b: Batch, trial) -> None:
        td = Path(b.run_dir)
        cand = list(td.rglob(trial.trial_dir))
        cls = failures.classify_trial_dir(cand[0]) if cand else failures.Classification(
            failures.PERMANENT, "trial directory not found", False, False)
        if cls.failure_class == failures.BUDGET:
            raise BudgetStop(
                f"budget failure observed in {trial.trial_dir}: {cls.reason}. "
                "Stopping the shard immediately; budget failures are NEVER retried.")
        if cls.failure_class == failures.TRANSIENT_PROVIDER and "429" in cls.evidence:
            self.real_429 += 1
            self.sch.note_rate_limit(b.profile)
        ok, why = self.ledger.may_retry(trial.trial_dir, cls)
        if not ok:
            self.retry_stats["permanent"] += 1
            return
        # Budget is re-checked before every retry -- a retry is new spend.
        try:
            self.budget.assert_can_continue({b.profile: 1}, what="retry")
        except BudgetStop:
            self.retry_stats["permanent"] += 1
            raise
        rec = self.ledger.open_retry(
            trial.trial_dir, f"{trial.trial_dir}--retry{self.ledger.retry_count(trial.trial_dir)+1}",
            cls, cell=lib.Cell(self.mode, b.profile, self.shard).key)
        self.retry_stats["attempted"] += 1
        # Re-queue as its own batch so the failed original is preserved untouched.
        self.batches[b.profile].append(Batch(
            profile=b.profile, index=1000 + rec.retry_number,
            start_base=b.start_base, base_count=0, expected_trials=1))

    def run(self) -> ControllerResult:
        self.plan_batches()
        try:
            self.budget.assert_can_continue(self._remaining_trials(), what="start this shard")
        except BudgetStop as exc:
            return ControllerResult(False, str(exc))

        while True:
            for p, b in list(self.inflight.items()):
                finished = b.proc is None or b.proc.poll() is not None
                if finished:
                    b.rc = 0 if b.proc is None else b.proc.returncode
                    del self.inflight[p]
                    try:
                        self.reap(b)
                    except BudgetStop as exc:
                        return self._stop(str(exc))

            pending = {p: [b for b in self.batches[p] if not b.done and b is not self.inflight.get(p)]
                       for p in self.profiles}
            for p in self.profiles:
                st = self.sch.states[p]
                st.queued = sum(b.expected_trials for b in pending[p])
                st.active = self.inflight[p].concurrency if p in self.inflight else 0
            if not any(pending.values()) and not self.inflight:
                break

            self.sch.note_healthy_window()
            decision = self.sch.allocate()
            for p in self.profiles:
                if p in self.inflight or not pending[p]:
                    continue
                n = decision.allocation.get(p, 0)
                if n <= 0:
                    continue
                try:
                    self.budget.assert_can_continue({p: pending[p][0].expected_trials},
                                                    what=f"launch {p} batch")
                except BudgetStop as exc:
                    return self._stop(str(exc))
                nxt = pending[p][0]
                self.launch(nxt, n)
                self.inflight[p] = nxt
            self.emit_status(decision)
            if self.dry_run:
                for p, b in list(self.inflight.items()):
                    b.rc = 0
                    self.completed_trials[p] += b.expected_trials
                    del self.inflight[p]
                continue
            time.sleep(5)

        self.emit_status(force=True)
        return ControllerResult(
            True, "", {p: [b.run_dir for b in self.batches[p] if b.run_dir] for p in self.profiles},
            dict(self.retry_stats), self.clock() - self.started)

    def _stop(self, reason: str) -> ControllerResult:
        for p, b in list(self.inflight.items()):
            if b.proc is not None:
                try:
                    b.proc.terminate()
                except Exception:
                    pass
        self.emit_status(force=True)
        print(f"\nSTOPPING CLEANLY: {reason}", file=sys.stderr, flush=True)
        return ControllerResult(False, reason,
                                {p: [b.run_dir for b in self.batches[p] if b.run_dir]
                                 for p in self.profiles},
                                dict(self.retry_stats), self.clock() - self.started)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=lib.MODES)
    ap.add_argument("--shard", type=int, choices=lib.SHARD_INDICES)
    ap.add_argument("--dry-run", action="store_true",
                    help="plan and log only; never launches Harbor")
    ap.add_argument("--limits", action="store_true", help="print limits and exit")
    ap.add_argument("--result-file", help="write the result JSON here (stdout stays live status)")
    args = ap.parse_args()
    if args.limits:
        print(sched.limits_table())
        return 0
    if not args.mode or not args.shard:
        ap.error("--mode and --shard are required unless --limits is given")
    c = Controller(args.mode, args.shard, dry_run=args.dry_run)
    res = c.run()
    blob = json.dumps(res.as_dict(), indent=2)
    if args.result_file:
        Path(args.result_file).write_text(blob + "\n")
    else:
        print(blob)
    return 0 if res.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
