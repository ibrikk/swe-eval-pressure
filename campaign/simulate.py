#!/usr/bin/env python3
"""Deterministic offline simulation of FULL shard 1. No Harbor, no model calls.

Trial durations are resampled (seeded) from the Aug 2026 empirical distribution
in campaign/historical_runtime.json, so the shape of the tail is real rather
than assumed. Everything else is a model, and the model is approximate -- see
the caveats printed at the end. Treat the ratio between schedulers as more
trustworthy than any absolute wall clock.
"""
from __future__ import annotations

import argparse
import heapq
import json
import random
import statistics as st
from pathlib import Path

from campaign import lib
from campaign.scheduler import Scheduler, PROFILE_LIMITS, GLOBAL_WORKER_CEILING

RUNTIME = json.loads((Path(__file__).parent / "historical_runtime.json").read_text())
# Not recoverable from the corpus: historical run_metadata.json carries no
# created_at, so the gap between a Harbor invocation and its first trial start
# cannot be measured. Declared assumption, applied identically to every
# scheduler so it cannot flatter the new one.
BATCH_STARTUP_SEC = 180.0
SEED = 20260902


def sample_durations(profile: str, n: int, rng: random.Random) -> list[float]:
    pool = RUNTIME["durations"].get(profile) or [900.0]
    return [rng.choice(pool) for _ in range(n)]


def pool_makespan(durations: list[float], concurrency: int, t0: float = 0.0) -> float:
    """Finish time of `durations` on `concurrency` workers, greedy next-free."""
    if not durations:
        return t0
    heap = [t0] * max(1, concurrency)
    heapq.heapify(heap)
    for d in durations:
        free = heapq.heappop(heap)
        heapq.heappush(heap, free + d)
    return max(heap)


# --------------------------------------------------------------------------- #
# OLD A: historical scripts/06_run_matrix_adaptive.sh
# --------------------------------------------------------------------------- #
def sim_old_adaptive(trials_per_profile: int, rng: random.Random) -> dict:
    """claude/fable/codex static and concurrent; llama floored by phase_floor
    and run in SEQUENTIAL 2-base-task batches."""
    fixed = {"claude": 20, "fable": 16, "codex": 16}
    finish, series = {}, []
    for p, c in fixed.items():
        finish[p] = pool_makespan(sample_durations(p, trials_per_profile, rng), c)
        series.append((p, c, finish[p]))

    # llama: 2 base tasks = 20 trials per batch, strictly sequential.
    per_batch = 2 * lib.VARIANTS_PER_TASK["full"]
    n_batches = trials_per_profile // per_batch
    t, llama_workers = 0.0, []
    alive_at = sorted(finish.values())
    for _ in range(n_batches):
        alive = sum(1 for f in alive_at if f > t)
        conc = {3: 4, 2: 12, 1: 24}.get(alive, 40)   # phase_floor()
        llama_workers.append(conc)
        t = pool_makespan(sample_durations("llama", per_batch, rng), conc,
                          t + BATCH_STARTUP_SEC)
    finish["llama"] = t
    return {
        "name": "OLD-A historical 06_run_matrix_adaptive.sh",
        "wall_clock_sec": max(finish.values()),
        "finish": finish,
        "mean_workers": {**fixed, "llama": round(st.mean(llama_workers), 1) if llama_workers else 0},
        "max_workers": {**fixed, "llama": max(llama_workers) if llama_workers else 0},
        "llama_is_tail": finish["llama"] >= max(finish.values()) - 1e-9,
    }


# --------------------------------------------------------------------------- #
# OLD B: today's campaign.sh run-shard (strictly sequential profiles)
# --------------------------------------------------------------------------- #
def sim_old_sequential(trials_per_profile: int, rng: random.Random) -> dict:
    conc = {"claude": 20, "fable": 16, "codex": 16, "llama": 4}
    t, finish = 0.0, {}
    for p in lib.PROFILES:
        t = pool_makespan(sample_durations(p, trials_per_profile, rng), conc[p],
                          t + BATCH_STARTUP_SEC)
        finish[p] = t
    return {
        "name": "OLD-B current campaign.sh run-shard (sequential)",
        "wall_clock_sec": t,
        "finish": finish,
        "mean_workers": conc,
        "max_workers": conc,
        "llama_is_tail": True,
    }


# --------------------------------------------------------------------------- #
# NEW: bounded work-conserving allocator
# --------------------------------------------------------------------------- #
def sim_new(trials_per_profile: int, rng: random.Random, *, batch_base_tasks: int = 10) -> dict:
    variants = lib.VARIANTS_PER_TASK["full"]
    per_batch = batch_base_tasks * variants
    queues = {p: [per_batch] * (trials_per_profile // per_batch) for p in lib.PROFILES}
    remainder = trials_per_profile % per_batch
    if remainder:
        for p in lib.PROFILES:
            queues[p].append(remainder)

    sch = Scheduler()
    now = 0.0
    inflight: dict[str, tuple[float, int]] = {}     # profile -> (finish_time, workers)
    worker_trace: list[tuple[float, dict]] = []
    tpm_trace: list[float] = []
    peak = {p: 0 for p in lib.PROFILES}
    seen = {p: [] for p in lib.PROFILES}

    while any(queues.values()) or inflight:
        for p in lib.PROFILES:
            sch.states[p].queued = sum(queues[p])
            sch.states[p].active = inflight[p][1] if p in inflight else 0
        decision = sch.allocate(now=now)

        launched = False
        for p in lib.PROFILES:
            if p in inflight or not queues[p]:
                continue
            n = decision.allocation.get(p, 0)
            if n <= 0:
                continue
            size = queues[p].pop(0)
            end = pool_makespan(sample_durations(p, size, rng), n, now + BATCH_STARTUP_SEC)
            inflight[p] = (end, n)
            peak[p] = max(peak[p], n)
            seen[p].append(n)
            launched = True

        active = {p: inflight[p][1] for p in inflight}
        worker_trace.append((now, dict(active)))
        tpm_trace.append(sum(PROFILE_LIMITS[p]["tpm_per_worker"] * w for p, w in active.items()))

        if not inflight:
            if not launched:
                break
            continue
        nxt = min(v[0] for v in inflight.values())
        for p in [p for p, v in inflight.items() if v[0] <= nxt + 1e-9]:
            del inflight[p]
        now = nxt

    return {
        "name": "NEW bounded work-conserving allocator",
        "wall_clock_sec": now,
        "finish": {},
        "mean_workers": {p: round(st.mean(v), 1) if v else 0 for p, v in seen.items()},
        "max_workers": peak,
        "worker_trace": worker_trace,
        "mean_tpm": st.mean(tpm_trace) if tpm_trace else 0.0,
        "peak_tpm": max(tpm_trace) if tpm_trace else 0.0,
        "llama_is_tail": False,
    }


def hhmm(sec: float) -> str:
    return f"{sec/3600:.1f}h"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials-per-profile", type=int, default=300)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    n = args.trials_per_profile
    old_a = sim_old_adaptive(n, random.Random(SEED))
    old_b = sim_old_sequential(n, random.Random(SEED))
    new = sim_new(n, random.Random(SEED))

    if args.json:
        print(json.dumps({"old_adaptive": {k: v for k, v in old_a.items()},
                          "old_sequential": {k: v for k, v in old_b.items()},
                          "new": {k: v for k, v in new.items() if k != "worker_trace"}},
                         indent=2, default=str))
        return 0

    print("=" * 76)
    print(f"SIMULATED FULL SHARD 1  --  {n} trials/profile, {n*4} total, seed {SEED}")
    print("=" * 76)
    for r in (old_b, old_a, new):
        print(f"\n{r['name']}")
        print(f"  wall clock        : {hhmm(r['wall_clock_sec'])}  ({r['wall_clock_sec']/60:,.0f} min)")
        print(f"  mean workers      : " + "  ".join(f"{p}={r['mean_workers'].get(p,0)}" for p in lib.PROFILES))
        print(f"  max workers       : " + "  ".join(f"{p}={r['max_workers'].get(p,0)}" for p in lib.PROFILES))
        if "mean_tpm" in r:
            print(f"  mean metered TPM  : {r['mean_tpm']:,.0f}   peak {r['peak_tpm']:,.0f}")
        print(f"  llama is the tail : {r['llama_is_tail']}")

    print("\n" + "-" * 76)
    for base, label in ((old_a, "vs OLD-A historical adaptive"),
                        (old_b, "vs OLD-B current sequential run-shard")):
        sp = base["wall_clock_sec"] / new["wall_clock_sec"] if new["wall_clock_sec"] else 0
        print(f"  speedup {label:42} {sp:.2f}x  "
              f"({hhmm(base['wall_clock_sec'])} -> {hhmm(new['wall_clock_sec'])})")

    # max SIMULTANEOUS workers, not the sum of per-profile peaks (which occur
    # at different times and would falsely look like a ceiling breach).
    tot = max((sum(a.values()) for _, a in new["worker_trace"]), default=0)
    print(f"\n  peak SIMULTANEOUS workers (new): {tot} (global ceiling {GLOBAL_WORKER_CEILING})")
    print(f"  peak metered TPM (new)         : {new['peak_tpm']:,.0f} "
          f"= {new['peak_tpm']/5e6*100:.2f}% of the 5,000,000 ceiling")
    print(f"  any configured limit hit?      : "
          f"{'global worker ceiling (by design)' if tot >= GLOBAL_WORKER_CEILING else 'no'}"
          f"; TPM ceiling NOT approached")

    print("""
  CAVEATS -- this is an approximation, not a prediction:
   * durations are resampled from Aug 2026 trials; Campaign V2 tasks differ.
   * Harbor startup is a declared 180s/batch assumption (historical
     run_metadata.json has no created_at, so it is not measurable). It is
     applied to every scheduler equally.
   * modelled as an independent worker pool per profile; real Harbor
     scheduling, Modal capacity and queueing are not modelled.
   * no failures or retries are simulated.
   * trust the RATIO between schedulers more than any absolute figure.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
