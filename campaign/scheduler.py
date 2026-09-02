#!/usr/bin/env python3
"""Bounded work-conserving global scheduler for Campaign V2.

Design follows audits/claude_opus5/campaign_v2/scheduler_audit.md.

The audit measured the Aug 2026 corpus and found:

  * peak METERED aggregate throughput was 467,693 TPM = 9.4% of the 5,000,000
    key ceiling, and there were ZERO real 429s in the entire historical record;
  * 47.0% of active minutes had only ONE profile running (39.3% had all four);
  * llama was pinned at concurrency 4 whenever the three "fixed" profiles were
    alive, and its batches ran sequentially.

So the historical loss was serialisation and static ceilings, NOT failure to
redistribute a token budget that sat ~90% idle. This scheduler therefore:

  * treats TPM as a SAFETY CEILING that can only throttle DOWN. Observing low
    TPM never causes growth -- growth is bounded solely by per-profile caps and
    demonstrated-healthy execution. A controller that ramped toward 4.6M would
    need ~10x historical concurrency and would run away into untested territory
    (Modal capacity, local process limits, upstream per-model limits);
  * is work-conserving across profiles: a profile with no queued work holds no
    capacity, and freed capacity is offered immediately to whoever still has a
    queue;
  * gives llama a real floor so it can never be starved again.

Concurrency cannot be retuned on an in-flight Harbor job (`harbor run -n` is a
launch-time flag), so the allocator is consulted at BATCH BOUNDARIES: each
decision applies to the next Harbor invocation for that profile.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict

from campaign import lib

# --------------------------------------------------------------------------- #
# limits -- every number here is derived from the historical corpus
# --------------------------------------------------------------------------- #
# floor  : the concurrency a profile is guaranteed while it still has queued
#          work. Chosen so no profile can be starved the way llama was.
# cap    : the most we will ever ask for. Set at or modestly above the
#          historically DEMONSTRATED-SAFE concurrency -- never a speculative
#          extrapolation, because nothing above these values has ever been run.
# tpm    : median METERED tokens/minute per worker (fresh input + cache
#          creation + output). Cache reads are excluded; see the audit.
PROFILE_LIMITS: dict[str, dict] = {
    # claude ran at 20 for hours with no 429 and no instability -> cap 24 is a
    # ~20% extension over demonstrated-safe. Floor 4 keeps it alive cheaply.
    "claude": {"floor": 4, "cap": 24, "tpm_per_worker": 6893, "sec_per_trial": 838},
    # fable ran at 16 clean -> cap 20. It is the most expensive profile per
    # trial ($3.65 mean), so it is also the first to be trimmed under pressure.
    "fable":  {"floor": 4, "cap": 20, "tpm_per_worker": 6481, "sec_per_trial": 810},
    # codex ran at 16 clean -> cap 20. Cheapest of the three paid profiles.
    "codex":  {"floor": 4, "cap": 20, "tpm_per_worker": 7291, "sec_per_trial": 703},
    # llama: historically pinned to 4 (median) but demonstrated up to 40 when it
    # ran alone. It is ~$0 and only ~1,881 metered TPM/worker, so it is nearly
    # free to run wide. Floor 8 is double the value that starved it; cap 40 is
    # exactly the demonstrated maximum, not an extrapolation.
    "llama":  {"floor": 8, "cap": 40, "tpm_per_worker": 1881, "sec_per_trial": 453},
}

# Historical peak concurrent workers was 20+16+16+4 = 56. 72 is a bounded
# (~1.3x) extension; the sum of per-profile caps (104) is deliberately NOT
# reachable, so the global ceiling stays the binding structural limit.
GLOBAL_WORKER_CEILING = int(os.environ.get("CAMPAIGN_GLOBAL_WORKERS", "72"))

HARD_TPM_CEILING = int(os.environ.get("CAMPAIGN_MAX_TPM", "5000000"))
SOFT_TPM_CEILING = int(os.environ.get("CAMPAIGN_TARGET_TPM", "4600000"))

# AIMD-ish response to real pressure. Growth is NOT part of this: capacity only
# ever returns to the cap after a cooldown with no adverse events.
BACKOFF_FACTOR = 0.5          # multiplicative decrease on real 429 / TPM breach
RECOVERY_STEP = 1             # additive increase per healthy adjustment window
ADJUST_COOLDOWN_SEC = int(os.environ.get("CAMPAIGN_ADJUST_COOLDOWN", "60"))
TPM_WINDOW_SEC = int(os.environ.get("CAMPAIGN_TPM_WINDOW", "300"))


# --------------------------------------------------------------------------- #
# rolling metered-TPM meter
# --------------------------------------------------------------------------- #
class TpmMeter:
    """Rolling window of METERED token consumption.

    Metered tokens = fresh input + cache creation + output. Cache READS are
    excluded: `total_cached_tokens` is exactly `total_cache_read_input_tokens`
    and counting it produced the nonsensical 227%-of-ceiling figure in the
    audit.

    RETROSPECTIVE. Every event fed here is reconstructed from a trajectory,
    which exists only once its trial has ended. `record` must therefore be
    given the timestamp the tokens were ACTUALLY spent (`campaign.tokens`
    extracts per-step timestamps for exactly this) and never the moment a batch
    happened to be reaped -- collapsing a 90-minute batch onto its exit instant
    is what produced the 12.3M-TPM phantom on 2026-09-02.
    """

    def __init__(self, window_sec: int = TPM_WINDOW_SEC):
        self.window_sec = window_sec
        self._events: list[tuple[float, str, int]] = []   # (ts, profile, tokens)

    @staticmethod
    def metered_tokens(final_metrics: dict) -> int:
        fm = final_metrics or {}
        prompt = fm.get("total_prompt_tokens") or 0
        completion = fm.get("total_completion_tokens") or 0
        cache_read = fm.get("total_cached_tokens") or 0
        return max(0, (prompt - cache_read) + completion)

    def record(self, profile: str, tokens: int, ts: float | None = None) -> None:
        self._events.append((time.time() if ts is None else ts, profile, int(tokens)))

    def _prune(self, now: float) -> None:
        cut = now - self.window_sec
        self._events = [e for e in self._events if e[0] >= cut]

    def tpm(self, now: float | None = None) -> float:
        now = time.time() if now is None else now
        self._prune(now)
        if not self._events:
            return 0.0
        return sum(e[2] for e in self._events) / (self.window_sec / 60.0)

    def tpm_by_profile(self, now: float | None = None) -> dict[str, float]:
        now = time.time() if now is None else now
        self._prune(now)
        out: dict[str, float] = {}
        for _, prof, tok in self._events:
            out[prof] = out.get(prof, 0.0) + tok
        return {k: v / (self.window_sec / 60.0) for k, v in out.items()}


# --------------------------------------------------------------------------- #
# scheduler state
# --------------------------------------------------------------------------- #
@dataclass
class ProfileState:
    profile: str
    queued: int = 0
    active: int = 0
    completed: int = 0
    failed: int = 0
    stalled: bool = False          # blocked for a non-retryable reason
    # headroom starts at the cap and is only reduced by observed pressure
    headroom: int = 0

    def __post_init__(self):
        if self.headroom == 0:
            self.headroom = PROFILE_LIMITS[self.profile]["cap"]

    @property
    def floor(self) -> int:
        return PROFILE_LIMITS[self.profile]["floor"]

    @property
    def cap(self) -> int:
        return PROFILE_LIMITS[self.profile]["cap"]

    @property
    def wants_work(self) -> bool:
        return self.queued > 0 and not self.stalled


@dataclass
class Decision:
    """One allocation decision, fully explainable for the controller log."""
    allocation: dict[str, int] = field(default_factory=dict)
    reason: str = ""
    projected_tpm: float = 0.0
    observed_tpm: float = 0.0
    throttled: bool = False
    total_workers: int = 0
    # True when `observed_tpm` came from trajectories that are only written when
    # a trial ENDS, i.e. it lags reality and must not gate live decisions.
    observed_is_retrospective: bool = True

    def as_dict(self):
        return asdict(self)


class Scheduler:
    """Work-conserving allocator with a TPM safety ceiling.

    Invariants the tests pin down:
      * never allocates a profile more than its cap, nor above the global ceiling
      * never allocates a profile with no queued work
      * a queued, non-stalled profile always gets at least min(floor, queued)
      * low observed TPM NEVER increases allocation
      * a real 429 or a TPM-ceiling breach reduces headroom multiplicatively
    """

    def __init__(self, profiles=lib.PROFILES, *, global_ceiling: int = GLOBAL_WORKER_CEILING,
                 soft_tpm: int = SOFT_TPM_CEILING, hard_tpm: int = HARD_TPM_CEILING,
                 live_tpm_signal: bool = False):
        self.states = {p: ProfileState(p) for p in profiles}
        self.global_ceiling = global_ceiling
        self.soft_tpm = soft_tpm
        self.hard_tpm = hard_tpm
        self.meter = TpmMeter()
        self.recent_429 = 0
        self.last_adjust = 0.0
        # Set True ONLY if a genuinely live, per-request metering source is
        # wired in. Token events reconstructed from trajectories are not that:
        # a trajectory is written when its trial ends, so the recent end of the
        # series is structurally empty while work is in flight. Throttling on
        # it is how the 2026-09-02 shard cut llama from 21 workers to 15 on a
        # reading of 12,298,656 TPM when real traffic that minute was 31,521.
        self.live_tpm_signal = bool(live_tpm_signal)

    # -- pressure signals ---------------------------------------------------
    def note_rate_limit(self, profile: str, *, now: float | None = None) -> None:
        """Record a CORROBORATED 429. Callers must not pass classifier guesses."""
        self.recent_429 += 1
        st = self.states[profile]
        st.headroom = max(st.floor, int(st.headroom * BACKOFF_FACTOR))
        self.last_adjust = time.time() if now is None else now

    def note_healthy_window(self, *, now: float | None = None) -> None:
        """Additively restore headroom after a clean cooldown period."""
        now = time.time() if now is None else now
        if now - self.last_adjust < ADJUST_COOLDOWN_SEC:
            return
        for st in self.states.values():
            if st.headroom < st.cap:
                st.headroom = min(st.cap, st.headroom + RECOVERY_STEP)
        self.recent_429 = 0
        self.last_adjust = now

    def projected_tpm(self, allocation: dict[str, int]) -> float:
        return float(sum(PROFILE_LIMITS[p]["tpm_per_worker"] * n
                         for p, n in allocation.items()))

    # -- the allocator ------------------------------------------------------
    def allocate(self, *, now: float | None = None) -> Decision:
        now = time.time() if now is None else now
        observed = self.meter.tpm(now)

        alloc = {p: 0 for p in self.states}

        # 0. IN-FLIGHT CAPACITY IS PINNED. `harbor run -n` cannot be retuned on a
        #    live job, so a profile with a batch in flight holds exactly the
        #    workers it launched with. Those workers still count against the
        #    global ceiling -- otherwise in-flight + newly launched can breach it.
        pinned = 0
        for st_ in self.states.values():
            if st_.active > 0:
                alloc[st_.profile] = st_.active
                pinned += st_.active

        wanting = [s for s in self.states.values() if s.wants_work and s.active == 0]
        if not wanting:
            reason = ("all queued profiles already have a batch in flight"
                      if pinned else "no profile has queued work")
            return Decision(alloc, reason, self.projected_tpm(alloc), observed, False, pinned)

        # 1. FLOORS. Every queued profile is guaranteed its floor. This is the
        #    anti-starvation guarantee; llama gets 8 even when all four are hot.
        for s in wanting:
            room = max(0, self.global_ceiling - sum(alloc.values()))
            alloc[s.profile] = min(s.floor, s.queued, s.headroom, room)

        # 2. WORK-CONSERVING FILL. Hand out what is left of the global ceiling.
        #    Order by metered cost per worker ascending, so the cheapest useful
        #    parallelism is bought first; llama (1,881 TPM/worker, $0) therefore
        #    expands before the expensive profiles rather than after them.
        #    Nothing here consults observed TPM: low utilisation must not drive
        #    growth. The bound is cap/headroom/queue, full stop.
        order = sorted(wanting, key=lambda s: PROFILE_LIMITS[s.profile]["tpm_per_worker"])
        changed = True
        while changed:
            changed = False
            for s in order:
                used = sum(alloc.values())
                if used >= self.global_ceiling:
                    break
                limit = min(s.cap, s.headroom, s.queued)
                if alloc[s.profile] >= limit:
                    continue
                trial = dict(alloc)
                trial[s.profile] += 1
                # 3. TPM SAFETY CEILING -- only ever blocks, never encourages.
                if self.projected_tpm(trial) > self.soft_tpm:
                    continue
                alloc = trial
                changed = True

        throttled = False
        reason = "work-conserving fill to caps"
        # 4. Reactive throttle on genuinely observed pressure. Gated on
        #    `live_tpm_signal`: a retrospective series may only be REPORTED.
        if self.live_tpm_signal and observed > self.soft_tpm:
            throttled = True
            reason = f"observed metered TPM {observed:,.0f} over soft ceiling {self.soft_tpm:,} - scaling down"
            for p in list(alloc):
                if self.states[p].active > 0:
                    continue          # cannot be retuned in flight; it drains instead
                if alloc[p] > 0:
                    alloc[p] = max(self.states[p].floor if self.states[p].wants_work else 0,
                                   int(alloc[p] * BACKOFF_FACTOR))
        elif self.recent_429:
            throttled = True
            reason = f"{self.recent_429} corroborated 429(s) in window - holding reduced headroom"

        if len(wanting) == 1:
            reason += f" (single profile {wanting[0].profile} may expand to its cap)"

        return Decision(alloc, reason, self.projected_tpm(alloc), observed,
                        throttled, sum(alloc.values()),
                        observed_is_retrospective=not self.live_tpm_signal)


def limits_table() -> str:
    rows = ["profile   floor  cap   metered TPM/worker   projected TPM at cap"]
    for p, v in PROFILE_LIMITS.items():
        rows.append(f"{p:8}  {v['floor']:>4}  {v['cap']:>3}   {v['tpm_per_worker']:>17,}   "
                    f"{v['tpm_per_worker']*v['cap']:>19,}")
    tot = sum(v["tpm_per_worker"] * v["cap"] for v in PROFILE_LIMITS.values())
    rows.append(f"{'TOTAL':8}  {'':>4}  {sum(v['cap'] for v in PROFILE_LIMITS.values()):>3}"
                f"   {'':>17}   {tot:>19,}")
    rows.append(f"global worker ceiling: {GLOBAL_WORKER_CEILING}   "
                f"soft TPM ceiling: {SOFT_TPM_CEILING:,}   hard: {HARD_TPM_CEILING:,}")
    return "\n".join(rows)


if __name__ == "__main__":
    print(limits_table())
    print()
    s = Scheduler()
    for p in lib.PROFILES:
        s.states[p].queued = 300
    print(json.dumps(s.allocate().as_dict(), indent=2))
