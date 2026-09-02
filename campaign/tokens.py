#!/usr/bin/env python3
"""Per-step token events with their REAL timestamps.

Why this module exists
----------------------
The controller used to meter tokens like this: when a Harbor batch finished, it
walked every trial in the finished run directory and recorded that trial's
`final_metrics` total against `batch.finished_at`. A 100-trial batch that ran
for ninety minutes therefore delivered ~90 minutes of token consumption to the
meter as a single instantaneous spike at the moment the subprocess exited.

Fed into a 300-second rolling window, that produces exactly the artifact seen on
2026-09-02: a TPM series of 4.97M -> 0 -> 12.30M -> 1.38M -> 0, where 12.3M is
2.46x the 5M hard ceiling that no real minute of gateway traffic ever came near.
Only 29 of 338 status samples were even non-zero. The controller then throttled
on it -- twice, once cutting llama from 21 workers to 15 -- because 100
trajectories happened to be stamped at the same instant.

The trajectories already carry what is needed. Every profile emits
`steps[].timestamp` alongside `steps[].metrics`, and the per-step metrics are
INCREMENTAL: they sum exactly to `final_metrics` (verified across all four
profiles). Attributing each step's tokens to its own timestamp reconstructs the
true minute-by-minute curve.

Coverage on the 2026-09-02 shard is 100%: llama leaves some steps without a
timestamp, but every such step carries zero tokens. `extract_events` still
reports what it could not attribute, so a future format change degrades loudly
instead of silently under-reporting.

Timestamp formats differ by profile -- claude/fable/codex emit '...Z', llama
emits '...+00:00' -- so both are parsed.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from campaign import lib


@dataclass(frozen=True)
class TokenEvent:
    ts: float           # epoch seconds, when the tokens were actually spent
    metered: int        # (prompt - cached) + completion
    prompt: int
    completion: int
    cached: int
    profile: str = ""
    trial_id: str = ""
    step: int = -1


@dataclass
class Extraction:
    events: list[TokenEvent]
    unattributed_metered: int = 0   # tokens with no usable timestamp
    steps_without_timestamp: int = 0

    @property
    def total_metered(self) -> int:
        return sum(e.metered for e in self.events) + self.unattributed_metered

    @property
    def coverage(self) -> float:
        """Fraction of metered tokens placed at a real timestamp."""
        tot = self.total_metered
        return 1.0 if tot == 0 else sum(e.metered for e in self.events) / tot


def parse_ts(value) -> float | None:
    """Parse a trajectory timestamp to epoch seconds. Returns None if unusable."""
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def metered_of(metrics: dict) -> tuple[int, int, int, int]:
    """(metered, prompt, completion, cached). Cache reads are not metered."""
    m = metrics or {}
    prompt = int(m.get("prompt_tokens") or 0)
    completion = int(m.get("completion_tokens") or 0)
    cached = int(m.get("cached_tokens") or 0)
    return max(0, prompt - cached) + completion, prompt, completion, cached


def extract_events(traj: dict, *, profile: str = "", trial_id: str = "") -> Extraction:
    """Per-step token events from one trajectory, at their real timestamps."""
    steps = (traj or {}).get("steps") or []
    events: list[TokenEvent] = []
    unattributed = 0
    missing_ts = 0
    for i, step in enumerate(steps):
        metered, prompt, completion, cached = metered_of(step.get("metrics"))
        ts = parse_ts(step.get("timestamp"))
        if ts is None:
            missing_ts += 1
            # Only tokens are lost, and only when a timestamp-less step actually
            # carries some. Never silently reassign them to another moment.
            unattributed += metered
            continue
        if metered <= 0:
            continue
        events.append(TokenEvent(ts, metered, prompt, completion, cached,
                                 profile, trial_id, i))
    events.sort(key=lambda e: e.ts)
    return Extraction(events, unattributed, missing_ts)


def extract_trial(trial_dir: Path, *, profile: str = "") -> Extraction:
    trial_dir = Path(trial_dir)
    traj = lib.jload(trial_dir / "agent" / "trajectory.json") or {}
    return extract_events(traj, profile=profile, trial_id=trial_dir.name)


def extract_run_dir(run_dir: Path, *, profile: str = "") -> Extraction:
    """Every token event in one Harbor output directory."""
    run_dir = Path(run_dir)
    events: list[TokenEvent] = []
    unattributed = missing = 0
    for trial in lib.scan_run_dir(run_dir):
        matches = [p for p in run_dir.rglob(trial.trial_dir) if p.is_dir()]
        if not matches:
            continue
        ex = extract_trial(matches[0], profile=profile)
        events.extend(ex.events)
        unattributed += ex.unattributed_metered
        missing += ex.steps_without_timestamp
    events.sort(key=lambda e: e.ts)
    return Extraction(events, unattributed, missing)


class TokenTimeline:
    """Token events kept in timestamp order, for true rolling-window TPM.

    This is a RETROSPECTIVE instrument. Events only become visible when their
    trajectory is written, which for most agents is when the trial ends -- so
    while a batch is in flight, the recent end of the timeline is empty and its
    TPM reads low. That is precisely why the controller must not throttle on it:
    see `Scheduler.allocate`, which throttles on projected per-worker TPM and
    corroborated 429s instead.
    """

    def __init__(self, window_sec: float = 300.0):
        self.window = float(window_sec)
        self._ts: list[float] = []
        self._ev: list[TokenEvent] = []
        self.unattributed_metered = 0

    def add(self, events) -> int:
        added = 0
        for e in events:
            i = bisect.bisect_right(self._ts, e.ts)
            self._ts.insert(i, e.ts)
            self._ev.insert(i, e)
            added += 1
        return added

    def add_extraction(self, ex: Extraction) -> int:
        self.unattributed_metered += ex.unattributed_metered
        return self.add(ex.events)

    def __len__(self) -> int:
        return len(self._ev)

    @property
    def total_metered(self) -> int:
        """Every metered token seen, attributed or not.

        Invariant worth stating: re-attributing events in time must never
        change this number. A timeline built from batch-end lumps and one built
        from per-step timestamps hold the SAME total; only the peak differs.
        """
        return sum(e.metered for e in self._ev) + self.unattributed_metered

    def span(self) -> tuple[float, float]:
        return (self._ts[0], self._ts[-1]) if self._ts else (0.0, 0.0)

    def tpm(self, at: float, *, window: float | None = None,
            profile: str | None = None) -> float:
        """Metered tokens per minute in the window ENDING at `at`."""
        win = float(window if window is not None else self.window)
        lo = bisect.bisect_left(self._ts, at - win)
        hi = bisect.bisect_right(self._ts, at)
        total = sum(e.metered for e in self._ev[lo:hi]
                    if profile is None or e.profile == profile)
        return total * 60.0 / win

    def peak_tpm(self, *, window: float | None = None,
                 step: float = 30.0) -> tuple[float, float]:
        """(peak_tpm, when). Scans the whole timeline on a fixed grid."""
        if not self._ts:
            return 0.0, 0.0
        win = float(window if window is not None else self.window)
        start, end = self._ts[0], self._ts[-1]
        best, best_at = 0.0, start
        t = start
        while t <= end + win:
            v = self.tpm(t, window=win)
            if v > best:
                best, best_at = v, t
            t += step
        return best, best_at

    def series(self, *, window: float | None = None,
               step: float = 60.0) -> list[tuple[float, float]]:
        if not self._ts:
            return []
        win = float(window if window is not None else self.window)
        out, t, end = [], self._ts[0], self._ts[-1]
        while t <= end + win:
            out.append((t, self.tpm(t, window=win)))
            t += step
        return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="True TPM from per-step timestamps.")
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--window", type=float, default=300.0)
    a = ap.parse_args()
    tl = TokenTimeline(a.window)
    for rd in a.run_dirs:
        prof = next((p for p in ("claude", "fable", "codex", "llama")
                     if f"-{p}-" in rd.name), "")
        ex = extract_run_dir(rd, profile=prof)
        tl.add_extraction(ex)
        print(f"{rd.name}: {len(ex.events)} events, coverage {ex.coverage:.4%}")
    peak, when = tl.peak_tpm()
    lo, hi = tl.span()
    print(f"\nevents={len(tl)} span={hi - lo:.0f}s "
          f"unattributed={tl.unattributed_metered}")
    print(f"true peak {a.window:.0f}s-window TPM = {peak:,.0f} "
          f"at {datetime.fromtimestamp(when, timezone.utc).isoformat()}")
