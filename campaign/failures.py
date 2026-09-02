#!/usr/bin/env python3
"""Independent failure classification and explicit retry lineage.

Harbor's own classifier is NOT trusted. The audit found a concrete false
positive: a codex `exit 1` on 2026-08-27 was raised as
`harbor.agents.installed.base.ApiRateLimitError` with no HTTP 429 anywhere in
the evidence -- `_classify_exec_error` is a heuristic over CLI output. Retrying
on that signal alone is how a budget exhaustion turns into a retry storm, which
is exactly the Aug-27 failure mode (hundreds of synthetic 2-step trials).

So: a Harbor exception CLASS NAME is never sufficient evidence. A retry needs
corroboration from the actual error text or HTTP status.

Precedence is deliberate and budget-first:
    BUDGET > PARTIAL_MODEL > TRANSIENT_PROVIDER > PRE_MODEL_INFRA > PERMANENT
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path

from campaign import lib

# failure classes
BUDGET = "budget"                      # never retried, stops the shard
TRANSIENT_PROVIDER = "transient_provider"   # 429 / 5xx / reset / timeout
PRE_MODEL_INFRA = "pre_model_infra"    # setup died before any model step
PARTIAL_MODEL = "partial_model"        # model did real work, then failed
PERMANENT = "permanent"                # anything we cannot justify retrying

RETRYABLE = (TRANSIENT_PROVIDER, PRE_MODEL_INFRA, PARTIAL_MODEL)
MAX_RETRIES = 2                        # 2 retries => at most 3 attempts total

# --- corroborating evidence patterns --------------------------------------- #
# Real 429 evidence: an actual status code or an actual provider error body.
# NOT an exception class name.
_RE_429 = re.compile(
    r"(?:\bHTTP/\d(?:\.\d)?\s+429\b)"
    r"|(?:\bstatus(?:_code)?[\"'\s:=]+429\b)"
    r"|(?:\b429\s+Too\s+Many\s+Requests\b)"
    r"|(?:\"type\"\s*:\s*\"rate_limit_error\")"
    r"|(?:\bRate\s*limit\s+(?:reached|exceeded)\b)",
    re.I,
)
_RE_5XX = re.compile(
    r"(?:\bHTTP/\d(?:\.\d)?\s+5\d\d\b)"
    r"|(?:\bstatus(?:_code)?[\"'\s:=]+5\d\d\b)"
    r"|(?:\b(?:internal\s+server\s+error|bad\s+gateway|service\s+unavailable|gateway\s+time-?out)\b)"
    r"|(?:\"type\"\s*:\s*\"(?:api_error|overloaded_error)\")",
    re.I,
)
_RE_NET = re.compile(
    r"(?:connection\s+reset(?:\s+by\s+peer)?)"
    r"|(?:\bECONNRESET\b)|(?:\bETIMEDOUT\b)|(?:\bEPIPE\b)"
    r"|(?:remote\s+end\s+closed\s+connection)"
    r"|(?:read\s+timed?\s*out)|(?:\bReadTimeout\b)|(?:\bConnectTimeout\b)"
    r"|(?:server\s+disconnected)",
    re.I,
)
# Setup died before the agent ever spoke to a model.
_RE_INFRA = re.compile(
    r"(?:failed\s+to\s+(?:build|start|create)\s+(?:image|container|sandbox|environment))"
    r"|(?:modal\S*\s*(?:error|exception|timeout))"
    r"|(?:environment\s+setup\s+failed)|(?:\bimage\s+build\s+failed\b)"
    r"|(?:docker[^\n]{0,40}(?:daemon|error))"
    r"|(?:container\s+(?:did\s+not\s+start|startup\s+failed|exited))"
    r"|(?:no\s+space\s+left\s+on\s+device)",
    re.I,
)
# Exception class names that Harbor emits. Recognised, but NEVER sufficient.
_RE_HARBOR_CLASSNAME = re.compile(
    r"\b(?:ApiRateLimitError|ApiError|AgentExecutionError|ExecError)\b")


@dataclass
class Classification:
    failure_class: str
    reason: str
    model_started: bool
    retryable: bool
    corroborated: bool = False
    evidence: str = ""

    def as_dict(self):
        return asdict(self)


def _text_of(trial_dir: Path) -> str:
    """Concatenate the real error surfaces of one trial directory."""
    parts = []
    for name in ("exception.txt", "trial.log", "job.log", "result.json"):
        p = trial_dir / name
        if p.is_file():
            try:
                parts.append(p.read_text(errors="replace"))
            except OSError:
                pass
    for name in ("agent/codex.txt", "agent/agent.log"):
        p = trial_dir / name
        if p.is_file():
            try:
                parts.append(p.read_text(errors="replace"))
            except OSError:
                pass
    return "\n".join(parts)


def model_started_from(trial_dir: Path) -> bool:
    """True if the model executed meaningful steps before failing."""
    traj = lib.jload(trial_dir / "agent" / "trajectory.json") or {}
    if not traj:
        return False
    if traj.get("steps"):
        return True
    fm = traj.get("final_metrics") or {}
    return bool((fm.get("total_steps") or 0) > 0
                or (fm.get("total_completion_tokens") or 0) > 0)


def classify(error_text: str, *, model_started: bool = False,
             http_status: int | None = None) -> Classification:
    """Classify from EVIDENCE, not from Harbor's verdict."""
    text = error_text or ""

    # 1. BUDGET first, always. Never retryable, and it outranks a 429 because a
    #    budget-exhausted gateway can also emit throttling-shaped noise.
    for marker in lib.BUDGET_MARKERS:
        if marker.lower() in text.lower():
            return Classification(BUDGET, f"budget marker {marker!r} present",
                                  model_started, False, True, marker)

    m429 = _RE_429.search(text)
    m5xx = _RE_5XX.search(text)
    mnet = _RE_NET.search(text)
    minfra = _RE_INFRA.search(text)
    status_429 = http_status == 429
    status_5xx = http_status is not None and 500 <= http_status < 600

    # 2. PARTIAL MODEL: real steps happened. Preserve the original attempt and
    #    retry only under explicit lineage -- never an in-place overwrite.
    if model_started:
        if m429 or status_429 or m5xx or status_5xx or mnet:
            ev = (m429 or m5xx or mnet)
            return Classification(
                PARTIAL_MODEL,
                "model executed steps then hit a corroborated transient provider fault",
                True, True, True, ev.group(0) if ev else f"http {http_status}")
        return Classification(
            PARTIAL_MODEL, "model executed steps then failed without transient evidence",
            True, False, False, "")

    # 3. TRANSIENT PROVIDER, only with corroboration.
    if m429 or status_429:
        return Classification(TRANSIENT_PROVIDER, "corroborated HTTP 429",
                              False, True, True,
                              m429.group(0) if m429 else f"http {http_status}")
    if m5xx or status_5xx:
        return Classification(TRANSIENT_PROVIDER, "corroborated provider 5xx",
                              False, True, True,
                              m5xx.group(0) if m5xx else f"http {http_status}")
    if mnet:
        return Classification(TRANSIENT_PROVIDER, "connection reset / gateway timeout",
                              False, True, True, mnet.group(0))

    # 4. PRE-MODEL INFRASTRUCTURE.
    if minfra:
        return Classification(PRE_MODEL_INFRA, "environment/container setup failure "
                              "before any model step", False, True, True, minfra.group(0))

    # 5. Harbor said "rate limit" but nothing corroborates it. This is the
    #    Aug-27 false positive. Do not retry it as transient.
    if _RE_HARBOR_CLASSNAME.search(text):
        cls = _RE_HARBOR_CLASSNAME.search(text).group(0)
        return Classification(
            PERMANENT,
            f"Harbor raised {cls} but no HTTP status, provider error body or "
            "network signature corroborates it - refusing to treat as transient",
            model_started, False, False, cls)

    return Classification(PERMANENT, "no recognised transient signature",
                          model_started, False, False, "")


def classify_trial_dir(trial_dir: Path) -> Classification:
    trial_dir = Path(trial_dir)
    return classify(_text_of(trial_dir), model_started=model_started_from(trial_dir))


# --------------------------------------------------------------------------- #
# retry ledger -- explicit lineage, append-only, never a silent replacement
# --------------------------------------------------------------------------- #
@dataclass
class RetryRecord:
    original_trial_id: str
    retry_trial_id: str
    retry_number: int
    failure_class: str
    failure_reason: str
    model_started: bool
    started_at: str
    finished_at: str = ""
    accepted_status: str = "pending"    # pending | accepted | failed | abandoned
    cell: str = ""
    evidence: str = ""

    def as_dict(self):
        return asdict(self)


class RetryLedger:
    """Append-only JSONL of retry lineage under the campaign namespace.

    Two rules the validator enforces against this file:
      * the failed original is preserved -- never deleted, never overwritten;
      * an accepted retry is reachable ONLY through explicit lineage, never by
        deduping on task_name.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        lib.assert_campaign_path(self.path.parent, "retry ledger directory")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def records(self) -> list[RetryRecord]:
        if not self.path.is_file():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if line:
                out.append(RetryRecord(**json.loads(line)))
        return out

    def retry_count(self, original_trial_id: str) -> int:
        return sum(1 for r in self.records() if r.original_trial_id == original_trial_id)

    def may_retry(self, original_trial_id: str, cls: Classification) -> tuple[bool, str]:
        if cls.failure_class == BUDGET:
            return False, "budget failures are never retried"
        if not cls.retryable:
            return False, cls.reason
        n = self.retry_count(original_trial_id)
        if n >= MAX_RETRIES:
            return False, f"retry budget exhausted ({n}/{MAX_RETRIES})"
        return True, f"retry {n + 1}/{MAX_RETRIES} for {cls.failure_class}"

    def open_retry(self, original_trial_id: str, retry_trial_id: str,
                   cls: Classification, *, cell: str = "") -> RetryRecord:
        rec = RetryRecord(
            original_trial_id=original_trial_id,
            retry_trial_id=retry_trial_id,
            retry_number=self.retry_count(original_trial_id) + 1,
            failure_class=cls.failure_class,
            failure_reason=cls.reason,
            model_started=cls.model_started,
            started_at=lib.now_iso(),
            cell=cell,
            evidence=cls.evidence,
        )
        with open(self.path, "a") as fh:
            fh.write(json.dumps(rec.as_dict()) + "\n")
        return rec

    def close_retry(self, retry_trial_id: str, accepted_status: str) -> None:
        recs = self.records()
        for r in recs:
            if r.retry_trial_id == retry_trial_id and r.accepted_status == "pending":
                r.accepted_status = accepted_status
                r.finished_at = lib.now_iso()
        self.path.write_text("".join(json.dumps(r.as_dict()) + "\n" for r in recs))

    def backoff_seconds(self, retry_number: int, base: float = 30.0) -> float:
        """Exponential backoff, capped. 30s, 60s (MAX_RETRIES == 2)."""
        return min(base * (2 ** max(0, retry_number - 1)), 600.0)


if __name__ == "__main__":
    import sys
    for arg in sys.argv[1:]:
        print(arg, "->", json.dumps(classify_trial_dir(Path(arg)).as_dict(), indent=2))
