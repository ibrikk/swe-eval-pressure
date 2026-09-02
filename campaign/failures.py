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

    BUDGET > PROVIDER_REFUSAL > DETERMINISTIC_INFRA > PARTIAL_MODEL
           > TRANSIENT_PROVIDER > PRE_MODEL_INFRA > PERMANENT

PROVIDER_REFUSAL and DETERMINISTIC_INFRA were added from the 2026-09-02 shard-1
evidence. Both TIGHTEN retry, never loosen it:

  * PROVIDER_REFUSAL - all 10 arms of base task 4fd11885 came back from Fable 5
    as `AgentSafetyRefusalError` with `stop_reason: refusal` and zero completion
    tokens, because the task (consolidating TruffleHog GitHub-token detectors)
    trips the provider's cyber safeguard. That is a deterministic function of
    the task text, so it is not retryable -- but the old code only reached
    "not retryable" by falling through to `partial_model` with the unhelpful
    reason "failed without transient evidence". Now it is named.

  * DETERMINISTIC_INFRA - the SAME content-addressed Modal image ids
    (im-x6EXQtv6..., im-Y3lpPwyq..., im-gerqAtzp...) failed the image build in
    all FOUR profiles independently, hours apart. An infrastructure failure is
    only transient until it demonstrably reproduces; once a signature has failed
    before, retrying it just buys the same failure again. `pre_model_infra` is
    therefore downgraded on reproduction, via `apply_reproduction_evidence`.

The one place classification was too STRICT is also fixed: an agent-install step
that dies because git/npm could not reach the network inside the sandbox
("could not read Username for 'https://github.com'") is a genuine transient
pre-model infrastructure fault, and used to fall through to PERMANENT.
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
PROVIDER_REFUSAL = "provider_refusal"  # provider safety-refused the task text
DETERMINISTIC_INFRA = "deterministic_infra"  # infra fault that has reproduced

RETRYABLE = (TRANSIENT_PROVIDER, PRE_MODEL_INFRA, PARTIAL_MODEL)
NEVER_RETRYABLE = (BUDGET, PROVIDER_REFUSAL, DETERMINISTIC_INFRA, PERMANENT)
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
# Provider SAFETY REFUSAL. Unlike the rate-limit case, accepting Harbor's class
# name here is conservative: it can only ever BLOCK a retry, never cause one. We
# still prefer body evidence, and match both.
_RE_REFUSAL = re.compile(
    r"(?:\bAgentSafetyRefusalError\b)"
    r"|(?:[\"']stop_reason[\"']\s*:\s*[\"']refusal[\"'])"
    r"|(?:safeguards\s+flagged\s+this\s+message)"
    r"|(?:[\"']type[\"']\s*:\s*[\"']refusal[\"'])",
    re.I,
)
# Agent INSTALL/bootstrap died reaching the network from inside the sandbox.
# This happens before any model step and is genuinely transient (the codex
# profile installs node via nvm, which clones from github.com).
_RE_AGENT_SETUP = re.compile(
    r"(?:could\s+not\s+read\s+Username\s+for)"
    r"|(?:failed\s+to\s+clone\s+\S+\s+repo)"
    r"|(?:NVM\s+failed\s+to\s+load)"
    r"|(?:expected\s+flush\s+after\s+ref\s+listing)"
    r"|(?:npm\s+ERR!\s+network)"
    r"|(?:E(?:AI_AGAIN|NOTFOUND)\b)"
    r"|(?:Temporary\s+failure\s+in\s+name\s+resolution)",
    re.I,
)
# Content-addressed identity of a reproducible infra fault. A Modal image id is
# derived from the build definition, so the same id failing twice is the same
# build failing twice -- not two unlucky draws.
_RE_MODAL_IMAGE = re.compile(r"\b(im-[A-Za-z0-9]{6,})\b")

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
    # Stable identity of a reproducible fault, "" when the failure has no
    # content-addressed identity. See `apply_reproduction_evidence`.
    signature: str = ""

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


def failure_signature(text: str) -> str:
    """Content-addressed identity of a reproducible fault, or "".

    Only content-addressed identifiers qualify. A Modal image id is a hash of
    the build definition, so `im-x6EXQtv6VljLdOL1LimybV` failing in the claude
    profile and in the llama profile six hours later is *the same build failing
    twice*, not two independent unlucky draws. A container id or a request id
    would NOT qualify -- those differ per attempt and would never match.
    """
    m = _RE_MODAL_IMAGE.search(text or "")
    return f"modal-image:{m.group(1)}" if m else ""


def apply_reproduction_evidence(cls: Classification,
                                seen_signatures) -> Classification:
    """Downgrade a 'transient' infra fault that has demonstrably reproduced.

    `seen_signatures` is the set of failure signatures already observed in this
    campaign (from the retry ledger and from earlier failures in this run). An
    infrastructure failure is a retry candidate only while it is plausibly a
    one-off; the second time the exact same content-addressed build fails, the
    evidence says it is deterministic and a retry only buys the same failure.

    This can only ever REMOVE a retry, never add one.
    """
    if cls.failure_class != PRE_MODEL_INFRA or not cls.signature:
        return cls
    if cls.signature not in set(seen_signatures):
        return cls
    return Classification(
        DETERMINISTIC_INFRA,
        f"infrastructure failure {cls.signature} has already failed in this "
        "campaign - reproduced, therefore deterministic, not retryable",
        cls.model_started, False, True, cls.evidence, cls.signature)


def model_started_from(trial_dir: Path) -> bool:
    """True if the model executed meaningful steps before failing.

    NOTE a provider safety refusal also lands here as True: the trajectory has
    `steps`, even though `total_completion_tokens` is 0. That is deliberate --
    the provider WAS reached, and the nuance is carried by the
    PROVIDER_REFUSAL class rather than by flipping this flag, which the
    validator and the corpus builder also read.
    """
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

    # 2. PROVIDER REFUSAL. Deterministic in the task text, so never retryable.
    #    Checked before PARTIAL_MODEL because a refusal *does* produce steps and
    #    would otherwise be swallowed by "failed without transient evidence".
    mref = _RE_REFUSAL.search(text)
    if mref:
        return Classification(
            PROVIDER_REFUSAL,
            "provider safety-refused the task content; deterministic in the "
            "task text, so a retry would refuse identically",
            model_started, False, True, mref.group(0))

    m429 = _RE_429.search(text)
    m5xx = _RE_5XX.search(text)
    mnet = _RE_NET.search(text)
    minfra = _RE_INFRA.search(text)
    status_429 = http_status == 429
    status_5xx = http_status is not None and 500 <= http_status < 600

    # 3. PARTIAL MODEL: real steps happened. Preserve the original attempt and
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

    # 4. TRANSIENT PROVIDER, only with corroboration.
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

    # 5. PRE-MODEL INFRASTRUCTURE. Retryable *on first sight only*; see
    #    `apply_reproduction_evidence` for the reproduction downgrade.
    if minfra:
        return Classification(PRE_MODEL_INFRA, "environment/container setup failure "
                              "before any model step", False, True, True,
                              minfra.group(0), failure_signature(text))

    # 6. AGENT INSTALL reached the network and failed. The codex profile
    #    installs node through nvm, which clones from github.com; when that
    #    clone fails the run dies before any model step. This used to fall
    #    through to PERMANENT, which was wrong -- it is a real transient.
    msetup = _RE_AGENT_SETUP.search(text)
    if msetup:
        return Classification(PRE_MODEL_INFRA,
                              "agent install/bootstrap failed on a network "
                              "operation before any model step",
                              False, True, True, msetup.group(0),
                              failure_signature(text))

    # 7. Harbor said "rate limit" but nothing corroborates it. This is the
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
    # Provenance of BOTH sides of the lineage, so the failed original can always
    # be found on disk and the retry's own artifacts can be audited.
    original_run_dir: str = ""
    retry_run_dir: str = ""
    retry_dataset: str = ""
    signature: str = ""

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

    def signatures(self) -> set:
        """Every failure signature this campaign has already recorded."""
        return {r.signature for r in self.records() if r.signature}

    def open_retry(self, original_trial_id: str, retry_trial_id: str,
                   cls: Classification, *, cell: str = "",
                   original_run_dir: str = "",
                   retry_dataset: str = "") -> RetryRecord:
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
            original_run_dir=str(original_run_dir),
            retry_dataset=str(retry_dataset),
            signature=cls.signature,
        )
        with open(self.path, "a") as fh:
            fh.write(json.dumps(rec.as_dict()) + "\n")
        return rec

    CLOSED_STATUS = ("accepted", "failed", "abandoned")

    def close_retry(self, retry_trial_id: str, accepted_status: str, *,
                    retry_run_dir: str = "") -> RetryRecord | None:
        """Close one open lineage record. Returns the record, or None.

        Every retry the controller launches MUST come back through here -- an
        eternally "pending" record is indistinguishable from a retry that was
        queued and silently dropped, which is exactly what the 2026-09-02 shard
        left behind (6 records, all pending, no finished_at).
        """
        if accepted_status not in self.CLOSED_STATUS:
            raise ValueError(f"accepted_status must be one of {self.CLOSED_STATUS}, "
                             f"got {accepted_status!r}")
        recs = self.records()
        closed = None
        for r in recs:
            if r.retry_trial_id == retry_trial_id and r.accepted_status == "pending":
                r.accepted_status = accepted_status
                r.finished_at = lib.now_iso()
                if retry_run_dir:
                    r.retry_run_dir = str(retry_run_dir)
                closed = r
        self.path.write_text("".join(json.dumps(r.as_dict()) + "\n" for r in recs))
        return closed

    def open_records(self) -> list[RetryRecord]:
        return [r for r in self.records() if r.accepted_status == "pending"]

    def backoff_seconds(self, retry_number: int, base: float = 30.0) -> float:
        """Exponential backoff, capped. 30s, 60s (MAX_RETRIES == 2)."""
        return min(base * (2 ** max(0, retry_number - 1)), 600.0)


if __name__ == "__main__":
    import sys
    for arg in sys.argv[1:]:
        print(arg, "->", json.dumps(classify_trial_dir(Path(arg)).as_dict(), indent=2))
