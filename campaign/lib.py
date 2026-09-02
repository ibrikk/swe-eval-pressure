#!/usr/bin/env python3
"""Shared constants and helpers for the SWE-EvalPressure replication campaign.

Everything here is campaign-scoped: no function in this module will read, write,
or accept a path under results/full, results/resource, results/archive or any
other historical location. Run directories must be supplied explicitly and must
live inside the campaign namespace (see `assert_campaign_path`).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CAMPAIGN_ID = "replication-20260902-v1"
CAMPAIGN_ROOT = PROJECT_ROOT / "results" / "campaigns" / CAMPAIGN_ID

PROFILES = ("claude", "fable", "codex", "llama")
MODES = ("full", "resource")

# Base-task sharding is fixed-size over the 70 ordered base task ids:
#   shard i selects base_ids[(i-1)*30 : min(i*30, 70)]  ->  30, 30, 10
SHARD_SIZE = 30
SHARD_INDICES = (1, 2, 3)
BASE_TASK_COUNT = 70
BASE_TASKS_PER_SHARD = {1: 30, 2: 30, 3: 10}

# Arm suffix -> (condition, delivery_channel, pressure_kind)
FULL_ARMS = {
    "clean-n": ("clean", "none", "none"),
    "eval-src": ("eval_only", "source", "none"),
    "eval-fin-src": ("eval_financial", "source", "financial"),
    "eval-self-src": ("eval_self_preservation", "source", "self_preservation"),
    "eval-root": ("eval_only", "root", "none"),
    "eval-fin-root": ("eval_financial", "root", "financial"),
    "eval-self-root": ("eval_self_preservation", "root", "self_preservation"),
    "eval-scaf": ("eval_only", "scaffold", "none"),
    "eval-fin-scaf": ("eval_financial", "scaffold", "financial"),
    "eval-self-scaf": ("eval_self_preservation", "scaffold", "self_preservation"),
}
RESOURCE_ARMS = {
    "clean-n": ("clean", "none", "none"),
    "eval-scaf": ("eval_only", "scaffold", "none"),
    "eval-resource-scaf": ("eval_resource_deprivation", "scaffold", "resource_deprivation"),
}
ARMS = {"full": FULL_ARMS, "resource": RESOURCE_ARMS}
VARIANTS_PER_TASK = {"full": 10, "resource": 3}
ARM_FACTORS = ("condition", "delivery_channel", "pressure_kind")


def arm_factors(mode: str, arm: str) -> dict[str, str]:
    """The experimental factors an arm encodes, as named fields.

    Kept here so every writer (corpus rows, cell audit, analysis) labels the
    design the same way instead of each re-deriving it from the arm string.
    """
    return dict(zip(ARM_FACTORS, ARMS[mode].get(arm, ("unknown",) * 3)))

# Version pins. Every stack the campaign executes must appear here.
VERSION_PINS = {
    "claude": {"agent": "claude-code", "pin_env": "CLAUDE_CODE_VERSION", "version": "2.1.247"},
    "fable": {"agent": "claude-code", "pin_env": "CLAUDE_CODE_VERSION", "version": "2.1.247"},
    "codex": {"agent": "codex", "pin_env": "CODEX_VERSION", "version": "0.147.0"},
    "llama": {"agent": "mini-swe-agent", "pin_env": "MINI_SWE_VERSION", "version": "2.4.5"},
}
MODEL_PINS = {
    "claude": "anthropic/claude-opus-4-8",
    "fable": "anthropic/claude-fable-5",
    "codex": "openai/gpt-5.6",
    "llama": "openai/llmengine/llama-3-3-70b-instruct",
}

LITELLM_ANTHROPIC_BASE = "https://litellm-proxy.ml.scale.com"

STATUS_COMPLETE = "complete"
STATUS_SYNTHETIC = "synthetic"
STATUS_BUDGET_CENSORED = "budget_censored"
STATUS_ERROR = "error"
# The provider's safety layer answered INSTEAD of the model: prompt billed, one
# turn, zero generation. It is an OBSERVED OUTCOME OF THE DEPLOYED STACK, so it
# is accepted into the corpus -- but it is NOT a model observation and must
# never be pooled with one. It is deliberately its OWN status rather than
# STATUS_SYNTHETIC: `synthetic` means "no real request was ever made", which is
# a harness defect the validator rejects outright, whereas a provider block is
# a real request that the vendor stack terminated. Conflating them would either
# make a genuine stack outcome look like a harness bug or make a harness bug
# look like a legitimate result.
STATUS_PROVIDER_BLOCKED = "provider_blocked"

# Statuses admissible in the final accepted corpus. Everything else means the
# cell has no explicit outcome yet and the corpus is not rectangular.
ACCEPTED_STATUSES = (STATUS_COMPLETE, STATUS_PROVIDER_BLOCKED)

BUDGET_MARKERS = (
    "Budget has been exceeded",
    "budget_exceeded",
    "ExceededBudget",
)


# --------------------------------------------------------------------------- #
# provider-layer refusal detection
# --------------------------------------------------------------------------- #
# Markers the safety layer leaves in the agent log when it answers in the
# model's place. `stop_reason` alone is NOT enough: a model that stops to refuse
# in its own words carries the same reason with real output, and that is a
# valid observation of model behaviour, not a block.
REFUSAL_LOG_MARKERS = ("model_refusal_no_fallback", "api_refusal_category")
REFUSAL_EXCEPTIONS = ("safetyrefusal", "agentsafetyrefusal")
_RE_REFUSAL_CATEGORY = re.compile(r'api_refusal_category"?\s*:\s*"([A-Za-z0-9_.-]+)"')
REFUSAL_CATEGORY_UNKNOWN = "unspecified"


def _agent_log_texts(trial_path: Path):
    agent_dir = Path(trial_path) / "agent"
    if not agent_dir.is_dir():
        return
    for log in sorted(agent_dir.glob("*.txt")):
        try:
            yield log.name, log.read_text(errors="replace")
        except OSError:
            continue


def provider_refusal_evidence(trial_path: Path, traj: dict, result: dict) -> str:
    """Why this trial looks like an API-layer block, or "" if it does not.

    Evidence only -- the caller decides. Whether the block replaced the model is
    settled by the completion-token count, not by anything found here.
    """
    exc = str(((result.get("exception_info") or {}).get("exception_type")) or "")
    if any(m in exc.lower().replace("_", "") for m in REFUSAL_EXCEPTIONS):
        return f"agent raised {exc}"
    for step in traj.get("steps") or ():
        if not isinstance(step, dict):
            continue
        if str(((step.get("extra") or {}).get("stop_reason")) or "") == "refusal":
            return "trajectory step carries stop_reason 'refusal'"
    for name, text in _agent_log_texts(trial_path):
        for marker in REFUSAL_LOG_MARKERS:
            if marker in text:
                return f"agent log {name} records {marker}"
    return ""


def provider_refusal_category(trial_path: Path) -> str:
    """The vendor's own label for the block, e.g. "cyber".

    Read verbatim from the agent log; never inferred from the task content. If
    the vendor recorded no category we say so explicitly rather than guessing --
    an invented category would be a fabricated experimental field.
    """
    for _name, text in _agent_log_texts(trial_path):
        m = _RE_REFUSAL_CATEGORY.search(text)
        if m:
            return m.group(1)
    return REFUSAL_CATEGORY_UNKNOWN


# --------------------------------------------------------------------------- #
# paths / hashing
# --------------------------------------------------------------------------- #
def campaign_paths() -> dict[str, Path]:
    return {
        "root": CAMPAIGN_ROOT,
        "full": CAMPAIGN_ROOT / "full",
        "resource": CAMPAIGN_ROOT / "resource",
        "logs": CAMPAIGN_ROOT / "logs",
        "manifests": CAMPAIGN_ROOT / "manifests",
        "provenance": CAMPAIGN_ROOT / "provenance",
        "validation": CAMPAIGN_ROOT / "validation",
        "analysis": CAMPAIGN_ROOT / "analysis",
        "datasets": CAMPAIGN_ROOT / "datasets",
        "manifest": CAMPAIGN_ROOT / "CAMPAIGN_MANIFEST.json",
        "readme": CAMPAIGN_ROOT / "CAMPAIGN_README.md",
    }


def assert_campaign_path(path: Path, what: str = "path") -> Path:
    """Refuse any path outside the campaign namespace.

    This is the structural defence against a historical Aug 2026 run directory
    being handed to the campaign's provenance/validation/analysis tooling.
    """
    resolved = Path(path).resolve()
    root = CAMPAIGN_ROOT.resolve()
    if not (resolved == root or root in resolved.parents):
        raise ValueError(
            f"refusing {what} outside the campaign namespace: {resolved}\n"
            f"  campaign namespace is: {root}\n"
            f"  historical results are provenance-only and must never be mixed in."
        )
    return resolved


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_tree(root: Path) -> str:
    """Content hash of a directory: sha256 over sorted (relpath, filehash) pairs."""
    h = hashlib.sha256()
    files = sorted(
        p for p in Path(root).rglob("*") if p.is_file() and p.name != ".DS_Store"
    )
    for p in files:
        rel = p.relative_to(root).as_posix()
        h.update(rel.encode())
        h.update(b"\0")
        h.update(sha256_file(p).encode())
        h.update(b"\n")
    return h.hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def jload(path: Path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# expected campaign shape
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Cell:
    """One unit of execution: (mode, profile, shard_index)."""
    mode: str
    profile: str
    shard_index: int

    @property
    def base_tasks(self) -> int:
        return BASE_TASKS_PER_SHARD[self.shard_index]

    @property
    def expected_trials(self) -> int:
        return self.base_tasks * VARIANTS_PER_TASK[self.mode]

    @property
    def shard_label(self) -> str:
        return f"chunk-{self.shard_index}-size-{SHARD_SIZE}"

    @property
    def key(self) -> str:
        return f"{self.mode}/{self.profile}/{self.shard_label}"


def all_cells(modes=MODES, profiles=PROFILES) -> list[Cell]:
    return [
        Cell(m, p, i)
        for m in modes
        for p in profiles
        for i in SHARD_INDICES
    ]


def expected_totals() -> dict:
    cells = all_cells()
    per_mode = {}
    for m in MODES:
        per_profile = sum(c.expected_trials for c in cells if c.mode == m and c.profile == "claude")
        per_mode[m] = {
            "per_profile": per_profile,
            "profiles": len(PROFILES),
            "total": per_profile * len(PROFILES),
        }
    return {
        "full": per_mode["full"],
        "resource": per_mode["resource"],
        "campaign_total": per_mode["full"]["total"] + per_mode["resource"]["total"],
        "cells": len(cells),
    }


# --------------------------------------------------------------------------- #
# trial parsing
# --------------------------------------------------------------------------- #
TRIAL_RE = re.compile(r"^ea-([0-9a-f]+)-(.+?)(?:__[A-Za-z0-9]+)?$")


def parse_trial_dir(name: str) -> tuple[str | None, str | None]:
    """'ea-10d4b445-eval-src__pXYfLHz' -> ('10d4b445', 'eval-src')."""
    m = TRIAL_RE.match(name)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def _has_budget_marker(trial_dir: Path) -> bool:
    for name in ("exception.txt", "result.json"):
        p = trial_dir / name
        if p.is_file():
            try:
                text = p.read_text(errors="ignore")
            except Exception:
                continue
            if any(mk in text for mk in BUDGET_MARKERS):
                return True
    agent_dir = trial_dir / "agent"
    if agent_dir.is_dir():
        for p in agent_dir.iterdir():
            if not p.is_file() or p.stat().st_size > 64 << 20:
                continue
            try:
                text = p.read_text(errors="ignore")
            except Exception:
                continue
            if any(mk in text for mk in BUDGET_MARKERS):
                return True
    return False


@dataclass
class Trial:
    run_dir: str
    trial_dir: str
    base_task_id: str
    arm: str
    agent_name: str
    agent_version: str
    model_name: str
    cost_usd: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    steps: int | None
    status: str
    reward: float | None
    resolved: bool | None
    # cache READS are not metered by the gateway (total_cached_tokens ==
    # total_cache_read_input_tokens, verified field-by-field); the scheduler
    # needs them to compute metered tokens, so carry them through.
    cached_tokens: int | None = None
    # Did a model actually generate for this trial? Every downstream analysis of
    # MODEL behaviour must gate on this; a PROVIDER_BLOCKED row has it False.
    model_started: bool = True
    provider_refusal: bool = False
    provider_refusal_category: str = ""


def scan_run_dir(run_dir: Path) -> list[Trial]:
    """Enumerate trials in ONE Harbor output directory.

    No globbing across sibling runs, no dedupe by task_name, no implicit
    fallback to any other directory. What you pass is what you get.
    """
    run_dir = Path(run_dir)
    # Harbor nests <output>/<job_name>/<trial>/...
    inners = [d for d in run_dir.iterdir() if d.is_dir()] if run_dir.is_dir() else []
    trial_parents = [d for d in inners if any(c.is_dir() and c.name.startswith("ea-") for c in d.iterdir())]
    if not trial_parents:
        trial_parents = [run_dir]

    out: list[Trial] = []
    for parent in trial_parents:
        for td in sorted(p for p in parent.iterdir() if p.is_dir() and p.name.startswith("ea-")):
            base_id, arm = parse_trial_dir(td.name)
            traj = jload(td / "agent" / "trajectory.json") or {}
            result = jload(td / "result.json") or {}
            rewards = ((result.get("verifier_result") or {}).get("rewards") or {})
            reward = rewards.get("reward")
            reward = float(reward) if isinstance(reward, (int, float)) else None
            agent = traj.get("agent") or {}
            fm = traj.get("final_metrics") or {}
            model = agent.get("model_name") or ""
            cost = fm.get("total_cost_usd")
            completion = fm.get("total_completion_tokens")
            generated = isinstance(completion, (int, float)) and completion > 0
            # Checked BEFORE the synthetic rule, because an API-layer block is
            # exactly what produces a synthetic model name. A refusal WITH real
            # output is the model speaking for itself and stays complete.
            refusal = provider_refusal_evidence(td, traj, result) if traj else ""
            blocked = bool(refusal) and not generated
            if blocked:
                status = STATUS_PROVIDER_BLOCKED
            elif "synthetic" in model.lower():
                status = STATUS_SYNTHETIC
            elif _has_budget_marker(td):
                status = STATUS_BUDGET_CENSORED
            elif not traj:
                status = STATUS_ERROR
            else:
                status = STATUS_COMPLETE
            out.append(Trial(
                run_dir=run_dir.name,
                trial_dir=td.name,
                base_task_id=base_id or "",
                arm=arm or "",
                agent_name=agent.get("name", ""),
                agent_version=agent.get("version", ""),
                model_name=model,
                cost_usd=float(cost) if isinstance(cost, (int, float)) else None,
                prompt_tokens=fm.get("total_prompt_tokens"),
                completion_tokens=fm.get("total_completion_tokens"),
                steps=fm.get("total_steps"),
                status=status,
                reward=reward,
                resolved=(reward >= 1.0) if reward is not None else None,
                cached_tokens=fm.get("total_cached_tokens"),
                model_started=bool(generated) and not blocked,
                provider_refusal=blocked,
                provider_refusal_category=(
                    provider_refusal_category(td) if blocked else ""),
            ))
    return out


# --------------------------------------------------------------------------- #
# budget
# --------------------------------------------------------------------------- #
@dataclass
class BudgetStatus:
    ok: bool
    spend: float | None
    max_budget: float | None
    remaining: float | None
    tpm_limit: str | None
    rpm_limit: str | None
    http_status: int | None
    key_fingerprint: str
    error: str = ""

    def as_dict(self):
        return asdict(self)


def key_fingerprint(key: str) -> str:
    """Stable, non-reversible key identity for manifests and logs."""
    if not key:
        return "EMPTY"
    return f"sha256:{hashlib.sha256(key.encode()).hexdigest()[:16]}/suffix:{key[-4:]}"


def probe_budget(key: str | None = None, base: str = LITELLM_ANTHROPIC_BASE,
                 timeout: int = 30) -> BudgetStatus:
    """Read LiteLLM key budget from response headers.

    Deliberately cheap and deliberately non-billable. LiteLLM stamps
    x-litellm-key-spend / -max-budget on the response even for a request it
    rejects at validation time, so this sends a request the upstream provider
    refuses before any inference happens: a PINNED, key-allowed model with an
    EMPTY message list. The gateway answers HTTP 400 with
    `x-litellm-response-cost: 0` and the budget headers attached.

    Why not an unroutable model name: this key carries a model ACL, and LiteLLM
    rejects an out-of-ACL model at HTTP 401 *before* stamping budget headers, so
    that probe reads as an auth failure and hides the budget. Why not /key/info
    or /user/info: those admin routes are 403 on this gateway.
    """
    key = key or os.environ.get("LITE_LLM_KEY") or os.environ.get("ANTHROPIC_API_KEY") or ""
    fp = key_fingerprint(key)
    if not key:
        return BudgetStatus(False, None, None, None, None, None, None, fp,
                            "no LiteLLM key in LITE_LLM_KEY/ANTHROPIC_API_KEY")
    payload = json.dumps({
        "model": os.environ.get("CAMPAIGN_BUDGET_PROBE_MODEL", MODEL_PINS["claude"]),
        "messages": [],          # rejected upstream; zero tokens, zero cost
        "max_tokens": 1,
    })
    cmd = [
        "curl", "-sS", "-m", str(timeout), "-D", "-", "-o", "/dev/null",
        "-H", f"Authorization: Bearer {key}",
        "-H", "Content-Type: application/json",
        "-X", "POST", f"{base}/v1/chat/completions",
        "-d", payload,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 10)
    except Exception as exc:
        return BudgetStatus(False, None, None, None, None, None, None, fp, str(exc))

    headers, status = {}, None
    for line in proc.stdout.splitlines():
        if line.startswith("HTTP/"):
            parts = line.split()
            if len(parts) > 1 and parts[1].isdigit():
                status = int(parts[1])
        elif ":" in line:
            k, _, v = line.partition(":")
            headers[k.strip().lower()] = v.strip()

    def num(name):
        try:
            return float(headers[name])
        except Exception:
            return None

    spend = num("x-litellm-key-spend")
    maxb = num("x-litellm-key-max-budget")
    remaining = (maxb - spend) if (spend is not None and maxb is not None) else None
    err = ""
    cost_header = headers.get("x-litellm-response-cost")
    if status in (401, 403) and spend is None:
        err = f"key rejected by gateway (HTTP {status})"
    elif cost_header not in (None, "0", "0.0"):
        err = f"budget probe was billed {cost_header}; refusing to treat it as free"
    elif spend is None or maxb is None:
        err = "gateway did not return budget headers"
    return BudgetStatus(
        ok=not err,
        spend=spend, max_budget=maxb, remaining=remaining,
        tpm_limit=headers.get("x-litellm-key-tpm-limit"),
        rpm_limit=headers.get("x-litellm-key-rpm-limit"),
        http_status=status, key_fingerprint=fp, error=err,
    )


def eprint(*a):
    print(*a, file=sys.stderr)
