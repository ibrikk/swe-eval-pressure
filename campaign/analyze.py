#!/usr/bin/env python3
"""Campaign-scoped analysis entry point.

Deliberately narrow. It refuses to run unless `campaign.validate` has passed,
and it reads exactly one input: provenance/corpus.jsonl, which itself was built
only from accepted attempts inside the campaign namespace.

Why so restrictive: the Aug 2026 analysis pipeline globbed a hard-coded list of
run directories that mixed a good shard 1, a budget-censored shard 2, an
archived abort and a repair run, then deduped by task_name. That silently
substituted salvaged trials for missing ones and made the corpus look complete.
Here there is no glob, no path argument, no dedupe, and no fallback: if the
validator has not signed off on 3,640 fresh trials, analysis does not run.

TWO POPULATIONS, NEVER POOLED
-----------------------------
The corpus holds every expected cell, but not every cell has a model behind it.
`provider_blocked` rows are outcomes of the deployed STACK: the vendor's safety
layer terminated the request before any model generated. They are real data
about the stack and they keep the design rectangular, but they contain zero
model behaviour.

So this module exposes two accessors and no third way in:

  stack_rows(rows)  every accepted observation -- use for end-to-end stack
                    questions (did the deployed system produce a solution?)
  model_rows(rows)  model_started only -- REQUIRED for anything about model
                    behaviour, reasoning, cue recognition, pressure response,
                    tokens, tools or trajectory shape

Every behavioural statistic below is computed on `model_rows`. Passing raw rows
to those functions raises: a mean over a blocked row would average in a token
count, step count and reward that no model produced.

THE 2026-09-02 BLOCK, AND THE SENSITIVITY ANALYSIS IT REQUIRES
--------------------------------------------------------------
One of the 70 base tasks (a TruffleHog credential-detector consolidation) was
blocked by fable's provider content filter on ALL TEN of its FULL arms --
including `clean-n`, which carries no cue and no injected content. The block is
therefore a property of the TASK, not of the pressure treatment: it cannot
confound a between-arm comparison, because it removed every arm equally.

It does make fable's per-arm n one lower than the other profiles'. To show that
this unbalanced cell is not driving anything, `sensitivity_complete_cases`
re-runs the same summary over only those base tasks where EVERY profile and arm
produced a model observation -- 69 of 70 as of the block. It is pre-specified,
not chosen after seeing the results, and it reproduces exactly the estimand a
task replacement would have bought, at zero cost to the corpus.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict

from campaign import lib


def require_validated(paths) -> dict:
    report_path = paths["validation"] / "validation_report.json"
    if not report_path.exists():
        raise SystemExit(
            "REFUSED: no validation report. Run `./campaign.sh validate "
            f"{lib.CAMPAIGN_ID}` first."
        )
    report = json.loads(report_path.read_text())
    if report.get("campaign_id") != lib.CAMPAIGN_ID:
        raise SystemExit(
            f"REFUSED: validation report is for campaign {report.get('campaign_id')!r}, "
            f"not {lib.CAMPAIGN_ID!r}."
        )
    if not report.get("ok"):
        failed = [c["id"] for c in report.get("checks", []) if not c.get("ok")]
        raise SystemExit(
            "REFUSED: validation did not pass. Failing checks: "
            + (", ".join(failed) or "unknown")
        )
    return report


def load_corpus(paths) -> list[dict]:
    corpus_path = paths["provenance"] / "corpus.jsonl"
    if not corpus_path.exists():
        raise SystemExit("REFUSED: provenance/corpus.jsonl missing; run validate first.")
    rows = []
    with corpus_path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit("REFUSED: corpus is empty.")
    bad = [r for r in rows if r.get("campaign_id") != lib.CAMPAIGN_ID]
    if bad:
        raise SystemExit(
            f"REFUSED: {len(bad)} corpus rows carry a foreign campaign_id."
        )
    return rows


class NonModelRowsInAnalysis(Exception):
    """Raised when model-behaviour code is handed a row with no model behind it."""


def stack_rows(rows: list[dict]) -> list[dict]:
    """Every accepted observation: the deployed stack's end-to-end outcomes.

    Includes provider-blocked cells. Correct for "did the shipped system solve
    the task"; WRONG for anything about the model.
    """
    return [r for r in rows if r.get("status") in lib.ACCEPTED_STATUSES]


def model_rows(rows: list[dict]) -> list[dict]:
    """Only cells where a model actually generated.

    THE gate for every behavioural analysis. Not a filter for convenience: a
    provider-blocked row has 0 completion tokens, 0 steps and reward 0 that
    belong to the safety layer, and pooling it with model output would report
    model behaviour that never occurred.
    """
    return [r for r in rows if r.get("model_started")]


def require_model_rows(rows: list[dict]) -> list[dict]:
    """Assert the caller already applied the gate. Fail loud, never filter."""
    intruders = [r for r in rows if not r.get("model_started")]
    if intruders:
        raise NonModelRowsInAnalysis(
            f"{len(intruders)} row(s) with model_started=false reached a "
            f"model-behaviour analysis (e.g. "
            f"{[r.get('trial_dir') for r in intruders[:3]]}). Model behaviour, "
            f"reasoning, cue recognition, pressure response, token, tool and "
            f"trajectory analyses must be run on analyze.model_rows(rows). "
            f"Use analyze.stack_rows(rows) for end-to-end stack questions.")
    return rows


def _rate(rows, pred) -> float | None:
    if not rows:
        return None
    return sum(1 for r in rows if pred(r)) / len(rows)


def _resolved(row) -> bool:
    return bool(row.get("resolved"))


def summarise(rows: list[dict]) -> dict:
    """Behavioural summary. `rows` MUST already be model_rows(...)."""
    require_model_rows(rows)
    by_mode_profile_arm: dict[tuple, list] = defaultdict(list)
    by_mode_profile: dict[tuple, list] = defaultdict(list)
    for r in rows:
        by_mode_profile_arm[(r["mode"], r["profile"], r["arm"])].append(r)
        by_mode_profile[(r["mode"], r["profile"])].append(r)

    arms = []
    for (mode, profile, arm), group in sorted(by_mode_profile_arm.items()):
        factors = lib.ARMS[mode].get(arm, ("?", "?", "?"))
        costs = [c for c in (r.get("cost_usd") for r in group) if c]
        steps = [s for s in (r.get("steps") for r in group) if s]
        arms.append({
            "mode": mode,
            "profile": profile,
            "arm": arm,
            "condition": factors[0],
            "delivery_channel": factors[1],
            "pressure_kind": factors[2],
            "n": len(group),
            "resolved_rate": _rate(group, _resolved),
            "mean_cost_usd": round(statistics.fmean(costs), 4) if costs else None,
            "mean_steps": round(statistics.fmean(steps), 2) if steps else None,
        })

    cells = []
    for (mode, profile), group in sorted(by_mode_profile.items()):
        cells.append({
            "mode": mode,
            "profile": profile,
            "n": len(group),
            "expected": lib.expected_totals()[mode]["per_profile"],
            "agent_versions": sorted({r.get("agent_version") for r in group if r.get("agent_version")}),
            "model_ids": sorted({r.get("model_name") for r in group if r.get("model_name")}),
            "resolved_rate": _rate(group, _resolved),
            "total_cost_usd": round(sum(r.get("cost_usd") or 0.0 for r in group), 2),
        })

    return {
        "campaign_id": lib.CAMPAIGN_ID,
        "generated_at": lib.now_iso(),
        "population": "model_observations",
        "n_trials": len(rows),
        "expected_trials": lib.expected_totals()["campaign_total"],
        "by_mode": dict(Counter(r["mode"] for r in rows)),
        "by_profile": dict(Counter(r["profile"] for r in rows)),
        "by_shard": dict(Counter(str(r["shard_index"]) for r in rows)),
        "cells": cells,
        "arms": arms,
        "note": (
            "FULL and RESOURCE are analysed as independent execution sets. "
            "RESOURCE control arms (clean-n, eval-scaf) are RESOURCE's own freshly "
            "executed trajectories; no FULL trajectory is pooled in."
        ),
    }


def stack_outcomes(rows: list[dict]) -> dict:
    """End-to-end outcomes of the DEPLOYED STACK, blocked cells included.

    The only place a provider-blocked cell contributes to a rate. The question
    here is "did the shipped system deliver a solution", and a vendor safety
    block is a genuine way for it not to -- so excluding those cells would
    flatter the stack. Reported alongside, never merged with, the model figures.
    """
    acc = stack_rows(rows)
    by: dict[tuple, list] = defaultdict(list)
    for r in acc:
        by[(r["mode"], r["profile"])].append(r)
    out = []
    for (mode, profile), group in sorted(by.items()):
        blocked = [r for r in group if r.get("provider_refusal")]
        out.append({
            "mode": mode,
            "profile": profile,
            "accepted_observations": len(group),
            "model_observations": sum(1 for r in group if r.get("model_started")),
            "provider_blocked": len(blocked),
            "provider_blocked_categories": sorted(
                {r.get("provider_refusal_category") or "" for r in blocked}),
            # Denominator is every accepted cell, so a blocked cell counts as an
            # unsolved task for the stack -- which is what it was.
            "stack_resolved_rate": _rate(group, _resolved),
        })
    return {
        "note": (
            "Stack-level view. Provider-blocked cells are counted as observed "
            "stack failures (no solution delivered). They are NOT model "
            "refusals and contribute to no model-behaviour statistic."),
        "by_mode_profile": out,
    }


def complete_case_base_tasks(rows: list[dict]) -> list[str]:
    """Base tasks where EVERY expected cell is a model observation.

    A base task is dropped only if some (profile, arm) cell of it has no model
    behind it -- the same criterion for every profile, so the surviving set is
    identical across models and no model is compared on a different task set.
    """
    expected: dict[str, set] = defaultdict(set)
    observed: dict[str, set] = defaultdict(set)
    for r in rows:
        key = (r["mode"], r["profile"], r["arm"])
        expected[r["base_task_id"]].add(key)
        if r.get("model_started"):
            observed[r["base_task_id"]].add(key)
    return sorted(b for b, want in expected.items() if observed[b] == want)


def sensitivity_complete_cases(rows: list[dict]) -> dict:
    """Pre-specified complete-case sensitivity analysis.

    Re-runs the whole behavioural summary over only the base tasks that every
    profile and arm executed. This is the estimand replacing the blocked task
    would have produced -- a perfectly balanced design across models -- obtained
    without discarding 30 valid trajectories or re-deriving the global cue
    assignment. If the headline conclusions hold here too, the one unbalanced
    base task is not driving them.

    Pre-specified means exactly that: the criterion is "every cell executed",
    fixed before the numbers were read, and not tuned to any result.
    """
    keep = set(complete_case_base_tasks(rows))
    dropped = sorted({r["base_task_id"] for r in rows} - keep)
    subset = [r for r in model_rows(rows) if r["base_task_id"] in keep]
    doc = summarise(subset) if subset else {"n_trials": 0}
    doc.update({
        "population": "model_observations_complete_case",
        "base_tasks_included": len(keep),
        "base_tasks_total": len(keep) + len(dropped),
        "base_tasks_dropped": dropped,
        "drop_reason": {
            b: sorted({f"{r['profile']}/{r['mode']}/{r['arm']}"
                       for r in rows
                       if r["base_task_id"] == b and not r.get("model_started")})
            for b in dropped},
        "note": (
            "Complete-case sensitivity analysis over base tasks executed by "
            "every profile in every arm. Pre-specified; the same task set is "
            "used for all models."),
    })
    return doc


def cross_mode_leak_check(rows: list[dict]) -> dict:
    """Prove no FULL trajectory was substituted into RESOURCE (or vice versa)."""
    run_ids: dict[str, set] = defaultdict(set)
    for r in rows:
        run_ids[r["run_id"]].add(r["mode"])
    shared = {rid: sorted(m) for rid, m in run_ids.items() if len(m) > 1}
    trial_keys: dict[tuple, set] = defaultdict(set)
    for r in rows:
        trial_keys[(r["mode"], r["profile"], r["base_task_id"], r["arm"])].add(r["trial_dir"])
    dupes = {"::".join(map(str, k)): sorted(v) for k, v in trial_keys.items() if len(v) > 1}
    return {
        "run_ids_shared_across_modes": shared,
        "duplicate_trial_dirs_per_cell": dupes,
        "ok": not shared and not dupes,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyse the validated campaign corpus.")
    ap.add_argument("--json", action="store_true", help="print the summary to stdout")
    args = ap.parse_args()

    paths = lib.campaign_paths()
    lib.assert_campaign_path(paths["root"], "campaign root")
    report = require_validated(paths)
    rows = load_corpus(paths)

    # The gate, applied once at the entry point. Everything behavioural below
    # runs on `behaviour`; `rows` itself is used only for accounting, the
    # stack-level view and the leak check, which are about cells, not models.
    behaviour = model_rows(rows)
    summary = summarise(behaviour)
    summary["accounting"] = {
        "accepted_observations": len(stack_rows(rows)),
        "model_observations": len(behaviour),
        "provider_blocked": sum(1 for r in rows if r.get("provider_refusal")),
        "expected_cells": lib.expected_totals()["campaign_total"],
    }
    summary["stack_outcomes"] = stack_outcomes(rows)
    summary["sensitivity_complete_cases"] = sensitivity_complete_cases(rows)
    leaks = cross_mode_leak_check(rows)
    summary["cross_mode_leak_check"] = leaks

    paths["analysis"].mkdir(parents=True, exist_ok=True)
    out = paths["analysis"] / "campaign_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")

    if not leaks["ok"]:
        lib.eprint("ANALYSIS FAILED: cross-mode leakage detected.")
        return 1

    acct = summary["accounting"]
    lib.eprint(
        f"[analyze] {acct['accepted_observations']}/{acct['expected_cells']} "
        f"accepted observations = {acct['model_observations']} model + "
        f"{acct['provider_blocked']} provider-blocked; behavioural statistics "
        f"use the {summary['n_trials']} model observations only; "
        f"validated at {report.get('generated_at')}"
    )
    sens = summary["sensitivity_complete_cases"]
    lib.eprint(
        f"[analyze] complete-case sensitivity: "
        f"{sens['base_tasks_included']}/{sens['base_tasks_total']} base tasks "
        f"executed by every profile in every arm"
    )
    lib.eprint(f"[analyze] wrote {out.relative_to(lib.PROJECT_ROOT)}")
    if args.json:
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
