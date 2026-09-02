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


def _rate(rows, pred) -> float | None:
    if not rows:
        return None
    return sum(1 for r in rows if pred(r)) / len(rows)


def _resolved(row) -> bool:
    return bool(row.get("resolved"))


def summarise(rows: list[dict]) -> dict:
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

    summary = summarise(rows)
    leaks = cross_mode_leak_check(rows)
    summary["cross_mode_leak_check"] = leaks

    paths["analysis"].mkdir(parents=True, exist_ok=True)
    out = paths["analysis"] / "campaign_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")

    if not leaks["ok"]:
        lib.eprint("ANALYSIS FAILED: cross-mode leakage detected.")
        return 1

    lib.eprint(
        f"[analyze] {summary['n_trials']}/{summary['expected_trials']} trials, "
        f"validated at {report.get('generated_at')}"
    )
    lib.eprint(f"[analyze] wrote {out.relative_to(lib.PROJECT_ROOT)}")
    if args.json:
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
