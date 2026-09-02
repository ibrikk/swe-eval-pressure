#!/usr/bin/env python3
"""Per-profile / per-mode cost model for the replication campaign.

Reads the Aug 2026 corpus FOR COST PROVENANCE ONLY. It extracts nothing but
`final_metrics` numbers and never returns a trial, a trajectory, a task result
or anything that could be mistaken for campaign data. The campaign's own
tooling (validate/provenance/analyze) cannot reach this module's inputs.

Trials that were budget-censored or synthetic are excluded from the estimate:
they under-report cost and would bias the campaign budget downwards, which is
precisely the failure that truncated the Aug 2026 study.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from campaign import lib

HISTORICAL_ROOTS = {
    "full": lib.PROJECT_ROOT / "results" / "full",
    "resource": lib.PROJECT_ROOT / "results" / "resource",
}


def _profile_of(run_name: str) -> str | None:
    for p in lib.PROFILES:
        if f"-{p}-" in run_name:
            return p
    return None


def collect() -> dict:
    """(mode, profile) -> list of per-trial cost/token records from healthy trials."""
    out: dict[tuple[str, str], list[dict]] = {}
    for mode, root in HISTORICAL_ROOTS.items():
        if not root.is_dir():
            continue
        for run in sorted(root.glob("swe-eval-pressure-*")):
            profile = _profile_of(run.name)
            if profile is None:
                continue
            for tj in run.glob("*/*/agent/trajectory.json"):
                traj = lib.jload(tj) or {}
                agent = traj.get("agent") or {}
                fm = traj.get("final_metrics") or {}
                model = (agent.get("model_name") or "")
                if "synthetic" in model.lower():
                    continue
                cost = fm.get("total_cost_usd")
                rec = {
                    "cost": float(cost) if isinstance(cost, (int, float)) else None,
                    "ptok": fm.get("total_prompt_tokens") or 0,
                    "ctok": fm.get("total_completion_tokens") or 0,
                    "steps": fm.get("total_steps") or 0,
                }
                out.setdefault((mode, profile), []).append(rec)
    return out


def estimate() -> dict:
    data = collect()
    cells = {}
    for mode in lib.MODES:
        per_profile_trials = lib.VARIANTS_PER_TASK[mode] * lib.BASE_TASK_COUNT
        for profile in lib.PROFILES:
            recs = data.get((mode, profile), [])
            costs = [r["cost"] for r in recs if r["cost"] and r["cost"] > 0]
            ptoks = [r["ptok"] for r in recs if r["ptok"]]
            n_priced = len(costs)
            mean = st.mean(costs) if costs else None
            p90 = sorted(costs)[int(0.9 * (len(costs) - 1))] if costs else None
            cells[f"{mode}/{profile}"] = {
                "mode": mode,
                "profile": profile,
                "historical_trials_observed": len(recs),
                "historical_trials_with_nonzero_cost": n_priced,
                "mean_usd_per_trial": round(mean, 4) if mean is not None else None,
                "p90_usd_per_trial": round(p90, 4) if p90 is not None else None,
                "median_prompt_tokens": int(st.median(ptoks)) if ptoks else None,
                "campaign_trials": per_profile_trials,
                "expected_usd": round(mean * per_profile_trials, 2) if mean is not None else 0.0,
                "p90_usd": round(p90 * per_profile_trials, 2) if p90 is not None else 0.0,
                "cost_reported_by_gateway": bool(n_priced),
            }

    exp = sum(c["expected_usd"] for c in cells.values())
    p90 = sum(c["p90_usd"] for c in cells.values())
    unpriced = sorted(k for k, c in cells.items() if not c["cost_reported_by_gateway"])
    return {
        "cells": cells,
        "expected_total_usd": round(exp, 2),
        "p90_total_usd": round(p90, 2),
        # Planning number: expectation plus a 20% contingency for retries,
        # verifier traffic and the unpriced llama cells.
        "planning_total_usd": round(exp * 1.20, 2),
        "unpriced_cells": unpriced,
        "note": (
            "llama cells are billed at $0 by this gateway (llmengine route reports "
            "x-litellm-response-cost 0), so their true cost is unknown-but-small; "
            "the 20% contingency in planning_total_usd covers them."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    est = estimate()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(est, indent=2) + "\n")
    if args.json:
        print(json.dumps(est, indent=2))
        return
    print(f"{'cell':18s}{'n_hist':>8s}{'mean$':>9s}{'p90$':>9s}{'trials':>8s}{'expected$':>12s}{'p90$tot':>12s}")
    for k in sorted(est["cells"]):
        c = est["cells"][k]
        f = lambda v: f"{v:.4f}" if v is not None else "-"
        print(f"{k:18s}{c['historical_trials_with_nonzero_cost']:>8d}"
              f"{f(c['mean_usd_per_trial']):>9s}{f(c['p90_usd_per_trial']):>9s}"
              f"{c['campaign_trials']:>8d}{c['expected_usd']:>12.2f}{c['p90_usd']:>12.2f}")
    print()
    print(f"  EXPECTED total  : ${est['expected_total_usd']:,.2f}")
    print(f"  P90 total       : ${est['p90_total_usd']:,.2f}")
    print(f"  PLANNING total  : ${est['planning_total_usd']:,.2f}  (expected + 20% contingency)")
    if est["unpriced_cells"]:
        print(f"  unpriced cells  : {', '.join(est['unpriced_cells'])}")


if __name__ == "__main__":
    main()
