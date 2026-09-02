#!/usr/bin/env python3
"""Single-shard execution planning for Campaign V2.

`./campaign.sh run-shard <id> <mode> <shard>` executes exactly one
(mode, shard) slice -- all four profiles -- and then stops. This module is the
fail-closed gate that runs BEFORE anything is launched. It answers one
question: "is it legitimate to execute this shard right now?"

It changes no rule. It only refuses earlier, and more cheaply, than the
existing preflight/validation gates would:

  * an unknown mode or a shard index outside 1..3 is rejected outright;
  * the prepared dataset and frozen cell manifest must exist for all four
    profiles, so a shard can never run against a half-prepared namespace;
  * a shard whose cells are ALREADY ACCEPTED in the attempt ledger is refused,
    because re-running it would supersede accepted trajectories. Overriding
    that requires the explicit `--new-attempt` workflow, which is recorded.

Nothing here launches Harbor, calls a model, or touches a historical result
directory. Planning is pure filesystem inspection.
"""
from __future__ import annotations

import argparse
import json
import sys

from campaign import lib


class ShardPlanError(Exception):
    """Refusal to plan a shard. Carries the operator-facing reason."""

    def __init__(self, message: str, *, detail: list[str] | None = None):
        super().__init__(message)
        self.detail = detail or []


# --------------------------------------------------------------------------- #
def _check_mode(mode: str) -> str:
    if mode not in lib.MODES:
        raise ShardPlanError(
            f"unknown mode {mode!r}",
            detail=[f"valid modes: {', '.join(lib.MODES)}"],
        )
    return mode


def _check_shard(shard) -> int:
    try:
        idx = int(str(shard).strip())
    except (TypeError, ValueError):
        raise ShardPlanError(
            f"shard index {shard!r} is not an integer",
            detail=[f"valid shard indices: {', '.join(str(i) for i in lib.SHARD_INDICES)}"],
        ) from None
    if idx not in lib.SHARD_INDICES:
        raise ShardPlanError(
            f"shard index {idx} is out of range",
            detail=[f"valid shard indices: {', '.join(str(i) for i in lib.SHARD_INDICES)}"],
        )
    return idx


def accepted_cells() -> dict[str, dict]:
    """Cell key -> accepted attempt, from the provenance ledger."""
    paths = lib.campaign_paths()
    doc = lib.jload(paths["provenance"] / "accepted_runs.json") or {}
    out = {}
    for a in doc.get("accepted", []):
        out[a["cell"]] = a
    return out


# --------------------------------------------------------------------------- #
def plan_shard(mode: str, shard, *, new_attempt: bool = False,
               require_datasets: bool = True) -> dict:
    """Validate one (mode, shard) slice and return its execution plan.

    Raises ShardPlanError if the shard must not be executed.
    """
    mode = _check_mode(mode)
    idx = _check_shard(shard)

    paths = lib.campaign_paths()
    already = accepted_cells()
    cells, missing, accepted_here = [], [], []

    for profile in lib.PROFILES:
        cell = lib.Cell(mode, profile, idx)
        dataset = paths["datasets"] / "_shards" / mode / profile / cell.shard_label
        manifest = paths["manifests"] / "cells" / f"{mode}__{profile}__{cell.shard_label}.json"
        prior = already.get(cell.key)
        if prior:
            accepted_here.append((cell.key, prior.get("attempt_id", "?")))
        if require_datasets:
            if not dataset.is_dir():
                missing.append(f"dataset  {dataset}")
            if not manifest.is_file():
                missing.append(f"manifest {manifest}")
        cells.append({
            "key": cell.key,
            "mode": mode,
            "profile": profile,
            "shard_index": idx,
            "shard_label": cell.shard_label,
            "dataset": str(dataset),
            "manifest": str(manifest),
            "expected_trials": cell.expected_trials,
            "already_accepted": bool(prior),
            "prior_attempt_id": prior.get("attempt_id") if prior else None,
        })

    if missing:
        raise ShardPlanError(
            f"{mode} shard {idx} is not prepared",
            detail=missing + [f"run: ./campaign.sh prepare {lib.CAMPAIGN_ID}"],
        )

    if accepted_here and not new_attempt:
        raise ShardPlanError(
            f"{mode} shard {idx} already has accepted attempts",
            detail=[f"{k} accepted as {a}" for k, a in accepted_here] + [
                "Re-running would supersede accepted trajectories.",
                "If that is genuinely intended, repeat the command with --new-attempt;",
                "the superseding attempt is then recorded explicitly in the ledger",
                "and the superseded one is preserved, never deleted.",
            ],
        )

    return {
        "campaign_id": lib.CAMPAIGN_ID,
        "mode": mode,
        "shard_index": idx,
        "shard_label": lib.Cell(mode, lib.PROFILES[0], idx).shard_label,
        "profiles": list(lib.PROFILES),
        "cells": cells,
        "expected_trials": sum(c["expected_trials"] for c in cells),
        "base_tasks": lib.BASE_TASKS_PER_SHARD[idx],
        "arms": len(lib.ARMS[mode]),
        "new_attempt": bool(new_attempt),
        "supersedes": [k for k, _ in accepted_here],
        "planned_at": lib.now_iso(),
    }


# --------------------------------------------------------------------------- #
def cmd_plan(args) -> int:
    try:
        plan = plan_shard(args.mode, args.shard, new_attempt=args.new_attempt,
                          require_datasets=not args.skip_dataset_check)
    except ShardPlanError as exc:
        lib.eprint(f"REFUSED: {exc}")
        for line in exc.detail:
            lib.eprint(f"         {line}")
        return 2

    if args.json:
        print(json.dumps(plan, indent=2))
    else:
        print(f"  campaign : {plan['campaign_id']}")
        print(f"  slice    : {plan['mode']} / {plan['shard_label']}")
        print(f"  profiles : {' '.join(plan['profiles'])}")
        print(f"  trials   : {plan['base_tasks']} base tasks x {plan['arms']} arms "
              f"x {len(plan['profiles'])} profiles = {plan['expected_trials']}")
        if plan["new_attempt"] and plan["supersedes"]:
            print(f"  NOTE     : new attempt, superseding {len(plan['supersedes'])} accepted cell(s)")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="validate and describe one shard's execution plan")
    p.add_argument("--mode", required=True)
    p.add_argument("--shard", required=True)
    p.add_argument("--new-attempt", action="store_true",
                   help="explicitly permit superseding already-accepted cells")
    p.add_argument("--skip-dataset-check", action="store_true",
                   help="plan without requiring the prepared dataset on disk")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_plan)
    args = ap.parse_args()
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
