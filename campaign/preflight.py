#!/usr/bin/env python3
"""Fail-closed preflight. Runs before the campaign and again before EVERY shard.

Checks, in order, and exits non-zero on the first failure:

  1. namespace     - campaign dirs + manifests exist and are for THIS campaign id
  2. integrity     - every task dir hash in the cell manifest still matches disk
  3. versions      - all four stacks explicitly pinned and present in the env
  4. models        - the pinned model id is exposed by the gateway
  5. budget        - remaining budget covers the REST OF THE WHOLE CAMPAIGN,
                     not just the next shard

Rule 5 is the one that matters. The Aug 2026 study died because shard 2 was
launched with enough budget to start and not enough to finish, producing 251
synthetic and hundreds of budget-censored trials that then had to be quarantined.
This refuses to start a shard unless the campaign as a whole can still complete.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from campaign import lib, cost_model

# Fraction of headroom demanded on top of the remaining planning cost.
SAFETY_MARGIN = 0.10


class Fail(Exception):
    pass


def check_namespace(paths) -> list[str]:
    notes = []
    if not paths["root"].is_dir():
        raise Fail(f"campaign namespace missing: {paths['root']} (run `./campaign.sh prepare`)")
    idx = paths["manifests"] / "cells_index.json"
    if not idx.is_file():
        raise Fail(f"cells index missing: {idx} (run `./campaign.sh prepare`)")
    data = json.loads(idx.read_text())
    if data.get("campaign_id") != lib.CAMPAIGN_ID:
        raise Fail(f"cells index is for campaign {data.get('campaign_id')!r}, expected {lib.CAMPAIGN_ID!r}")
    if len(data["cells"]) != len(lib.all_cells()):
        raise Fail(f"cells index has {len(data['cells'])} cells, expected {len(lib.all_cells())}")
    notes.append(f"namespace OK: {len(data['cells'])} cells, "
                 f"{sum(c['expected_trials'] for c in data['cells'])} trials")
    return notes


def check_integrity(paths, cells: list[lib.Cell], quick: bool) -> list[str]:
    notes = []
    for cell in cells:
        mf = paths["manifests"] / "cells" / f"{cell.mode}__{cell.profile}__{cell.shard_label}.json"
        if not mf.is_file():
            raise Fail(f"missing cell manifest: {mf}")
        cm = json.loads(mf.read_text())
        if cm["campaign_id"] != lib.CAMPAIGN_ID:
            raise Fail(f"{mf.name}: wrong campaign_id {cm['campaign_id']!r}")
        shard_dir = lib.PROJECT_ROOT / cm["dataset_path"]
        if not shard_dir.is_dir():
            raise Fail(f"{cell.key}: dataset snapshot missing at {shard_dir}")
        got = lib.sha256_file(shard_dir / "manifest.json")
        if got != cm["dataset_manifest_sha256"]:
            raise Fail(f"{cell.key}: dataset manifest changed since prepare\n"
                       f"  expected {cm['dataset_manifest_sha256']}\n  found    {got}")
        rows = cm["trials"]
        subset = rows if not quick else rows[:: max(1, len(rows) // 10)]
        for r in subset:
            td = lib.PROJECT_ROOT / r["snapshot_task_path"]
            if not td.is_dir():
                raise Fail(f"{cell.key}: task dir missing: {r['task_dir']}")
            if lib.sha256_tree(td) != r["task_content_sha256"]:
                raise Fail(f"{cell.key}: task content changed: {r['task_dir']}")
        notes.append(f"integrity OK: {cell.key} ({len(subset)}/{len(rows)} task dirs hashed)")
    return notes


def check_versions(profiles) -> list[str]:
    notes = []
    for p in profiles:
        pin = lib.VERSION_PINS[p]
        env_val = os.environ.get(pin["pin_env"], "")
        if not env_val:
            raise Fail(
                f"{p}: {pin['pin_env']} is not set. Every stack must be explicitly "
                f"pinned before the campaign may run. Expected {pin['version']!r}."
            )
        if env_val != pin["version"]:
            raise Fail(f"{p}: {pin['pin_env']}={env_val!r} but campaign pins {pin['version']!r}")
        notes.append(f"version OK: {p} -> {pin['agent']} {pin['version']} (via {pin['pin_env']})")
    return notes


def check_models(profiles, key: str) -> list[str]:
    notes = []
    base = lib.LITELLM_ANTHROPIC_BASE
    try:
        proc = subprocess.run(
            ["curl", "-sS", "-m", "30", "-H", f"Authorization: Bearer {key}", f"{base}/v1/models"],
            capture_output=True, text=True, timeout=45)
        ids = {m.get("id", "") for m in json.loads(proc.stdout).get("data", [])}
    except Exception as exc:
        raise Fail(f"could not list gateway models: {exc}")
    for p in profiles:
        model = lib.MODEL_PINS[p]
        # litellm strips a leading `openai/` provider prefix before dispatch, so
        # accept either the full id or the stripped wire name.
        wire = model.split("/", 1)[1] if model.startswith("openai/") else model
        if model not in ids and wire not in ids:
            raise Fail(f"{p}: neither {model!r} nor {wire!r} is exposed by the gateway")
        notes.append(f"model OK: {p} -> {model}")
    return notes


def remaining_campaign_cost(paths, est: dict) -> tuple[float, list[str]]:
    """Planning cost of cells that are not yet validated-complete."""
    done = set()
    state = paths["provenance"] / "accepted_runs.json"
    if state.is_file():
        for entry in json.loads(state.read_text()).get("accepted", []):
            if entry.get("status") == "complete":
                done.add(entry["cell"])
    total, pending = 0.0, []
    for cell in lib.all_cells():
        if cell.key in done:
            continue
        c = est["cells"][f"{cell.mode}/{cell.profile}"]
        per_trial = c["mean_usd_per_trial"] or 0.0
        total += per_trial * cell.expected_trials * 1.20  # same contingency as planning
        pending.append(cell.key)
    return total, pending


def check_budget(paths, est: dict, key: str | None) -> list[str]:
    status = lib.probe_budget(key)
    if not status.ok:
        raise Fail(f"budget probe failed: {status.error} (key {status.key_fingerprint})")
    need, pending = remaining_campaign_cost(paths, est)
    required = need * (1 + SAFETY_MARGIN)
    notes = [
        f"key             : {status.key_fingerprint}",
        f"max budget      : ${status.max_budget:,.2f}",
        f"spend           : ${status.spend:,.2f}",
        f"remaining       : ${status.remaining:,.2f}",
        f"tpm/rpm         : {status.tpm_limit}/{status.rpm_limit}",
        f"cells pending   : {len(pending)}",
        f"planning cost   : ${need:,.2f}  (remaining cells, incl. 20% contingency)",
        f"required        : ${required:,.2f}  (+{int(SAFETY_MARGIN*100)}% safety margin)",
    ]
    if status.remaining < required:
        raise Fail(
            "INSUFFICIENT BUDGET - refusing to launch.\n  " + "\n  ".join(notes) +
            f"\n  short by ${required - status.remaining:,.2f}.\n"
            "  Raise the key budget or reduce campaign scope; do NOT start a partial run."
        )
    p90 = est["p90_total_usd"]
    if status.remaining < p90:
        notes.append(
            f"WARNING: the p90 (pessimistic) whole-campaign cost is ${p90:,.2f}, which "
            f"exceeds remaining budget ${status.remaining:,.2f}. The expected-cost plan "
            "fits, but a heavy-tail run will not. Re-run preflight between shards."
        )
    return notes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=lib.MODES)
    ap.add_argument("--profile", choices=lib.PROFILES)
    ap.add_argument("--shard", type=int, choices=lib.SHARD_INDICES)
    ap.add_argument("--quick", action="store_true",
                    help="hash a 10%% sample of task dirs instead of all (per-shard use)")
    ap.add_argument("--skip-budget", action="store_true", help="offline structural checks only")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    paths = lib.campaign_paths()
    cells = lib.all_cells()
    if args.mode:
        cells = [c for c in cells if c.mode == args.mode]
    if args.profile:
        cells = [c for c in cells if c.profile == args.profile]
    if args.shard:
        cells = [c for c in cells if c.shard_index == args.shard]
    if not cells:
        print("no cells selected", file=sys.stderr)
        sys.exit(2)
    profiles = sorted({c.profile for c in cells})
    key = os.environ.get("LITE_LLM_KEY") or os.environ.get("ANTHROPIC_API_KEY") or ""

    notes, failures = [], []
    try:
        notes += check_namespace(paths)
        notes += check_integrity(paths, cells, args.quick)
        notes += check_versions(profiles)
        if not args.skip_budget:
            notes += check_models(profiles, key)
            est = cost_model.estimate()
            notes += check_budget(paths, est, key)
    except Fail as exc:
        failures.append(str(exc))

    result = {"campaign_id": lib.CAMPAIGN_ID, "checked_at": lib.now_iso(),
              "cells": [c.key for c in cells], "notes": notes,
              "failures": failures, "ok": not failures}
    log = paths["logs"] / "preflight.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as fh:
        fh.write(json.dumps(result) + "\n")

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for n in notes:
            print(f"  {n}")
        for f in failures:
            print(f"\nPREFLIGHT FAILED:\n  {f}", file=sys.stderr)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
