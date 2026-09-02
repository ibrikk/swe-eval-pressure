#!/usr/bin/env python3
"""Attempt ledger and provenance builder.

WHAT WENT WRONG BEFORE
----------------------
`audits/claude_opus5/scripts/01_build_inventory.py` hard-codes a RUN_DIRS list
that mixes a good shard-1 run, a partial shard-2 run, a budget-failed shard-2
run, an archived abort and a historical repair run - and then dedupes by
`task_name`. Because the failed attempt and the good attempt produce directories
with the same task name, whichever one the loop saw last silently won. A failed
trial could therefore be promoted into the "primary corpus" with no trace.

THE FIX, STRUCTURALLY
---------------------
1. NO GLOBBING.        Runs enter the corpus only via an explicit, append-only
                       ledger (`provenance/attempts.jsonl`).
2. NO PATHS OUTSIDE.   Every run dir is checked with `lib.assert_campaign_path`,
                       so a historical Aug 2026 directory cannot be admitted at all.
3. ONE WINNER PER CELL.Exactly one attempt per cell may carry status "complete".
                       A second is a hard error, not a dedupe.
4. NO DEDUPE BY NAME.  Within the accepted set, a repeated (cell, base_task, arm)
                       raises. Collisions are reported, never resolved silently.
5. SUPERSEDING IS EXPLICIT. A failed attempt is marked `superseded_by` the retry;
                       it stays in the ledger as evidence and stays out of the corpus.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from campaign import lib

ALLOWED_STATUS = ("complete", "failed", "aborted", "superseded")


def _paths():
    p = lib.campaign_paths()
    p["attempts"] = p["provenance"] / "attempts.jsonl"
    p["accepted"] = p["provenance"] / "accepted_runs.json"
    p["corpus"] = p["provenance"] / "corpus.jsonl"
    return p


def _load_attempts(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


# --------------------------------------------------------------------------- #
# record
# --------------------------------------------------------------------------- #
def cmd_record(args) -> int:
    paths = _paths()
    paths["provenance"].mkdir(parents=True, exist_ok=True)
    run_dir = Path(args.run_dir)
    lib.assert_campaign_path(run_dir, "run directory")

    cell = lib.Cell(args.mode, args.profile, args.shard)
    trials = lib.scan_run_dir(run_dir) if run_dir.is_dir() else []
    counts = {}
    for t in trials:
        counts[t.status] = counts.get(t.status, 0) + 1

    complete = counts.get(lib.STATUS_COMPLETE, 0)
    status = args.status
    if status == "auto":
        status = "complete" if complete == cell.expected_trials and len(trials) == cell.expected_trials else "failed"

    meta = lib.jload(run_dir / "run_metadata.json") or {}
    versions = sorted({t.agent_version for t in trials if t.agent_version})
    models = sorted({t.model_name for t in trials if t.model_name})

    existing = _load_attempts(paths["attempts"])
    attempt_id = f"{cell.mode}-{cell.profile}-s{cell.shard_index}-a{sum(1 for e in existing if e['cell'] == cell.key) + 1:02d}"

    entry = {
        "campaign_id": lib.CAMPAIGN_ID,
        "cell": cell.key,
        "mode": cell.mode,
        "profile": cell.profile,
        "shard": cell.shard_label,
        "shard_index": cell.shard_index,
        "attempt_id": attempt_id,
        "run_id": run_dir.name,
        "run_dir": str(run_dir.resolve().relative_to(lib.PROJECT_ROOT)),
        "model_id": meta.get("model") or (models[0] if len(models) == 1 else models),
        "agent": meta.get("agent"),
        "agent_version_requested": meta.get("agent_version_requested"),
        "agent_version_observed": versions[0] if len(versions) == 1 else versions,
        "started_at": args.started_at or meta.get("created_at"),
        "finished_at": args.finished_at or lib.now_iso(),
        "expected_trials": cell.expected_trials,
        "observed_trials": len(trials),
        "status_counts": counts,
        "status": status,
        "superseded_by": None,
        "recorded_at": lib.now_iso(),
        "note": args.note or "",
    }

    # A new complete attempt supersedes any earlier attempt for the same cell.
    if status == "complete":
        for e in existing:
            if e["cell"] == cell.key and e["superseded_by"] is None and e["attempt_id"] != attempt_id:
                e["superseded_by"] = attempt_id
                if e["status"] == "complete":
                    e["status"] = "superseded"
        paths["attempts"].write_text("".join(json.dumps(e) + "\n" for e in existing))

    with open(paths["attempts"], "a") as fh:
        fh.write(json.dumps(entry) + "\n")

    _rebuild_accepted(paths)
    print(json.dumps({k: entry[k] for k in
                      ("cell", "attempt_id", "run_id", "status", "observed_trials",
                       "expected_trials", "status_counts", "agent_version_observed")}, indent=2))
    return 0 if status == "complete" else 1


def _rebuild_accepted(paths) -> dict:
    attempts = _load_attempts(paths["attempts"])
    accepted = [a for a in attempts if a["status"] == "complete" and a["superseded_by"] is None]
    by_cell = {}
    conflicts = []
    for a in accepted:
        if a["cell"] in by_cell:
            conflicts.append(a["cell"])
        by_cell[a["cell"]] = a
    doc = {
        "campaign_id": lib.CAMPAIGN_ID,
        "rebuilt_at": lib.now_iso(),
        "accepted": accepted,
        "rejected": [a for a in attempts if a["status"] != "complete" or a["superseded_by"]],
        "conflicting_cells": sorted(set(conflicts)),
        "cells_complete": len(by_cell),
        "cells_expected": len(lib.all_cells()),
    }
    paths["accepted"].write_text(json.dumps(doc, indent=2) + "\n")
    return doc


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def cmd_build(args) -> int:
    paths = _paths()
    if not paths["accepted"].is_file():
        print("no accepted runs; nothing to build", file=sys.stderr)
        return 1
    doc = _rebuild_accepted(paths)

    errors = []
    if doc["conflicting_cells"]:
        errors.append(f"multiple accepted attempts for cells: {doc['conflicting_cells']} "
                      "- resolve explicitly, this tool will not pick one for you")

    rows, seen = [], {}
    for a in doc["accepted"]:
        run_dir = lib.PROJECT_ROOT / a["run_dir"]
        try:
            lib.assert_campaign_path(run_dir, "accepted run directory")
        except ValueError as exc:
            errors.append(str(exc))
            continue
        for t in lib.scan_run_dir(run_dir):
            key = (a["cell"], t.base_task_id, t.arm)
            if key in seen:
                errors.append(
                    f"duplicate cell {key} appears in both {seen[key]} and {a['attempt_id']} "
                    "- NOT deduping; fix the ledger")
                continue
            seen[key] = a["attempt_id"]
            rows.append({
                "campaign_id": lib.CAMPAIGN_ID,
                "cell": a["cell"], "mode": a["mode"], "profile": a["profile"],
                "shard": a["shard"], "attempt_id": a["attempt_id"], "run_id": a["run_id"],
                "base_task_id": t.base_task_id, "arm": t.arm,
                "trial_dir": t.trial_dir, "status": t.status,
                "agent_name": t.agent_name, "agent_version": t.agent_version,
                "model_name": t.model_name, "cost_usd": t.cost_usd,
                "prompt_tokens": t.prompt_tokens, "completion_tokens": t.completion_tokens,
                "steps": t.steps,
                "reward": t.reward, "resolved": t.resolved,
                "shard_index": a["shard_index"],
            })

    # FAIL CLOSED. A duplicate is skipped rather than deduped, which can leave a
    # coincidentally complete-looking corpus. Writing it anyway would let the
    # validator pass on a corpus the builder already knows is wrong - exactly the
    # Aug 2026 failure mode one layer up. So: no corpus at all when errors exist,
    # and any stale corpus from a previous build is removed.
    if errors:
        if paths["corpus"].exists():
            paths["corpus"].unlink()
    else:
        paths["corpus"].write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = {
        "campaign_id": lib.CAMPAIGN_ID, "built_at": lib.now_iso(),
        "cells_complete": doc["cells_complete"], "cells_expected": doc["cells_expected"],
        "trials": 0 if errors else len(rows), "rows_scanned": len(rows),
        "errors": errors, "ok": not errors,
        "corpus": str(paths["corpus"].relative_to(lib.PROJECT_ROOT)),
    }
    (paths["provenance"] / "build_report.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    for e in errors:
        print(f"ERROR: {e}", file=sys.stderr)
    return 1 if errors else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("record", help="record one execution attempt for one cell")
    r.add_argument("--mode", required=True, choices=lib.MODES)
    r.add_argument("--profile", required=True, choices=lib.PROFILES)
    r.add_argument("--shard", required=True, type=int, choices=lib.SHARD_INDICES)
    r.add_argument("--run-dir", required=True)
    r.add_argument("--status", default="auto", choices=("auto",) + ALLOWED_STATUS)
    r.add_argument("--started-at")
    r.add_argument("--finished-at")
    r.add_argument("--note")
    r.set_defaults(fn=cmd_record)

    b = sub.add_parser("build", help="rebuild the campaign corpus from accepted attempts")
    b.set_defaults(fn=cmd_build)

    args = ap.parse_args()
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
