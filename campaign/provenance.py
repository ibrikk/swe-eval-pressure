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
6. SCOPE IS EXPLICIT.  An attempt says whether it is answerable for a whole
                       profile/shard cell (`full_cell`) or only for the cells a
                       repair plan listed (`repair`). Judging a repair against
                       the whole cell is what made a 103/103 repair record as
                       `failed`; see `repair_scope`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from campaign import cells, lib

ALLOWED_STATUS = ("complete", "failed", "aborted", "superseded")

# An attempt's SCOPE. `full_cell` is a run-shard invocation: it is answerable
# for every trial in the profile/shard cell. `repair` is a repair-shard
# invocation: it is answerable ONLY for the cells its source repair plan listed,
# and judging it against the whole 300-trial cell is the accounting bug this
# module used to have. See `repair_scope`.
ATTEMPT_KIND_FULL = "full_cell"
ATTEMPT_KIND_REPAIR = "repair"
ATTEMPT_KINDS = (ATTEMPT_KIND_FULL, ATTEMPT_KIND_REPAIR)

PLAN_BASIS_FILE = "repair_plan"
PLAN_BASIS_RECONSTRUCTED = "reconstructed_from_run_dirs"


def _paths():
    p = lib.campaign_paths()
    p["attempts"] = p["provenance"] / "attempts.jsonl"
    p["accepted"] = p["provenance"] / "accepted_runs.json"
    p["corpus"] = p["provenance"] / "corpus.jsonl"
    p["closures"] = p["provenance"] / "cell_closures.jsonl"
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
# repair scope
#
# THE ACCOUNTING BUG THIS SECTION EXISTS TO FIX
# ---------------------------------------------
# `campaign.sh record_attempts()` records run-shard and repair-shard attempts
# through the SAME `provenance record --status auto` call. The old "auto" rule
# was:
#
#     complete iff accepted == cell.expected_trials
#
# `cell.expected_trials` is the size of the whole profile/shard cell (300 for
# FULL shard 1). A repair invocation only ever produces the cells its plan
# listed, so a repair that executed 103/103 planned cells perfectly was compared
# against 300 and recorded as `failed`. That non-zero return propagated out of
# record_attempts into campaign.sh, which then died FATAL on a repair that had
# in fact succeeded.
#
# A repair attempt is answerable for its PLAN, not for the cell. The plan is
# therefore part of the attempt's provenance, and the completion rule reads it.
# --------------------------------------------------------------------------- #
def load_repair_plan(path) -> dict:
    """Read a frozen repair plan, refusing anything outside the campaign."""
    path = Path(path)
    lib.assert_campaign_path(path.parent, "repair plan directory")
    plan = lib.jload(path)
    if not isinstance(plan, dict):
        raise ValueError(f"unreadable repair plan: {path}")
    if plan.get("campaign_id") != lib.CAMPAIGN_ID:
        raise ValueError(
            f"repair plan {path} belongs to campaign {plan.get('campaign_id')!r}, "
            f"not {lib.CAMPAIGN_ID!r}")
    return plan


def planned_repair_cells(plan: dict, profile: str) -> list[str]:
    """Cell keys this plan asked THIS profile to repair. Order is stable."""
    return sorted(c["cell_key"] for c in (plan.get("by_profile") or {}).get(profile, []))


def reconstruct_repair_cells(audit_result: dict, profile: str,
                             run_dir_names) -> list[str]:
    """Planned cells inferred from what an attempt's run dirs actually touched.

    Used ONLY when the source plan file is unavailable -- notably when
    `repair-shard`'s own step (8) re-audit has already overwritten it with the
    post-repair plan. The result is labelled `reconstructed_from_run_dirs`
    everywhere it is recorded, never passed off as the frozen plan.
    """
    names = set(run_dir_names)
    out = []
    for key, rec in audit_result["records"].items():
        if rec.cell.profile != profile:
            continue
        if any(o.run_dir in names for o in rec.observations):
            out.append(key)
    return sorted(out)


def repair_scope(cell, planned: list[str], run_dir_names, *, audit_result,
                 plan_path=None, plan=None, basis=PLAN_BASIS_FILE) -> dict:
    """Judge a repair attempt against its OWN planned cells.

    Complete iff every planned cell now holds an accepted outcome that this
    attempt's run directories produced. A planned cell that came back
    PROVIDER_BLOCKED counts as accepted and is reported on its own line: a
    block is an explicit end-to-end stack outcome, and demanding it flip to
    COMPLETE_VALID would make the attempt closeable only by re-running a
    refusal until it complies.
    """
    names = set(run_dir_names)
    recs = audit_result["records"]
    completed, blocked, outstanding, unknown = [], [], [], []
    for key in planned:
        rec = recs.get(key)
        if rec is None:
            unknown.append(key)
            continue
        here = [o for o in rec.observations
                if o.run_dir in names and o.status in cells.ACCEPTED_STATUSES]
        if any(o.status == cells.COMPLETE_VALID for o in here):
            completed.append(key)
        elif here:
            blocked.append(key)
        else:
            outstanding.append(key)
    observed = len(completed) + len(blocked)
    return {
        "attempt_kind": ATTEMPT_KIND_REPAIR,
        "attempt_scope": "repair_subset",
        "repair_plan_basis": basis,
        "repair_plan": (str(Path(plan_path).resolve().relative_to(lib.PROJECT_ROOT))
                        if plan_path else None),
        "repair_plan_sha256": (lib.sha256_file(Path(plan_path))
                               if plan_path and Path(plan_path).is_file() else None),
        "repair_plan_generated_at": (plan or {}).get("generated_at"),
        # The four numbers requirement 3 asks for, stated separately from the
        # profile/shard cell size so neither can be mistaken for the other.
        "planned_repair_cells": len(planned),
        "expected_repair_cells": len(planned),
        "observed_repair_cells": observed,
        "planned_repair_cell_keys": planned,
        "completed_repair_cell_keys": completed,
        "provider_blocked_repair_cell_keys": blocked,
        "outstanding_repair_cell_keys": outstanding,
        "unplanned_or_unknown_cell_keys": unknown,
        # The profile/shard cell is still on record -- as context, never as the
        # yardstick this attempt is measured by.
        "profile_shard_expected_trials": cell.expected_trials,
        "repair_complete": bool(planned) and observed == len(planned)
                           and not outstanding and not unknown,
    }


# --------------------------------------------------------------------------- #
# record
# --------------------------------------------------------------------------- #
def cmd_record(args) -> int:
    paths = _paths()
    paths["provenance"].mkdir(parents=True, exist_ok=True)
    # One attempt may span SEVERAL Harbor invocations: the adaptive controller
    # splits a cell into batches so concurrency can be re-planned at batch
    # boundaries. They are all part of ONE attempt for ONE cell -- this is not a
    # merge across attempts and never a dedupe across task_name.
    raw_dirs = args.run_dir if isinstance(args.run_dir, list) else [args.run_dir]
    run_dirs = [Path(d) for d in raw_dirs if d]
    for d in run_dirs:
        lib.assert_campaign_path(d, "run directory")
    run_dir = run_dirs[0]

    cell = lib.Cell(args.mode, args.profile, args.shard)
    trials = []
    for d in run_dirs:
        if d.is_dir():
            trials.extend(lib.scan_run_dir(d))
    counts = {}
    for t in trials:
        counts[t.status] = counts.get(t.status, 0) + 1

    complete = counts.get(lib.STATUS_COMPLETE, 0)
    # A provider block is an EXPLICIT outcome, not an outstanding cell: the
    # request reached the vendor and the vendor's stack terminated it. Refusing
    # to accept the attempt would leave the shard permanently un-closeable and
    # push the operator toward re-running the block until it complies, which is
    # exactly the outcome-conditioned acceptance the campaign forbids. It is
    # counted here and reported separately everywhere downstream.
    blocked = counts.get(lib.STATUS_PROVIDER_BLOCKED, 0)
    accepted = complete + blocked

    # SCOPE FIRST, THEN STATUS. A repair attempt is judged against the cells its
    # plan listed; a run-shard attempt against the whole profile/shard cell.
    scope = None
    plan_path = getattr(args, "repair_plan", None)
    if plan_path:
        plan = load_repair_plan(plan_path)
        names = [d.name for d in run_dirs]
        audit_result = cells.audit(cell.mode, cell.shard_index, profiles=[cell.profile])
        planned = planned_repair_cells(plan, cell.profile)
        scope = repair_scope(cell, planned, names, audit_result=audit_result,
                             plan_path=plan_path, plan=plan, basis=PLAN_BASIS_FILE)

    status = args.status
    if status == "auto":
        if scope is not None:
            status = "complete" if scope["repair_complete"] else "failed"
        else:
            status = ("complete" if accepted == cell.expected_trials
                      and len(trials) == cell.expected_trials else "failed")

    meta = {}
    for d in run_dirs:
        meta = lib.jload(d / "run_metadata.json") or meta
    versions = sorted({t.agent_version for t in trials if t.agent_version})
    # Only trials where a model actually generated can testify to which model
    # ran; a blocked trial carries the safety layer's placeholder name.
    models = sorted({t.model_name for t in trials if t.model_name and t.model_started})

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
        "run_dirs": [str(d.resolve().relative_to(lib.PROJECT_ROOT)) for d in run_dirs],
        "model_id": meta.get("model") or (models[0] if len(models) == 1 else models),
        "agent": meta.get("agent"),
        "agent_version_requested": meta.get("agent_version_requested"),
        "agent_version_observed": versions[0] if len(versions) == 1 else versions,
        "started_at": args.started_at or meta.get("created_at"),
        "finished_at": args.finished_at or lib.now_iso(),
        # For a repair attempt this is the attempt's OWN scope, so
        # expected/observed are comparable within the record. The profile/shard
        # cell size travels alongside as `profile_shard_expected_trials`.
        "expected_trials": (scope["expected_repair_cells"] if scope
                            else cell.expected_trials),
        "observed_trials": len(trials),
        "accepted_observations": accepted,
        "model_observations": complete,
        "provider_blocked": blocked,
        "status_counts": counts,
        "status": status,
        "attempt_kind": ATTEMPT_KIND_REPAIR if scope else ATTEMPT_KIND_FULL,
        "superseded_by": None,
        "recorded_at": lib.now_iso(),
        "note": args.note or "",
    }
    if scope:
        entry.update(scope)
        # A repair COMPLEMENTS the attempt that left the cells outstanding; it
        # does not replace it. Naming it here is what keeps the two halves of a
        # repaired cell joinable without globbing.
        entry["complements_attempt_ids"] = sorted(
            e["attempt_id"] for e in existing
            if e["cell"] == cell.key and e["attempt_id"] != attempt_id)

    # A new complete FULL-CELL attempt supersedes any earlier full-cell attempt
    # for the same cell: it re-ran the whole cell, so the earlier one is
    # replaced wholesale. A repair supersedes NOTHING -- it covers a subset, and
    # marking the original superseded would discard the majority of the cell
    # that the original attempt is still the only evidence for.
    if status == "complete" and not scope:
        for e in existing:
            if (e["cell"] == cell.key and e["superseded_by"] is None
                    and e["attempt_id"] != attempt_id
                    and e.get("attempt_kind", ATTEMPT_KIND_FULL) == ATTEMPT_KIND_FULL):
                e["superseded_by"] = attempt_id
                if e["status"] == "complete":
                    e["status"] = "superseded"
        paths["attempts"].write_text("".join(json.dumps(e) + "\n" for e in existing))

    with open(paths["attempts"], "a") as fh:
        fh.write(json.dumps(entry) + "\n")

    _rebuild_accepted(paths)
    keys = ["cell", "attempt_id", "run_id", "attempt_kind", "status",
            "observed_trials", "expected_trials", "status_counts",
            "agent_version_observed"]
    if scope:
        keys[7:7] = ["planned_repair_cells", "observed_repair_cells",
                     "expected_repair_cells", "repair_plan", "repair_plan_basis",
                     "profile_shard_expected_trials"]
    print(json.dumps({k: entry[k] for k in keys}, indent=2))
    return 0 if status == "complete" else 1


def _is_repair(a: dict) -> bool:
    return a.get("attempt_kind") == ATTEMPT_KIND_REPAIR


def _rebuild_accepted(paths) -> dict:
    attempts = _load_attempts(paths["attempts"])
    live = [a for a in attempts if a["status"] == "complete" and a["superseded_by"] is None]
    # `accepted` means "this ONE attempt covers the whole profile/shard cell",
    # which is what campaign.sh, campaign.shard and campaign.validate all read
    # it as. A complete repair attempt covers only its planned subset, so it is
    # listed separately -- promoting it here would tell the shard validator to
    # expect 300 trials from a run that legitimately produced 103.
    accepted = [a for a in live if not _is_repair(a)]
    repairs = [a for a in live if _is_repair(a)]
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
        # Complete repair attempts, in the ledger and addressable, but never
        # mistaken for a whole-cell acceptance.
        "accepted_repair_attempts": repairs,
        "repair_cells_completed": sum(a.get("observed_repair_cells", 0) for a in repairs),
        "rejected": [a for a in attempts if a["status"] != "complete" or a["superseded_by"]],
        "conflicting_cells": sorted(set(conflicts)),
        "cells_complete": len(by_cell),
        "cells_expected": len(lib.all_cells()),
        "cell_closures": _load_attempts(paths["closures"]),
    }
    paths["accepted"].write_text(json.dumps(doc, indent=2) + "\n")
    return doc


# --------------------------------------------------------------------------- #
# reconcile
#
# Finalising a repair that ALREADY RAN, offline. No Harbor, no model call, no
# re-execution: it reads the cells on disk and corrects the ledger's bookkeeping
# for them. Two jobs:
#
#   1. Repair attempts mislabelled by the old whole-cell "auto" rule get an
#      explicit CORRECTION record -- a proper repair-attempt record naming the
#      attempt it corrects. The original stays in the ledger, marked superseded,
#      with its original numbers untouched. Nothing is deleted, nothing is
#      edited in place beyond the standard explicit supersede stamp.
#
#   2. When the cell-level audit shows a profile/shard cell fully accounted for
#      -- original valid observations + accepted provider blocks + valid repair
#      observations == every expected cell -- a CLOSURE record states that sum
#      and names the attempts that contributed each part.
#
# The completeness decision is the cell audit's, not this function's. Reconcile
# records it; it never declares it.
# --------------------------------------------------------------------------- #
def _legacy_repair_candidates(attempts, mode, shard, attempt_ids=None):
    """Attempts recorded by repair-shard before repair scope existed."""
    out = []
    for a in attempts:
        if a.get("mode") != mode or a.get("shard_index") != int(shard):
            continue
        if a.get("superseded_by"):
            continue
        if attempt_ids:
            if a.get("attempt_id") in attempt_ids:
                out.append(a)
            continue
        # Already scoped correctly -- nothing to correct.
        if a.get("attempt_kind") == ATTEMPT_KIND_REPAIR:
            continue
        if "repair-shard" in (a.get("note") or ""):
            out.append(a)
    return out


def _cell_closure(cell, audit_result, repair_run_dir_names, contributing) -> dict:
    """State the cell-level completeness sum for one profile/shard cell."""
    names = set(repair_run_dir_names)
    original_valid, repair_valid, blocked, outstanding = [], [], [], []
    for key, rec in audit_result["records"].items():
        if rec.cell.profile != cell.profile:
            continue
        if rec.status == cells.PROVIDER_BLOCKED:
            blocked.append(key)
        elif rec.status == cells.COMPLETE_VALID:
            valid = [o for o in rec.observations if o.status == cells.COMPLETE_VALID]
            if valid and valid[0].run_dir in names:
                repair_valid.append(key)
            else:
                original_valid.append(key)
        else:
            outstanding.append(key)
    total = len(original_valid) + len(blocked) + len(repair_valid)
    return {
        "campaign_id": lib.CAMPAIGN_ID,
        "record_kind": "cell_closure",
        "cell": cell.key,
        "mode": cell.mode,
        "profile": cell.profile,
        "shard": cell.shard_label,
        "shard_index": cell.shard_index,
        "basis": "campaign.cells audit (cell level)",
        # Requirement 4, spelled out as an arithmetic statement rather than a
        # verdict: the four terms and the total they must reach.
        "original_valid_observations": len(original_valid),
        "accepted_provider_blocked": len(blocked),
        "valid_repair_observations": len(repair_valid),
        "accounted_cells": total,
        "expected_cells": cell.expected_trials,
        "outstanding_cells": len(outstanding),
        "outstanding_cell_keys": sorted(outstanding)[:50],
        "complete": total == cell.expected_trials and not outstanding,
        "contributing_attempt_ids": sorted(contributing),
        "closed_at": lib.now_iso(),
    }


def cmd_reconcile(args) -> int:
    paths = _paths()
    paths["provenance"].mkdir(parents=True, exist_ok=True)
    mode, shard = args.mode, int(args.shard)
    apply = bool(args.apply)
    profiles = list(getattr(args, "profile", None) or lib.PROFILES)

    audit_result = cells.audit(mode, shard, profiles=profiles)
    v = cells.validate_shard_complete(audit_result)
    attempts = _load_attempts(paths["attempts"])
    targets = [a for a in _legacy_repair_candidates(attempts, mode, shard,
                                                   set(args.attempt_id or []))
               if a.get("profile") in profiles]

    print(f"reconcile {mode} shard {shard}   ({'APPLY' if apply else 'DRY RUN'})")
    print("  cell-level audit (authoritative)")
    print(f"    expected cells                     : {v['expected']}")
    print(f"    accepted observations              : {v['accepted_observations']}")
    print(f"      model observations               : {v['model_observations']}")
    print(f"      provider blocked                 : {v['provider_blocked']}")
    print(f"    missing                            : {v['missing']}")
    print(f"    outstanding                        : {v['outstanding_total']}")
    for prob in v["problems"]:
        print(f"    ! {prob}")
    print(f"  repair attempts to correct           : {len(targets)}")
    if not targets:
        print("    (none: every repair attempt already carries repair scope)")

    corrections, closures = [], []
    seq = {}
    for a in attempts:
        seq[a["cell"]] = seq.get(a["cell"], 0) + 1

    for old in targets:
        cell = lib.Cell(old["mode"], old["profile"], old["shard_index"])
        run_names = [Path(d).name for d in (old.get("run_dirs") or [old["run_dir"]])]

        # The frozen plan first; the run dirs only if it is gone. repair-shard's
        # own step (8) re-audit overwrites the plan file in place, so for a
        # repair that has already finished the frozen copy is usually absent --
        # which is exactly why the executed plan is now snapshotted before
        # launch. The basis is recorded either way; it is never guessed at.
        plan_path = args.source_repair_plan or cells.repair_plan_path(mode, shard, paths=lib.campaign_paths())
        plan, planned, basis = None, [], PLAN_BASIS_RECONSTRUCTED
        try:
            plan = load_repair_plan(plan_path)
            planned = planned_repair_cells(plan, cell.profile)
            if planned:
                basis = PLAN_BASIS_FILE
        except (ValueError, OSError):
            plan = None
        if not planned:
            plan, basis = None, PLAN_BASIS_RECONSTRUCTED
            planned = reconstruct_repair_cells(audit_result, cell.profile, run_names)

        scope = repair_scope(cell, planned, run_names, audit_result=audit_result,
                             plan_path=(plan_path if basis == PLAN_BASIS_FILE else None),
                             plan=plan, basis=basis)
        seq[cell.key] = seq.get(cell.key, 0) + 1
        new_id = f"{cell.mode}-{cell.profile}-s{cell.shard_index}-a{seq[cell.key]:02d}"
        status = "complete" if scope["repair_complete"] else "failed"

        entry = {k: old[k] for k in old if k not in
                 ("attempt_id", "status", "superseded_by", "recorded_at", "note",
                  "expected_trials")}
        entry.update(scope)
        entry.update({
            "attempt_id": new_id,
            "expected_trials": scope["expected_repair_cells"],
            "status": status,
            "superseded_by": None,
            "recorded_at": lib.now_iso(),
            "record_kind": "repair_attempt_correction",
            "corrects_attempt_id": old["attempt_id"],
            "corrects_bookkeeping": {
                "expected_trials": old.get("expected_trials"),
                "observed_trials": old.get("observed_trials"),
                "status": old.get("status"),
            },
            "correction_reason": (
                "recorded by campaign.sh repair-shard through the whole-cell "
                "'auto' rule, which compared this repair attempt's "
                f"{old.get('observed_trials')} repair cells against the entire "
                f"profile/shard expected_trials={old.get('expected_trials')} and "
                "marked a successful repair 'failed'. The repair executed its "
                "planned cells; only the bookkeeping was wrong."),
            "note": (f"correction of {old['attempt_id']} by "
                     f"`campaign.provenance reconcile --mode {mode} --shard {shard}`; "
                     "no cell was re-run"),
        })
        corrections.append((old, entry))

        print(f"    {old['attempt_id']} -> {new_id}  {cell.profile:<7} "
              f"planned={scope['planned_repair_cells']} "
              f"observed={scope['observed_repair_cells']} "
              f"expected={scope['expected_repair_cells']} "
              f"basis={basis} status={status}")
        if scope["outstanding_repair_cell_keys"]:
            print(f"      ! {len(scope['outstanding_repair_cell_keys'])} planned cell(s) "
                  "did not complete in this attempt")

    # Closures, per profile/shard cell, from the cell audit.
    all_repair_names = set()
    for old, _new in corrections:
        all_repair_names.update(Path(d).name for d in (old.get("run_dirs") or [old["run_dir"]]))
    for a in attempts:
        if _is_repair(a) and a.get("mode") == mode and a.get("shard_index") == shard:
            all_repair_names.update(Path(d).name for d in (a.get("run_dirs") or [a["run_dir"]]))

    print("  cell closures")
    for profile in profiles:
        cell = lib.Cell(mode, profile, shard)
        contributing = [e["attempt_id"] for _o, e in corrections if _o["cell"] == cell.key]
        contributing += [a["attempt_id"] for a in attempts if a["cell"] == cell.key]
        c = _cell_closure(cell, audit_result, all_repair_names, set(contributing))
        closures.append(c)
        print(f"    {profile:<7} original_valid={c['original_valid_observations']:>4} "
              f"+ blocked={c['accepted_provider_blocked']:>3} "
              f"+ repair_valid={c['valid_repair_observations']:>4} "
              f"= {c['accounted_cells']:>4} / {c['expected_cells']} "
              f"{'OK' if c['complete'] else 'INCOMPLETE'}")

    total = sum(c["accounted_cells"] for c in closures)
    expected = sum(c["expected_cells"] for c in closures)
    print(f"    {'TOTAL':<7} {total} / {expected}")

    ok = (not v["problems"]
          and all(c["complete"] for c in closures)
          and all(e["status"] == "complete" for _o, e in corrections))

    if not apply:
        print("  DRY RUN: nothing written. Re-run with --apply to record.")
        return 0 if ok else 1

    # Append-only. The originals keep their numbers; they gain the standard
    # explicit supersede stamp and a reason, and stay in the ledger as evidence.
    by_id = {c[0]["attempt_id"]: c[1] for c in corrections}
    for a in attempts:
        new = by_id.get(a["attempt_id"])
        if new is not None:
            a["superseded_by"] = new["attempt_id"]
            a["status"] = "superseded"
            a["superseded_reason"] = new["correction_reason"]
    paths["attempts"].write_text("".join(json.dumps(a) + "\n" for a in attempts))
    with open(paths["attempts"], "a") as fh:
        for _old, new in corrections:
            fh.write(json.dumps(new) + "\n")
    with open(paths["closures"], "a") as fh:
        for c in closures:
            fh.write(json.dumps(c) + "\n")
    _rebuild_accepted(paths)
    print(f"  wrote {len(corrections)} correction record(s) and {len(closures)} closure(s)")
    print(f"    {paths['attempts'].relative_to(lib.PROJECT_ROOT)}")
    print(f"    {paths['closures'].relative_to(lib.PROJECT_ROOT)}")
    return 0 if ok else 1


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
        attempt_dirs = [lib.PROJECT_ROOT / d for d in (a.get("run_dirs") or [a["run_dir"]])]
        bad = False
        for d in attempt_dirs:
            try:
                lib.assert_campaign_path(d, "accepted run directory")
            except ValueError as exc:
                errors.append(str(exc)); bad = True
        if bad:
            continue
        scanned = []
        for d in attempt_dirs:
            scanned.extend(lib.scan_run_dir(d))
        for t in scanned:
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
                # Design coordinates travel with every row, blocked ones
                # included, so a blocked cell keeps its address in the design.
                **lib.arm_factors(a["mode"], t.arm),
                # The analysis gate. Rows with model_started False carry no
                # model behaviour and are excluded from every behavioural
                # statistic; see campaign.analyze.model_rows.
                "model_started": t.model_started,
                "provider_refusal": t.provider_refusal,
                "provider_refusal_category": t.provider_refusal_category,
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
    blocked_rows = [r for r in rows if r["status"] == lib.STATUS_PROVIDER_BLOCKED]
    model_rows = [r for r in rows if r["model_started"]]
    out = {
        "campaign_id": lib.CAMPAIGN_ID, "built_at": lib.now_iso(),
        "cells_complete": doc["cells_complete"], "cells_expected": doc["cells_expected"],
        "trials": 0 if errors else len(rows), "rows_scanned": len(rows),
        # Reported separately, always. `accepted_observations` says the corpus
        # is rectangular; `model_observations` says how much of it is model
        # behaviour. Never quote the first where the second is meant.
        "accepted_observations": 0 if errors else len(rows),
        "model_observations": 0 if errors else len(model_rows),
        "provider_blocked": 0 if errors else len(blocked_rows),
        "provider_blocked_categories": sorted(
            {r["provider_refusal_category"] for r in blocked_rows}),
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
    r.add_argument("--run-dir", required=True, action="append")
    r.add_argument("--status", default="auto", choices=("auto",) + ALLOWED_STATUS)
    r.add_argument("--repair-plan",
                   help="source repair plan that scoped this attempt. Marks the "
                        "attempt as a REPAIR: --status auto then completes iff "
                        "every planned repair cell completed validly, instead of "
                        "comparing it against the whole profile/shard cell.")
    r.add_argument("--started-at")
    r.add_argument("--finished-at")
    r.add_argument("--note")
    r.set_defaults(fn=cmd_record)

    rc = sub.add_parser(
        "reconcile",
        help="offline: correct repair-attempt bookkeeping and close cells from "
             "the cell-level audit. Never runs a trial.")
    rc.add_argument("--mode", required=True, choices=lib.MODES)
    rc.add_argument("--shard", required=True, type=int, choices=lib.SHARD_INDICES)
    rc.add_argument("--profile", action="append", choices=lib.PROFILES,
                    help="restrict to these profiles (default: all four)")
    rc.add_argument("--attempt-id", action="append",
                    help="correct exactly these attempts (default: auto-detect "
                         "repair-shard attempts that predate repair scope)")
    rc.add_argument("--source-repair-plan",
                    help="the plan the repair executed against. If absent or "
                         "already overwritten, the scope is reconstructed from "
                         "the attempt's run dirs and labelled as such.")
    rc.add_argument("--apply", action="store_true",
                    help="write the records (default: dry run, writes nothing)")
    rc.set_defaults(fn=cmd_reconcile)

    b = sub.add_parser("build", help="rebuild the campaign corpus from accepted attempts")
    b.set_defaults(fn=cmd_build)

    args = ap.parse_args()
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
