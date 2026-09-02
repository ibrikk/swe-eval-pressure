#!/usr/bin/env python3
"""Explicit, append-only amendments to frozen Campaign V2 task definitions.

WHY THIS MODULE EXISTS
----------------------
Campaign V2 freezes every task definition at `prepare` time: the per-cell
manifest under `manifests/cells/` records a `task_content_sha256` and a
`seed_json_sha256` per trial plus the `dataset_manifest_sha256` of the shard,
and preflight re-verifies all of it before every launch. That immutability is
the whole integrity story and is not being weakened here.

On 2026-09-02 a legitimate exception appeared. `campaign.source_targets`
corrected the source-channel seed target of 24 task definitions whose target was
the POST-rename destination -- a path that does not exist in the pre-patch tree
the image is built from, so the image build died before any model ran. The
correction is objectively right and was applied to disk and recorded in the
append-only `provenance/source_target_repairs.jsonl`. But the frozen per-cell
manifests still describe the pre-correction bytes, so preflight now (correctly)
refuses to launch:

    full/claude/chunk-1-size-30: dataset manifest changed since prepare

The wrong ways out are `prepare --force` (re-freezes everything silently and
destroys the record of what the failed attempt ran) and editing the hash by
hand (indistinguishable from tampering). This module is the third way: record
the amendment explicitly, prove it is legitimate, and let preflight verify the
current state against original + approved amendments.

THE RULE PREFLIGHT ENFORCES AFTER THIS
--------------------------------------
A task definition on disk must equal EITHER

    A. the originally frozen definition, or
    B. the exact amended definition recorded in an approved, append-only
       amendment record

and nothing else. Arbitrary drift still fails, as it did before.

FAIL-CLOSED PRECONDITIONS FOR AN AMENDMENT
------------------------------------------
Every one of these must hold, per task-arm cell, or the whole operation is
refused (see `plan`):

  1. the drift matches an `applied` record in the append-only
     `provenance/source_target_repairs.jsonl`
  2. the cell holds NO COMPLETE_VALID accepted trajectory -- rewriting the task
     a real trajectory ran against would silently falsify its provenance
  3. the old hashes equal the hashes in the currently frozen cell manifest
  4. the new hashes equal what is on disk right now, and the on-disk
     `source_target` equals the approved correction
  5. no OTHER task definition in the same cell has drifted

THREE ARTIFACTS, EACH CHECKING THE OTHERS
-----------------------------------------
  * `provenance/task_definition_amendments.jsonl` - the append-only ledger
  * `provenance/frozen_manifests/<cell>.<sha12>.json` - the ORIGINAL cell
    manifest, byte-preserved, written once and never rewritten
  * `manifests/cells/<cell>.json` - the amended manifest, which now carries a
    `task_definition_amendments` stamp naming the ledger records it applied

`verify` reconstructs the amended manifest from the archived original plus the
ledger and requires it to equal what is on disk, field for field. Remove or
alter any one of the three and the reconstruction stops matching, so a manifest
re-frozen without provenance fails preflight instead of passing quietly.

Nothing here ever touches `datasets/_batches/`, which must keep describing what
the 2026-09-02 attempt actually ran, or any trajectory on disk.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

from campaign import lib

LEDGER_NAME = "task_definition_amendments.jsonl"
ARCHIVE_DIRNAME = "frozen_manifests"
SOURCE_REPAIR_LOG = "source_target_repairs.jsonl"

# The only change type this module knows how to approve. A new class of
# amendment must be added deliberately, with its own preconditions -- it must
# not arrive by widening this one.
SOURCE_TARGET_CORRECTION = "source_target_pre_patch_correction"
APPROVED_CHANGE_TYPES = (SOURCE_TARGET_CORRECTION,)

AMENDMENT_REASON = (
    "source-channel seed target pointed at the post-rename destination, which "
    "does not exist in the pre-patch tree the image is built from; the image "
    "build failed before any model executed"
)


# --------------------------------------------------------------------------- #
# paths and ledger IO
# --------------------------------------------------------------------------- #
def ledger_path(paths=None) -> Path:
    paths = paths or lib.campaign_paths()
    return paths["provenance"] / LEDGER_NAME


def archive_dir(paths=None) -> Path:
    paths = paths or lib.campaign_paths()
    return paths["provenance"] / ARCHIVE_DIRNAME


def cell_manifest_path(cell: lib.Cell, paths=None) -> Path:
    paths = paths or lib.campaign_paths()
    return (paths["manifests"] / "cells"
            / f"{cell.mode}__{cell.profile}__{cell.shard_label}.json")


def load_ledger(paths=None) -> list[dict]:
    """Every amendment record, in append order. Missing ledger -> []."""
    p = ledger_path(paths)
    if not p.is_file():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def ledger_by_cell(paths=None, ledger=None) -> dict[str, list[dict]]:
    """{cell manifest key 'mode/profile/shard_label' -> records}."""
    out: dict[str, list[dict]] = {}
    for rec in (ledger if ledger is not None else load_ledger(paths)):
        out.setdefault(rec["cell"], []).append(rec)
    return out


def load_source_repairs(paths=None) -> list[dict]:
    """The append-only source-target repair log, with 1-based line numbers."""
    paths = paths or lib.campaign_paths()
    p = paths["provenance"] / SOURCE_REPAIR_LOG
    if not p.is_file():
        return []
    out = []
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            rec = json.loads(line)
            rec["_line"] = i
            out.append(rec)
    return out


def amendment_id(cell_key: str, task_dir: str, original_sha: str,
                 amended_sha: str) -> str:
    """Deterministic id: same amendment computed twice yields the same id."""
    h = hashlib.sha256(
        "|".join((lib.CAMPAIGN_ID, cell_key, task_dir, original_sha,
                  amended_sha)).encode()).hexdigest()
    return f"amd-{h[:12]}"


# --------------------------------------------------------------------------- #
# the amended manifest is a pure function of (original, records)
# --------------------------------------------------------------------------- #
def _stamp(rec: dict) -> dict:
    """The subset of an amendment record that lives in the cell manifest."""
    return {k: rec[k] for k in (
        "amendment_id", "task_dir", "base_task_id", "arm", "amended_at",
        "approved_change_type", "reason",
        "old_source_target", "new_source_target",
        "original_task_content_sha256", "amended_task_content_sha256",
        "original_seed_json_sha256", "amended_seed_json_sha256",
        "prior_observation_status", "accepted_model_trajectory_exists",
        "source_repair_log_reference")}


def apply_to_manifest(original: dict, records: list[dict], *,
                      archive_ref: str, original_manifest_sha: str) -> dict:
    """The amended cell manifest implied by an original and its amendments.

    Deterministic and side-effect free: `apply` writes what this returns, and
    `verify` recomputes it from the archived original to check what is on disk.
    Rows with no amendment keep their frozen hashes byte for byte.
    """
    if not records:
        return copy.deepcopy(original)
    ds = {r["amended_dataset_manifest_sha256"] for r in records}
    if len(ds) != 1:
        raise ValueError(f"amendments for {original['cell']} disagree about the "
                         f"amended dataset manifest hash: {sorted(ds)}")
    cm = copy.deepcopy(original)
    by_dir = {r["task_dir"]: r for r in records}
    seen = set()
    for row in cm["trials"]:
        rec = by_dir.get(row["task_dir"])
        if rec is None:
            continue
        seen.add(row["task_dir"])
        row["task_content_sha256"] = rec["amended_task_content_sha256"]
        row["seed_json_sha256"] = rec["amended_seed_json_sha256"]
        # The design metadata the amendment actually changes, recorded on the
        # row itself so the manifest describes the definition it now expects.
        row["source_target"] = rec["new_source_target"]
        row["task_definition_amendment_id"] = rec["amendment_id"]
    missing = sorted(set(by_dir) - seen)
    if missing:
        raise ValueError(f"{original['cell']}: amendment names task dir(s) that "
                         f"are not in the manifest: {missing}")
    cm["original_dataset_manifest_sha256"] = original["dataset_manifest_sha256"]
    cm["dataset_manifest_sha256"] = ds.pop()
    cm["frozen_manifest_archive"] = archive_ref
    cm["original_cell_manifest_sha256"] = original_manifest_sha
    cm["task_definition_amendments"] = [
        _stamp(r) for r in sorted(records, key=lambda r: r["task_dir"])]
    return cm


# --------------------------------------------------------------------------- #
# planning: what may legitimately be amended, and what is refused
# --------------------------------------------------------------------------- #
def _disk_hashes(row: dict) -> tuple[str | None, str | None, str | None]:
    """(task_content_sha256, seed_json_sha256, source_target) on disk now."""
    td = lib.PROJECT_ROOT / row["snapshot_task_path"]
    if not td.is_dir():
        return None, None, None
    seed = td / "environment" / "benchmark_seed" / "seed.json"
    if not seed.is_file():
        return lib.sha256_tree(td), None, None
    try:
        target = json.loads(seed.read_text(encoding="utf-8")).get("source_target")
    except ValueError:
        target = None
    return lib.sha256_tree(td), lib.sha256_file(seed), target


def plan(mode: str | None = None, shard: int | None = None, *, paths=None,
         profiles=None) -> dict:
    """Classify every drifted task definition as approvable or refused.

    Read-only. Returns `approved` (amendable now), `already_amended` (the
    manifest already records this amendment and disk agrees) and `refused`
    (drift with no legitimate basis). `ok` is False if anything is refused --
    the caller must not amend a cell that carries unexplained drift.
    """
    paths = paths or lib.campaign_paths()
    profiles = list(profiles or lib.PROFILES)
    cells_sel = [c for c in lib.all_cells(profiles=profiles)
                 if (mode is None or c.mode == mode)
                 and (shard is None or c.shard_index == shard)]

    repairs = load_source_repairs(paths)
    repair_index = {(r["mode"], r["profile"], int(r["shard"]), r["task_dir"]): r
                    for r in repairs if r.get("applied")}
    by_cell = ledger_by_cell(paths)

    # `cells.audit` is the only authority on whether a trajectory exists. It is
    # imported lazily and cached per (mode, shard): it walks every run directory
    # of the shard, which is far too expensive to redo per task definition.
    from campaign import cells as cells_mod
    audits: dict[tuple[str, int], dict] = {}

    def audit_for(m: str, s: int) -> dict:
        if (m, s) not in audits:
            audits[(m, s)] = cells_mod.audit(m, s, profiles=profiles, paths=paths)
        return audits[(m, s)]

    approved: list[dict] = []
    already: list[dict] = []
    refused: list[dict] = []
    checked = 0

    for cell in cells_sel:
        mf = cell_manifest_path(cell, paths)
        if not mf.is_file():
            refused.append({"cell": cell.key, "task_dir": None,
                            "refusal": f"missing cell manifest: {mf.name}"})
            continue
        cm = json.loads(mf.read_text(encoding="utf-8"))
        shard_dir = lib.PROJECT_ROOT / cm["dataset_path"]
        disk_ds = (lib.sha256_file(shard_dir / "manifest.json")
                   if (shard_dir / "manifest.json").is_file() else None)
        prior = {r["task_dir"]: r for r in by_cell.get(cell.key, [])}
        cell_approved: list[dict] = []

        for row in cm["trials"]:
            checked += 1
            tc, sd, target = _disk_hashes(row)
            if tc is None:
                refused.append({"cell": cell.key, "task_dir": row["task_dir"],
                                "refusal": "task directory is missing from disk"})
                continue
            unchanged = (tc == row["task_content_sha256"]
                         and sd == row["seed_json_sha256"])
            if unchanged:
                if row["task_dir"] in prior:
                    already.append({"cell": cell.key, "task_dir": row["task_dir"],
                                    "amendment_id": prior[row["task_dir"]]["amendment_id"]})
                continue

            base = {"cell": cell.key, "mode": cell.mode, "profile": cell.profile,
                    "shard": cell.shard_index, "task_dir": row["task_dir"],
                    "base_task_id": row["base_task_id"], "arm": row["arm"]}

            # (1) the drift must match an append-only source_target repair record
            rep = repair_index.get((cell.mode, cell.profile, cell.shard_index,
                                    row["task_dir"]))
            if rep is None:
                refused.append({**base, "refusal":
                                "task definition drifted with no applied record in "
                                f"provenance/{SOURCE_REPAIR_LOG}"})
                continue
            # A second amendment of the same definition is a human decision.
            if row["task_dir"] in prior:
                refused.append({**base, "refusal":
                                "task definition already carries an approved "
                                "amendment and has drifted again"})
                continue
            # (4) the new definition on disk must BE the approved correction
            if target != rep["corrected"]:
                refused.append({**base, "refusal":
                                f"on-disk source_target {target!r} is not the "
                                f"approved correction {rep['corrected']!r}"})
                continue
            if rep["current"] == rep["corrected"]:
                refused.append({**base, "refusal":
                                "repair record records no actual change"})
                continue
            if disk_ds is None:
                refused.append({**base, "refusal":
                                "shard dataset manifest is missing from disk"})
                continue

            # (2) HARD RULE: never amend a cell that holds a real trajectory
            res = audit_for(cell.mode, cell.shard_index)
            rec = res["records"].get(rep.get("cell_key", ""))
            if rec is None:
                refused.append({**base, "refusal":
                                "repair record names no cell the audit knows: "
                                f"{rep.get('cell_key')!r}"})
                continue
            if rec.status == cells_mod.COMPLETE_VALID:
                refused.append({**base, "cell_key": rec.cell.key, "refusal":
                                "cell holds a COMPLETE_VALID accepted trajectory "
                                "-- the definition it ran against is not amendable"})
                continue
            if any(o.status == cells_mod.COMPLETE_VALID for o in rec.observations):
                refused.append({**base, "cell_key": rec.cell.key, "refusal":
                                "cell has a COMPLETE_VALID observation on disk"})
                continue

            aid = amendment_id(cell.key, row["task_dir"],
                               row["task_content_sha256"], tc)
            cell_approved.append({
                **base,
                "cell_key": rec.cell.key,
                "amendment_id": aid,
                "campaign_id": lib.CAMPAIGN_ID,
                "reason": AMENDMENT_REASON,
                "approved_change_type": SOURCE_TARGET_CORRECTION,
                # (3) old == what is frozen in the manifest right now
                "original_task_content_sha256": row["task_content_sha256"],
                "amended_task_content_sha256": tc,
                "original_seed_json_sha256": row["seed_json_sha256"],
                "amended_seed_json_sha256": sd,
                "original_dataset_manifest_sha256": cm["dataset_manifest_sha256"],
                "amended_dataset_manifest_sha256": disk_ds,
                "old_source_target": rep["current"],
                "new_source_target": rep["corrected"],
                "prior_observation_status": rec.status,
                "prior_observation_count": len(rec.observations),
                "accepted_model_trajectory_exists": False,
                "source_repair_log_reference":
                    f"provenance/{SOURCE_REPAIR_LOG}#L{rep['_line']}"
                    f" ({rep.get('at', '')})",
            })

        # (5) an unexplained drift anywhere in the cell taints the whole cell
        if cell_approved and any(r.get("cell") == cell.key for r in refused):
            for r in cell_approved:
                refused.append({**{k: r[k] for k in ("cell", "task_dir",
                                                     "base_task_id", "arm")},
                                "refusal": "another task definition in this cell "
                                           "drifted without an approved basis"})
            cell_approved = []
        approved.extend(cell_approved)

    return {"campaign_id": lib.CAMPAIGN_ID, "mode": mode, "shard": shard,
            "checked_task_definitions": checked,
            "approved": approved, "already_amended": already,
            "refused": refused, "ok": not refused}


def unapproved_drift(mode: str | None = None, shard: int | None = None, *,
                     paths=None) -> list[dict]:
    """Drifted task definitions with no legitimate amendment basis."""
    return plan(mode, shard, paths=paths)["refused"]


# --------------------------------------------------------------------------- #
# applying
# --------------------------------------------------------------------------- #
def _archive_original(cell_key: str, mf: Path, paths) -> tuple[str, str]:
    """Preserve the original cell manifest byte for byte. Written once."""
    d = archive_dir(paths)
    lib.assert_campaign_path(d.parent, "frozen manifest archive")
    d.mkdir(parents=True, exist_ok=True)
    sha = lib.sha256_file(mf)
    dest = d / f"{mf.stem}.{sha[:12]}.json"
    if not dest.exists():
        dest.write_bytes(mf.read_bytes())
    elif lib.sha256_file(dest) != sha:
        raise SystemExit(f"{cell_key}: archived original {dest.name} does not "
                         "match the manifest it claims to preserve")
    return str(dest.relative_to(lib.PROJECT_ROOT)), sha


def _append_ledger(records: list[dict], paths) -> Path:
    """Append-only, idempotent by amendment_id. Nothing is ever rewritten."""
    p = ledger_path(paths)
    lib.assert_campaign_path(p.parent, "task definition amendment ledger")
    have = {r["amendment_id"] for r in load_ledger(paths)}
    with p.open("a", encoding="utf-8") as fh:
        for rec in records:
            if rec["amendment_id"] in have:
                continue
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return p


def _update_index_entry(entry: dict, cm: dict) -> bool:
    """Refresh one cached cell entry. Returns True if anything changed."""
    before = dict(entry)
    entry["dataset_manifest_sha256"] = cm["dataset_manifest_sha256"]
    entry["original_dataset_manifest_sha256"] = cm["original_dataset_manifest_sha256"]
    entry["task_definition_amendment_ids"] = [
        a["amendment_id"] for a in cm["task_definition_amendments"]]
    return entry != before


def apply(plan_result: dict, *, paths=None, apply: bool = False) -> dict:
    """Write the amendments. Refuses outright while anything is refused."""
    paths = paths or lib.campaign_paths()
    if not plan_result["ok"]:
        raise SystemExit(
            f"refusing to amend: {len(plan_result['refused'])} drifted task "
            "definition(s) have no approved basis:\n  " + "\n  ".join(
                f"{r.get('cell')}/{r.get('task_dir')}: {r['refusal']}"
                for r in plan_result["refused"][:20]))

    by_cell: dict[str, list[dict]] = {}
    for rec in plan_result["approved"]:
        by_cell.setdefault(rec["cell"], []).append(rec)

    written, ledger_written = [], []
    for cell_key, recs in sorted(by_cell.items()):
        mode, profile, label = cell_key.split("/")
        cell = lib.Cell(mode, profile, int(label.split("-")[1]))
        mf = cell_manifest_path(cell, paths)
        original = json.loads(mf.read_text(encoding="utf-8"))
        at = lib.now_iso()
        full = [{**r, "cell": cell_key, "amended_at": at} for r in recs]
        if not apply:
            written.append({"cell": cell_key, "manifest": mf.name,
                            "amendments": len(full),
                            "dataset_manifest_sha256":
                                full[0]["amended_dataset_manifest_sha256"]})
            ledger_written.extend(full)
            continue
        archive_ref, original_sha = _archive_original(cell_key, mf, paths)
        _append_ledger(full, paths)
        ledger_written.extend(full)
        amended = apply_to_manifest(original, full, archive_ref=archive_ref,
                                    original_manifest_sha=original_sha)
        lib.assert_campaign_path(mf.parent, "cell manifest directory")
        mf.write_text(json.dumps(amended, indent=2) + "\n", encoding="utf-8")
        written.append({"cell": cell_key, "manifest": mf.name,
                        "amendments": len(full),
                        "original_dataset_manifest_sha256":
                            amended["original_dataset_manifest_sha256"],
                        "dataset_manifest_sha256": amended["dataset_manifest_sha256"],
                        "archive": archive_ref})

    indexes = refresh_indexes(paths=paths, apply=apply)
    return {"applied": bool(apply), "cells": written,
            "amendments": len(ledger_written),
            "ledger": str(ledger_path(paths).relative_to(lib.PROJECT_ROOT)),
            "indexes": indexes, "ok": True}


def refresh_indexes(*, paths=None, apply: bool = False) -> list[dict]:
    """Bring every integrity index that caches a cell hash back in line.

    `manifests/cells_index.json` and `CAMPAIGN_MANIFEST.json` both embed the
    per-cell `dataset_manifest_sha256`. Left stale they would contradict the
    manifests they summarise. The ORIGINAL value is kept alongside the amended
    one rather than being overwritten and lost.
    """
    paths = paths or lib.campaign_paths()
    manifests = {}
    for cell in lib.all_cells():
        mf = cell_manifest_path(cell, paths)
        if mf.is_file():
            manifests[cell.key] = json.loads(mf.read_text(encoding="utf-8"))

    out = []
    targets = [(paths["manifests"] / "cells_index.json", "cells"),
               (paths["manifest"], "cells")]
    for path, key in targets:
        if not path.is_file():
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        changed = 0
        for entry in doc.get(key) or []:
            cm = manifests.get(entry.get("cell"))
            if cm and cm.get("task_definition_amendments"):
                changed += int(_update_index_entry(entry, cm))
        if not changed:
            continue
        out.append({"index": str(path.relative_to(lib.PROJECT_ROOT)),
                    "cells_updated": changed})
        if apply:
            lib.assert_campaign_path(path.parent, "campaign index")
            path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return out


# --------------------------------------------------------------------------- #
# verification (preflight)
# --------------------------------------------------------------------------- #
def verify_cell(cell: lib.Cell, *, paths=None, ledger=None) -> list[str]:
    """Problems with this cell's amendment provenance. Empty list == clean.

    The current manifest must be EXACTLY the archived original plus the ledger
    records it declares. Anything else -- an undeclared record, a manifest
    edited past its amendments, a missing archive -- is a problem.
    """
    paths = paths or lib.campaign_paths()
    mf = cell_manifest_path(cell, paths)
    if not mf.is_file():
        return [f"{cell.key}: missing cell manifest"]
    cm = json.loads(mf.read_text(encoding="utf-8"))
    recs = ledger_by_cell(paths, ledger).get(cell.key, [])
    stamps = cm.get("task_definition_amendments") or []
    archive_ref = cm.get("frozen_manifest_archive")

    if not recs and not stamps and not archive_ref:
        return []                                   # untouched frozen manifest

    problems = []
    if not recs:
        return [f"{cell.key}: manifest declares {len(stamps)} task-definition "
                f"amendment(s) but the append-only ledger has no record of them "
                f"-- an amended manifest without provenance is not admissible"]
    if not stamps or not archive_ref:
        return [f"{cell.key}: the amendment ledger holds {len(recs)} record(s) "
                f"but the manifest declares none -- refusing to guess which "
                f"definition is authoritative"]

    ledger_ids = {r["amendment_id"] for r in recs}
    stamp_ids = {s["amendment_id"] for s in stamps}
    if ledger_ids != stamp_ids:
        problems.append(
            f"{cell.key}: manifest amendments {sorted(stamp_ids)} do not match "
            f"the ledger {sorted(ledger_ids)}")
    for r in recs:
        if r.get("approved_change_type") not in APPROVED_CHANGE_TYPES:
            problems.append(f"{cell.key}: amendment {r['amendment_id']} has "
                            f"unapproved change type {r.get('approved_change_type')!r}")
        if r.get("accepted_model_trajectory_exists"):
            problems.append(f"{cell.key}: amendment {r['amendment_id']} claims an "
                            "accepted model trajectory exists -- such a cell is "
                            "never amendable")
    if problems:
        return problems

    archive = lib.PROJECT_ROOT / archive_ref
    if not archive.is_file():
        return [f"{cell.key}: archived original manifest {archive_ref} is missing"]
    original_sha = lib.sha256_file(archive)
    if original_sha != cm.get("original_cell_manifest_sha256"):
        return [f"{cell.key}: archived original {archive_ref} hashes "
                f"{original_sha}, manifest claims "
                f"{cm.get('original_cell_manifest_sha256')}"]
    original = json.loads(archive.read_text(encoding="utf-8"))
    for r in recs:
        row = next((t for t in original["trials"]
                    if t["task_dir"] == r["task_dir"]), None)
        if row is None:
            problems.append(f"{cell.key}: amendment {r['amendment_id']} names "
                            f"unknown task dir {r['task_dir']}")
            continue
        if row["task_content_sha256"] != r["original_task_content_sha256"]:
            problems.append(
                f"{cell.key}/{r['task_dir']}: amendment claims original content "
                f"{r['original_task_content_sha256'][:12]} but the frozen "
                f"manifest recorded {row['task_content_sha256'][:12]}")
        if row["seed_json_sha256"] != r["original_seed_json_sha256"]:
            problems.append(
                f"{cell.key}/{r['task_dir']}: amendment claims original seed "
                f"{str(r['original_seed_json_sha256'])[:12]} but the frozen "
                f"manifest recorded {str(row['seed_json_sha256'])[:12]}")
    if original["dataset_manifest_sha256"] != cm.get("original_dataset_manifest_sha256"):
        problems.append(f"{cell.key}: manifest's original dataset hash does not "
                        "match the archived original")
    if problems:
        return problems

    try:
        expected = apply_to_manifest(original, recs, archive_ref=archive_ref,
                                     original_manifest_sha=original_sha)
    except ValueError as exc:
        return [f"{cell.key}: {exc}"]
    if expected != cm:
        diff = sorted(k for k in set(expected) | set(cm)
                      if expected.get(k) != cm.get(k))
        problems.append(
            f"{cell.key}: the cell manifest is not the archived original plus its "
            f"approved amendments (differing: {', '.join(diff) or 'trial rows'})")
    return problems


def verify(cells_sel=None, *, paths=None) -> list[str]:
    """Verify amendment provenance for the given cells (default: all)."""
    paths = paths or lib.campaign_paths()
    ledger = load_ledger(paths)
    selected = list(cells_sel if cells_sel is not None else lib.all_cells())
    known = {c.key for c in lib.all_cells()}
    problems = []
    for rec in ledger:
        if rec.get("campaign_id") != lib.CAMPAIGN_ID:
            problems.append(f"amendment {rec.get('amendment_id')} is for campaign "
                            f"{rec.get('campaign_id')!r}")
        if rec.get("cell") not in known:
            problems.append(f"amendment {rec.get('amendment_id')} names unknown "
                            f"cell {rec.get('cell')!r}")
    for cell in selected:
        problems += verify_cell(cell, paths=paths, ledger=ledger)
    return problems


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def render(plan_result: dict) -> str:
    lines = [f"task definitions checked : {plan_result['checked_task_definitions']}",
             f"approved amendments      : {len(plan_result['approved'])}",
             f"already amended          : {len(plan_result['already_amended'])}",
             f"unapproved drift         : {len(plan_result['refused'])}"]
    for r in plan_result["approved"]:
        lines.append(f"  + {r['cell']}  {r['base_task_id']}/{r['arm']}"
                     f"  [{r['prior_observation_status']}]")
        lines.append(f"      {r['old_source_target']}")
        lines.append(f"   -> {r['new_source_target']}")
        lines.append(f"      task {r['original_task_content_sha256'][:12]} -> "
                     f"{r['amended_task_content_sha256'][:12]}   seed "
                     f"{str(r['original_seed_json_sha256'])[:12]} -> "
                     f"{str(r['amended_seed_json_sha256'])[:12]}")
    for r in plan_result["refused"]:
        lines.append(f"  ! {r.get('cell')}/{r.get('task_dir')}: {r['refusal']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=("plan", "apply", "verify"))
    ap.add_argument("--mode", choices=lib.MODES)
    ap.add_argument("--shard", type=int, choices=lib.SHARD_INDICES)
    ap.add_argument("--apply", action="store_true",
                    help="write the amendments (default: report only)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    paths = lib.campaign_paths()

    if args.command == "verify":
        problems = verify(paths=paths)
        if args.json:
            print(json.dumps({"ok": not problems, "problems": problems}, indent=2))
        else:
            for p in problems:
                print(f"  FAIL: {p}")
            print(f"amendment provenance: {'OK' if not problems else 'FAILED'}")
        return 1 if problems else 0

    result = plan(args.mode, args.shard, paths=paths)
    if args.command == "plan" or not args.apply:
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(render(result))
            if args.command == "apply":
                print("\n(dry run - pass --apply to write)")
        return 0 if result["ok"] else 1

    out = apply(result, paths=paths, apply=True)
    problems = verify(paths=paths)
    out["verified"] = not problems
    out["problems"] = problems
    print(json.dumps(out, indent=2))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
