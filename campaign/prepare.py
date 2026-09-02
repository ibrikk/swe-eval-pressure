#!/usr/bin/env python3
"""Campaign prepare: namespace, immutable dataset snapshot, shards, cell manifests.

Dataset integrity strategy (decision recorded in CAMPAIGN_MANIFEST.json):

    B - campaign-local immutable snapshot, built with hardlinks.

Rationale over "just point at generated/ read-only":
  * `./lab.sh prepare` rebuilds generated/<mode>/<profile> with rmtree + rewrite.
    A hardlink snapshot survives that unharmed, because the rebuild creates NEW
    inodes; a read-only reference would silently start pointing at new content
    mid-campaign.
  * Hardlinks cost ~0 bytes (same filesystem), so the 809 MB corpus is snapshotted
    for free. There is no size argument for the weaker option.
  * Hardlinks still share an inode, so an in-place edit WOULD propagate. That
    residual hole is closed by content hashes: every task directory's sha256 is
    frozen in the cell manifest and re-verified by `campaign.py preflight`
    before every single shard launch.

Nothing here mutates generated/ or results/full or results/resource.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from campaign import lib


def link_or_copy(src: str, dst: str) -> str:
    try:
        os.link(src, dst)
        return dst
    except OSError:
        return shutil.copy2(src, dst)


def snapshot_dataset(mode: str, profile: str, dest_root: Path, force: bool) -> Path:
    src = lib.PROJECT_ROOT / "generated" / mode / profile
    if not (src / "manifest.json").is_file():
        raise SystemExit(f"missing generated dataset: {src}/manifest.json")
    dest = dest_root / mode / profile
    if dest.exists():
        if not force:
            return dest
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest, copy_function=link_or_copy)
    return dest


def shard_snapshot(snapshot: Path, mode: str, profile: str, shard_index: int,
                   shards_root: Path, force: bool) -> Path:
    manifest = json.loads((snapshot / "manifest.json").read_text())
    base_ids, seen = [], set()
    for item in manifest["tasks"]:
        bid = str(item["base_task_id"])
        if bid not in seen:
            seen.add(bid)
            base_ids.append(bid)
    if len(base_ids) != lib.BASE_TASK_COUNT:
        raise SystemExit(f"{mode}/{profile}: expected {lib.BASE_TASK_COUNT} base tasks, got {len(base_ids)}")

    start = (shard_index - 1) * lib.SHARD_SIZE
    end = min(start + lib.SHARD_SIZE, len(base_ids))
    if start >= len(base_ids):
        raise SystemExit(f"shard {shard_index} starts beyond {len(base_ids)} base tasks")
    selected = set(base_ids[start:end])
    tasks = [t for t in manifest["tasks"] if str(t["base_task_id"]) in selected]

    expected_base = lib.BASE_TASKS_PER_SHARD[shard_index]
    expected_trials = expected_base * lib.VARIANTS_PER_TASK[mode]
    if len(selected) != expected_base or len(tasks) != expected_trials:
        raise SystemExit(
            f"{mode}/{profile}/shard{shard_index}: sharding produced "
            f"{len(selected)} base tasks / {len(tasks)} trials, "
            f"expected {expected_base} / {expected_trials}"
        )

    label = f"chunk-{shard_index}-size-{lib.SHARD_SIZE}"
    target = shards_root / mode / profile / label
    if target.exists():
        if not force:
            return target
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for item in tasks:
        shutil.copytree(snapshot / item["directory"], target / item["directory"],
                        copy_function=link_or_copy)

    shard_manifest = dict(manifest)
    shard_manifest["tasks"] = tasks
    shard_manifest["base_task_count"] = len(selected)
    shard_manifest["shard"] = {
        "type": "fixed_size", "index": shard_index,
        "base_task_size": lib.SHARD_SIZE,
        "start_base_task": start + 1, "end_base_task": end,
    }
    shard_manifest["campaign_id"] = lib.CAMPAIGN_ID
    (target / "manifest.json").write_text(
        json.dumps(shard_manifest, indent=2, ensure_ascii=False) + "\n")
    return target


def build_cell_manifest(cell: lib.Cell, shard_dir: Path, snapshot: Path) -> dict:
    """Per-cell manifest: one row per trial, enough to detect any drift."""
    manifest = json.loads((shard_dir / "manifest.json").read_text())
    arms = lib.ARMS[cell.mode]

    # Resource controls are byte-identical duplicates of the FULL task of the
    # same name; record that lineage explicitly so analysis can never silently
    # pool them across modes.
    full_snapshot = snapshot.parent.parent / "full" / cell.profile

    rows = []
    for item in manifest["tasks"]:
        directory = item["directory"]
        task_dir = shard_dir / directory
        base_id, arm = lib.parse_trial_dir(directory)
        if arm not in arms:
            raise SystemExit(f"{cell.key}: unexpected arm {arm!r} in {directory}")
        condition, channel, pressure = arms[arm]
        seed = task_dir / "environment" / "benchmark_seed" / "seed.json"
        parent = None
        if cell.mode == "resource" and arm in ("clean-n", "eval-scaf"):
            candidate = full_snapshot / directory
            if candidate.is_dir():
                parent = f"full/{cell.profile}/{directory}"
        rows.append({
            "campaign_id": lib.CAMPAIGN_ID,
            "mode": cell.mode,
            "shard": cell.shard_label,
            "shard_index": cell.shard_index,
            "profile": cell.profile,
            "base_task_id": base_id,
            "arm": arm,
            "condition": condition,
            "delivery_channel": channel,
            "pressure_kind": pressure,
            "task_dir": directory,
            "source_task_path": f"generated/{cell.mode}/{cell.profile}/{directory}",
            "snapshot_task_path": str(task_dir.relative_to(lib.PROJECT_ROOT)),
            "task_content_sha256": lib.sha256_tree(task_dir),
            "seed_json_sha256": lib.sha256_file(seed) if seed.is_file() else None,
            "resource_derivation_parent": parent,
        })

    if len(rows) != cell.expected_trials:
        raise SystemExit(f"{cell.key}: {len(rows)} rows, expected {cell.expected_trials}")

    seen = set()
    for r in rows:
        k = (r["base_task_id"], r["arm"])
        if k in seen:
            raise SystemExit(f"{cell.key}: duplicate cell {k}")
        seen.add(k)

    counts = {}
    for r in rows:
        counts[r["arm"]] = counts.get(r["arm"], 0) + 1
    if set(counts) != set(arms) or len(set(counts.values())) != 1:
        raise SystemExit(f"{cell.key}: unbalanced arms {counts}")

    return {
        "campaign_id": lib.CAMPAIGN_ID,
        "cell": cell.key,
        "mode": cell.mode,
        "profile": cell.profile,
        "shard": cell.shard_label,
        "shard_index": cell.shard_index,
        "base_task_count": cell.base_tasks,
        "expected_trials": cell.expected_trials,
        "arms": sorted(arms),
        "arm_counts": counts,
        "dataset_path": str(shard_dir.relative_to(lib.PROJECT_ROOT)),
        "dataset_manifest_sha256": lib.sha256_file(shard_dir / "manifest.json"),
        "model": lib.MODEL_PINS[cell.profile],
        "agent_version_pinned": lib.VERSION_PINS[cell.profile]["version"],
        "trials": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="rebuild snapshot/shards even if present")
    ap.add_argument("--modes", nargs="*", default=list(lib.MODES))
    ap.add_argument("--profiles", nargs="*", default=list(lib.PROFILES))
    args = ap.parse_args()

    paths = lib.campaign_paths()
    for name in ("root", "full", "resource", "logs", "manifests", "provenance",
                 "validation", "analysis", "datasets"):
        paths[name].mkdir(parents=True, exist_ok=True)
    (paths["manifests"] / "cells").mkdir(exist_ok=True)

    dsroot = paths["datasets"]
    shards_root = dsroot / "_shards"

    cells = [c for c in lib.all_cells(args.modes, args.profiles)]
    summary = []
    for mode in args.modes:
        for profile in args.profiles:
            snap = snapshot_dataset(mode, profile, dsroot, args.force)
            lib.eprint(f"[snapshot] {mode}/{profile} -> {snap.relative_to(lib.PROJECT_ROOT)}")

    for cell in cells:
        snap = dsroot / cell.mode / cell.profile
        shard_dir = shard_snapshot(snap, cell.mode, cell.profile, cell.shard_index,
                                   shards_root, args.force)
        cm = build_cell_manifest(cell, shard_dir, snap)
        out = paths["manifests"] / "cells" / f"{cell.mode}__{cell.profile}__{cell.shard_label}.json"
        out.write_text(json.dumps(cm, indent=2) + "\n")
        summary.append({
            "cell": cell.key,
            "expected_trials": cell.expected_trials,
            "dataset_manifest_sha256": cm["dataset_manifest_sha256"],
            "manifest": str(out.relative_to(lib.PROJECT_ROOT)),
        })
        lib.eprint(f"[cell] {cell.key}: {cell.expected_trials} trials -> {out.name}")

    (paths["manifests"] / "cells_index.json").write_text(
        json.dumps({"campaign_id": lib.CAMPAIGN_ID, "generated_at": lib.now_iso(),
                    "cells": summary}, indent=2) + "\n")

    write_campaign_manifest(paths, summary)
    write_campaign_readme(paths)
    print(json.dumps({"cells": len(summary),
                      "trials": sum(s["expected_trials"] for s in summary)}, indent=2))


def write_campaign_manifest(paths, summary) -> None:
    from campaign import cost_model
    est = cost_model.estimate()
    exp = lib.expected_totals()
    budget = lib.probe_budget() if os.environ.get("CAMPAIGN_PROBE_BUDGET") == "1" else None

    doc = {
        "campaign_id": lib.CAMPAIGN_ID,
        "schema_version": "1.0",
        "created_at": lib.now_iso(),
        "purpose": (
            "Second, completely fresh experimental campaign for SWE-EvalPressure. "
            "Every trajectory in this namespace is newly executed. No trajectory, "
            "partial run, salvaged shard or repair attempt from the Aug 2026 corpus "
            "is reused, pooled or referenced as data."
        ),
        "design": {
            "full": {
                "base_tasks": lib.BASE_TASK_COUNT,
                "arms": lib.VARIANTS_PER_TASK["full"],
                "arm_names": list(lib.FULL_ARMS),
                "per_profile": exp["full"]["per_profile"],
                "profiles": list(lib.PROFILES),
                "total": exp["full"]["total"],
            },
            "resource": {
                "base_tasks": lib.BASE_TASK_COUNT,
                "arms": lib.VARIANTS_PER_TASK["resource"],
                "arm_names": list(lib.RESOURCE_ARMS),
                "per_profile": exp["resource"]["per_profile"],
                "profiles": list(lib.PROFILES),
                "total": exp["resource"]["total"],
                "self_contained": True,
                "self_contained_rationale": (
                    "clean-n and eval-scaf task definitions are byte-identical to the "
                    "FULL cells of the same name (verified: 140/140 identical seed.json "
                    "hashes per profile). They are nevertheless executed independently "
                    "so RESOURCE has its own freshly executed control, eval-only "
                    "comparator and treatment. No FULL trajectory may be substituted."
                ),
            },
            "campaign_total": exp["campaign_total"],
        },
        "sharding": {
            "type": "fixed_size_base_task",
            "shard_size": lib.SHARD_SIZE,
            "shards": {
                str(i): {
                    "base_tasks": lib.BASE_TASKS_PER_SHARD[i],
                    "base_task_range": (
                        f"{(i-1)*lib.SHARD_SIZE+1}-{min(i*lib.SHARD_SIZE, lib.BASE_TASK_COUNT)}"
                    ),
                    "full_per_profile": lib.BASE_TASKS_PER_SHARD[i] * lib.VARIANTS_PER_TASK["full"],
                    "full_all_profiles": lib.BASE_TASKS_PER_SHARD[i] * lib.VARIANTS_PER_TASK["full"] * len(lib.PROFILES),
                    "resource_per_profile": lib.BASE_TASKS_PER_SHARD[i] * lib.VARIANTS_PER_TASK["resource"],
                    "resource_all_profiles": lib.BASE_TASKS_PER_SHARD[i] * lib.VARIANTS_PER_TASK["resource"] * len(lib.PROFILES),
                }
                for i in lib.SHARD_INDICES
            },
        },
        "version_pins": {
            p: {
                "agent": lib.VERSION_PINS[p]["agent"],
                "agent_version": lib.VERSION_PINS[p]["version"],
                "pin_env": lib.VERSION_PINS[p]["pin_env"],
                "model": lib.MODEL_PINS[p],
            }
            for p in lib.PROFILES
        },
        "runtime_pins": {
            "harbor": "0.20.0",
            "mini_swe_litellm": "1.83.0",
            "uv_bootstrap": "0.7.13",
            "modal_vm_runtime": True,
            "harbor_repeats": 1,
        },
        "factors": {
            "cue_assignment_seed": 20260805,
            "financial_message_index": 3,
            "self_preservation_message_index": 5,
            "resource_deprivation_message_index": 10,
        },
        "dataset_integrity": {
            "strategy": "campaign_local_hardlink_snapshot_plus_content_hashes",
            "snapshot_root": str(paths["datasets"].relative_to(lib.PROJECT_ROOT)),
            "verified_before_every_shard": True,
        },
        "cost_model": est,
        "budget_at_prepare": budget.as_dict() if budget else None,
        "cells": summary,
        "policy": {
            "reuse_historical_trajectories": False,
            "silent_salvage": False,
            "dedupe_by_task_name": False,
            "backfill_across_attempts": False,
            "partial_shard_acceptance": False,
            "on_failed_attempt": "preserve as FAILED and stop; replacement gets a new attempt_id",
        },
    }
    paths["manifest"].write_text(json.dumps(doc, indent=2) + "\n")
    lib.eprint(f"[manifest] {paths['manifest'].relative_to(lib.PROJECT_ROOT)}")


def write_campaign_readme(paths) -> None:
    exp = lib.expected_totals()
    pins = "\n".join(
        f"| {p} | `{lib.VERSION_PINS[p]['agent']}` | `{lib.VERSION_PINS[p]['version']}` "
        f"| `{lib.VERSION_PINS[p]['pin_env']}` | `{lib.MODEL_PINS[p]}` |"
        for p in lib.PROFILES
    )
    text = f"""# Campaign `{lib.CAMPAIGN_ID}`

Second, completely fresh experimental campaign for SWE-EvalPressure.

## Hard rule

**Every trajectory under this directory is newly executed by this campaign.**
Nothing from the Aug 2026 corpus (`results/full`, `results/resource`,
`results/archive`, `results/quarantine`, `results/failed_repair_attempts`) is
reused, pooled, salvaged or backfilled. The tooling enforces this structurally:
`campaign/lib.py:assert_campaign_path` refuses any run directory outside
`results/campaigns/{lib.CAMPAIGN_ID}/`.

## Design

| mode | base tasks | arms | per profile | profiles | total |
|---|---|---|---|---|---|
| full | {lib.BASE_TASK_COUNT} | {lib.VARIANTS_PER_TASK['full']} | {exp['full']['per_profile']} | {len(lib.PROFILES)} | **{exp['full']['total']}** |
| resource | {lib.BASE_TASK_COUNT} | {lib.VARIANTS_PER_TASK['resource']} | {exp['resource']['per_profile']} | {len(lib.PROFILES)} | **{exp['resource']['total']}** |
| | | | | | **{exp['campaign_total']}** |

FULL arms: {', '.join('`' + a + '`' for a in lib.FULL_ARMS)}

RESOURCE arms: {', '.join('`' + a + '`' for a in lib.RESOURCE_ARMS)}

### RESOURCE is deliberately self-contained

`clean-n` and `eval-scaf` task definitions in RESOURCE are byte-identical to the
FULL cells of the same name (verified: 140/140 matching `seed.json` hashes per
profile). They are executed **independently anyway**. RESOURCE gets its own
freshly executed control, eval-only comparator and resource-deprivation
treatment, so RESOURCE analysis never depends on a FULL trajectory. The
duplication is recorded per trial as `resource_derivation_parent` for
transparency; it is never used to substitute data.

## Sharding

Fixed-size over the 70 ordered base task ids, 30 per shard.

| shard | base tasks | FULL /profile | FULL all | RESOURCE /profile | RESOURCE all |
|---|---|---|---|---|---|
| 1 | 1-30 | 300 | 1200 | 90 | 360 |
| 2 | 31-60 | 300 | 1200 | 90 | 360 |
| 3 | 61-70 | 100 | 400 | 30 | 120 |

## Version pins

| profile | agent | version | pin env | model |
|---|---|---|---|---|
{pins}

Runtime: Harbor `0.20.0`, Modal VM runtime, `mini-swe-agent` extra `litellm==1.83.0`,
uv bootstrap `0.7.13`, `harbor_repeats=1`.

## Layout

```
{lib.CAMPAIGN_ID}/
  CAMPAIGN_MANIFEST.json   frozen design, pins, cost model, policy
  CAMPAIGN_README.md       this file
  datasets/                hardlink snapshot of generated/ + pre-built shards
  full/                    Harbor job outputs, mode=full
  resource/                Harbor job outputs, mode=resource
  manifests/cells/         one frozen manifest per (mode, profile, shard)
  provenance/              attempts.jsonl, accepted_runs.json, corpus.jsonl
  validation/              validation_report.json
  analysis/                analysis outputs
  logs/                    preflight + runner logs
```

## Operating it

```
./campaign.sh prepare      {lib.CAMPAIGN_ID}
./campaign.sh preflight    {lib.CAMPAIGN_ID}
./campaign.sh run-full     {lib.CAMPAIGN_ID}
./campaign.sh run-resource {lib.CAMPAIGN_ID}
./campaign.sh validate     {lib.CAMPAIGN_ID}
./campaign.sh analyze      {lib.CAMPAIGN_ID}
```

`run-full` and `run-resource` preflight before every shard, run all four
profiles for that shard, validate, and **stop on the first failure**. A failed
attempt is preserved as FAILED; the replacement gets a new `attempt_id`. There
is no backfilling and no partial-shard acceptance.
"""
    paths["readme"].write_text(text)
    lib.eprint(f"[readme] {paths['readme'].relative_to(lib.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
