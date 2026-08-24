#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


def link_or_copy(src: str, dst: str) -> str:
    try:
        os.link(src, dst)
        return dst
    except OSError:
        return shutil.copy2(src, dst)


def ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def evenly_partition(items: list[str], index: int, total: int) -> list[str]:
    n = len(items)
    start = (index - 1) * n // total
    end = index * n // total
    return items[start:end]


def main() -> None:
    ap = argparse.ArgumentParser(description="Partition an already-selected dataset across LiteLLM keys.")
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--total", type=int, required=True)
    args = ap.parse_args()

    source = args.source.resolve()
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing source manifest: {manifest_path}")
    if args.total < 1 or args.index < 1 or args.index > args.total:
        raise SystemExit("key shard uses 1-based indexing and must satisfy 1 <= index <= total")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_ids = ordered_unique([str(item["base_task_id"]) for item in manifest.get("tasks", [])])
    selected = evenly_partition(base_ids, args.index, args.total)
    if not selected:
        raise SystemExit("selected key shard is empty")
    selected_set = set(selected)
    tasks = [item for item in manifest["tasks"] if str(item["base_task_id"]) in selected_set]

    source_id = hashlib.sha256(str(source).encode()).hexdigest()[:10]
    source_label = source.name.replace("/", "-")
    target = (
        args.project_root
        / "generated"
        / "_key_shards"
        / args.mode
        / args.profile
        / f"{source_label}-{source_id}"
        / f"key-{args.index}-of-{args.total}"
    )
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)

    for item in tasks:
        src = source / item["directory"]
        dst = target / item["directory"]
        shutil.copytree(src, dst, copy_function=link_or_copy)

    try:
        parent_rel = str(source.relative_to(args.project_root))
    except ValueError:
        parent_rel = str(source)
    ultimate_source = manifest.get("source_dataset") or parent_rel

    shard_manifest = dict(manifest)
    shard_manifest["tasks"] = tasks
    shard_manifest["base_task_count"] = len(selected)
    shard_manifest["source_dataset"] = ultimate_source
    shard_manifest["parent_dataset"] = parent_rel
    shard_manifest["key_shard"] = {
        "index": args.index,
        "total": args.total,
        "base_task_count": len(selected),
    }
    (target / "manifest.json").write_text(
        json.dumps(shard_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(target)


if __name__ == "__main__":
    main()
