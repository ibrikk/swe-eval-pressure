#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
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


def parse_fraction(value: str) -> tuple[int, int]:
    try:
        index_text, total_text = value.split("/", 1)
        index, total = int(index_text), int(total_text)
    except Exception as exc:
        raise argparse.ArgumentTypeError("shard must look like 1/3") from exc
    if total < 1 or index < 1 or index > total:
        raise argparse.ArgumentTypeError("shard uses 1-based indexing and must satisfy 1 <= index <= total")
    return index, total


def evenly_partition(items: list[str], index: int, total: int) -> list[str]:
    n = len(items)
    start = (index - 1) * n // total
    end = index * n // total
    return items[start:end]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a base-task-aware runnable shard of a generated dataset.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--profile", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--shard", type=parse_fraction, help="Balanced 1-based shard, e.g. 1/3")
    group.add_argument("--shard-size", type=int, help="Number of base tasks per fixed-size shard")
    parser.add_argument("--shard-index", type=int, help="1-based index used with --shard-size")
    args = parser.parse_args()

    source = args.project_root / "generated" / args.mode / args.profile
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing generated dataset: {manifest_path}; run ./lab.sh prepare {args.mode} first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_ids = ordered_unique([str(item["base_task_id"]) for item in manifest["tasks"]])

    if args.shard is not None:
        index, total = args.shard
        selected = evenly_partition(base_ids, index, total)
        label = f"shard-{index}-of-{total}"
        shard_meta = {"type": "balanced", "index": index, "total": total}
    else:
        if args.shard_size is None or args.shard_size < 1:
            raise SystemExit("--shard-size must be >= 1")
        if args.shard_index is None or args.shard_index < 1:
            raise SystemExit("--shard-index is required with --shard-size and uses 1-based indexing")
        start = (args.shard_index - 1) * args.shard_size
        end = min(start + args.shard_size, len(base_ids))
        if start >= len(base_ids):
            raise SystemExit(
                f"shard index {args.shard_index} starts beyond the {len(base_ids)} available base tasks"
            )
        selected = base_ids[start:end]
        label = f"chunk-{args.shard_index}-size-{args.shard_size}"
        shard_meta = {
            "type": "fixed_size",
            "index": args.shard_index,
            "base_task_size": args.shard_size,
            "start_base_task": start + 1,
            "end_base_task": end,
        }

    selected_set = set(selected)
    tasks = [item for item in manifest["tasks"] if item["base_task_id"] in selected_set]
    target = args.project_root / "generated" / "_shards" / args.mode / args.profile / label
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)

    for item in tasks:
        src = source / item["directory"]
        dst = target / item["directory"]
        shutil.copytree(src, dst, copy_function=link_or_copy)

    shard_manifest = dict(manifest)
    shard_manifest["tasks"] = tasks
    shard_manifest["base_task_count"] = len(selected)
    shard_manifest["shard"] = shard_meta
    shard_manifest["source_dataset"] = str(source.relative_to(args.project_root))
    (target / "manifest.json").write_text(
        json.dumps(shard_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(
        f"Prepared {label}: {len(selected)} base tasks, {len(tasks)} trajectories -> {target}",
        file=sys.stderr,
    )
    print(target)


if __name__ == "__main__":
    main()
