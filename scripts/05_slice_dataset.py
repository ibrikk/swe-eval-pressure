#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def link_or_copy(src: str, dst: str) -> str:
    try:
        os.link(src, dst)
        return dst
    except OSError:
        return shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a contiguous base-task-aware subset of an already generated dataset."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-index", type=int, required=True, help="Zero-based base-task start index")
    parser.add_argument("--base-task-count", type=int, required=True)
    parser.add_argument("--label", default="adaptive_llama_batch")
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing source manifest: {manifest_path}")
    if args.start_index < 0:
        raise SystemExit("--start-index must be >= 0")
    if args.base_task_count < 1:
        raise SystemExit("--base-task-count must be >= 1")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = manifest.get("tasks", [])
    base_ids = ordered_unique([str(item["base_task_id"]) for item in tasks])
    end = min(args.start_index + args.base_task_count, len(base_ids))
    if args.start_index >= len(base_ids):
        raise SystemExit(
            f"start index {args.start_index} is beyond the {len(base_ids)} available base tasks"
        )
    selected = base_ids[args.start_index:end]
    selected_set = set(selected)
    subset = [item for item in tasks if str(item["base_task_id"]) in selected_set]

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    for item in subset:
        src = source / item["directory"]
        dst = output / item["directory"]
        shutil.copytree(src, dst, copy_function=link_or_copy)

    out_manifest = dict(manifest)
    out_manifest["tasks"] = subset
    out_manifest["base_task_count"] = len(selected)
    out_manifest["adaptive_batch"] = {
        "label": args.label,
        "start_index_zero_based": args.start_index,
        "end_index_exclusive": end,
        "base_task_count": len(selected),
        "base_task_ids": selected,
    }
    if not out_manifest.get("source_dataset"):
        try:
            out_manifest["source_dataset"] = str(source.relative_to(project_root))
        except ValueError:
            out_manifest["source_dataset"] = ""

    (output / "manifest.json").write_text(
        json.dumps(out_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
