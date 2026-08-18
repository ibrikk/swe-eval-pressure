#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a temporary install-only Harbor dataset with one representative "
            "variant per SWE-Atlas base task."
        )
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.project_root.resolve()
    source = args.source.resolve()
    output = args.output.resolve()
    manifest_path = source / "manifest.json"

    if not manifest_path.is_file():
        raise SystemExit(f"missing source manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = manifest.get("tasks", [])
    if not isinstance(tasks, list) or not tasks:
        raise SystemExit(f"source manifest has no tasks: {manifest_path}")

    by_base: dict[str, list[dict]] = {}
    order: list[str] = []
    for task in tasks:
        base = str(task.get("base_task_id", "")).strip()
        directory = str(task.get("directory", "")).strip()
        if not base or not directory:
            raise SystemExit("manifest task missing base_task_id or directory")
        if base not in by_base:
            by_base[base] = []
            order.append(base)
        by_base[base].append(task)

    # Prefer the clean variant because install-only is testing task-image/agent
    # compatibility, not cue delivery. Fall back deterministically if a mode has
    # no clean variant.
    selected: list[dict] = []
    for base in order:
        variants = by_base[base]
        clean = [t for t in variants if t.get("condition") == "clean"]
        selected.append((clean or variants)[0])

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    for task in selected:
        directory = task["directory"]
        src = source / directory
        dst = output / directory
        if not (src / "task.toml").is_file():
            raise SystemExit(f"missing Harbor task directory: {src}")
        shutil.copytree(src, dst)

    reduced = dict(manifest)
    try:
        reduced["source_dataset"] = str(source.relative_to(root))
    except ValueError:
        reduced["source_dataset"] = str(source)
    reduced["install_only_representative"] = True
    reduced["base_task_count"] = len(selected)
    reduced["variants_per_task"] = 1
    reduced["tasks"] = selected
    (output / "manifest.json").write_text(
        json.dumps(reduced, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(output)


if __name__ == "__main__":
    main()
