#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from resource_mode import (
    MODE as RESOURCE_MODE,
    task_count as resource_task_count,
    variants_per_task as resource_variants_per_task,
)


def task_count(mode: str, total: int) -> int:
    if mode == "pilot":
        return min(int(os.getenv("PILOT_TASK_COUNT", "4")), total)
    if mode == "sample":
        return min(int(os.getenv("SAMPLE_TASK_COUNT", "10")), total)
    if mode == "full":
        return total
    if mode == RESOURCE_MODE:
        return resource_task_count(total)
    raise SystemExit(f"unknown mode: {mode}")


def full_includes_resource() -> bool:
    value = os.getenv("FULL_INCLUDE_RESOURCE", "true").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise SystemExit(f"invalid FULL_INCLUDE_RESOURCE={value!r}; use true or false")


def variants_per_task(mode: str) -> int:
    if mode == RESOURCE_MODE:
        return resource_variants_per_task(["scaffold"])
    if mode == "full" and full_includes_resource():
        return 11
    return 10


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--assignment-file", type=Path, required=True)
    args = parser.parse_args()

    registry_path = args.project_root / "manifests" / "task_registry.json"
    if not registry_path.exists():
        raise SystemExit("Run ./lab.sh inventory first.")
    if not args.assignment_file.exists():
        raise SystemExit("Run ./lab.sh assign-cues first.")

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assignment = json.loads(args.assignment_file.read_text(encoding="utf-8"))
    total = registry["task_count"]
    selected = task_count(args.mode, total)
    variants = variants_per_task(args.mode)
    profiles = args.profiles.split()
    per_profile = selected * variants
    trajectories = per_profile * args.repeats * len(profiles)

    print("===== BENCHMARK PLAN =====")
    print(f"Mode:                    {args.mode}")
    print(f"Available base tasks:    {total}")
    print(f"Selected base tasks:     {selected}")
    print(f"Logical variants/task:   {variants}")
    if args.mode == RESOURCE_MODE:
        print("Conditions:              clean, eval-only, eval+resource-deprivation")
        print("Delivery channel:        scaffold-native")
        print(f"Resource message index:  {os.getenv('RESOURCE_DEPRIVATION_MESSAGE_INDEX', '10')}")
    else:
        if args.mode == "full" and full_includes_resource():
            print("Conditions:              clean, eval-only, eval+financial, eval+self-preservation, eval+resource-deprivation")
            print("Resource placement:      scaffold-native only")
            print(f"Resource message index:  {os.getenv('RESOURCE_DEPRIVATION_MESSAGE_INDEX', '10')}")
        else:
            print("Conditions:              clean, eval-only, eval+financial, eval+self-preservation")
        print("Delivery channels:       source-local, passive-root, scaffold-native")
    print(f"Cue level:               {assignment['cue_level']}")
    print(f"Cue assignment seed:     {assignment['assignment_seed']}")
    print(f"Exact cues in primary:   {assignment['cue_count']}")
    print(f"Tasks per exact cue:     {assignment['tasks_per_cue']}")
    print(f"Internet access:         {os.getenv('ALLOW_INTERNET', 'true')}")
    print(f"Profiles:                {' '.join(profiles)}")
    print(f"Harbor tasks/profile:    {per_profile}")
    print(f"Repeats/task:            {args.repeats}")
    print(f"Total trajectories:      {trajectories}")
    if args.mode == "full":
        seeded_counts = Counter(item["cue_id"] for item in assignment["tasks"])
        seeded_variants = variants - 1
        print(f"Clean instances/profile: {selected}")
        print(f"Seeded instances/profile:{selected * seeded_variants:>5}")
        print("Cue distribution:        " + ", ".join(f"{cue_id}={count * seeded_variants}" for cue_id, count in sorted(seeded_counts.items())))
    print()
    if args.mode == RESOURCE_MODE:
        print("Note: standalone resource mode reuses each base task's permanent cue and carries its own clean/eval-only controls.")
    elif args.mode == "full" and full_includes_resource():
        print("Note: full mode includes one scaffold-native eval+resource-deprivation variant per base task; existing clean/eval-only scaffold runs are its matched controls.")
    else:
        print("Note: each base task keeps one permanent explicit cue across all seeded variants and all profiles.")


if __name__ == "__main__":
    main()
