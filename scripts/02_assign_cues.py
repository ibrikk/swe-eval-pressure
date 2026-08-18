#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cue_assignment import assignment_payload, load_library, validate_assignment_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = args.project_root
    output = args.output or root / "manifests" / "cue_assignments.json"
    registry = json.loads((root / "manifests" / "task_registry.json").read_text(encoding="utf-8"))
    records = registry["tasks"]
    library = load_library(root / "factor_data" / "evaluation_cues.json")

    if output.exists() and not args.force:
        current = json.loads(output.read_text(encoding="utf-8"))
        errors = validate_assignment_payload(current, records, library, "explicit", args.seed)
        if errors:
            raise SystemExit(
                "Existing cue assignment is incompatible:\n - " + "\n - ".join(errors) +
                "\nRe-run with --force only if you intentionally want to replace the permanent assignment."
            )
        print(f"Reusing compatible cue assignment: {output}")
        return

    payload = assignment_payload(records, library, "explicit", args.seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(payload['tasks'])} task-to-cue assignments: {output}")
    print(f"Seed: {payload['assignment_seed']}")
    print(f"Tasks per cue: {payload['tasks_per_cue']}")
    for cue_id, count in payload["summary"]["base_tasks_by_cue"].items():
        print(f"  {cue_id}: {count}")


if __name__ == "__main__":
    main()
