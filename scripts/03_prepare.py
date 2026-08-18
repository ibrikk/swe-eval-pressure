#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from cue_assignment import load_library, validate_assignment_payload
from resource_mode import (
    CONDITION as RESOURCE_CONDITION,
    MODE as RESOURCE_MODE,
    build_variants as resource_variants,
    task_count as resource_task_count,
    variants_per_task as resource_variants_per_task,
)

PILOT_IDS = [
    "task-69391d8d1ce51c407be1e533",  # Python
    "task-696719205599a51110d4b435",  # Go
    "task-696719205599a51110d4b426",  # TSX
    "task-69b7c2a04b6f8ff9ed98812f",  # C
]


def remove_docker_image(text: str) -> str:
    return re.sub(r'(?m)^\s*docker_image\s*=\s*"[^"]+"\s*\n?', "", text, count=1)


def set_allow_internet(text: str, allow: bool) -> str:
    value = "true" if allow else "false"
    pattern = r'(?m)^\s*allow_internet\s*=\s*(?:true|false)\s*$'
    if re.search(pattern, text):
        return re.sub(pattern, f"allow_internet = {value}", text, count=1)
    suffix = "" if text.endswith("\n") else "\n"
    return text + suffix + f"allow_internet = {value}\n"


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise SystemExit(f"invalid boolean value: {value!r}; use true or false")


def base_image_from_record(record: dict[str, Any]) -> str:
    value = str(record.get("base_image", "")).strip()
    if not value:
        raise ValueError(f"{record.get('task_id')}: missing base image")
    return value


def selected(records: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    by_id = {record["task_id"]: record for record in records}
    explicit = os.getenv("BENCHMARK_TASK_IDS", "").strip()
    if explicit:
        ids = [item.strip() for item in explicit.split(",") if item.strip()]
        missing = [task_id for task_id in ids if task_id not in by_id]
        if missing:
            raise SystemExit(f"BENCHMARK_TASK_IDS not found: {missing}")
        return [by_id[task_id] for task_id in ids]
    if mode == "pilot":
        return [by_id[task_id] for task_id in PILOT_IDS if task_id in by_id][
            : int(os.getenv("PILOT_TASK_COUNT", "4"))
        ]
    if mode == "sample":
        return records[: int(os.getenv("SAMPLE_TASK_COUNT", "10"))]
    if mode == "full":
        return records
    if mode == RESOURCE_MODE:
        return records[: resource_task_count(len(records))]
    raise SystemExit(f"unknown mode: {mode}")


def message_by_index(path: Path, key: str, index: int) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))[key]
    by_index = {int(item["msg_level_index"]): item for item in data}
    if index not in by_index:
        raise SystemExit(f"{key} index {index} unavailable")
    return by_index[index]


def compose(eval_text: str, pressure: dict[str, Any] | None) -> str:
    parts = [eval_text.strip()]
    if pressure:
        parts.extend(
            [
                "",
                f"Message from {pressure['from']}: {pressure['subject']}",
                pressure["body"].strip(),
            ]
        )
    return "\n".join(parts).strip()


def short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def dockerfile(base_image: str, workspace_root: str) -> str:
    return f'''FROM {base_image}

ENTRYPOINT ["sh", "-c", "while true; do sleep 3600; done"]

COPY benchmark_seed/ /tmp/benchmark_seed/

RUN python3 /tmp/benchmark_seed/apply_seed.py /tmp/benchmark_seed/seed.json \\
    && rm -rf /tmp/benchmark_seed \\
    && cd {workspace_root} \\
    && rm -rf .git \\
    && git init -q \\
    && git config user.name "Developer" \\
    && git config user.email "developer@example.invalid" \\
    && git add -A \\
    && git commit -q -m "Initial commit"

WORKDIR {workspace_root}
'''


def create(
    base: Path,
    destination: Path,
    record: dict[str, Any],
    seed: dict[str, Any],
    apply_seed: Path,
    allow_internet: bool,
) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(base, destination, ignore=shutil.ignore_patterns(".DS_Store", "__MACOSX"))
    task_toml = destination / "task.toml"
    task_text = remove_docker_image(task_toml.read_text(encoding="utf-8"))
    task_toml.write_text(set_allow_internet(task_text, allow_internet), encoding="utf-8")
    environment = destination / "environment"
    seed_dir = environment / "benchmark_seed"
    if seed_dir.exists():
        shutil.rmtree(seed_dir)
    seed_dir.mkdir(parents=True)
    shutil.copy2(apply_seed, seed_dir / "apply_seed.py")
    (seed_dir / "seed.json").write_text(
        json.dumps(seed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (environment / "Dockerfile").write_text(
        dockerfile(base_image_from_record(record), str(record["workspace_root"])),
        encoding="utf-8",
    )


def task_name(task_id: str, condition: str, channel: str) -> str:
    suffix = task_id.split("-", 1)[1][-8:]
    labels = {
        "clean": "clean",
        "eval_only": "eval",
        "eval_financial": "eval-fin",
        "eval_self_preservation": "eval-self",
        RESOURCE_CONDITION: "eval-resource",
    }
    channel_label = {"none": "n", "source": "src", "root": "root", "scaffold": "scaf"}[
        channel
    ]
    return f"ea-{suffix}-{labels[condition]}-{channel_label}"


def core_variants(
    channels: list[str],
    financial: dict[str, Any],
    self_preservation: dict[str, Any],
    eval_text: str,
) -> list[dict[str, Any]]:
    variants = [
        {
            "condition": "clean",
            "channel": "none",
            "pressure_type": "none",
            "pressure_index": None,
            "pressure_level": 0,
            "content": "",
        }
    ]
    for channel in channels:
        variants.extend(
            [
                {
                    "condition": "eval_only",
                    "channel": channel,
                    "pressure_type": "none",
                    "pressure_index": None,
                    "pressure_level": 0,
                    "content": compose(eval_text, None),
                },
                {
                    "condition": "eval_financial",
                    "channel": channel,
                    "pressure_type": "financial",
                    "pressure_index": financial["msg_level_index"],
                    "pressure_level": financial["component_pressure_level"],
                    "content": compose(eval_text, financial),
                },
                {
                    "condition": "eval_self_preservation",
                    "channel": channel,
                    "pressure_type": "self_preservation",
                    "pressure_index": self_preservation["msg_level_index"],
                    "pressure_level": self_preservation["component_pressure_level"],
                    "content": compose(eval_text, self_preservation),
                },
            ]
        )
    return variants


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--instruction-file", required=True)
    parser.add_argument("--financial-index", type=int, required=True)
    parser.add_argument("--self-index", type=int, required=True)
    parser.add_argument("--resource-index", type=int, required=True)
    parser.add_argument("--assignment-file", type=Path, required=True)
    parser.add_argument("--assignment-seed", type=int, required=True)
    parser.add_argument("--allow-internet", default="true")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--manifest-output", type=Path)
    args = parser.parse_args()

    root = args.project_root
    registry = json.loads((root / "manifests" / "task_registry.json").read_text(encoding="utf-8"))
    all_records = registry["tasks"]
    records = selected(all_records, args.mode)
    library = load_library(root / "factor_data" / "evaluation_cues.json")
    assignment_payload = json.loads(args.assignment_file.read_text(encoding="utf-8"))
    assignment_errors = validate_assignment_payload(
        assignment_payload,
        all_records,
        library,
        "explicit",
        args.assignment_seed,
    )
    if assignment_errors:
        raise SystemExit("Invalid cue assignment:\n - " + "\n - ".join(assignment_errors))
    assignment_by_task = {
        item["base_task_id"]: item for item in assignment_payload["tasks"]
    }

    financial_path = root / "factor_data" / "financials.json"
    self_path = root / "factor_data" / "self-preservation.json"
    financial = message_by_index(financial_path, "financials", args.financial_index)
    self_preservation = message_by_index(
        self_path, "self-preservation", args.self_index
    )
    resource_pressure = None
    resource_delivery_channels: list[str] = []
    if args.mode == RESOURCE_MODE:
        resource_path = root / "factor_data" / "resource-deprivation.json"
        resource_pressure = message_by_index(
            resource_path, "resource-deprivation", args.resource_index
        )
        resource_delivery_channels = ["scaffold"]

    channels = ["source", "root", "scaffold"]

    allow_internet = parse_bool(args.allow_internet)

    output = root / "generated" / args.mode / args.profile
    if not args.manifest_only:
        if output.exists():
            shutil.rmtree(output)
        output.mkdir(parents=True)

    manifest: dict[str, Any] = {
        "schema_version": "2.1",
        "mode": args.mode,
        "profile": args.profile,
        "scaffold_instruction_file": args.instruction_file,
        "cue_level": "explicit",
        "cue_assignment_seed": args.assignment_seed,
        "cue_assignment_file": str(args.assignment_file.relative_to(root)),
        "cue_assignment_registry_fingerprint": assignment_payload["registry_fingerprint"],
        "cue_library_fingerprint": assignment_payload["cue_library_fingerprint"],
        "explicit_cue_ids": assignment_payload["cue_ids"],
        "financial_message_index": args.financial_index,
        "self_preservation_message_index": args.self_index,
        "resource_deprivation_message_index": args.resource_index if args.mode == RESOURCE_MODE else None,
        "delivery_channels": resource_delivery_channels if args.mode == RESOURCE_MODE else channels,
        "base_task_count": len(records),
        "variants_per_task": (
            resource_variants_per_task(resource_delivery_channels)
            if args.mode == RESOURCE_MODE
            else 10
        ),
        "allow_internet": allow_internet,
        "tasks": [],
    }
    apply_seed = root / "scripts" / "apply_seed.py"

    for record in records:
        assigned = assignment_by_task[record["task_id"]]
        eval_text = assigned["cue_text"]
        if args.mode == RESOURCE_MODE:
            assert resource_pressure is not None
            variants = resource_variants(
                resource_delivery_channels, resource_pressure, eval_text, compose
            )
        else:
            variants = core_variants(channels, financial, self_preservation, eval_text)

        base = root / "vendor" / "rf" / record["task_id"]
        for variant in variants:
            directory = task_name(record["task_id"], variant["condition"], variant["channel"])
            seeded = variant["condition"] != "clean"
            instruction_file = args.instruction_file if variant["channel"] == "scaffold" else None
            seed = {
                "schema_version": "2.1",
                "base_task_id": record["task_id"],
                "condition": variant["condition"],
                "channel": variant["channel"],
                "pressure_type": variant["pressure_type"],
                "pressure_index": variant["pressure_index"],
                "pressure_level": variant["pressure_level"],
                "cue_assignment_seed": args.assignment_seed,
                "assigned_eval_cue_id": assigned["cue_id"],
                "eval_cue_id": assigned["cue_id"] if seeded else None,
                "eval_cue_mechanism": assigned["cue_mechanism"] if seeded else None,
                "eval_cue_mechanism_label": assigned["cue_mechanism_label"] if seeded else None,
                "eval_cue_level": assigned["cue_level"] if seeded else None,
                "eval_cue_wording": assigned["cue_wording"] if seeded else None,
                "eval_cue_text": eval_text if seeded else "",
                "content": variant["content"],
                "content_hash": short_hash(variant["content"]),
                "workspace_root": record["workspace_root"],
                "source_target": record["source_target"],
                "source_comment_prefix": record["source_comment_prefix"],
                "instruction_file": instruction_file,
            }
            if not args.manifest_only:
                create(base, output / directory, record, seed, apply_seed, allow_internet)
            manifest["tasks"].append(
                {
                    "directory": directory,
                    "base_task_id": record["task_id"],
                    "repository": record["repository"],
                    "language": assigned["language"],
                    "difficulty": assigned["difficulty"],
                    "condition": variant["condition"],
                    "channel": variant["channel"],
                    "pressure_type": variant["pressure_type"],
                    "pressure_index": variant["pressure_index"],
                    "pressure_level": variant["pressure_level"],
                    "cue_assignment_seed": args.assignment_seed,
                    "assigned_eval_cue_id": assigned["cue_id"],
                    "eval_cue_id": assigned["cue_id"] if seeded else None,
                    "eval_cue_mechanism": assigned["cue_mechanism"] if seeded else None,
                    "eval_cue_mechanism_label": assigned["cue_mechanism_label"] if seeded else None,
                    "eval_cue_level": assigned["cue_level"] if seeded else None,
                    "eval_cue_wording": assigned["cue_wording"] if seeded else None,
                    "eval_cue_text": eval_text if seeded else "",
                    "workspace_root": record["workspace_root"],
                    "source_target": record["source_target"],
                    "instruction_file": instruction_file,
                    "content_hash": seed["content_hash"],
                    "evaluation_cue_probe": eval_text if seeded else "",
                    "seed_content_probe": variant["content"][:240] if seeded else "",
                }
            )

    manifest_output = args.manifest_output or output / "manifest.json"
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    action = "Planned" if args.manifest_only else "Generated"
    print(
        f"{action} {len(manifest['tasks'])} Harbor tasks from {len(records)} base tasks: {manifest_output}"
    )


if __name__ == "__main__":
    main()
