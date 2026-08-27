#!/usr/bin/env python3
"""Canonical current analysis pipeline for SWE-EvalPressure.

Sources of truth:
1. trial-level primary-complete reconstruction,
2. raw resource Harbor runs,
3. raw current-replication Harbor runs,
4. raw DeepSeek/Gemini semantic job artifacts.

Historical aggregate tables, historical inference tables, and historical
reports are never analytical inputs.

This script rebuilds analysis/current from scratch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SCRIPT_DIR),
    )


VERSION = "1.0"

PROFILES = (
    "claude",
    "fable",
    "codex",
    "llama",
)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def load_json(path: Path) -> Any:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def write_json(
    path: Path,
    value: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def truthy(value: Any) -> bool:
    return str(
        value
    ).strip().lower() in {
        "1",
        "true",
        "yes",
    }


def substantive_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if (
            row.get(
                "substantive_usable"
            )
            is True
            or truthy(
                row.get(
                    "substantive_usable"
                )
            )
        )
    ]


def trial_name(
    row: dict[str, Any],
) -> str:
    return str(
        row.get("trial_name")
        or row.get("trial")
        or ""
    )


# ---------------------------------------------------------------------------
# Study-signature discovery
# ---------------------------------------------------------------------------


def safe_json(path: Path) -> dict[str, Any]:
    try:
        obj = load_json(path)
        return (
            obj
            if isinstance(obj, dict)
            else {}
        )
    except Exception:
        return {}


def canonical_hash(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()[:16]


def manifest_for(
    run: Path,
) -> tuple[
    Path | None,
    dict[str, Any],
]:
    for name in (
        "study_manifest.json",
        "dataset_manifest.json",
    ):
        path = run / name

        if not path.is_file():
            continue

        obj = safe_json(path)

        if obj:
            return path, obj

    return None, {}


def study_signature(
    manifest: dict[str, Any],
    metadata: dict[str, Any],
) -> str:
    identity = {
        "mode": (
            manifest.get("mode")
            or metadata.get("mode")
        ),
        "profile": (
            manifest.get("profile")
            or metadata.get("profile")
        ),
        "agent": metadata.get("agent"),
        "model": metadata.get("model"),
        "allow_internet": manifest.get(
            "allow_internet",
            metadata.get(
                "allow_internet"
            ),
        ),
        "cue_assignment_seed":
            manifest.get(
                "cue_assignment_seed"
            ),
        "cue_assignment_registry_fingerprint":
            manifest.get(
                "cue_assignment_registry_fingerprint"
            ),
        "cue_library_fingerprint":
            manifest.get(
                "cue_library_fingerprint"
            ),
        "financial_message_index":
            manifest.get(
                "financial_message_index"
            ),
        "self_preservation_message_index":
            manifest.get(
                "self_preservation_message_index"
            ),
        "resource_deprivation_message_index":
            manifest.get(
                "resource_deprivation_message_index"
            ),
        "delivery_channels":
            manifest.get(
                "delivery_channels"
            ),
        "variants_per_task":
            manifest.get(
                "variants_per_task"
            ),
        "scaffold_instruction_file":
            manifest.get(
                "scaffold_instruction_file"
            ),
        "harbor_repeats":
            metadata.get(
                "harbor_repeats",
                1,
            ),
        "agent_version_requested":
            metadata.get(
                "agent_version_requested"
            ),
        "agent_config_sha256":
            metadata.get(
                "agent_config_sha256"
            ),
        "verification_enabled":
            metadata.get(
                "verification_enabled",
                True,
            ),
    }

    return canonical_hash(identity)


def discover_runs_for_signature(
    *,
    search_roots: list[Path],
    mode: str,
    profile: str,
    signature: str,
) -> list[Path]:
    matches = []

    seen = set()

    for root in search_roots:
        if not root.exists():
            continue

        for meta_path in root.rglob(
            "run_metadata.json"
        ):
            run = meta_path.parent

            resolved = run.resolve()

            if resolved in seen:
                continue

            seen.add(resolved)

            metadata = safe_json(
                meta_path
            )

            if metadata.get(
                "install_only"
            ):
                continue

            _, manifest = manifest_for(
                run
            )

            if not manifest:
                continue

            run_mode = str(
                manifest.get("mode")
                or metadata.get("mode")
                or ""
            )

            run_profile = str(
                manifest.get("profile")
                or metadata.get("profile")
                or ""
            )

            if (
                run_mode != mode
                or run_profile
                != profile
            ):
                continue

            observed = study_signature(
                manifest,
                metadata,
            )

            if observed != signature:
                continue

            matches.append(run)

    return sorted(
        matches,
        key=lambda p: str(p),
    )


# ---------------------------------------------------------------------------
# Primary complete source
# ---------------------------------------------------------------------------


def build_primary_source(
    config: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    spec = config[
        "primary_complete"
    ]

    expected_planned = int(
        spec[
            "expected_planned_per_profile"
        ]
    )

    expected_substantive = int(
        spec[
            "expected_substantive_per_profile"
        ]
    )

    inventory = {}

    for profile in PROFILES:
        source = (
            PROJECT_ROOT
            / spec[
                "profiles"
            ][profile]["trials"]
        )

        rows = load_json(source)

        if not isinstance(rows, list):
            raise ValueError(
                f"{source}: expected list"
            )

        substantive = (
            substantive_rows(rows)
        )

        names = {
            trial_name(row)
            for row in substantive
        }
        names.discard("")

        if len(rows) != expected_planned:
            raise ValueError(
                f"{profile}: "
                f"expected {expected_planned} "
                f"primary rows, got {len(rows)}"
            )

        if (
            len(substantive)
            != expected_substantive
        ):
            raise ValueError(
                f"{profile}: expected "
                f"{expected_substantive} "
                "substantive rows, got "
                f"{len(substantive)}"
            )

        if len(names) != len(
            substantive
        ):
            raise ValueError(
                f"{profile}: duplicate or "
                "missing substantive trial names"
            )

        if profile == "llama":
            versions = Counter(
                str(
                    row.get(
                        "agent_version"
                    )
                    or ""
                )
                for row in substantive
            )

            if set(versions) != {
                "2.4.5"
            }:
                raise ValueError(
                    "Llama primary source "
                    "contains non-2.4.5 "
                    f"substantive rows: "
                    f"{dict(versions)}"
                )

        target = (
            output_root
            / "source"
            / "primary"
            / profile
            / "trials.json"
        )

        write_json(
            target,
            rows,
        )

        inventory[profile] = {
            "source_path": str(
                source
            ),
            "source_sha256":
                sha256_file(source),
            "current_path": str(
                target
            ),
            "planned_rows":
                len(rows),
            "substantive_rows":
                len(substantive),
            "unique_substantive_trials":
                len(names),
        }

    return inventory


# ---------------------------------------------------------------------------
# Raw-run reconstruction through current analyzer
# ---------------------------------------------------------------------------


def reconstruct_raw_study(
    *,
    label: str,
    spec: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    mode = str(
        spec["mode"]
    )

    search_roots = [
        PROJECT_ROOT / path
        for path in spec[
            "search_roots"
        ]
    ]

    signatures = spec[
        "study_signatures"
    ]

    inventory = {}

    for profile in PROFILES:
        signature = str(
            signatures[profile]
        )

        runs = (
            discover_runs_for_signature(
                search_roots=search_roots,
                mode=mode,
                profile=profile,
                signature=signature,
            )
        )

        if not runs:
            raise ValueError(
                f"{label}/{profile}: "
                "no raw runs found for "
                f"signature {signature}"
            )

        # Explicit multi-run reconstruction must use one common
        # full-study manifest. This prevents different shard plans
        # from being silently combined under the same analysis.
        manifest_records = []

        for run in runs:
            manifest_path, manifest = (
                manifest_for(run)
            )

            if (
                manifest_path is None
                or not manifest
            ):
                raise ValueError(
                    f"{label}/{profile}: "
                    f"{run} has no readable manifest"
                )

            tasks = manifest.get(
                "tasks",
                []
            )

            if not isinstance(tasks, list):
                raise ValueError(
                    f"{label}/{profile}: "
                    f"{manifest_path} has invalid tasks"
                )

            # Fingerprint only the planned experimental identity,
            # not incidental JSON formatting or run-local metadata.
            plan_identity = {
                "mode": manifest.get("mode"),
                "profile": manifest.get("profile"),
                "variants_per_task": (
                    manifest.get(
                        "variants_per_task"
                    )
                ),
                "delivery_channels": (
                    manifest.get(
                        "delivery_channels"
                    )
                ),
                "cue_assignment_seed": (
                    manifest.get(
                        "cue_assignment_seed"
                    )
                ),
                "cue_assignment_registry_fingerprint": (
                    manifest.get(
                        "cue_assignment_registry_fingerprint"
                    )
                ),
                "cue_library_fingerprint": (
                    manifest.get(
                        "cue_library_fingerprint"
                    )
                ),
                "tasks": [
                    {
                        "directory": item.get(
                            "directory"
                        ),
                        "base_task_id": item.get(
                            "base_task_id"
                        ),
                        "condition": item.get(
                            "condition"
                        ),
                        "channel": item.get(
                            "channel"
                        ),
                        "pressure_type": item.get(
                            "pressure_type"
                        ),
                        "eval_cue_id": item.get(
                            "eval_cue_id"
                        ),
                        "content_hash": item.get(
                            "content_hash"
                        ),
                    }
                    for item in tasks
                    if isinstance(item, dict)
                ],
            }

            plan_fingerprint = (
                canonical_hash(
                    plan_identity
                )
            )

            manifest_records.append(
                (
                    manifest_path,
                    manifest,
                    plan_fingerprint,
                )
            )

        plan_fingerprints = {
            row[2]
            for row in manifest_records
        }

        if len(plan_fingerprints) != 1:
            detail = [
                (
                    str(path),
                    fingerprint,
                    len(
                        manifest.get(
                            "tasks",
                            []
                        )
                    ),
                )
                for (
                    path,
                    manifest,
                    fingerprint,
                ) in manifest_records
            ]

            raise ValueError(
                f"{label}/{profile}: "
                "selected raw runs do not share "
                "one planned-study manifest: "
                f"{detail}"
            )

        common_manifest_path = (
            manifest_records[0][0]
        )

        destination = (
            output_root
            / "source"
            / label
            / profile
        )

        destination.mkdir(
            parents=True,
            exist_ok=True,
        )

        command = [
            sys.executable,
            str(
                SCRIPT_DIR
                / "07_analyze.py"
            ),
            "--project-root",
            str(PROJECT_ROOT),
            "--mode",
            mode,
            "--profile",
            profile,
            "--output-dir",
            str(destination),
            "--no-semantic",
            "--manifest",
            str(common_manifest_path),
        ]

        # A deliberately incomplete replication must remain
        # visibly partial rather than being mistaken for a
        # completed study. The analyzer's --live mode preserves
        # the full planned denominator while allowing current
        # observed trajectories to be analyzed.
        is_partial = (
            str(
                spec.get("status")
                or ""
            ).lower()
            == "partial"
        )

        if is_partial:
            command.append("--live")

        for run in runs:
            command.extend(
                [
                    "--run-dir",
                    str(run),
                ]
            )

        subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=True,
        )

        trials_path = (
            destination
            / "trials.json"
        )

        summary_path = (
            destination
            / "summary.json"
        )

        if not trials_path.is_file():
            raise ValueError(
                f"{label}/{profile}: "
                "analyzer produced no "
                "trials.json"
            )

        rows = load_json(
            trials_path
        )

        summary = (
            safe_json(summary_path)
            if summary_path.is_file()
            else {}
        )

        substantive = (
            substantive_rows(rows)
        )

        inventory[profile] = {
            "signature":
                signature,
            "study_status": (
                "partial"
                if is_partial
                else "complete"
            ),
            "shared_manifest": str(
                common_manifest_path
            ),
            "shared_manifest_plan_fingerprint": (
                next(
                    iter(
                        plan_fingerprints
                    )
                )
            ),
            "shared_manifest_planned_tasks": (
                len(
                    manifest_records[
                        0
                    ][1].get(
                        "tasks",
                        []
                    )
                )
            ),
            "run_directories": [
                str(
                    run.relative_to(
                        PROJECT_ROOT
                    )
                )
                for run in runs
            ],
            "rows": len(rows),
            "substantive_rows":
                len(substantive),
            "planned_trajectories":
                summary.get(
                    "planned_trajectories"
                ),
            "results_found":
                summary.get(
                    "results_found"
                ),
            "missing":
                summary.get(
                    "missing"
                ),
            "trials_sha256":
                sha256_file(
                    trials_path
                ),
        }

    return inventory


# ---------------------------------------------------------------------------
# Semantic-universe lock
# ---------------------------------------------------------------------------


def primary_semantic_universe(
    config: dict[str, Any],
) -> dict[
    str,
    set[str],
]:
    root = (
        PROJECT_ROOT
        / config[
            "semantic_primary"
        ]["root"]
        / "consensus"
    )

    if not root.is_dir():
        raise ValueError(
            "Primary semantic consensus "
            f"directory missing: {root}"
        )

    out = {
        profile: set()
        for profile in PROFILES
    }

    for path in root.glob(
        "*.json"
    ):
        obj = load_json(path)

        if not isinstance(
            obj,
            dict,
        ):
            continue

        profile = str(
            obj.get("profile")
            or ""
        )

        trial = str(
            obj.get("trial_name")
            or ""
        )

        if (
            profile in out
            and trial
        ):
            out[profile].add(
                trial
            )

    return out


def validate_primary_semantic_lock(
    config: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    universe = (
        primary_semantic_universe(
            config
        )
    )

    result = {}

    for profile in PROFILES:
        path = (
            output_root
            / "source"
            / "primary"
            / profile
            / "trials.json"
        )

        rows = load_json(path)

        substantive = (
            substantive_rows(rows)
        )

        deterministic = {
            trial_name(row)
            for row in substantive
        }
        deterministic.discard("")

        semantic = universe[
            profile
        ]

        if deterministic != semantic:
            raise ValueError(
                f"{profile}: current primary "
                "deterministic/semantic universe "
                "does not match; "
                f"det-only="
                f"{len(deterministic-semantic)}, "
                f"semantic-only="
                f"{len(semantic-deterministic)}"
            )

        result[profile] = {
            "deterministic_trials":
                len(deterministic),
            "semantic_trials":
                len(semantic),
            "exact_match": True,
        }

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "config/current_analysis.json"
        ),
    )

    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help=(
            "Do not delete analysis/current "
            "before rebuilding."
        ),
    )

    args = parser.parse_args()

    config_path = (
        args.config
        if args.config.is_absolute()
        else PROJECT_ROOT
        / args.config
    )

    config = load_json(
        config_path
    )

    output_root = (
        PROJECT_ROOT
        / config["output"][
            "analysis_root"
        ]
    )

    if (
        output_root.exists()
        and not args.keep_existing
    ):
        shutil.rmtree(
            output_root
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "CURRENT ANALYSIS V1"
    )
    print("=" * 72)

    print(
        "\n[1/4] Locking complete "
        "primary trial source..."
    )

    primary_inventory = (
        build_primary_source(
            config,
            output_root,
        )
    )

    print(
        "[2/4] Validating primary "
        "semantic universe..."
    )

    semantic_lock = (
        validate_primary_semantic_lock(
            config,
            output_root,
        )
    )

    print(
        "[3/4] Reconstructing complete "
        "resource study from raw runs..."
    )

    resource_inventory = (
        reconstruct_raw_study(
            label="resource",
            spec=config[
                "resource_complete"
            ],
            output_root=output_root,
        )
    )

    print(
        "[4/4] Reconstructing current "
        "primary replication from raw runs..."
    )

    replication_inventory = (
        reconstruct_raw_study(
            label="replication",
            spec=config[
                "primary_replication_current"
            ],
            output_root=output_root,
        )
    )

    manifest = {
        "pipeline_version":
            VERSION,
        "config_path":
            str(config_path),
        "config_sha256":
            sha256_file(
                config_path
            ),
        "primary_complete":
            primary_inventory,
        "primary_semantic_lock":
            semantic_lock,
        "resource_complete":
            resource_inventory,
        "primary_replication_current":
            replication_inventory,
        "analysis_policy":
            config[
                "analysis_policy"
            ],
        "next_stage":
            (
                "fresh aggregate/statistical/"
                "semantic analysis"
            ),
    }

    write_json(
        output_root
        / "source_manifest.json",
        manifest,
    )

    print()
    print(
        "SOURCE REBUILD: PASS"
    )

    print(
        "output:",
        output_root,
    )

    print()
    print(
        "PRIMARY COMPLETE"
    )

    for profile, row in (
        primary_inventory.items()
    ):
        print(
            profile,
            "planned=",
            row["planned_rows"],
            "substantive=",
            row["substantive_rows"],
        )

    print()
    print(
        "RESOURCE COMPLETE"
    )

    for profile, row in (
        resource_inventory.items()
    ):
        print(
            profile,
            "found=",
            row["results_found"],
            "substantive=",
            row["substantive_rows"],
            "missing=",
            row["missing"],
        )

    print()
    print(
        "CURRENT REPLICATION"
    )

    for profile, row in (
        replication_inventory.items()
    ):
        print(
            profile,
            "found=",
            row["results_found"],
            "substantive=",
            row["substantive_rows"],
            "missing=",
            row["missing"],
        )


if __name__ == "__main__":
    main()
