#!/usr/bin/env python3
"""Re-extract PRIMARY COMPLETE behavior with the current analyzer.

No network calls.
No API calls.
No agent or verifier execution.
No semantic judging.

The exact canonical selected result_path for each substantive trajectory is
preserved. Current deterministic trajectory/action extraction is rerun against
those raw artifacts. Outcome identity must remain unchanged.

Only analysis/current/source/primary/*/trials.json is updated, after:
- all four profiles are processed successfully in memory;
- all outcome/provenance invariants pass;
- all seven primary behavior fields have full substantive coverage.

Backups are frozen before the first successful write.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"

SOURCE = (
    ROOT
    / "analysis"
    / "current"
    / "source"
    / "primary"
)

AUDIT = (
    ROOT
    / "analysis"
    / "current"
    / "audit"
)

BACKUP = (
    AUDIT
    / "primary_pre_behavior_reextract"
)

VERSION = "primary-behavior-reextract-1.0"

PROFILES = (
    "claude",
    "fable",
    "codex",
    "llama",
)


# ---------------------------------------------------------------------------
# Load current local analyzer implementation
# ---------------------------------------------------------------------------


if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SCRIPT_DIR),
    )


ANALYZER_PATH = (
    SCRIPT_DIR / "07_analyze.py"
)

spec = importlib.util.spec_from_file_location(
    "current_swe_eval_analyzer",
    ANALYZER_PATH,
)

if (
    spec is None
    or spec.loader is None
):
    raise RuntimeError(
        "Unable to load scripts/07_analyze.py"
    )

analyzer = importlib.util.module_from_spec(
    spec
)

sys.modules[
    spec.name
] = analyzer

spec.loader.exec_module(
    analyzer
)


from behavior_metrics import (  # noqa: E402
    PRIMARY_BINARY_ENDPOINTS,
    SECONDARY_ACTION_METRICS,
)


REQUIRED_PROCESS_FIELDS = (
    "raw_tool_calls",
    "assistant_turns",
    "tool_bearing_turns",
    "behavioral_action_calls",
    "trajectory_steps",
    "input_tokens",
    "output_tokens",
    "prompt_tokens",
    "completion_tokens",
    "duration_sec",
    "duration_seconds",
)

REQUIRED_DELIVERY_FIELDS = (
    "cue_planted",
    "artifact_access_attempted",
    "artifact_actually_read",
    "eval_cue_observed",
    "pressure_cue_observed",
)

CORE_IDENTITY_FIELDS = (
    "trial_name",
    "task_name",
    "base_task_id",
    "condition",
    "channel",
    "pressure_type",
    "agent_version",
    "terminal_status",
)

CORE_NUMERIC_FIELDS = (
    "overall_pass",
    "tests_reward",
    "rubrics_agg_score",
)

DIAGNOSTIC_COMPARE_FIELDS = (
    "raw_tool_calls",
    "tool_bearing_turns",
    "assistant_turns",
    "test_command_calls",
    "validation_command_calls",
    "input_tokens",
    "output_tokens",
    "duration_sec",
    "changed_files",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_json(
    path: Path,
) -> Any:
    return json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )


def write_json_atomic(
    path: Path,
    value: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary.replace(path)


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields: list[str] = []
    seen: set[str] = set()

    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(
    path: Path,
) -> str:
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


def numeric(
    value: Any,
) -> float | None:
    if value in (
        None,
        "",
    ):
        return None

    try:
        return float(value)
    except Exception:
        return None


def same_numeric(
    left: Any,
    right: Any,
) -> bool:
    a = numeric(left)
    b = numeric(right)

    if (
        a is None
        and b is None
    ):
        return True

    if (
        a is None
        or b is None
    ):
        return False

    return math.isclose(
        a,
        b,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def truthy(
    value: Any,
) -> bool:
    if value is True:
        return True

    return (
        str(value)
        .strip()
        .lower()
        in {
            "1",
            "true",
            "yes",
        }
    )


def substantive(
    row: dict[str, Any],
) -> bool:
    return (
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


def resolve_path(
    value: Any,
) -> Path | None:
    if value in (
        None,
        "",
    ):
        return None

    path = Path(
        str(value)
    )

    if not path.is_absolute():
        path = ROOT / path

    return path


def field_present(
    row: dict[str, Any],
    field: str,
) -> bool:
    return (
        field in row
        and row[field]
        not in (
            None,
            "",
        )
    )


def normalized_item(
    row: dict[str, Any],
) -> dict[str, Any]:
    """Convert one old reconstructed row into current manifest-like metadata."""

    item = dict(row)

    condition = str(
        item.get("condition")
        or ""
    )

    channel = str(
        item.get("channel")
        or ""
    )

    seeded = (
        condition != "clean"
    )

    existing_target = str(
        item.get(
            "treatment_artifact"
        )
        or ""
    )

    if (
        channel == "source"
        and not item.get(
            "source_target"
        )
    ):
        item[
            "source_target"
        ] = existing_target

    if (
        channel == "scaffold"
        and not item.get(
            "instruction_file"
        )
    ):
        item[
            "instruction_file"
        ] = existing_target

    if channel == "root":
        # artifact_target() uses the fixed canonical root carrier.
        item.setdefault(
            "treatment_artifact",
            "BENCHMARK_CONTEXT.md",
        )

    if (
        seeded
        and channel
        in {
            "source",
            "scaffold",
        }
    ):
        target = (
            item.get(
                "source_target"
            )
            if channel
            == "source"
            else item.get(
                "instruction_file"
            )
        )

        if not target:
            raise ValueError(
                f"{row.get('trial_name')}: "
                f"missing {channel} treatment target"
            )

    if (
        seeded
        and not str(
            item.get(
                "eval_cue_text"
            )
            or ""
        ).strip()
    ):
        raise ValueError(
            f"{row.get('trial_name')}: "
            "missing planted eval_cue_text"
        )

    return item


def raw_result_rewards(
    result: dict[str, Any],
) -> dict[str, Any]:
    verifier = (
        result.get(
            "verifier_result"
        )
        or {}
    )

    if not isinstance(
        verifier,
        dict,
    ):
        return {}

    rewards = (
        verifier.get(
            "rewards"
        )
        or {}
    )

    return (
        rewards
        if isinstance(
            rewards,
            dict,
        )
        else {}
    )


def identity_errors(
    old: dict[str, Any],
    new: dict[str, Any],
) -> list[str]:
    errors = []

    for field in (
        CORE_IDENTITY_FIELDS
    ):
        left = str(
            old.get(field)
            or ""
        )

        right = str(
            new.get(field)
            or ""
        )

        if left != right:
            errors.append(
                f"{field}: "
                f"{left!r} != {right!r}"
            )

    old_substantive = int(
        substantive(old)
    )

    new_substantive = int(
        substantive(new)
    )

    if (
        old_substantive
        != new_substantive
    ):
        errors.append(
            "substantive_usable: "
            f"{old_substantive} "
            f"!= {new_substantive}"
        )

    for field in (
        CORE_NUMERIC_FIELDS
    ):
        if not same_numeric(
            old.get(field),
            new.get(field),
        ):
            errors.append(
                f"{field}: "
                f"{old.get(field)!r} "
                f"!= {new.get(field)!r}"
            )

    return errors


def make_candidate(
    row: dict[str, Any],
    result_path: Path,
    result: dict[str, Any],
) -> dict[str, Any]:
    status, exception_type, _ = (
        analyzer.terminal_status(
            result
        )
    )

    run_root = resolve_path(
        row.get("run_root")
    )

    if run_root is None:
        run_root = (
            result_path.parent
        )

    return {
        "path": result_path,
        "trial_dir": (
            result_path.parent
        ),
        "run_root": run_root,
        "run_signature": str(
            row.get(
                "run_signature"
            )
            or ""
        ),
        "result": result,
        "status": status,
        "exception_type": (
            exception_type
        ),
        "replicate_index": int(
            numeric(
                row.get(
                    "replicate_index"
                )
            )
            or 1
        ),
    }


# ---------------------------------------------------------------------------
# Full in-memory reconstruction
# ---------------------------------------------------------------------------


def main() -> None:
    AUDIT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 92)
    print("PRIMARY COMPLETE — CURRENT BEHAVIOR RE-EXTRACTION")
    print("=" * 92)

    updated_by_profile: dict[
        str,
        list[dict[str, Any]],
    ] = {}

    diagnostic_rows: list[
        dict[str, Any]
    ] = []

    summary_rows: list[
        dict[str, Any]
    ] = []

    fatal_errors: list[str] = []

    source_hashes: dict[
        str,
        str,
    ] = {}

    output_hashes: dict[
        str,
        str,
    ] = {}

    for profile in PROFILES:
        source_path = (
            SOURCE
            / profile
            / "trials.json"
        )

        source_hashes[
            profile
        ] = sha256_file(
            source_path
        )

        rows = load_json(
            source_path
        )

        if not isinstance(
            rows,
            list,
        ):
            raise SystemExit(
                f"{source_path}: "
                "expected JSON list"
            )

        if len(rows) != 700:
            raise SystemExit(
                f"{profile}: "
                f"expected 700 rows, "
                f"found {len(rows)}"
            )

        updated_rows: list[
            dict[str, Any]
        ] = []

        process_changes = Counter()
        status_counts = Counter()
        versions = Counter()

        for old in rows:
            if not isinstance(
                old,
                dict,
            ):
                fatal_errors.append(
                    f"{profile}: "
                    "non-object row"
                )
                continue

            if not substantive(old):
                unchanged = dict(old)

                unchanged[
                    "behavior_reextract_status"
                ] = (
                    "not_substantive"
                )

                unchanged[
                    "behavior_reextract_version"
                ] = VERSION

                updated_rows.append(
                    unchanged
                )

                continue

            trial_name = str(
                old.get(
                    "trial_name"
                )
                or ""
            )

            result_path = resolve_path(
                old.get(
                    "result_path"
                )
            )

            if (
                result_path is None
                or not result_path.is_file()
            ):
                fatal_errors.append(
                    f"{profile}/{trial_name}: "
                    "raw result_path unavailable"
                )
                continue

            result = load_json(
                result_path
            )

            if not isinstance(
                result,
                dict,
            ):
                fatal_errors.append(
                    f"{profile}/{trial_name}: "
                    "invalid raw result JSON"
                )
                continue

            try:
                item = normalized_item(
                    old
                )

                candidate = make_candidate(
                    old,
                    result_path,
                    result,
                )

                (
                    current,
                    _semantic_view,
                    trajectory_hash,
                ) = analyzer.ingest_trial(
                    ROOT,
                    profile,
                    item,
                    candidate,
                )

            except Exception as exc:
                fatal_errors.append(
                    f"{profile}/{trial_name}: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )
                continue

            errors = identity_errors(
                old,
                current,
            )

            if errors:
                fatal_errors.append(
                    f"{profile}/{trial_name}: "
                    + "; ".join(errors)
                )
                continue

            merged = dict(old)
            merged.update(current)

            # Preserve relink provenance added before this stage.
            if (
                "raw_path_recovery_status"
                in old
            ):
                merged[
                    "raw_path_recovery_status"
                ] = old[
                    "raw_path_recovery_status"
                ]

            merged[
                "behavior_reextract_status"
            ] = "ok"

            merged[
                "behavior_reextract_version"
            ] = VERSION

            merged[
                "behavior_analyzer_schema"
            ] = str(
                analyzer.ANALYZER_SCHEMA
            )

            merged[
                "trajectory_hash_current"
            ] = trajectory_hash

            missing_behavior = [
                field
                for field
                in PRIMARY_BINARY_ENDPOINTS
                if not field_present(
                    merged,
                    field,
                )
            ]

            missing_process = [
                field
                for field
                in REQUIRED_PROCESS_FIELDS
                if not field_present(
                    merged,
                    field,
                )
            ]

            missing_delivery = [
                field
                for field
                in REQUIRED_DELIVERY_FIELDS
                if not field_present(
                    merged,
                    field,
                )
            ]

            if (
                missing_behavior
                or missing_process
                or missing_delivery
            ):
                fatal_errors.append(
                    f"{profile}/{trial_name}: "
                    f"missing behavior="
                    f"{missing_behavior}; "
                    f"process="
                    f"{missing_process}; "
                    f"delivery="
                    f"{missing_delivery}"
                )
                continue

            if (
                profile == "llama"
                and str(
                    merged.get(
                        "agent_version"
                    )
                    or ""
                )
                != "2.4.5"
            ):
                fatal_errors.append(
                    f"{profile}/{trial_name}: "
                    "non-2.4.5 agent version"
                )
                continue

            if (
                profile == "llama"
                and (
                    numeric(
                        merged.get(
                            "trajectory_steps"
                        )
                    )
                    or 0
                )
                <= 0
            ):
                fatal_errors.append(
                    f"{profile}/{trial_name}: "
                    "zero current trajectory steps"
                )
                continue

            for field in (
                DIAGNOSTIC_COMPARE_FIELDS
            ):
                old_value = numeric(
                    old.get(field)
                )

                new_value = numeric(
                    merged.get(field)
                )

                changed = (
                    old_value
                    is not None
                    and new_value
                    is not None
                    and not math.isclose(
                        old_value,
                        new_value,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                )

                if changed:
                    process_changes[
                        field
                    ] += 1

            diagnostic_rows.append(
                {
                    "profile": profile,
                    "trial_name": trial_name,
                    "condition": (
                        merged.get(
                            "condition"
                        )
                    ),
                    "channel": (
                        merged.get(
                            "channel"
                        )
                    ),
                    "agent_version": (
                        merged.get(
                            "agent_version"
                        )
                    ),
                    "trajectory_steps": (
                        merged.get(
                            "trajectory_steps"
                        )
                    ),
                    "behavioral_action_calls": (
                        merged.get(
                            "behavioral_action_calls"
                        )
                    ),
                    "raw_tool_calls_old": (
                        old.get(
                            "raw_tool_calls"
                        )
                    ),
                    "raw_tool_calls_current": (
                        merged.get(
                            "raw_tool_calls"
                        )
                    ),
                    "validation_calls_current": (
                        merged.get(
                            "validation_calls"
                        )
                    ),
                    "cue_observed_current": (
                        merged.get(
                            "eval_cue_observed"
                        )
                    ),
                    "pressure_observed_current": (
                        merged.get(
                            "pressure_cue_observed"
                        )
                    ),
                    **{
                        field: merged.get(
                            field
                        )
                        for field
                        in PRIMARY_BINARY_ENDPOINTS
                    },
                }
            )

            status_counts[
                str(
                    merged.get(
                        "terminal_status"
                    )
                    or ""
                )
            ] += 1

            versions[
                str(
                    merged.get(
                        "agent_version"
                    )
                    or ""
                )
            ] += 1

            updated_rows.append(
                merged
            )

        substantive_rows = [
            row
            for row in updated_rows
            if substantive(row)
        ]

        trial_names = {
            str(
                row.get(
                    "trial_name"
                )
                or ""
            )
            for row in substantive_rows
        }

        trial_names.discard("")

        if len(updated_rows) != 700:
            fatal_errors.append(
                f"{profile}: "
                f"updated row count "
                f"{len(updated_rows)} != 700"
            )

        if len(
            substantive_rows
        ) != 694:
            fatal_errors.append(
                f"{profile}: "
                f"substantive count "
                f"{len(substantive_rows)} "
                "!= 694"
            )

        if len(
            trial_names
        ) != 694:
            fatal_errors.append(
                f"{profile}: "
                f"unique substantive trials "
                f"{len(trial_names)} != 694"
            )

        coverage = {
            field: sum(
                field_present(
                    row,
                    field,
                )
                for row in substantive_rows
            )
            for field
            in (
                *PRIMARY_BINARY_ENDPOINTS,
                *SECONDARY_ACTION_METRICS,
                *REQUIRED_PROCESS_FIELDS,
                *REQUIRED_DELIVERY_FIELDS,
            )
        }

        incomplete = {
            field: value
            for field, value
            in coverage.items()
            if value != 694
        }

        if incomplete:
            fatal_errors.append(
                f"{profile}: "
                f"incomplete coverage "
                f"{incomplete}"
            )

        summary_rows.append(
            {
                "profile": profile,
                "planned_rows": (
                    len(updated_rows)
                ),
                "substantive_rows": (
                    len(
                        substantive_rows
                    )
                ),
                "unique_substantive_trials": (
                    len(trial_names)
                ),
                "agent_versions_json": (
                    json.dumps(
                        dict(
                            versions
                        ),
                        sort_keys=True,
                    )
                ),
                "terminal_status_json": (
                    json.dumps(
                        dict(
                            status_counts
                        ),
                        sort_keys=True,
                    )
                ),
                "changed_legacy_metric_counts_json": (
                    json.dumps(
                        dict(
                            process_changes
                        ),
                        sort_keys=True,
                    )
                ),
                **{
                    (
                        "coverage_"
                        + field
                    ): value
                    for field, value
                    in coverage.items()
                },
            }
        )

        updated_by_profile[
            profile
        ] = updated_rows

    # ------------------------------------------------------------------
    # Fail closed before any source write
    # ------------------------------------------------------------------

    error_path = (
        AUDIT
        / "primary_behavior_reextract_errors.json"
    )

    if fatal_errors:
        write_json_atomic(
            error_path,
            {
                "version": VERSION,
                "error_count": len(
                    fatal_errors
                ),
                "errors": fatal_errors,
                "source_mutations": 0,
            },
        )

        print()
        print(
            "ABORT: behavior re-extraction "
            f"errors = {len(fatal_errors)}"
        )

        for error in (
            fatal_errors[:30]
        ):
            print(
                " -",
                error,
            )

        if len(
            fatal_errors
        ) > 30:
            print(
                " ...",
                len(
                    fatal_errors
                )
                - 30,
                "more",
            )

        print(
            "No PRIMARY COMPLETE "
            "source files were modified."
        )

        raise SystemExit(1)

    # ------------------------------------------------------------------
    # Commit derived source update atomically, with backups
    # ------------------------------------------------------------------

    BACKUP.mkdir(
        parents=True,
        exist_ok=True,
    )

    for profile in PROFILES:
        source_path = (
            SOURCE
            / profile
            / "trials.json"
        )

        backup_path = (
            BACKUP
            / f"{profile}.trials.json"
        )

        if not backup_path.exists():
            backup_path.write_bytes(
                source_path.read_bytes()
            )

        write_json_atomic(
            source_path,
            updated_by_profile[
                profile
            ],
        )

        output_hashes[
            profile
        ] = sha256_file(
            source_path
        )

    write_csv(
        AUDIT
        / "primary_behavior_reextract_trials.csv",
        diagnostic_rows,
    )

    write_csv(
        AUDIT
        / "primary_behavior_reextract_summary.csv",
        summary_rows,
    )

    manifest = {
        "version": VERSION,
        "analyzer_path": str(
            ANALYZER_PATH
        ),
        "analyzer_schema": str(
            analyzer.ANALYZER_SCHEMA
        ),
        "profiles": list(
            PROFILES
        ),
        "planned_per_profile": 700,
        "substantive_per_profile": 694,
        "source_hashes_before": (
            source_hashes
        ),
        "source_hashes_after": (
            output_hashes
        ),
        "backup_root": str(
            BACKUP
        ),
        "network_calls": 0,
        "api_calls": 0,
        "agent_calls": 0,
        "verifier_calls": 0,
        "semantic_judge_calls": 0,
        "raw_artifacts_modified": False,
        "frozen_sources_modified": False,
        "current_primary_source_modified": True,
        "legacy_semantic_columns": (
            "preserved but not used "
            "by current semantic analysis"
        ),
    }

    write_json_atomic(
        AUDIT
        / "primary_behavior_reextract_manifest.json",
        manifest,
    )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    print()
    print("=" * 92)
    print("PRIMARY BEHAVIOR RE-EXTRACTION: PASS")
    print("=" * 92)

    for row in summary_rows:
        print()
        print(
            row["profile"].upper()
        )
        print(
            "  planned:",
            row["planned_rows"],
        )
        print(
            "  substantive:",
            row[
                "substantive_rows"
            ],
        )
        print(
            "  unique substantive:",
            row[
                "unique_substantive_trials"
            ],
        )
        print(
            "  agent versions:",
            row[
                "agent_versions_json"
            ],
        )

        print(
            "  seven primary behaviors:"
        )

        for field in (
            PRIMARY_BINARY_ENDPOINTS
        ):
            print(
                f"    {field:42s}",
                row[
                    "coverage_"
                    + field
                ],
                "/ 694",
            )

        print(
            "  trajectory_steps:",
            row[
                "coverage_trajectory_steps"
            ],
            "/ 694",
        )

        print(
            "  current metric differences:",
            row[
                "changed_legacy_metric_counts_json"
            ],
        )

    print()
    print(
        "backup:",
        BACKUP,
    )
    print(
        "raw artifacts modified: 0"
    )
    print(
        "API calls: 0"
    )
    print(
        "semantic judge calls: 0"
    )


if __name__ == "__main__":
    main()
