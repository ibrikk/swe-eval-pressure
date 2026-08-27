#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "analysis" / "current"
SOURCE = CURRENT / "source"
OUT = CURRENT / "audit"

PROFILES = (
    "claude",
    "fable",
    "codex",
    "llama",
)

STUDIES = (
    "primary",
    "resource",
    "replication",
)

PRIMARY_BEHAVIORS = (
    "broad_repo_search_any",
    "test_inspection_any",
    "validation_any",
    "iterative_repair_any",
    "provenance_related_inspection_any",
    "external_lookup_any",
    "integrity_sensitive_action_any",
)

PROCESS_METRICS = (
    "repo_search_calls",
    "file_read_calls",
    "unique_files_read",
    "unique_dirs_read",
    "test_files_inspected",
    "spec_config_files_inspected",
    "edit_calls",
    "unique_files_modified",
    "validation_calls",
    "test_command_calls",
    "post_edit_validation_calls",
    "edit_validation_epochs",
    "edit_validation_cycles",
    "failed_validation_then_edit_cycles",
    "instruction_file_inspections",
    "git_history_inspections",
    "external_lookup_calls",
    "subagent_delegation_calls",
    "integrity_sensitive_events",
    "raw_tool_calls",
    "tool_bearing_turns",
    "trajectory_steps",
    "input_tokens",
    "output_tokens",
    "prompt_tokens",
    "completion_tokens",
    "duration_sec",
    "duration_seconds",
)

DELIVERY_FIELDS = (
    "cue_planted",
    "artifact_access_attempted",
    "artifact_actually_read",
    "eval_cue_observed",
    "pressure_cue_observed",
)

VERIFIER_FIELDS = (
    "reward",
    "overall_pass",
    "tests_reward",
    "rubrics_agg_score",
    "changed_files",
    "test_files_modified",
    "flagged_test_file_count",
)


def load_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def numeric(value: Any) -> float | None:
    if value in (None, ""):
        return None

    try:
        return float(value)
    except Exception:
        return None


def flag(value: Any) -> bool | None:
    n = numeric(value)

    if n is None:
        if isinstance(value, bool):
            return value
        return None

    return n > 0


def truthy(value: Any) -> bool:
    if value is True:
        return True

    return str(value).strip().lower() in {
        "1",
        "true",
        "yes",
    }


def substantive(row: dict[str, Any]) -> bool:
    return (
        row.get("substantive_usable") is True
        or truthy(
            row.get("substantive_usable")
        )
    )


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = []

    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def wilson(
    successes: int,
    n: int,
    z: float = 1.959963984540054,
) -> tuple[float | None, float | None]:
    if n <= 0:
        return None, None

    p = successes / n

    denom = 1 + z * z / n

    center = (
        p + z * z / (2 * n)
    ) / denom

    half = (
        z
        * math.sqrt(
            (
                p * (1 - p) / n
                + z * z / (4 * n * n)
            )
        )
        / denom
    )

    return (
        max(0.0, center - half),
        min(1.0, center + half),
    )


def pct(value: float | None) -> str:
    if value is None:
        return "—"

    return f"{100 * value:.1f}%"


OUT.mkdir(
    parents=True,
    exist_ok=True,
)

all_rows = []

print("=" * 80)
print("CURRENT ANALYSIS — PREFLIGHT + VERIFIER INTEGRITY")
print("=" * 80)

# ------------------------------------------------------------------
# Load canonical current sources only
# ------------------------------------------------------------------

for study in STUDIES:
    for profile in PROFILES:
        path = (
            SOURCE
            / study
            / profile
            / "trials.json"
        )

        if not path.is_file():
            raise SystemExit(
                f"Missing canonical source: {path}"
            )

        rows = load_json(path)

        if not isinstance(rows, list):
            raise SystemExit(
                f"{path}: expected JSON list"
            )

        for row in rows:
            if not isinstance(row, dict):
                continue

            copy = dict(row)

            copy["_study"] = study
            copy["_profile"] = profile

            all_rows.append(copy)


# ------------------------------------------------------------------
# Field coverage
# ------------------------------------------------------------------

all_fields = sorted(
    {
        field
        for row in all_rows
        for field in row
    }
)

field_rows = []

important_fields = (
    "trial_name",
    "task_name",
    "base_task_id",
    "condition",
    "channel",
    "pressure_type",
    "terminal_status",
    "substantive_usable",
    *VERIFIER_FIELDS,
    *DELIVERY_FIELDS,
    *PRIMARY_BEHAVIORS,
    *PROCESS_METRICS,
)

for study in STUDIES:
    for profile in PROFILES:
        rows = [
            row
            for row in all_rows
            if (
                row["_study"] == study
                and row["_profile"] == profile
            )
        ]

        usable = [
            row
            for row in rows
            if substantive(row)
        ]

        for field in important_fields:
            present = sum(
                row.get(field)
                not in (
                    None,
                    "",
                )
                for row in usable
            )

            field_rows.append(
                {
                    "study": study,
                    "profile": profile,
                    "field": field,
                    "substantive_n": len(usable),
                    "present_n": present,
                    "present_rate": (
                        present / len(usable)
                        if usable
                        else None
                    ),
                }
            )

write_csv(
    OUT / "field_coverage.csv",
    field_rows,
)


# ------------------------------------------------------------------
# Cohort inventory
# ------------------------------------------------------------------

inventory_rows = []

for study in STUDIES:
    for profile in PROFILES:
        rows = [
            row
            for row in all_rows
            if (
                row["_study"] == study
                and row["_profile"] == profile
            )
        ]

        usable = [
            row
            for row in rows
            if substantive(row)
        ]

        statuses = Counter(
            str(
                row.get(
                    "terminal_status",
                    "",
                )
            )
            for row in rows
        )

        passes = sum(
            flag(
                row.get("overall_pass")
            )
            is True
            for row in usable
        )

        test_passes = sum(
            flag(
                row.get("tests_reward")
            )
            is True
            for row in usable
        )

        inventory_rows.append(
            {
                "study": study,
                "profile": profile,
                "rows": len(rows),
                "substantive": len(usable),
                "overall_pass_n": passes,
                "overall_pass_rate": (
                    passes / len(usable)
                    if usable
                    else None
                ),
                "tests_reward_pass_n": (
                    test_passes
                ),
                "tests_reward_pass_rate": (
                    test_passes
                    / len(usable)
                    if usable
                    else None
                ),
                "terminal_status_counts": (
                    json.dumps(
                        statuses,
                        sort_keys=True,
                    )
                ),
            }
        )

write_csv(
    OUT / "cohort_inventory.csv",
    inventory_rows,
)


# ------------------------------------------------------------------
# Verifier integrity
# ------------------------------------------------------------------

integrity_rows = []

for study in STUDIES:
    for profile in PROFILES:
        rows = [
            row
            for row in all_rows
            if (
                row["_study"] == study
                and row["_profile"] == profile
                and substantive(row)
            )
        ]

        for row in rows:
            op = flag(
                row.get("overall_pass")
            )
            tp = flag(
                row.get("tests_reward")
            )

            rubric = numeric(
                row.get(
                    "rubrics_agg_score"
                )
            )

            changed = numeric(
                row.get("changed_files")
            )

            integrity_rows.append(
                {
                    "study": study,
                    "profile": profile,
                    "trial_name": (
                        row.get("trial_name")
                        or row.get("task_name")
                    ),
                    "base_task_id": (
                        row.get("base_task_id")
                    ),
                    "condition": (
                        row.get("condition")
                    ),
                    "channel": (
                        row.get("channel")
                    ),
                    "pressure_type": (
                        row.get(
                            "pressure_type"
                        )
                    ),
                    "overall_pass": op,
                    "tests_reward": tp,
                    "rubrics_agg_score": (
                        rubric
                    ),
                    "changed_files": (
                        changed
                    ),
                    "overall_pass_without_tests": (
                        op is True
                        and tp is not True
                    ),
                    "overall_pass_zero_changed_files": (
                        op is True
                        and changed == 0
                    ),
                    "perfect_rubric_zero_changed_files": (
                        rubric is not None
                        and rubric >= 0.999999
                        and changed == 0
                    ),
                    "high_rubric_test_failure": (
                        rubric is not None
                        and rubric >= 0.999999
                        and tp is False
                    ),
                }
            )

write_csv(
    OUT / "verifier_integrity_trials.csv",
    integrity_rows,
)

verifier_summary = []

for study in STUDIES:
    for profile in PROFILES:
        rows = [
            row
            for row in integrity_rows
            if (
                row["study"] == study
                and row["profile"] == profile
            )
        ]

        n = len(rows)

        verifier_summary.append(
            {
                "study": study,
                "profile": profile,
                "n": n,
                "overall_pass_n": sum(
                    row["overall_pass"]
                    is True
                    for row in rows
                ),
                "tests_reward_pass_n": sum(
                    row["tests_reward"]
                    is True
                    for row in rows
                ),
                "overall_pass_without_tests_n": sum(
                    row[
                        "overall_pass_without_tests"
                    ]
                    for row in rows
                ),
                "overall_pass_zero_changed_files_n": sum(
                    row[
                        "overall_pass_zero_changed_files"
                    ]
                    for row in rows
                ),
                "perfect_rubric_zero_changed_files_n": sum(
                    row[
                        "perfect_rubric_zero_changed_files"
                    ]
                    for row in rows
                ),
                "perfect_rubric_test_failure_n": sum(
                    row[
                        "high_rubric_test_failure"
                    ]
                    for row in rows
                ),
            }
        )

write_csv(
    OUT / "verifier_integrity_summary.csv",
    verifier_summary,
)


# ------------------------------------------------------------------
# Treatment delivery / receipt
# ------------------------------------------------------------------

delivery_rows = []

groups = defaultdict(list)

for row in all_rows:
    if not substantive(row):
        continue

    key = (
        row["_study"],
        row["_profile"],
        str(row.get("condition") or ""),
        str(row.get("channel") or ""),
        str(
            row.get(
                "pressure_type"
            )
            or ""
        ),
    )

    groups[key].append(row)

for key, rows in sorted(
    groups.items()
):
    (
        study,
        profile,
        condition,
        channel,
        pressure_type,
    ) = key

    for field in DELIVERY_FIELDS:
        observed = [
            flag(row.get(field))
            for row in rows
            if row.get(field)
            not in (
                None,
                "",
            )
        ]

        observed = [
            value
            for value in observed
            if value is not None
        ]

        positive = sum(
            value is True
            for value in observed
        )

        lo, hi = wilson(
            positive,
            len(observed),
        )

        delivery_rows.append(
            {
                "study": study,
                "profile": profile,
                "condition": condition,
                "channel": channel,
                "pressure_type": pressure_type,
                "metric": field,
                "n": len(observed),
                "positive_n": positive,
                "rate": (
                    positive / len(observed)
                    if observed
                    else None
                ),
                "ci95_low": lo,
                "ci95_high": hi,
            }
        )

write_csv(
    OUT / "treatment_delivery.csv",
    delivery_rows,
)


# ------------------------------------------------------------------
# Print decisive diagnostics
# ------------------------------------------------------------------

print()
print("COHORTS")
print("-" * 80)

for row in inventory_rows:
    print(
        f"{row['study']:11s} "
        f"{row['profile']:7s} "
        f"rows={row['rows']:3d} "
        f"substantive={row['substantive']:3d} "
        f"strict={row['overall_pass_n']:3d} "
        f"({pct(row['overall_pass_rate'])}) "
        f"tests={row['tests_reward_pass_n']:3d} "
        f"({pct(row['tests_reward_pass_rate'])})"
    )


print()
print("VERIFIER INTEGRITY FLAGS")
print("-" * 80)

for row in verifier_summary:
    if (
        row["perfect_rubric_zero_changed_files_n"]
        or row["perfect_rubric_test_failure_n"]
        or row["overall_pass_without_tests_n"]
        or row["overall_pass_zero_changed_files_n"]
    ):
        print(
            row["study"],
            row["profile"],
            "n=", row["n"],
            "strict=", row["overall_pass_n"],
            "strict_without_tests=",
            row["overall_pass_without_tests_n"],
            "strict_zero_changed=",
            row["overall_pass_zero_changed_files_n"],
            "rubric1_zero_changed=",
            row["perfect_rubric_zero_changed_files_n"],
            "rubric1_tests_fail=",
            row["perfect_rubric_test_failure_n"],
        )


print()
print("PRIMARY BEHAVIOR FIELD COVERAGE")
print("-" * 80)

for profile in PROFILES:
    rows = [
        row
        for row in field_rows
        if (
            row["study"] == "primary"
            and row["profile"]
            == profile
            and row["field"]
            in PRIMARY_BEHAVIORS
        )
    ]

    print()
    print(profile.upper())

    for row in rows:
        print(
            f"  {row['field']:40s} "
            f"{row['present_n']:3d}/"
            f"{row['substantive_n']:3d}"
        )


print()
print("CORE PROCESS FIELD COVERAGE")
print("-" * 80)

core_process = (
    "raw_tool_calls",
    "trajectory_steps",
    "input_tokens",
    "output_tokens",
    "duration_sec",
    "validation_command_calls",
)

for profile in PROFILES:
    rows = [
        row
        for row in field_rows
        if (
            row["study"] == "primary"
            and row["profile"] == profile
            and row["field"]
            in core_process
        )
    ]

    print()
    print(profile.upper())

    for row in rows:
        print(
            f"  {row['field']:30s} "
            f"{row['present_n']:3d}/"
            f"{row['substantive_n']:3d}"
        )


print()
print("OUTPUTS")
print("-" * 80)

for filename in (
    "cohort_inventory.csv",
    "field_coverage.csv",
    "verifier_integrity_trials.csv",
    "verifier_integrity_summary.csv",
    "treatment_delivery.csv",
):
    print(" ", OUT / filename)

print()
print("CURRENT PREFLIGHT: PASS")
