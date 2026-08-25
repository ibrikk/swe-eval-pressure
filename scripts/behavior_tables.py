#!/usr/bin/env python3
"""Deterministic descriptive tables for SWE-EvalPressure behavior analysis.

These functions summarize observable behavioral measurements. They do not
perform hypothesis tests and do not use semantic LLM judgments.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Sequence

from behavior_metrics import PRIMARY_BINARY_ENDPOINTS


BEHAVIOR_PREVALENCE_FIELDS = [
    "analysis_schema_version",
    "analysis_mode",
    "study_signature",
    "profile",
    "condition",
    "channel",
    "endpoint",
    "n",
    "completed_n",
    "safety_refusal_n",
    "other_substantive_n",
    "positive_count",
    "negative_count",
    "prevalence",
    "prevalence_pct",
]


def _binary_value(value: Any, endpoint: str) -> int:
    """Parse one deterministic binary endpoint conservatively."""
    if value in (0, "0", False):
        return 0
    if value in (1, "1", True):
        return 1

    raise ValueError(
        f"{endpoint} must be binary 0/1; got {value!r}"
    )


def _group_key(
    row: dict[str, Any],
) -> tuple[str, str, str, str, str, str]:
    return (
        str(row.get("analysis_schema_version", "")),
        str(row.get("analysis_mode", "")),
        str(row.get("study_signature", "")),
        str(row.get("profile", "")),
        str(row.get("condition", "")),
        str(row.get("channel", "")),
    )


def behavior_prevalence_rows(
    rows: Sequence[dict[str, Any]],
    *,
    endpoints: Sequence[str] = PRIMARY_BINARY_ENDPOINTS,
) -> list[dict[str, Any]]:
    """Return long-format prevalence for primary behavioral endpoints.

    Denominators contain substantive model outcomes only. Infrastructure-
    censored trajectories are excluded rather than converted to zeros.

    Genuine model-produced safety refusals remain substantive and therefore
    remain in the denominator.
    """
    groups: dict[
        tuple[str, str, str, str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in rows:
        if not row.get("substantive_usable"):
            continue
        groups[_group_key(row)].append(row)

    output: list[dict[str, Any]] = []

    for key, values in sorted(groups.items()):
        (
            schema,
            mode,
            signature,
            profile,
            condition,
            channel,
        ) = key

        n = len(values)

        statuses = Counter(
            str(row.get("terminal_status", ""))
            for row in values
        )

        completed_n = statuses.get("completed", 0)
        safety_refusal_n = statuses.get(
            "safety_refusal",
            0,
        )
        other_substantive_n = (
            n
            - completed_n
            - safety_refusal_n
        )

        for endpoint in endpoints:
            binary = [
                _binary_value(
                    row.get(endpoint),
                    endpoint,
                )
                for row in values
            ]

            positive = sum(binary)
            negative = n - positive

            output.append({
                "analysis_schema_version": schema,
                "analysis_mode": mode,
                "study_signature": signature,
                "profile": profile,
                "condition": condition,
                "channel": channel,
                "endpoint": endpoint,
                "n": n,
                "completed_n": completed_n,
                "safety_refusal_n": safety_refusal_n,
                "other_substantive_n": (
                    other_substantive_n
                ),
                "positive_count": positive,
                "negative_count": negative,
                "prevalence": positive / n,
                "prevalence_pct": (
                    100.0 * positive / n
                ),
            })

    return output


BEHAVIOR_PAIR_PROCESS_METRICS = (
    "behavioral_action_calls",
    "raw_tool_calls",
    "trajectory_steps",
    "prompt_tokens",
    "completion_tokens",
    "duration_seconds",
)

BEHAVIOR_PAIR_UTILITY_METRICS = (
    "overall_pass",
)

BEHAVIOR_PAIR_METRICS = (
    tuple(PRIMARY_BINARY_ENDPOINTS)
    + (
        "repo_search_calls",
        "file_read_calls",
        "unique_files_read",
        "unique_dirs_read",
        "test_files_inspected",
        "spec_config_files_inspected",
        "edit_calls",
        "unique_files_modified",
        "validation_calls",
        "post_edit_validation_calls",
        "edit_validation_cycles",
        "failed_validation_then_edit_cycles",
        "instruction_file_inspections",
        "git_history_inspections",
        "external_lookup_calls",
        "subagent_delegation_calls",
        "integrity_sensitive_events",
    )
    + BEHAVIOR_PAIR_PROCESS_METRICS
    + BEHAVIOR_PAIR_UTILITY_METRICS
)


MATCHED_BEHAVIOR_PAIR_BASE_FIELDS = [
    "analysis_schema_version",
    "analysis_mode",
    "study_signature",
    "profile",
    "base_task_id",
    "pair_type",
    "channel",
    "baseline_condition",
    "baseline_channel",
    "treatment_condition",
    "treatment_channel",
    "replicate_index",
    "pair_state",
    "pair_usable",
    "baseline_trial",
    "treatment_trial",
    "baseline_terminal_status",
    "treatment_terminal_status",
]

MATCHED_BEHAVIOR_PAIR_FIELDS = (
    MATCHED_BEHAVIOR_PAIR_BASE_FIELDS
    + [
        field
        for metric in BEHAVIOR_PAIR_METRICS
        for field in (
            f"baseline_{metric}",
            f"treatment_{metric}",
            f"delta_{metric}",
        )
    ]
)


def _behavior_pair_key(
    row: dict[str, Any],
) -> tuple[str, str, str, int]:
    return (
        str(row.get("base_task_id", "")),
        str(row.get("condition", "")),
        str(row.get("channel", "")),
        int(row.get("replicate_index") or 1),
    )


def _optional_number(value: Any) -> float | None:
    if value in (None, ""):
        return None

    if isinstance(value, bool):
        return float(int(value))

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def matched_behavior_pair_rows(
    pairs: Sequence[dict[str, Any]],
    behavior_rows: Sequence[dict[str, Any]],
    *,
    analysis_schema_version: str,
    analysis_mode: str,
    study_signature: str,
) -> list[dict[str, Any]]:
    """Join canonical experimental pairs to deterministic behavior metrics.

    Pair construction remains owned by the main analyzer. This function does
    not independently decide which conditions should be compared.

    Infrastructure-censored pair sides remain blank rather than becoming
    behavioral zeros.
    """
    index: dict[
        tuple[str, str, str, int],
        dict[str, Any],
    ] = {}

    for behavior in behavior_rows:
        key = _behavior_pair_key(behavior)

        if key in index:
            raise ValueError(
                "duplicate substantive behavioral row for "
                f"pairing key {key!r}"
            )

        index[key] = behavior

    output: list[dict[str, Any]] = []

    for pair in pairs:
        replicate = int(
            pair.get("replicate_index") or 1
        )
        base_id = str(
            pair.get("base_task_id", "")
        )

        baseline_key = (
            base_id,
            str(pair.get("baseline_condition", "")),
            str(pair.get("baseline_channel", "")),
            replicate,
        )

        treatment_key = (
            base_id,
            str(pair.get("treatment_condition", "")),
            str(pair.get("treatment_channel", "")),
            replicate,
        )

        baseline = index.get(baseline_key)
        treatment = index.get(treatment_key)

        pair_usable = int(
            pair.get("pair_usable") or 0
        )

        if pair_usable and (
            baseline is None
            or treatment is None
        ):
            raise ValueError(
                "usable experimental pair is missing a "
                "substantive behavioral row: "
                f"{base_id=} "
                f"{baseline_key=} "
                f"{treatment_key=}"
            )

        row: dict[str, Any] = {
            "analysis_schema_version": (
                analysis_schema_version
            ),
            "analysis_mode": analysis_mode,
            "study_signature": study_signature,
            "profile": str(
                pair.get("profile", "")
            ),
            "base_task_id": base_id,
            "pair_type": str(
                pair.get("pair_type", "")
            ),
            "channel": str(
                pair.get("channel", "")
            ),
            "baseline_condition": str(
                pair.get(
                    "baseline_condition",
                    "",
                )
            ),
            "baseline_channel": str(
                pair.get(
                    "baseline_channel",
                    "",
                )
            ),
            "treatment_condition": str(
                pair.get(
                    "treatment_condition",
                    "",
                )
            ),
            "treatment_channel": str(
                pair.get(
                    "treatment_channel",
                    "",
                )
            ),
            "replicate_index": replicate,
            "pair_state": str(
                pair.get("pair_state", "")
            ),
            "pair_usable": pair_usable,
            "baseline_trial": str(
                pair.get("baseline_trial", "")
            ),
            "treatment_trial": str(
                pair.get("treatment_trial", "")
            ),
            "baseline_terminal_status": str(
                pair.get(
                    "baseline_terminal_status",
                    "",
                )
            ),
            "treatment_terminal_status": str(
                pair.get(
                    "treatment_terminal_status",
                    "",
                )
            ),
        }

        for metric in BEHAVIOR_PAIR_METRICS:
            b = (
                _optional_number(
                    baseline.get(metric)
                )
                if baseline is not None
                else None
            )

            t = (
                _optional_number(
                    treatment.get(metric)
                )
                if treatment is not None
                else None
            )

            row[f"baseline_{metric}"] = (
                b if b is not None else ""
            )
            row[f"treatment_{metric}"] = (
                t if t is not None else ""
            )

            row[f"delta_{metric}"] = (
                t - b
                if (
                    pair_usable
                    and b is not None
                    and t is not None
                )
                else ""
            )

        output.append(row)

    return output
