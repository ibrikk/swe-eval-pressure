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
