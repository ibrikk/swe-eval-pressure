#!/usr/bin/env python3
"""Primary matched binary inference for SWE-EvalPressure behavior analysis.

This module operates only on same-task matched behavioral pairs produced by
the canonical analyzer.

Primary inferential families follow the frozen behavioral analysis plan:

Historical full study, per profile:
    7 primary binary endpoints x all planned historical
    eval / financial / self-preservation matched contrasts.

Resource extension, per profile:
    7 primary binary endpoints x resource-vs-eval/scaffold.

The standalone resource-mode clean-vs-eval comparison is a contemporaneous
control comparison but is not added post hoc to the frozen resource primary
multiplicity family.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

from behavior_metrics import PRIMARY_BINARY_ENDPOINTS
from behavior_stats import (
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_BOOTSTRAP_SEED,
    holm_adjust,
    summarize_binary_pairs,
)


DEFAULT_ALPHA = 0.05

HISTORICAL_PRIMARY_PAIR_TYPES = {
    "eval_effect",
    "financial_effect",
    "self_preservation_effect",
}

RESOURCE_PRIMARY_PAIR_TYPES = {
    "resource_effect",
}


BEHAVIOR_BINARY_EFFECT_FIELDS = [
    "analysis_schema_version",
    "analysis_mode",
    "study_signature",
    "profile",
    "analysis_tier",
    "family_name",
    "multiplicity_method",
    "family_size",
    "tested_family_size",
    "pair_type",
    "channel",
    "baseline_condition",
    "baseline_channel",
    "treatment_condition",
    "treatment_channel",
    "endpoint",
    "n_pairs",
    "baseline_positive",
    "treatment_positive",
    "baseline_prevalence",
    "treatment_prevalence",
    "baseline_prevalence_pct",
    "treatment_prevalence_pct",
    "risk_difference",
    "risk_difference_pp",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "bootstrap_ci_low_pp",
    "bootstrap_ci_high_pp",
    "treatment_only_positive",
    "baseline_only_positive",
    "discordant_pairs",
    "mcnemar_exact_p",
    "holm_adjusted_p",
    "alpha",
    "unadjusted_reject",
    "adjusted_reject",
    "bootstrap_replicates",
    "bootstrap_seed",
]


BEHAVIOR_MULTIPLICITY_FIELDS = [
    "analysis_schema_version",
    "analysis_mode",
    "study_signature",
    "profile",
    "family_name",
    "multiplicity_method",
    "alpha",
    "family_size",
    "tested_family_size",
    "untested_family_size",
    "unadjusted_rejections",
    "adjusted_rejections",
]


def _binary(value: Any, field: str) -> int:
    if value in (0, 0.0, "0", False):
        return 0

    if value in (1, 1.0, "1", True):
        return 1

    raise ValueError(
        f"{field} must be binary 0/1; got {value!r}"
    )


def _primary_family(
    mode: str,
    pair_type: str,
) -> str | None:
    if (
        mode == "full"
        and pair_type in HISTORICAL_PRIMARY_PAIR_TYPES
    ):
        return "historical_primary_binary_holm"

    if pair_type in RESOURCE_PRIMARY_PAIR_TYPES:
        return "resource_primary_binary_holm"

    return None


def _planned_groups(
    pair_rows: Sequence[dict[str, Any]],
    *,
    mode: str,
) -> dict[
    tuple[str, str],
    list[dict[str, Any]],
]:
    """Return planned primary contrast groups.

    Group key is (pair_type, channel). Missing/censored task pairs remain in
    the group so the planned hypothesis family is not silently reduced.
    """
    groups: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in pair_rows:
        pair_type = str(
            row.get("pair_type", "")
        )

        if _primary_family(
            mode,
            pair_type,
        ) is None:
            continue

        key = (
            pair_type,
            str(row.get("channel", "")),
        )

        groups[key].append(row)

    return dict(groups)


def primary_binary_effect_rows(
    pair_rows: Sequence[dict[str, Any]],
    *,
    analysis_schema_version: str,
    analysis_mode: str,
    study_signature: str,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    alpha: float = DEFAULT_ALPHA,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Compute primary matched binary effects and Holm families."""

    if not 0.0 < alpha < 1.0:
        raise ValueError(
            "alpha must be in (0, 1)"
        )

    groups = _planned_groups(
        pair_rows,
        mode=analysis_mode,
    )

    effects: list[dict[str, Any]] = []

    for (
        pair_type,
        channel,
    ), group in sorted(groups.items()):
        first = group[0]

        family_name = _primary_family(
            analysis_mode,
            pair_type,
        )

        assert family_name is not None

        usable = [
            row
            for row in group
            if int(
                row.get("pair_usable") or 0
            )
        ]

        for endpoint in PRIMARY_BINARY_ENDPOINTS:
            output: dict[str, Any] = {
                "analysis_schema_version": (
                    analysis_schema_version
                ),
                "analysis_mode": analysis_mode,
                "study_signature": (
                    study_signature
                ),
                "profile": str(
                    first.get("profile", "")
                ),
                "analysis_tier": (
                    "primary_binary"
                ),
                "family_name": family_name,
                "multiplicity_method": "holm",
                "family_size": "",
                "tested_family_size": "",
                "pair_type": pair_type,
                "channel": channel,
                "baseline_condition": str(
                    first.get(
                        "baseline_condition",
                        "",
                    )
                ),
                "baseline_channel": str(
                    first.get(
                        "baseline_channel",
                        "",
                    )
                ),
                "treatment_condition": str(
                    first.get(
                        "treatment_condition",
                        "",
                    )
                ),
                "treatment_channel": str(
                    first.get(
                        "treatment_channel",
                        "",
                    )
                ),
                "endpoint": endpoint,
                "n_pairs": len(usable),
                "baseline_positive": "",
                "treatment_positive": "",
                "baseline_prevalence": "",
                "treatment_prevalence": "",
                "baseline_prevalence_pct": "",
                "treatment_prevalence_pct": "",
                "risk_difference": "",
                "risk_difference_pp": "",
                "bootstrap_ci_low": "",
                "bootstrap_ci_high": "",
                "bootstrap_ci_low_pp": "",
                "bootstrap_ci_high_pp": "",
                "treatment_only_positive": "",
                "baseline_only_positive": "",
                "discordant_pairs": "",
                "mcnemar_exact_p": "",
                "holm_adjusted_p": "",
                "alpha": alpha,
                "unadjusted_reject": "",
                "adjusted_reject": "",
                "bootstrap_replicates": (
                    bootstrap_replicates
                ),
                "bootstrap_seed": (
                    bootstrap_seed
                ),
            }

            if usable:
                baseline = [
                    _binary(
                        row.get(
                            f"baseline_{endpoint}"
                        ),
                        f"baseline_{endpoint}",
                    )
                    for row in usable
                ]

                treatment = [
                    _binary(
                        row.get(
                            f"treatment_{endpoint}"
                        ),
                        f"treatment_{endpoint}",
                    )
                    for row in usable
                ]

                result = summarize_binary_pairs(
                    baseline,
                    treatment,
                    bootstrap_replicates=(
                        bootstrap_replicates
                    ),
                    bootstrap_seed=(
                        bootstrap_seed
                    ),
                )

                output.update({
                    "baseline_positive": (
                        result.control_positive
                    ),
                    "treatment_positive": (
                        result.treatment_positive
                    ),
                    "baseline_prevalence": (
                        result.control_prevalence
                    ),
                    "treatment_prevalence": (
                        result.treatment_prevalence
                    ),
                    "baseline_prevalence_pct": (
                        100.0
                        * result.control_prevalence
                    ),
                    "treatment_prevalence_pct": (
                        100.0
                        * result.treatment_prevalence
                    ),
                    "risk_difference": (
                        result.delta
                    ),
                    "risk_difference_pp": (
                        result.delta_pp
                    ),
                    "bootstrap_ci_low": (
                        result.bootstrap_ci_low
                    ),
                    "bootstrap_ci_high": (
                        result.bootstrap_ci_high
                    ),
                    "bootstrap_ci_low_pp": (
                        result.bootstrap_ci_low_pp
                    ),
                    "bootstrap_ci_high_pp": (
                        result.bootstrap_ci_high_pp
                    ),
                    "treatment_only_positive": (
                        result.treatment_only_positive
                    ),
                    "baseline_only_positive": (
                        result.control_only_positive
                    ),
                    "discordant_pairs": (
                        result.discordant_pairs
                    ),
                    "mcnemar_exact_p": (
                        result.mcnemar_exact_p
                    ),
                    "unadjusted_reject": int(
                        result.mcnemar_exact_p
                        < alpha
                    ),
                })

            effects.append(output)

    by_family: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in effects:
        by_family[
            (
                str(row["profile"]),
                str(row["family_name"]),
            )
        ].append(row)

    multiplicity: list[dict[str, Any]] = []

    for (
        profile,
        family_name,
    ), family_rows in sorted(
        by_family.items()
    ):
        family_size = len(family_rows)

        tested_rows = [
            row
            for row in family_rows
            if row["mcnemar_exact_p"] != ""
        ]

        tested_family_size = len(
            tested_rows
        )

        # Untested planned hypotheses are represented conservatively as p=1
        # during adjustment. This preserves the frozen family size even in a
        # partial/live reconstruction.
        p_values = [
            float(row["mcnemar_exact_p"])
            if row["mcnemar_exact_p"] != ""
            else 1.0
            for row in family_rows
        ]

        adjusted = holm_adjust(
            p_values
        )

        for row, adjusted_p in zip(
            family_rows,
            adjusted,
        ):
            row["family_size"] = family_size
            row[
                "tested_family_size"
            ] = tested_family_size

            if row["mcnemar_exact_p"] == "":
                continue

            row[
                "holm_adjusted_p"
            ] = adjusted_p

            row[
                "adjusted_reject"
            ] = int(
                adjusted_p < alpha
            )

        multiplicity.append({
            "analysis_schema_version": (
                analysis_schema_version
            ),
            "analysis_mode": analysis_mode,
            "study_signature": (
                study_signature
            ),
            "profile": profile,
            "family_name": family_name,
            "multiplicity_method": "holm",
            "alpha": alpha,
            "family_size": family_size,
            "tested_family_size": (
                tested_family_size
            ),
            "untested_family_size": (
                family_size
                - tested_family_size
            ),
            "unadjusted_rejections": sum(
                int(
                    row[
                        "unadjusted_reject"
                    ]
                )
                for row in tested_rows
            ),
            "adjusted_rejections": sum(
                int(
                    row[
                        "adjusted_reject"
                    ]
                )
                for row in tested_rows
            ),
        })

    return effects, multiplicity
