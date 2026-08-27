#!/usr/bin/env python3
"""Canonical current SWE-EvalPressure statistical analysis.

Inputs:
- analysis/current/source/{primary,resource,replication}
- raw DeepSeek/Gemini primary semantic jobs
- raw DeepSeek/Gemini resource semantic jobs

No old aggregate, inference, semantic-summary, or HTML outputs are inputs.
No network/API/model/verifier calls.

Primary treatment contrasts remain same-task matched comparisons.
Semantic-state subgroup analyses are explicitly descriptive/post-treatment.
The incomplete Aug-26 rerun is descriptive and is not pooled with PRIMARY.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"

OUT = (
    ROOT
    / "analysis"
    / "current"
    / "results"
)

CORE_PATH = (
    SCRIPT_DIR
    / "27_current_core_analysis.py"
)


# ---------------------------------------------------------------------------
# Load tested/source-building helpers from the provisional core implementation
# ---------------------------------------------------------------------------


spec = importlib.util.spec_from_file_location(
    "current_core_base",
    CORE_PATH,
)

if (
    spec is None
    or spec.loader is None
):
    raise RuntimeError(
        f"Unable to import {CORE_PATH}"
    )

core = importlib.util.module_from_spec(
    spec
)

sys.modules[
    spec.name
] = core

spec.loader.exec_module(
    core
)


VERSION = "current-analysis-2.0"

PROFILES = tuple(
    core.PROFILES
)

CAPABLE = tuple(
    core.CAPABLE
)

PRIMARY_SEMANTIC_FIELDS = tuple(
    core.PRIMARY_SEMANTIC_FIELDS
)

RESOURCE_SEMANTIC_FIELDS = tuple(
    core.RESOURCE_SEMANTIC_FIELDS
)

BEHAVIOR_FIELDS = tuple(
    core.BEHAVIOR_FIELDS
)


# One canonical field for each observable process construct.
# Aliases such as prompt_tokens/input_tokens are never included together.
CANONICAL_PROCESS_METRICS = (
    "raw_tool_calls",
    "behavioral_action_calls",
    "trajectory_steps",
    "repo_search_calls",
    "file_read_calls",
    "unique_files_read",
    "unique_dirs_read",
    "test_files_inspected",
    "spec_config_files_inspected",
    "edit_calls",
    "unique_files_modified",
    "test_command_calls",
    "validation_calls",
    "post_edit_validation_calls",
    "edit_validation_cycles",
    "failed_validation_then_edit_cycles",
    "instruction_file_inspections",
    "git_history_inspections",
    "external_lookup_calls",
    "subagent_delegation_calls",
    "integrity_sensitive_events",
    "input_tokens",
    "output_tokens",
    "duration_sec",
)

FOCUSED_RESOURCE_METRICS = (
    "raw_tool_calls",
    "input_tokens",
    "validation_calls",
)

BINARY_METRICS = (
    "overall_pass",
    "tests_pass",
    *BEHAVIOR_FIELDS,
)

SIGN_FLIP_DRAWS = 50_000
SIGN_FLIP_SEED = 20260827

SAID_DID_DISPLAY_MIN_N = 5
SAID_DID_FIGURE_MIN_N = 10


# ---------------------------------------------------------------------------
# Paired inferential helpers
# ---------------------------------------------------------------------------


def deterministic_seed(
    text: str,
) -> int:
    digest = hashlib.sha256(
        (
            str(
                SIGN_FLIP_SEED
            )
            + "|"
            + text
        ).encode(
            "utf-8"
        )
    ).hexdigest()

    return int(
        digest[:16],
        16,
    )


def paired_sign_flip_p(
    deltas: list[float],
    *,
    seed_text: str,
) -> float | None:
    """Two-sided paired sign-flip test for a zero mean difference.

    Exact enumeration is used when feasible. Otherwise a deterministic
    Monte Carlo randomization test with add-one correction is used.
    """

    values = [
        float(value)
        for value in deltas
        if value is not None
    ]

    if not values:
        return None

    observed = abs(
        sum(values)
        / len(values)
    )

    if math.isclose(
        observed,
        0.0,
        abs_tol=1e-15,
    ):
        return 1.0

    n = len(values)

    if n <= 20:
        total = 1 << n
        extreme = 0

        for mask in range(total):
            estimate = sum(
                (
                    value
                    if (
                        mask
                        & (
                            1 << index
                        )
                    )
                    else -value
                )
                for index, value
                in enumerate(values)
            ) / n

            if (
                abs(estimate)
                >= observed
                - 1e-15
            ):
                extreme += 1

        return extreme / total

    rng = random.Random(
        deterministic_seed(
            seed_text
        )
    )

    extreme = 0

    for _ in range(
        SIGN_FLIP_DRAWS
    ):
        estimate = sum(
            (
                value
                if rng.getrandbits(1)
                else -value
            )
            for value in values
        ) / n

        if (
            abs(estimate)
            >= observed
            - 1e-15
        ):
            extreme += 1

    return (
        extreme + 1
    ) / (
        SIGN_FLIP_DRAWS
        + 1
    )


# ---------------------------------------------------------------------------
# Corrected binary effects and multiplicity
# ---------------------------------------------------------------------------


def corrected_binary_effects(
    pairs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = core.binary_effect_rows(
        pairs
    )

    families: dict[
        tuple[str, ...],
        list[tuple[int, float]],
    ] = defaultdict(list)

    for index, row in enumerate(rows):
        raw_p = row.get(
            "mcnemar_p"
        )

        row[
            "mcnemar_p_raw"
        ] = raw_p

        row["mcnemar_p"] = None
        row["holm_p"] = None
        row[
            "multiplicity_family"
        ] = ""
        row["family_size"] = ""
        row["family_scope"] = ""

        study = str(
            row["study"]
        )

        profile = str(
            row["profile"]
        )

        contrast = str(
            row["contrast"]
        )

        metric = str(
            row["metric"]
        )

        if study == "replication":
            row[
                "inferential_role"
            ] = (
                "descriptive_partial_replication"
            )
            continue

        if profile == "llama":
            row[
                "inferential_role"
            ] = (
                "descriptive_capability_floor"
            )
            continue

        if study == "primary":
            family = (
                "primary",
                profile,
                metric,
                "nine_planned_matched_contrasts",
            )

            row[
                "inferential_role"
            ] = (
                "planned_matched_inference"
            )

            row[
                "family_scope"
            ] = (
                "within_model_across_"
                "3_contrasts_x_3_placements"
            )

        elif (
            study == "resource"
            and contrast
            in {
                "resource_deprivation",
                "resource_evaluation_disclosure",
            }
        ):
            family = (
                "resource",
                contrast,
                metric,
                "three_capable_models",
            )

            row[
                "inferential_role"
            ] = (
                "primary_resource_inference"
                if contrast
                == "resource_deprivation"
                else
                "secondary_resource_inference"
            )

            row[
                "family_scope"
            ] = (
                "across_claude_fable_codex"
            )

        else:
            row[
                "inferential_role"
            ] = "descriptive"
            continue

        row[
            "multiplicity_family"
        ] = "|".join(
            family
        )

        row["mcnemar_p"] = raw_p

        if raw_p is not None:
            families[family].append(
                (
                    index,
                    float(raw_p),
                )
            )

    for family, values in (
        families.items()
    ):
        adjusted = core.holm(
            values
        )

        family_size = len(values)

        for index, _ in values:
            rows[index][
                "holm_p"
            ] = adjusted[index]

            rows[index][
                "family_size"
            ] = family_size

    return rows


# ---------------------------------------------------------------------------
# Canonical process effects
# ---------------------------------------------------------------------------


def canonical_process_effects(
    pairs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[
        tuple[str, str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for pair in pairs:
        groups[
            (
                str(
                    pair["study"]
                ),
                str(
                    pair["profile"]
                ),
                str(
                    pair["contrast"]
                ),
                str(
                    pair["placement"]
                ),
            )
        ].append(pair)

    rows = []

    for key, values in sorted(
        groups.items()
    ):
        (
            study,
            profile,
            contrast,
            placement,
        ) = key

        for metric in (
            CANONICAL_PROCESS_METRICS
        ):
            if not core.process_metric_available(
                values,
                metric,
            ):
                continue

            observations = []

            for pair in values:
                baseline = core.numeric(
                    pair[
                        "baseline"
                    ].get(metric)
                )

                treatment = core.numeric(
                    pair[
                        "treatment"
                    ].get(metric)
                )

                if (
                    baseline is None
                    or treatment is None
                ):
                    continue

                observations.append(
                    (
                        baseline,
                        treatment,
                        treatment
                        - baseline,
                    )
                )

            if not observations:
                continue

            deltas = [
                delta
                for _, _, delta
                in observations
            ]

            seed_text = (
                f"process|{study}|"
                f"{profile}|{contrast}|"
                f"{placement}|{metric}"
            )

            ci_low, ci_high = (
                core.paired_bootstrap_ci(
                    deltas,
                    seed_text=seed_text,
                )
            )

            if study == "replication":
                raw_p = None
                role = (
                    "descriptive_partial_replication"
                )

            else:
                raw_p = (
                    paired_sign_flip_p(
                        deltas,
                        seed_text=seed_text,
                    )
                )

                role = (
                    "exploratory_paired_inference"
                )

            rows.append(
                {
                    "study": study,
                    "profile": profile,
                    "contrast": contrast,
                    "placement": placement,
                    "metric": metric,
                    "matched_n": len(
                        observations
                    ),
                    "baseline_mean": core.mean(
                        [
                            baseline
                            for (
                                baseline,
                                _,
                                _,
                            )
                            in observations
                        ]
                    ),
                    "treatment_mean": core.mean(
                        [
                            treatment
                            for (
                                _,
                                treatment,
                                _,
                            )
                            in observations
                        ]
                    ),
                    "mean_delta": core.mean(
                        deltas
                    ),
                    "median_delta": core.median(
                        deltas
                    ),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "positive_delta_n": sum(
                        delta > 0
                        for delta in deltas
                    ),
                    "zero_delta_n": sum(
                        delta == 0
                        for delta in deltas
                    ),
                    "negative_delta_n": sum(
                        delta < 0
                        for delta in deltas
                    ),
                    "positive_fraction": (
                        sum(
                            delta > 0
                            for delta in deltas
                        )
                        / len(deltas)
                    ),
                    "zero_fraction": (
                        sum(
                            delta == 0
                            for delta in deltas
                        )
                        / len(deltas)
                    ),
                    "negative_fraction": (
                        sum(
                            delta < 0
                            for delta in deltas
                        )
                        / len(deltas)
                    ),
                    "sign_flip_p": raw_p,
                    "bh_q": None,
                    "inferential_role": role,
                    "fdr_family": "",
                }
            )

    families: dict[
        tuple[str, ...],
        list[tuple[int, float]],
    ] = defaultdict(list)

    for index, row in enumerate(rows):
        if (
            row["study"]
            == "replication"
            or row[
                "sign_flip_p"
            ]
            is None
        ):
            continue

        family = (
            str(
                row["study"]
            ),
            str(
                row["profile"]
            ),
            str(
                row["contrast"]
            ),
            str(
                row["placement"]
            ),
            "canonical_process_metrics",
        )

        row[
            "fdr_family"
        ] = "|".join(
            family
        )

        families[family].append(
            (
                index,
                float(
                    row[
                        "sign_flip_p"
                    ]
                ),
            )
        )

    for _, values in (
        families.items()
    ):
        adjusted = core.bh(
            values
        )

        for index, _ in values:
            rows[index][
                "bh_q"
            ] = adjusted[index]

    return rows


def focused_resource_process(
    process_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []

    for source in process_rows:
        if not (
            source["study"]
            == "resource"
            and source["contrast"]
            == "resource_deprivation"
            and source["profile"]
            in CAPABLE
            and source["metric"]
            in FOCUSED_RESOURCE_METRICS
        ):
            continue

        row = dict(source)

        row[
            "focused_family"
        ] = (
            "resource_3_models_x_"
            "3_process_metrics_holm"
        )

        row["focused_holm_p"] = None

        row[
            "inferential_role"
        ] = (
            "focused_resource_"
            "exploratory_inference"
        )

        rows.append(row)

    pvalues = [
        (
            index,
            float(
                row[
                    "sign_flip_p"
                ]
            ),
        )
        for index, row
        in enumerate(rows)
        if row[
            "sign_flip_p"
        ]
        is not None
    ]

    adjusted = core.holm(
        pvalues
    )

    for index, _ in pvalues:
        rows[index][
            "focused_holm_p"
        ] = adjusted[index]

    return rows


# ---------------------------------------------------------------------------
# Behavior effects
# ---------------------------------------------------------------------------


def corrected_behavior_effects(
    pairs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = core.behavior_effects(
        pairs
    )

    for row in rows:
        row[
            "mcnemar_p_raw"
        ] = row.get(
            "mcnemar_p"
        )

        if (
            row["study"]
            == "replication"
        ):
            row[
                "inferential_role"
            ] = (
                "descriptive_partial_replication"
            )

            row[
                "mcnemar_p"
            ] = None

            row["holm_p"] = None

            row[
                "multiplicity_family"
            ] = ""

        else:
            row[
                "inferential_role"
            ] = (
                "seven_behavior_"
                "matched_inference"
            )

    return rows


# ---------------------------------------------------------------------------
# Semantic reliability by profile and cell
# ---------------------------------------------------------------------------


def semantic_agreement_breakdowns(
    jobs: list[dict[str, Any]],
    fields: tuple[str, ...],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    by_trial: dict[
        tuple[str, str, str],
        dict[str, dict[str, Any]],
    ] = defaultdict(dict)

    for job in jobs:
        by_trial[
            (
                str(
                    job["study"]
                ),
                str(
                    job["profile"]
                ),
                str(
                    job["trial_name"]
                ),
            )
        ][
            str(
                job["judge_family"]
            )
        ] = job

    profile_pairs: dict[
        tuple[str, str, str],
        list[tuple[str, str]],
    ] = defaultdict(list)

    cell_pairs: dict[
        tuple[
            str,
            str,
            str,
            str,
            str,
            str,
        ],
        list[tuple[str, str]],
    ] = defaultdict(list)

    coverage: dict[
        tuple[str, str, str, str, str, str],
        Counter,
    ] = defaultdict(Counter)

    for (
        study,
        profile,
        _trial,
    ), judges in sorted(
        by_trial.items()
    ):
        deepseek = judges.get(
            "deepseek"
        )

        gemini = judges.get(
            "gemini"
        )

        metadata = (
            deepseek
            or gemini
            or {}
        )

        condition = str(
            metadata.get(
                "condition"
            )
            or ""
        )

        placement = str(
            metadata.get(
                "placement"
            )
            or ""
        )

        pressure_type = str(
            metadata.get(
                "pressure_type"
            )
            or ""
        )

        for field in fields:
            deepseek_label = (
                deepseek.get(field)
                if (
                    deepseek
                    and deepseek.get(
                        "valid"
                    )
                    == 1
                )
                else None
            )

            gemini_label = (
                gemini.get(field)
                if (
                    gemini
                    and gemini.get(
                        "valid"
                    )
                    == 1
                )
                else None
            )

            coverage_key = (
                study,
                profile,
                condition,
                placement,
                pressure_type,
                field,
            )

            coverage[
                coverage_key
            ]["total"] += 1

            if (
                deepseek_label
                is not None
                and gemini_label
                is not None
            ):
                coverage[
                    coverage_key
                ]["both_valid"] += 1

                pair = (
                    str(
                        deepseek_label
                    ),
                    str(
                        gemini_label
                    ),
                )

                profile_pairs[
                    (
                        study,
                        profile,
                        field,
                    )
                ].append(pair)

                cell_pairs[
                    coverage_key
                ].append(pair)

                if (
                    deepseek_label
                    == gemini_label
                ):
                    coverage[
                        coverage_key
                    ]["agreement"] += 1

                else:
                    coverage[
                        coverage_key
                    ]["disagreement"] += 1

            elif (
                deepseek_label
                is not None
                or gemini_label
                is not None
            ):
                coverage[
                    coverage_key
                ]["one_valid"] += 1

            else:
                coverage[
                    coverage_key
                ]["neither_valid"] += 1

    profile_rows = []

    for (
        study,
        profile,
        field,
    ), pairs in sorted(
        profile_pairs.items()
    ):
        agreement = sum(
            left == right
            for left, right in pairs
        )

        profile_rows.append(
            {
                "study": study,
                "profile": profile,
                "field": field,
                "both_valid_n": len(
                    pairs
                ),
                "agreement_n": agreement,
                "raw_agreement": (
                    agreement
                    / len(pairs)
                ),
                "cohen_kappa": (
                    core.cohen_kappa(
                        pairs
                    )
                ),
                "gwet_ac1": (
                    core.gwet_ac1(
                        pairs
                    )
                ),
            }
        )

    cell_rows = []

    for key, pairs in sorted(
        cell_pairs.items()
    ):
        (
            study,
            profile,
            condition,
            placement,
            pressure_type,
            field,
        ) = key

        agreement = sum(
            left == right
            for left, right in pairs
        )

        cell_rows.append(
            {
                "study": study,
                "profile": profile,
                "condition": condition,
                "placement": placement,
                "pressure_type": pressure_type,
                "field": field,
                "both_valid_n": len(
                    pairs
                ),
                "agreement_n": agreement,
                "raw_agreement": (
                    agreement
                    / len(pairs)
                ),
                "cohen_kappa": (
                    core.cohen_kappa(
                        pairs
                    )
                ),
                "gwet_ac1": (
                    core.gwet_ac1(
                        pairs
                    )
                ),
            }
        )

    coverage_rows = []

    for key, counts in sorted(
        coverage.items()
    ):
        (
            study,
            profile,
            condition,
            placement,
            pressure_type,
            field,
        ) = key

        total = counts[
            "total"
        ]

        coverage_rows.append(
            {
                "study": study,
                "profile": profile,
                "condition": condition,
                "placement": placement,
                "pressure_type": pressure_type,
                "field": field,
                "trajectory_n": total,
                "both_valid_n": counts[
                    "both_valid"
                ],
                "one_valid_n": counts[
                    "one_valid"
                ],
                "neither_valid_n": counts[
                    "neither_valid"
                ],
                "agreement_n": counts[
                    "agreement"
                ],
                "disagreement_n": counts[
                    "disagreement"
                ],
                "strict_consensus_coverage": (
                    counts[
                        "agreement"
                    ]
                    / total
                    if total
                    else None
                ),
            }
        )

    return (
        profile_rows,
        cell_rows,
        coverage_rows,
    )


# ---------------------------------------------------------------------------
# Enriched said-X / did-Y
# ---------------------------------------------------------------------------


def metric_value(
    row: dict[str, Any],
    metric: str,
) -> float | None:
    if metric == "overall_pass":
        value = core.overall_pass(
            row
        )

    elif metric == "tests_pass":
        value = core.tests_pass(
            row
        )

    elif metric in BEHAVIOR_FIELDS:
        value = core.binary_value(
            row,
            metric,
        )

    else:
        value = core.numeric(
            row.get(metric)
        )

    return (
        float(value)
        if value is not None
        else None
    )


def build_said_did(
    pairs: list[dict[str, Any]],
    consensus_rows: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    semantic = core.semantic_lookup(
        consensus_rows
    )

    fields_by_study = {
        "primary": (
            PRIMARY_SEMANTIC_FIELDS
        ),
        "resource": (
            RESOURCE_SEMANTIC_FIELDS
        ),
    }

    metrics = (
        "overall_pass",
        "tests_pass",
        *BEHAVIOR_FIELDS,
        *CANONICAL_PROCESS_METRICS,
    )

    pair_rows = []

    for pair in pairs:
        study = str(
            pair["study"]
        )

        if study not in (
            fields_by_study
        ):
            continue

        treatment = pair[
            "treatment"
        ]

        semantic_row = semantic.get(
            (
                study,
                str(
                    pair["profile"]
                ),
                str(
                    treatment.get(
                        "trial_name"
                    )
                    or ""
                ),
            )
        )

        if semantic_row is None:
            continue

        output = {
            "study": study,
            "profile": pair[
                "profile"
            ],
            "base_task_id": pair[
                "base_task_id"
            ],
            "contrast": pair[
                "contrast"
            ],
            "placement": pair[
                "placement"
            ],
            "baseline_trial": (
                pair[
                    "baseline"
                ].get(
                    "trial_name"
                )
            ),
            "treatment_trial": (
                treatment.get(
                    "trial_name"
                )
            ),
            "semantic_states_are_post_treatment": 1,
            "causal_interpretation_allowed": 0,
        }

        for field in (
            fields_by_study[
                study
            ]
        ):
            for suffix in (
                "status",
                "label",
                "deepseek",
                "gemini",
            ):
                output[
                    f"{field}__{suffix}"
                ] = semantic_row.get(
                    f"{field}__{suffix}"
                )

        for metric in metrics:
            baseline_value = (
                metric_value(
                    pair[
                        "baseline"
                    ],
                    metric,
                )
            )

            treatment_value = (
                metric_value(
                    treatment,
                    metric,
                )
            )

            output[
                f"baseline_{metric}"
            ] = baseline_value

            output[
                f"treatment_{metric}"
            ] = treatment_value

            output[
                f"delta_{metric}"
            ] = (
                treatment_value
                - baseline_value
                if (
                    baseline_value
                    is not None
                    and treatment_value
                    is not None
                )
                else None
            )

        pair_rows.append(output)

    groups: dict[
        tuple[
            str,
            str,
            str,
            str,
            str,
            str,
            str,
            str,
        ],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in pair_rows:
        fields = fields_by_study[
            row["study"]
        ]

        for field in fields:
            semantic_sources = {
                "strict_consensus": (
                    row.get(
                        f"{field}__label"
                    )
                    if row.get(
                        f"{field}__status"
                    )
                    == "agreement"
                    else None
                ),
                "deepseek": row.get(
                    f"{field}__deepseek"
                ),
                "gemini": row.get(
                    f"{field}__gemini"
                ),
            }

            for (
                semantic_source,
                label,
            ) in (
                semantic_sources.items()
            ):
                if label is None:
                    continue

                for metric in metrics:
                    delta = row.get(
                        f"delta_{metric}"
                    )

                    if delta is None:
                        continue

                    groups[
                        (
                            str(
                                row["study"]
                            ),
                            str(
                                row["profile"]
                            ),
                            str(
                                row["contrast"]
                            ),
                            str(
                                row["placement"]
                            ),
                            field,
                            semantic_source,
                            str(label),
                            metric,
                        )
                    ].append(row)

    summary_rows = []

    for key, values in sorted(
        groups.items()
    ):
        (
            study,
            profile,
            contrast,
            placement,
            field,
            semantic_source,
            label,
            metric,
        ) = key

        multiplier = (
            100.0
            if metric
            in BINARY_METRICS
            else 1.0
        )

        baseline_values = [
            float(
                row[
                    f"baseline_{metric}"
                ]
            )
            * multiplier
            for row in values
        ]

        treatment_values = [
            float(
                row[
                    f"treatment_{metric}"
                ]
            )
            * multiplier
            for row in values
        ]

        deltas = [
            float(
                row[
                    f"delta_{metric}"
                ]
            )
            * multiplier
            for row in values
        ]

        seed_text = (
            "said-did|"
            f"{study}|{profile}|"
            f"{contrast}|{placement}|"
            f"{field}|"
            f"{semantic_source}|"
            f"{label}|{metric}"
        )

        ci_low, ci_high = (
            core.paired_bootstrap_ci(
                deltas,
                seed_text=seed_text,
            )
        )

        n = len(deltas)

        summary_rows.append(
            {
                "study": study,
                "profile": profile,
                "contrast": contrast,
                "placement": placement,
                "semantic_field": field,
                "semantic_source": (
                    semantic_source
                ),
                "semantic_label": label,
                "metric": metric,
                "unit": (
                    "percentage_points"
                    if metric
                    in BINARY_METRICS
                    else "raw_units"
                ),
                "n_pairs": n,
                "baseline_mean": (
                    core.mean(
                        baseline_values
                    )
                ),
                "treatment_mean": (
                    core.mean(
                        treatment_values
                    )
                ),
                "mean_delta": (
                    core.mean(
                        deltas
                    )
                ),
                "median_delta": (
                    core.median(
                        deltas
                    )
                ),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "increased_n": sum(
                    delta > 0
                    for delta in deltas
                ),
                "unchanged_n": sum(
                    delta == 0
                    for delta in deltas
                ),
                "decreased_n": sum(
                    delta < 0
                    for delta in deltas
                ),
                "increased_fraction": (
                    sum(
                        delta > 0
                        for delta in deltas
                    )
                    / n
                ),
                "unchanged_fraction": (
                    sum(
                        delta == 0
                        for delta in deltas
                    )
                    / n
                ),
                "decreased_fraction": (
                    sum(
                        delta < 0
                        for delta in deltas
                    )
                    / n
                ),
                "display_eligible": int(
                    n
                    >= SAID_DID_DISPLAY_MIN_N
                ),
                "figure_eligible": int(
                    n
                    >= SAID_DID_FIGURE_MIN_N
                ),
                "inferential_status": (
                    "descriptive_post_treatment"
                ),
                "causal_interpretation_allowed": 0,
            }
        )

    return (
        pair_rows,
        summary_rows,
    )


# ---------------------------------------------------------------------------
# Intervention → effect → interpretation catalog
# ---------------------------------------------------------------------------


def uncertainty_status(
    low: Any,
    high: Any,
) -> str:
    low_value = core.numeric(
        low
    )

    high_value = core.numeric(
        high
    )

    if (
        low_value is None
        or high_value is None
    ):
        return "not_estimated"

    if low_value > 0:
        return "increase_supported"

    if high_value < 0:
        return "decrease_supported"

    return "direction_uncertain"


def interpretation_text(
    metric: str,
    status: str,
) -> str:
    construct = {
        "overall_pass": (
            "strict benchmark success"
        ),
        "tests_pass": (
            "test/comparison success"
        ),
        "broad_repo_search_any": (
            "broad repository-search prevalence"
        ),
        "test_inspection_any": (
            "test/config inspection prevalence"
        ),
        "validation_any": (
            "validation prevalence"
        ),
        "iterative_repair_any": (
            "iterative repair prevalence"
        ),
        "provenance_related_inspection_any": (
            "provenance-inspection prevalence"
        ),
        "external_lookup_any": (
            "external-lookup prevalence"
        ),
        "integrity_sensitive_action_any": (
            "integrity-sensitive action prevalence"
        ),
        "raw_tool_calls": (
            "raw tool-use volume"
        ),
        "behavioral_action_calls": (
            "observable task-action volume"
        ),
        "trajectory_steps": (
            "trajectory-step volume"
        ),
        "validation_calls": (
            "validation-command volume"
        ),
        "input_tokens": (
            "input-token volume"
        ),
        "output_tokens": (
            "output-token volume"
        ),
        "duration_sec": (
            "wall-clock duration"
        ),
    }.get(
        metric,
        metric.replace(
            "_",
            " ",
        ),
    )

    if status == "increase_supported":
        return (
            f"{construct} increased "
            "under treatment."
        )

    if status == "decrease_supported":
        return (
            f"{construct} decreased "
            "under treatment."
        )

    if status == "direction_uncertain":
        return (
            f"The estimated change in "
            f"{construct} remains uncertain."
        )

    return (
        f"No uncertainty interval was "
        f"available for {construct}."
    )


def effect_catalog(
    binary_rows: list[dict[str, Any]],
    behavior_rows: list[dict[str, Any]],
    process_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []

    for row in binary_rows:
        status = uncertainty_status(
            row.get(
                "ci95_low_pp"
            ),
            row.get(
                "ci95_high_pp"
            ),
        )

        output.append(
            {
                "study": row["study"],
                "profile": row["profile"],
                "intervention": (
                    row["contrast"]
                ),
                "placement": (
                    row["placement"]
                ),
                "outcome_family": (
                    "performance"
                ),
                "metric": row["metric"],
                "unit": (
                    "percentage_points"
                ),
                "matched_n": (
                    row["matched_n"]
                ),
                "effect": (
                    row["effect_pp"]
                ),
                "ci95_low": (
                    row[
                        "ci95_low_pp"
                    ]
                ),
                "ci95_high": (
                    row[
                        "ci95_high_pp"
                    ]
                ),
                "raw_p": row.get(
                    "mcnemar_p"
                ),
                "adjusted_p": row.get(
                    "holm_p"
                ),
                "inferential_role": (
                    row.get(
                        "inferential_role"
                    )
                ),
                "uncertainty_status": (
                    status
                ),
                "conservative_interpretation": (
                    interpretation_text(
                        row["metric"],
                        status,
                    )
                ),
                "causal_interpretation_allowed": (
                    int(
                        row["study"]
                        != "replication"
                    )
                ),
            }
        )

    for row in behavior_rows:
        status = uncertainty_status(
            row.get(
                "ci95_low_pp"
            ),
            row.get(
                "ci95_high_pp"
            ),
        )

        output.append(
            {
                "study": row["study"],
                "profile": row["profile"],
                "intervention": (
                    row["contrast"]
                ),
                "placement": (
                    row["placement"]
                ),
                "outcome_family": (
                    "behavior_prevalence"
                ),
                "metric": row["metric"],
                "unit": (
                    "percentage_points"
                ),
                "matched_n": (
                    row["matched_n"]
                ),
                "effect": (
                    row["effect_pp"]
                ),
                "ci95_low": (
                    row[
                        "ci95_low_pp"
                    ]
                ),
                "ci95_high": (
                    row[
                        "ci95_high_pp"
                    ]
                ),
                "raw_p": row.get(
                    "mcnemar_p"
                ),
                "adjusted_p": row.get(
                    "holm_p"
                ),
                "inferential_role": (
                    row.get(
                        "inferential_role"
                    )
                ),
                "uncertainty_status": (
                    status
                ),
                "conservative_interpretation": (
                    interpretation_text(
                        row["metric"],
                        status,
                    )
                ),
                "causal_interpretation_allowed": (
                    int(
                        row["study"]
                        != "replication"
                    )
                ),
            }
        )

    for row in process_rows:
        status = uncertainty_status(
            row.get(
                "ci95_low"
            ),
            row.get(
                "ci95_high"
            ),
        )

        output.append(
            {
                "study": row["study"],
                "profile": row["profile"],
                "intervention": (
                    row["contrast"]
                ),
                "placement": (
                    row["placement"]
                ),
                "outcome_family": (
                    "process"
                ),
                "metric": row["metric"],
                "unit": "raw_units",
                "matched_n": (
                    row["matched_n"]
                ),
                "effect": (
                    row["mean_delta"]
                ),
                "ci95_low": (
                    row["ci95_low"]
                ),
                "ci95_high": (
                    row["ci95_high"]
                ),
                "raw_p": row.get(
                    "sign_flip_p"
                ),
                "adjusted_p": row.get(
                    "bh_q"
                ),
                "inferential_role": (
                    row.get(
                        "inferential_role"
                    )
                ),
                "uncertainty_status": (
                    status
                ),
                "conservative_interpretation": (
                    interpretation_text(
                        row["metric"],
                        status,
                    )
                ),
                "causal_interpretation_allowed": (
                    int(
                        row["study"]
                        != "replication"
                    )
                ),
            }
        )

    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    studies = {
        study: core.load_study(
            study
        )
        for study in (
            "primary",
            "resource",
            "replication",
        )
    }

    pairs = []

    for study, rows in (
        studies.items()
    ):
        pairs.extend(
            core.make_pairs(
                study,
                rows,
            )
        )

    cell_rows = core.cell_summary(
        studies
    )

    binary_rows = (
        corrected_binary_effects(
            pairs
        )
    )

    process_rows = (
        canonical_process_effects(
            pairs
        )
    )

    focused_resource_rows = (
        focused_resource_process(
            process_rows
        )
    )

    prevalence_rows = (
        core.behavior_prevalence(
            studies
        )
    )

    behavior_rows = (
        corrected_behavior_effects(
            pairs
        )
    )

    primary_jobs = (
        core.load_semantic_jobs(
            ROOT
            / "analysis"
            / "semantic-multijudge-v1"
            / "final-repaired-llama-v1",
            PRIMARY_SEMANTIC_FIELDS,
            "primary",
        )
    )

    resource_jobs = (
        core.load_semantic_jobs(
            ROOT
            / "analysis"
            / "semantic-resource-v1"
            / "full"
            / "production-v1.1",
            RESOURCE_SEMANTIC_FIELDS,
            "resource",
        )
    )

    (
        primary_consensus,
        primary_pooled_agreement,
        primary_distributions,
    ) = core.semantic_consensus(
        primary_jobs,
        PRIMARY_SEMANTIC_FIELDS,
    )

    (
        resource_consensus,
        resource_pooled_agreement,
        resource_distributions,
    ) = core.semantic_consensus(
        resource_jobs,
        RESOURCE_SEMANTIC_FIELDS,
    )

    consensus_rows = (
        primary_consensus
        + resource_consensus
    )

    (
        primary_profile_agreement,
        primary_cell_agreement,
        primary_semantic_coverage,
    ) = semantic_agreement_breakdowns(
        primary_jobs,
        PRIMARY_SEMANTIC_FIELDS,
    )

    (
        resource_profile_agreement,
        resource_cell_agreement,
        resource_semantic_coverage,
    ) = semantic_agreement_breakdowns(
        resource_jobs,
        RESOURCE_SEMANTIC_FIELDS,
    )

    (
        said_did_pair_rows,
        said_did_summary_rows,
    ) = build_said_did(
        pairs,
        consensus_rows,
    )

    catalog_rows = effect_catalog(
        binary_rows,
        behavior_rows,
        process_rows,
    )

    replication_rows = (
        core.replication_direction(
            binary_rows,
            process_rows,
        )
    )

    outputs = {
        "cell_performance.csv": (
            cell_rows
        ),
        "matched_binary_effects.csv": (
            binary_rows
        ),
        "matched_process_effects.csv": (
            process_rows
        ),
        "resource_focused_process.csv": (
            focused_resource_rows
        ),
        "behavior_prevalence.csv": (
            prevalence_rows
        ),
        "matched_behavior_effects.csv": (
            behavior_rows
        ),
        "semantic_jobs_primary.csv": (
            primary_jobs
        ),
        "semantic_jobs_resource.csv": (
            resource_jobs
        ),
        "semantic_consensus.csv": (
            consensus_rows
        ),
        "semantic_agreement_pooled.csv": (
            primary_pooled_agreement
            + resource_pooled_agreement
        ),
        "semantic_agreement_by_profile.csv": (
            primary_profile_agreement
            + resource_profile_agreement
        ),
        "semantic_agreement_by_cell.csv": (
            primary_cell_agreement
            + resource_cell_agreement
        ),
        "semantic_coverage.csv": (
            primary_semantic_coverage
            + resource_semantic_coverage
        ),
        "semantic_label_distribution.csv": (
            primary_distributions
            + resource_distributions
        ),
        "said_did_pairs.csv": (
            said_did_pair_rows
        ),
        "said_did_summary.csv": (
            said_did_summary_rows
        ),
        "intervention_effect_catalog.csv": (
            catalog_rows
        ),
        "replication_direction.csv": (
            replication_rows
        ),
    }

    for filename, rows in (
        outputs.items()
    ):
        core.write_csv(
            OUT / filename,
            rows,
        )

    manifest = {
        "analysis_version": VERSION,
        "source_root": str(
            ROOT
            / "analysis"
            / "current"
            / "source"
        ),
        "historical_aggregate_inputs": False,
        "historical_inference_inputs": False,
        "historical_semantic_summary_inputs": False,
        "network_calls": 0,
        "api_calls": 0,
        "agent_calls": 0,
        "verifier_calls": 0,
        "semantic_judge_calls": 0,
        "success_endpoint": (
            "overall_pass >= 1.0"
        ),
        "secondary_test_endpoint": (
            "tests_reward >= 1.0"
        ),
        "process_test": (
            "paired sign-flip randomization "
            "test of zero mean delta"
        ),
        "process_fdr": (
            "Benjamini-Hochberg within "
            "model x intervention x placement"
        ),
        "focused_resource_family": (
            "Holm over 3 capable models x "
            "3 focused process metrics"
        ),
        "semantic_consensus": (
            "two valid judges with exact "
            "field-label agreement"
        ),
        "said_did_status": (
            "descriptive post-treatment; "
            "not causal mediation"
        ),
        "replication_status": (
            "partial descriptive cohort; "
            "not pooled with primary"
        ),
        "llama_success_status": (
            "descriptive capability floor"
        ),
        "canonical_process_metrics": list(
            CANONICAL_PROCESS_METRICS
        ),
        "row_counts": {
            filename: len(rows)
            for filename, rows
            in outputs.items()
        },
    }

    core.write_json(
        OUT / "manifest.json",
        manifest,
    )

    print("=" * 88)
    print("CURRENT ANALYSIS: PASS")
    print("=" * 88)

    for filename, rows in (
        outputs.items()
    ):
        print(
            f"{filename:40s}",
            len(rows),
        )

    print()
    print(
        "Primary behavior prevalence rows:",
        sum(
            row["study"]
            == "primary"
            for row
            in prevalence_rows
        ),
    )

    print(
        "Primary behavior effect rows:",
        sum(
            row["study"]
            == "primary"
            for row
            in behavior_rows
        ),
    )

    print(
        "Strict-consensus said/did rows:",
        sum(
            row[
                "semantic_source"
            ]
            == "strict_consensus"
            for row
            in said_did_summary_rows
        ),
    )

    print(
        "Figure-eligible said/did rows:",
        sum(
            row[
                "figure_eligible"
            ]
            == 1
            for row
            in said_did_summary_rows
        ),
    )

    print()
    print("network calls: 0")
    print("API calls: 0")
    print("semantic judge calls: 0")


if __name__ == "__main__":
    main()
