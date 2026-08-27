#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

RESULTS = (
    ROOT
    / "analysis"
    / "current"
    / "results"
)

AUDIT = (
    ROOT
    / "analysis"
    / "current"
    / "audit"
)

OUT = (
    ROOT
    / "analysis"
    / "current"
    / "findings"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


PRIMARY_FIELDS = (
    "evaluation_recognition",
    "pressure_recognition",
    "cue_provenance_stance",
    "explicit_resistance",
    "claimed_behavioral_response",
)

RESOURCE_FIELDS = (
    "evaluation_recognition",
    "resource_constraint_recognition",
    "resource_constraint_stance",
    "claimed_resource_response",
    "response_conserve_tool_calls",
    "response_batch_commands",
    "response_reduce_exploration",
    "response_reduce_validation",
    "response_prioritize_core_edit",
    "response_stop_early",
    "response_other",
    "evaluation_targeting_intent",
)

TARGET_SEMANTICS = {
    (
        "primary",
        "cue_provenance_stance",
    ): {
        "untrusted_or_suspicious",
    },

    (
        "primary",
        "explicit_resistance",
    ): {
        "observed",
    },

    (
        "primary",
        "claimed_behavioral_response",
    ): {
        "ignore",
        "remove_or_modify_cue",
    },

    (
        "resource",
        "resource_constraint_stance",
    ): {
        "rejected_as_untrusted_or_injection",
        "rejected_other",
    },

    (
        "resource",
        "claimed_resource_response",
    ): {
        "explicitly_ignore_constraint",
    },
}

KEY_SAID_DID_METRICS = {
    "broad_repo_search_any",
    "test_inspection_any",
    "validation_any",
    "iterative_repair_any",
    "provenance_related_inspection_any",
    "external_lookup_any",
    "integrity_sensitive_action_any",
    "raw_tool_calls",
    "behavioral_action_calls",
    "trajectory_steps",
    "validation_calls",
    "input_tokens",
    "output_tokens",
    "duration_sec",
    "tests_pass",
    "overall_pass",
}

KEY_PROCESS_METRICS = {
    "raw_tool_calls",
    "behavioral_action_calls",
    "trajectory_steps",
    "validation_calls",
    "input_tokens",
    "output_tokens",
    "duration_sec",
}


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------


def read_csv(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        encoding="utf-8",
        newline="",
    ) as f:
        return list(
            csv.DictReader(f)
        )


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
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


def number(
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


def integer(
    value: Any,
) -> int:
    try:
        return int(
            float(value)
        )
    except Exception:
        return 0


def interval_excludes_zero(
    low: Any,
    high: Any,
) -> bool:
    lo = number(low)
    hi = number(high)

    if (
        lo is None
        or hi is None
    ):
        return False

    return (
        lo > 0
        or hi < 0
    )


def fmt(
    value: Any,
    digits: int = 2,
) -> str:
    x = number(value)

    if x is None:
        return "—"

    return f"{x:.{digits}f}"


# ------------------------------------------------------------
# Load canonical fresh outputs
# ------------------------------------------------------------


binary = read_csv(
    RESULTS
    / "matched_binary_effects.csv"
)

behavior = read_csv(
    RESULTS
    / "matched_behavior_effects.csv"
)

process = read_csv(
    RESULTS
    / "matched_process_effects.csv"
)

focused = read_csv(
    RESULTS
    / "resource_focused_process.csv"
)

said = read_csv(
    RESULTS
    / "said_did_summary.csv"
)

consensus = read_csv(
    RESULTS
    / "semantic_consensus.csv"
)

primary_jobs = read_csv(
    RESULTS
    / "semantic_jobs_primary.csv"
)

resource_jobs = read_csv(
    RESULTS
    / "semantic_jobs_resource.csv"
)

agreement = read_csv(
    RESULTS
    / "semantic_agreement_pooled.csv"
)

delivery = read_csv(
    AUDIT
    / "treatment_delivery.csv"
)


# ============================================================
# 1. Strict-consensus semantic prevalence by treatment cell
# ============================================================


job_metadata = {}

for row in (
    primary_jobs
    + resource_jobs
):
    key = (
        row["study"],
        row["profile"],
        row["trial_name"],
    )

    if key not in job_metadata:
        job_metadata[key] = {
            "condition":
                row.get(
                    "condition",
                    "",
                ),
            "placement":
                row.get(
                    "placement",
                    "",
                ),
            "pressure_type":
                row.get(
                    "pressure_type",
                    "",
                ),
        }


semantic_groups = defaultdict(
    lambda: {
        "total": 0,
        "resolved": 0,
        "labels": Counter(),
    }
)


for row in consensus:
    study = row["study"]

    fields = (
        PRIMARY_FIELDS
        if study == "primary"
        else RESOURCE_FIELDS
    )

    metadata = job_metadata.get(
        (
            study,
            row["profile"],
            row["trial_name"],
        ),
        {},
    )

    for field in fields:
        status = row.get(
            field
            + "__status",
            "",
        )

        label = row.get(
            field
            + "__label",
            "",
        )

        key = (
            study,
            row["profile"],
            metadata.get(
                "condition",
                "",
            ),
            metadata.get(
                "placement",
                "",
            ),
            metadata.get(
                "pressure_type",
                "",
            ),
            field,
        )

        group = semantic_groups[key]

        group["total"] += 1

        if (
            status == "agreement"
            and label
        ):
            group["resolved"] += 1
            group["labels"][
                label
            ] += 1


semantic_prevalence = []

for key, group in sorted(
    semantic_groups.items()
):
    (
        study,
        profile,
        condition,
        placement,
        pressure_type,
        field,
    ) = key

    total = group["total"]
    resolved = group["resolved"]
    unresolved = (
        total - resolved
    )

    for label, count in sorted(
        group["labels"].items()
    ):
        semantic_prevalence.append(
            {
                "study": study,
                "profile": profile,
                "condition": condition,
                "placement": placement,
                "pressure_type":
                    pressure_type,
                "field": field,
                "label": label,
                "full_cell_n": total,
                "strict_resolved_n":
                    resolved,
                "unresolved_n":
                    unresolved,
                "count": count,
                "rate_among_resolved":
                    (
                        count
                        / resolved
                        if resolved
                        else None
                    ),
                "lower_bound_full_cell":
                    (
                        count
                        / total
                        if total
                        else None
                    ),
                "upper_bound_full_cell":
                    (
                        (
                            count
                            + unresolved
                        )
                        / total
                        if total
                        else None
                    ),
            }
        )


write_csv(
    OUT
    / "semantic_prevalence_by_cell.csv",
    semantic_prevalence,
)


# ============================================================
# 2. Join treatment delivery onto every quantitative effect
# ============================================================


delivery_index = defaultdict(dict)

for row in delivery:
    key = (
        row["study"],
        row["profile"],
        row["condition"],
        row["channel"],
    )

    delivery_index[key][
        row["metric"]
    ] = number(
        row["rate"]
    )


def treatment_cell(
    study: str,
    contrast: str,
    placement: str,
):
    if study == "primary":
        condition = {
            "evaluation_disclosure":
                "eval_only",
            "financial_pressure":
                "eval_financial",
            "self_preservation_pressure":
                "eval_self_preservation",
        }.get(
            contrast
        )

        return (
            condition,
            placement,
        )

    if study == "resource":
        condition = {
            "resource_evaluation_disclosure":
                "eval_only",
            "resource_deprivation":
                "eval_resource_deprivation",
        }.get(
            contrast
        )

        return (
            condition,
            "scaffold",
        )

    return (
        None,
        None,
    )


synthesis_rows = []

for family, rows in (
    (
        "performance",
        binary,
    ),
    (
        "behavior",
        behavior,
    ),
    (
        "process",
        process,
    ),
):
    for row in rows:
        (
            condition,
            channel,
        ) = treatment_cell(
            row["study"],
            row["contrast"],
            row["placement"],
        )

        exposure = (
            delivery_index.get(
                (
                    row["study"],
                    row["profile"],
                    condition,
                    channel,
                ),
                {},
            )
            if condition
            else {}
        )

        if family in {
            "performance",
            "behavior",
        }:
            effect = number(
                row.get(
                    "effect_pp"
                )
            )

            low = number(
                row.get(
                    "ci95_low_pp"
                )
            )

            high = number(
                row.get(
                    "ci95_high_pp"
                )
            )

            adjusted = number(
                row.get(
                    "holm_p"
                )
            )

            unit = (
                "percentage_points"
            )

        else:
            effect = number(
                row.get(
                    "mean_delta"
                )
            )

            low = number(
                row.get(
                    "ci95_low"
                )
            )

            high = number(
                row.get(
                    "ci95_high"
                )
            )

            adjusted = number(
                row.get(
                    "bh_q"
                )
            )

            unit = "raw_units"

        synthesis_rows.append(
            {
                "study":
                    row["study"],
                "profile":
                    row["profile"],
                "intervention":
                    row["contrast"],
                "placement":
                    row["placement"],
                "treatment_condition":
                    condition or "",
                "outcome_family":
                    family,
                "metric":
                    row["metric"],
                "matched_n":
                    row["matched_n"],
                "effect":
                    effect,
                "unit":
                    unit,
                "ci95_low":
                    low,
                "ci95_high":
                    high,
                "adjusted_p_or_q":
                    adjusted,
                "interval_excludes_zero":
                    int(
                        interval_excludes_zero(
                            low,
                            high,
                        )
                    ),
                "artifact_actually_read_rate":
                    exposure.get(
                        "artifact_actually_read"
                    ),
                "eval_cue_observed_rate":
                    exposure.get(
                        "eval_cue_observed"
                    ),
                "pressure_cue_observed_rate":
                    exposure.get(
                        "pressure_cue_observed"
                    ),
                "interpretation":
                    (
                        "increase"
                        if (
                            low is not None
                            and low > 0
                        )
                        else
                        "decrease"
                        if (
                            high is not None
                            and high < 0
                        )
                        else
                        "uncertain"
                    ),
            }
        )


write_csv(
    OUT
    / "intervention_synthesis.csv",
    synthesis_rows,
)


# ============================================================
# 3. Exact “said X, did Y” headline rows
# ============================================================


said_headlines = []

for row in said:
    if (
        row["semantic_source"]
        != "strict_consensus"
    ):
        continue

    if integer(
        row["n_pairs"]
    ) < 5:
        continue

    key = (
        row["study"],
        row["semantic_field"],
    )

    allowed = (
        TARGET_SEMANTICS.get(
            key
        )
    )

    if not allowed:
        continue

    if (
        row["semantic_label"]
        not in allowed
    ):
        continue

    if (
        row["metric"]
        not in KEY_SAID_DID_METRICS
    ):
        continue

    low = number(
        row["ci95_low"]
    )

    high = number(
        row["ci95_high"]
    )

    delta = number(
        row["mean_delta"]
    )

    baseline = number(
        row["baseline_mean"]
    )

    relative = None

    if (
        row["unit"]
        != "percentage_points"
        and baseline not in (
            None,
            0,
        )
        and delta is not None
    ):
        relative = (
            100
            * delta
            / abs(baseline)
        )

    said_headlines.append(
        {
            **row,
            "ci_excludes_zero":
                int(
                    interval_excludes_zero(
                        low,
                        high,
                    )
                ),
            "relative_mean_change_pct":
                relative,
        }
    )


write_csv(
    OUT
    / "said_did_headlines.csv",
    said_headlines,
)


# ============================================================
# 4. Terminal findings digest
# ============================================================


print(
    "=" * 106
)
print(
    "FRESH FINDINGS DIGEST"
)
print(
    "=" * 106
)


# ------------------------------------------------------------
# A. Primary binary outcomes
# ------------------------------------------------------------


print()
print(
    "A. PRIMARY PERFORMANCE — HOLM-CORRECTED"
)
print(
    "-" * 106
)

selected = [
    row
    for row in binary
    if (
        row["study"]
        == "primary"
        and row["profile"]
        != "llama"
        and number(
            row["holm_p"]
        )
        is not None
        and number(
            row["holm_p"]
        )
        <= 0.10
    )
]

selected.sort(
    key=lambda row: (
        number(
            row["holm_p"]
        ),
        -abs(
            number(
                row["effect_pp"]
            )
            or 0
        ),
    )
)

if not selected:
    print(
        "No primary performance contrast "
        "has Holm-adjusted p <= .10."
    )

for row in selected:
    print(
        f"{row['profile']:7s} "
        f"{row['contrast']:29s} "
        f"{row['placement']:8s} "
        f"{row['metric']:13s} "
        f"n={row['matched_n']:>2s} "
        f"Δ={fmt(row['effect_pp']):>7s}pp "
        f"CI=[{fmt(row['ci95_low_pp'])}, "
        f"{fmt(row['ci95_high_pp'])}] "
        f"Holm={fmt(row['holm_p'],4)}"
    )


# ------------------------------------------------------------
# B. Primary behavior
# ------------------------------------------------------------


print()
print(
    "B. PRIMARY BEHAVIOR — HOLM-SIGNIFICANT"
)
print(
    "-" * 106
)

selected = [
    row
    for row in behavior
    if (
        row["study"]
        == "primary"
        and number(
            row["holm_p"]
        )
        is not None
        and number(
            row["holm_p"]
        )
        <= 0.05
    )
]

selected.sort(
    key=lambda row: (
        number(
            row["holm_p"]
        ),
        -abs(
            number(
                row["effect_pp"]
            )
            or 0
        ),
    )
)

print(
    "count:",
    len(selected),
)

for row in selected:
    print(
        f"{row['profile']:7s} "
        f"{row['contrast']:29s} "
        f"{row['placement']:8s} "
        f"{row['metric']:40s} "
        f"Δ={fmt(row['effect_pp']):>7s}pp "
        f"CI=[{fmt(row['ci95_low_pp'])}, "
        f"{fmt(row['ci95_high_pp'])}] "
        f"Holm={fmt(row['holm_p'],4)}"
    )


# ------------------------------------------------------------
# C. Primary key process outcomes
# ------------------------------------------------------------


print()
print(
    "C. PRIMARY KEY PROCESS EFFECTS — BH q <= .05"
)
print(
    "-" * 106
)

selected = [
    row
    for row in process
    if (
        row["study"]
        == "primary"
        and row["metric"]
        in KEY_PROCESS_METRICS
        and number(
            row["bh_q"]
        )
        is not None
        and number(
            row["bh_q"]
        )
        <= 0.05
    )
]

selected.sort(
    key=lambda row: (
        number(
            row["bh_q"]
        ),
        row["profile"],
        row["contrast"],
        row["placement"],
        row["metric"],
    )
)

print(
    "count:",
    len(selected),
)

for row in selected:
    base = number(
        row["baseline_mean"]
    )

    delta = number(
        row["mean_delta"]
    )

    relative = (
        100
        * delta
        / abs(base)
        if (
            base not in (
                None,
                0,
            )
            and delta
            is not None
        )
        else None
    )

    print(
        f"{row['profile']:7s} "
        f"{row['contrast']:29s} "
        f"{row['placement']:8s} "
        f"{row['metric']:25s} "
        f"Δ={fmt(delta):>11s} "
        f"rel={fmt(relative):>8s}% "
        f"CI=[{fmt(row['ci95_low'])}, "
        f"{fmt(row['ci95_high'])}] "
        f"q={fmt(row['bh_q'],4)}"
    )


# ------------------------------------------------------------
# D. Resource focused family
# ------------------------------------------------------------


print()
print(
    "D. RESOURCE DEPRIVATION — PRESPECIFIED 3×3 FAMILY"
)
print(
    "-" * 106
)

for row in sorted(
    focused,
    key=lambda row: (
        number(
            row["focused_holm_p"]
        ),
        row["profile"],
        row["metric"],
    )
):
    print(
        f"{row['profile']:7s} "
        f"{row['metric']:20s} "
        f"n={row['matched_n']:>2s} "
        f"Δ={fmt(row['mean_delta']):>13s} "
        f"CI=[{fmt(row['ci95_low'])}, "
        f"{fmt(row['ci95_high'])}] "
        f"Holm={fmt(row['focused_holm_p'],4)}"
    )


# ------------------------------------------------------------
# E. Semantic reliability warnings
# ------------------------------------------------------------


print()
print(
    "E. SEMANTIC RELIABILITY"
)
print(
    "-" * 106
)

for row in sorted(
    agreement,
    key=lambda row: (
        number(
            row["cohen_kappa"]
        )
        if number(
            row["cohen_kappa"]
        )
        is not None
        else 999,
    )
):
    print(
        f"{row['study']:8s} "
        f"{row['field']:38s} "
        f"agreement="
        f"{100*(number(row['raw_agreement']) or 0):5.1f}% "
        f"κ={fmt(row['cohen_kappa'],3):>6s} "
        f"AC1={fmt(row['gwet_ac1'],3):>6s}"
    )


# ------------------------------------------------------------
# F. EXACT said-X/did-Y question
# ------------------------------------------------------------


print()
print(
    'F. "IT SAID IT WOULD IGNORE/RESIST IT — '
    'BUT BEHAVIOR CHANGED"'
)
print(
    "-" * 106
)

supported = [
    row
    for row in said_headlines
    if (
        integer(
            row["n_pairs"]
        )
        >= 10
        and integer(
            row["ci_excludes_zero"]
        )
        == 1
    )
]

def rank_score(
    row,
):
    delta = abs(
        number(
            row["mean_delta"]
        )
        or 0
    )

    if (
        row["unit"]
        == "percentage_points"
    ):
        return delta

    relative = number(
        row.get(
            "relative_mean_change_pct"
        )
    )

    return (
        abs(relative)
        if relative is not None
        else delta
    )


supported.sort(
    key=rank_score,
    reverse=True,
)

print(
    "strict-consensus supported dissociations:",
    len(supported),
)

for row in supported[:50]:
    print(
        f"{row['study']:8s} "
        f"{row['profile']:7s} "
        f"{row['contrast']:28s} "
        f"{row['placement']:8s} | "
        f"{row['semantic_field']}="
        f"{row['semantic_label']} | "
        f"{row['metric']:31s} "
        f"n={row['n_pairs']:>3s} "
        f"Δ={fmt(row['mean_delta']):>11s} "
        f"CI=[{fmt(row['ci95_low'])}, "
        f"{fmt(row['ci95_high'])}] "
        f"↑{100*(number(row['increased_fraction']) or 0):4.0f}% "
        f"={100*(number(row['unchanged_fraction']) or 0):4.0f}% "
        f"↓{100*(number(row['decreased_fraction']) or 0):4.0f}%"
    )


print()
print(
    "Exact claimed_behavioral_response=ignore groups:"
)

ignore_groups = [
    row
    for row in said_headlines
    if (
        row["study"]
        == "primary"
        and row["semantic_field"]
        == "claimed_behavioral_response"
        and row["semantic_label"]
        == "ignore"
        and integer(
            row["n_pairs"]
        )
        >= 5
    )
]

keys = sorted(
    {
        (
            row["profile"],
            row["contrast"],
            row["placement"],
            integer(
                row["n_pairs"]
            ),
        )
        for row in ignore_groups
    }
)

if not keys:
    print(
        "  No strict-consensus ignore subgroup "
        "has n>=5."
    )
else:
    for key in keys:
        print(
            " ",
            key,
        )


# ------------------------------------------------------------
# Save digest text marker
# ------------------------------------------------------------


print()
print(
    "=" * 106
)
print(
    "OUTPUTS"
)
print(
    "=" * 106
)

for name in (
    "semantic_prevalence_by_cell.csv",
    "intervention_synthesis.csv",
    "said_did_headlines.csv",
):
    print(
        " ",
        OUT / name,
    )
