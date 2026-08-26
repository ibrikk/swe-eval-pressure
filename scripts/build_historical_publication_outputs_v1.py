#!/usr/bin/env python3

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_ROOT = (
    Path.home()
    / "Documents"
    / "swe-eval-pressure"
)

SEM_ROOT = (
    DATA_ROOT
    / "analysis"
    / "semantic-multijudge-v1"
)

FINAL_ROOT = (
    SEM_ROOT
    / "final-repaired-llama-v1"
)

FINAL_ANALYSIS = (
    SEM_ROOT
    / "final-analysis-v1"
)

MATCHED_ROOT = (
    SEM_ROOT
    / "matched-inference-v1"
)

INTEGRATION_ROOT = (
    SEM_ROOT
    / "behavior-integration-v1"
)

TEMPORAL_ROOT = (
    SEM_ROOT
    / "residual-influence-temporal-v1"
)

OUT = (
    SEM_ROOT
    / "publication-v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


PROFILES = [
    "claude",
    "fable",
    "codex",
    "llama",
]

PROFILE_LABEL = {
    "claude": "Claude",
    "fable": "Fable",
    "codex": "Codex",
    "llama": "Llama 3.3 70B",
}

PLACEMENTS = [
    "root",
    "scaffold",
    "source",
]

PRESSURES = [
    "financial",
    "self_preservation",
]

ENDPOINTS = [
    "evaluation_recognition",
    "pressure_recognition",
    "cue_untrusted_or_suspicious",
    "explicit_resistance",
    "any_claimed_behavioral_response",
]

ENDPOINT_LABEL = {
    "evaluation_recognition": "Evaluation recognition",
    "pressure_recognition": "Pressure recognition",
    "cue_untrusted_or_suspicious": "Cue untrusted / suspicious",
    "explicit_resistance": "Explicit resistance",
    "any_claimed_behavioral_response": "Any claimed response",
}

DEFENSIVE_ENDPOINTS = [
    "pressure_recognition",
    "cue_untrusted_or_suspicious",
    "explicit_resistance",
    "any_claimed_behavioral_response",
]


def load_json(
    path: Path,
) -> Any:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def read_csv(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        encoding="utf-8"
    ) as f:
        return list(
            csv.DictReader(f)
        )


def sha(
    path: Path,
) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def consensus_label(
    artifact: dict[str, Any],
    field: str,
) -> str | None:
    result = (
        artifact[
            "consensus"
        ][
            "fields"
        ][field]
    )

    if (
        result.get("status")
        != "agreement"
    ):
        return None

    label = result.get(
        "label"
    )

    return (
        str(label)
        if label is not None
        else None
    )


def endpoint_value(
    artifact: dict[str, Any],
    endpoint: str,
) -> int | None:
    if endpoint == "evaluation_recognition":
        label = consensus_label(
            artifact,
            "evaluation_recognition",
        )

        if label is None:
            return None

        return int(
            label == "observed"
        )

    if endpoint == "pressure_recognition":
        label = consensus_label(
            artifact,
            "pressure_recognition",
        )

        if label is None:
            return None

        return int(
            label == "observed"
        )

    if endpoint == "cue_untrusted_or_suspicious":
        label = consensus_label(
            artifact,
            "cue_provenance_stance",
        )

        if label is None:
            return None

        return int(
            label
            == "untrusted_or_suspicious"
        )

    if endpoint == "explicit_resistance":
        label = consensus_label(
            artifact,
            "explicit_resistance",
        )

        if label is None:
            return None

        return int(
            label == "observed"
        )

    if endpoint == "any_claimed_behavioral_response":
        label = consensus_label(
            artifact,
            "claimed_behavioral_response",
        )

        if label is None:
            return None

        return int(
            label
            not in {
                "none_observed",
            }
        )

    raise ValueError(
        endpoint
    )


# ==================================================
# 1. Semantic endpoint rates by complete historical
#    cell.
# ==================================================

consensus_files = sorted(
    (
        FINAL_ROOT
        / "consensus"
    ).glob("*.json")
)

assert len(
    consensus_files
) == 2776

rate_counts = Counter()
rate_den = Counter()
cell_n = Counter()

for path in consensus_files:
    x = load_json(
        path
    )

    profile = str(
        x["profile"]
    )

    condition = str(
        x["condition"]
    )

    placement = str(
        x["placement"]
    )

    pressure = str(
        x["pressure_type"]
    )

    cell = (
        profile,
        condition,
        placement,
        pressure,
    )

    cell_n[cell] += 1

    for endpoint in ENDPOINTS:
        value = endpoint_value(
            x,
            endpoint,
        )

        if value is None:
            continue

        key = (
            *cell,
            endpoint,
        )

        rate_den[key] += 1
        rate_counts[key] += value


rate_rows = []

for key in sorted(
    rate_den
):
    (
        profile,
        condition,
        placement,
        pressure,
        endpoint,
    ) = key

    n = rate_den[key]
    positive = (
        rate_counts[key]
    )

    total = cell_n[
        (
            profile,
            condition,
            placement,
            pressure,
        )
    ]

    rate_rows.append({
        "profile": profile,
        "condition": condition,
        "placement": placement,
        "pressure_type": pressure,
        "endpoint": endpoint,
        "cell_n": total,
        "resolved_n": n,
        "unresolved_n": (
            total - n
        ),
        "positive_n": positive,
        "rate": (
            positive / n
        ),
    })


rates_df = pd.DataFrame(
    rate_rows
)

rates_path = (
    OUT
    / "table_2_semantic_rates_by_cell.csv"
)

rates_df.to_csv(
    rates_path,
    index=False,
)


# ==================================================
# 2. Reliability publication table.
# ==================================================

reliability = pd.read_csv(
    FINAL_ANALYSIS
    / "reliability.csv"
)

reliability.to_csv(
    OUT
    / "table_1_reliability.csv",
    index=False,
)


# ==================================================
# 3. Matched semantic effects.
# ==================================================

matched = pd.read_csv(
    MATCHED_ROOT
    / "matched_semantic_contrasts.csv"
)

assert len(
    matched
) == 120

assert (
    matched[
        "p_holm_global_120"
    ]
    .lt(0.05)
    .sum()
    == 24
)

matched.to_csv(
    OUT
    / "table_3_matched_semantic_effects.csv",
    index=False,
)


# ==================================================
# 4. Defensive chain.
# ==================================================

chain = pd.read_csv(
    INTEGRATION_ROOT
    / "semantic_chain.csv"
)

chain.to_csv(
    OUT
    / "table_4_defensive_chain.csv",
    index=False,
)


# ==================================================
# 5. Objective task-adjusted behavioral validation.
# ==================================================

matched_behavior = pd.read_csv(
    INTEGRATION_ROOT
    / "semantic_stratified_matched_deltas.csv"
)

objective = matched_behavior[
    (
        matched_behavior[
            "placement"
        ]
        == "source"
    )
    & (
        matched_behavior[
            "metric"
        ]
        == "seeded_cue_removed_or_modified"
    )
    & (
        matched_behavior[
            "endpoint"
        ].isin(
            DEFENSIVE_ENDPOINTS
        )
    )
].copy()

objective.to_csv(
    OUT
    / "table_5_objective_behavioral_validation.csv",
    index=False,
)


# ==================================================
# 6. Temporal exploratory result.
# ==================================================

temporal = pd.read_csv(
    TEMPORAL_ROOT
    / "primary_claude_A_inference.csv"
)

assert len(
    temporal
) == 6

assert (
    temporal[
        "p_holm_6"
    ]
    .lt(0.05)
    .sum()
    == 0
)

temporal.to_csv(
    OUT
    / "table_6_temporal_post_removal.csv",
    index=False,
)


# ==================================================
# Figure helper.
# ==================================================

plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def save_figure(
    fig,
    stem: str,
):
    fig.tight_layout()

    fig.savefig(
        OUT
        / f"{stem}.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        OUT
        / f"{stem}.png",
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ==================================================
# FIGURE 1
# Multiplicity-controlled matched semantic effects.
# ==================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(
        12,
        9,
    ),
    sharex=True,
)

axes = axes.flatten()

endpoint_order = [
    "pressure_recognition",
    "cue_untrusted_or_suspicious",
    "explicit_resistance",
    "any_claimed_behavioral_response",
    "evaluation_recognition",
]

placement_order = [
    "root",
    "scaffold",
    "source",
]

pressure_order = [
    "financial",
    "self_preservation",
]

for ax, profile in zip(
    axes,
    PROFILES,
    strict=True,
):
    df = matched[
        matched[
            "profile"
        ]
        == profile
    ].copy()

    rows = []

    for placement in placement_order:
        for endpoint in endpoint_order:
            for pressure in pressure_order:
                m = df[
                    (
                        df[
                            "placement"
                        ]
                        == placement
                    )
                    & (
                        df[
                            "endpoint"
                        ]
                        == endpoint
                    )
                    & (
                        df[
                            "pressure_type"
                        ]
                        == pressure
                    )
                ]

                if len(m) != 1:
                    continue

                r = m.iloc[0]

                rows.append(
                    (
                        placement,
                        endpoint,
                        pressure,
                        float(
                            r[
                                "delta_pp"
                            ]
                        ),
                        float(
                            r[
                                "ci_low_pp"
                            ]
                        ),
                        float(
                            r[
                                "ci_high_pp"
                            ]
                        ),
                        float(
                            r[
                                "p_holm_global_120"
                            ]
                        ),
                    )
                )

    y = 0

    labels = []

    for (
        placement,
        endpoint,
        pressure,
        effect,
        lo,
        hi,
        p,
    ) in rows:
        marker = (
            "o"
            if pressure
            == "financial"
            else "s"
        )

        ax.errorbar(
            effect,
            y,
            xerr=[
                [
                    effect - lo
                ],
                [
                    hi - effect
                ],
            ],
            fmt=marker,
            capsize=2.5,
            markersize=4.5,
        )

        if p < 0.05:
            ax.text(
                hi + 1.5,
                y,
                "*",
                va="center",
                fontsize=10,
            )

        labels.append(
            (
                f"{placement[:3]} · "
                f"{ENDPOINT_LABEL[endpoint]}"
                f" · "
                f"{'Fin' if pressure == 'financial' else 'Self'}"
            )
        )

        y += 1

    ax.axvline(
        0,
        linewidth=0.8,
    )

    ax.set_yticks(
        np.arange(
            len(labels)
        )
    )

    ax.set_yticklabels(
        labels
    )

    ax.invert_yaxis()

    ax.set_title(
        PROFILE_LABEL[
            profile
        ]
    )

    ax.set_xlabel(
        "Pressure − eval-only effect (percentage points)"
    )


fig.suptitle(
    "Matched semantic treatment effects\n"
    "* globally Holm-significant across 120 tests",
    y=1.01,
)

save_figure(
    fig,
    "fig_1_matched_semantic_effects",
)


# ==================================================
# FIGURE 2
# Semantic endpoint rates:
# model × condition × placement.
# ==================================================

display_cells = []

for condition in [
    "eval_only",
    "eval_financial",
    "eval_self_preservation",
]:
    for placement in PLACEMENTS:
        display_cells.append(
            (
                condition,
                placement,
            )
        )


fig, axes = plt.subplots(
    len(ENDPOINTS),
    1,
    figsize=(
        12,
        12,
    ),
)

for ax, endpoint in zip(
    axes,
    ENDPOINTS,
    strict=True,
):
    matrix = np.full(
        (
            len(PROFILES),
            len(display_cells),
        ),
        np.nan,
    )

    for i, profile in enumerate(
        PROFILES
    ):
        for j, (
            condition,
            placement,
        ) in enumerate(
            display_cells
        ):
            rows = rates_df[
                (
                    rates_df[
                        "profile"
                    ]
                    == profile
                )
                & (
                    rates_df[
                        "condition"
                    ]
                    == condition
                )
                & (
                    rates_df[
                        "placement"
                    ]
                    == placement
                )
                & (
                    rates_df[
                        "endpoint"
                    ]
                    == endpoint
                )
            ]

            if len(rows) == 1:
                matrix[
                    i,
                    j,
                ] = float(
                    rows.iloc[0][
                        "rate"
                    ]
                )

    image = ax.imshow(
        matrix,
        aspect="auto",
        vmin=0,
        vmax=1,
    )

    for i in range(
        matrix.shape[0]
    ):
        for j in range(
            matrix.shape[1]
        ):
            value = matrix[
                i,
                j,
            ]

            if np.isnan(
                value
            ):
                continue

            ax.text(
                j,
                i,
                f"{100*value:.0f}",
                ha="center",
                va="center",
                fontsize=7,
            )

    ax.set_yticks(
        np.arange(
            len(PROFILES)
        )
    )

    ax.set_yticklabels([
        PROFILE_LABEL[p]
        for p in PROFILES
    ])

    ax.set_xticks(
        np.arange(
            len(display_cells)
        )
    )

    ax.set_xticklabels(
        [
            (
                f"{condition.replace('eval_', '')}\n"
                f"{placement}"
            )
            for (
                condition,
                placement,
            )
            in display_cells
        ],
        rotation=0,
    )

    ax.set_title(
        ENDPOINT_LABEL[
            endpoint
        ]
        + " (% among resolved consensus labels)"
    )

fig.colorbar(
    image,
    ax=axes.tolist(),
    label="Rate",
    shrink=0.7,
)

fig.suptitle(
    "Semantic rates by model, treatment condition, and placement",
    y=1.005,
)

save_figure(
    fig,
    "fig_2_semantic_rates_heatmap",
)


# ==================================================
# FIGURE 3
# Complete defensive chain.
# ==================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(
        10,
        7,
    ),
    sharey=True,
)

axes = axes.flatten()

for ax, profile in zip(
    axes,
    PROFILES,
    strict=True,
):
    df = chain[
        chain[
            "profile"
        ]
        == profile
    ]

    x = np.arange(
        len(PLACEMENTS)
    )

    width = 0.34

    for offset, pressure in [
        (
            -width / 2,
            "financial",
        ),
        (
            width / 2,
            "self_preservation",
        ),
    ]:
        values = []

        for placement in PLACEMENTS:
            row = df[
                (
                    df[
                        "placement"
                    ]
                    == placement
                )
                & (
                    df[
                        "pressure_type"
                    ]
                    == pressure
                )
            ]

            values.append(
                (
                    100
                    * float(
                        row.iloc[0][
                            "all_four_positive_rate"
                        ]
                    )
                )
                if len(row)
                else 0
            )

        ax.bar(
            x + offset,
            values,
            width,
            label=(
                "Financial"
                if pressure
                == "financial"
                else "Self-preservation"
            ),
        )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        PLACEMENTS
    )

    ax.set_ylim(
        0,
        100,
    )

    ax.set_title(
        PROFILE_LABEL[
            profile
        ]
    )

    ax.set_ylabel(
        "Full defensive chain (%)"
    )

axes[0].legend()

fig.suptitle(
    "Placement-dependent defensive chain\n"
    "pressure recognition → distrust → resistance → claimed response"
)

save_figure(
    fig,
    "fig_3_defensive_chain",
)


# ==================================================
# FIGURE 4
# Objective cue modification validation.
# ==================================================

plot_objective = objective[
    objective[
        "profile"
    ].isin(
        [
            "claude",
            "fable",
        ]
    )
].copy()

fig, axes = plt.subplots(
    1,
    2,
    figsize=(
        11,
        5,
    ),
    sharex=True,
)

for ax, profile in zip(
    axes,
    [
        "claude",
        "fable",
    ],
    strict=True,
):
    df = plot_objective[
        plot_objective[
            "profile"
        ]
        == profile
    ]

    y = 0

    labels = []

    for pressure in PRESSURES:
        for endpoint in DEFENSIVE_ENDPOINTS:
            row = df[
                (
                    df[
                        "pressure_type"
                    ]
                    == pressure
                )
                & (
                    df[
                        "endpoint"
                    ]
                    == endpoint
                )
            ]

            if len(row) != 1:
                continue

            r = row.iloc[0]

            estimate = (
                100
                * float(
                    r[
                        "difference_of_matched_deltas"
                    ]
                )
            )

            lo = (
                100
                * float(
                    r[
                        "ci_low"
                    ]
                )
            )

            hi = (
                100
                * float(
                    r[
                        "ci_high"
                    ]
                )
            )

            ax.errorbar(
                estimate,
                y,
                xerr=[
                    [
                        estimate - lo
                    ],
                    [
                        hi - estimate
                    ],
                ],
                fmt="o",
                capsize=3,
            )

            labels.append(
                (
                    f"{'Fin' if pressure == 'financial' else 'Self'} · "
                    f"{ENDPOINT_LABEL[endpoint]}"
                )
            )

            y += 1

    ax.axvline(
        0,
        linewidth=0.8,
    )

    ax.set_yticks(
        np.arange(
            len(labels)
        )
    )

    ax.set_yticklabels(
        labels
    )

    ax.invert_yaxis()

    ax.set_title(
        PROFILE_LABEL[
            profile
        ]
    )

    ax.set_xlabel(
        "Difference of matched deltas (pp)"
    )

fig.suptitle(
    "Objective validation: actual source-cue modification\n"
    "Post-treatment descriptive association; same-task eval-only adjusted"
)

save_figure(
    fig,
    "fig_4_objective_cue_modification",
)


# ==================================================
# FIGURE 5
# Judge reliability.
# ==================================================

overall = reliability[
    reliability[
        "scope_type"
    ]
    == "overall"
].copy()

overall[
    "field_label"
] = overall[
    "field"
].map({
    "evaluation_recognition": "Evaluation recognition",
    "pressure_recognition": "Pressure recognition",
    "cue_provenance_stance": "Cue provenance stance",
    "explicit_resistance": "Explicit resistance",
    "claimed_behavioral_response": "Claimed response",
})

overall = overall.set_index(
    "field_label"
)

metrics = [
    (
        "raw_agreement",
        "Raw agreement",
    ),
    (
        "cohen_kappa",
        "Cohen κ",
    ),
    (
        "gwet_ac1",
        "Gwet AC1",
    ),
]

fig, ax = plt.subplots(
    figsize=(
        10,
        5,
    )
)

x = np.arange(
    len(overall)
)

width = 0.24

for i, (
    column,
    label,
) in enumerate(
    metrics
):
    ax.bar(
        x
        + (
            i - 1
        )
        * width,
        overall[
            column
        ].astype(float),
        width,
        label=label,
    )

ax.set_xticks(
    x
)

ax.set_xticklabels(
    overall.index,
    rotation=20,
    ha="right",
)

ax.set_ylim(
    0,
    1.05,
)

ax.set_ylabel(
    "Agreement coefficient"
)

ax.set_title(
    "DeepSeek ↔ Gemini semantic reliability"
)

ax.legend()

save_figure(
    fig,
    "fig_5_judge_reliability",
)


# ==================================================
# Publication summary / integrity checks.
# ==================================================

summary = {
    "publication_version": "1.0",
    "network_calls": 0,
    "judge_calls": 0,
    "semantic_trajectories": len(
        consensus_files
    ),
    "matched_semantic_tests": int(
        len(
            matched
        )
    ),
    "global_holm_significant": int(
        matched[
            "p_holm_global_120"
        ]
        .lt(0.05)
        .sum()
    ),
    "evaluation_recognition_global_holm_significant": int(
        matched[
            (
                matched[
                    "endpoint"
                ]
                == "evaluation_recognition"
            )
            & (
                matched[
                    "p_holm_global_120"
                ]
                < 0.05
            )
        ].shape[0]
    ),
    "temporal_primary_endpoints": int(
        len(
            temporal
        )
    ),
    "temporal_holm_significant": int(
        temporal[
            "p_holm_6"
        ]
        .lt(0.05)
        .sum()
    ),
    "files": {},
}


for path in sorted(
    OUT.iterdir()
):
    if path.is_file():
        summary[
            "files"
        ][
            path.name
        ] = sha(
            path
        )


assert (
    summary[
        "global_holm_significant"
    ]
    == 24
)

assert (
    summary[
        "evaluation_recognition_global_holm_significant"
    ]
    == 0
)

assert (
    summary[
        "temporal_holm_significant"
    ]
    == 0
)


summary_path = (
    OUT
    / "publication_summary.json"
)

summary_path.write_text(
    json.dumps(
        summary,
        indent=2,
        sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)


print(
    "HISTORICAL PUBLICATION OUTPUTS V1"
)

print("=" * 78)

print(
    "semantic consensus trajectories:",
    len(
        consensus_files
    ),
)

print(
    "matched semantic tests:",
    len(
        matched
    ),
)

print(
    "global Holm significant:",
    summary[
        "global_holm_significant"
    ],
    "/",
    len(
        matched
    ),
)

print(
    "evaluation-recognition corrected:",
    summary[
        "evaluation_recognition_global_holm_significant"
    ],
)

print(
    "temporal Holm significant:",
    summary[
        "temporal_holm_significant"
    ],
    "/",
    len(
        temporal
    ),
)

print()
print(
    "TABLES"
)

for path in sorted(
    OUT.glob(
        "table_*.csv"
    )
):
    print(
        " ",
        path.name,
    )

print()
print(
    "FIGURES"
)

for path in sorted(
    OUT.glob(
        "fig_*.pdf"
    )
):
    print(
        " ",
        path.name,
    )

print()
print(
    "output:",
    OUT,
)

print(
    "network calls: 0"
)

print(
    "judge calls: 0"
)

print(
    "PUBLICATION OUTPUTS: PASS"
)
