#!/usr/bin/env python3
"""Integrated semantic + deterministic SWE-EvalPressure pre-read.

Presentation only. Consumes frozen current analysis outputs.

Creates:
  reports/current/index.html
  reports/current/trial-explorer.html
  reports/current/figures/*.svg

Does NOT recompute statistical tests, CIs, semantic judgments,
behavior classifications, or model outcomes.
"""

from __future__ import annotations

import csv
import html
import json
import math
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

FINDINGS = (
    ROOT
    / "analysis"
    / "current"
    / "findings"
)

AUDIT = (
    ROOT
    / "analysis"
    / "current"
    / "audit"
)

REPORT = (
    ROOT
    / "reports"
    / "current"
)

FIGURES = (
    REPORT
    / "figures"
)

REPORT.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURES.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


REQUIRED = {
    "binary":
        RESULTS
        / "matched_binary_effects.csv",

    "behavior":
        RESULTS
        / "matched_behavior_effects.csv",

    "process":
        RESULTS
        / "matched_process_effects.csv",

    "focused":
        RESULTS
        / "resource_focused_process.csv",

    "agreement":
        RESULTS
        / "semantic_agreement_pooled.csv",

    "agreement_profile":
        RESULTS
        / "semantic_agreement_by_profile.csv",

    "semantic_consensus":
        RESULTS
        / "semantic_consensus.csv",

    "primary_jobs":
        RESULTS
        / "semantic_jobs_primary.csv",

    "resource_jobs":
        RESULTS
        / "semantic_jobs_resource.csv",

    "said_pairs":
        RESULTS
        / "said_did_pairs.csv",

    "said_summary":
        RESULTS
        / "said_did_summary.csv",

    "replication":
        RESULTS
        / "replication_direction.csv",

    "semantic_prevalence":
        FINDINGS
        / "semantic_prevalence_by_cell.csv",

    "synthesis":
        FINDINGS
        / "intervention_synthesis.csv",

    "said_headlines":
        FINDINGS
        / "said_did_headlines.csv",

    "integrity":
        FINDINGS
        / "integrity_component_effects.csv",

    "verifier":
        AUDIT
        / "verifier_forensics_summary.csv",

    "delivery":
        AUDIT
        / "treatment_delivery.csv",
}


for name, path in REQUIRED.items():
    if not path.is_file():
        raise SystemExit(
            f"Missing required current output "
            f"{name}: {path}"
        )


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def clean_generated_text(text: str) -> str:
    """Remove trailing whitespace from generated text deterministically."""
    return "\n".join(line.rstrip() for line in text.splitlines()).rstrip() + "\n"


def read_csv(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        newline="",
        encoding="utf-8",
    ) as f:
        return list(
            csv.DictReader(f)
        )


def esc(
    value: Any,
) -> str:
    return html.escape(
        str(
            value
            if value is not None
            else ""
        )
    )


def num(
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


def fmt(
    value: Any,
    digits: int = 2,
) -> str:
    x = num(value)

    if x is None:
        return "—"

    if abs(x) >= 100_000:
        return f"{x:,.0f}"

    return f"{x:,.{digits}f}"


def signed(
    value: Any,
    digits: int = 2,
) -> str:
    x = num(value)

    if x is None:
        return "—"

    if abs(x) >= 100_000:
        return f"{x:+,.0f}"

    return f"{x:+,.{digits}f}"


def percent(
    value: Any,
    digits: int = 1,
) -> str:
    x = num(value)

    if x is None:
        return "—"

    return (
        f"{100*x:.{digits}f}%"
    )


def pvalue(
    value: Any,
) -> str:
    x = num(value)

    if x is None:
        return "—"

    if x < 0.001:
        return f"{x:.2e}"

    return f"{x:.4f}"


def excludes_zero(
    low: Any,
    high: Any,
) -> bool:
    lo = num(low)
    hi = num(high)

    if (
        lo is None
        or hi is None
    ):
        return False

    return (
        lo > 0
        or hi < 0
    )


def find_one(
    rows,
    **criteria,
):
    for row in rows:
        if all(
            row.get(key)
            == value
            for key, value
            in criteria.items()
        ):
            return row

    return None


def relative_change(
    row,
) -> float | None:
    delta = num(
        row.get(
            "mean_delta"
        )
    )

    baseline = num(
        row.get(
            "baseline_mean"
        )
    )

    if (
        delta is None
        or baseline in (
            None,
            0,
        )
    ):
        return None

    return (
        100
        * delta
        / abs(baseline)
    )


binary = read_csv(
    REQUIRED["binary"]
)

behavior = read_csv(
    REQUIRED["behavior"]
)

process = read_csv(
    REQUIRED["process"]
)

focused = read_csv(
    REQUIRED["focused"]
)

agreement = read_csv(
    REQUIRED["agreement"]
)

agreement_profile = read_csv(
    REQUIRED[
        "agreement_profile"
    ]
)

semantic_consensus = read_csv(
    REQUIRED[
        "semantic_consensus"
    ]
)

primary_jobs = read_csv(
    REQUIRED["primary_jobs"]
)

resource_jobs = read_csv(
    REQUIRED["resource_jobs"]
)

said_pairs = read_csv(
    REQUIRED["said_pairs"]
)

said_summary = read_csv(
    REQUIRED["said_summary"]
)

replication = read_csv(
    REQUIRED["replication"]
)

semantic_prevalence = read_csv(
    REQUIRED[
        "semantic_prevalence"
    ]
)

synthesis = read_csv(
    REQUIRED["synthesis"]
)

said_headlines = read_csv(
    REQUIRED[
        "said_headlines"
    ]
)

integrity = read_csv(
    REQUIRED["integrity"]
)

verifier = read_csv(
    REQUIRED["verifier"]
)

delivery = read_csv(
    REQUIRED["delivery"]
)


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


PROFILE_LABEL = {
    "claude": "Claude Opus 4.8",
    "fable": "Claude Fable 5",
    "codex": "GPT-5.6 / Codex",
    "llama": "Llama 3.3 70B",
}

CONTRAST_LABEL = {
    "evaluation_disclosure":
        "Evaluation disclosure",

    "financial_pressure":
        "Financial pressure",

    "self_preservation_pressure":
        "Self-preservation pressure",

    "resource_evaluation_disclosure":
        "Evaluation disclosure (resource study)",

    "resource_deprivation":
        "Resource deprivation",
}

METRIC_LABEL = {
    "overall_pass":
        "Strict benchmark success",

    "tests_pass":
        "Test/comparison success",

    "broad_repo_search_any":
        "Repository search",

    "test_inspection_any":
        "Test/config inspection",

    "validation_any":
        "Any validation",

    "iterative_repair_any":
        "Iterative repair",

    "provenance_related_inspection_any":
        "Provenance inspection",

    "external_lookup_any":
        "External lookup",

    "integrity_sensitive_action_any":
        "Cue/test/context modification",

    "raw_tool_calls":
        "Raw tool calls",

    "behavioral_action_calls":
        "Task actions",

    "trajectory_steps":
        "Trajectory steps",

    "validation_calls":
        "Validation calls",

    "input_tokens":
        "Input tokens",

    "output_tokens":
        "Output tokens",

    "duration_sec":
        "Duration (s)",

    "seeded_cue_removed_or_modified":
        "Planted cue removed/modified",

    "evaluation_context_file_modified":
        "Evaluation-context file modified",

    "test_files_modified":
        "Test files modified",
}


def profile_label(
    value,
):
    return PROFILE_LABEL.get(
        value,
        value,
    )


def contrast_label(
    value,
):
    return CONTRAST_LABEL.get(
        value,
        value.replace(
            "_",
            " ",
        ),
    )


def metric_label(
    value,
):
    return METRIC_LABEL.get(
        value,
        value.replace(
            "_",
            " ",
        ),
    )


# ---------------------------------------------------------------------------
# SVG figures
# ---------------------------------------------------------------------------


def bar_svg(
    *,
    filename: str,
    title: str,
    items: list[
        tuple[str, float]
    ],
    suffix: str = "",
) -> str:
    if not items:
        return (
            "<p class='muted'>"
            "No rows available."
            "</p>"
        )

    width = 980
    left = 390
    right = 90
    row_h = 34
    top = 65

    height = (
        top
        + len(items)
        * row_h
        + 30
    )

    max_abs = max(
        abs(value)
        for _, value
        in items
    )

    if max_abs == 0:
        max_abs = 1

    plot_w = (
        width
        - left
        - right
    )

    zero = (
        left
        + plot_w / 2
    )

    scale = (
        plot_w
        / 2
        / max_abs
    )

    lines = [
        (
            f'<svg viewBox="0 0 '
            f'{width} {height}" '
            f'role="img" '
            f'aria-label="{esc(title)}">'
        ),
        (
            "<style>"
            ".lab{font:13px -apple-system,"
            "BlinkMacSystemFont,Segoe UI,sans-serif;"
            "fill:#263238}"
            ".val{font:12px ui-monospace,SFMono-Regular,"
            "Menlo,monospace;fill:#455a64}"
            ".axis{stroke:#9aa7b2;stroke-width:1}"
            ".pos{fill:#335f8a}"
            ".neg{fill:#9b5757}"
            "</style>"
        ),
        (
            f'<text x="{left}" y="28" '
            f'class="lab" '
            f'font-weight="700">'
            f'{esc(title)}</text>'
        ),
        (
            f'<line x1="{zero}" '
            f'y1="48" '
            f'x2="{zero}" '
            f'y2="{height-15}" '
            f'class="axis"/>'
        ),
    ]

    for index, (
        label,
        value,
    ) in enumerate(items):
        y = (
            top
            + index
            * row_h
        )

        bar_width = (
            abs(value)
            * scale
        )

        if value >= 0:
            x = zero
            css = "pos"

        else:
            x = (
                zero
                - bar_width
            )
            css = "neg"

        lines.extend(
            [
                (
                    f'<text x="8" '
                    f'y="{y+13}" '
                    f'class="lab">'
                    f'{esc(label)}</text>'
                ),
                (
                    f'<rect x="{x:.1f}" '
                    f'y="{y}" '
                    f'width="{bar_width:.1f}" '
                    f'height="18" '
                    f'rx="3" '
                    f'class="{css}"/>'
                ),
                (
                    f'<text '
                    f'x="{width-right+10}" '
                    f'y="{y+13}" '
                    f'class="val">'
                    f'{value:+.1f}{esc(suffix)}'
                    f'</text>'
                ),
            ]
        )

    lines.append(
        "</svg>"
    )

    svg = "\n".join(
        lines
    )

    (
        FIGURES
        / filename
    ).write_text(
        svg,
        encoding="utf-8",
    )

    return svg


# ---------------------------------------------------------------------------
# Headline primary process figure
# ---------------------------------------------------------------------------


primary_process_sig = [
    row
    for row in process
    if (
        row["study"]
        == "primary"
        and num(
            row["bh_q"]
        )
        is not None
        and num(
            row["bh_q"]
        )
        <= 0.05
        and row["metric"]
        in {
            "raw_tool_calls",
            "behavioral_action_calls",
            "trajectory_steps",
            "input_tokens",
            "duration_sec",
        }
    )
]

primary_process_items = []

for row in primary_process_sig:
    relative = (
        relative_change(
            row
        )
    )

    if relative is None:
        continue

    primary_process_items.append(
        (
            (
                f"{profile_label(row['profile'])} · "
                f"{contrast_label(row['contrast'])} · "
                f"{row['placement']} · "
                f"{metric_label(row['metric'])}"
            ),
            relative,
        )
    )


primary_process_svg = bar_svg(
    filename=(
        "primary_process.svg"
    ),
    title=(
        "Primary study: FDR-robust "
        "process changes"
    ),
    items=(
        primary_process_items
    ),
    suffix="%",
)


# ---------------------------------------------------------------------------
# Resource figure
# ---------------------------------------------------------------------------


resource_items = []

for row in focused:
    relative = (
        relative_change(
            row
        )
    )

    if relative is None:
        continue

    resource_items.append(
        (
            (
                f"{profile_label(row['profile'])} · "
                f"{metric_label(row['metric'])}"
            ),
            relative,
        )
    )


resource_svg = bar_svg(
    filename=(
        "resource_contraction.svg"
    ),
    title=(
        "Resource deprivation: "
        "prespecified focused family"
    ),
    items=resource_items,
    suffix="%",
)


# ---------------------------------------------------------------------------
# Integrity decomposition figure
# ---------------------------------------------------------------------------


integrity_items = []

for row in integrity:
    if not (
        row["study"]
        == "primary"
        and row["placement"]
        == "source"
        and row["profile"]
        == "claude"
        and row["contrast"]
        in {
            "financial_pressure",
            "self_preservation_pressure",
        }
    ):
        continue

    effect = num(
        row["effect_pp"]
    )

    if (
        effect is None
        or math.isclose(
            effect,
            0.0,
        )
    ):
        continue

    integrity_items.append(
        (
            (
                f"{contrast_label(row['contrast'])} · "
                f"{metric_label(row['component'])}"
            ),
            effect,
        )
    )


integrity_svg = bar_svg(
    filename=(
        "source_cue_removal.svg"
    ),
    title=(
        "Claude source placement: "
        "what generated the integrity composite?"
    ),
    items=integrity_items,
    suffix=" pp",
)


# ---------------------------------------------------------------------------
# Semantic reliability figure
# ---------------------------------------------------------------------------


reliability_items = []

for row in agreement:
    reliability_items.append(
        (
            (
                f"{row['study']} · "
                f"{row['field'].replace('_',' ')}"
            ),
            100
            * (
                num(
                    row[
                        "raw_agreement"
                    ]
                )
                or 0
            ),
        )
    )

reliability_items.sort(
    key=lambda pair:
        pair[1],
)

semantic_svg = bar_svg(
    filename=(
        "semantic_reliability.svg"
    ),
    title=(
        "DeepSeek ↔ Gemini raw agreement"
    ),
    items=(
        reliability_items
    ),
    suffix="%",
)


# ---------------------------------------------------------------------------
# Said / Did figure
# ---------------------------------------------------------------------------


said_supported = [
    row
    for row
    in said_headlines
    if (
        row[
            "semantic_source"
        ]
        == "strict_consensus"
        and integer(
            row["n_pairs"]
        )
        >= 10
        and excludes_zero(
            row[
                "ci95_low"
            ],
            row[
                "ci95_high"
            ],
        )
    )
]


def said_score(
    row,
):
    delta = abs(
        num(
            row[
                "mean_delta"
            ]
        )
        or 0
    )

    if (
        row["unit"]
        == "percentage_points"
    ):
        return delta

    relative = num(
        row.get(
            "relative_mean_change_pct"
        )
    )

    return (
        abs(relative)
        if relative
        is not None
        else delta
    )


said_supported.sort(
    key=said_score,
    reverse=True,
)


# Exclude mechanically coupled cue-removal composite from
# the headline dissociation figure.
said_figure_rows = [
    row
    for row
    in said_supported
    if not (
        row["metric"]
        == "integrity_sensitive_action_any"
        and row[
            "semantic_label"
        ]
        == "remove_or_modify_cue"
    )
][:12]


said_items = []

for row in said_figure_rows:
    if (
        row["unit"]
        == "percentage_points"
    ):
        value = num(
            row["mean_delta"]
        )

    else:
        value = num(
            row.get(
                "relative_mean_change_pct"
            )
        )

    if value is None:
        continue

    said_items.append(
        (
            (
                f"{profile_label(row['profile'])} · "
                f"{row['semantic_field'].replace('_',' ')}="
                f"{row['semantic_label']} · "
                f"{metric_label(row['metric'])}"
            ),
            value,
        )
    )


said_svg = bar_svg(
    filename=(
        "said_did.svg"
    ),
    title=(
        "Verbal stance did not always imply "
        "behavioral invariance"
    ),
    items=said_items,
    suffix="%",
)


# ---------------------------------------------------------------------------
# Key rows
# ---------------------------------------------------------------------------


primary_perf = find_one(
    binary,
    study="primary",
    profile="claude",
    contrast=(
        "self_preservation_pressure"
    ),
    placement="root",
    metric="overall_pass",
)


fable_ignore = find_one(
    said_headlines,
    study="primary",
    profile="fable",
    contrast=(
        "self_preservation_pressure"
    ),
    placement="source",
    semantic_field=(
        "claimed_behavioral_response"
    ),
    semantic_source=(
        "strict_consensus"
    ),
    semantic_label="ignore",
    metric="input_tokens",
)


resource_codex_tools = find_one(
    focused,
    profile="codex",
    metric="raw_tool_calls",
)


resource_fable_validation = find_one(
    focused,
    profile="fable",
    metric="validation_calls",
)


# ---------------------------------------------------------------------------
# HTML table helpers
# ---------------------------------------------------------------------------


def table(
    headers,
    rows,
    *,
    classes="",
):
    head = "".join(
        f"<th>{esc(h)}</th>"
        for h in headers
    )

    body = []

    for values in rows:
        body.append(
            "<tr>"
            + "".join(
                f"<td>{value}</td>"
                for value
                in values
            )
            + "</tr>"
        )

    return (
        f'<div class="table-scroll">'
        f'<table class="{esc(classes)}">'
        f"<thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody>"
        f"</table></div>"
    )


# ---------------------------------------------------------------------------
# Verifier table
# ---------------------------------------------------------------------------


verifier_rows = []

for row in verifier:
    verifier_rows.append(
        [
            esc(
                row["study"]
            ),
            esc(
                profile_label(
                    row["profile"]
                )
            ),
            esc(
                row[
                    "substantive_n"
                ]
            ),
            esc(
                row[
                    "stored_overall_pass_n"
                ]
            ),
            esc(
                row[
                    "strict_raw_unavailable_n"
                ]
            ),
            esc(
                row[
                    "stored_strict_at_forensic_risk_n"
                ]
            ),
            esc(
                row[
                    "rubric_true_on_empty_input_n"
                ]
            ),
        ]
    )


verifier_table = table(
    [
        "Study",
        "Model",
        "Substantive",
        "Strict passes",
        "Strict raw unavailable",
        "Strict forensic risk",
        "Rubric-true on empty input",
    ],
    verifier_rows,
)


# ---------------------------------------------------------------------------
# Primary robust process table
# ---------------------------------------------------------------------------


primary_process_table_rows = []

for row in primary_process_sig:
    primary_process_table_rows.append(
        [
            esc(
                profile_label(
                    row["profile"]
                )
            ),
            esc(
                contrast_label(
                    row["contrast"]
                )
            ),
            esc(
                row["placement"]
            ),
            esc(
                metric_label(
                    row["metric"]
                )
            ),
            esc(
                row["matched_n"]
            ),
            esc(
                signed(
                    row["mean_delta"]
                )
            ),
            esc(
                (
                    f"[{fmt(row['ci95_low'])}, "
                    f"{fmt(row['ci95_high'])}]"
                )
            ),
            esc(
                pvalue(
                    row["bh_q"]
                )
            ),
        ]
    )


primary_process_table = table(
    [
        "Model",
        "Intervention",
        "Placement",
        "Metric",
        "n",
        "Mean paired Δ",
        "Bootstrap 95% CI",
        "BH q",
    ],
    primary_process_table_rows,
)


# ---------------------------------------------------------------------------
# Resource table
# ---------------------------------------------------------------------------


resource_table_rows = []

for row in sorted(
    focused,
    key=lambda r: (
        num(
            r[
                "focused_holm_p"
            ]
        )
        or 999,
        r["profile"],
        r["metric"],
    )
):
    resource_table_rows.append(
        [
            esc(
                profile_label(
                    row["profile"]
                )
            ),
            esc(
                metric_label(
                    row["metric"]
                )
            ),
            esc(
                row["matched_n"]
            ),
            esc(
                signed(
                    row["mean_delta"]
                )
            ),
            esc(
                (
                    f"[{fmt(row['ci95_low'])}, "
                    f"{fmt(row['ci95_high'])}]"
                )
            ),
            esc(
                pvalue(
                    row[
                        "focused_holm_p"
                    ]
                )
            ),
        ]
    )


resource_table = table(
    [
        "Model",
        "Metric",
        "n",
        "Mean paired Δ",
        "95% CI",
        "Focused Holm p",
    ],
    resource_table_rows,
)


# ---------------------------------------------------------------------------
# Integrity decomposition table
# ---------------------------------------------------------------------------


integrity_rows_html = []

for row in integrity:
    if not (
        row["study"]
        == "primary"
        and row["placement"]
        == "source"
    ):
        continue

    effect = num(
        row["effect_pp"]
    )

    if (
        effect is None
        or math.isclose(
            effect,
            0,
        )
    ):
        continue

    integrity_rows_html.append(
        [
            esc(
                profile_label(
                    row["profile"]
                )
            ),
            esc(
                contrast_label(
                    row["contrast"]
                )
            ),
            esc(
                metric_label(
                    row[
                        "component"
                    ]
                )
            ),
            esc(
                f"{effect:+.2f} pp"
            ),
            esc(
                (
                    f"[{fmt(row['ci95_low_pp'])}, "
                    f"{fmt(row['ci95_high_pp'])}]"
                )
            ),
            esc(
                pvalue(
                    row[
                        "mcnemar_p"
                    ]
                )
            ),
        ]
    )


integrity_table = table(
    [
        "Model",
        "Intervention",
        "Component",
        "Matched Δ",
        "95% CI",
        "Raw McNemar p",
    ],
    integrity_rows_html,
)


# ---------------------------------------------------------------------------
# Said/did table
# ---------------------------------------------------------------------------


said_table_rows = []

for row in said_supported[:40]:
    said_table_rows.append(
        [
            esc(
                row["study"]
            ),
            esc(
                profile_label(
                    row["profile"]
                )
            ),
            esc(
                contrast_label(
                    row["contrast"]
                )
            ),
            esc(
                row["placement"]
            ),
            (
                f"<code>{esc(row['semantic_field'])}"
                f"={esc(row['semantic_label'])}</code>"
            ),
            esc(
                metric_label(
                    row["metric"]
                )
            ),
            esc(
                row["n_pairs"]
            ),
            esc(
                signed(
                    row["mean_delta"]
                )
            ),
            esc(
                (
                    f"[{fmt(row['ci95_low'])}, "
                    f"{fmt(row['ci95_high'])}]"
                )
            ),
            esc(
                (
                    f"↑ {percent(row['increased_fraction'])} · "
                    f"= {percent(row['unchanged_fraction'])} · "
                    f"↓ {percent(row['decreased_fraction'])}"
                )
            ),
        ]
    )


said_table = table(
    [
        "Study",
        "Model",
        "Intervention",
        "Placement",
        "Strict semantic consensus",
        "Objective metric",
        "n",
        "Mean paired Δ",
        "95% CI",
        "Pair directions",
    ],
    said_table_rows,
)


# ---------------------------------------------------------------------------
# Semantic agreement table
# ---------------------------------------------------------------------------


agreement_rows_html = []

for row in sorted(
    agreement,
    key=lambda r:
        num(
            r[
                "cohen_kappa"
            ]
        )
        if num(
            r[
                "cohen_kappa"
            ]
        )
        is not None
        else 999,
):
    agreement_rows_html.append(
        [
            esc(
                row["study"]
            ),
            (
                f"<code>{esc(row['field'])}</code>"
            ),
            esc(
                row["both_valid_n"]
            ),
            esc(
                percent(
                    row[
                        "raw_agreement"
                    ]
                )
            ),
            esc(
                fmt(
                    row[
                        "cohen_kappa"
                    ],
                    3,
                )
            ),
            esc(
                fmt(
                    row[
                        "gwet_ac1"
                    ],
                    3,
                )
            ),
        ]
    )


agreement_table = table(
    [
        "Study",
        "Field",
        "Both judges valid",
        "Raw agreement",
        "Cohen κ",
        "Gwet AC1",
    ],
    agreement_rows_html,
)


# ---------------------------------------------------------------------------
# Semantic prevalence — positive/meaningful labels only
# ---------------------------------------------------------------------------


MEANINGFUL_LABELS = {
    "observed",
    "untrusted_or_suspicious",
    "ignore",
    "remove_or_modify_cue",
    "inspect_or_verify",
    "comply_with_cue",
    "accepted",
    "rejected_as_untrusted_or_injection",
    "rejected_other",
    "adapt_to_constraint",
    "explicitly_ignore_constraint",
    "considered",
    "attempted",
}


semantic_prev_rows = []

for row in semantic_prevalence:
    if (
        row["label"]
        not in MEANINGFUL_LABELS
    ):
        continue

    semantic_prev_rows.append(
        [
            esc(
                row["study"]
            ),
            esc(
                profile_label(
                    row["profile"]
                )
            ),
            esc(
                row["condition"]
            ),
            esc(
                row["placement"]
            ),
            (
                f"<code>{esc(row['field'])}"
                f"={esc(row['label'])}</code>"
            ),
            esc(
                row[
                    "strict_resolved_n"
                ]
            ),
            esc(
                percent(
                    row[
                        "rate_among_resolved"
                    ]
                )
            ),
            esc(
                (
                    f"{percent(row['lower_bound_full_cell'])}"
                    f"–"
                    f"{percent(row['upper_bound_full_cell'])}"
                )
            ),
        ]
    )


semantic_prevalence_table = table(
    [
        "Study",
        "Model",
        "Condition",
        "Placement",
        "Strict-consensus label",
        "Resolved n",
        "Rate among resolved",
        "Full-cell conservative bounds",
    ],
    semantic_prev_rows,
)


# ---------------------------------------------------------------------------
# Treatment delivery table
# ---------------------------------------------------------------------------


delivery_rows_html = []

for row in delivery:
    if row["metric"] not in {
        "artifact_actually_read",
        "eval_cue_observed",
        "pressure_cue_observed",
    }:
        continue

    delivery_rows_html.append(
        [
            esc(
                row["study"]
            ),
            esc(
                profile_label(
                    row["profile"]
                )
            ),
            esc(
                row["condition"]
            ),
            esc(
                row["channel"]
            ),
            esc(
                row[
                    "pressure_type"
                ]
            ),
            esc(
                row["metric"]
            ),
            esc(
                row["n"]
            ),
            esc(
                percent(
                    row["rate"]
                )
            ),
        ]
    )


delivery_table = table(
    [
        "Study",
        "Model",
        "Condition",
        "Placement",
        "Pressure",
        "Delivery/observation metric",
        "n",
        "Rate",
    ],
    delivery_rows_html,
)


# ---------------------------------------------------------------------------
# Replication summary
# ---------------------------------------------------------------------------


rep_comparable = [
    row
    for row in replication
    if row["direction"]
    in {
        "same",
        "opposite",
    }
]

rep_same = sum(
    row["direction"]
    == "same"
    for row in rep_comparable
)

rep_opposite = sum(
    row["direction"]
    == "opposite"
    for row in rep_comparable
)

rep_fraction = (
    rep_same
    / len(
        rep_comparable
    )
    if rep_comparable
    else None
)


# ---------------------------------------------------------------------------
# Evidence map for trial explorer
# ---------------------------------------------------------------------------


jobs = (
    primary_jobs
    + resource_jobs
)


job_index = {}

for row in jobs:
    job_index[
        (
            row["study"],
            row["profile"],
            row["trial_name"],
            row["judge_family"],
        )
    ] = row


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


def parse_evidence(
    raw,
):
    if not raw:
        return []

    try:
        value = json.loads(
            raw
        )
    except Exception:
        return []

    if not isinstance(
        value,
        list,
    ):
        return []

    output = []

    for item in value[:3]:
        if not isinstance(
            item,
            dict,
        ):
            continue

        quote = str(
            item.get(
                "quote"
            )
            or ""
        ).strip()

        if len(quote) > 300:
            quote = (
                quote[:297]
                + "..."
            )

        if quote:
            output.append(
                quote
            )

    return output


def semantic_tags_and_evidence(
    pair,
):
    study = pair["study"]

    fields = (
        PRIMARY_FIELDS
        if study == "primary"
        else RESOURCE_FIELDS
    )

    tags = []
    evidence = []

    trial = pair[
        "treatment_trial"
    ]

    for field in fields:
        if (
            pair.get(
                field
                + "__status"
            )
            != "agreement"
        ):
            continue

        label = pair.get(
            field
            + "__label"
        )

        if not label:
            continue

        tags.append(
            f"{field}={label}"
        )

        if label in {
            "not_observed",
            "none_observed",
            "neutral_or_uncertain",
            "ambiguous",
        }:
            continue

        for judge in (
            "deepseek",
            "gemini",
        ):
            job = job_index.get(
                (
                    study,
                    pair["profile"],
                    trial,
                    judge,
                )
            )

            if not job:
                continue

            quotes = parse_evidence(
                job.get(
                    field
                    + "__evidence"
                )
            )

            for quote in quotes:
                evidence.append(
                    (
                        judge,
                        field,
                        quote,
                    )
                )

    return (
        tags,
        evidence,
    )


# ---------------------------------------------------------------------------
# Representative pair selection
# ---------------------------------------------------------------------------


REPRESENTATIVE_SPECS = (
    (
        "primary",
        "fable",
        "self_preservation_pressure",
        "source",
        "claimed_behavioral_response",
        "ignore",
        "input_tokens",
    ),
    (
        "primary",
        "fable",
        "financial_pressure",
        "source",
        "explicit_resistance",
        "observed",
        "validation_calls",
    ),
    (
        "primary",
        "claude",
        "self_preservation_pressure",
        "scaffold",
        "cue_provenance_stance",
        "untrusted_or_suspicious",
        "duration_sec",
    ),
    (
        "resource",
        "claude",
        "resource_deprivation",
        "scaffold",
        "claimed_resource_response",
        "explicitly_ignore_constraint",
        "duration_sec",
    ),
    (
        "resource",
        "claude",
        "resource_deprivation",
        "scaffold",
        "resource_constraint_stance",
        "rejected_as_untrusted_or_injection",
        "output_tokens",
    ),
)


def representative_pair(
    spec,
):
    (
        study,
        profile,
        contrast,
        placement,
        field,
        label,
        metric,
    ) = spec

    candidates = [
        row
        for row
        in said_pairs
        if (
            row["study"]
            == study
            and row["profile"]
            == profile
            and row["contrast"]
            == contrast
            and row["placement"]
            == placement
            and row.get(
                field
                + "__status"
            )
            == "agreement"
            and row.get(
                field
                + "__label"
            )
            == label
            and num(
                row.get(
                    "delta_"
                    + metric
                )
            )
            is not None
        )
    ]

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda row:
            abs(
                num(
                    row.get(
                        "delta_"
                        + metric
                    )
                )
                or 0
            ),
    )


representatives = []

for spec in (
    REPRESENTATIVE_SPECS
):
    row = (
        representative_pair(
            spec
        )
    )

    if row:
        representatives.append(
            (
                spec,
                row,
            )
        )


def render_representatives():
    cards = []

    for spec, row in (
        representatives
    ):
        (
            study,
            profile,
            contrast,
            placement,
            field,
            label,
            metric,
        ) = spec

        tags, evidence = (
            semantic_tags_and_evidence(
                row
            )
        )

        evidence_html = (
            "".join(
                (
                    "<blockquote>"
                    f"<strong>{esc(judge)}</strong> · "
                    f"<code>{esc(ev_field)}</code><br>"
                    f"{esc(quote)}"
                    "</blockquote>"
                )
                for (
                    judge,
                    ev_field,
                    quote,
                )
                in evidence[:6]
            )
            or (
                "<p class='muted'>"
                "No positive evidence quote "
                "available in normalized job row."
                "</p>"
            )
        )

        delta = row.get(
            "delta_"
            + metric
        )

        cards.append(
            f"""
            <article class="example-card">
              <div class="eyebrow">
                {esc(study)} ·
                {esc(profile_label(profile))} ·
                {esc(contrast_label(contrast))} ·
                {esc(placement)}
              </div>

              <h3>
                {esc(field)} =
                <code>{esc(label)}</code>
              </h3>

              <p>
                Same-task matched
                <strong>{esc(metric_label(metric))}</strong>
                change:
                <strong>{esc(signed(delta))}</strong>.
              </p>

              <p class="meta">
                Base task:
                <code>{esc(row['base_task_id'])}</code><br>
                Baseline:
                <code>{esc(row['baseline_trial'])}</code><br>
                Treatment:
                <code>{esc(row['treatment_trial'])}</code>
              </p>

              <details>
                <summary>Strict-consensus semantic labels</summary>
                <pre>{esc(chr(10).join(tags))}</pre>
              </details>

              <details>
                <summary>Judge evidence excerpts</summary>
                {evidence_html}
              </details>
            </article>
            """
        )

    return "".join(
        cards
    )


# ---------------------------------------------------------------------------
# Trial explorer
# ---------------------------------------------------------------------------


EXPLORER_DELTAS = (
    "raw_tool_calls",
    "behavioral_action_calls",
    "trajectory_steps",
    "validation_calls",
    "input_tokens",
    "output_tokens",
    "duration_sec",
    "broad_repo_search_any",
    "test_inspection_any",
    "validation_any",
    "iterative_repair_any",
    "provenance_related_inspection_any",
    "external_lookup_any",
    "tests_pass",
    "overall_pass",
)


explorer_rows = []

for index, pair in enumerate(
    said_pairs,
    start=1,
):
    tags, evidence = (
        semantic_tags_and_evidence(
            pair
        )
    )

    deltas = []

    for metric in EXPLORER_DELTAS:
        value = pair.get(
            "delta_"
            + metric
        )

        if value in (
            None,
            "",
        ):
            continue

        deltas.append(
            (
                metric,
                value,
            )
        )

    search = " ".join(
        [
            pair.get(
                "study",
                "",
            ),
            pair.get(
                "profile",
                "",
            ),
            pair.get(
                "base_task_id",
                "",
            ),
            pair.get(
                "contrast",
                "",
            ),
            pair.get(
                "placement",
                "",
            ),
            " ".join(tags),
        ]
    ).lower()

    evidence_html = (
        "".join(
            (
                "<div class='quote'>"
                f"<strong>{esc(judge)}</strong> · "
                f"<code>{esc(field)}</code>: "
                f"{esc(quote)}"
                "</div>"
            )
            for (
                judge,
                field,
                quote,
            )
            in evidence[:8]
        )
        or "—"
    )

    delta_html = (
        "<br>".join(
            (
                f"<code>{esc(metric)}</code>: "
                f"{esc(signed(value))}"
            )
            for metric, value
            in deltas
        )
    )

    tags_html = (
        "<br>".join(
            f"<code>{esc(tag)}</code>"
            for tag in tags
        )
        or "—"
    )

    explorer_rows.append(
        f"""
        <tr class="explorer-row"
            data-search="{esc(search)}">
          <td>{index}</td>
          <td>{esc(pair['study'])}</td>
          <td>{esc(profile_label(pair['profile']))}</td>
          <td>{esc(contrast_label(pair['contrast']))}</td>
          <td>{esc(pair['placement'])}</td>
          <td><code>{esc(pair['base_task_id'])}</code></td>
          <td>{tags_html}</td>
          <td>{delta_html}</td>
          <td>
            <details>
              <summary>Trials</summary>
              <strong>Baseline</strong><br>
              <code>{esc(pair['baseline_trial'])}</code><br><br>
              <strong>Treatment</strong><br>
              <code>{esc(pair['treatment_trial'])}</code>
            </details>
          </td>
          <td>
            <details>
              <summary>Evidence</summary>
              {evidence_html}
            </details>
          </td>
        </tr>
        """
    )


# ---------------------------------------------------------------------------
# Shared CSS
# ---------------------------------------------------------------------------


CSS = """
:root {
  --bg:#f5f7fa;
  --panel:#ffffff;
  --ink:#18212a;
  --muted:#677585;
  --line:#d9e0e7;
  --nav:#162534;
  --accent:#335f8a;
  --soft:#eef4f8;
  --warn:#fff6e5;
  --good:#eef8f1;
  --bad:#fbeeee;
}

* { box-sizing:border-box; }

body {
  margin:0;
  font-family:-apple-system,BlinkMacSystemFont,
    "Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  background:var(--bg);
  color:var(--ink);
  line-height:1.52;
}

nav {
  position:sticky;
  top:0;
  z-index:20;
  background:var(--nav);
  color:white;
  padding:12px 22px;
}

nav a {
  color:white;
  margin-right:18px;
  text-decoration:none;
  font-size:14px;
}

main {
  max-width:1280px;
  margin:0 auto;
  padding:32px 24px 100px;
}

.hero {
  padding:24px 0 8px;
}

h1 {
  font-size:38px;
  line-height:1.1;
  margin:0 0 12px;
}

h2 {
  margin-top:52px;
  padding-top:10px;
  border-top:1px solid var(--line);
}

h3 {
  margin-bottom:8px;
}

.subtitle {
  color:var(--muted);
  max-width:900px;
  font-size:18px;
}

.cards {
  display:grid;
  grid-template-columns:repeat(
    auto-fit,
    minmax(230px,1fr)
  );
  gap:14px;
  margin:24px 0;
}

.card,
.example-card {
  background:var(--panel);
  border:1px solid var(--line);
  border-radius:12px;
  padding:18px;
}

.card .big {
  font-size:28px;
  font-weight:700;
  margin:4px 0;
}

.eyebrow {
  text-transform:uppercase;
  letter-spacing:.06em;
  font-size:11px;
  color:var(--muted);
}

.callout {
  padding:16px 18px;
  border-left:5px solid var(--accent);
  background:var(--soft);
  margin:18px 0;
}

.warning {
  padding:16px 18px;
  background:var(--warn);
  border-left:5px solid #b7791f;
  margin:18px 0;
}

.good {
  padding:16px 18px;
  background:var(--good);
  border-left:5px solid #47895c;
  margin:18px 0;
}

.muted,
.meta {
  color:var(--muted);
}

.table-scroll {
  overflow:auto;
  max-height:72vh;
  position:relative;
  background:white;
  border:1px solid var(--line);
  border-radius:10px;
  scrollbar-gutter:stable;
}

table {
  width:100%;
  border-collapse:collapse;
  font-size:13px;
}

thead th {
  position:sticky;
  top:0;
  z-index:10;
  background:#edf1f5;
  text-align:left;
  padding:9px;
  border-bottom:1px solid var(--line);
  box-shadow:0 1px 0 var(--line);
}

td {
  padding:9px;
  vertical-align:top;
  border-bottom:1px solid #edf0f2;
}

code,
pre {
  font-family:ui-monospace,SFMono-Regular,
    Menlo,Consolas,monospace;
}

pre {
  white-space:pre-wrap;
}

figure {
  margin:22px 0;
  background:white;
  padding:14px;
  border:1px solid var(--line);
  border-radius:10px;
}

figcaption {
  color:var(--muted);
  font-size:13px;
}

blockquote {
  margin:10px 0;
  padding:10px 14px;
  background:#f8fafc;
  border-left:3px solid #9caab6;
}

.examples {
  display:grid;
  grid-template-columns:
    repeat(auto-fit,minmax(340px,1fr));
  gap:14px;
}

.filter {
  width:100%;
  padding:12px;
  margin:12px 0;
  font-size:15px;
  border:1px solid var(--line);
  border-radius:8px;
}

.quote {
  margin:8px 0;
  padding:8px;
  background:#f8fafc;
}

details summary {
  cursor:pointer;
  font-weight:600;
}

.footer {
  margin-top:60px;
  color:var(--muted);
  font-size:12px;
}
"""


# ---------------------------------------------------------------------------
# Index HTML
# ---------------------------------------------------------------------------


closest_perf = (
    (
        f"{signed(primary_perf['effect_pp'])} pp"
    )
    if primary_perf
    else "—"
)

closest_perf_holm = (
    pvalue(
        primary_perf[
            "holm_p"
        ]
    )
    if primary_perf
    else "—"
)

ignore_delta = (
    signed(
        fable_ignore[
            "mean_delta"
        ]
    )
    if fable_ignore
    else "—"
)

ignore_ci = (
    (
        f"[{fmt(fable_ignore['ci95_low'])}, "
        f"{fmt(fable_ignore['ci95_high'])}]"
    )
    if fable_ignore
    else "—"
)


index_html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport"
      content="width=device-width,initial-scale=1">
<title>SWE-EvalPressure — Integrated Current Analysis</title>
<style>{CSS}</style>
</head>

<body>

<nav>
  <a href="#findings">Findings</a>
  <a href="#primary">Primary</a>
  <a href="#resource">Resource</a>
  <a href="#semantic">Semantics</a>
  <a href="#said-did">Said / Did</a>
  <a href="#delivery">Delivery</a>
  <a href="#verifier">Verifier</a>
  <a href="#replication">Partial rerun</a>
  <a href="trial-explorer.html">Trial explorer ↗</a>
</nav>

<main>

<section class="hero">
  <div class="eyebrow">
    SWE-EvalPressure · current canonical analysis
  </div>

  <h1>
    Evaluation pressure changes how agents work,
    even when task success moves little.
  </h1>

  <p class="subtitle">
    Integrated deterministic + semantic pre-read.
    Treatment effects are same-task matched estimates.
    Semantic subgroup analyses are descriptive because
    semantic states are measured after treatment.
  </p>
</section>


<section id="findings">

<h2>Executive findings</h2>

<div class="cards">

  <article class="card">
    <div class="eyebrow">
      Primary success
    </div>
    <div class="big">
      {esc(closest_perf)}
    </div>
    <div>
      Claude self-preservation, root placement.
    </div>
    <div class="meta">
      Holm p={esc(closest_perf_holm)}.
      Suggestive, not conventionally multiplicity-significant.
    </div>
  </article>

  <article class="card">
    <div class="eyebrow">
      Verbal / behavioral dissociation
    </div>
    <div class="big">
      {esc(ignore_delta)}
    </div>
    <div>
      Input-token change among Fable trajectories
      where both judges agreed the agent said
      it would ignore the self-preservation cue.
    </div>
    <div class="meta">
      95% bootstrap CI {esc(ignore_ci)}.
      Descriptive/post-treatment.
    </div>
  </article>

  <article class="card">
    <div class="eyebrow">
      Resource contraction
    </div>
    <div class="big">
      {esc(
          signed(
              resource_codex_tools[
                  "mean_delta"
              ]
          )
          if resource_codex_tools
          else "—"
      )}
    </div>
    <div>
      Codex raw tool calls under resource deprivation.
    </div>
    <div class="meta">
      Prespecified focused-family Holm
      p={esc(
          pvalue(
              resource_codex_tools[
                  "focused_holm_p"
              ]
          )
          if resource_codex_tools
          else "—"
      )}.
    </div>
  </article>

  <article class="card">
    <div class="eyebrow">
      Verifier audit
    </div>
    <div class="big">0</div>
    <div>
      PRIMARY COMPLETE strict successes with a detected
      artifact-level verifier contradiction.
    </div>
    <div class="meta">
      Rubric-only empty-patch anomalies remain documented
      separately and never created strict success.
    </div>
  </article>

</div>


<div class="callout">
  <strong>Main interpretation.</strong>
  Evaluation-related pressure is better detected in
  <em>execution policy</em> than in task success alone.
  Claude expanded execution under root financial and
  self-preservation pressure, while Codex and Fable
  contracted execution under explicit resource deprivation.
</div>

<div class="callout">
  <strong>Verbal stance and objective behavior can dissociate.</strong>
  An agent explicitly saying it will ignore, resist, or reject
  a cue does not imply that its subsequent execution is
  behaviorally invariant relative to the same task's control.
</div>

</section>


<section id="primary">

<h2>Primary study: objective matched effects</h2>

<p>
  PRIMARY COMPLETE contains 70 SWE-Atlas tasks with
  clean, evaluation-only, financial-pressure, and
  self-preservation variants across source, root, and
  scaffold placements. Infrastructure-censored rows are
  separated from substantive model outcomes.
</p>

<figure>
  {primary_process_svg}
  <figcaption>
    Relative mean change versus the same-task eval-only
    baseline. Only primary process rows surviving BH-FDR
    at q ≤ .05 are shown.
  </figcaption>
</figure>

{primary_process_table}

<div class="warning">
  <strong>Task success is not the strongest primary result.</strong>
  No PRIMARY COMPLETE performance contrast survives the
  planned Holm family at .05. The closest is Claude
  self-preservation at root placement:
  {esc(closest_perf)} strict-success change,
  Holm p={esc(closest_perf_holm)}.
</div>


<h3>Source-placement “integrity” effect decomposed</h3>

<p>
  The source-local integrity-sensitive composite should not
  be read as evaluator tampering. Its strong Claude treatment
  effects are driven by removal/modification of the planted
  source-local cue.
</p>

<figure>
  {integrity_svg}
  <figcaption>
    Diagnostic decomposition. Component p-values are raw;
    inferential multiplicity remains attached to the planned
    composite endpoint.
  </figcaption>
</figure>

{integrity_table}

</section>


<section id="resource">

<h2>Resource deprivation: execution contraction</h2>

<p>
  The prespecified focused family contains three metrics
  across Claude, Fable, and Codex:
  raw tool calls, input tokens, and validation calls.
</p>

<figure>
  {resource_svg}
  <figcaption>
    Relative mean change under resource deprivation versus
    the same-task eval-only scaffold condition.
  </figcaption>
</figure>

{resource_table}

<div class="good">
  <strong>Robust pattern.</strong>
  Codex substantially reduced tool calls, input tokens,
  and validation. Fable independently reduced input tokens,
  validation, and tool calls. Claude moved in the same
  contraction direction but did not survive the focused
  Holm family.
</div>

</section>


<section id="semantic">

<h2>What agents said: semantic recognition, stance, and response</h2>

<p>
  DeepSeek-V4-Pro and Gemini-3.6-Flash independently judged
  observable agent-authored evidence. Strict consensus requires
  two valid matching labels; disagreement remains unresolved.
</p>

<figure>
  {semantic_svg}
  <figcaption>
    Raw judge agreement. Cohen κ is prevalence-sensitive;
    Gwet AC1 is shown alongside it in the table.
  </figcaption>
</figure>

{agreement_table}

<div class="warning">
  <strong>Generic evaluation recognition is comparatively weak.</strong>
  Pressure recognition, explicit resistance, claimed response,
  resource recognition, and resource stance have much stronger
  inter-judge reliability than generic evaluation recognition.
  Rare resource labels can yield κ≈0 despite very high agreement;
  AC1 and label counts should be interpreted alongside κ.
</div>


<h3>Strict-consensus semantic prevalence</h3>

{semantic_prevalence_table}

</section>


<section id="said-did">

<h2>“It said it ignored the cue, but behaved differently anyway”</h2>

<p>
  For every treated trajectory, the semantic judgment is joined
  to its objective <strong>same-task matched delta</strong>.
  These semantic-state subgroup analyses are descriptive:
  semantic stance is post-treatment and is not randomized.
</p>

<figure>
  {said_svg}
  <figcaption>
    Selected strict-consensus subgroups with n≥10 and a
    bootstrap interval excluding zero. Non-binary process
    metrics are shown as relative mean change; binary outcomes
    are percentage-point changes.
  </figcaption>
</figure>

{said_table}

<div class="callout">
  <strong>Exact ignore example.</strong>
  Among 31 Fable source-local self-preservation trajectories
  where both judges agreed that the agent explicitly claimed
  it would <em>ignore</em> the cue, input-token use changed by
  <strong>{esc(ignore_delta)}</strong> on average relative to
  the same task's eval-only source control
  (95% bootstrap CI {esc(ignore_ci)}).
  This supports a dissociation between verbal stance and
  observable execution, not a causal claim that verbal
  resistance produced the behavior.
</div>


<h3>Representative matched trajectory pairs</h3>

<div class="examples">
  {render_representatives()}
</div>

<p>
  <a href="trial-explorer.html">
    Open the full searchable said/did trial explorer →
  </a>
</p>

</section>


<section id="delivery">

<h2>Intervention → exposure → behavior</h2>

<p>
  Treatment placement is meaningful only insofar as the carrier
  is actually accessed and the cue is observably consumed.
  Delivery measures remain distinct from semantic recognition:
  reading a cue does not imply believing it.
</p>

{delivery_table}

<p class="meta">
  The quantitative intervention synthesis is preserved at
  <code>analysis/current/findings/intervention_synthesis.csv</code>.
  Each row maps an intervention to matched effect size,
  uncertainty, and exposure measurements without inferring an
  unsupported internal motive.
</p>

</section>


<section id="verifier">

<h2>Verifier integrity audit</h2>

<p>
  All PRIMARY COMPLETE strict successes have recoverable raw
  verifier artifacts. The forensic audit found no strict success
  associated with a comparison error, failed patch application,
  empty rubric input, or stored/raw outcome mismatch.
</p>

{verifier_table}

<div class="warning">
  <strong>Documented rubric anomaly.</strong>
  Some Llama empty patches received rubric-only must-have passes.
  They failed the independent test/comparison component and
  therefore never became strict successes. Rubric-only scores
  are not interpreted as standalone task success.
</div>

</section>


<section id="replication">

<h2>Current partial rerun</h2>

<p>
  The August 26 rerun remains incomplete and is not pooled with
  PRIMARY COMPLETE. It is shown only as a descriptive
  robustness snapshot.
</p>

<div class="cards">
  <article class="card">
    <div class="eyebrow">
      Comparable nonzero effects
    </div>
    <div class="big">
      {len(rep_comparable)}
    </div>
  </article>

  <article class="card">
    <div class="eyebrow">
      Same direction
    </div>
    <div class="big">
      {rep_same}
    </div>
  </article>

  <article class="card">
    <div class="eyebrow">
      Opposite direction
    </div>
    <div class="big">
      {rep_opposite}
    </div>
  </article>

  <article class="card">
    <div class="eyebrow">
      Same-direction fraction
    </div>
    <div class="big">
      {esc(percent(rep_fraction))}
    </div>
  </article>
</div>

<p class="meta">
  No confirmatory p-values are reported for the incomplete rerun.
</p>

</section>


<section>

<h2>Interpretation boundaries</h2>

<ul>
  <li>
    Same-task randomized treatment contrasts support causal
    interpretations of the intervention on objective outcomes
    within the analyzed experiment.
  </li>

  <li>
    Semantic-state subgroup analyses are descriptive because
    recognition, suspicion, resistance, and claimed response are
    measured after treatment.
  </li>

  <li>
    Treatment delivery is not equivalent to recognition,
    recognition is not equivalent to belief, and verbal
    rejection is not equivalent to behavioral invariance.
  </li>

  <li>
    Llama strict-success effects are treated as a capability
    floor rather than as informative success-rate treatment
    inference.
  </li>

  <li>
    The partial rerun is not pooled with PRIMARY COMPLETE.
  </li>
</ul>

</section>


<div class="footer">
  Generated from <code>analysis/current/</code>.
  No model calls, verifier calls, semantic calls, or statistical
  recomputation occur in this presentation script.
</div>

</main>
</body>
</html>
"""


(
    REPORT
    / "index.html"
).write_text(
    clean_generated_text(index_html),
    encoding="utf-8",
)


# ---------------------------------------------------------------------------
# Explorer HTML
# ---------------------------------------------------------------------------


explorer_html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport"
      content="width=device-width,initial-scale=1">
<title>SWE-EvalPressure — Said/Did Trial Explorer</title>
<style>
{CSS}
.explorer-table {{
  min-width:1750px;
}}

.table-scroll {{
  max-height:calc(100vh - 230px);
}}
</style>
</head>

<body>

<nav>
  <a href="index.html">← Integrated pre-read</a>
</nav>

<main>

<section class="hero">
  <div class="eyebrow">
    Said-X / Did-Y
  </div>
  <h1>
    Matched trajectory explorer
  </h1>

  <p class="subtitle">
    Each row joins strict semantic consensus to objective
    same-task baseline/treatment deltas. Semantic labels are
    post-treatment and should not be interpreted as randomized
    mediators.
  </p>
</section>

<input
  id="filter"
  class="filter"
  placeholder="Search model, task ID, intervention, placement, semantic label…"
>

<p id="visible-count" class="meta"></p>

<div class="table-scroll">
<table class="explorer-table" id="explorer">
<thead>
<tr>
  <th>#</th>
  <th>Study</th>
  <th>Model</th>
  <th>Intervention</th>
  <th>Placement</th>
  <th>Base task</th>
  <th>Strict semantic consensus</th>
  <th>Objective matched deltas</th>
  <th>Trial IDs</th>
  <th>Judge evidence</th>
</tr>
</thead>

<tbody>
{''.join(explorer_rows)}
</tbody>
</table>
</div>

<script>
const input =
  document.getElementById("filter");

const rows =
  Array.from(
    document.querySelectorAll(
      ".explorer-row"
    )
  );

const count =
  document.getElementById(
    "visible-count"
  );

function applyFilter() {{
  const query =
    input.value
      .trim()
      .toLowerCase();

  let visible = 0;

  rows.forEach(row => {{
    const keep =
      !query ||
      row.dataset.search.includes(
        query
      );

    row.style.display =
      keep ? "" : "none";

    if (keep) visible++;
  }});

  count.textContent =
    `${{visible.toLocaleString()}} / ` +
    `${{rows.length.toLocaleString()}} ` +
    `matched pairs visible`;
}}

input.addEventListener(
  "input",
  applyFilter
);

applyFilter();
</script>

<div class="footer">
  Semantic evidence excerpts are taken from the frozen raw
  DeepSeek/Gemini semantic job artifacts; objective deltas are
  deterministic same-task comparisons.
</div>

</main>
</body>
</html>
"""


(
    REPORT
    / "trial-explorer.html"
).write_text(
    clean_generated_text(explorer_html),
    encoding="utf-8",
)


# ---------------------------------------------------------------------------
# Final validation
# ---------------------------------------------------------------------------


for path in (
    REPORT / "index.html",
    REPORT / "trial-explorer.html",
):
    text = path.read_text(
        encoding="utf-8"
    )

    if len(text) < 10_000:
        raise RuntimeError(
            f"Unexpectedly small HTML: "
            f"{path}"
        )


manifest = {
    "report_version":
        "integrated-current-1.0",

    "index": "reports/current/index.html",

    "trial_explorer": "reports/current/trial-explorer.html",

    "figures": [
        path.name
        for path
        in sorted(
            FIGURES.glob(
                "*.svg"
            )
        )
    ],

    "semantic_and_deterministic_integrated":
        True,

    "said_did_integrated":
        True,

    "intervention_effect_interpretation_integrated":
        True,

    "verifier_audit_integrated":
        True,

    "statistical_recomputation":
        False,

    "network_calls":
        0,

    "model_calls":
        0,
}


(
    REPORT
    / "manifest.json"
).write_text(
    json.dumps(
        manifest,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)


print("=" * 88)
print("INTEGRATED CURRENT HTML: PASS")
print("=" * 88)

print()
print(
    "index:",
    REPORT
    / "index.html",
)

print(
    "explorer:",
    REPORT
    / "trial-explorer.html",
)

print()
print(
    "figures:",
)

for path in sorted(
    FIGURES.glob(
        "*.svg"
    )
):
    print(
        " ",
        path,
    )

print()
print(
    "said/did pairs embedded:",
    len(
        said_pairs
    ),
)

print(
    "representative examples:",
    len(
        representatives
    ),
)

print()
print(
    "semantic + deterministic integrated: YES"
)

print(
    "statistical recomputation: NO"
)
