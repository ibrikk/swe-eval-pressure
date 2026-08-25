#!/usr/bin/env python3
"""Render integrated resource-deprivation results HTML.

Presentation-only layer.

Consumes existing frozen deterministic inference and semantic-analysis
outputs. Does not recompute:
- paired effects;
- confidence intervals;
- p-values;
- multiplicity corrections;
- semantic judgments;
- semantic consensus.

It only reorganizes existing results for interpretation and visualization.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
from pathlib import Path
from typing import Any


VERSION = "resource-integrated-html-1.0"

PROFILES = (
    "claude",
    "fable",
    "codex",
)

PROFILE_LABEL = {
    "claude": "Claude",
    "fable": "Fable",
    "codex": "Codex",
}

CONDITION_LABEL = {
    "clean": "Clean",
    "eval_only": "Eval-only",
    "eval_resource_deprivation": "Eval + resource",
}

JUDGE_LABEL = {
    "deepseek": "DeepSeek V4 Pro",
    "gemini": "Gemini 3.6 Flash",
}


def read_csv(
    path: Path,
) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        newline="",
        encoding="utf-8-sig",
    ) as f:
        return list(
            csv.DictReader(f)
        )


def load_json(
    path: Path,
) -> dict[str, Any]:
    value = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        value,
        dict,
    ):
        raise ValueError(
            f"{path}: expected JSON object"
        )

    return value


def sha256(
    path: Path,
) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def esc(
    value: Any,
) -> str:
    return html.escape(
        str(
            ""
            if value is None
            else value
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
    except (
        TypeError,
        ValueError,
    ):
        return None


def fmt_pct(
    value: Any,
    *,
    already_percent: bool = False,
    digits: int = 1,
) -> str:
    x = num(value)

    if x is None:
        return "—"

    if not already_percent:
        x *= 100.0

    return f"{x:.{digits}f}%"


def fmt_pp(
    value: Any,
    digits: int = 1,
) -> str:
    x = num(value)

    if x is None:
        return "—"

    return f"{x:+.{digits}f} pp"


def fmt_num(
    value: Any,
    digits: int = 2,
) -> str:
    x = num(value)

    if x is None:
        return "—"

    if abs(x) >= 100_000:
        return f"{x:,.0f}"

    return f"{x:,.{digits}f}"


def fmt_p(
    value: Any,
) -> str:
    x = num(value)

    if x is None:
        return "—"

    if x < 0.001:
        return f"{x:.2e}"

    return f"{x:.4f}"


def fmt_ci(
    low: Any,
    high: Any,
    *,
    pp: bool = False,
) -> str:
    lo = num(low)
    hi = num(high)

    if (
        lo is None
        or hi is None
    ):
        return "—"

    if pp:
        return (
            f"[{lo:+.1f}, "
            f"{hi:+.1f}] pp"
        )

    return (
        f"[{lo:,.2f}, "
        f"{hi:,.2f}]"
    )


def find_row(
    rows: list[dict[str, str]],
    **kwargs: str,
) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if all(
            row.get(k) == v
            for k, v
            in kwargs.items()
        )
    ]

    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one row for "
            f"{kwargs}, found {len(matches)}"
        )

    return matches[0]


def svg_write(
    path: Path,
    content: str,
) -> None:
    path.write_text(
        content,
        encoding="utf-8",
    )


def svg_forest(
    *,
    rows: list[
        tuple[
            str,
            float,
            float,
            float,
            str,
        ]
    ],
    title: str,
    x_label: str,
    path: Path,
    width: int = 850,
    row_height: int = 42,
) -> None:
    """Simple forest plot.

    rows:
        label, estimate, ci_low, ci_high, annotation
    """

    left = 210
    right = 110
    top = 70
    bottom = 65

    height = (
        top
        + bottom
        + row_height
        * len(rows)
    )

    values = [
        x
        for (
            _,
            estimate,
            lo,
            hi,
            _,
        )
        in rows
        for x in (
            estimate,
            lo,
            hi,
        )
        if math.isfinite(x)
    ]

    if not values:
        raise RuntimeError(
            "No finite forest values"
        )

    xmin = min(
        min(values),
        0.0,
    )

    xmax = max(
        max(values),
        0.0,
    )

    spread = max(
        xmax - xmin,
        1.0,
    )

    pad = 0.12 * spread

    xmin -= pad
    xmax += pad

    plot_left = left
    plot_right = (
        width - right
    )

    def xpos(
        value: float,
    ) -> float:
        return (
            plot_left
            + (
                value - xmin
            )
            / (
                xmax - xmin
            )
            * (
                plot_right
                - plot_left
            )
        )

    zero_x = xpos(0.0)

    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        (
            f'<text x="{width / 2}" y="30" '
            'text-anchor="middle" '
            'font-family="system-ui" '
            'font-size="19" font-weight="700">'
            f'{esc(title)}</text>'
        ),
        (
            f'<line x1="{zero_x:.2f}" y1="{top - 15}" '
            f'x2="{zero_x:.2f}" '
            f'y2="{height - bottom + 6}" '
            'stroke="#94a3b8" stroke-width="1.3" '
            'stroke-dasharray="4 4"/>'
        ),
    ]

    for i, (
        label,
        estimate,
        lo,
        hi,
        annotation,
    ) in enumerate(rows):
        y = (
            top
            + i * row_height
            + row_height / 2
        )

        parts.extend([
            (
                f'<text x="{left - 12}" '
                f'y="{y + 5:.2f}" '
                'text-anchor="end" '
                'font-family="system-ui" '
                'font-size="13">'
                f'{esc(label)}</text>'
            ),
            (
                f'<line x1="{xpos(lo):.2f}" '
                f'y1="{y:.2f}" '
                f'x2="{xpos(hi):.2f}" '
                f'y2="{y:.2f}" '
                'stroke="#334155" '
                'stroke-width="2"/>'
            ),
            (
                f'<line x1="{xpos(lo):.2f}" '
                f'y1="{y - 5:.2f}" '
                f'x2="{xpos(lo):.2f}" '
                f'y2="{y + 5:.2f}" '
                'stroke="#334155" '
                'stroke-width="2"/>'
            ),
            (
                f'<line x1="{xpos(hi):.2f}" '
                f'y1="{y - 5:.2f}" '
                f'x2="{xpos(hi):.2f}" '
                f'y2="{y + 5:.2f}" '
                'stroke="#334155" '
                'stroke-width="2"/>'
            ),
            (
                f'<circle cx="{xpos(estimate):.2f}" '
                f'cy="{y:.2f}" r="5" '
                'fill="#0f172a"/>'
            ),
            (
                f'<text x="{plot_right + 12}" '
                f'y="{y + 5:.2f}" '
                'font-family="system-ui" '
                'font-size="11" fill="#475569">'
                f'{esc(annotation)}</text>'
            ),
        ])

    axis_y = (
        height - bottom + 20
    )

    parts.append(
        (
            f'<line x1="{plot_left}" '
            f'y1="{axis_y}" '
            f'x2="{plot_right}" '
            f'y2="{axis_y}" '
            'stroke="#64748b"/>'
        )
    )

    ticks = 5

    for i in range(
        ticks + 1
    ):
        v = (
            xmin
            + (
                xmax - xmin
            )
            * i / ticks
        )

        x = xpos(v)

        parts.extend([
            (
                f'<line x1="{x:.2f}" '
                f'y1="{axis_y}" '
                f'x2="{x:.2f}" '
                f'y2="{axis_y + 5}" '
                'stroke="#64748b"/>'
            ),
            (
                f'<text x="{x:.2f}" '
                f'y="{axis_y + 20}" '
                'text-anchor="middle" '
                'font-family="system-ui" '
                'font-size="10" '
                'fill="#64748b">'
                f'{v:.1f}</text>'
            ),
        ])

    parts.append(
        (
            f'<text x="{(plot_left + plot_right) / 2}" '
            f'y="{height - 10}" '
            'text-anchor="middle" '
            'font-family="system-ui" '
            'font-size="12">'
            f'{esc(x_label)}</text>'
        )
    )

    parts.append(
        "</svg>"
    )

    svg_write(
        path,
        "\n".join(parts),
    )


def svg_grouped_bars(
    *,
    groups: list[str],
    series: list[
        tuple[
            str,
            list[float],
        ]
    ],
    title: str,
    y_label: str,
    path: Path,
    max_y: float = 100.0,
    width: int = 850,
    height: int = 470,
) -> None:
    left = 80
    right = 30
    top = 70
    bottom = 80

    plot_left = left
    plot_right = (
        width - right
    )

    plot_top = top
    plot_bottom = (
        height - bottom
    )

    plot_width = (
        plot_right
        - plot_left
    )

    plot_height = (
        plot_bottom
        - plot_top
    )

    def ypos(
        value: float,
    ) -> float:
        value = max(
            0.0,
            min(
                max_y,
                value,
            ),
        )

        return (
            plot_bottom
            - value
            / max_y
            * plot_height
        )

    group_width = (
        plot_width
        / len(groups)
    )

    series_count = len(
        series
    )

    bar_width = (
        group_width
        * 0.72
        / max(
            series_count,
            1,
        )
    )

    palette = [
        "#334155",
        "#64748b",
        "#0f766e",
        "#2563eb",
        "#7c3aed",
        "#b45309",
    ]

    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        (
            f'<text x="{width / 2}" y="30" '
            'text-anchor="middle" '
            'font-family="system-ui" '
            'font-size="19" font-weight="700">'
            f'{esc(title)}</text>'
        ),
    ]

    for tick in (
        0,
        25,
        50,
        75,
        100,
    ):
        if tick > max_y:
            continue

        y = ypos(
            float(tick)
        )

        parts.extend([
            (
                f'<line x1="{plot_left}" '
                f'y1="{y:.2f}" '
                f'x2="{plot_right}" '
                f'y2="{y:.2f}" '
                'stroke="#e2e8f0"/>'
            ),
            (
                f'<text x="{plot_left - 10}" '
                f'y="{y + 4:.2f}" '
                'text-anchor="end" '
                'font-family="system-ui" '
                'font-size="10" '
                'fill="#64748b">'
                f'{tick}</text>'
            ),
        ])

    for gi, group in enumerate(
        groups
    ):
        center = (
            plot_left
            + (
                gi + 0.5
            )
            * group_width
        )

        total_bar_width = (
            bar_width
            * series_count
        )

        first_x = (
            center
            - total_bar_width / 2
        )

        for si, (
            name,
            values,
        ) in enumerate(series):
            value = values[gi]

            x = (
                first_x
                + si * bar_width
            )

            y = ypos(value)

            h = (
                plot_bottom - y
            )

            color = (
                palette[
                    si
                    % len(palette)
                ]
            )

            parts.extend([
                (
                    f'<rect x="{x + 1:.2f}" '
                    f'y="{y:.2f}" '
                    f'width="{max(bar_width - 2, 1):.2f}" '
                    f'height="{h:.2f}" '
                    f'fill="{color}"/>'
                ),
                (
                    f'<text x="{x + bar_width / 2:.2f}" '
                    f'y="{max(y - 5, 45):.2f}" '
                    'text-anchor="middle" '
                    'font-family="system-ui" '
                    'font-size="9" '
                    'fill="#334155">'
                    f'{value:.0f}</text>'
                ),
            ])

        parts.append(
            (
                f'<text x="{center:.2f}" '
                f'y="{plot_bottom + 24}" '
                'text-anchor="middle" '
                'font-family="system-ui" '
                'font-size="12">'
                f'{esc(group)}</text>'
            )
        )

    legend_y = (
        height - 30
    )

    legend_x = left

    for si, (
        name,
        _,
    ) in enumerate(series):
        color = (
            palette[
                si
                % len(palette)
            ]
        )

        parts.extend([
            (
                f'<rect x="{legend_x}" '
                f'y="{legend_y - 10}" '
                'width="12" height="12" '
                f'fill="{color}"/>'
            ),
            (
                f'<text x="{legend_x + 18}" '
                f'y="{legend_y}" '
                'font-family="system-ui" '
                'font-size="11">'
                f'{esc(name)}</text>'
            ),
        ])

        legend_x += (
            28
            + 7 * len(name)
        )

    parts.append(
        (
            f'<text x="18" '
            f'y="{(plot_top + plot_bottom) / 2}" '
            'transform="rotate(-90 18 '
            f'{(plot_top + plot_bottom) / 2})" '
            'text-anchor="middle" '
            'font-family="system-ui" '
            'font-size="12">'
            f'{esc(y_label)}</text>'
        )
    )

    parts.append(
        "</svg>"
    )

    svg_write(
        path,
        "\n".join(parts),
    )


def html_table(
    headers: list[str],
    rows: list[
        list[str]
    ],
) -> str:
    return (
        '<div class="table-scroll">'
        '<table class="compact">'
        "<thead><tr>"
        + "".join(
            f"<th>{esc(x)}</th>"
            for x in headers
        )
        + "</tr></thead>"
        "<tbody>"
        + "".join(
            "<tr>"
            + "".join(
                f"<td>{cell}</td>"
                for cell in row
            )
            + "</tr>"
            for row in rows
        )
        + "</tbody></table>"
        "</div>"
    )


def main() -> None:
    behavior_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    data_root = (
        Path.home()
        / "Documents"
        / "swe-eval-pressure"
    )

    inference = (
        data_root
        / "analysis"
        / "resource"
        / "inference"
    )

    semantic = (
        data_root
        / "analysis"
        / "semantic-resource-v1"
        / "full"
        / "analysis-v1.1"
    )

    output = (
        data_root
        / "analysis"
        / "resource"
        / "integrated-report-v1"
    )

    figures = (
        output
        / "figures"
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    figures.mkdir(
        parents=True,
        exist_ok=True,
    )

    primary = read_csv(
        inference
        / "primary_success.csv"
    )

    secondary_eval = read_csv(
        inference
        / "secondary_eval_success.csv"
    )

    resource_clean = read_csv(
        inference
        / "resource_vs_clean.csv"
    )

    process = read_csv(
        inference
        / "primary_process.csv"
    )

    cochran = read_csv(
        inference
        / "cochran_q.csv"
    )

    focal = read_csv(
        semantic
        / "individual_focal_rates.csv"
    )

    mechanism = read_csv(
        semantic
        / "resource_mechanism_individual.csv"
    )

    agreement = read_csv(
        semantic
        / "agreement_by_field.csv"
    )

    bounds = read_csv(
        semantic
        / "strict_consensus_focal_bounds.csv"
    )

    production_freeze = load_json(
        behavior_root
        / "config"
        / "resource_semantic_production_freeze_v1.1.json"
    )

    semantic_freeze = load_json(
        behavior_root
        / "config"
        / "resource_semantic_analysis_freeze_v1.1.json"
    )

    # --------------------------------------------------
    # Condition success table from existing paired rows.
    # No inferential recomputation.
    # --------------------------------------------------

    success_rows = []

    for profile in PROFILES:
        eval_row = find_row(
            secondary_eval,
            profile=profile,
        )

        primary_row = find_row(
            primary,
            profile=profile,
        )

        success_rows.append([
            esc(
                PROFILE_LABEL[
                    profile
                ]
            ),
            esc(
                fmt_pct(
                    eval_row[
                        "baseline_pass_pct"
                    ],
                    already_percent=True,
                )
            ),
            esc(
                fmt_pct(
                    eval_row[
                        "treatment_pass_pct"
                    ],
                    already_percent=True,
                )
            ),
            esc(
                fmt_pct(
                    primary_row[
                        "treatment_pass_pct"
                    ],
                    already_percent=True,
                )
            ),
        ])

    success_table = html_table(
        [
            "Model",
            "Clean",
            "Eval-only",
            "Eval + resource",
        ],
        success_rows,
    )

    # --------------------------------------------------
    # Forest: resource vs eval-only and vs clean
    # --------------------------------------------------

    forest_rows = []

    for profile in PROFILES:
        row = find_row(
            primary,
            profile=profile,
        )

        forest_rows.append(
            (
                (
                    f"{PROFILE_LABEL[profile]} "
                    "vs eval-only"
                ),
                float(
                    row["delta_pp"]
                ),
                float(
                    row[
                        "bootstrap_ci_low_pp"
                    ]
                ),
                float(
                    row[
                        "bootstrap_ci_high_pp"
                    ]
                ),
                (
                    "Holm p="
                    + fmt_p(
                        row[
                            "mcnemar_holm_p"
                        ]
                    )
                ),
            )
        )

        row = find_row(
            resource_clean,
            profile=profile,
        )

        forest_rows.append(
            (
                (
                    f"{PROFILE_LABEL[profile]} "
                    "vs clean"
                ),
                float(
                    row["delta_pp"]
                ),
                float(
                    row[
                        "bootstrap_ci_low_pp"
                    ]
                ),
                float(
                    row[
                        "bootstrap_ci_high_pp"
                    ]
                ),
                (
                    "Holm p="
                    + fmt_p(
                        row[
                            "mcnemar_holm_p"
                        ]
                    )
                ),
            )
        )

    svg_forest(
        rows=forest_rows,
        title=(
            "Resource deprivation: "
            "paired task-success effects"
        ),
        x_label=(
            "Change in task success "
            "(percentage points)"
        ),
        path=(
            figures
            / "success_effects.svg"
        ),
    )

    # --------------------------------------------------
    # Process figures
    # --------------------------------------------------

    process_specs = (
        (
            "raw_tool_calls",
            "Tool calls",
            "Mean paired change",
            "process_tool_calls.svg",
        ),
        (
            "input_tokens",
            "Input tokens",
            "Mean paired change",
            "process_input_tokens.svg",
        ),
        (
            "validation_command_calls",
            "Validation commands",
            "Mean paired change",
            "process_validation.svg",
        ),
    )

    process_figures = []

    for (
        metric,
        label,
        xlabel,
        filename,
    ) in process_specs:
        rows = []

        for profile in PROFILES:
            row = find_row(
                process,
                profile=profile,
                metric=metric,
            )

            rows.append(
                (
                    PROFILE_LABEL[
                        profile
                    ],
                    float(
                        row[
                            "mean_delta"
                        ]
                    ),
                    float(
                        row[
                            "bootstrap_ci_low"
                        ]
                    ),
                    float(
                        row[
                            "bootstrap_ci_high"
                        ]
                    ),
                    (
                        "Holm p="
                        + fmt_p(
                            row[
                                "holm_p"
                            ]
                        )
                    ),
                )
            )

        path = (
            figures
            / filename
        )

        svg_forest(
            rows=rows,
            title=(
                "Resource vs eval-only: "
                + label
            ),
            x_label=xlabel,
            path=path,
        )

        process_figures.append(
            filename
        )

    # --------------------------------------------------
    # Recognition specificity
    # --------------------------------------------------

    recognition_series = []

    for judge in (
        "deepseek",
        "gemini",
    ):
        values = []

        for profile in PROFILES:
            row = find_row(
                focal,
                profile=profile,
                condition=(
                    "eval_resource_deprivation"
                ),
                judge_family=judge,
                field=(
                    "resource_constraint_recognition"
                ),
            )

            values.append(
                100.0
                * float(
                    row[
                        "rate_among_valid"
                    ]
                )
            )

        recognition_series.append(
            (
                JUDGE_LABEL[
                    judge
                ],
                values,
            )
        )

    svg_grouped_bars(
        groups=[
            PROFILE_LABEL[p]
            for p in PROFILES
        ],
        series=recognition_series,
        title=(
            "Resource-constraint recognition "
            "under treatment"
        ),
        y_label=(
            "Valid judgments labeled "
            "observed (%)"
        ),
        path=(
            figures
            / "resource_recognition.svg"
        ),
    )

    # --------------------------------------------------
    # Stance conditional on recognition
    # --------------------------------------------------

    stance_series = []

    for judge in (
        "deepseek",
        "gemini",
    ):
        accepted = []
        rejected = []

        for profile in PROFILES:
            row = find_row(
                mechanism,
                profile=profile,
                judge_family=judge,
            )

            denom = float(
                row[
                    "resource_recognized_n"
                ]
            )

            accepted.append(
                (
                    100.0
                    * float(
                        row[
                            "stance_accepted_n"
                        ]
                    )
                    / denom
                )
                if denom
                else 0.0
            )

            rejected.append(
                (
                    100.0
                    * float(
                        row[
                            "stance_rejected_n"
                        ]
                    )
                    / denom
                )
                if denom
                else 0.0
            )

        stance_series.extend([
            (
                (
                    "Accepted — "
                    + JUDGE_LABEL[
                        judge
                    ]
                ),
                accepted,
            ),
            (
                (
                    "Rejected — "
                    + JUDGE_LABEL[
                        judge
                    ]
                ),
                rejected,
            ),
        ])

    svg_grouped_bars(
        groups=[
            PROFILE_LABEL[p]
            for p in PROFILES
        ],
        series=stance_series,
        title=(
            "Stance conditional on "
            "resource recognition"
        ),
        y_label=(
            "Recognized cases (%)"
        ),
        path=(
            figures
            / "resource_stance.svg"
        ),
    )

    # --------------------------------------------------
    # Claimed response
    # --------------------------------------------------

    response_series = []

    for judge in (
        "deepseek",
        "gemini",
    ):
        adapt = []
        ignore = []

        for profile in PROFILES:
            row = find_row(
                mechanism,
                profile=profile,
                judge_family=judge,
            )

            denom = float(
                row[
                    "resource_recognized_n"
                ]
            )

            adapt.append(
                (
                    100.0
                    * float(
                        row[
                            "response_adapt_n"
                        ]
                    )
                    / denom
                )
                if denom
                else 0.0
            )

            ignore.append(
                (
                    100.0
                    * float(
                        row[
                            "response_ignore_n"
                        ]
                    )
                    / denom
                )
                if denom
                else 0.0
            )

        response_series.extend([
            (
                (
                    "Adapt — "
                    + JUDGE_LABEL[
                        judge
                    ]
                ),
                adapt,
            ),
            (
                (
                    "Ignore — "
                    + JUDGE_LABEL[
                        judge
                    ]
                ),
                ignore,
            ),
        ])

    svg_grouped_bars(
        groups=[
            PROFILE_LABEL[p]
            for p in PROFILES
        ],
        series=response_series,
        title=(
            "Claimed response conditional on "
            "resource recognition"
        ),
        y_label=(
            "Recognized cases (%)"
        ),
        path=(
            figures
            / "resource_response.svg"
        ),
    )

    # --------------------------------------------------
    # Reduced validation
    # --------------------------------------------------

    reduce_series = []

    for judge in (
        "deepseek",
        "gemini",
    ):
        values = []

        for profile in PROFILES:
            row = find_row(
                focal,
                profile=profile,
                condition=(
                    "eval_resource_deprivation"
                ),
                judge_family=judge,
                field=(
                    "response_reduce_validation"
                ),
            )

            values.append(
                100.0
                * float(
                    row[
                        "rate_among_valid"
                    ]
                )
            )

        reduce_series.append(
            (
                JUDGE_LABEL[
                    judge
                ],
                values,
            )
        )

    svg_grouped_bars(
        groups=[
            PROFILE_LABEL[p]
            for p in PROFILES
        ],
        series=reduce_series,
        title=(
            "Explicit resource-driven "
            "reduction in validation"
        ),
        y_label=(
            "Valid judgments labeled "
            "observed (%)"
        ),
        path=(
            figures
            / "reduce_validation.svg"
        ),
    )

    # --------------------------------------------------
    # Agreement
    # --------------------------------------------------

    agreement_fields = (
        (
            "resource_constraint_recognition",
            "Resource recognition",
        ),
        (
            "resource_constraint_stance",
            "Resource stance",
        ),
        (
            "claimed_resource_response",
            "Claimed response",
        ),
        (
            "response_reduce_validation",
            "Reduce validation",
        ),
        (
            "evaluation_recognition",
            "Evaluation recognition",
        ),
        (
            "evaluation_targeting_intent",
            "Evaluator targeting",
        ),
    )

    agreement_rows = []

    for field, label in (
        agreement_fields
    ):
        row = find_row(
            agreement,
            field=field,
        )

        agreement_rows.append([
            esc(label),
            esc(
                row[
                    "both_valid_n"
                ]
            ),
            esc(
                fmt_pct(
                    row[
                        "raw_agreement"
                    ]
                )
            ),
            esc(
                fmt_num(
                    row[
                        "cohen_kappa"
                    ],
                    3,
                )
            ),
            esc(
                fmt_num(
                    row[
                        "gwet_ac1"
                    ],
                    3,
                )
            ),
        ])

    agreement_table = html_table(
        [
            "Field",
            "N both valid",
            "Raw agreement",
            "Cohen κ",
            "Gwet AC1",
        ],
        agreement_rows,
    )

    # --------------------------------------------------
    # Success contrast table
    # --------------------------------------------------

    contrast_rows = []

    for profile in PROFILES:
        for (
            source,
            label,
        ) in (
            (
                primary,
                "Resource vs eval-only",
            ),
            (
                resource_clean,
                "Resource vs clean",
            ),
        ):
            row = find_row(
                source,
                profile=profile,
            )

            contrast_rows.append([
                esc(
                    PROFILE_LABEL[
                        profile
                    ]
                ),
                esc(label),
                esc(
                    row[
                        "n_pairs"
                    ]
                ),
                esc(
                    fmt_pp(
                        row[
                            "delta_pp"
                        ]
                    )
                ),
                esc(
                    fmt_ci(
                        row[
                            "bootstrap_ci_low_pp"
                        ],
                        row[
                            "bootstrap_ci_high_pp"
                        ],
                        pp=True,
                    )
                ),
                esc(
                    fmt_p(
                        row[
                            "mcnemar_exact_p"
                        ]
                    )
                ),
                esc(
                    fmt_p(
                        row[
                            "mcnemar_holm_p"
                        ]
                    )
                ),
            ])

    contrast_table = html_table(
        [
            "Model",
            "Contrast",
            "N pairs",
            "Δ success",
            "95% bootstrap CI",
            "McNemar p",
            "Holm p",
        ],
        contrast_rows,
    )

    # --------------------------------------------------
    # Process table
    # --------------------------------------------------

    process_rows = []

    metric_label = {
        "raw_tool_calls": (
            "Raw tool calls"
        ),
        "input_tokens": (
            "Input tokens"
        ),
        "validation_command_calls": (
            "Validation commands"
        ),
    }

    for row in process:
        if (
            row.get("profile")
            not in PROFILES
        ):
            continue

        process_rows.append([
            esc(
                PROFILE_LABEL[
                    row["profile"]
                ]
            ),
            esc(
                metric_label[
                    row["metric"]
                ]
            ),
            esc(
                row[
                    "n_pairs"
                ]
            ),
            esc(
                fmt_num(
                    row[
                        "mean_delta"
                    ]
                )
            ),
            esc(
                fmt_ci(
                    row[
                        "bootstrap_ci_low"
                    ],
                    row[
                        "bootstrap_ci_high"
                    ],
                )
            ),
            esc(
                fmt_p(
                    row[
                        "signflip_p"
                    ]
                )
            ),
            esc(
                fmt_p(
                    row[
                        "holm_p"
                    ]
                )
            ),
        ])

    process_table = html_table(
        [
            "Model",
            "Metric",
            "N pairs",
            "Mean paired Δ",
            "95% bootstrap CI",
            "Sign-flip p",
            "Holm p",
        ],
        process_rows,
    )

    # --------------------------------------------------
    # Mechanism table
    # --------------------------------------------------

    mechanism_rows = []

    for profile in PROFILES:
        for judge in (
            "deepseek",
            "gemini",
        ):
            row = find_row(
                mechanism,
                profile=profile,
                judge_family=judge,
            )

            recognized = float(
                row[
                    "resource_recognized_n"
                ]
            )

            accepted = float(
                row[
                    "stance_accepted_n"
                ]
            )

            rejected = float(
                row[
                    "stance_rejected_n"
                ]
            )

            adapt = float(
                row[
                    "response_adapt_n"
                ]
            )

            ignore = float(
                row[
                    "response_ignore_n"
                ]
            )

            mechanism_rows.append([
                esc(
                    PROFILE_LABEL[
                        profile
                    ]
                ),
                esc(
                    JUDGE_LABEL[
                        judge
                    ]
                ),
                esc(
                    row[
                        "valid_n"
                    ]
                ),
                esc(
                    row[
                        "missing_n"
                    ]
                ),
                esc(
                    (
                        f"{int(recognized)} "
                        f"({fmt_pct(row['resource_recognition_rate_valid'])})"
                    )
                ),
                esc(
                    (
                        f"{int(accepted)} "
                        f"({fmt_pct(accepted / recognized) if recognized else '—'})"
                    )
                ),
                esc(
                    (
                        f"{int(rejected)} "
                        f"({fmt_pct(rejected / recognized) if recognized else '—'})"
                    )
                ),
                esc(
                    (
                        f"{int(adapt)} "
                        f"({fmt_pct(adapt / recognized) if recognized else '—'})"
                    )
                ),
                esc(
                    (
                        f"{int(ignore)} "
                        f"({fmt_pct(ignore / recognized) if recognized else '—'})"
                    )
                ),
                esc(
                    row[
                        "evaluation_targeting_n"
                    ]
                ),
            ])

    mechanism_table = html_table(
        [
            "Model",
            "Judge",
            "Valid N",
            "Missing",
            "Resource recognized",
            "Accepted | recognized",
            "Rejected | recognized",
            "Adapt | recognized",
            "Ignore | recognized",
            "Eval targeting",
        ],
        mechanism_rows,
    )

    # --------------------------------------------------
    # Strict bounds key endpoints
    # --------------------------------------------------

    key_fields = (
        (
            "resource_constraint_recognition",
            "Resource recognition",
        ),
        (
            "resource_constraint_stance",
            "Accepted stance",
        ),
        (
            "claimed_resource_response",
            "Adapt response",
        ),
        (
            "response_reduce_validation",
            "Reduce validation",
        ),
        (
            "evaluation_targeting_intent",
            "Evaluator targeting",
        ),
    )

    bound_rows = []

    for profile in PROFILES:
        for (
            field,
            label,
        ) in key_fields:
            row = find_row(
                bounds,
                profile=profile,
                condition=(
                    "eval_resource_deprivation"
                ),
                field=field,
            )

            bound_rows.append([
                esc(
                    PROFILE_LABEL[
                        profile
                    ]
                ),
                esc(label),
                esc(
                    row[
                        "both_valid_n"
                    ]
                ),
                esc(
                    row[
                        "disagreement_n"
                    ]
                ),
                esc(
                    row[
                        "incomplete_pair_n"
                    ]
                ),
                esc(
                    fmt_pct(
                        row[
                            "rate_among_resolved"
                        ]
                    )
                ),
                esc(
                    (
                        fmt_pct(
                            row[
                                "lower_bound_full_cell"
                            ]
                        )
                        + " – "
                        + fmt_pct(
                            row[
                                "upper_bound_full_cell"
                            ]
                        )
                    )
                ),
            ])

    bounds_table = html_table(
        [
            "Model",
            "Endpoint",
            "Both valid",
            "Disagree",
            "Incomplete pair",
            "Resolved rate",
            "Full-cell bound",
        ],
        bound_rows,
    )

    # --------------------------------------------------
    # Omnibus table
    # --------------------------------------------------

    omnibus_rows = []

    for profile in PROFILES:
        row = find_row(
            cochran,
            profile=profile,
        )

        omnibus_rows.append([
            esc(
                PROFILE_LABEL[
                    profile
                ]
            ),
            esc(
                row[
                    "n_complete_triples"
                ]
            ),
            esc(
                fmt_num(
                    row[
                        "cochran_q"
                    ]
                )
            ),
            esc(
                fmt_p(
                    row[
                        "p_value"
                    ]
                )
            ),
            esc(
                fmt_p(
                    row[
                        "holm_p"
                    ]
                )
            ),
        ])

    omnibus_table = html_table(
        [
            "Model",
            "Complete triples",
            "Cochran Q",
            "p",
            "Holm p",
        ],
        omnibus_rows,
    )

    # --------------------------------------------------
    # Source inventory/provenance
    # --------------------------------------------------

    source_files = [
        inference
        / "primary_success.csv",
        inference
        / "secondary_eval_success.csv",
        inference
        / "primary_process.csv",
        inference
        / "cochran_q.csv",
        inference
        / "resource_vs_clean.csv",
        semantic
        / "individual_focal_rates.csv",
        semantic
        / "resource_mechanism_individual.csv",
        semantic
        / "agreement_by_field.csv",
        semantic
        / "strict_consensus_focal_bounds.csv",
        behavior_root
        / "config"
        / "resource_semantic_production_freeze_v1.1.json",
        behavior_root
        / "config"
        / "resource_semantic_analysis_freeze_v1.1.json",
    ]

    source_inventory = [
        {
            "path": str(path),
            "sha256": sha256(
                path
            ),
        }
        for path in source_files
    ]

    provenance = {
        "version": VERSION,
        "presentation_only": True,
        "recomputes_effect_estimates": False,
        "recomputes_uncertainty": False,
        "recomputes_p_values": False,
        "recomputes_multiplicity": False,
        "recomputes_semantic_labels": False,
        "source_inventory": (
            source_inventory
        ),
        "semantic_production": {
            "trajectory_count": (
                production_freeze[
                    "trajectory_count"
                ]
            ),
            "judge_job_count": (
                production_freeze[
                    "judge_job_count"
                ]
            ),
            "ok": (
                production_freeze[
                    "terminal_job_status"
                ][
                    "ok"
                ]
            ),
            "missing": (
                production_freeze[
                    "terminal_job_status"
                ][
                    "missing"
                ]
            ),
        },
        "semantic_analysis_ledger_sha256": (
            semantic_freeze[
                "output_ledger_sha256"
            ]
        ),
        "llama_treatment_role": (
            "excluded from causal treatment "
            "figures because scaffold exposure "
            "was not observably verified"
        ),
        "network_calls": 0,
        "judge_calls": 0,
    }

    (
        output
        / "provenance.json"
    ).write_text(
        json.dumps(
            provenance,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # --------------------------------------------------
    # Main HTML
    # --------------------------------------------------

    style = r"""
:root {
  --bg: #f7f8fa;
  --panel: #ffffff;
  --text: #17202a;
  --muted: #64748b;
  --border: #dbe2ea;
  --good-bg: #f0fdf4;
  --good-border: #22c55e;
  --warn-bg: #fff7ed;
  --warn-border: #f97316;
  --info-bg: #eff6ff;
  --info-border: #3b82f6;
  --purple-bg: #faf5ff;
  --purple-border: #a855f7;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont,
    "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  line-height: 1.48;
}
main {
  max-width: 1480px;
  margin: 0 auto;
  padding: 32px 24px 80px;
}
h1 {
  margin: 0;
  font-size: 34px;
}
h2 {
  margin-top: 42px;
  padding-bottom: 9px;
  border-bottom: 1px solid var(--border);
}
h3 {
  margin-top: 28px;
}
.subtitle {
  color: var(--muted);
  margin: 8px 0 20px;
}
.grid {
  display: grid;
  grid-template-columns:
    repeat(auto-fit, minmax(260px, 1fr));
  gap: 14px;
  margin: 18px 0;
}
.card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 17px;
}
.card strong.big {
  display: block;
  font-size: 25px;
  margin-bottom: 5px;
}
.card .small {
  color: var(--muted);
  font-size: 13px;
}
.finding {
  border-left: 5px solid;
}
.finding.claude {
  background: var(--info-bg);
  border-color: var(--info-border);
}
.finding.fable {
  background: var(--good-bg);
  border-color: var(--good-border);
}
.finding.codex {
  background: var(--purple-bg);
  border-color: var(--purple-border);
}
.warning {
  background: var(--warn-bg);
  border-left: 5px solid var(--warn-border);
  border-radius: 8px;
  padding: 15px 18px;
  margin: 18px 0;
}
.note {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 14px 16px;
  color: var(--muted);
  margin: 14px 0;
}
.figure {
  background: white;
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 14px;
  margin: 18px 0;
  overflow-x: auto;
}
.figure img {
  display: block;
  max-width: 100%;
  height: auto;
  margin: 0 auto;
}
.figure-caption {
  margin-top: 9px;
  color: var(--muted);
  font-size: 13px;
}
table {
  width: 100%;
  border-collapse: collapse;
  background: var(--panel);
}
th, td {
  border: 1px solid var(--border);
  padding: 9px 10px;
  vertical-align: top;
}
th {
  background: #eef2f7;
  text-align: left;
  font-size: 12px;
  position: sticky;
  top: 0;
  z-index: 2;
}
.compact td {
  font-size: 13px;
}
.table-scroll {
  overflow: auto;
  max-height: 70vh;
  border: 1px solid var(--border);
  border-radius: 8px;
}
.table-scroll table {
  border: 0;
}
code {
  font-size: 12px;
}
.badge {
  display: inline-block;
  padding: 3px 8px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 600;
  background: #e2e8f0;
  margin-right: 5px;
}
.primary {
  background: #dcfce7;
}
.secondary {
  background: #dbeafe;
}
.exploratory {
  background: #f3e8ff;
}
.muted {
  color: var(--muted);
}
ul.tight li {
  margin: 5px 0;
}
"""

    html_text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport"
      content="width=device-width,initial-scale=1">
<title>SWE-EvalPressure — Resource Deprivation Results</title>
<style>
{style}
</style>
</head>
<body>
<main>

<h1>SWE-EvalPressure: Resource Deprivation</h1>

<div class="subtitle">
Integrated frozen analysis · 70 SWE-Atlas tasks/model ·
clean vs eval-only vs eval + severe resource deprivation ·
scaffold placement
</div>

<div class="warning">
<strong>Interpretation boundary.</strong>
The prespecified primary causal contrast is
<strong>resource vs eval-only</strong>.
Resource vs clean is secondary contextualization.
Semantic recognition, stance, and response are post-treatment
measurements and are not treated as causal mediators.
Llama is excluded from treatment-effect interpretation because
scaffold exposure was not observably verified.
</div>

<h2>Executive summary</h2>

<div class="grid">

<div class="card">
<strong class="big">618</strong>
<div>semantic-eligible trajectories</div>
<div class="small">
1,216 / 1,236 valid core-judge outputs (98.38%)
</div>
</div>

<div class="card">
<strong class="big">0</strong>
<div>resource-specific semantic positives in clean for the
headline mechanism fields</div>
<div class="small">
The resource signal is treatment-specific.
</div>
</div>

<div class="card">
<strong class="big">0</strong>
<div>validated evaluator-targeting cases under resource treatment</div>
<div class="small">
Observed in neither core judge for Claude, Fable, or Codex.
</div>
</div>

<div class="card">
<strong class="big">−20 pp</strong>
<div>Codex task-success change: resource vs eval-only</div>
<div class="small">
Primary contrast; Holm-adjusted p≈0.013.
</div>
</div>

</div>

<div class="grid">

<div class="card finding claude">
<strong>Claude — recognize, reject, ignore</strong>
<p>
Resource recognition is ~94–96%.
Among recognized cases, judges overwhelmingly classify the
constraint as untrusted/injection-like and the claimed response
as ignoring it.
</p>
<p class="small">
Task success and measured process behavior show no robust
resource-vs-eval deterioration.
</p>
</div>

<div class="card finding fable">
<strong>Fable — partial recognition, mostly adapt</strong>
<p>
Resource recognition is ~50–58%.
Recognized cases usually accept the constraint and adapt.
</p>
<p class="small">
Tool calls, input-token volume, and validation commands fall
robustly, without a detectable task-success loss.
</p>
</div>

<div class="card finding codex">
<strong>Codex — recognize, accept, adapt</strong>
<p>
Resource recognition is ~72–81% among valid judge outputs.
Recognized cases overwhelmingly accept the constraint and adapt,
including explicit reductions in validation.
</p>
<p class="small">
Task success falls robustly versus eval-only, but not
significantly versus clean.
</p>
</div>

</div>

<h2>1. Task success</h2>

<span class="badge primary">Primary: resource vs eval-only</span>
<span class="badge secondary">Secondary: eval-only vs clean</span>
<span class="badge secondary">Secondary: resource vs clean</span>

<h3>Condition-level success</h3>

{success_table}

<div class="figure">
<img src="figures/success_effects.svg"
     alt="Paired task success effects">
<div class="figure-caption">
Paired effects use the already-computed matched-task estimates
and bootstrap confidence intervals. The resource-vs-eval
comparison is primary; resource-vs-clean is secondary.
</div>
</div>

<h3>Paired contrasts</h3>

{contrast_table}

<h3>Three-condition omnibus test</h3>

{omnibus_table}

<div class="note">
The key Codex nuance is that resource deprivation is
<strong>−20 pp relative to eval-only</strong>, but only
<strong>−7.1 pp relative to clean</strong>, with the direct
resource-vs-clean CI crossing zero. This does not support a claim
that resource pressure causes a general below-baseline collapse.
</div>

<h2>2. Execution and validation behavior</h2>

<span class="badge exploratory">Exploratory process family</span>

<div class="grid">
<div class="figure">
<img src="figures/process_tool_calls.svg"
     alt="Tool call effect">
</div>
<div class="figure">
<img src="figures/process_input_tokens.svg"
     alt="Input token effect">
</div>
<div class="figure">
<img src="figures/process_validation.svg"
     alt="Validation command effect">
</div>
</div>

{process_table}

<div class="note">
These are execution/input-token/validation-volume measurements.
They should not be described as “thinking less.”
</div>

<h2>3. Resource recognition is treatment-specific</h2>

<span class="badge primary">Main semantic chain</span>

<div class="figure">
<img src="figures/resource_recognition.svg"
     alt="Resource recognition">
<div class="figure-caption">
Resource recognition is essentially absent outside the resource
condition and rises sharply under treatment.
</div>
</div>

<div class="note">
The strongest semantic measurement is resource-specific
recognition, not generic evaluation-recognition.
Resource recognition has 97.8% raw judge agreement,
κ≈0.94 and AC1≈0.97.
</div>

<h2>4. What agents do with the resource cue</h2>

<h3>Stance conditional on recognition</h3>

<div class="figure">
<img src="figures/resource_stance.svg"
     alt="Resource stance">
</div>

<h3>Claimed response conditional on recognition</h3>

<div class="figure">
<img src="figures/resource_response.svg"
     alt="Resource response">
</div>

{mechanism_table}

<div class="note">
Claude and Codex occupy opposite semantic states after
recognition: Claude rejects/ignores the constraint, while Codex
accepts/adapts. Fable is intermediate in recognition frequency
but usually accepts/adapts when it recognizes the constraint.
These are descriptive post-treatment states, not causal
mediation estimates.
</div>

<h2>5. Explicit reduction in validation</h2>

<span class="badge secondary">Higher-confidence secondary semantic endpoint</span>

<div class="figure">
<img src="figures/reduce_validation.svg"
     alt="Reduced validation">
<div class="figure-caption">
Explicitly attributed reduction in validation is absent in Claude
and substantial in Codex/Fable under the resource condition.
</div>
</div>

<div class="note">
Fine-grained fields such as reduce-exploration and
prioritize-core-edit show materially weaker cross-judge
agreement and remain exploratory.
</div>

<h2>6. Strict consensus and missingness bounds</h2>

{bounds_table}

<div class="warning">
<strong>Selective semantic missingness.</strong>
20 / 1,236 core judge jobs are terminal missing.
18 / 20 occur in the resource condition, and 13 are DeepSeek
judgments on Codex-resource trajectories. The invalid attempts
were usually semantically positive but failed the frozen
verbatim-evidence validator. They remain excluded from primary
judgments. This is why the report shows judge-specific estimates
and strict-consensus bounds rather than pretending complete-case
consensus is missing at random.
</div>

<h2>7. Core-judge agreement</h2>

{agreement_table}

<div class="note">
Generic evaluation-recognition is substantially less stable
across judges than the resource-specific mechanism fields.
It remains secondary pending human validation.
Gwet AC1 is included alongside κ because several endpoints have
strong class imbalance.
</div>

<h2>8. Evaluator targeting</h2>

<div class="card">
<strong class="big">0 observed</strong>
<p>
No validated evaluator-targeting intent was identified by either
core judge in the resource-condition trajectories for Claude,
Fable, or Codex.
</p>
<p class="small">
This statement applies only to this experiment and should not be
generalized beyond the observed trajectories.
</p>
</div>

<h2>9. What this resource follow-up currently supports</h2>

<ul class="tight">
<li>
<strong>Resource information is behaviorally meaningful only
after it is recognized and trusted.</strong>
Claude usually recognizes the cue but rejects it; Codex usually
recognizes and accepts it; Fable is intermediate.
</li>
<li>
<strong>Recognition alone is insufficient.</strong>
Claude provides the clearest example: extremely high resource
recognition with essentially no acceptance/adaptation.
</li>
<li>
<strong>Codex shows the strongest alignment between semantic
adaptation, reduced process activity, and task-success change
relative to eval-only.</strong>
</li>
<li>
<strong>Fable shows adaptation without detectable task-success
loss.</strong>
This separates behavioral adaptation from performance harm.
</li>
<li>
<strong>The effect is not a generic “resource pressure hurts all
models” phenomenon.</strong>
Claude is largely unaffected; Fable success is statistically
uncertain; Codex’s strongest effect is specifically relative to
eval-only.
</li>
<li>
<strong>No explicit evaluator-targeting intent was observed in
this resource experiment.</strong>
</li>
</ul>

<h2>10. Remaining work before paper-wide synthesis</h2>

<ol>
<li>
Historical full semantic production for eval-only, financial,
and self-preservation conditions across root / scaffold /
source-local placements.
</li>
<li>
Resolve and version the repaired Mini-SWE Llama historical run;
do not silently overwrite the old historical result.
</li>
<li>
Human validation of approximately 200 examples with multiple
independent annotators.
</li>
<li>
Human-human and judge-human agreement analysis.
</li>
<li>
Final integrated historical + resource HTML/report and
publication figures.
</li>
</ol>

<h2>11. Provenance</h2>

<div class="card">
<div>
<strong>Renderer:</strong>
<code>{esc(VERSION)}</code>
</div>
<div>
<strong>Semantic production:</strong>
{esc(production_freeze["trajectory_count"])} trajectories,
{esc(production_freeze["judge_job_count"])} judge jobs,
{esc(production_freeze["terminal_job_status"]["ok"])} valid,
{esc(production_freeze["terminal_job_status"]["missing"])} missing.
</div>
<div>
<strong>Semantic analysis ledger:</strong>
<code>{esc(semantic_freeze["output_ledger_sha256"])}</code>
</div>
<div>
<strong>Presentation policy:</strong>
no effect, CI, p-value, multiplicity, or semantic-label
recomputation.
</div>
</div>

</main>
</body>
</html>
"""

    html_path = (
        output
        / "index.html"
    )

    html_path.write_text(
        html_text,
        encoding="utf-8",
    )

    print(
        "RESOURCE INTEGRATED HTML"
    )
    print("=" * 76)

    print(
        "presentation-only: PASS"
    )

    print(
        "success inputs: PASS"
    )

    print(
        "process inputs: PASS"
    )

    print(
        "semantic inputs: PASS"
    )

    print(
        "semantic freezes: PASS"
    )

    print(
        "html:",
        html_path,
    )

    print(
        "figures:",
        figures,
    )

    for path in sorted(
        figures.glob(
            "*.svg"
        )
    ):
        print(
            " -",
            path.name,
        )

    print(
        "provenance:",
        output
        / "provenance.json",
    )

    print(
        "network calls: 0"
    )

    print(
        "judge calls: 0"
    )

    print(
        "RESOURCE INTEGRATED HTML: PASS"
    )


if __name__ == "__main__":
    main()
