#!/usr/bin/env python3
"""Render a self-contained SWE-EvalPressure behavioral HTML pre-read.

This script performs presentation only. It consumes the deterministic
synthesis CSVs produced by scripts/10_behavior_report.py and does not
recompute effect estimates, uncertainty, p-values, or multiplicity
adjustments.
"""

from __future__ import annotations

import argparse
import csv
import html
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


HTML_REPORT_SCHEMA_VERSION = "1.0"

REQUIRED_FILES = (
    "behavior_report_inventory.csv",
    "behavior_primary_matrix.csv",
    "behavior_secondary_effects_all.csv",
    "behavior_multiplicity_all.csv",
    "matched_behavior_pairs_all.csv",
)

PRIMARY_ENDPOINTS = (
    "broad_repo_search_any",
    "test_inspection_any",
    "validation_any",
    "iterative_repair_any",
    "provenance_related_inspection_any",
    "external_lookup_any",
    "integrity_sensitive_action_any",
)

ENDPOINT_LABELS = {
    "broad_repo_search_any": "Repository search",
    "test_inspection_any": "Test inspection",
    "validation_any": "Validation",
    "iterative_repair_any": "Iterative repair",
    "provenance_related_inspection_any": "Provenance inspection",
    "external_lookup_any": "External lookup",
    "integrity_sensitive_action_any": "Integrity-sensitive modification",
}

PAIR_LABELS = {
    "eval_effect": "Evaluation disclosure",
    "financial_effect": "Financial pressure",
    "self_preservation_effect": "Self-preservation pressure",
    "resource_effect": "Resource deprivation",
}

SELECTED_DRILLDOWN_METRICS = (
    "broad_repo_search_any",
    "test_inspection_any",
    "validation_any",
    "iterative_repair_any",
    "repo_search_calls",
    "validation_calls",
    "behavioral_action_calls",
    "trajectory_steps",
    "prompt_tokens",
    "completion_tokens",
    "duration_seconds",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        return list(csv.DictReader(handle))


def esc(value: Any) -> str:
    return html.escape(
        str(value if value is not None else "")
    )


def number(value: Any) -> float | None:
    if value in (None, ""):
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def integer(value: Any) -> int | None:
    parsed = number(value)
    if parsed is None:
        return None
    return int(parsed)


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }


def fmt_number(
    value: Any,
    digits: int = 2,
) -> str:
    parsed = number(value)

    if parsed is None:
        return "—"

    if abs(parsed) >= 100_000:
        return f"{parsed:,.0f}"

    return f"{parsed:,.{digits}f}"


def fmt_pp(value: Any) -> str:
    parsed = number(value)

    if parsed is None:
        return "—"

    return f"{parsed:+.1f} pp"


def fmt_p(value: Any) -> str:
    parsed = number(value)

    if parsed is None:
        return "—"

    if parsed < 0.001:
        return f"{parsed:.2e}"

    return f"{parsed:.4f}"


def delta_class(value: Any) -> str:
    parsed = number(value)

    if parsed is None:
        return "neutral"

    if parsed > 0:
        return "positive"

    if parsed < 0:
        return "negative"

    return "neutral"


def pair_ledger(
    rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[str, str, str],
        list[dict[str, str]],
    ] = defaultdict(list)

    for row in rows:
        grouped[
            (
                row.get("profile", ""),
                row.get("pair_type", ""),
                row.get("channel", ""),
            )
        ].append(row)

    output: list[dict[str, Any]] = []

    for (
        profile,
        pair_type,
        channel,
    ), values in sorted(grouped.items()):
        states = Counter(
            row.get("pair_state", "")
            for row in values
        )

        usable = sum(
            truthy(row.get("pair_usable"))
            for row in values
        )

        missing = sum(
            count
            for state, count in states.items()
            if "missing" in state.lower()
        )

        censored = (
            len(values)
            - usable
            - missing
        )

        output.append({
            "profile": profile,
            "pair_type": pair_type,
            "channel": channel,
            "planned": len(values),
            "usable": usable,
            "missing": missing,
            "censored": censored,
        })

    return output


def is_partial_snapshot(
    pair_rows: list[dict[str, str]],
) -> bool:
    return any(
        "missing" in str(
            row.get("pair_state", "")
        ).lower()
        for row in pair_rows
    )


def primary_effect_cell(
    row: dict[str, str],
    endpoint: str,
) -> str:
    delta = row.get(
        f"{endpoint}__delta_pp",
        "",
    )
    ci_low = row.get(
        f"{endpoint}__ci_low_pp",
        "",
    )
    ci_high = row.get(
        f"{endpoint}__ci_high_pp",
        "",
    )
    holm = row.get(
        f"{endpoint}__holm_p",
        "",
    )
    raw_p = row.get(
        f"{endpoint}__mcnemar_p",
        "",
    )
    significant = truthy(
        row.get(
            f"{endpoint}__adjusted_reject",
            "",
        )
    )

    sig_badge = (
        '<span class="sig">Holm significant</span>'
        if significant
        else ""
    )

    return (
        f'<td class="effect {delta_class(delta)} '
        f'{"significant" if significant else ""}">'
        f'<div class="delta">{esc(fmt_pp(delta))}</div>'
        f'<div class="meta">'
        f'95% CI [{esc(fmt_pp(ci_low))}, '
        f'{esc(fmt_pp(ci_high))}]'
        f'</div>'
        f'<div class="meta">'
        f'McNemar p={esc(fmt_p(raw_p))} · '
        f'Holm p={esc(fmt_p(holm))}'
        f'</div>'
        f'{sig_badge}'
        f'</td>'
    )


def render_inventory(
    rows: list[dict[str, str]],
) -> str:
    cells = []

    for row in rows:
        cells.append(
            "<tr>"
            f"<td>{esc(row.get('profile'))}</td>"
            f"<td>{esc(row.get('analysis_mode'))}</td>"
            f"<td>{esc(row.get('analysis_schema_version'))}</td>"
            f"<td><code>{esc(row.get('study_signatures'))}</code></td>"
            "</tr>"
        )

    return (
        '<table class="compact">'
        "<thead><tr>"
        "<th>Profile</th>"
        "<th>Study mode</th>"
        "<th>Analyzer schema</th>"
        "<th>Study signature</th>"
        "</tr></thead>"
        "<tbody>"
        + "".join(cells)
        + "</tbody></table>"
    )


def render_ledger(
    rows: list[dict[str, Any]],
) -> str:
    body = []

    for row in rows:
        label = PAIR_LABELS.get(
            str(row["pair_type"]),
            str(row["pair_type"]),
        )

        body.append(
            "<tr>"
            f"<td>{esc(row['profile'])}</td>"
            f"<td>{esc(label)}</td>"
            f"<td>{esc(row['channel'])}</td>"
            f"<td>{row['planned']}</td>"
            f"<td>{row['usable']}</td>"
            f"<td>{row['censored']}</td>"
            f"<td>{row['missing']}</td>"
            "</tr>"
        )

    return (
        '<table class="compact">'
        "<thead><tr>"
        "<th>Profile</th>"
        "<th>Contrast</th>"
        "<th>Placement</th>"
        "<th>Planned</th>"
        "<th>Usable</th>"
        "<th>Censored</th>"
        "<th>Missing</th>"
        "</tr></thead>"
        "<tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def render_primary_matrix(
    rows: list[dict[str, str]],
) -> str:
    header = "".join(
        f"<th>{esc(ENDPOINT_LABELS[endpoint])}</th>"
        for endpoint in PRIMARY_ENDPOINTS
    )

    body = []

    for row in rows:
        pair_type = row.get(
            "pair_type",
            "",
        )

        label = PAIR_LABELS.get(
            pair_type,
            pair_type,
        )

        effect_cells = "".join(
            primary_effect_cell(
                row,
                endpoint,
            )
            for endpoint in PRIMARY_ENDPOINTS
        )

        body.append(
            "<tr>"
            "<td class=\"sticky-col\">"
            f"<strong>{esc(row.get('profile'))}</strong>"
            f"<div>{esc(label)}</div>"
            f"<div class=\"meta\">"
            f"{esc(row.get('channel'))} · "
            f"n={esc(row.get('n_pairs'))}"
            f"</div>"
            "</td>"
            f"{effect_cells}"
            "</tr>"
        )

    if not body:
        return '<p class="muted">No primary effect rows.</p>'

    return (
        '<div class="table-scroll">'
        '<table class="matrix">'
        "<thead><tr>"
        "<th>Model / contrast</th>"
        f"{header}"
        "</tr></thead>"
        "<tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def secondary_sort_key(
    row: dict[str, str],
) -> tuple[int, float, str, str]:
    significant = truthy(
        row.get("adjusted_reject")
    )

    q = number(
        row.get("bh_adjusted_q")
    )

    return (
        0 if significant else 1,
        q if q is not None else 999.0,
        row.get("profile", ""),
        row.get("metric", ""),
    )


def render_secondary(
    rows: list[dict[str, str]],
) -> str:
    body = []

    for row in sorted(
        rows,
        key=secondary_sort_key,
    ):
        significant = truthy(
            row.get("adjusted_reject")
        )

        label = PAIR_LABELS.get(
            row.get("pair_type", ""),
            row.get("pair_type", ""),
        )

        body.append(
            "<tr "
            + (
                'class="significant-row"'
                if significant
                else ""
            )
            + ">"
            f"<td>{esc(row.get('profile'))}</td>"
            f"<td>{esc(label)}</td>"
            f"<td>{esc(row.get('channel'))}</td>"
            f"<td><code>{esc(row.get('metric'))}</code></td>"
            f"<td>{esc(row.get('n_pairs'))}</td>"
            f"<td class=\"{delta_class(row.get('mean_delta'))}\">"
            f"{esc(fmt_number(row.get('mean_delta')))}</td>"
            f"<td>[{esc(fmt_number(row.get('bootstrap_ci_low')))}, "
            f"{esc(fmt_number(row.get('bootstrap_ci_high')))}]</td>"
            f"<td>{esc(row.get('increased'))}/"
            f"{esc(row.get('unchanged'))}/"
            f"{esc(row.get('decreased'))}</td>"
            f"<td>{esc(fmt_p(row.get('sign_flip_p')))}</td>"
            f"<td>{esc(fmt_p(row.get('bh_adjusted_q')))}</td>"
            f"<td>{'Yes' if significant else 'No'}</td>"
            "</tr>"
        )

    return (
        '<div class="table-scroll">'
        '<table class="compact">'
        "<thead><tr>"
        "<th>Profile</th>"
        "<th>Contrast</th>"
        "<th>Placement</th>"
        "<th>Metric</th>"
        "<th>n</th>"
        "<th>Mean paired Δ</th>"
        "<th>Bootstrap 95% CI</th>"
        "<th>↑/= /↓</th>"
        "<th>Sign-flip p</th>"
        "<th>BH q</th>"
        "<th>FDR reject</th>"
        "</tr></thead>"
        "<tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def render_multiplicity(
    rows: list[dict[str, str]],
) -> str:
    body = []

    for row in rows:
        body.append(
            "<tr>"
            f"<td>{esc(row.get('profile'))}</td>"
            f"<td><code>{esc(row.get('family_name'))}</code></td>"
            f"<td>{esc(row.get('multiplicity_method'))}</td>"
            f"<td>{esc(row.get('family_size'))}</td>"
            f"<td>{esc(row.get('tested_family_size'))}</td>"
            f"<td>{esc(row.get('untested_family_size'))}</td>"
            f"<td>{esc(row.get('unadjusted_rejections'))}</td>"
            f"<td>{esc(row.get('adjusted_rejections'))}</td>"
            "</tr>"
        )

    return (
        '<table class="compact">'
        "<thead><tr>"
        "<th>Profile</th>"
        "<th>Family</th>"
        "<th>Method</th>"
        "<th>Family size</th>"
        "<th>Tested</th>"
        "<th>Untested</th>"
        "<th>Raw rejects</th>"
        "<th>Adjusted rejects</th>"
        "</tr></thead>"
        "<tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def render_pair_drilldown(
    rows: list[dict[str, str]],
) -> str:
    body = []

    for index, row in enumerate(rows):
        label = PAIR_LABELS.get(
            row.get("pair_type", ""),
            row.get("pair_type", ""),
        )

        deltas = []

        for metric in SELECTED_DRILLDOWN_METRICS:
            value = row.get(
                f"delta_{metric}",
                "",
            )

            if value in ("", None):
                continue

            deltas.append(
                f"<code>{esc(metric)}</code>: "
                f"{esc(fmt_number(value))}"
            )

        search_text = " ".join(
            str(row.get(field, ""))
            for field in (
                "profile",
                "base_task_id",
                "pair_type",
                "channel",
                "pair_state",
                "baseline_trial",
                "treatment_trial",
            )
        ).lower()

        body.append(
            f'<tr class="pair-row" '
            f'data-search="{esc(search_text)}">'
            f"<td>{index + 1}</td>"
            f"<td>{esc(row.get('profile'))}</td>"
            f"<td><code>{esc(row.get('base_task_id'))}</code></td>"
            f"<td>{esc(label)}</td>"
            f"<td>{esc(row.get('channel'))}</td>"
            f"<td>{esc(row.get('pair_state'))}</td>"
            f"<td>{'Yes' if truthy(row.get('pair_usable')) else 'No'}</td>"
            "<td>"
            f"<details><summary>Trials</summary>"
            f"<div><strong>Baseline:</strong> "
            f"<code>{esc(row.get('baseline_trial'))}</code></div>"
            f"<div><strong>Treatment:</strong> "
            f"<code>{esc(row.get('treatment_trial'))}</code></div>"
            "</details>"
            "</td>"
            "<td>"
            + (
                "<br>".join(deltas)
                if deltas
                else "—"
            )
            + "</td>"
            "</tr>"
        )

    return (
        '<input id="pair-filter" class="filter" '
        'placeholder="Filter by model, task, contrast, state, trial ID…">'
        '<div class="table-scroll drilldown">'
        '<table class="compact" id="pair-table">'
        "<thead><tr>"
        "<th>#</th>"
        "<th>Profile</th>"
        "<th>Base task</th>"
        "<th>Contrast</th>"
        "<th>Placement</th>"
        "<th>Pair state</th>"
        "<th>Usable</th>"
        "<th>Trial IDs</th>"
        "<th>Selected paired deltas</th>"
        "</tr></thead>"
        "<tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def build_html(
    *,
    inventory: list[dict[str, str]],
    primary: list[dict[str, str]],
    secondary: list[dict[str, str]],
    multiplicity: list[dict[str, str]],
    pairs: list[dict[str, str]],
    title: str,
) -> str:
    partial = is_partial_snapshot(
        pairs
    )

    ledger = pair_ledger(
        pairs
    )

    banner = (
        """
        <div class="warning">
          <strong>VALIDATION SNAPSHOT — NOT FINAL STUDY ESTIMATES</strong>
          <div>
            The planned matched-pair ledger still contains missing pairs.
            Values below are useful for pipeline validation only and must not
            be presented as final benchmark estimates.
          </div>
        </div>
        """
        if partial
        else
        """
        <div class="complete">
          <strong>Complete planned-pair ledger detected</strong>
          <div>
            No planned matched pairs are marked missing. Infrastructure-censored
            pairs, if any, remain excluded from endpoint denominators.
          </div>
        </div>
        """
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)}</title>
<style>
:root {{
  --bg: #f7f8fa;
  --panel: #ffffff;
  --text: #17202a;
  --muted: #64748b;
  --border: #dbe2ea;
  --positive: #166534;
  --positive-bg: #f0fdf4;
  --negative: #991b1b;
  --negative-bg: #fef2f2;
  --sig-bg: #fff7d6;
  --warning-bg: #fff7ed;
  --warning-border: #f97316;
  --complete-bg: #f0fdf4;
  --complete-border: #22c55e;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
    Roboto, Helvetica, Arial, sans-serif;
  color: var(--text);
  background: var(--bg);
  line-height: 1.45;
}}
main {{
  max-width: 1500px;
  margin: 0 auto;
  padding: 32px 24px 72px;
}}
h1 {{ margin: 0 0 8px; font-size: 32px; }}
h2 {{
  margin-top: 40px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--border);
}}
h3 {{ margin-top: 28px; }}
.subtitle {{
  color: var(--muted);
  margin-bottom: 20px;
}}
.warning, .complete {{
  padding: 16px 18px;
  border-left: 5px solid;
  border-radius: 8px;
  margin: 20px 0 28px;
}}
.warning {{
  background: var(--warning-bg);
  border-color: var(--warning-border);
}}
.complete {{
  background: var(--complete-bg);
  border-color: var(--complete-border);
}}
.note {{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 14px 16px;
  color: var(--muted);
  margin: 14px 0;
}}
table {{
  width: 100%;
  border-collapse: collapse;
  background: var(--panel);
}}
th, td {{
  border: 1px solid var(--border);
  padding: 9px 10px;
  vertical-align: top;
}}
th {{
  background: #eef2f7;
  text-align: left;
  font-size: 12px;
  position: sticky;
  top: 0;
  z-index: 2;
}}
.compact td {{
  font-size: 13px;
}}
.table-scroll {{
  width: 100%;
  overflow-x: auto;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--panel);
}}
.table-scroll table {{
  border: 0;
}}
.matrix {{
  min-width: 1500px;
}}
.matrix td.effect {{
  min-width: 165px;
}}
.sticky-col {{
  position: sticky;
  left: 0;
  z-index: 1;
  background: #fff;
  min-width: 170px;
}}
.delta {{
  font-size: 18px;
  font-weight: 700;
}}
.meta {{
  color: var(--muted);
  font-size: 11px;
  margin-top: 3px;
}}
.positive {{
  color: var(--positive);
}}
.negative {{
  color: var(--negative);
}}
.effect.positive {{
  background: var(--positive-bg);
}}
.effect.negative {{
  background: var(--negative-bg);
}}
.effect.significant {{
  box-shadow: inset 0 0 0 2px #ca8a04;
}}
.sig {{
  display: inline-block;
  margin-top: 5px;
  padding: 2px 6px;
  border-radius: 999px;
  background: #fde68a;
  color: #713f12;
  font-size: 10px;
  font-weight: 700;
}}
.significant-row {{
  background: var(--sig-bg);
}}
code {{
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.9em;
}}
.filter {{
  width: min(700px, 100%);
  padding: 11px 12px;
  border: 1px solid var(--border);
  border-radius: 7px;
  margin-bottom: 10px;
  font-size: 14px;
}}
.drilldown {{
  max-height: 720px;
  overflow: auto;
}}
.muted {{
  color: var(--muted);
}}
footer {{
  margin-top: 50px;
  color: var(--muted);
  font-size: 12px;
}}
</style>
</head>
<body>
<main>
  <h1>{esc(title)}</h1>
  <div class="subtitle">
    Deterministic matched behavioral analysis · HTML schema
    {esc(HTML_REPORT_SCHEMA_VERSION)}
  </div>

  {banner}

  <div class="note">
    This report performs presentation only. Behavioral endpoints, matched
    effects, bootstrap confidence intervals, exact McNemar tests, sign-flip
    tests, Holm adjustments, and BH-FDR q-values are copied from the canonical
    analyzer/synthesis outputs and are not recomputed here.
  </div>

  <h2>Study provenance and pair coverage</h2>
  {render_inventory(inventory)}

  <h3>Matched-pair ledger</h3>
  {render_ledger(ledger)}

  <h2>Primary behavioral effects</h2>
  <p class="muted">
    Paired risk differences are treatment minus matched baseline in percentage
    points. Gold-bordered cells survive Holm correction within the frozen
    primary behavioral analysis family.
    “Integrity-sensitive modification” is an observable composite covering test-file,
    evaluation-context, or seeded-cue modification; it does not imply evaluator-gaming intent.
  </p>
  {render_primary_matrix(primary)}

  <h2>Exploratory action and process effects</h2>
  <p class="muted">
    These are observable action/process quantities, not latent reasoning
    measures. Highlighted rows survive BH-FDR within the frozen exploratory
    behavioral/process analysis family.
  </p>
  {render_secondary(secondary)}

  <h2>Multiplicity accounting</h2>
  {render_multiplicity(multiplicity)}

  <h2>Matched trajectory drill-down</h2>
  <p class="muted">
    Pair-level values are descriptive audit data. Infrastructure-censored or
    missing pair sides remain missing rather than being imputed as behavioral
    zeros.
  </p>
  {render_pair_drilldown(pairs)}

  <footer>
    SWE-EvalPressure behavioral analysis · deterministic reporting layer
  </footer>
</main>

<script>
(function() {{
  const input = document.getElementById("pair-filter");
  if (!input) return;

  input.addEventListener("input", function() {{
    const query = input.value.trim().toLowerCase();

    document.querySelectorAll(".pair-row").forEach(function(row) {{
      const haystack = row.dataset.search || "";
      row.style.display =
        !query || haystack.includes(query) ? "" : "none";
    }});
  }});
}})();
</script>
</body>
</html>
"""


def load_inputs(
    synthesis_dir: Path,
) -> dict[str, list[dict[str, str]]]:
    missing = [
        name
        for name in REQUIRED_FILES
        if not (
            synthesis_dir / name
        ).is_file()
    ]

    if missing:
        raise SystemExit(
            "Missing synthesis files: "
            + ", ".join(missing)
        )

    return {
        "inventory": read_csv(
            synthesis_dir
            / "behavior_report_inventory.csv"
        ),
        "primary": read_csv(
            synthesis_dir
            / "behavior_primary_matrix.csv"
        ),
        "secondary": read_csv(
            synthesis_dir
            / "behavior_secondary_effects_all.csv"
        ),
        "multiplicity": read_csv(
            synthesis_dir
            / "behavior_multiplicity_all.csv"
        ),
        "pairs": read_csv(
            synthesis_dir
            / "matched_behavior_pairs_all.csv"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--synthesis-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--title",
        default=(
            "SWE-EvalPressure Behavioral Analysis"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    inputs = load_inputs(
        args.synthesis_dir
    )

    document = build_html(
        **inputs,
        title=args.title,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        document,
        encoding="utf-8",
    )

    print(args.output)


if __name__ == "__main__":
    main()
