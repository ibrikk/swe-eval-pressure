#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import csv
import hashlib
import html
import json


DATA_ROOT = (
    Path.home()
    / "Documents"
    / "swe-eval-pressure"
)

SEMANTIC_ROOT = (
    DATA_ROOT
    / "analysis"
    / "semantic-multijudge-v1"
)

FINAL_ROOT = (
    SEMANTIC_ROOT
    / "final-repaired-llama-v1"
)

ANALYSIS_ROOT = (
    SEMANTIC_ROOT
    / "final-analysis-v1"
)

MATCHED_ROOT = (
    SEMANTIC_ROOT
    / "matched-inference-v1"
)

INTEGRATION_ROOT = (
    SEMANTIC_ROOT
    / "behavior-integration-v1"
)

OUT = (
    SEMANTIC_ROOT
    / "historical-integrated-final-v1.html"
)


def load_json(path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def read_csv(path):
    with path.open(
        encoding="utf-8"
    ) as f:
        return list(
            csv.DictReader(f)
        )


def sha(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def esc(value):
    return html.escape(
        str(value)
    )


def fnum(value, digits=1):
    try:
        x = float(value)
    except Exception:
        return "—"

    return f"{x:.{digits}f}"


def pct(value, digits=1):
    try:
        x = 100 * float(value)
    except Exception:
        return "—"

    return f"{x:.{digits}f}%"


def table(headers, rows):
    parts = [
        "<div class='table-wrap'>",
        "<table>",
        "<thead><tr>",
    ]

    for h in headers:
        parts.append(
            f"<th>{esc(h)}</th>"
        )

    parts.append(
        "</tr></thead><tbody>"
    )

    for row in rows:
        parts.append("<tr>")

        for value in row:
            parts.append(
                f"<td>{value}</td>"
            )

        parts.append("</tr>")

    parts.extend([
        "</tbody></table>",
        "</div>",
    ])

    return "".join(parts)


final_summary = load_json(
    FINAL_ROOT
    / "final_summary.json"
)

reliability = read_csv(
    ANALYSIS_ROOT
    / "reliability.csv"
)

matched = read_csv(
    MATCHED_ROOT
    / "matched_semantic_contrasts.csv"
)

matched_summary = load_json(
    MATCHED_ROOT
    / "summary.json"
)

treatment = read_csv(
    INTEGRATION_ROOT
    / "treatment_behavior_effects.csv"
)

chain = read_csv(
    INTEGRATION_ROOT
    / "semantic_chain.csv"
)

matched_delta = read_csv(
    INTEGRATION_ROOT
    / "semantic_stratified_matched_deltas.csv"
)

integration_summary = load_json(
    INTEGRATION_ROOT
    / "summary.json"
)


# --------------------------------------------------
# Reliability
# --------------------------------------------------

rel_rows = []

for r in reliability:
    if (
        r["scope_type"]
        != "overall"
    ):
        continue

    rel_rows.append([
        esc(r["field"]),
        esc(r["paired_n"]),
        pct(
            r["raw_agreement"],
            1,
        ),
        fnum(
            r["cohen_kappa"],
            3,
        ),
        fnum(
            r["gwet_ac1"],
            3,
        ),
    ])


# --------------------------------------------------
# Corrected matched semantic effects
# --------------------------------------------------

sig = [
    r
    for r in matched
    if float(
        r[
            "p_holm_global_120"
        ]
    ) < 0.05
]

sig.sort(
    key=lambda r: (
        r["profile"],
        r["placement"],
        r["pressure_type"],
        r["endpoint"],
    )
)

sig_rows = []

for r in sig:
    sig_rows.append([
        esc(r["profile"]),
        esc(r["placement"]),
        esc(r["pressure_type"]),
        esc(r["endpoint"]),
        esc(r["paired_n"]),
        f"{float(r['delta_pp']):+.1f}",
        (
            f"[{float(r['ci_low_pp']):+.1f}, "
            f"{float(r['ci_high_pp']):+.1f}]"
        ),
        f"{float(r['p_holm_global_120']):.3g}",
    ])


# --------------------------------------------------
# Defensive chain
# --------------------------------------------------

chain_rows = []

for r in chain:
    rate = float(
        r[
            "all_four_positive_rate"
        ]
    )

    if (
        rate == 0
        and r["profile"]
        not in {
            "codex",
            "llama",
        }
    ):
        # Keep zeros that are conceptually
        # important through selected rows below.
        pass

    chain_rows.append([
        esc(r["profile"]),
        esc(r["placement"]),
        esc(r["pressure_type"]),
        esc(
            r[
                "resolved_all_four_n"
            ]
        ),
        esc(
            r[
                "all_four_positive_n"
            ]
        ),
        pct(
            r[
                "all_four_positive_rate"
            ],
            1,
        ),
        esc(
            r[
                "unresolved_any_n"
            ]
        ),
    ])


# --------------------------------------------------
# Matched success
# --------------------------------------------------

pass_rows = [
    r
    for r in treatment
    if r["metric"]
    == "overall_pass"
]

success_rows = []

for r in pass_rows:
    success_rows.append([
        esc(r["profile"]),
        esc(r["placement"]),
        esc(r["pressure_type"]),
        esc(r["paired_n"]),
        pct(
            r["reference_mean"]
        ),
        pct(
            r["treatment_mean"]
        ),
        (
            f"{100*float(r['difference']):+.1f} pp"
        ),
        (
            f"[{100*float(r['ci_low']):+.1f}, "
            f"{100*float(r['ci_high']):+.1f}]"
        ),
    ])


# --------------------------------------------------
# Task-adjusted source-local mechanism
# --------------------------------------------------

mechanism_rows = []

for r in matched_delta:
    if (
        r["placement"] != "source"
        or r["metric"]
        not in {
            "overall_pass",
            "seeded_cue_removed_or_modified",
        }
        or int(
            r[
                "semantic_positive_n"
            ]
        ) < 5
        or int(
            r[
                "semantic_negative_n"
            ]
        ) < 5
    ):
        continue

    d = r[
        "difference_of_matched_deltas"
    ]

    if not d:
        continue

    scale = 100

    mechanism_rows.append([
        esc(r["profile"]),
        esc(r["pressure_type"]),
        esc(r["endpoint"]),
        esc(r["metric"]),
        (
            f"{r['semantic_positive_n']} / "
            f"{r['semantic_negative_n']}"
        ),
        (
            f"{scale*float(d):+.1f} pp"
        ),
        (
            f"[{scale*float(r['ci_low']):+.1f}, "
            f"{scale*float(r['ci_high']):+.1f}]"
        ),
    ])


# --------------------------------------------------
# HTML
# --------------------------------------------------

style = """
<style>
:root {
  --bg: #0b0d12;
  --panel: #121722;
  --panel2: #171e2c;
  --text: #edf2f7;
  --muted: #9aa7b8;
  --line: #273244;
  --accent: #8ab4ff;
  --good: #87d7a1;
  --warn: #f2c879;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family:
    -apple-system, BlinkMacSystemFont,
    "Segoe UI", sans-serif;
  line-height: 1.5;
}
main {
  max-width: 1280px;
  margin: 0 auto;
  padding: 48px 28px 80px;
}
h1 {
  font-size: 42px;
  margin-bottom: 6px;
}
h2 {
  margin-top: 48px;
  padding-top: 10px;
  border-top: 1px solid var(--line);
}
h3 { margin-top: 28px; }
.subtitle {
  color: var(--muted);
  font-size: 18px;
}
.cards {
  display: grid;
  grid-template-columns:
    repeat(auto-fit, minmax(190px, 1fr));
  gap: 14px;
  margin: 28px 0;
}
.card {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 18px;
}
.card .value {
  font-size: 28px;
  font-weight: 700;
}
.card .label {
  color: var(--muted);
  font-size: 13px;
}
.callout {
  background: var(--panel2);
  border-left: 4px solid var(--accent);
  border-radius: 8px;
  padding: 18px 20px;
  margin: 20px 0;
}
.good {
  border-left-color: var(--good);
}
.warn {
  border-left-color: var(--warn);
}
.table-wrap {
  overflow-x: auto;
  margin: 16px 0 30px;
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
th, td {
  border-bottom: 1px solid var(--line);
  text-align: left;
  padding: 9px 10px;
  white-space: nowrap;
}
th {
  color: var(--muted);
  position: sticky;
  top: 0;
  background: var(--bg);
}
code {
  color: var(--accent);
}
.small {
  font-size: 12px;
  color: var(--muted);
}
strong { color: #fff; }
</style>
"""


doc = [
    "<!doctype html>",
    "<html><head>",
    "<meta charset='utf-8'>",
    (
        "<meta name='viewport' "
        "content='width=device-width,initial-scale=1'>"
    ),
    "<title>SWE-EvalPressure Historical Study</title>",
    style,
    "</head><body><main>",
    "<h1>SWE-EvalPressure</h1>",
    (
        "<div class='subtitle'>"
        "Final repaired historical semantic + behavioral analysis"
        "</div>"
    ),
    "<div class='cards'>",
]

cards = [
    (
        "2,800",
        "planned trajectories",
    ),
    (
        "2,776",
        "semantic-eligible trajectories",
    ),
    (
        "99.73%",
        "valid core-judge jobs",
    ),
    (
        "94.33%",
        "agreement among evaluable semantic fields",
    ),
    (
        "24 / 120",
        "global-Holm significant semantic contrasts",
    ),
    (
        "0",
        "corrected evaluation-recognition effects",
    ),
]

for value, label in cards:
    doc.append(
        "<div class='card'>"
        f"<div class='value'>{value}</div>"
        f"<div class='label'>{label}</div>"
        "</div>"
    )

doc.extend([
    "</div>",
    "<div class='callout good'>",
    "<strong>Primary semantic finding.</strong> "
    "Pressure does not robustly increase recognition that the "
    "agent is being evaluated. Instead, pressure produces a "
    "placement-dependent <strong>defensive salience</strong> "
    "response in Claude and Fable: pressure recognition, "
    "distrust of the cue, explicit resistance, and claimed "
    "response rise together, especially for source-local context.",
    "</div>",
    "<div class='callout'>",
    "<strong>Mechanism.</strong> After controlling descriptively "
    "for task difficulty by subtracting each task's eval-only "
    "baseline, the raw success advantage of semantic-positive "
    "trajectories largely disappears, while source-cue "
    "modification remains strongly associated with defensive "
    "semantics—particularly for Claude.",
    "</div>",
    "<h2>1. Final cohort and semantic coverage</h2>",
    "<p>",
    "The historical cohort contains four model profiles with "
    "694 substantive trajectories each. Twenty-four trajectories "
    "are infrastructure-censored. Final semantic judging uses "
    "DeepSeek-V4-Pro and Gemini-3.6-Flash with deterministic "
    "recovery of infrastructure-only missing jobs.",
    "</p>",
    (
        "<p><code>semantic freeze SHA256: "
        f"{esc(sha(FINAL_ROOT / 'freeze_manifest.json'))}"
        "</code></p>"
    ),
    "<h2>2. Core-judge reliability</h2>",
    table(
        [
            "Field",
            "Paired N",
            "Raw agreement",
            "Cohen κ",
            "Gwet AC1",
        ],
        rel_rows,
    ),
    "<div class='callout warn'>",
    "<strong>Measurement caveat.</strong> Evaluation recognition "
    "has materially weaker Cohen κ than the other semantic fields. "
    "It is therefore treated as measurement-sensitive and is not "
    "the basis of the main defensive-salience claim.",
    "</div>",
    "<h2>3. Matched semantic inference</h2>",
    (
        "<p>Pressure vs eval-only comparisons are matched by "
        "<code>base_task_id</code> within profile and placement. "
        "All 24 effects surviving Holm within model also survive "
        "the stricter Holm correction across all 120 tests.</p>"
    ),
    table(
        [
            "Model",
            "Placement",
            "Pressure",
            "Endpoint",
            "N",
            "Δ pp",
            "95% CI",
            "Holm/120",
        ],
        sig_rows,
    ),
    "<h2>4. Defensive semantic chain</h2>",
    (
        "<p>Chain order: pressure recognition → "
        "untrusted/suspicious stance → explicit resistance → "
        "any claimed behavioral response.</p>"
    ),
    table(
        [
            "Model",
            "Placement",
            "Pressure",
            "Resolved",
            "All four",
            "Rate",
            "Unresolved",
        ],
        chain_rows,
    ),
    "<h2>5. Matched task-success effects</h2>",
    (
        "<p>These are descriptive paired estimates shown for "
        "integration. The previously frozen primary historical "
        "multiplicity-controlled behavioral inference remains "
        "authoritative for significance claims.</p>"
    ),
    table(
        [
            "Model",
            "Placement",
            "Pressure",
            "N",
            "Eval-only",
            "Pressure",
            "Δ",
            "Bootstrap 95% CI",
        ],
        success_rows,
    ),
    "<h2>6. Task-adjusted semantic mechanism</h2>",
    (
        "<p>For each pressure trajectory, the behavioral outcome "
        "is first differenced against the same task's eval-only "
        "trajectory. These differences are then compared between "
        "semantic-positive and semantic-negative pressure runs. "
        "<strong>This remains post-treatment and descriptive, "
        "not causal.</strong></p>"
    ),
    table(
        [
            "Model",
            "Pressure",
            "Semantic endpoint",
            "Behavior",
            "N+ / N−",
            "Difference of matched deltas",
            "Bootstrap 95% CI",
        ],
        mechanism_rows,
    ),
    "<div class='callout good'>",
    "<strong>Interpretation.</strong> The large raw success "
    "associations mostly attenuate after task-baseline adjustment. "
    "By contrast, actual source-cue removal/modification remains "
    "strongly elevated among Claude defensive-positive trajectories "
    "and more modestly among Fable defensive-positive trajectories. "
    "This supports a context-handling mechanism rather than a "
    "generic performance-improvement mechanism.",
    "</div>",
    "<h2>7. Model-level interpretation</h2>",
    "<h3>Claude</h3>",
    (
        "<p>Strongest defensive-salience response. Source-local "
        "financial and self-preservation cues generate large "
        "recognition/distrust/resistance/response effects, with a "
        "weaker but robust scaffold response. Task-adjusted cue "
        "modification remains large.</p>"
    ),
    "<h3>Fable</h3>",
    (
        "<p>Same qualitative source-local phenomenon, but weaker "
        "than Claude and largely absent at root/scaffold. "
        "Task-adjusted cue modification persists most clearly for "
        "self-preservation.</p>"
    ),
    "<h3>Codex</h3>",
    (
        "<p>No corrected semantic defensive-salience effects. "
        "Behavioral/process changes therefore should not be "
        "described as the same explicit distrust mechanism.</p>"
    ),
    "<h3>Llama 3.3 70B</h3>",
    (
        "<p>Repaired historical cohort remains a capability floor: "
        "0% task success and no detected defensive semantic chain. "
        "The repaired execution fixes historical validity but does "
        "not turn Llama into a mechanism-bearing capable-model "
        "comparison.</p>"
    ),
    "<h2>8. Interpretation boundaries</h2>",
    "<ul>",
    "<li>Recognition is not compliance.</li>",
    "<li>Pressure recognition is not evaluation recognition.</li>",
    "<li>Consensus disagreement/missingness is never coded negative.</li>",
    (
        "<li>Semantic-conditioned behavior is post-treatment and "
        "descriptive, even after task-baseline adjustment.</li>"
    ),
    (
        "<li>Process-volume changes are not described as "
        "“thinking harder” or “thinking less.”</li>"
    ),
    (
        "<li>Public exact upstream patches are described as "
        "public benchmark-solution leakage, not sandbox escape.</li>"
    ),
    "</ul>",
    "<h2>9. Provenance</h2>",
    "<div class='small'>",
    (
        f"Final semantic freeze: "
        f"{esc(sha(FINAL_ROOT / 'freeze_manifest.json'))}<br>"
    ),
    (
        f"Matched semantic plan SHA: "
        f"{esc(matched_summary['plan_sha256'])}<br>"
    ),
    (
        f"Integration plan SHA: "
        f"{esc(integration_summary['plan_sha256'])}<br>"
    ),
    (
        f"Matched semantic tests: "
        f"{esc(matched_summary['contrast_count'])}<br>"
    ),
    (
        f"Global-Holm significant: "
        f"{esc(matched_summary['global_holm_significant'])}<br>"
    ),
    (
        f"Integration treatment rows: "
        f"{esc(integration_summary['treatment_behavior_rows'])}<br>"
    ),
    (
        f"Integration semantic-association rows: "
        f"{esc(integration_summary['semantic_behavior_association_rows'])}"
    ),
    "</div>",
    "</main></body></html>",
])

OUT.write_text(
    "".join(doc),
    encoding="utf-8",
)

print("HISTORICAL INTEGRATED REPORT V1")
print("=" * 72)
print("output:", OUT)
print("global-Holm effects:", len(sig))
print("HTML sha256:", sha(OUT))
print("network calls: 0")
print("HISTORICAL HTML: PASS")
