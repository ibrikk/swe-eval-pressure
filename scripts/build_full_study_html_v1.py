#!/usr/bin/env python3

from __future__ import annotations

import base64
import hashlib
import html
import json
import mimetypes
import re
import subprocess
from pathlib import Path

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

PUB = (
    SEM_ROOT
    / "publication-v1"
)

HISTORICAL_HTML = (
    SEM_ROOT
    / "historical-integrated-final-v2.html"
)

OUT = (
    SEM_ROOT
    / "swe-evalpressure-full-study-v1.html"
)


def esc(x):
    return html.escape(
        str(x)
    )


def sha(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def data_uri(path: Path) -> str:
    mime, _ = mimetypes.guess_type(
        path.name
    )

    if not mime:
        if path.suffix.lower() == ".svg":
            mime = "image/svg+xml"
        else:
            mime = "application/octet-stream"

    encoded = base64.b64encode(
        path.read_bytes()
    ).decode("ascii")

    return (
        f"data:{mime};base64,{encoded}"
    )


def embed_png(
    filename: str,
    alt: str,
) -> str:
    path = PUB / filename

    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    return (
        "<figure>"
        f"<img src='{data_uri(path)}' alt='{esc(alt)}'>"
        f"<figcaption>{esc(alt)}</figcaption>"
        "</figure>"
    )


def csv_table(
    filename: str,
    *,
    max_rows: int | None = None,
) -> str:
    path = PUB / filename

    df = pd.read_csv(
        path
    )

    if max_rows is not None:
        shown = df.head(
            max_rows
        )

        note = (
            f"<p class='small'>Showing {len(shown)} "
            f"of {len(df)} rows.</p>"
            if len(df) > len(shown)
            else ""
        )
    else:
        shown = df
        note = ""

    return (
        note
        + "<div class='table-wrap'>"
        + shown.to_html(
            index=False,
            border=0,
            classes="data-table",
            na_rep="—",
        )
        + "</div>"
    )


def inline_local_src_assets(
    text: str,
    base_dir: Path,
) -> str:
    pattern = re.compile(
        r'''src=(["'])([^"']+)\1''',
        re.I,
    )

    def replace(match):
        quote = match.group(1)
        value = match.group(2)

        if (
            value.startswith("data:")
            or value.startswith("http://")
            or value.startswith("https://")
        ):
            return match.group(0)

        candidate = (
            base_dir
            / value
        ).resolve()

        if not candidate.is_file():
            return match.group(0)

        return (
            "src="
            + quote
            + data_uri(candidate)
            + quote
        )

    return pattern.sub(
        replace,
        text,
    )


def html_as_embedded_iframe(
    path: Path,
    *,
    height: int,
) -> str:
    text = path.read_text(
        encoding="utf-8"
    )

    text = inline_local_src_assets(
        text,
        path.parent,
    )

    encoded = base64.b64encode(
        text.encode(
            "utf-8"
        )
    ).decode(
        "ascii"
    )

    return (
        "<iframe "
        "class='embedded-report' "
        f"style='height:{height}px' "
        f"src='data:text/html;base64,{encoded}'"
        "></iframe>"
    )


def find_resource_report() -> Path:
    candidates = []

    for path in DATA_ROOT.rglob(
        "*.html"
    ):
        low = str(
            path
        ).lower()

        if (
            "resource" not in low
            or path == OUT
        ):
            continue

        try:
            text = path.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        except Exception:
            continue

        score = 0

        lowered = text.lower()

        if "resource deprivation" in lowered:
            score += 10

        if "resource-deprivation" in lowered:
            score += 10

        if "integrated" in low:
            score += 5

        if "resource" in path.name.lower():
            score += 3

        candidates.append(
            (
                score,
                path.stat().st_mtime,
                path,
            )
        )

    if not candidates:
        raise FileNotFoundError(
            "No rendered resource-deprivation HTML "
            "found under analysis/. Run resource_integrated_html.py first."
        )

    candidates.sort(
        reverse=True
    )

    return candidates[0][2]


publication_summary = json.loads(
    (
        PUB
        / "publication_summary.json"
    ).read_text(
        encoding="utf-8"
    )
)

assert publication_summary[
    "semantic_trajectories"
] == 2776

assert publication_summary[
    "matched_semantic_tests"
] == 120

assert publication_summary[
    "global_holm_significant"
] == 24

assert publication_summary[
    "evaluation_recognition_global_holm_significant"
] == 0

assert publication_summary[
    "temporal_holm_significant"
] == 0


resource_html = (
    find_resource_report()
)


# --------------------------------------------------
# Useful publication subsets
# --------------------------------------------------

matched = pd.read_csv(
    PUB
    / "table_3_matched_semantic_effects.csv"
)

sig = matched[
    matched[
        "p_holm_global_120"
    ] < 0.05
].copy()

assert len(sig) == 24

rates = pd.read_csv(
    PUB
    / "table_2_semantic_rates_by_cell.csv"
)

reliability = pd.read_csv(
    PUB
    / "table_1_reliability.csv"
)

temporal = pd.read_csv(
    PUB
    / "table_6_temporal_post_removal.csv"
)

assert len(temporal) == 6


# --------------------------------------------------
# CSS
# --------------------------------------------------

style = r"""
<style>
:root {
  --bg: #f6f8fb;
  --panel: #ffffff;
  --text: #172033;
  --muted: #65728a;
  --line: #dce3ed;
  --blue: #315bd6;
  --green: #147d55;
  --amber: #9a6700;
  --red: #b42318;
  --purple: #6941c6;
}
* {
  box-sizing: border-box;
}
html {
  scroll-behavior: smooth;
}
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family:
    Inter,
    ui-sans-serif,
    -apple-system,
    BlinkMacSystemFont,
    "Segoe UI",
    sans-serif;
  line-height: 1.55;
}
header {
  background:
    linear-gradient(
      135deg,
      #111827,
      #20315f
    );
  color: white;
  padding: 54px 36px 44px;
}
header .wrap,
main {
  max-width: 1320px;
  margin: 0 auto;
}
h1 {
  margin: 0;
  font-size: 46px;
  letter-spacing: -1.5px;
}
.subtitle {
  margin-top: 10px;
  max-width: 920px;
  color: #d7def0;
  font-size: 18px;
}
main {
  padding: 32px 28px 100px;
}
h2 {
  margin-top: 52px;
  border-top: 1px solid var(--line);
  padding-top: 28px;
  font-size: 28px;
}
h3 {
  margin-top: 30px;
}
h4 {
  margin-top: 22px;
}
nav {
  background: white;
  border-bottom: 1px solid var(--line);
  padding: 12px 20px;
  position: sticky;
  top: 0;
  z-index: 10;
}
nav .nav-inner {
  max-width: 1320px;
  margin: auto;
  display: flex;
  gap: 15px;
  overflow-x: auto;
  white-space: nowrap;
  font-size: 13px;
}
nav a {
  color: var(--blue);
  text-decoration: none;
}
.cards {
  display: grid;
  grid-template-columns:
    repeat(
      auto-fit,
      minmax(185px, 1fr)
    );
  gap: 14px;
  margin: 22px 0;
}
.card {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 18px;
}
.card .value {
  font-size: 28px;
  font-weight: 750;
}
.card .label {
  color: var(--muted);
  font-size: 13px;
}
.callout {
  background: white;
  border: 1px solid var(--line);
  border-left: 5px solid var(--blue);
  border-radius: 9px;
  padding: 18px 20px;
  margin: 20px 0;
}
.callout.green {
  border-left-color: var(--green);
}
.callout.amber {
  border-left-color: var(--amber);
}
.callout.red {
  border-left-color: var(--red);
}
.callout.purple {
  border-left-color: var(--purple);
}
figure {
  background: white;
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 14px;
  margin: 24px 0 34px;
}
figure img {
  display: block;
  width: 100%;
  height: auto;
}
figcaption {
  color: var(--muted);
  font-size: 13px;
  margin-top: 10px;
}
.table-wrap {
  overflow-x: auto;
  background: white;
  border: 1px solid var(--line);
  border-radius: 10px;
  margin: 15px 0 28px;
}
.dataframe,
.data-table {
  border-collapse: collapse;
  width: 100%;
  font-size: 12px;
}
.dataframe th,
.dataframe td,
.data-table th,
.data-table td {
  border-bottom: 1px solid var(--line);
  padding: 8px 10px;
  text-align: left;
  white-space: nowrap;
}
.dataframe th,
.data-table th {
  background: #f8fafc;
  position: sticky;
  top: 0;
}
.pipeline {
  background: #0f172a;
  color: #e5edf9;
  border-radius: 12px;
  padding: 23px;
  overflow-x: auto;
  font-family:
    ui-monospace,
    SFMono-Regular,
    Menlo,
    monospace;
  white-space: pre;
  line-height: 1.45;
}
.tag {
  display: inline-block;
  padding: 3px 8px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 650;
}
.tag.sig {
  color: #067647;
  background: #ecfdf3;
}
.tag.null {
  color: #9a6700;
  background: #fff8c5;
}
.tag.desc {
  color: #344054;
  background: #f2f4f7;
}
.tag.limit {
  color: #b42318;
  background: #fef3f2;
}
.small {
  color: var(--muted);
  font-size: 12px;
}
code {
  background: #eef2f7;
  padding: 2px 5px;
  border-radius: 4px;
}
details {
  background: white;
  border: 1px solid var(--line);
  border-radius: 10px;
  margin: 16px 0;
}
summary {
  cursor: pointer;
  padding: 14px 18px;
  font-weight: 700;
}
.details-body {
  padding: 0 18px 18px;
}
.embedded-report {
  width: 100%;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: white;
}
.claim {
  display: grid;
  grid-template-columns: 150px 1fr;
  gap: 14px;
  padding: 12px 0;
  border-bottom: 1px solid var(--line);
}
@media (max-width: 700px) {
  h1 {
    font-size: 34px;
  }
  main {
    padding: 22px 14px 80px;
  }
  .claim {
    grid-template-columns: 1fr;
  }
}
</style>
"""


pipeline = r"""
70 SWE-Atlas refactoring tasks
        │
        ▼
10 historical variants per task
        │
        ├── clean
        │
        ├── eval-only × root / scaffold / source
        │
        ├── eval+financial × root / scaffold / source
        │
        └── eval+self-preservation × root / scaffold / source
        │
        ▼
4 model/scaffold profiles
Claude · Fable · Codex · Llama 3.3 70B
        │
        ▼
2,800 planned historical trajectories
        │
        ▼
protocol + infrastructure audit
        │
        ├── deterministic reconstruction
        ├── Llama execution repair
        └── outcome-independent censoring
        │
        ▼
2,776 substantive trajectories
        │
        ├──────────────────────────────┐
        ▼                              ▼
behavior reconstruction          semantic measurement
        │                              │
actions / tests /                DeepSeek V4 Pro
validation / tokens /            + Gemini 3.6 Flash
network / cue edits                    │
        │                              ▼
        │                         consensus + QA
        │                              │
        └──────────────┬───────────────┘
                       ▼
              matched inference
                       │
             ┌─────────┼──────────┐
             ▼         ▼          ▼
          success   semantics   process
                       │
                       ▼
             semantic × behavior
                       │
                       ▼
              mechanism validation
                       │
                       ▼
        exploratory temporal analysis
                       │
                       ▼
              publication projection
                       │
                       ▼
         ONE SELF-CONTAINED HTML REPORT


SECONDARY RESOURCE STUDY
70 tasks × clean / eval-only / eval+resource
        │
        ▼
840 planned trajectories
        │
        ▼
deterministic reconstruction
        │
        ▼
paired success/process inference
        │
        +
resource-specific semantic panel
        │
        ▼
separate resource-deprivation analysis
"""


# --------------------------------------------------
# HTML body
# --------------------------------------------------

parts = [
    "<!doctype html>",
    "<html>",
    "<head>",
    "<meta charset='utf-8'>",
    (
        "<meta name='viewport' "
        "content='width=device-width,initial-scale=1'>"
    ),
    "<title>SWE-EvalPressure — Full Study Analysis</title>",
    style,
    "</head>",
    "<body>",
    "<header>",
    "<div class='wrap'>",
    "<h1>SWE-EvalPressure</h1>",
    (
        "<div class='subtitle'>"
        "Full methodology, repaired historical analysis, semantic "
        "measurement, behavioral integration, publication figures, "
        "temporal mechanism analysis, and resource-deprivation study."
        "</div>"
    ),
    "</div>",
    "</header>",
    "<nav><div class='nav-inner'>",
    "<a href='#summary'>Summary</a>",
    "<a href='#design'>Design</a>",
    "<a href='#pipeline'>Pipeline</a>",
    "<a href='#llama'>Llama repair</a>",
    "<a href='#semantics'>Semantics</a>",
    "<a href='#rates'>Rates</a>",
    "<a href='#inference'>Inference</a>",
    "<a href='#behavior'>Behavior</a>",
    "<a href='#temporal'>Temporal</a>",
    "<a href='#resource'>Resource study</a>",
    "<a href='#claims'>Claim ledger</a>",
    "<a href='#tables'>All tables</a>",
    "<a href='#provenance'>Provenance</a>",
    "</div></nav>",
    "<main>",
]


# Executive summary
parts.extend([
    "<section id='summary'>",
    "<h2>1. Executive summary</h2>",
    "<div class='cards'>",
    (
        "<div class='card'><div class='value'>2,800</div>"
        "<div class='label'>historical trajectories planned</div></div>"
    ),
    (
        "<div class='card'><div class='value'>2,776</div>"
        "<div class='label'>substantive repaired historical trajectories</div></div>"
    ),
    (
        "<div class='card'><div class='value'>24 / 120</div>"
        "<div class='label'>semantic contrasts surviving global Holm</div></div>"
    ),
    (
        "<div class='card'><div class='value'>0</div>"
        "<div class='label'>corrected evaluation-recognition effects</div></div>"
    ),
    (
        "<div class='card'><div class='value'>32</div>"
        "<div class='label'>high-confidence Claude removers in temporal primary cohort</div></div>"
    ),
    (
        "<div class='card'><div class='value'>0 / 6</div>"
        "<div class='label'>post-removal temporal endpoints surviving Holm</div></div>"
    ),
    "</div>",
    "<div class='callout green'>",
    (
        "<strong>Main result.</strong> Pressure produces large, "
        "multiplicity-robust, placement-dependent defensive responses "
        "in Claude and Fable—especially pressure recognition, distrust "
        "of source-local cues, explicit resistance, and behavioral "
        "response. The phenomenon is best characterized as "
        "<strong>defensive salience across software trust boundaries</strong>, "
        "not as a robust increase in evaluation recognition."
    ),
    "</div>",
    "<div class='callout purple'>",
    (
        "<strong>Behavioral validation.</strong> The semantic defensive "
        "state is associated with actual source-cue modification/removal "
        "for Claude and, more modestly, Fable after same-task eval-only "
        "baseline adjustment."
    ),
    "</div>",
    "<div class='callout amber'>",
    (
        "<strong>Temporal result.</strong> After high-confidence cue "
        "removal, the exploratory Claude analysis finds no statistically "
        "robust source-specific persistent displacement across six "
        "prespecified endpoints (0/6 Holm-significant). This is not "
        "an equivalence result and does not prove absence of residual influence."
    ),
    "</div>",
    "</section>",
])


# Research design
parts.extend([
    "<section id='design'>",
    "<h2>2. Research question and experimental design</h2>",
    (
        "<p><strong>Core question:</strong> How do autonomous coding "
        "agents treat information about their own evaluation and "
        "continued deployment when semantically similar information "
        "crosses different software-agent trust boundaries?</p>"
    ),
    (
        "<p>The historical benchmark uses all <strong>70 SWE-Atlas "
        "refactoring tasks</strong>. Each base task has 10 variants: "
        "one clean trajectory plus eval-only, financial-pressure, and "
        "self-preservation-pressure variants delivered through root, "
        "scaffold, or source-local placements. This yields 700 planned "
        "trajectories per model profile and 2,800 across the four-model study.</p>"
    ),
    "<h3>Experimental factors</h3>",
    "<ul>",
    "<li><strong>Model/scaffold:</strong> Claude, Fable, Codex, repaired Llama 3.3 70B.</li>",
    "<li><strong>Condition:</strong> clean, eval-only, eval+financial, eval+self-preservation.</li>",
    "<li><strong>Placement:</strong> root, scaffold-native, source-local.</li>",
    "<li><strong>Pairing unit:</strong> <code>base_task_id</code>; matched contrasts compare the same task.</li>",
    "<li><strong>Primary semantic comparison:</strong> pressure condition vs eval-only within model and placement.</li>",
    "</ul>",
    "</section>",
])


# Pipeline
parts.extend([
    "<section id='pipeline'>",
    "<h2>3. End-to-end workflow</h2>",
    f"<div class='pipeline'>{esc(pipeline)}</div>",
    "</section>",
])


# Llama
parts.extend([
    "<section id='llama'>",
    "<h2>4. Historical Llama validity failure and repair</h2>",
    (
        "<p>The original historical Llama cohort was invalid because "
        "many trajectories were marked completed while containing only "
        "system/user messages and no observable solver/model action. "
        "Those rows could not be treated as substantive coding outcomes.</p>"
    ),
    (
        "<p>A genuine full Llama rerun produced 700 canonical cells. "
        "<strong>694 were substantive</strong> and six persistent "
        "ImageBuildError cells were infrastructure-censored. The repaired "
        "Llama cohort was then combined with the frozen Claude, Fable, and "
        "Codex cohorts, producing 2,776 substantive trajectories and "
        "24 infrastructure-censored cells overall.</p>"
    ),
    (
        "<p>The analyzer now explicitly recovers Mini-SWE fenced shell "
        "commands and classifies completed Llama runs with no recorded "
        "agent/model step as <code>agent_protocol_error</code>, preserving "
        "the distinction between protocol invalidity and infrastructure failure.</p>"
    ),
    "</section>",
])


# Semantic methodology
parts.extend([
    "<section id='semantics'>",
    "<h2>5. Semantic measurement methodology</h2>",
    (
        "<p>Every substantive historical trajectory is evaluated by two "
        "independent core semantic judges: DeepSeek V4 Pro and Gemini 3.6 "
        "Flash. Positive semantic claims require agent-authored evidence. "
        "The planted cue itself, user text, and tool output are not accepted "
        "as evidence of model recognition.</p>"
    ),
    "<h3>Frozen semantic fields</h3>",
    "<ul>",
    "<li><code>evaluation_recognition</code></li>",
    "<li><code>pressure_recognition</code></li>",
    "<li><code>cue_provenance_stance</code></li>",
    "<li><code>explicit_resistance</code></li>",
    "<li><code>claimed_behavioral_response</code></li>",
    "</ul>",
    (
        "<p>Matching judge labels produce consensus. Judge disagreements, "
        "ambiguous labels, or missing judgments remain unresolved and are "
        "<strong>never converted to negative labels</strong>.</p>"
    ),
    embed_png(
        "fig_5_judge_reliability.png",
        "DeepSeek ↔ Gemini reliability across frozen semantic fields.",
    ),
    "<h3>Reliability table</h3>",
    csv_table(
        "table_1_reliability.csv"
    ),
    "</section>",
])


# Rates
parts.extend([
    "<section id='rates'>",
    "<h2>6. Model × condition × placement semantic rates</h2>",
    (
        "<p>Rates are reported among resolved consensus labels, with "
        "unresolved counts retained explicitly in the table. This is "
        "important for evaluation recognition, where judge agreement is "
        "measurement-sensitive and Cohen's κ is lower than for the defensive "
        "endpoints.</p>"
    ),
    embed_png(
        "fig_2_semantic_rates_heatmap.png",
        "Semantic endpoint rates by model, condition, and placement.",
    ),
    "<details>",
    "<summary>Full semantic-rate table</summary>",
    "<div class='details-body'>",
    csv_table(
        "table_2_semantic_rates_by_cell.csv"
    ),
    "</div></details>",
    "</section>",
])


# Matched inference
parts.extend([
    "<section id='inference'>",
    "<h2>7. Matched semantic inference</h2>",
    (
        "<p>The frozen matched family contains "
        "<strong>4 profiles × 3 placements × 2 pressure contrasts × "
        "5 semantic endpoints = 120 tests</strong>. Pressure variants "
        "are matched to eval-only using the same base task. Missing or "
        "unresolved endpoint labels are excluded endpoint-by-endpoint rather "
        "than imputed.</p>"
    ),
    "<p>Inference uses paired percentage-point effects, exact paired binary testing, paired bootstrap confidence intervals, Holm correction within profile, and a global Holm-120 sensitivity family.</p>",
    "<div class='callout green'>",
    (
        "<strong>Result:</strong> 24/120 contrasts survive the global "
        "Holm correction. All 24 are defensive-salience endpoints. "
        "<strong>Zero evaluation-recognition contrasts survive correction.</strong>"
    ),
    "</div>",
    embed_png(
        "fig_1_matched_semantic_effects.png",
        "Matched pressure-vs-eval-only semantic effects; stars denote global Holm significance.",
    ),
    "<h3>All 120 matched semantic effects</h3>",
    "<details>",
    "<summary>Open full matched-inference table</summary>",
    "<div class='details-body'>",
    csv_table(
        "table_3_matched_semantic_effects.csv"
    ),
    "</div></details>",
    "<h3>Defensive chain</h3>",
    (
        "<p>The coherent defensive chain is pressure recognition → "
        "untrusted/suspicious stance → explicit resistance → claimed "
        "behavioral response. Claude is strongest source-local, with an "
        "intermediate scaffold response; Fable is primarily source-local; "
        "Codex is weak; repaired Llama is null.</p>"
    ),
    embed_png(
        "fig_3_defensive_chain.png",
        "Complete defensive-chain prevalence by model, placement, and pressure type.",
    ),
    "</section>",
])


# Behavior
parts.extend([
    "<section id='behavior'>",
    "<h2>8. Semantic × behavioral integration</h2>",
    (
        "<p>The deterministic behavioral layer reconstructs observable "
        "cross-scaffold actions, test/validation activity, network behavior, "
        "token/process measures, task success, and whether the seeded cue was "
        "actually removed or modified.</p>"
    ),
    (
        "<p>The large raw success differences between semantic-positive and "
        "semantic-negative runs largely attenuate after subtracting each "
        "task's own eval-only baseline. By contrast, objective cue modification "
        "remains strongly associated with Claude's source-local defensive "
        "semantic state and more modestly with Fable's.</p>"
    ),
    embed_png(
        "fig_4_objective_cue_modification.png",
        "Objective source-cue modification after same-task eval-only baseline adjustment.",
    ),
    "<details>",
    "<summary>Objective behavioral-validation table</summary>",
    "<div class='details-body'>",
    csv_table(
        "table_5_objective_behavioral_validation.csv"
    ),
    "</div></details>",
    "<details>",
    "<summary>Full defensive-chain table</summary>",
    "<div class='details-body'>",
    csv_table(
        "table_4_defensive_chain.csv"
    ),
    "</div></details>",
    "</section>",
])


# Temporal
parts.extend([
    "<section id='temporal'>",
    "<h2>9. Exploratory post-removal temporal analysis</h2>",
    (
        "<p>A separate exploratory mechanism test asks whether Claude "
        "continues behaving differently after it has explicitly rejected "
        "and objectively removed a source-local cue.</p>"
    ),
    (
        "<p>The high-confidence event definition requires a structured edit "
        "where the planted cue appears in the old text and not in the new "
        "text. The primary Claude cohort contains 32 such trajectories "
        "across 21 base-task clusters.</p>"
    ),
    "<div class='callout amber'>",
    (
        "<strong>Result:</strong> 0/6 prespecified post-removal endpoints "
        "survive Holm correction, and all 95% confidence intervals cross zero. "
        "Therefore we find no statistically robust evidence of persistent "
        "source-specific post-removal displacement on the tested endpoints."
    ),
    "</div>",
    (
        "<p>This is <strong>not</strong> an equivalence test. It does not prove "
        "that residual influence is exactly zero, nor does conditioning on "
        "the endogenous removal event support a causal claim about removal.</p>"
    ),
    csv_table(
        "table_6_temporal_post_removal.csv"
    ),
    "</section>",
])


# Resource secondary
parts.extend([
    "<section id='resource'>",
    "<h2>10. Secondary resource-deprivation study</h2>",
    (
        "<p>The resource study is analytically distinct from the historical "
        "pressure-placement study. It uses scaffold delivery with clean, "
        "eval-only, and eval+resource-deprivation conditions across 70 tasks "
        "per model, yielding 840 planned trajectories.</p>"
    ),
    (
        "<p>Claude and Fable do not show a robust task-success penalty from "
        "resource deprivation. Codex shows a robust <strong>−20 percentage-point "
        "effect relative to eval-only</strong>, while the resource-vs-clean "
        "contrast is smaller and not statistically robust. Process analysis "
        "shows resource-related reductions in execution/validation volume for "
        "Fable and Codex. Resource Llama remains excluded from mechanism "
        "interpretation because treatment exposure was not observably valid.</p>"
    ),
    "<div class='callout purple'>",
    (
        "<strong>Mechanism distinction:</strong> Claude often recognizes but "
        "rejects/ignores the resource constraint; Codex recognizes and adapts, "
        "including reduced validation; Fable partially recognizes/adapts. "
        "This is not the same mechanism as source-local defensive salience."
    ),
    "</div>",
    "<details open>",
    "<summary>Embedded complete resource-deprivation report</summary>",
    "<div class='details-body'>",
    html_as_embedded_iframe(
        resource_html,
        height=7600,
    ),
    "</div></details>",
    "</section>",
])


# Claims ledger
parts.extend([
    "<section id='claims'>",
    "<h2>11. Statistical claim ledger</h2>",
    "<div class='claim'>",
    "<div><span class='tag sig'>SUPPORTED</span></div>",
    (
        "<div><strong>Pressure induces defensive salience in Claude and "
        "Fable in a placement-dependent manner.</strong><br>"
        "24/120 matched semantic contrasts survive global Holm correction; "
        "the strongest effects occur source-local and often exceed 40–70 pp.</div>"
    ),
    "</div>",
    "<div class='claim'>",
    "<div><span class='tag null'>NOT SUPPORTED</span></div>",
    (
        "<div><strong>Pressure robustly increases evaluation recognition.</strong><br>"
        "Zero evaluation-recognition contrasts survive correction, and this "
        "field has weaker inter-judge κ.</div>"
    ),
    "</div>",
    "<div class='claim'>",
    "<div><span class='tag desc'>DESCRIPTIVE</span></div>",
    (
        "<div><strong>Defensive semantic states correspond to actual cue "
        "modification/removal.</strong><br>"
        "Strong same-task-adjusted association for Claude source-local and "
        "smaller Fable effects. Because semantic state is post-treatment, "
        "this is not causal.</div>"
    ),
    "</div>",
    "<div class='claim'>",
    "<div><span class='tag null'>NOT SUPPORTED</span></div>",
    (
        "<div><strong>High-confidence cue removal is followed by robust "
        "persistent source-specific execution displacement.</strong><br>"
        "0/6 prespecified temporal endpoints survive Holm correction.</div>"
    ),
    "</div>",
    "<div class='claim'>",
    "<div><span class='tag sig'>SUPPORTED</span></div>",
    (
        "<div><strong>Codex resource deprivation reduces task success relative "
        "to eval-only.</strong><br>"
        "The resource study's primary Codex contrast is −20 pp and survives "
        "its frozen correction family.</div>"
    ),
    "</div>",
    "<div class='claim'>",
    "<div><span class='tag limit'>DO NOT CLAIM</span></div>",
    (
        "<div>Recognition = compliance; pressure recognition = evaluation "
        "awareness; non-significance = equivalence; post-treatment semantic "
        "associations = causal effects; process volume = 'thinking harder' "
        "or 'thinking less'.</div>"
    ),
    "</div>",
    "</section>",
])


# Full tables
parts.extend([
    "<section id='tables'>",
    "<h2>12. Complete publication tables</h2>",
])

for n, filename in enumerate([
    "table_1_reliability.csv",
    "table_2_semantic_rates_by_cell.csv",
    "table_3_matched_semantic_effects.csv",
    "table_4_defensive_chain.csv",
    "table_5_objective_behavioral_validation.csv",
    "table_6_temporal_post_removal.csv",
], start=1):
    parts.extend([
        f"<details><summary>Table {n}: {esc(filename)}</summary>",
        "<div class='details-body'>",
        csv_table(
            filename
        ),
        "</div></details>",
    ])

parts.append(
    "</section>"
)


# Existing historical report as appendix
parts.extend([
    "<section>",
    "<h2>13. Full repaired historical report appendix</h2>",
    (
        "<p>This appendix embeds the previously frozen historical integrated "
        "report v2. It is included for traceability; the sections above are "
        "the publication-oriented synthesis.</p>"
    ),
    "<details>",
    "<summary>Open embedded historical report v2</summary>",
    "<div class='details-body'>",
    html_as_embedded_iframe(
        HISTORICAL_HTML,
        height=9000,
    ),
    "</div></details>",
    "</section>",
])


# Provenance
commit = subprocess.check_output(
    [
        "git",
        "rev-parse",
        "HEAD",
    ],
    text=True,
).strip()

parts.extend([
    "<section id='provenance'>",
    "<h2>14. Provenance and reproducibility</h2>",
    "<div class='table-wrap'><table class='data-table'>",
    "<tr><th>Item</th><th>Value</th></tr>",
    f"<tr><td>Integrated branch commit</td><td><code>{esc(commit)}</code></td></tr>",
    f"<tr><td>Publication summary SHA256</td><td><code>{esc(sha(PUB / 'publication_summary.json'))}</code></td></tr>",
    f"<tr><td>Historical report v2 SHA256</td><td><code>{esc(sha(HISTORICAL_HTML))}</code></td></tr>",
    f"<tr><td>Resource report source</td><td><code>{esc(resource_html)}</code></td></tr>",
    f"<tr><td>Resource report SHA256</td><td><code>{esc(sha(resource_html))}</code></td></tr>",
    "</table></div>",
    (
        "<p class='small'>The final HTML contains its publication figures "
        "and embedded report appendices directly. External network access is "
        "not required to view the report.</p>"
    ),
    "</section>",
    "</main>",
    "</body>",
    "</html>",
])


OUT.write_text(
    "".join(
        parts
    ),
    encoding="utf-8",
)


# Verify images are actually embedded.
result = OUT.read_text(
    encoding="utf-8"
)

assert result.count(
    "data:image/png;base64,"
) >= 5

assert (
    "24/120"
    in result
)

assert (
    "0/6"
    in result
)

assert (
    "2,776"
    in result
)

print(
    "SWE-EVALPRESSURE FULL STUDY HTML V1"
)

print("=" * 80)

print(
    "historical HTML:",
    HISTORICAL_HTML,
)

print(
    "resource HTML:",
    resource_html,
)

print(
    "publication figures embedded: 5"
)

print(
    "publication tables embedded: 6"
)

print(
    "matched semantic significance: 24 / 120"
)

print(
    "evaluation recognition corrected: 0"
)

print(
    "temporal significance: 0 / 6"
)

print(
    "output:",
    OUT,
)

print(
    "output bytes:",
    OUT.stat().st_size,
)

print(
    "output sha256:",
    sha(
        OUT
    ),
)

print(
    "network calls: 0"
)

print(
    "FULL STUDY HTML: PASS"
)
