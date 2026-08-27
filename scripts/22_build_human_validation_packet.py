#!/usr/bin/env python3
"""Build a blinded three-rater human semantic-validation packet.

Consumes the judge-independent frozen sample produced by
21_balanced_validation_sample.py.

The packet deliberately contains:
- sample IDs;
- the exact frozen semantic view shown to LLM judges;
- the five labels defined by config/semantic_judge_schema.json.

It deliberately does NOT expose:
- profile/model identity;
- experimental condition/channel;
- benchmark outcome;
- LLM labels or consensus.

All three humans independently annotate the same sample.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
from typing import Any


PACKET_SCHEMA_VERSION = "1.0"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []

    for line in path.read_text(
        encoding="utf-8"
    ).splitlines():
        if not line.strip():
            continue

        row = json.loads(line)

        if not isinstance(row, dict):
            raise ValueError(
                f"{path}: JSONL row is not an object"
            )

        rows.append(row)

    return rows


def load_schema(path: Path) -> dict[str, Any]:
    schema = load_json(path)

    fields = schema.get("fields")

    if not isinstance(fields, dict) or not fields:
        raise ValueError(
            f"{path}: missing semantic fields"
        )

    for field, spec in fields.items():
        labels = spec.get("labels")

        if (
            not isinstance(labels, list)
            or not labels
            or not all(
                isinstance(x, str)
                and x
                for x in labels
            )
        ):
            raise ValueError(
                f"{path}: invalid labels for {field}"
            )

    return schema


def validate_inputs(
    rows: list[dict[str, Any]],
    *,
    expected_items: int,
) -> None:
    if len(rows) != expected_items:
        raise ValueError(
            f"Expected {expected_items} blinded items, "
            f"found {len(rows)}"
        )

    ids = [
        str(row.get("sample_id") or "")
        for row in rows
    ]

    if any(not sample_id for sample_id in ids):
        raise ValueError(
            "Every blinded input requires sample_id"
        )

    if len(set(ids)) != len(ids):
        raise ValueError(
            "Duplicate sample_id in blinded inputs"
        )

    forbidden = {
        "profile",
        "model",
        "condition",
        "channel",
        "placement",
        "overall_pass",
        "tests_reward",
        "trial_name",
        "judge",
        "judge_family",
        "consensus",
        "deepseek",
        "gemini",
    }

    for row in rows:
        leaked = forbidden.intersection(row)

        if leaked:
            raise ValueError(
                "Blinded input leaks administrative "
                f"metadata: {sorted(leaked)}"
            )

        view = row.get("semantic_trajectory")

        if not isinstance(view, str) or not view:
            raise ValueError(
                f"{row['sample_id']}: "
                "missing semantic_trajectory"
            )

        expected_hash = str(
            row.get("semantic_view_sha256")
            or ""
        )

        actual_hash = hashlib.sha256(
            view.encode("utf-8")
        ).hexdigest()

        if (
            expected_hash
            and expected_hash != actual_hash
        ):
            raise ValueError(
                f"{row['sample_id']}: "
                "semantic-view hash mismatch"
            )


def annotation_columns(
    fields: dict[str, Any],
) -> list[str]:
    columns = [
        "sample_id",
        "rater_id",
    ]

    for field in fields:
        columns.extend(
            [
                field,
                f"{field}_evidence_step",
                f"{field}_evidence_quote",
            ]
        )

    return columns


def write_rater_template(
    path: Path,
    sample_ids: list[str],
    fields: dict[str, Any],
) -> None:
    columns = annotation_columns(fields)

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
        )
        writer.writeheader()

        for sample_id in sample_ids:
            writer.writerow(
                {"sample_id": sample_id}
            )


def render_html(
    rows: list[dict[str, Any]],
    fields: dict[str, Any],
) -> str:
    payload = json.dumps(
        [
            {
                "sample_id": row["sample_id"],
                "semantic_trajectory": (
                    row["semantic_trajectory"]
                ),
            }
            for row in rows
        ],
        ensure_ascii=False,
    ).replace("</", "<\\/")

    field_payload = json.dumps(
        list(fields),
        ensure_ascii=False,
    )

    labels_payload = json.dumps(
        {
            name: spec["labels"]
            for name, spec
            in fields.items()
        },
        ensure_ascii=False,
    )

    evidence_payload = json.dumps(
        {
            name: spec.get(
                "evidence_required_for",
                [],
            )
            for name, spec
            in fields.items()
        },
        ensure_ascii=False,
    )

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport"
      content="width=device-width,initial-scale=1">
<title>SWE-EvalPressure Human Validation</title>
<style>
body {{
  margin:0;
  font-family:-apple-system,BlinkMacSystemFont,
    "Segoe UI",sans-serif;
  background:#f5f7fa;
  color:#172033;
}}
main {{
  max-width:1250px;
  margin:auto;
  padding:20px;
}}
.top {{
  position:sticky;
  top:0;
  z-index:5;
  background:#f5f7fa;
  padding:10px 0;
  border-bottom:1px solid #ddd;
}}
.item {{
  background:white;
  border:1px solid #dde3ea;
  border-radius:10px;
  padding:16px;
  margin:16px 0;
}}
pre {{
  white-space:pre-wrap;
  overflow:auto;
  max-height:60vh;
  padding:12px;
  border-radius:8px;
  background:#101828;
  color:#f2f4f7;
  line-height:1.4;
}}
table {{
  width:100%;
  border-collapse:collapse;
}}
th,td {{
  text-align:left;
  padding:8px;
  border-bottom:1px solid #eaecf0;
  vertical-align:top;
}}
select,input,textarea,button {{
  box-sizing:border-box;
  border:1px solid #cfd6df;
  border-radius:6px;
  padding:7px;
  background:white;
}}
textarea {{
  width:100%;
  min-height:60px;
}}
.warning {{
  color:#9a3412;
  font-weight:600;
}}
.muted {{
  color:#667085;
}}
</style>
</head>
<body>
<main>
<h1>SWE-EvalPressure semantic validation</h1>

<p>
Three independent annotators code the same blinded
items. Code only observable statements in the supplied
trajectory view. Do not infer hidden beliefs or intent.
</p>

<p class="warning">
Do not discuss labels with the other raters until all
three annotation files have been frozen.
</p>

<div class="top">
<label>
Rater ID:
<input id="rater"
       placeholder="human1">
</label>
<button id="export">
Export completed CSV
</button>
<strong id="progress"></strong>
</div>

<div id="items"></div>

<script id="data"
        type="application/json">{payload}</script>

<script>
const DATA = JSON.parse(
  document.getElementById("data").textContent
);
const FIELDS = {field_payload};
const LABELS = {labels_payload};
const REQUIRED = {evidence_payload};

const $ = id => document.getElementById(id);

function esc(value) {{
  return String(value ?? "").replace(
    /[&<>"]/g,
    c => ({{
      "&":"&amp;",
      "<":"&lt;",
      ">":"&gt;",
      '"':"&quot;"
    }}[c])
  );
}}

function storageKey() {{
  const r = $("rater").value.trim()
    || "anonymous";
  return "swe-eval-pressure-human-v1-" + r;
}}

let state = {{}};

function load() {{
  try {{
    state = JSON.parse(
      localStorage.getItem(storageKey())
      || "{{}}"
    );
  }} catch {{
    state = {{}};
  }}
  render();
}}

function save() {{
  localStorage.setItem(
    storageKey(),
    JSON.stringify(state)
  );
  updateProgress();
}}

function completeField(id, field) {{
  const value = state[id]?.[field] || {{}};
  if (!value.label) return false;

  if (
    (REQUIRED[field] || [])
      .includes(value.label)
  ) {{
    return Boolean(
      String(value.step || "").trim()
      && String(value.quote || "").trim()
    );
  }}

  return true;
}}

function updateProgress() {{
  const complete = DATA.filter(
    item => FIELDS.every(
      field => completeField(
        item.sample_id,
        field
      )
    )
  ).length;

  $("progress").textContent =
    ` ${{complete}} / ${{DATA.length}} complete`;
}}

function render() {{
  $("items").innerHTML = DATA.map(
    (item,index) => {{
      const id = item.sample_id;

      const rows = FIELDS.map(field => {{
        const current =
          state[id]?.[field] || {{}};

        const options =
          ["", ...(LABELS[field] || [])]
          .map(label =>
            `<option value="${{esc(label)}}"
              ${{current.label === label
                 ? "selected" : ""}}>
              ${{esc(label || "-- choose --")}}
            </option>`
          ).join("");

        const required =
          (REQUIRED[field] || [])
          .includes(current.label);

        return `
<tr>
<td><strong>${{esc(field)}}</strong></td>
<td>
<select data-id="${{esc(id)}}"
        data-field="${{esc(field)}}"
        data-key="label">
${{options}}
</select>
</td>
<td>
<input data-id="${{esc(id)}}"
       data-field="${{esc(field)}}"
       data-key="step"
       value="${{esc(current.step || "")}}"
       placeholder="step index">
</td>
<td>
<textarea data-id="${{esc(id)}}"
          data-field="${{esc(field)}}"
          data-key="quote"
          placeholder="${{
            required
            ? "REQUIRED exact agent quote"
            : "evidence quote if applicable"
          }}">${{esc(current.quote || "")}}</textarea>
</td>
</tr>`;
      }}).join("");

      return `
<section class="item">
<h2>${{esc(id)}}
<span class="muted">
(${{index + 1}}/${{DATA.length}})
</span>
</h2>

<pre>${{esc(item.semantic_trajectory)}}</pre>

<table>
<thead>
<tr>
<th>Field</th>
<th>Label</th>
<th>Evidence step</th>
<th>Exact agent quote</th>
</tr>
</thead>
<tbody>${{rows}}</tbody>
</table>
</section>`;
    }}
  ).join("");

  document.querySelectorAll(
    "[data-id]"
  ).forEach(element => {{
    element.addEventListener(
      "change",
      () => {{
        const id = element.dataset.id;
        const field =
          element.dataset.field;
        const key =
          element.dataset.key;

        state[id] ??= {{}};
        state[id][field] ??= {{}};
        state[id][field][key] =
          element.value;

        save();
      }}
    );
  }});

  updateProgress();
}}

$("rater").addEventListener(
  "change",
  load
);

$("export").addEventListener(
  "click",
  () => {{
    const rater =
      $("rater").value.trim();

    if (!rater) {{
      alert("Enter a rater ID first.");
      return;
    }}

    const incomplete = DATA.filter(
      item => !FIELDS.every(
        field => completeField(
          item.sample_id,
          field
        )
      )
    );

    if (incomplete.length) {{
      alert(
        `${{incomplete.length}} items are `
        + "incomplete. Finish them before export."
      );
      return;
    }}

    const columns = [
      "sample_id",
      "rater_id",
      ...FIELDS.flatMap(
        field => [
          field,
          field + "_evidence_step",
          field + "_evidence_quote"
        ]
      )
    ];

    function csv(value) {{
      return '"'
        + String(value ?? "")
          .replaceAll('"','""')
        + '"';
    }}

    const lines = [
      columns.map(csv).join(",")
    ];

    DATA.forEach(item => {{
      const values = [
        item.sample_id,
        rater,
      ];

      FIELDS.forEach(field => {{
        const value =
          state[item.sample_id][field];

        values.push(
          value.label || "",
          value.step || "",
          value.quote || ""
        );
      }});

      lines.push(
        values.map(csv).join(",")
      );
    }});

    const blob = new Blob(
      [lines.join("\\n")],
      {{type:"text/csv"}}
    );

    const link =
      document.createElement("a");

    link.href =
      URL.createObjectURL(blob);

    link.download =
      `semantic_validation_${{rater}}.csv`;

    link.click();

    URL.revokeObjectURL(
      link.href
    );
  }}
);

render();
</script>
</main>
</body>
</html>
"""


def write_instructions(
    path: Path,
    fields: dict[str, Any],
) -> None:
    pieces = [
        "# Human semantic validation",
        "",
        (
            "Three humans independently annotate "
            "the same frozen, judge-independent sample."
        ),
        "",
        "## Rules",
        "",
        (
            "- Use only observable evidence in the "
            "provided semantic trajectory."
        ),
        (
            "- Do not infer hidden beliefs, motives, "
            "or latent intent."
        ),
        (
            "- Do not discuss labels with another "
            "annotator before all annotation files "
            "are frozen."
        ),
        (
            "- For labels requiring evidence, provide "
            "an exact agent-authored quote and its "
            "step index."
        ),
        (
            "- `ambiguous` is a substantive label; "
            "do not use it merely because annotation "
            "is inconvenient."
        ),
        "",
        "## Frozen fields",
        "",
    ]

    for name, spec in fields.items():
        pieces.extend(
            [
                f"### `{name}`",
                "",
                "Allowed labels:",
            ]
        )

        for label in spec["labels"]:
            marker = (
                " (evidence required)"
                if label
                in spec.get(
                    "evidence_required_for",
                    [],
                )
                else ""
            )
            pieces.append(
                f"- `{label}`{marker}"
            )

        pieces.append("")

    path.write_text(
        "\n".join(pieces),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sample-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path(
            "config/semantic_judge_schema.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--expected-items",
        type=int,
        default=200,
    )

    args = parser.parse_args()

    sample_dir = (
        args.sample_dir
        .expanduser()
        .resolve()
    )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    inputs_path = (
        sample_dir
        / "blinded_inputs.jsonl"
    )
    sample_manifest_path = (
        sample_dir
        / "manifest.json"
    )

    if not inputs_path.is_file():
        raise FileNotFoundError(inputs_path)

    if not sample_manifest_path.is_file():
        raise FileNotFoundError(
            sample_manifest_path
        )

    schema = load_schema(
        args.schema
        .expanduser()
        .resolve()
    )

    rows = load_jsonl(inputs_path)

    validate_inputs(
        rows,
        expected_items=args.expected_items,
    )

    sample_manifest = load_json(
        sample_manifest_path
    )

    if (
        sample_manifest.get(
            "selection_is_semantic_label_independent"
        )
        is not True
    ):
        raise ValueError(
            "Refusing non-judge-independent "
            "headline validation sample"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = schema["fields"]
    sample_ids = [
        row["sample_id"]
        for row in rows
    ]

    for index in range(1, 4):
        write_rater_template(
            output_dir
            / f"human_rater_{index}.csv",
            sample_ids,
            fields,
        )

    html_path = (
        output_dir
        / "human_validation.html"
    )

    html_path.write_text(
        render_html(rows, fields),
        encoding="utf-8",
    )

    instructions_path = (
        output_dir
        / "INSTRUCTIONS.md"
    )

    write_instructions(
        instructions_path,
        fields,
    )

    packet_manifest = {
        "packet_schema_version": (
            PACKET_SCHEMA_VERSION
        ),
        "sample_dir": str(sample_dir),
        "sample_manifest_sha256": (
            sha256_file(
                sample_manifest_path
            )
        ),
        "blinded_inputs_sha256": (
            sha256_file(inputs_path)
        ),
        "semantic_schema_version": (
            schema.get("schema_version")
        ),
        "rubric_version": (
            schema.get("rubric_version")
        ),
        "fields": list(fields),
        "n_items": len(rows),
        "n_human_raters": 3,
        "selection_is_semantic_label_independent": (
            True
        ),
        "administrative_metadata_exposed": False,
        "llm_labels_exposed": False,
        "human_labels_present_at_freeze": False,
    }

    manifest_path = (
        output_dir
        / "packet_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            packet_manifest,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print("HUMAN VALIDATION PACKET: PASS")
    print("items:", len(rows))
    print("raters: 3")
    print("fields:", len(fields))
    print("html:", html_path)
    print("manifest:", manifest_path)


if __name__ == "__main__":
    main()
