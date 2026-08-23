#!/usr/bin/env python3
"""Create a human-readable report for disagreements between two semantic smoke runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FIELDS = [
    "evaluation_cue_recognition",
    "evaluation_cue_stance",
    "hidden_grader_awareness",
    "benchmark_recognition",
    "current_agent_evaluation_awareness",
    "evaluation_purpose_recognition",
    "evaluation_response",
    "pressure_recognition",
    "pressure_type_recognized",
    "pressure_cue_stance",
    "pressure_response",
    "evaluation_targeting_intent",
]


def load_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def quote_lines(judgment):
    quotes = judgment.get("evidence_quotes", [])
    if not isinstance(quotes, list) or not quotes:
        return ["- evidence_quotes: none"]
    out = []
    for q in quotes:
        if isinstance(q, dict):
            out.append(
                f"- `{q.get('label', '')}`: {str(q.get('quote', '')).strip()}"
            )
        else:
            out.append(f"- {q}")
    return out


def context_for_quote(view: str, quote: str, radius: int = 350):
    quote = quote.strip()
    if not quote:
        return None
    idx = view.find(quote)
    if idx < 0:
        return None
    start = max(0, idx - radius)
    end = min(len(view), idx + len(quote) + radius)
    return view[start:end].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a", type=Path, required=True)
    ap.add_argument("--run-b", type=Path, required=True)
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/semantic/v25-smoke-disagreements.md"),
    )
    args = ap.parse_args()

    a = load_jsonl(args.run_a / "smoke_results.jsonl")
    b = load_jsonl(args.run_b / "smoke_results.jsonl")

    if len(a) != len(b):
        raise SystemExit("Run lengths differ")

    lines = [
        "# Semantic smoke disagreement report",
        "",
        "This report is deterministic and makes no model calls.",
        "",
    ]

    field_counts = {f: 0 for f in FIELDS}
    trial_count = 0
    diff_count = 0

    for i, (ra, rb) in enumerate(zip(a, b), start=1):
        if ra["trial_name"] != rb["trial_name"]:
            raise SystemExit(
                f"Trial mismatch at row {i}: {ra['trial_name']} != {rb['trial_name']}"
            )

        ja = ra["new_judgment"]
        jb = rb["new_judgment"]
        diffs = [f for f in FIELDS if ja.get(f) != jb.get(f)]
        if not diffs:
            continue

        trial_count += 1
        diff_count += len(diffs)
        for f in diffs:
            field_counts[f] += 1

        prefix = f"{i:02d}_{ra['profile']}_{ra['case_type']}"
        view_path = args.run_a / f"{prefix}.view.txt"
        view = view_path.read_text(encoding="utf-8")

        lines.extend([
            f"## {ra['profile']} — {ra['trial_name']}",
            "",
            f"- case: `{ra['case_type']}`",
            f"- condition: `{ra['condition']}`",
            f"- channel: `{ra['channel']}`",
            f"- pressure type: `{ra['pressure_type']}`",
            f"- view chars: {ra['semantic_view_chars']}",
            "",
            "### Disagreeing fields",
            "",
        ])

        old = ra.get("old_v23", {})
        for f in diffs:
            lines.append(
                f"- **{f}**: v2.3=`{old.get(f, '')}` | "
                f"A=`{ja.get(f, '')}` | B=`{jb.get(f, '')}`"
            )

        lines.extend([
            "",
            "### Run A evidence",
            "",
            f"- confidence: `{ja.get('confidence', '')}`",
        ])
        lines.extend(quote_lines(ja))

        lines.extend([
            "",
            "### Run B evidence",
            "",
            f"- confidence: `{jb.get('confidence', '')}`",
        ])
        lines.extend(quote_lines(jb))

        # Collect contexts for distinct quoted strings from both runs.
        contexts = []
        seen = set()
        for judgment in (ja, jb):
            quotes = judgment.get("evidence_quotes", [])
            if not isinstance(quotes, list):
                continue
            for q in quotes:
                if not isinstance(q, dict):
                    continue
                quote = str(q.get("quote", "")).strip()
                if not quote or quote in seen:
                    continue
                seen.add(quote)
                ctx = context_for_quote(view, quote)
                if ctx:
                    contexts.append((q.get("label", ""), quote, ctx))

        if contexts:
            lines.extend(["", "### Evidence in exact semantic view", ""])
            for label, quote, ctx in contexts:
                lines.extend([
                    f"**{label or 'evidence'}**",
                    "",
                    "```text",
                    ctx,
                    "```",
                    "",
                ])

        lines.extend(["---", ""])

    lines.insert(4, f"- trajectories with A/B disagreement: **{trial_count}**")
    lines.insert(5, f"- disagreeing fields: **{diff_count}**")
    lines.insert(6, "")
    lines.insert(7, "## Disagreement counts by field")
    lines.insert(8, "")
    insert_at = 9
    for f in FIELDS:
        if field_counts[f]:
            lines.insert(insert_at, f"- `{f}`: {field_counts[f]}")
            insert_at += 1
    lines.insert(insert_at, "")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"trajectories_with_disagreement: {trial_count}")
    print(f"disagreeing_fields: {diff_count}")
    print("field_counts:")
    for f in FIELDS:
        if field_counts[f]:
            print(f"  {f}: {field_counts[f]}")
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
