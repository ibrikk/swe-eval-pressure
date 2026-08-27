#!/usr/bin/env python3
"""Semantic judge/human reliability analysis.

Primary validation design:
- the same frozen sample is independently rated by:
    DeepSeek
    Gemini
    Human 1
    Human 2
    Human 3
- labels and allowed categories come from
  config/semantic_judge_schema.json
- missing labels are never treated as negative labels

Outputs per semantic field:
- LLM↔LLM raw agreement, Cohen's kappa, Gwet AC1
- human pairwise raw agreement / Cohen's kappa
- 3-human Fleiss' kappa and multirater Gwet AC1
- human-majority↔each LLM raw agreement, Cohen's kappa, Gwet AC1
- human-majority↔2-LLM consensus agreement
- disagreement / unresolved counts
- confusion matrices
- class precision / recall / F1
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VERSION = "1.0"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_schema(path: Path) -> dict[str, Any]:
    schema = load_json(path)

    fields = schema.get("fields")
    if not isinstance(fields, dict) or not fields:
        raise ValueError(f"{path}: missing fields")

    for field, spec in fields.items():
        labels = spec.get("labels")
        if (
            not isinstance(labels, list)
            or not labels
            or not all(isinstance(x, str) and x for x in labels)
        ):
            raise ValueError(
                f"{path}: invalid labels for {field}"
            )

    return schema


def load_ratings_csv(
    path: Path,
    fields: dict[str, Any],
    *,
    default_rater: str | None = None,
) -> dict[str, dict[str, str | None]]:
    rows: dict[str, dict[str, str | None]] = {}

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required = {"sample_id", *fields}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path}: missing columns {sorted(missing)}"
            )

        for raw in reader:
            sample_id = str(raw.get("sample_id") or "").strip()
            if not sample_id:
                raise ValueError(
                    f"{path}: row without sample_id"
                )
            if sample_id in rows:
                raise ValueError(
                    f"{path}: duplicate sample_id {sample_id}"
                )

            row: dict[str, str | None] = {}

            for field, spec in fields.items():
                value = str(raw.get(field) or "").strip()
                if not value:
                    row[field] = None
                    continue

                if value not in spec["labels"]:
                    raise ValueError(
                        f"{path}: {sample_id}/{field}: "
                        f"invalid label {value!r}"
                    )

                row[field] = value

            rows[sample_id] = row

    return rows


def observed_pairs(
    a: dict[str, str | None],
    b: dict[str, str | None],
    ids: list[str],
) -> list[tuple[str, str]]:
    pairs = []

    for sample_id in ids:
        av = a.get(sample_id)
        bv = b.get(sample_id)

        if av is None or bv is None:
            continue

        pairs.append((av, bv))

    return pairs


def raw_agreement(
    pairs: list[tuple[str, str]],
) -> float | None:
    if not pairs:
        return None

    return sum(a == b for a, b in pairs) / len(pairs)


def cohen_kappa(
    pairs: list[tuple[str, str]],
    labels: list[str],
) -> float | None:
    if not pairs:
        return None

    n = len(pairs)
    p_o = sum(a == b for a, b in pairs) / n

    ca = Counter(a for a, _ in pairs)
    cb = Counter(b for _, b in pairs)

    p_e = sum(
        (ca[label] / n) * (cb[label] / n)
        for label in labels
    )

    denominator = 1.0 - p_e

    if math.isclose(denominator, 0.0):
        if math.isclose(p_o, 1.0):
            return 1.0
        return None

    return (p_o - p_e) / denominator


def gwet_ac1_two(
    pairs: list[tuple[str, str]],
    labels: list[str],
) -> float | None:
    """Gwet's AC1 for two nominal raters."""
    if not pairs:
        return None

    n = len(pairs)
    k = len(labels)

    if k < 2:
        return None

    p_o = sum(a == b for a, b in pairs) / n

    ca = Counter(a for a, _ in pairs)
    cb = Counter(b for _, b in pairs)

    p_bar = {
        label: (
            ca[label] + cb[label]
        ) / (2.0 * n)
        for label in labels
    }

    p_e = (
        sum(
            p * (1.0 - p)
            for p in p_bar.values()
        )
        / (k - 1)
    )

    denominator = 1.0 - p_e

    if math.isclose(denominator, 0.0):
        return None

    return (p_o - p_e) / denominator


def fleiss_kappa(
    matrix: list[list[int]],
) -> float | None:
    """Fleiss' kappa for equal-size nominal-rating rows."""
    if not matrix:
        return None

    n_subjects = len(matrix)
    n_categories = len(matrix[0])

    if n_categories < 2:
        return None

    n_ratings = sum(matrix[0])

    if n_ratings < 2:
        return None

    if any(
        len(row) != n_categories
        or sum(row) != n_ratings
        for row in matrix
    ):
        raise ValueError(
            "Fleiss matrix requires equal ratings per subject"
        )

    p_i = [
        (
            sum(count * count for count in row)
            - n_ratings
        )
        / (
            n_ratings
            * (n_ratings - 1)
        )
        for row in matrix
    ]

    p_bar = sum(p_i) / n_subjects

    category_totals = [
        sum(row[j] for row in matrix)
        for j in range(n_categories)
    ]

    denominator = n_subjects * n_ratings

    p_j = [
        total / denominator
        for total in category_totals
    ]

    p_e = sum(p * p for p in p_j)

    if math.isclose(1.0 - p_e, 0.0):
        if math.isclose(p_bar, 1.0):
            return 1.0
        return None

    return (p_bar - p_e) / (1.0 - p_e)


def gwet_ac1_multi(
    ratings: list[list[str]],
    labels: list[str],
) -> float | None:
    """Multirater nominal Gwet AC1.

    Rows must contain the same number of nonmissing ratings.
    """
    if not ratings:
        return None

    r = len(ratings[0])
    k = len(labels)

    if r < 2 or k < 2:
        return None

    if any(len(row) != r for row in ratings):
        raise ValueError(
            "AC1 requires equal number of ratings per item"
        )

    p_o_rows = []

    total = Counter()

    for row in ratings:
        counts = Counter(row)
        total.update(row)

        agreements = sum(
            count * (count - 1)
            for count in counts.values()
        )

        p_o_rows.append(
            agreements / (r * (r - 1))
        )

    p_o = sum(p_o_rows) / len(p_o_rows)

    denominator = len(ratings) * r

    p = {
        label: total[label] / denominator
        for label in labels
    }

    p_e = (
        sum(
            value * (1.0 - value)
            for value in p.values()
        )
        / (k - 1)
    )

    if math.isclose(1.0 - p_e, 0.0):
        return None

    return (p_o - p_e) / (1.0 - p_e)


def majority_of_three(
    values: list[str | None],
) -> str | None:
    valid = [value for value in values if value is not None]

    if len(valid) != 3:
        return None

    counts = Counter(valid)
    label, count = counts.most_common(1)[0]

    if count >= 2:
        return label

    return None


def two_rater_consensus(
    a: str | None,
    b: str | None,
) -> str | None:
    if a is None or b is None:
        return None

    if a != b:
        return None

    return a


def confusion_rows(
    pairs: list[tuple[str, str]],
    labels: list[str],
    *,
    field: str,
    comparison: str,
) -> list[dict[str, Any]]:
    counts = Counter(pairs)

    rows = []

    for reference in labels:
        for prediction in labels:
            rows.append(
                {
                    "field": field,
                    "comparison": comparison,
                    "reference": reference,
                    "prediction": prediction,
                    "count": counts[
                        (reference, prediction)
                    ],
                }
            )

    return rows


def class_metric_rows(
    pairs: list[tuple[str, str]],
    labels: list[str],
    *,
    field: str,
    comparison: str,
) -> list[dict[str, Any]]:
    rows = []

    for label in labels:
        tp = sum(
            reference == label and prediction == label
            for reference, prediction in pairs
        )
        fp = sum(
            reference != label and prediction == label
            for reference, prediction in pairs
        )
        fn = sum(
            reference == label and prediction != label
            for reference, prediction in pairs
        )

        precision = (
            tp / (tp + fp)
            if tp + fp
            else None
        )
        recall = (
            tp / (tp + fn)
            if tp + fn
            else None
        )

        if (
            precision is not None
            and recall is not None
            and precision + recall
        ):
            f1 = (
                2.0
                * precision
                * recall
                / (precision + recall)
            )
        else:
            f1 = None

        rows.append(
            {
                "field": field,
                "comparison": comparison,
                "label": label,
                "support": tp + fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    return rows


def comparison_row(
    *,
    field: str,
    comparison: str,
    pairs: list[tuple[str, str]],
    labels: list[str],
) -> dict[str, Any]:
    return {
        "field": field,
        "comparison": comparison,
        "n": len(pairs),
        "raw_agreement": raw_agreement(pairs),
        "cohen_kappa": cohen_kappa(
            pairs,
            labels,
        ),
        "gwet_ac1": gwet_ac1_two(
            pairs,
            labels,
        ),
    }


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def build_html(
    agreement: list[dict[str, Any]],
    human_irr: list[dict[str, Any]],
    unresolved: list[dict[str, Any]],
) -> str:
    agreement_rows = "\n".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(fmt(row[key]))}</td>"
            for key in (
                "field",
                "comparison",
                "n",
                "raw_agreement",
                "cohen_kappa",
                "gwet_ac1",
            )
        )
        + "</tr>"
        for row in agreement
    )

    irr_rows = "\n".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(fmt(row[key]))}</td>"
            for key in (
                "field",
                "n_complete",
                "fleiss_kappa",
                "gwet_ac1",
                "human_majority_resolved",
                "human_majority_unresolved",
            )
        )
        + "</tr>"
        for row in human_irr
    )

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>SWE-EvalPressure semantic reliability</title>
<style>
body {{
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  max-width:1250px;
  margin:auto;
  padding:28px;
  color:#172033;
}}
table {{
  width:100%;
  border-collapse:collapse;
  margin:18px 0 34px;
}}
th,td {{
  text-align:left;
  padding:8px;
  border-bottom:1px solid #ddd;
}}
th {{
  background:#f5f7fa;
}}
code {{
  background:#f2f4f7;
  padding:2px 4px;
}}
</style>
</head>
<body>
<h1>SWE-EvalPressure semantic reliability</h1>

<p>
All agreement statistics use only jointly observed
ratings. Missing ratings are never recoded as negative.
Human-majority labels require an actual 2-of-3 majority.
Three-way human disagreements remain unresolved.
</p>

<h2>Two-rater comparisons</h2>
<table>
<thead>
<tr>
<th>Field</th>
<th>Comparison</th>
<th>N</th>
<th>Raw agreement</th>
<th>Cohen κ</th>
<th>Gwet AC1</th>
</tr>
</thead>
<tbody>
{agreement_rows}
</tbody>
</table>

<h2>Three-human reliability</h2>
<table>
<thead>
<tr>
<th>Field</th>
<th>N complete</th>
<th>Fleiss κ</th>
<th>Gwet AC1</th>
<th>Majority resolved</th>
<th>Unresolved</th>
</tr>
</thead>
<tbody>
{irr_rows}
</tbody>
</table>

<p>
Detailed confusion matrices, class-level metrics,
disagreement cases, and frozen machine-readable summaries
are stored beside this report.
</p>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--schema",
        type=Path,
        default=Path(
            "config/semantic_judge_schema.json"
        ),
    )

    parser.add_argument(
        "--human",
        action="append",
        type=Path,
        required=True,
        help=(
            "Human annotation CSV. "
            "Supply exactly three times."
        ),
    )

    parser.add_argument(
        "--deepseek",
        type=Path,
        required=True,
        help=(
            "Normalized DeepSeek CSV with "
            "sample_id + semantic fields."
        ),
    )

    parser.add_argument(
        "--gemini",
        type=Path,
        required=True,
        help=(
            "Normalized Gemini CSV with "
            "sample_id + semantic fields."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    if len(args.human) != 3:
        raise ValueError(
            "Exactly three --human files are required"
        )

    schema = load_schema(args.schema)
    fields = schema["fields"]

    humans = [
        load_ratings_csv(path, fields)
        for path in args.human
    ]

    deepseek = load_ratings_csv(
        args.deepseek,
        fields,
    )
    gemini = load_ratings_csv(
        args.gemini,
        fields,
    )

    id_sets = [
        set(ratings)
        for ratings in [
            *humans,
            deepseek,
            gemini,
        ]
    ]

    if len({frozenset(x) for x in id_sets}) != 1:
        raise ValueError(
            "All five rating files must contain "
            "the exact same sample_id set"
        )

    ids = sorted(id_sets[0])

    agreement_rows: list[dict[str, Any]] = []
    irr_rows: list[dict[str, Any]] = []
    confusion: list[dict[str, Any]] = []
    class_metrics: list[dict[str, Any]] = []
    disagreement_rows: list[dict[str, Any]] = []
    majority_rows: list[dict[str, Any]] = []

    for field, spec in fields.items():
        labels = list(spec["labels"])

        h = [
            {
                sample_id: human[sample_id][field]
                for sample_id in ids
            }
            for human in humans
        ]

        ds = {
            sample_id: deepseek[sample_id][field]
            for sample_id in ids
        }
        gm = {
            sample_id: gemini[sample_id][field]
            for sample_id in ids
        }

        majority: dict[str, str | None] = {}
        llm_consensus: dict[str, str | None] = {}

        complete_human_ratings: list[list[str]] = []
        fleiss_matrix: list[list[int]] = []

        resolved = 0
        unresolved = 0

        for sample_id in ids:
            human_values = [
                h_i[sample_id]
                for h_i in h
            ]

            m = majority_of_three(
                human_values
            )
            majority[sample_id] = m

            c = two_rater_consensus(
                ds[sample_id],
                gm[sample_id],
            )
            llm_consensus[sample_id] = c

            if all(
                value is not None
                for value in human_values
            ):
                typed = [
                    str(value)
                    for value in human_values
                ]

                complete_human_ratings.append(
                    typed
                )

                counts = Counter(typed)

                fleiss_matrix.append(
                    [
                        counts[label]
                        for label in labels
                    ]
                )

            if m is None:
                unresolved += 1
            else:
                resolved += 1

            majority_rows.append(
                {
                    "sample_id": sample_id,
                    "field": field,
                    "human1": human_values[0],
                    "human2": human_values[1],
                    "human3": human_values[2],
                    "human_majority": m,
                    "deepseek": ds[sample_id],
                    "gemini": gm[sample_id],
                    "llm_consensus": c,
                }
            )

            values = {
                "human1": human_values[0],
                "human2": human_values[1],
                "human3": human_values[2],
                "human_majority": m,
                "deepseek": ds[sample_id],
                "gemini": gm[sample_id],
                "llm_consensus": c,
            }

            distinct = {
                value
                for value in values.values()
                if value is not None
            }

            if (
                len(distinct) > 1
                or m is None
                or c is None
            ):
                disagreement_rows.append(
                    {
                        "sample_id": sample_id,
                        "field": field,
                        **values,
                    }
                )

        irr_rows.append(
            {
                "field": field,
                "n_complete": len(
                    complete_human_ratings
                ),
                "fleiss_kappa": (
                    fleiss_kappa(
                        fleiss_matrix
                    )
                ),
                "gwet_ac1": (
                    gwet_ac1_multi(
                        complete_human_ratings,
                        labels,
                    )
                ),
                "human_majority_resolved": resolved,
                "human_majority_unresolved": unresolved,
            }
        )

        comparisons = [
            (
                "deepseek_vs_gemini",
                ds,
                gm,
                False,
            ),
            (
                "human1_vs_human2",
                h[0],
                h[1],
                False,
            ),
            (
                "human1_vs_human3",
                h[0],
                h[2],
                False,
            ),
            (
                "human2_vs_human3",
                h[1],
                h[2],
                False,
            ),
            (
                "human_majority_vs_deepseek",
                majority,
                ds,
                True,
            ),
            (
                "human_majority_vs_gemini",
                majority,
                gm,
                True,
            ),
            (
                "human_majority_vs_llm_consensus",
                majority,
                llm_consensus,
                True,
            ),
        ]

        for (
            comparison,
            left,
            right,
            reference_is_left,
        ) in comparisons:
            pairs = observed_pairs(
                left,
                right,
                ids,
            )

            agreement_rows.append(
                comparison_row(
                    field=field,
                    comparison=comparison,
                    pairs=pairs,
                    labels=labels,
                )
            )

            if reference_is_left:
                confusion.extend(
                    confusion_rows(
                        pairs,
                        labels,
                        field=field,
                        comparison=comparison,
                    )
                )

                class_metrics.extend(
                    class_metric_rows(
                        pairs,
                        labels,
                        field=field,
                        comparison=comparison,
                    )
                )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        output_dir
        / "agreement_by_field.csv",
        agreement_rows,
    )
    write_csv(
        output_dir
        / "human_irr_by_field.csv",
        irr_rows,
    )
    write_csv(
        output_dir
        / "confusion_matrices.csv",
        confusion,
    )
    write_csv(
        output_dir
        / "class_metrics.csv",
        class_metrics,
    )
    write_csv(
        output_dir
        / "disagreement_items.csv",
        disagreement_rows,
    )
    write_csv(
        output_dir
        / "human_majority.csv",
        majority_rows,
    )

    summary = {
        "version": VERSION,
        "n_items": len(ids),
        "n_humans": 3,
        "llm_judges": [
            "deepseek",
            "gemini",
        ],
        "fields": list(fields),
        "missing_is_not_negative": True,
        "human_majority_rule": (
            "2-of-3; otherwise unresolved"
        ),
        "llm_consensus_rule": (
            "2-of-2 exact match; otherwise unresolved"
        ),
        "agreement": agreement_rows,
        "human_irr": irr_rows,
    }

    (
        output_dir
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    (
        output_dir
        / "semantic_validation.html"
    ).write_text(
        build_html(
            agreement_rows,
            irr_rows,
            disagreement_rows,
        ),
        encoding="utf-8",
    )

    print("SEMANTIC RELIABILITY: PASS")
    print("items:", len(ids))
    print("fields:", len(fields))
    print("humans: 3")
    print(
        "two-rater comparisons:",
        len(agreement_rows),
    )
    print("output:", output_dir)


if __name__ == "__main__":
    main()
