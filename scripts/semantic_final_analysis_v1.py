#!/usr/bin/env python3
"""Final historical semantic analysis.

Network-free.

Reads ONLY the materialized final historical semantic freeze and produces:
- judge coverage;
- DeepSeek/Gemini label-level reliability;
- Cohen's kappa;
- Gwet AC1;
- per-profile reliability;
- consensus label distributions;
- judge-specific label distributions;
- key semantic endpoint tables.

Disagreement and missing are never coded as negative labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ANALYZER_VERSION = "1.0"


def load_json(path: Path) -> Any:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def safe_div(
    numerator: int | float,
    denominator: int | float,
) -> float | None:
    if denominator == 0:
        return None

    return numerator / denominator


def cohen_kappa(
    left: list[str],
    right: list[str],
) -> float | None:
    if len(left) != len(right):
        raise ValueError(
            "paired rating lengths differ"
        )

    n = len(left)

    if n == 0:
        return None

    labels = sorted(
        set(left) | set(right)
    )

    po = sum(
        a == b
        for a, b in zip(
            left,
            right,
            strict=True,
        )
    ) / n

    l = Counter(left)
    r = Counter(right)

    pe = sum(
        (l[label] / n)
        * (r[label] / n)
        for label in labels
    )

    if abs(1.0 - pe) < 1e-15:
        return (
            1.0
            if abs(1.0 - po) < 1e-15
            else None
        )

    return (
        (po - pe)
        / (1.0 - pe)
    )


def gwet_ac1(
    left: list[str],
    right: list[str],
) -> float | None:
    """Gwet AC1 for two raters, nominal categories."""
    if len(left) != len(right):
        raise ValueError(
            "paired rating lengths differ"
        )

    n = len(left)

    if n == 0:
        return None

    labels = sorted(
        set(left) | set(right)
    )

    q = len(labels)

    po = sum(
        a == b
        for a, b in zip(
            left,
            right,
            strict=True,
        )
    ) / n

    if q <= 1:
        return 1.0

    l = Counter(left)
    r = Counter(right)

    # Overall marginal probability for each
    # category across both raters.
    p = {
        label: (
            l[label]
            + r[label]
        )
        / (2.0 * n)
        for label in labels
    }

    pe = sum(
        value
        * (1.0 - value)
        for value in p.values()
    ) / (q - 1)

    if abs(1.0 - pe) < 1e-15:
        return (
            1.0
            if abs(1.0 - po) < 1e-15
            else None
        )

    return (
        (po - pe)
        / (1.0 - pe)
    )


def fmt(value: float | None) -> str:
    if value is None:
        return "NA"

    return f"{value:.4f}"


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        raise ValueError(
            f"no rows for {path}"
        )

    columns = list(
        rows[0].keys()
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
        )

        writer.writeheader()
        writer.writerows(rows)


def job_label(
    artifact: dict[str, Any],
    field: str,
) -> str | None:
    if artifact.get("status") != "ok":
        return None

    entry = artifact.get(
        "final_cache_entry"
    )

    if not isinstance(entry, dict):
        return None

    judgment = entry.get(
        "judgment"
    )

    if not isinstance(
        judgment,
        dict,
    ):
        return None

    value = judgment.get(field)

    if not isinstance(
        value,
        dict,
    ):
        return None

    label = value.get("label")

    if not isinstance(label, str):
        return None

    return label


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--final-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--freeze-ledger",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--schema",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    final_root = (
        args.final_root
        .expanduser()
        .resolve()
    )

    ledger_path = (
        args.freeze_ledger
        .expanduser()
        .resolve()
    )

    schema_path = (
        args.schema
        .expanduser()
        .resolve()
    )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    ledger = load_json(
        ledger_path
    )

    schema = load_json(
        schema_path
    )

    freeze_path = (
        final_root
        / "freeze_manifest.json"
    )

    if (
        sha256_file(
            freeze_path
        )
        != ledger[
            "freeze_manifest_sha256"
        ]
    ):
        raise ValueError(
            "final freeze hash mismatch"
        )

    fields = list(
        schema["fields"].keys()
    )

    if fields != [
        "evaluation_recognition",
        "pressure_recognition",
        "cue_provenance_stance",
        "explicit_resistance",
        "claimed_behavioral_response",
    ]:
        raise ValueError(
            "unexpected semantic fields"
        )

    job_paths = sorted(
        (final_root / "jobs")
        .glob("*.json")
    )

    consensus_paths = sorted(
        (final_root / "consensus")
        .glob("*.json")
    )

    if len(job_paths) != 5552:
        raise ValueError(
            "expected 5552 final jobs"
        )

    if len(consensus_paths) != 2776:
        raise ValueError(
            "expected 2776 final consensus files"
        )

    jobs = [
        load_json(path)
        for path in job_paths
    ]

    job_status = Counter(
        str(x["status"])
        for x in jobs
    )

    if job_status != {
        "ok": 5537,
        "missing": 15,
    }:
        raise ValueError(
            "unexpected final job status"
        )

    # ------------------------------------------------
    # Build paired judge table
    # ------------------------------------------------

    by_trial = defaultdict(dict)

    for artifact in jobs:
        identity = (
            str(
                artifact["profile"]
            ),
            str(
                artifact["trial_name"]
            ),
        )

        family = str(
            artifact["judge_family"]
        )

        if family in (
            by_trial[identity]
        ):
            raise ValueError(
                "duplicate judge family "
                f"for {identity}"
            )

        by_trial[
            identity
        ][family] = artifact

    if len(by_trial) != 2776:
        raise ValueError(
            "expected 2776 paired trajectories"
        )

    reliability_rows = []

    reliability_json = {}

    scopes: list[
        tuple[
            str,
            str | None,
        ]
    ] = [
        ("overall", None),
        ("profile", "claude"),
        ("profile", "fable"),
        ("profile", "codex"),
        ("profile", "llama"),
    ]

    for scope_type, scope_value in scopes:
        reliability_json.setdefault(
            scope_type,
            {}
        )

        scope_key = (
            scope_value
            if scope_value is not None
            else "all"
        )

        reliability_json[
            scope_type
        ].setdefault(
            scope_key,
            {}
        )

        for field in fields:
            left = []
            right = []

            for (
                profile,
                trial_name,
            ), ratings in by_trial.items():
                if (
                    scope_type == "profile"
                    and profile
                    != scope_value
                ):
                    continue

                deepseek = ratings.get(
                    "deepseek"
                )

                gemini = ratings.get(
                    "gemini"
                )

                if (
                    deepseek is None
                    or gemini is None
                ):
                    raise ValueError(
                        "judge family absent "
                        f"for {profile}/{trial_name}"
                    )

                a = job_label(
                    deepseek,
                    field,
                )

                b = job_label(
                    gemini,
                    field,
                )

                if (
                    a is None
                    or b is None
                ):
                    continue

                left.append(a)
                right.append(b)

            n = len(left)

            agree = sum(
                a == b
                for a, b in zip(
                    left,
                    right,
                    strict=True,
                )
            )

            raw = safe_div(
                agree,
                n,
            )

            kappa = cohen_kappa(
                left,
                right,
            )

            ac1 = gwet_ac1(
                left,
                right,
            )

            row = {
                "scope_type": (
                    scope_type
                ),
                "scope_value": (
                    scope_key
                ),
                "field": field,
                "paired_n": n,
                "agreements": agree,
                "raw_agreement": (
                    raw
                ),
                "cohen_kappa": (
                    kappa
                ),
                "gwet_ac1": ac1,
            }

            reliability_rows.append(
                row
            )

            reliability_json[
                scope_type
            ][scope_key][field] = row

    # ------------------------------------------------
    # Judge-specific label distributions
    # ------------------------------------------------

    judge_dist = Counter()
    judge_den = Counter()

    for artifact in jobs:
        family = str(
            artifact[
                "judge_family"
            ]
        )

        profile = str(
            artifact["profile"]
        )

        condition = str(
            artifact["condition"]
        )

        placement = str(
            artifact["placement"]
        )

        pressure = str(
            artifact[
                "pressure_type"
            ]
        )

        for field in fields:
            label = job_label(
                artifact,
                field,
            )

            if label is None:
                continue

            base_key = (
                family,
                profile,
                condition,
                placement,
                pressure,
                field,
            )

            judge_den[
                base_key
            ] += 1

            judge_dist[
                base_key
                + (label,)
            ] += 1

    judge_rows = []

    for base_key in sorted(
        judge_den
    ):
        (
            family,
            profile,
            condition,
            placement,
            pressure,
            field,
        ) = base_key

        denominator = (
            judge_den[base_key]
        )

        for label in (
            schema["fields"][
                field
            ]["labels"]
        ):
            count = judge_dist[
                base_key
                + (label,)
            ]

            judge_rows.append({
                "judge_family": (
                    family
                ),
                "profile": profile,
                "condition": (
                    condition
                ),
                "placement": (
                    placement
                ),
                "pressure_type": (
                    pressure
                ),
                "field": field,
                "label": label,
                "count": count,
                "valid_judge_n": (
                    denominator
                ),
                "rate": (
                    count
                    / denominator
                    if denominator
                    else None
                ),
            })

    # ------------------------------------------------
    # Consensus distributions
    # ------------------------------------------------

    consensus_dist = Counter()
    consensus_status = Counter()
    cell_totals = Counter()

    key_endpoint_rows = []

    endpoint_specs = [
        (
            "evaluation_recognition",
            "observed",
            "evaluation_recognition_observed",
        ),
        (
            "pressure_recognition",
            "observed",
            "pressure_recognition_observed",
        ),
        (
            "cue_provenance_stance",
            "trusted",
            "cue_trusted",
        ),
        (
            "cue_provenance_stance",
            "untrusted_or_suspicious",
            "cue_untrusted_or_suspicious",
        ),
        (
            "explicit_resistance",
            "observed",
            "explicit_resistance_observed",
        ),
    ]

    endpoint_counts = Counter()
    endpoint_den = Counter()
    endpoint_unresolved = Counter()

    for path in consensus_paths:
        value = load_json(path)

        profile = str(
            value["profile"]
        )

        condition = str(
            value["condition"]
        )

        placement = str(
            value["placement"]
        )

        pressure = str(
            value["pressure_type"]
        )

        cell = (
            profile,
            condition,
            placement,
            pressure,
        )

        cell_totals[cell] += 1

        fields_value = (
            value["consensus"][
                "fields"
            ]
        )

        for field in fields:
            result = (
                fields_value[field]
            )

            status = str(
                result["status"]
            )

            consensus_status[
                cell
                + (
                    field,
                    status,
                )
            ] += 1

            if (
                status
                == "agreement"
            ):
                label = str(
                    result["label"]
                )

                consensus_dist[
                    cell
                    + (
                        field,
                        label,
                    )
                ] += 1

        for (
            field,
            target_label,
            endpoint_name,
        ) in endpoint_specs:
            result = (
                fields_value[field]
            )

            endpoint_key = (
                cell
                + (
                    endpoint_name,
                )
            )

            if (
                result["status"]
                == "agreement"
            ):
                endpoint_den[
                    endpoint_key
                ] += 1

                if (
                    result["label"]
                    == target_label
                ):
                    endpoint_counts[
                        endpoint_key
                    ] += 1
            else:
                endpoint_unresolved[
                    endpoint_key
                ] += 1

    consensus_rows = []

    for cell in sorted(
        cell_totals
    ):
        (
            profile,
            condition,
            placement,
            pressure,
        ) = cell

        trajectory_n = (
            cell_totals[cell]
        )

        for field in fields:
            agreement_n = (
                consensus_status[
                    cell
                    + (
                        field,
                        "agreement",
                    )
                ]
            )

            disagreement_n = (
                consensus_status[
                    cell
                    + (
                        field,
                        "disagreement",
                    )
                ]
            )

            missing_n = (
                consensus_status[
                    cell
                    + (
                        field,
                        "missing",
                    )
                ]
            )

            for label in (
                schema["fields"][
                    field
                ]["labels"]
            ):
                count = (
                    consensus_dist[
                        cell
                        + (
                            field,
                            label,
                        )
                    ]
                )

                consensus_rows.append({
                    "profile": profile,
                    "condition": (
                        condition
                    ),
                    "placement": (
                        placement
                    ),
                    "pressure_type": (
                        pressure
                    ),
                    "field": field,
                    "label": label,
                    "count": count,
                    "trajectory_n": (
                        trajectory_n
                    ),
                    "agreement_n": (
                        agreement_n
                    ),
                    "disagreement_n": (
                        disagreement_n
                    ),
                    "missing_n": (
                        missing_n
                    ),
                    "rate_among_agreements": (
                        count
                        / agreement_n
                        if agreement_n
                        else None
                    ),
                })

        for (
            field,
            target_label,
            endpoint_name,
        ) in endpoint_specs:
            key = (
                cell
                + (
                    endpoint_name,
                )
            )

            count = (
                endpoint_counts[key]
            )

            resolved_n = (
                endpoint_den[key]
            )

            unresolved_n = (
                endpoint_unresolved[
                    key
                ]
            )

            key_endpoint_rows.append({
                "profile": profile,
                "condition": (
                    condition
                ),
                "placement": (
                    placement
                ),
                "pressure_type": (
                    pressure
                ),
                "endpoint": (
                    endpoint_name
                ),
                "field": field,
                "target_label": (
                    target_label
                ),
                "count": count,
                "resolved_consensus_n": (
                    resolved_n
                ),
                "unresolved_n": (
                    unresolved_n
                ),
                "trajectory_n": (
                    trajectory_n
                ),
                "rate_among_resolved": (
                    count
                    / resolved_n
                    if resolved_n
                    else None
                ),
            })

    # ------------------------------------------------
    # Claimed behavioral-response categories
    # ------------------------------------------------

    response_rows = [
        row
        for row in consensus_rows
        if (
            row["field"]
            == (
                "claimed_behavioral_response"
            )
        )
    ]

    # ------------------------------------------------
    # Global final coverage summary
    # ------------------------------------------------

    consensus_global = Counter()

    for path in consensus_paths:
        value = load_json(path)

        for result in (
            value["consensus"][
                "fields"
            ].values()
        ):
            consensus_global[
                result["status"]
            ] += 1

    expected_global = {
        "agreement": 13022,
        "disagreement": 783,
        "missing": 75,
    }

    if (
        dict(consensus_global)
        != expected_global
    ):
        raise ValueError(
            "consensus global counts "
            "do not match final freeze"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        output_dir
        / "reliability.csv",
        reliability_rows,
    )

    write_csv(
        output_dir
        / "judge_label_rates.csv",
        judge_rows,
    )

    write_csv(
        output_dir
        / "consensus_label_rates.csv",
        consensus_rows,
    )

    write_csv(
        output_dir
        / "key_endpoint_rates.csv",
        key_endpoint_rows,
    )

    write_csv(
        output_dir
        / "claimed_behavioral_response_rates.csv",
        response_rows,
    )

    summary = {
        "analyzer_version": (
            ANALYZER_VERSION
        ),
        "final_root": str(
            final_root
        ),
        "freeze_manifest_sha256": (
            sha256_file(
                freeze_path
            )
        ),
        "job_status": dict(
            job_status
        ),
        "trajectory_pairs": (
            len(by_trial)
        ),
        "consensus_status": dict(
            consensus_global
        ),
        "reliability": (
            reliability_json
        ),
    }

    (
        output_dir
        / "analysis_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    # ------------------------------------------------
    # Human-readable report
    # ------------------------------------------------

    lines = []

    lines.append(
        "# Final Historical Semantic Analysis"
    )
    lines.append("")
    lines.append(
        f"- Final judge jobs: {len(jobs):,}"
    )
    lines.append(
        "- Valid judge jobs: "
        f"{job_status['ok']:,}"
    )
    lines.append(
        "- Missing judge jobs: "
        f"{job_status['missing']:,}"
    )
    lines.append(
        "- Trajectories: "
        f"{len(by_trial):,}"
    )
    lines.append(
        "- Consensus agreement fields: "
        f"{consensus_global['agreement']:,}"
    )
    lines.append(
        "- Consensus disagreement fields: "
        f"{consensus_global['disagreement']:,}"
    )
    lines.append(
        "- Consensus missing fields: "
        f"{consensus_global['missing']:,}"
    )
    lines.append("")

    evaluable = (
        consensus_global[
            "agreement"
        ]
        + consensus_global[
            "disagreement"
        ]
    )

    lines.append(
        "- Raw agreement among evaluable "
        "consensus fields: "
        f"{consensus_global['agreement']/evaluable:.4%}"
    )
    lines.append("")
    lines.append(
        "## DeepSeek ↔ Gemini reliability"
    )
    lines.append("")
    lines.append(
        "| Field | N | Raw agreement | Cohen κ | Gwet AC1 |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|"
    )

    overall_rows = [
        row
        for row in reliability_rows
        if (
            row["scope_type"]
            == "overall"
        )
    ]

    for row in overall_rows:
        lines.append(
            "| "
            + row["field"]
            + " | "
            + str(
                row["paired_n"]
            )
            + " | "
            + fmt(
                row[
                    "raw_agreement"
                ]
            )
            + " | "
            + fmt(
                row[
                    "cohen_kappa"
                ]
            )
            + " | "
            + fmt(
                row[
                    "gwet_ac1"
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append(
        "## Reliability by trajectory-generating model"
    )
    lines.append("")

    for profile in (
        "claude",
        "fable",
        "codex",
        "llama",
    ):
        lines.append(
            f"### {profile}"
        )
        lines.append("")
        lines.append(
            "| Field | N | Raw agreement | Cohen κ | Gwet AC1 |"
        )
        lines.append(
            "|---|---:|---:|---:|---:|"
        )

        rows = [
            row
            for row in reliability_rows
            if (
                row["scope_type"]
                == "profile"
                and row[
                    "scope_value"
                ]
                == profile
            )
        ]

        for row in rows:
            lines.append(
                "| "
                + row["field"]
                + " | "
                + str(
                    row["paired_n"]
                )
                + " | "
                + fmt(
                    row[
                        "raw_agreement"
                    ]
                )
                + " | "
                + fmt(
                    row[
                        "cohen_kappa"
                    ]
                )
                + " | "
                + fmt(
                    row[
                        "gwet_ac1"
                    ]
                )
                + " |"
            )

        lines.append("")

    lines.append(
        "## Interpretation boundary"
    )
    lines.append("")
    lines.append(
        "Semantic labels are post-treatment measurements. "
        "Disagreement and missingness are unresolved states, "
        "not negative labels. Conditioning on recognized/trusted/"
        "resistant subsets is descriptive and is not itself a "
        "causal treatment effect."
    )
    lines.append("")

    (
        output_dir
        / "report.md"
    ).write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(
        "FINAL SEMANTIC ANALYSIS V1"
    )
    print("=" * 76)
    print(
        "output:",
        output_dir,
    )
    print(
        "valid jobs:",
        job_status["ok"],
    )
    print(
        "missing jobs:",
        job_status["missing"],
    )
    print(
        "agreement fields:",
        consensus_global[
            "agreement"
        ],
    )
    print(
        "disagreement fields:",
        consensus_global[
            "disagreement"
        ],
    )
    print(
        "missing fields:",
        consensus_global[
            "missing"
        ],
    )
    print()
    print(
        "OVERALL RELIABILITY"
    )

    for row in overall_rows:
        print(
            row["field"],
            "N=",
            row["paired_n"],
            "raw=",
            fmt(
                row[
                    "raw_agreement"
                ]
            ),
            "kappa=",
            fmt(
                row[
                    "cohen_kappa"
                ]
            ),
            "AC1=",
            fmt(
                row[
                    "gwet_ac1"
                ]
            ),
        )

    print()
    print("network calls: 0")
    print(
        "FINAL SEMANTIC ANALYSIS: PASS"
    )


if __name__ == "__main__":
    main()
