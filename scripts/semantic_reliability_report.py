#!/usr/bin/env python3
"""Outcome-blind reliability summary for semantic judging.

This report inspects only:
- terminal job availability;
- judge identity;
- retry counts and failure modes;
- consensus agreement/disagreement/missing status;
- append-only execution provenance.

It does NOT inspect semantic labels, evidence contents,
treatment effects, pass/fail outcomes, or behavioral outcomes.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


REPORT_VERSION = "1.0"


def load_json(
    path: Path,
) -> dict[str, Any]:
    value = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(value, dict):
        raise ValueError(
            f"{path}: expected JSON object"
        )

    return value


def load_jobs(
    root: Path,
    source: str,
) -> list[dict[str, Any]]:
    rows = []

    for path in sorted(
        (root / "jobs").glob(
            "*.json"
        )
    ):
        artifact = load_json(path)

        rows.append({
            "source": source,
            "path": str(path),
            "cache_key": str(
                artifact["cache_key"]
            ),
            "status": str(
                artifact["status"]
            ),
            "judge_family": str(
                artifact[
                    "judge_family"
                ]
            ),
            "judge_model": str(
                artifact[
                    "judge_model"
                ]
            ),
            "attempt_count": int(
                artifact[
                    "attempt_count"
                ]
            ),
            "attempts": (
                artifact.get(
                    "attempts",
                    [],
                )
            ),
        })

    return rows


def attempt_failure_summary(
    rows: list[
        dict[str, Any]
    ],
) -> dict[str, Any]:
    status_counts = Counter()
    exception_counts = Counter()
    validation_counts = Counter()

    failed_attempts = 0

    for row in rows:
        for attempt in row[
            "attempts"
        ]:
            status = str(
                attempt.get(
                    "status",
                    "",
                )
            )

            if status == "ok":
                continue

            failed_attempts += 1

            status_counts[
                status or "<empty>"
            ] += 1

            exception = str(
                attempt.get(
                    "exception_type",
                    "",
                )
                or ""
            )

            if exception:
                exception_counts[
                    exception
                ] += 1

            validation = str(
                attempt.get(
                    "validation_error",
                    "",
                )
                or ""
            )

            if validation:
                validation_counts[
                    validation
                ] += 1

    return {
        "failed_attempts": (
            failed_attempts
        ),
        "status_counts": dict(
            sorted(
                status_counts.items()
            )
        ),
        "exception_type_counts": (
            dict(
                sorted(
                    exception_counts.items()
                )
            )
        ),
        "validation_error_counts": (
            dict(
                sorted(
                    validation_counts.items()
                )
            )
        ),
    }


def consensus_summary(
    dose_root: Path,
) -> dict[str, Any]:
    paths = sorted(
        (
            dose_root
            / "consensus"
        ).glob(
            "*.json"
        )
    )

    statuses = Counter()

    for path in paths:
        value = load_json(path)

        fields = (
            value[
                "consensus"
            ]["fields"]
        )

        for result in (
            fields.values()
        ):
            statuses[
                str(
                    result[
                        "status"
                    ]
                )
            ] += 1

    total = sum(
        statuses.values()
    )

    agreement = statuses[
        "agreement"
    ]

    disagreement = statuses[
        "disagreement"
    ]

    missing = statuses[
        "missing"
    ]

    evaluable = (
        agreement
        + disagreement
    )

    return {
        "trajectory_files": (
            len(paths)
        ),
        "field_comparisons": (
            total
        ),
        "agreement": agreement,
        "disagreement": (
            disagreement
        ),
        "missing": missing,
        "evaluable": evaluable,
        "raw_agreement_given_evaluable": (
            agreement / evaluable
            if evaluable
            else None
        ),
        "missing_field_rate": (
            missing / total
            if total
            else None
        ),
    }


def run_history_summary(
    dose_root: Path,
) -> list[dict[str, Any]]:
    rows = []

    for path in sorted(
        (
            dose_root
            / "runs"
        ).glob(
            "run-*.json"
        )
    ):
        value = load_json(path)

        rows.append({
            "path": str(path),
            "run_id": (
                value.get(
                    "run_id"
                )
            ),
            "new_jobs": int(
                value.get(
                    "new_jobs",
                    0,
                )
            ),
            "provider_attempts": int(
                value.get(
                    "provider_attempts",
                    0,
                )
            ),
            "after": dict(
                value.get(
                    "after",
                    {},
                )
            ),
        })

    return rows


def build_report(
    *,
    pilot_root: Path,
    dose_root: Path,
) -> dict[str, Any]:
    pilot_rows = load_jobs(
        pilot_root,
        "inherited_pilot",
    )

    dose_rows = load_jobs(
        dose_root,
        "dose_writable",
    )

    rows = (
        pilot_rows
        + dose_rows
    )

    if len(pilot_rows) != 20:
        raise ValueError(
            "expected 20 inherited "
            "pilot judge jobs"
        )

    if len(dose_rows) != 180:
        raise ValueError(
            "expected 180 writable "
            "dose judge jobs"
        )

    if len(rows) != 200:
        raise ValueError(
            "expected 200 effective "
            "Stage C judge jobs"
        )

    cache_keys = [
        row["cache_key"]
        for row in rows
    ]

    if len(
        set(cache_keys)
    ) != 200:
        raise ValueError(
            "effective Stage C cache "
            "keys are not unique"
        )

    status_counts = Counter(
        row["status"]
        for row in rows
    )

    if status_counts != {
        "ok": 199,
        "missing": 1,
    }:
        raise ValueError(
            "unexpected Stage C "
            f"terminal status counts: "
            f"{dict(status_counts)}"
        )

    attempt_distribution = (
        Counter(
            row[
                "attempt_count"
            ]
            for row in rows
        )
    )

    total_attempts = sum(
        row["attempt_count"]
        for row in rows
    )

    retry_jobs = sum(
        row[
            "attempt_count"
        ] > 1
        for row in rows
    )

    per_judge: dict[
        str,
        dict[str, Any],
    ] = {}

    models = sorted({
        row["judge_model"]
        for row in rows
    })

    for model in models:
        subset = [
            row
            for row in rows
            if (
                row[
                    "judge_model"
                ]
                == model
            )
        ]

        counts = Counter(
            row["status"]
            for row in subset
        )

        per_judge[model] = {
            "jobs": len(subset),
            "ok": counts["ok"],
            "missing": (
                counts["missing"]
            ),
            "coverage": (
                counts["ok"]
                / len(subset)
            ),
            "provider_attempts": (
                sum(
                    row[
                        "attempt_count"
                    ]
                    for row
                    in subset
                )
            ),
            "retry_jobs": sum(
                row[
                    "attempt_count"
                ] > 1
                for row
                in subset
            ),
        }

    failures = (
        attempt_failure_summary(
            rows
        )
    )

    consensus = (
        consensus_summary(
            dose_root
        )
    )

    history = (
        run_history_summary(
            dose_root
        )
    )

    if (
        consensus[
            "trajectory_files"
        ]
        != 100
        or consensus[
            "field_comparisons"
        ]
        != 500
    ):
        raise ValueError(
            "unexpected consensus "
            "coverage"
        )

    if not any(
        (
            run["new_jobs"] == 180
            and run[
                "provider_attempts"
            ] == 188
        )
        for run in history
    ):
        raise ValueError(
            "original Stage C run "
            "record not found"
        )

    if not any(
        (
            run["new_jobs"] == 0
            and run[
                "provider_attempts"
            ] == 0
        )
        for run in history
    ):
        raise ValueError(
            "zero-call resume run "
            "record not found"
        )

    return {
        "report_version": (
            REPORT_VERSION
        ),
        "scope": (
            "stage_c_semantic_measurement_"
            "reliability_only"
        ),
        "outcome_blind": True,
        "effective_trajectories": 100,
        "effective_judge_jobs": 200,
        "inherited_pilot_jobs": (
            len(pilot_rows)
        ),
        "new_dose_jobs": (
            len(dose_rows)
        ),
        "terminal_job_status": dict(
            status_counts
        ),
        "judge_job_coverage": (
            status_counts["ok"]
            / len(rows)
        ),
        "attempts": {
            "provider_attempts": (
                total_attempts
            ),
            "excess_attempts": (
                total_attempts
                - len(rows)
            ),
            "retry_jobs": retry_jobs,
            "retry_job_rate": (
                retry_jobs
                / len(rows)
            ),
            "attempt_count_distribution": (
                {
                    str(k): v
                    for k, v
                    in sorted(
                        attempt_distribution.items()
                    )
                }
            ),
        },
        "per_judge": per_judge,
        "failed_attempts": failures,
        "consensus": consensus,
        "run_history": history,
    }


def percent(
    value: float | None,
) -> str:
    if value is None:
        return "NA"

    return (
        f"{100.0 * value:.2f}%"
    )


def markdown_report(
    report: dict[str, Any],
) -> str:
    attempts = report[
        "attempts"
    ]

    consensus = report[
        "consensus"
    ]

    lines = [
        "# Stage C Semantic Judge Reliability",
        "",
        (
            "This report is outcome-blind. "
            "It summarizes measurement "
            "availability, retries, validation "
            "failures, judge coverage, "
            "consensus status, and resume "
            "provenance only."
        ),
        "",
        "## Coverage",
        "",
        (
            f"- Effective trajectories: "
            f"{report['effective_trajectories']}"
        ),
        (
            f"- Core judge jobs: "
            f"{report['effective_judge_jobs']}"
        ),
        (
            f"- Valid judge jobs: "
            f"{report['terminal_job_status'].get('ok', 0)}"
        ),
        (
            f"- Missing judge jobs: "
            f"{report['terminal_job_status'].get('missing', 0)}"
        ),
        (
            "- Judge-job coverage: "
            + percent(
                report[
                    "judge_job_coverage"
                ]
            )
        ),
        "",
        "## Retry behavior",
        "",
        (
            f"- Provider attempts: "
            f"{attempts['provider_attempts']}"
        ),
        (
            f"- Excess attempts: "
            f"{attempts['excess_attempts']}"
        ),
        (
            f"- Jobs requiring retry: "
            f"{attempts['retry_jobs']} "
            f"({percent(attempts['retry_job_rate'])})"
        ),
        (
            "- Attempt-count distribution: "
            f"{attempts['attempt_count_distribution']}"
        ),
        "",
        "## Core-judge consensus",
        "",
        (
            f"- Field comparisons: "
            f"{consensus['field_comparisons']}"
        ),
        (
            f"- Agreement: "
            f"{consensus['agreement']}"
        ),
        (
            f"- Disagreement: "
            f"{consensus['disagreement']}"
        ),
        (
            f"- Missing: "
            f"{consensus['missing']}"
        ),
        (
            "- Raw agreement among evaluable "
            "fields: "
            + percent(
                consensus[
                    "raw_agreement_given_evaluable"
                ]
            )
        ),
        (
            "- Missing-field rate: "
            + percent(
                consensus[
                    "missing_field_rate"
                ]
            )
        ),
        "",
        "## Interpretation",
        "",
        (
            "Stage C is operationally suitable "
            "for production-scale semantic "
            "measurement. Terminal missingness "
            "is retained as missing rather than "
            "converted to a negative label or "
            "retried beyond the frozen attempt "
            "budget."
        ),
        "",
    ]

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pilot-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--dose-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--json-output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--md-output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    report = build_report(
        pilot_root=(
            args.pilot_root
            .expanduser()
            .resolve()
        ),
        dose_root=(
            args.dose_root
            .expanduser()
            .resolve()
        ),
    )

    args.json_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.md_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.json_output.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    args.md_output.write_text(
        markdown_report(
            report
        ),
        encoding="utf-8",
    )

    print(
        "effective trajectories:",
        report[
            "effective_trajectories"
        ],
    )

    print(
        "effective judge jobs:",
        report[
            "effective_judge_jobs"
        ],
    )

    print(
        "valid judge jobs:",
        report[
            "terminal_job_status"
        ]["ok"],
    )

    print(
        "missing judge jobs:",
        report[
            "terminal_job_status"
        ]["missing"],
    )

    print(
        "provider attempts:",
        report[
            "attempts"
        ][
            "provider_attempts"
        ],
    )

    print(
        "retry jobs:",
        report[
            "attempts"
        ]["retry_jobs"],
    )

    print(
        "agreement:",
        report[
            "consensus"
        ]["agreement"],
    )

    print(
        "disagreement:",
        report[
            "consensus"
        ]["disagreement"],
    )

    print(
        "missing fields:",
        report[
            "consensus"
        ]["missing"],
    )

    print(
        "raw agreement among "
        "evaluable fields:",
        percent(
            report[
                "consensus"
            ][
                "raw_agreement_given_evaluable"
            ]
        ),
    )

    print()
    print(
        "STAGE C RELIABILITY "
        "REPORT: PASS"
    )


if __name__ == "__main__":
    main()
