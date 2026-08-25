#!/usr/bin/env python3
"""Reconstruct the frozen 840-cell resource-deprivation study.

This script establishes analysis denominators and provenance only.
It does not estimate treatment effects.

Classification policy is loaded from:
config/resource_reconstruction_policy_v1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


RECONSTRUCTION_VERSION = "1.0"

EXPECTED_TOTAL = 840
EXPECTED_PROFILES = (
    "claude",
    "fable",
    "codex",
    "llama",
)
EXPECTED_CONDITIONS = (
    "clean",
    "eval_only",
    "eval_resource",
)
EXPECTED_PER_PROFILE = 210
EXPECTED_PER_CONDITION = 70
EXPECTED_BASE_TASKS = 70


def load_json(
    path: Path,
) -> Any:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def resolve_result(
    *,
    result_root: Path,
    trial_name: str,
) -> Path:
    inner = (
        result_root
        / result_root.name
    )

    exact = (
        inner
        / trial_name
        / "result.json"
    )

    if exact.is_file():
        return exact

    matches = sorted(
        path / "result.json"
        for path
        in inner.glob(
            f"{trial_name}__*"
        )
        if (
            path.is_dir()
            and (
                path
                / "result.json"
            ).is_file()
        )
    )

    if len(matches) != 1:
        raise ValueError(
            f"{trial_name}: expected "
            "exactly one Harbor result; "
            f"found {len(matches)}"
        )

    return matches[0]


def find_trajectory(
    trial_dir: Path,
) -> Path | None:
    preferred = (
        "agent/trajectory.json",
        "agent/minisuite_trajectory.json",
        "agent/mini-swe-agent.trajectory.json",
    )

    for name in preferred:
        path = (
            trial_dir
            / name
        )

        if path.is_file():
            return path

    matches = sorted(
        path
        for path
        in trial_dir.glob(
            "**/*trajectory*.json"
        )
        if path.is_file()
    )

    if len(matches) == 1:
        return matches[0]

    if matches:
        # Deterministic fallback only.
        return matches[0]

    return None


def parse_dt(
    value: Any,
) -> datetime | None:
    if not value:
        return None

    try:
        text = str(value).replace(
            "Z",
            "+00:00",
        )

        return datetime.fromisoformat(
            text
        )

    except Exception:
        return None


def duration_seconds(
    start: Any,
    finish: Any,
) -> float | None:
    a = parse_dt(start)
    b = parse_dt(finish)

    if (
        a is None
        or b is None
    ):
        return None

    return (
        b - a
    ).total_seconds()


def numeric(
    value: Any,
) -> float | None:
    if value is None:
        return None

    try:
        return float(value)

    except Exception:
        return None


def classification(
    *,
    exception_type: str | None,
    policy: dict[str, Any],
) -> str:
    if not exception_type:
        return (
            "substantive_model_outcome"
        )

    if exception_type in set(
        policy[
            "substantive_model_outcome_exceptions"
        ]
    ):
        return (
            "substantive_model_outcome"
        )

    if exception_type in set(
        policy[
            "infrastructure_censored_exceptions"
        ]
    ):
        return (
            "infrastructure_censored"
        )

    if exception_type in set(
        policy[
            "needs_adjudication_exceptions"
        ]
    ):
        return (
            "needs_adjudication"
        )

    return "needs_adjudication"


def excerpt(
    value: Any,
    limit: int = 500,
) -> str:
    if value is None:
        return ""

    text = str(value)

    if len(text) <= limit:
        return text

    return (
        text[:limit]
        + " ... [truncated]"
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw-manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--policy",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    raw_manifest_path = (
        args.raw_manifest
        .expanduser()
        .resolve()
    )

    policy_path = (
        args.policy
        .expanduser()
        .resolve()
    )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    manifest = load_json(
        raw_manifest_path
    )

    policy = load_json(
        policy_path
    )

    planned = (
        manifest[
            "planned_trials"
        ]
    )

    if len(planned) != EXPECTED_TOTAL:
        raise ValueError(
            "raw manifest does not "
            "contain 840 planned cells"
        )

    shard_roots = {
        (
            str(
                shard["profile"]
            ),
            int(
                shard["chunk"]
            ),
        ): Path(
            shard["result_root"]
        )
        for shard
        in manifest["shards"]
    }

    rows = []

    for planned_row in planned:
        profile = str(
            planned_row["profile"]
        )

        condition = str(
            planned_row["condition"]
        )

        chunk = int(
            planned_row["chunk"]
        )

        planned_name = str(
            planned_row[
                "trial_name"
            ]
        )

        result_path = (
            resolve_result(
                result_root=(
                    shard_roots[
                        (
                            profile,
                            chunk,
                        )
                    ]
                ),
                trial_name=(
                    planned_name
                ),
            )
        )

        value = load_json(
            result_path
        )

        if (
            str(
                value.get(
                    "task_name"
                )
            )
            != planned_name
        ):
            raise ValueError(
                f"{planned_name}: "
                "Harbor task_name mismatch"
            )

        harbor_trial_name = str(
            value.get(
                "trial_name",
                "",
            )
        )

        if not (
            harbor_trial_name
            == planned_name
            or harbor_trial_name.startswith(
                planned_name + "__"
            )
        ):
            raise ValueError(
                f"{planned_name}: "
                "Harbor trial_name mismatch"
            )

        exception = (
            value.get(
                "exception_info"
            )
        )

        if isinstance(
            exception,
            dict,
        ):
            exception_type = (
                str(
                    exception.get(
                        "exception_type",
                        "",
                    )
                )
                or None
            )

            exception_message = (
                exception.get(
                    "exception_message"
                )
            )

            exception_traceback = (
                exception.get(
                    "exception_traceback"
                )
            )

        else:
            exception_type = None
            exception_message = None
            exception_traceback = None

        analysis_class = (
            classification(
                exception_type=(
                    exception_type
                ),
                policy=policy,
            )
        )

        verifier_result = (
            value.get(
                "verifier_result"
            )
        )

        rewards = (
            verifier_result.get(
                "rewards"
            )
            if isinstance(
                verifier_result,
                dict,
            )
            else None
        )

        if not isinstance(
            rewards,
            dict,
        ):
            rewards = {}

        overall_pass = numeric(
            rewards.get(
                "overall_pass"
            )
        )

        tests_reward = numeric(
            rewards.get(
                "tests_reward"
            )
        )

        must_have_pass = numeric(
            rewards.get(
                "must_have_pass"
            )
        )

        rubric_score = numeric(
            rewards.get(
                "rubrics_agg_score"
            )
        )

        verifier_outcome_available = (
            overall_pass is not None
        )

        trajectory_path = (
            find_trajectory(
                result_path.parent
            )
        )

        trajectory_available = (
            trajectory_path
            is not None
        )

        agent_result = (
            value.get(
                "agent_result"
            )
        )

        if not isinstance(
            agent_result,
            dict,
        ):
            agent_result = {}

        agent_execution = (
            value.get(
                "agent_execution"
            )
        )

        behavior_available = (
            analysis_class
            == "substantive_model_outcome"
            and trajectory_available
        )

        performance_available = (
            analysis_class
            == "substantive_model_outcome"
            and verifier_outcome_available
        )

        row = {
            "profile": profile,
            "chunk": chunk,
            "base_task": str(
                planned_row[
                    "base_task"
                ]
            ),
            "condition": condition,
            "planned_trial_name": (
                planned_name
            ),
            "harbor_trial_name": (
                harbor_trial_name
            ),
            "analysis_class": (
                analysis_class
            ),
            "exception_type": (
                exception_type
            ),
            "exception_message_excerpt": (
                excerpt(
                    exception_message
                )
            ),
            "exception_traceback_sha256": (
                hashlib.sha256(
                    str(
                        exception_traceback
                    ).encode(
                        "utf-8",
                        errors="replace",
                    )
                ).hexdigest()
                if exception_traceback
                else None
            ),
            "result_path": str(
                result_path
            ),
            "result_sha256": (
                sha256_file(
                    result_path
                )
            ),
            "trajectory_path": (
                str(
                    trajectory_path
                )
                if trajectory_path
                else None
            ),
            "trajectory_sha256": (
                sha256_file(
                    trajectory_path
                )
                if trajectory_path
                else None
            ),
            "trajectory_available": (
                trajectory_available
            ),
            "agent_result_available": (
                bool(
                    value.get(
                        "agent_result"
                    )
                )
            ),
            "agent_execution_available": (
                isinstance(
                    agent_execution,
                    dict,
                )
            ),
            "verifier_outcome_available": (
                verifier_outcome_available
            ),
            "behavior_available": (
                behavior_available
            ),
            "performance_available": (
                performance_available
            ),
            "overall_pass": (
                overall_pass
            ),
            "tests_reward": (
                tests_reward
            ),
            "must_have_pass": (
                must_have_pass
            ),
            "rubrics_agg_score": (
                rubric_score
            ),
            "input_tokens": numeric(
                agent_result.get(
                    "n_input_tokens"
                )
            ),
            "cache_tokens": numeric(
                agent_result.get(
                    "n_cache_tokens"
                )
            ),
            "output_tokens": numeric(
                agent_result.get(
                    "n_output_tokens"
                )
            ),
            "cost_usd": numeric(
                agent_result.get(
                    "cost_usd"
                )
            ),
            "duration_sec": (
                duration_seconds(
                    value.get(
                        "started_at"
                    ),
                    value.get(
                        "finished_at"
                    ),
                )
            ),
        }

        rows.append(
            row
        )

    if len(rows) != EXPECTED_TOTAL:
        raise ValueError(
            "reconstruction did not "
            "produce 840 rows"
        )

    identities = {
        (
            row["profile"],
            row[
                "planned_trial_name"
            ],
        )
        for row in rows
    }

    if len(identities) != EXPECTED_TOTAL:
        raise ValueError(
            "duplicate reconstructed "
            "trial identities"
        )

    for profile in EXPECTED_PROFILES:
        subset = [
            row
            for row in rows
            if row["profile"]
            == profile
        ]

        if (
            len(subset)
            != EXPECTED_PER_PROFILE
        ):
            raise ValueError(
                f"{profile}: expected "
                "210 reconstructed cells"
            )

        counts = Counter(
            row["condition"]
            for row in subset
        )

        if counts != {
            "clean": (
                EXPECTED_PER_CONDITION
            ),
            "eval_only": (
                EXPECTED_PER_CONDITION
            ),
            "eval_resource": (
                EXPECTED_PER_CONDITION
            ),
        }:
            raise ValueError(
                f"{profile}: condition "
                f"coverage mismatch: "
                f"{counts}"
            )

    grouped: dict[
        tuple[str, str],
        dict[str, dict[str, Any]],
    ] = defaultdict(dict)

    for row in rows:
        key = (
            row["profile"],
            row["base_task"],
        )

        if (
            row["condition"]
            in grouped[key]
        ):
            raise ValueError(
                "duplicate condition "
                f"within matched triple: "
                f"{key}"
            )

        grouped[key][
            row["condition"]
        ] = row

    if len(grouped) != 280:
        raise ValueError(
            "expected 280 "
            "profile/base-task triples"
        )

    triples = []

    expected_condition_set = set(
        EXPECTED_CONDITIONS
    )

    for (
        profile,
        base_task,
    ), cells in sorted(
        grouped.items()
    ):
        if (
            set(cells)
            != expected_condition_set
        ):
            raise ValueError(
                f"{profile}/{base_task}: "
                "incomplete condition triple"
            )

        cell_rows = [
            cells[
                condition
            ]
            for condition
            in EXPECTED_CONDITIONS
        ]

        triples.append({
            "profile": profile,
            "base_task": base_task,
            "substantive_triple": all(
                row[
                    "analysis_class"
                ]
                == (
                    "substantive_"
                    "model_outcome"
                )
                for row
                in cell_rows
            ),
            "performance_triple": all(
                row[
                    "performance_available"
                ]
                for row
                in cell_rows
            ),
            "behavior_triple": all(
                row[
                    "behavior_available"
                ]
                for row
                in cell_rows
            ),
            "cell_status": {
                condition: {
                    "analysis_class": (
                        cells[
                            condition
                        ][
                            "analysis_class"
                        ]
                    ),
                    "exception_type": (
                        cells[
                            condition
                        ][
                            "exception_type"
                        ]
                    ),
                    "verifier_outcome_available": (
                        cells[
                            condition
                        ][
                            "verifier_outcome_available"
                        ]
                    ),
                    "trajectory_available": (
                        cells[
                            condition
                        ][
                            "trajectory_available"
                        ]
                    ),
                }
                for condition
                in EXPECTED_CONDITIONS
            },
        })

    class_counts = Counter(
        row["analysis_class"]
        for row in rows
    )

    exception_counts = Counter(
        row["exception_type"]
        for row in rows
        if row[
            "exception_type"
        ]
    )

    by_profile = {}

    for profile in EXPECTED_PROFILES:
        subset = [
            row
            for row in rows
            if row["profile"]
            == profile
        ]

        by_profile[
            profile
        ] = {
            "planned": len(subset),
            "class_counts": dict(
                Counter(
                    row[
                        "analysis_class"
                    ]
                    for row
                    in subset
                )
            ),
            "exception_counts": dict(
                Counter(
                    row[
                        "exception_type"
                    ]
                    for row
                    in subset
                    if row[
                        "exception_type"
                    ]
                )
            ),
            "verifier_outcome_available": sum(
                bool(
                    row[
                        "verifier_outcome_available"
                    ]
                )
                for row
                in subset
            ),
            "trajectory_available": sum(
                bool(
                    row[
                        "trajectory_available"
                    ]
                )
                for row
                in subset
            ),
        }

    by_profile_condition = {}

    for profile in EXPECTED_PROFILES:
        for condition in (
            EXPECTED_CONDITIONS
        ):
            subset = [
                row
                for row in rows
                if (
                    row["profile"]
                    == profile
                    and row[
                        "condition"
                    ]
                    == condition
                )
            ]

            key = (
                profile
                + "×"
                + condition
            )

            by_profile_condition[
                key
            ] = {
                "planned": len(
                    subset
                ),
                "class_counts": dict(
                    Counter(
                        row[
                            "analysis_class"
                        ]
                        for row
                        in subset
                    )
                ),
                "exception_counts": dict(
                    Counter(
                        row[
                            "exception_type"
                        ]
                        for row
                        in subset
                        if row[
                            "exception_type"
                        ]
                    )
                ),
                "verifier_outcome_available": sum(
                    bool(
                        row[
                            "verifier_outcome_available"
                        ]
                    )
                    for row
                    in subset
                ),
                "trajectory_available": sum(
                    bool(
                        row[
                            "trajectory_available"
                        ]
                    )
                    for row
                    in subset
                ),
            }

    adjudication = [
        row
        for row in rows
        if (
            row[
                "analysis_class"
            ]
            == "needs_adjudication"
        )
    ]

    summary = {
        "reconstruction_version": (
            RECONSTRUCTION_VERSION
        ),
        "source_raw_manifest_sha256": (
            sha256_file(
                raw_manifest_path
            )
        ),
        "policy_sha256": (
            sha256_file(
                policy_path
            )
        ),
        "planned_trajectories": (
            len(rows)
        ),
        "profiles": list(
            EXPECTED_PROFILES
        ),
        "conditions": list(
            EXPECTED_CONDITIONS
        ),
        "base_tasks_per_profile": (
            EXPECTED_BASE_TASKS
        ),
        "matched_triples": (
            len(triples)
        ),
        "analysis_class_counts": (
            dict(class_counts)
        ),
        "exception_counts": (
            dict(exception_counts)
        ),
        "verifier_outcome_available": sum(
            bool(
                row[
                    "verifier_outcome_available"
                ]
            )
            for row in rows
        ),
        "trajectory_available": sum(
            bool(
                row[
                    "trajectory_available"
                ]
            )
            for row in rows
        ),
        "performance_available": sum(
            bool(
                row[
                    "performance_available"
                ]
            )
            for row in rows
        ),
        "behavior_available": sum(
            bool(
                row[
                    "behavior_available"
                ]
            )
            for row in rows
        ),
        "fully_substantive_triples": sum(
            bool(
                triple[
                    "substantive_triple"
                ]
            )
            for triple
            in triples
        ),
        "complete_performance_triples": sum(
            bool(
                triple[
                    "performance_triple"
                ]
            )
            for triple
            in triples
        ),
        "complete_behavior_triples": sum(
            bool(
                triple[
                    "behavior_triple"
                ]
            )
            for triple
            in triples
        ),
        "needs_adjudication": (
            len(adjudication)
        ),
        "by_profile": (
            by_profile
        ),
        "by_profile_condition": (
            by_profile_condition
        ),
        "network_calls": 0,
        "effect_estimation": False
    }

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        output_dir
        / "trials.json"
    ).write_text(
        json.dumps(
            rows,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    (
        output_dir
        / "triples.json"
    ).write_text(
        json.dumps(
            triples,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    (
        output_dir
        / "adjudication.json"
    ).write_text(
        json.dumps(
            adjudication,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

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

    print(
        "RESOURCE RECONSTRUCTION"
    )
    print("=" * 72)

    print(
        "planned trajectories:",
        len(rows),
    )
    print(
        "matched triples:",
        len(triples),
    )

    print()
    print(
        "analysis classes:",
        dict(class_counts),
    )

    print(
        "true exception types:",
        dict(exception_counts),
    )

    print()
    print(
        "verifier outcomes:",
        summary[
            "verifier_outcome_available"
        ],
    )

    print(
        "trajectories found:",
        summary[
            "trajectory_available"
        ],
    )

    print(
        "performance-available cells:",
        summary[
            "performance_available"
        ],
    )

    print(
        "behavior-available cells:",
        summary[
            "behavior_available"
        ],
    )

    print()
    print(
        "fully substantive triples:",
        summary[
            "fully_substantive_triples"
        ],
    )

    print(
        "complete performance triples:",
        summary[
            "complete_performance_triples"
        ],
    )

    print(
        "complete behavior triples:",
        summary[
            "complete_behavior_triples"
        ],
    )

    print(
        "needs adjudication:",
        len(adjudication),
    )

    print()
    print(
        "network calls: 0"
    )
    print(
        "effect estimation: 0"
    )
    print(
        "RESOURCE RECONSTRUCTION: PASS"
    )


if __name__ == "__main__":
    main()
