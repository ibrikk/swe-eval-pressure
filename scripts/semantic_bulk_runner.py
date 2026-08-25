#!/usr/bin/env python3
"""Resumable semantic bulk execution planner.

Stage B initially provides:
- deterministic 10-trajectory pilot selection;
- exactly one trajectory from each treatment cell;
- deterministic profile assignment;
- two core-judge jobs per selected trajectory;
- cache/artifact path planning;
- dry-run accounting.

The dry-run path makes NO network calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


RUNNER_VERSION = "1.0"

PILOT_CELLS = (
    ("clean", "none"),
    ("eval_only", "root"),
    ("eval_only", "scaffold"),
    ("eval_only", "source"),
    ("eval_financial", "root"),
    ("eval_financial", "scaffold"),
    ("eval_financial", "source"),
    (
        "eval_self_preservation",
        "root",
    ),
    (
        "eval_self_preservation",
        "scaffold",
    ),
    (
        "eval_self_preservation",
        "source",
    ),
)

# Deliberately fixed before semantic outcomes are inspected.
#
# Distribution:
# claude = 3
# fable  = 3
# codex  = 2
# llama  = 2
PILOT_PROFILES = (
    "claude",
    "fable",
    "codex",
    "llama",
    "claude",
    "fable",
    "codex",
    "llama",
    "claude",
    "fable",
)


def selection_hash(
    *,
    profile: str,
    trial_name: str,
) -> str:
    value = (
        f"{profile}|{trial_name}"
    )

    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def collapse_manifest_jobs(
    jobs: list[dict[str, Any]],
) -> dict[
    tuple[str, str],
    dict[str, Any],
]:
    trials: dict[
        tuple[str, str],
        dict[str, Any],
    ] = {}

    for job in jobs:
        key = (
            str(job["profile"]),
            str(job["trial_name"]),
        )

        if key not in trials:
            trials[key] = {
                "profile": str(
                    job["profile"]
                ),
                "trial_name": str(
                    job["trial_name"]
                ),
                "condition": str(
                    job.get(
                        "condition",
                        "",
                    )
                ),
                "placement": str(
                    job.get(
                        "placement",
                        "",
                    )
                ),
                "pressure_type": str(
                    job.get(
                        "pressure_type",
                        "",
                    )
                ),
                "trajectory_path": str(
                    job[
                        "trajectory_path"
                    ]
                ),
                "trajectory_hash": str(
                    job[
                        "trajectory_hash"
                    ]
                ),
                "jobs": [],
            }

        trial = trials[key]

        for field in (
            "condition",
            "placement",
            "pressure_type",
            "trajectory_path",
            "trajectory_hash",
        ):
            if (
                trial[field]
                != str(
                    job.get(
                        field,
                        "",
                    )
                )
            ):
                raise ValueError(
                    f"inconsistent {field} "
                    f"for {key}"
                )

        trial["jobs"].append(job)

    for key, trial in trials.items():
        families = {
            str(
                job["judge_family"]
            )
            for job in trial["jobs"]
        }

        if families != {
            "deepseek",
            "gemini",
        }:
            raise ValueError(
                f"{key}: unexpected core "
                f"judge families {families}"
            )

        if len(
            trial["jobs"]
        ) != 2:
            raise ValueError(
                f"{key}: expected 2 "
                "core judge jobs"
            )

        trial["jobs"].sort(
            key=lambda job: (
                str(
                    job[
                        "judge_family"
                    ]
                ),
                str(
                    job[
                        "judge_model"
                    ]
                ),
            )
        )

    return trials


def select_pilot_trials(
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    trials = collapse_manifest_jobs(
        manifest["jobs"]
    )

    selected: list[
        dict[str, Any]
    ] = []

    for (
        profile,
        (
            condition,
            placement,
        ),
    ) in zip(
        PILOT_PROFILES,
        PILOT_CELLS,
        strict=True,
    ):
        candidates = [
            trial
            for trial in trials.values()
            if (
                trial["profile"]
                == profile
                and trial[
                    "condition"
                ]
                == condition
                and trial[
                    "placement"
                ]
                == placement
            )
        ]

        if not candidates:
            raise ValueError(
                "no pilot candidate for "
                f"{profile} × "
                f"{condition} × "
                f"{placement}"
            )

        candidates.sort(
            key=lambda trial: (
                selection_hash(
                    profile=profile,
                    trial_name=(
                        trial[
                            "trial_name"
                        ]
                    ),
                ),
                trial["trial_name"],
            )
        )

        chosen = dict(
            candidates[0]
        )

        chosen[
            "selection_hash"
        ] = selection_hash(
            profile=profile,
            trial_name=(
                chosen["trial_name"]
            ),
        )

        selected.append(
            chosen
        )

    identities = {
        (
            trial["profile"],
            trial["trial_name"],
        )
        for trial in selected
    }

    if len(identities) != 10:
        raise ValueError(
            "pilot contains duplicate "
            "trajectory identities"
        )

    observed_cells = {
        (
            trial["condition"],
            trial["placement"],
        )
        for trial in selected
    }

    if observed_cells != set(
        PILOT_CELLS
    ):
        raise ValueError(
            "pilot does not cover "
            "exactly the ten treatment "
            "cells"
        )

    return selected


def job_artifact_path(
    output_dir: Path,
    job: dict[str, Any],
) -> Path:
    return (
        output_dir
        / "jobs"
        / (
            str(
                job["cache_key"]
            )
            + ".json"
        )
    )


def artifact_is_complete(
    path: Path,
    *,
    expected_cache_key: str,
) -> bool:
    if not path.is_file():
        return False

    try:
        data = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return False

    return (
        isinstance(data, dict)
        and data.get(
            "cache_key"
        )
        == expected_cache_key
        and data.get(
            "status"
        )
        == "ok"
        and isinstance(
            data.get(
                "final_cache_entry"
            ),
            dict,
        )
        and data[
            "final_cache_entry"
        ].get(
            "status"
        )
        == "ok"
    )


def build_pilot_plan(
    *,
    manifest: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    trials = select_pilot_trials(
        manifest
    )

    jobs: list[
        dict[str, Any]
    ] = []

    for trial in trials:
        for source_job in trial[
            "jobs"
        ]:
            job = dict(
                source_job
            )

            artifact = (
                job_artifact_path(
                    output_dir,
                    job,
                )
            )

            cache_hit = (
                artifact_is_complete(
                    artifact,
                    expected_cache_key=(
                        str(
                            job[
                                "cache_key"
                            ]
                        )
                    ),
                )
            )

            job[
                "artifact_path"
            ] = str(artifact)

            job[
                "cache_hit"
            ] = cache_hit

            jobs.append(job)

    jobs.sort(
        key=lambda job: (
            str(
                job["condition"]
            ),
            str(
                job["placement"]
            ),
            str(
                job["profile"]
            ),
            str(
                job[
                    "trial_name"
                ]
            ),
            str(
                job[
                    "judge_family"
                ]
            ),
        )
    )

    cache_hits = sum(
        bool(
            job["cache_hit"]
        )
        for job in jobs
    )

    pending = (
        len(jobs)
        - cache_hits
    )

    return {
        "runner_version": (
            RUNNER_VERSION
        ),
        "mode": "pilot_10",
        "network_calls": 0,
        "trajectory_count": (
            len(trials)
        ),
        "job_count": len(jobs),
        "cache_hits": cache_hits,
        "pending_jobs": pending,
        "output_dir": str(
            output_dir
        ),
        "profile_counts": dict(
            Counter(
                trial["profile"]
                for trial in trials
            )
        ),
        "cell_counts": {
            (
                f"{trial['condition']}"
                "×"
                f"{trial['placement']}"
            ): 1
            for trial in trials
        },
        "trials": trials,
        "jobs": jobs,
    }


def print_plan(
    plan: dict[str, Any],
) -> None:
    print(
        "SEMANTIC BULK STAGE B "
        "PILOT DRY RUN"
    )
    print("=" * 72)

    print(
        "trajectories:",
        plan[
            "trajectory_count"
        ],
    )

    print(
        "judge jobs:",
        plan[
            "job_count"
        ],
    )

    print(
        "cache hits:",
        plan[
            "cache_hits"
        ],
    )

    print(
        "pending jobs:",
        plan[
            "pending_jobs"
        ],
    )

    print(
        "network calls:",
        plan[
            "network_calls"
        ],
    )

    print()
    print("SELECTED TRAJECTORIES")

    for index, trial in enumerate(
        plan["trials"],
        1,
    ):
        print(
            f"{index:02d}",
            trial["profile"],
            trial["condition"],
            trial["placement"],
            trial["trial_name"],
        )

    print()
    print(
        "profile counts:",
        plan[
            "profile_counts"
        ],
    )

    print()
    print(
        "STAGE B PILOT DRY RUN: PASS"
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--plan-output",
        type=Path,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        required=True,
    )

    args = parser.parse_args()

    manifest = json.loads(
        args.manifest.read_text(
            encoding="utf-8"
        )
    )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    plan = build_pilot_plan(
        manifest=manifest,
        output_dir=output_dir,
    )

    if plan[
        "trajectory_count"
    ] != 10:
        raise ValueError(
            "expected exactly "
            "10 pilot trajectories"
        )

    if plan[
        "job_count"
    ] != 20:
        raise ValueError(
            "expected exactly "
            "20 pilot jobs"
        )

    if plan[
        "network_calls"
    ] != 0:
        raise ValueError(
            "dry-run attempted "
            "network activity"
        )

    if args.plan_output:
        args.plan_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        args.plan_output.write_text(
            json.dumps(
                plan,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    print_plan(plan)

    if args.plan_output:
        print(
            "plan:",
            args.plan_output,
        )


if __name__ == "__main__":
    main()
