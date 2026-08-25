#!/usr/bin/env python3
"""Outcome-blind deterministic pilot selection for resource semantics.

Select exactly one trajectory from each:
    3 capable profiles x 3 resource-study conditions = 9 trajectories.

Each trajectory receives both frozen core judges = 18 jobs.

NO network/API calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


PILOT_VERSION = "resource-pilot-1.0"

PROFILES = (
    "claude",
    "fable",
    "codex",
)

CONDITIONS = (
    "clean",
    "eval_only",
    "eval_resource_deprivation",
)

EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "a9d065e77fb13f7c55b2871f1d850047"
    "894d8f455b2d54aafe9e32b71e1ec62f"
)

EXPECTED_TRAJECTORIES = 9
EXPECTED_JOBS = 18


def sha256_file(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def selection_hash(
    *,
    profile: str,
    condition: str,
    trial_name: str,
) -> str:
    identity = (
        f"{PILOT_VERSION}|"
        f"{profile}|"
        f"{condition}|"
        f"{trial_name}"
    )

    return hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()


def collapse_jobs(
    jobs: list[dict[str, Any]],
) -> dict[
    tuple[str, str],
    dict[str, Any],
]:
    trials = {}

    for job in jobs:
        profile = str(
            job["profile"]
        )
        trial_name = str(
            job["trial_name"]
        )

        key = (
            profile,
            trial_name,
        )

        if key not in trials:
            trials[key] = {
                "profile": profile,
                "trial_name": trial_name,
                "condition": str(
                    job["condition"]
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
                str(trial[field])
                != str(
                    job.get(
                        field,
                        "",
                    )
                )
            ):
                raise ValueError(
                    f"{key}: inconsistent "
                    f"{field}"
                )

        trial["jobs"].append(
            dict(job)
        )

    for key, trial in trials.items():
        jobs_for_trial = trial[
            "jobs"
        ]

        if len(jobs_for_trial) != 2:
            raise ValueError(
                f"{key}: expected exactly "
                "2 core judge jobs"
            )

        families = {
            str(
                job["judge_family"]
            )
            for job
            in jobs_for_trial
        }

        if families != {
            "deepseek",
            "gemini",
        }:
            raise ValueError(
                f"{key}: unexpected "
                f"judge families {families}"
            )

        jobs_for_trial.sort(
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


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        type=Path,
        default=(
            Path.home()
            / "Documents"
            / "swe-eval-pressure"
        ),
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--plan-output",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--spec-output",
        type=Path,
        default=(
            Path("config")
            / "resource_semantic_pilot_v1.json"
        ),
    )

    args = parser.parse_args()

    data_root = (
        args.data_root
        .expanduser()
        .resolve()
    )

    manifest_path = (
        args.manifest
        if args.manifest
        is not None
        else (
            data_root
            / "analysis"
            / "semantic-resource-v1"
            / "manifests"
            / "resource-stage-a.json"
        )
    )

    manifest_path = (
        manifest_path
        .expanduser()
        .resolve()
    )

    actual_hash = sha256_file(
        manifest_path
    )

    if (
        actual_hash
        != EXPECTED_SOURCE_MANIFEST_SHA256
    ):
        raise ValueError(
            "Stage A manifest hash "
            "does not match frozen "
            "pre-pilot value:\n"
            f"expected "
            f"{EXPECTED_SOURCE_MANIFEST_SHA256}\n"
            f"actual   {actual_hash}"
        )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    expected = {
        "semantic_eligible_trajectories": 618,
        "core_judge_jobs": 1236,
        "unique_cache_keys": 1236,
        "primary_panel_size": 2,
    }

    for field, wanted in (
        expected.items()
    ):
        actual = int(
            manifest[field]
        )

        if actual != wanted:
            raise ValueError(
                f"{field}: expected "
                f"{wanted}, found "
                f"{actual}"
            )

    if (
        manifest[
            "semantic_schema_version"
        ]
        != "resource-1.0"
    ):
        raise ValueError(
            "unexpected semantic "
            "schema version"
        )

    if (
        manifest[
            "rubric_version"
        ]
        != "resource-1.0"
    ):
        raise ValueError(
            "unexpected rubric version"
        )

    trials = collapse_jobs(
        manifest["jobs"]
    )

    selected = []

    for profile in PROFILES:
        for condition in CONDITIONS:
            candidates = [
                trial
                for trial
                in trials.values()
                if (
                    trial["profile"]
                    == profile
                    and trial[
                        "condition"
                    ]
                    == condition
                )
            ]

            if not candidates:
                raise ValueError(
                    "no candidate for "
                    f"{profile} x "
                    f"{condition}"
                )

            candidates.sort(
                key=lambda trial: (
                    selection_hash(
                        profile=profile,
                        condition=condition,
                        trial_name=(
                            trial[
                                "trial_name"
                            ]
                        ),
                    ),
                    trial[
                        "trial_name"
                    ],
                )
            )

            chosen = dict(
                candidates[0]
            )

            chosen[
                "selection_hash"
            ] = selection_hash(
                profile=profile,
                condition=condition,
                trial_name=(
                    chosen[
                        "trial_name"
                    ]
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

    if (
        len(identities)
        != EXPECTED_TRAJECTORIES
    ):
        raise ValueError(
            "pilot trajectory "
            "identities are not unique"
        )

    observed_cells = {
        (
            trial["profile"],
            trial["condition"],
        )
        for trial in selected
    }

    expected_cells = {
        (
            profile,
            condition,
        )
        for profile
        in PROFILES
        for condition
        in CONDITIONS
    }

    if (
        observed_cells
        != expected_cells
    ):
        raise ValueError(
            "pilot does not cover "
            "exactly 3x3 cells"
        )

    pilot_jobs = []

    for trial in selected:
        for job in trial[
            "jobs"
        ]:
            pilot_jobs.append(
                dict(job)
            )

    if (
        len(pilot_jobs)
        != EXPECTED_JOBS
    ):
        raise ValueError(
            f"expected "
            f"{EXPECTED_JOBS} jobs, "
            f"found "
            f"{len(pilot_jobs)}"
        )

    if (
        len(
            {
                str(
                    job[
                        "cache_key"
                    ]
                )
                for job
                in pilot_jobs
            }
        )
        != EXPECTED_JOBS
    ):
        raise ValueError(
            "pilot cache keys "
            "are not unique"
        )

    profile_counts = dict(
        Counter(
            trial["profile"]
            for trial
            in selected
        )
    )

    condition_counts = dict(
        Counter(
            trial["condition"]
            for trial
            in selected
        )
    )

    judge_counts = dict(
        Counter(
            job["judge_model"]
            for job
            in pilot_jobs
        )
    )

    primary_judges = sorted(
        {
            (
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
            for job
            in pilot_jobs
        }
    )

    serializable_trials = []

    for trial in selected:
        value = {
            key: val
            for key, val
            in trial.items()
            if key != "jobs"
        }

        value["judge_jobs"] = [
            {
                "judge_family": (
                    job[
                        "judge_family"
                    ]
                ),
                "judge_model": (
                    job[
                        "judge_model"
                    ]
                ),
                "cache_key": (
                    job[
                        "cache_key"
                    ]
                ),
            }
            for job
            in trial["jobs"]
        ]

        serializable_trials.append(
            value
        )

    spec = {
        "pilot_version": (
            PILOT_VERSION
        ),
        "selection_status": (
            "frozen_before_resource_"
            "semantic_outcomes"
        ),
        "selection_method": (
            "For each capable profile "
            "x condition cell, choose "
            "the trajectory with the "
            "lexicographically smallest "
            "SHA256 of "
            "'resource-pilot-1.0|"
            "profile|condition|trial_name'."
        ),
        "source_manifest_path": str(
            manifest_path
        ),
        "source_manifest_sha256": (
            actual_hash
        ),
        "semantic_schema_version": (
            manifest[
                "semantic_schema_version"
            ]
        ),
        "rubric_version": (
            manifest[
                "rubric_version"
            ]
        ),
        "semantic_view_version": (
            manifest[
                "semantic_view_version"
            ]
        ),
        "trajectory_count": (
            len(selected)
        ),
        "core_judge_jobs": (
            len(pilot_jobs)
        ),
        "profile_counts": (
            profile_counts
        ),
        "condition_counts": (
            condition_counts
        ),
        "judge_counts": (
            judge_counts
        ),
        "primary_judges": [
            {
                "family": family,
                "model": model,
            }
            for family, model
            in primary_judges
        ],
        "outcome_based_selection": (
            False
        ),
        "verifier_outcomes_used_for_selection": (
            False
        ),
        "semantic_outputs_used_for_selection": (
            False
        ),
        "network_calls": 0,
        "judge_calls": 0,
        "trials": (
            serializable_trials
        ),
    }

    plan = {
        "pilot_version": (
            PILOT_VERSION
        ),
        "source_manifest_sha256": (
            actual_hash
        ),
        "trajectory_count": (
            len(selected)
        ),
        "job_count": (
            len(pilot_jobs)
        ),
        "profile_counts": (
            profile_counts
        ),
        "condition_counts": (
            condition_counts
        ),
        "judge_counts": (
            judge_counts
        ),
        "network_calls": 0,
        "judge_calls": 0,
        "trials": (
            serializable_trials
        ),
        "jobs": (
            pilot_jobs
        ),
    }

    spec_output = (
        args.spec_output
        .expanduser()
        .resolve()
    )

    spec_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    spec_output.write_text(
        json.dumps(
            spec,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    plan_output = (
        args.plan_output
        if args.plan_output
        is not None
        else (
            data_root
            / "analysis"
            / "semantic-resource-v1"
            / "pilot"
            / "resource-pilot-plan.json"
        )
    )

    plan_output = (
        plan_output
        .expanduser()
        .resolve()
    )

    plan_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plan_output.write_text(
        json.dumps(
            plan,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "RESOURCE SEMANTIC PILOT SELECTION"
    )
    print("=" * 72)

    print(
        "source manifest sha256:",
        actual_hash,
    )
    print(
        "trajectories:",
        len(selected),
    )
    print(
        "judge jobs:",
        len(pilot_jobs),
    )
    print(
        "profile counts:",
        profile_counts,
    )
    print(
        "condition counts:",
        condition_counts,
    )
    print(
        "judge counts:",
        judge_counts,
    )

    print()
    print(
        "SELECTED TRAJECTORIES"
    )

    for i, trial in enumerate(
        selected,
        1,
    ):
        print(
            f"{i:02d}",
            trial["profile"],
            trial["condition"],
            trial["trial_name"],
            "selection=",
            trial[
                "selection_hash"
            ][:12],
        )

    print()
    print(
        "spec:",
        spec_output,
    )
    print(
        "spec sha256:",
        sha256_file(
            spec_output
        ),
    )
    print(
        "plan:",
        plan_output,
    )
    print(
        "plan sha256:",
        sha256_file(
            plan_output
        ),
    )

    print()
    print("network calls: 0")
    print("judge calls: 0")
    print(
        "RESOURCE SEMANTIC PILOT "
        "SELECTION: PASS"
    )


if __name__ == "__main__":
    main()
