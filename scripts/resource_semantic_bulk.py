#!/usr/bin/env python3
"""Zero-call Stage A indexing for resource semantic judging."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import semantic_bulk as bulk
import semantic_panel as panel
from semantic_view import (
    SEMANTIC_VIEW_SCHEMA_VERSION,
)


INDEX_VERSION = "resource-1.1"

PRIMARY_PROFILES = (
    "claude",
    "fable",
    "codex",
)

EXCLUDED_PROFILE = "llama"

EXPECTED_PLANNED = {
    "claude": 210,
    "fable": 210,
    "codex": 210,
    "llama": 210,
}

EXPECTED_SUBSTANTIVE = {
    "claude": 203,
    "fable": 205,
    "codex": 210,
    "llama": 205,
}

EXPECTED_STUDY_PLANNED = 840
EXPECTED_STUDY_SUBSTANTIVE = 823
EXPECTED_ELIGIBLE = 618
EXPECTED_JOBS = 1236
EXPECTED_PANEL = 2

CONDITIONS = {
    "clean",
    "eval_only",
    "eval_resource_deprivation",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def git_head(path: Path) -> str:
    return subprocess.check_output(
        [
            "git",
            "-C",
            str(path),
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()


def load_rows(
    data_root: Path,
    profile: str,
) -> tuple[
    Path,
    list[dict[str, Any]],
]:
    path = (
        data_root
        / "analysis"
        / "resource"
        / profile
        / "trials.json"
    )

    rows = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(rows, list):
        raise ValueError(
            f"{path}: expected list"
        )

    if not all(
        isinstance(row, dict)
        for row in rows
    ):
        raise ValueError(
            f"{path}: non-object row"
        )

    return path, rows


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
        "--output",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    data_root = (
        args.data_root
        .expanduser()
        .resolve()
    )

    schema_path = (
        data_root
        / "config"
        / "resource_semantic_schema_v1.json"
    )

    fidelity_path = (
        data_root
        / "config"
        / "resource_fidelity_v1.json"
    )

    schema = panel.load_schema(
        schema_path
    )

    if (
        schema["schema_version"]
        != "resource-1.1"
    ):
        raise ValueError(
            "unexpected resource "
            "schema_version"
        )

    if (
        schema["rubric_version"]
        != "resource-1.1"
    ):
        raise ValueError(
            "unexpected resource "
            "rubric_version"
        )

    judges = schema[
        "primary_judges"
    ]

    if len(judges) != EXPECTED_PANEL:
        raise ValueError(
            "primary panel size mismatch"
        )

    if int(
        schema["consensus"][
            "primary_panel_size"
        ]
    ) != EXPECTED_PANEL:
        raise ValueError(
            "consensus panel size mismatch"
        )

    if {
        judge["model"]
        for judge in judges
    } != {
        "azure_ai/DeepSeek-V4-Pro",
        "gemini/gemini-3.6-flash",
    }:
        raise ValueError(
            "unexpected core judge panel"
        )

    all_profile_rows = {}
    trials_hashes = {}
    condition_counts = {}
    planned_counts = {}
    substantive_counts = {}
    censored_counts = {}

    for profile in (
        *PRIMARY_PROFILES,
        EXCLUDED_PROFILE,
    ):
        path, rows = load_rows(
            data_root,
            profile,
        )

        all_profile_rows[
            profile
        ] = rows

        trials_hashes[
            profile
        ] = sha256_file(path)

        planned_counts[
            profile
        ] = len(rows)

        substantive = [
            row
            for row in rows
            if bool(
                row.get(
                    "substantive_usable"
                )
            )
        ]

        substantive_counts[
            profile
        ] = len(substantive)

        censored_counts[
            profile
        ] = (
            len(rows)
            - len(substantive)
        )

        condition_counts[
            profile
        ] = dict(
            Counter(
                str(
                    row.get(
                        "condition"
                    )
                )
                for row
                in substantive
            )
        )

        if (
            len(rows)
            != EXPECTED_PLANNED[
                profile
            ]
        ):
            raise ValueError(
                f"{profile}: planned "
                f"{len(rows)} != "
                f"{EXPECTED_PLANNED[profile]}"
            )

        if (
            len(substantive)
            != EXPECTED_SUBSTANTIVE[
                profile
            ]
        ):
            raise ValueError(
                f"{profile}: substantive "
                f"{len(substantive)} != "
                f"{EXPECTED_SUBSTANTIVE[profile]}"
            )

        unexpected_conditions = (
            set(
                condition_counts[
                    profile
                ]
            )
            - CONDITIONS
        )

        if unexpected_conditions:
            raise ValueError(
                f"{profile}: unexpected "
                "conditions "
                f"{sorted(unexpected_conditions)}"
            )

    study_planned = sum(
        planned_counts.values()
    )

    study_substantive = sum(
        substantive_counts.values()
    )

    if (
        study_planned
        != EXPECTED_STUDY_PLANNED
    ):
        raise ValueError(
            "study planned denominator "
            "mismatch"
        )

    if (
        study_substantive
        != EXPECTED_STUDY_SUBSTANTIVE
    ):
        raise ValueError(
            "study substantive denominator "
            "mismatch"
        )

    jobs = []

    seen_trials = set()

    for profile in PRIMARY_PROFILES:
        rows = [
            row
            for row
            in all_profile_rows[
                profile
            ]
            if bool(
                row.get(
                    "substantive_usable"
                )
            )
        ]

        for row in rows:
            trial_name = str(
                row.get(
                    "trial_name"
                )
                or ""
            )

            if not trial_name:
                raise ValueError(
                    f"{profile}: missing "
                    "trial_name"
                )

            identity = (
                profile,
                trial_name,
            )

            if identity in seen_trials:
                raise ValueError(
                    f"duplicate trial "
                    f"{identity}"
                )

            seen_trials.add(identity)

            for judge in judges:
                job = bulk.build_job(
                    row=row,
                    profile=profile,
                    judge=judge,
                    schema=schema,
                    data_root=data_root,
                )

                job[
                    "resource_index_version"
                ] = INDEX_VERSION

                jobs.append(job)

    eligible = len(
        seen_trials
    )

    if eligible != EXPECTED_ELIGIBLE:
        raise ValueError(
            f"eligible={eligible}, "
            f"expected={EXPECTED_ELIGIBLE}"
        )

    if len(jobs) != EXPECTED_JOBS:
        raise ValueError(
            f"jobs={len(jobs)}, "
            f"expected={EXPECTED_JOBS}"
        )

    cache_keys = [
        str(job["cache_key"])
        for job in jobs
    ]

    if (
        len(set(cache_keys))
        != EXPECTED_JOBS
    ):
        raise ValueError(
            "cache keys are not unique"
        )

    trial_judge = [
        (
            job["profile"],
            job["trial_name"],
            job["judge_model"],
        )
        for job in jobs
    ]

    if (
        len(set(trial_judge))
        != EXPECTED_JOBS
    ):
        raise ValueError(
            "duplicate trial/judge jobs"
        )

    jobs.sort(
        key=lambda job: (
            str(job["profile"]),
            str(job["trial_name"]),
            str(job["judge_family"]),
            str(job["judge_model"]),
        )
    )

    judge_counts = dict(
        Counter(
            str(job["judge_model"])
            for job in jobs
        )
    )

    manifest = {
        "resource_index_version": (
            INDEX_VERSION
        ),
        "selection_status": (
            "indexed_before_resource_"
            "semantic_outcomes"
        ),
        "data_root": str(
            data_root
        ),
        "semantic_schema_path": str(
            schema_path
        ),
        "semantic_schema_sha256": (
            sha256_file(
                schema_path
            )
        ),
        "fidelity_spec_path": str(
            fidelity_path
        ),
        "fidelity_spec_sha256": (
            sha256_file(
                fidelity_path
            )
        ),
        "semantic_schema_version": (
            schema["schema_version"]
        ),
        "rubric_version": (
            schema["rubric_version"]
        ),
        "semantic_view_version": (
            SEMANTIC_VIEW_SCHEMA_VERSION
        ),
        "primary_profiles": list(
            PRIMARY_PROFILES
        ),
        "excluded_profiles": {
            "llama": (
                "capability_floor_and_"
                "scaffold_consumption_not_"
                "observably_established"
            )
        },
        "planned_counts": (
            planned_counts
        ),
        "substantive_counts": (
            substantive_counts
        ),
        "censored_counts": (
            censored_counts
        ),
        "condition_counts": (
            condition_counts
        ),
        "study_planned_trajectories": (
            study_planned
        ),
        "study_substantive_trajectories": (
            study_substantive
        ),
        "semantic_eligible_trajectories": (
            eligible
        ),
        "excluded_llama_substantive": (
            substantive_counts[
                "llama"
            ]
        ),
        "primary_panel_size": (
            EXPECTED_PANEL
        ),
        "judge_counts": (
            judge_counts
        ),
        "core_judge_jobs": len(
            jobs
        ),
        "unique_cache_keys": len(
            set(cache_keys)
        ),
        "input_trials_sha256": (
            trials_hashes
        ),
        "resource_data_git_head": (
            git_head(
                data_root
            )
        ),
        "behavior_code_git_head": (
            git_head(
                Path.cwd()
            )
        ),
        "network_calls": 0,
        "judge_calls": 0,
        "jobs": jobs,
    }

    output = args.output

    if output is None:
        output = (
            data_root
            / "analysis"
            / "semantic-resource-v1"
            / "manifests"
            / "resource-stage-a.json"
        )

    output = (
        output
        .expanduser()
        .resolve()
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "RESOURCE SEMANTIC STAGE A"
    )
    print("=" * 72)

    print(
        "study planned:",
        study_planned,
    )

    print(
        "study substantive:",
        study_substantive,
    )

    print()

    for profile in (
        *PRIMARY_PROFILES,
        EXCLUDED_PROFILE,
    ):
        print(
            profile,
            "planned=",
            planned_counts[
                profile
            ],
            "substantive=",
            substantive_counts[
                profile
            ],
            "censored=",
            censored_counts[
                profile
            ],
            "conditions=",
            condition_counts[
                profile
            ],
        )

    print()
    print(
        "semantic eligible:",
        eligible,
    )
    print(
        "core judge jobs:",
        len(jobs),
    )
    print(
        "unique cache keys:",
        len(set(cache_keys)),
    )
    print(
        "judge counts:",
        judge_counts,
    )
    print(
        "llama primary jobs: 0"
    )

    print()
    print(
        "schema version:",
        schema["schema_version"],
    )
    print(
        "rubric version:",
        schema["rubric_version"],
    )
    print(
        "semantic view:",
        SEMANTIC_VIEW_SCHEMA_VERSION,
    )

    print()
    print(
        "manifest:",
        output,
    )
    print(
        "manifest sha256:",
        sha256_file(output),
    )

    print()
    print("network calls: 0")
    print("judge calls: 0")
    print(
        "RESOURCE SEMANTIC STAGE A: PASS"
    )


if __name__ == "__main__":
    main()
