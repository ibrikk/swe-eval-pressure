#!/usr/bin/env python3
"""Deterministic bulk semantic-job indexing.

Stage A only:
- load the frozen usable historical trajectories;
- resolve and hash every trajectory;
- enumerate one job per core judge;
- compute the existing judge-specific cache key;
- validate uniqueness and expected coverage.

This module makes NO network/API calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import semantic_panel as panel
from semantic_view import (
    SEMANTIC_VIEW_SCHEMA_VERSION,
)


BULK_INDEX_VERSION = "1.0"

PROFILES = (
    "claude",
    "fable",
    "codex",
    "llama",
)

EXPECTED_USABLE_PER_PROFILE = 694
EXPECTED_USABLE_TOTAL = 2776

FROZEN_RELATIVE = Path(
    "analysis"
) / "frozen" / (
    "semantic-v26-primary-final-20260823"
)


def default_data_root() -> Path:
    raw = os.getenv("DATA_ROOT")

    if raw:
        return Path(raw).expanduser().resolve()

    return (
        Path.home()
        / "Documents"
        / "swe-eval-pressure"
    ).resolve()


def frozen_root(
    data_root: Path,
) -> Path:
    return (
        data_root
        / FROZEN_RELATIVE
    )


def load_profile_rows(
    *,
    data_root: Path,
    profile: str,
) -> list[dict[str, Any]]:
    path = (
        frozen_root(data_root)
        / profile
        / "trials.json"
    )

    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(data, list):
        raise ValueError(
            f"{path}: trials.json "
            "must contain a list"
        )

    rows = [
        row
        for row in data
        if isinstance(row, dict)
    ]

    if len(rows) != len(data):
        raise ValueError(
            f"{path}: non-object trial row"
        )

    return rows


def normalize_path(
    value: str,
    *,
    data_root: Path,
) -> Path:
    path = Path(value).expanduser()

    if not path.is_absolute():
        path = data_root / path

    return path


def resolve_trajectory_path(
    row: dict[str, Any],
    *,
    data_root: Path,
) -> Path:
    result_raw = str(
        row.get("result_path", "")
        or ""
    )

    if not result_raw:
        raise ValueError(
            f"{row.get('trial_name')}: "
            "missing result_path"
        )

    result_path = normalize_path(
        result_raw,
        data_root=data_root,
    )

    trial_dir = result_path.parent

    relative_raw = str(
        row.get("trajectory_file", "")
        or ""
    )

    if relative_raw:
        relative = Path(
            relative_raw
        ).expanduser()

        candidate = (
            relative
            if relative.is_absolute()
            else trial_dir / relative
        )

        if candidate.is_file():
            return candidate.resolve()

    known = [
        trial_dir
        / "agent"
        / "trajectory.json",
        trial_dir
        / "agent"
        / "minisuite_trajectory.json",
        trial_dir
        / "agent"
        / "mini-swe-agent.trajectory.json",
    ]

    existing_known = [
        path
        for path in known
        if path.is_file()
    ]

    if len(existing_known) == 1:
        return existing_known[
            0
        ].resolve()

    if len(existing_known) > 1:
        raise ValueError(
            f"{row.get('trial_name')}: "
            "multiple known trajectory files"
        )

    fallback = sorted(
        trial_dir.rglob(
            "*trajectory*.json"
        )
    )

    if len(fallback) != 1:
        raise ValueError(
            f"{row.get('trial_name')}: "
            "expected exactly one fallback "
            f"trajectory, found {len(fallback)}"
        )

    return fallback[0].resolve()


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def build_job(
    *,
    row: dict[str, Any],
    profile: str,
    judge: dict[str, str],
    schema: dict[str, Any],
    data_root: Path,
) -> dict[str, Any]:
    trial_name = str(
        row.get("trial_name", "")
        or ""
    )

    if not trial_name:
        raise ValueError(
            f"{profile}: trial_name missing"
        )

    trajectory_path = (
        resolve_trajectory_path(
            row,
            data_root=data_root,
        )
    )

    trajectory_hash = (
        sha256_file(
            trajectory_path
        )
    )

    cache_key = (
        panel.judge_cache_key(
            trial_name=trial_name,
            trajectory_hash=(
                trajectory_hash
            ),
            model=judge["model"],
            schema=schema,
            semantic_view_version=(
                SEMANTIC_VIEW_SCHEMA_VERSION
            ),
        )
    )

    placement = (
        row.get("placement")
        or row.get("channel")
        or ""
    )

    return {
        "bulk_index_version": (
            BULK_INDEX_VERSION
        ),
        "profile": profile,
        "trial_name": trial_name,
        "condition": row.get(
            "condition",
            "",
        ),
        "placement": placement,
        "pressure_type": row.get(
            "pressure_type",
            "",
        ),
        "trajectory_path": str(
            trajectory_path
        ),
        "trajectory_hash": (
            trajectory_hash
        ),
        "judge_family": judge[
            "family"
        ],
        "judge_model": judge[
            "model"
        ],
        "semantic_schema_version": (
            schema["schema_version"]
        ),
        "rubric_version": (
            schema["rubric_version"]
        ),
        "semantic_view_version": (
            SEMANTIC_VIEW_SCHEMA_VERSION
        ),
        "cache_key": cache_key,
    }


def build_manifest(
    *,
    data_root: Path,
    strict_counts: bool = True,
) -> dict[str, Any]:
    schema = panel.load_schema()

    judges = schema[
        "primary_judges"
    ]

    expected_panel = int(
        schema["consensus"][
            "primary_panel_size"
        ]
    )

    if len(judges) != expected_panel:
        raise ValueError(
            "primary judge count does not "
            "match primary_panel_size"
        )

    families = [
        judge["family"]
        for judge in judges
    ]

    if len(set(families)) != len(
        families
    ):
        raise ValueError(
            "primary judge families "
            "are not unique"
        )

    jobs: list[
        dict[str, Any]
    ] = []

    planned_counts: dict[
        str,
        int,
    ] = {}

    trial_counts: dict[
        str,
        int,
    ] = {}

    censored_counts: dict[
        str,
        int,
    ] = {}

    seen_trials: set[
        tuple[str, str]
    ] = set()

    for profile in PROFILES:
        all_rows = load_profile_rows(
            data_root=data_root,
            profile=profile,
        )

        # Frozen trials.json preserves the complete planned
        # reconstruction denominator, including infrastructure-
        # censored/error rows. Semantic judging is defined only
        # for substantively usable trajectories.
        if not all(
            "substantive_usable" in row
            for row in all_rows
        ):
            raise ValueError(
                f"{profile}: trials.json lacks "
                "substantive_usable on one or more rows"
            )

        rows = [
            row
            for row in all_rows
            if bool(
                row.get(
                    "substantive_usable"
                )
            )
        ]

        planned_counts[
            profile
        ] = len(all_rows)

        trial_counts[
            profile
        ] = len(rows)

        censored_counts[
            profile
        ] = (
            len(all_rows)
            - len(rows)
        )

        if strict_counts:
            if len(all_rows) != 700:
                raise ValueError(
                    f"{profile}: expected 700 "
                    "planned/reconstructed rows, "
                    f"found {len(all_rows)}"
                )

            if (
                len(rows)
                != EXPECTED_USABLE_PER_PROFILE
            ):
                raise ValueError(
                    f"{profile}: expected "
                    f"{EXPECTED_USABLE_PER_PROFILE} "
                    "usable trajectories, "
                    f"found {len(rows)}"
                )

            if (
                censored_counts[profile]
                != 6
            ):
                raise ValueError(
                    f"{profile}: expected 6 "
                    "infrastructure-censored/error "
                    "rows, found "
                    f"{censored_counts[profile]}"
                )

        for row in rows:
            trial_name = str(
                row.get(
                    "trial_name",
                    "",
                )
                or ""
            )

            trial_identity = (
                profile,
                trial_name,
            )

            if trial_identity in (
                seen_trials
            ):
                raise ValueError(
                    f"duplicate trial: "
                    f"{trial_identity}"
                )

            seen_trials.add(
                trial_identity
            )

            for judge in judges:
                jobs.append(
                    build_job(
                        row=row,
                        profile=profile,
                        judge=judge,
                        schema=schema,
                        data_root=(
                            data_root
                        ),
                    )
                )

    usable_total = sum(
        trial_counts.values()
    )

    if (
        strict_counts
        and usable_total
        != EXPECTED_USABLE_TOTAL
    ):
        raise ValueError(
            "expected "
            f"{EXPECTED_USABLE_TOTAL} "
            "usable trajectories, "
            f"found {usable_total}"
        )

    cache_keys = [
        job["cache_key"]
        for job in jobs
    ]

    if len(set(cache_keys)) != len(
        cache_keys
    ):
        duplicates = [
            key
            for key, count in Counter(
                cache_keys
            ).items()
            if count > 1
        ]

        raise ValueError(
            "duplicate semantic cache keys: "
            f"{len(duplicates)}"
        )

    pair_ids = [
        (
            job["profile"],
            job["trial_name"],
            job["judge_model"],
        )
        for job in jobs
    ]

    if len(set(pair_ids)) != len(
        pair_ids
    ):
        raise ValueError(
            "duplicate trial/judge jobs"
        )

    expected_jobs = (
        usable_total
        * expected_panel
    )

    if len(jobs) != expected_jobs:
        raise ValueError(
            f"expected {expected_jobs} "
            f"jobs, found {len(jobs)}"
        )

    jobs.sort(
        key=lambda job: (
            job["profile"],
            job["trial_name"],
            job["judge_family"],
            job["judge_model"],
        )
    )

    judge_counts = Counter(
        job["judge_model"]
        for job in jobs
    )

    return {
        "bulk_index_version": (
            BULK_INDEX_VERSION
        ),
        "data_root": str(
            data_root
        ),
        "frozen_root": str(
            frozen_root(data_root)
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
        "profiles": list(
            PROFILES
        ),
        "planned_counts": (
            planned_counts
        ),
        "trial_counts": (
            trial_counts
        ),
        "censored_counts": (
            censored_counts
        ),
        "planned_trajectories": sum(
            planned_counts.values()
        ),
        "usable_trajectories": (
            usable_total
        ),
        "censored_or_error": sum(
            censored_counts.values()
        ),
        "primary_panel_size": (
            expected_panel
        ),
        "judge_counts": dict(
            judge_counts
        ),
        "job_count": len(jobs),
        "unique_cache_keys": len(
            set(cache_keys)
        ),
        "jobs": jobs,
    }


def summary_lines(
    manifest: dict[str, Any],
) -> list[str]:
    lines = [
        "SEMANTIC BULK STAGE A",
        "=" * 72,
        (
            "planned trajectories: "
            f"{manifest['planned_trajectories']}"
        ),
        (
            "usable trajectories: "
            f"{manifest['usable_trajectories']}"
        ),
        (
            "censored/error: "
            f"{manifest['censored_or_error']}"
        ),
        (
            "primary panel size: "
            f"{manifest['primary_panel_size']}"
        ),
        (
            "judge jobs: "
            f"{manifest['job_count']}"
        ),
        (
            "unique cache keys: "
            f"{manifest['unique_cache_keys']}"
        ),
        "",
        "per profile:",
    ]

    for profile in PROFILES:
        lines.append(
            f"  {profile}: "
            f"planned={manifest['planned_counts'][profile]} "
            f"usable={manifest['trial_counts'][profile]} "
            f"censored={manifest['censored_counts'][profile]}"
        )

    lines.extend([
        "",
        "per judge:",
    ])

    for model, count in sorted(
        manifest[
            "judge_counts"
        ].items()
    ):
        lines.append(
            f"  {model}: {count}"
        )

    lines.extend([
        "",
        "network calls: 0",
        "STAGE A DRY RUN: PASS",
    ])

    return lines


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root(),
    )

    parser.add_argument(
        "--output",
        type=Path,
    )

    parser.add_argument(
        "--no-strict-counts",
        action="store_true",
    )

    args = parser.parse_args()

    data_root = (
        args.data_root
        .expanduser()
        .resolve()
    )

    manifest = build_manifest(
        data_root=data_root,
        strict_counts=(
            not args.no_strict_counts
        ),
    )

    if args.output:
        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        args.output.write_text(
            json.dumps(
                manifest,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    print(
        "\n".join(
            summary_lines(
                manifest
            )
        )
    )

    if args.output:
        print(
            "manifest:",
            args.output,
        )


if __name__ == "__main__":
    main()
