#!/usr/bin/env python3
"""Execute the frozen full semantic production universe.

Properties:
- 2,800 planned trajectories remain the reconstruction denominator;
- only 2,776 substantively usable trajectories are judged;
- exactly 5,552 core-judge jobs;
- multiple inherited caches are read-only;
- terminal inherited missingness remains missing;
- only the full-production root is writable;
- strict request-body compatibility is verified for every cache hit;
- bounded retries are inherited from the frozen pilot executor;
- optional job batching supports safe resumable production;
- dry-run performs NO network/API calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import semantic_dose_execute as dose
import semantic_orchestrator as orch
import semantic_panel as panel
import semantic_pilot_execute as base


FULL_EXECUTOR_VERSION = "1.0"

EXPECTED_PLANNED = 2800
EXPECTED_USABLE = 2776
EXPECTED_CENSORED = 24
EXPECTED_JOBS = 5552
EXPECTED_PANEL = 2

PROFILES = (
    "claude",
    "fable",
    "codex",
    "llama",
)


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


def sha256_file(
    path: Path,
) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def validate_manifest(
    manifest: dict[str, Any],
) -> None:
    expected = {
        "planned_trajectories": (
            EXPECTED_PLANNED
        ),
        "usable_trajectories": (
            EXPECTED_USABLE
        ),
        "censored_or_error": (
            EXPECTED_CENSORED
        ),
        "primary_panel_size": (
            EXPECTED_PANEL
        ),
        "job_count": (
            EXPECTED_JOBS
        ),
        "unique_cache_keys": (
            EXPECTED_JOBS
        ),
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
                f"{wanted}, found {actual}"
            )

    for profile in PROFILES:
        if int(
            manifest[
                "planned_counts"
            ][profile]
        ) != 700:
            raise ValueError(
                f"{profile}: planned "
                "count mismatch"
            )

        if int(
            manifest[
                "trial_counts"
            ][profile]
        ) != 694:
            raise ValueError(
                f"{profile}: usable "
                "count mismatch"
            )

        if int(
            manifest[
                "censored_counts"
            ][profile]
        ) != 6:
            raise ValueError(
                f"{profile}: censored "
                "count mismatch"
            )


def validate_spec(
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    spec: dict[str, Any],
) -> None:
    if (
        spec.get(
            "selection_status"
        )
        != (
            "frozen_historical_"
            "semantic_universe"
        )
    ):
        raise ValueError(
            "full universe is not "
            "marked frozen"
        )

    if (
        sha256_file(
            manifest_path
        )
        != spec[
            "source_manifest_sha256"
        ]
    ):
        raise ValueError(
            "full manifest hash mismatch"
        )

    checks = {
        "bulk_index_version": (
            manifest[
                "bulk_index_version"
            ]
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
    }

    for field, value in (
        checks.items()
    ):
        if (
            str(spec[field])
            != str(value)
        ):
            raise ValueError(
                f"{field} mismatch"
            )

    numeric = {
        "planned_trajectories": (
            EXPECTED_PLANNED
        ),
        (
            "semantic_eligible_"
            "trajectories"
        ): EXPECTED_USABLE,
        "censored_or_error": (
            EXPECTED_CENSORED
        ),
        "primary_panel_size": (
            EXPECTED_PANEL
        ),
        "core_judge_jobs": (
            EXPECTED_JOBS
        ),
    }

    for field, wanted in (
        numeric.items()
    ):
        if int(
            spec[field]
        ) != wanted:
            raise ValueError(
                f"{field} mismatch"
            )

    policy = spec[
        "request_policy"
    ]

    schema = panel.load_schema()

    if (
        policy["temperature"]
        != schema[
            "request"
        ]["temperature"]
    ):
        raise ValueError(
            "temperature policy mismatch"
        )

    if (
        policy[
            "response_format"
        ]
        != schema[
            "request"
        ]["response_format"]
    ):
        raise ValueError(
            "response-format "
            "policy mismatch"
        )

    if int(
        policy["max_tokens"]
    ) != int(
        base.MAX_TOKENS
    ):
        raise ValueError(
            "max_tokens policy mismatch"
        )

    if int(
        policy["timeout_seconds"]
    ) != int(
        base.TIMEOUT_SECONDS
    ):
        raise ValueError(
            "timeout policy mismatch"
        )

    if int(
        policy[
            "max_attempts_per_job"
        ]
    ) != int(
        orch.DEFAULT_MAX_ATTEMPTS
    ):
        raise ValueError(
            "attempt-budget "
            "policy mismatch"
        )

    if [
        float(x)
        for x in policy[
            "retry_delays_seconds"
        ]
    ] != [
        float(x)
        for x in (
            orch.DEFAULT_RETRY_DELAYS
        )
    ]:
        raise ValueError(
            "retry-delay policy mismatch"
        )


def full_jobs(
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    jobs = [
        dict(job)
        for job in manifest["jobs"]
    ]

    if len(jobs) != EXPECTED_JOBS:
        raise ValueError(
            "expected exactly "
            f"{EXPECTED_JOBS} jobs"
        )

    keys = [
        str(job["cache_key"])
        for job in jobs
    ]

    if len(
        set(keys)
    ) != EXPECTED_JOBS:
        raise ValueError(
            "full cache keys "
            "are not unique"
        )

    identities = [
        (
            str(job["profile"]),
            str(job["trial_name"]),
            str(job["judge_model"]),
        )
        for job in jobs
    ]

    if len(
        set(identities)
    ) != EXPECTED_JOBS:
        raise ValueError(
            "full job identities "
            "are not unique"
        )

    jobs.sort(
        key=lambda job: (
            str(job["profile"]),
            str(job["trial_name"]),
            str(
                job["judge_family"]
            ),
            str(job["judge_model"]),
        )
    )

    return jobs


def locate_job(
    *,
    output_dir: Path,
    inherited_roots: list[Path],
    data_root: Path,
    schema: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any]:
    writable = (
        dose.writable_path(
            output_dir=output_dir,
            job=job,
        )
    )

    status, artifact = (
        dose.terminal_artifact(
            path=writable,
            job=job,
            data_root=data_root,
            schema=schema,
        )
    )

    if status == "ok":
        return {
            "state": "writable_ok",
            "path": str(writable),
            "artifact": artifact,
            "root": str(
                output_dir
            ),
        }

    if status == "missing":
        return {
            "state": (
                "writable_missing"
            ),
            "path": str(writable),
            "artifact": artifact,
            "root": str(
                output_dir
            ),
        }

    if status == "nonterminal":
        return {
            "state": "partial",
            "path": str(writable),
            "artifact": artifact,
            "root": str(
                output_dir
            ),
        }

    if base.partial_attempts(
        output_dir=output_dir,
        cache_key=str(
            job["cache_key"]
        ),
    ):
        return {
            "state": "partial",
            "path": str(writable),
            "artifact": None,
            "root": str(
                output_dir
            ),
        }

    hits = []

    for root in inherited_roots:
        inherited = (
            dose.inherited_path(
                inherited_root=root,
                job=job,
            )
        )

        inherited_status, (
            inherited_artifact
        ) = dose.terminal_artifact(
            path=inherited,
            job=job,
            data_root=data_root,
            schema=schema,
        )

        if (
            inherited_status
            == "nonterminal"
        ):
            raise RuntimeError(
                "nonterminal artifact "
                "inside inherited cache: "
                f"{inherited}"
            )

        if base.partial_attempts(
            output_dir=root,
            cache_key=str(
                job["cache_key"]
            ),
        ) and (
            inherited_status
            not in {
                "ok",
                "missing",
            }
        ):
            raise RuntimeError(
                "partial attempt state "
                "inside inherited cache: "
                f"{root}"
            )

        if inherited_status in {
            "ok",
            "missing",
        }:
            hits.append({
                "status": (
                    inherited_status
                ),
                "artifact": (
                    inherited_artifact
                ),
                "path": str(
                    inherited
                ),
                "root": str(root),
            })

    if len(hits) > 1:
        raise ValueError(
            "same cache key found "
            "in multiple inherited roots: "
            f"{job['cache_key']}"
        )

    if hits:
        hit = hits[0]

        return {
            "state": (
                "inherited_ok"
                if hit["status"]
                == "ok"
                else (
                    "inherited_missing"
                )
            ),
            "path": hit["path"],
            "artifact": (
                hit["artifact"]
            ),
            "root": hit["root"],
        }

    return {
        "state": "pending",
        "path": "",
        "artifact": None,
        "root": "",
    }


def state_counts(
    *,
    jobs: list[
        dict[str, Any]
    ],
    output_dir: Path,
    inherited_roots: list[Path],
    data_root: Path,
    schema: dict[str, Any],
) -> Counter:
    return Counter(
        locate_job(
            output_dir=output_dir,
            inherited_roots=(
                inherited_roots
            ),
            data_root=data_root,
            schema=schema,
            job=job,
        )["state"]
        for job in jobs
    )


def print_state(
    counts: Counter,
) -> None:
    for state in (
        "inherited_ok",
        "inherited_missing",
        "writable_ok",
        "writable_missing",
        "partial",
        "pending",
    ):
        print(
            f"{state}:",
            counts.get(
                state,
                0,
            ),
        )


def write_consensus(
    *,
    jobs: list[
        dict[str, Any]
    ],
    output_dir: Path,
    inherited_roots: list[Path],
    data_root: Path,
    schema: dict[str, Any],
) -> Counter:
    by_trial: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = {}

    for job in jobs:
        identity = (
            str(job["profile"]),
            str(job["trial_name"]),
        )

        by_trial.setdefault(
            identity,
            [],
        ).append(job)

    if len(
        by_trial
    ) != EXPECTED_USABLE:
        raise ValueError(
            "expected exactly "
            f"{EXPECTED_USABLE} "
            "trajectory groups"
        )

    statuses = Counter()

    for identity, trial_jobs in (
        sorted(
            by_trial.items()
        )
    ):
        trial_jobs = sorted(
            trial_jobs,
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
            ),
        )

        if len(
            trial_jobs
        ) != EXPECTED_PANEL:
            raise ValueError(
                f"{identity}: expected "
                "two core judges"
            )

        entries = []
        sources = []

        for job in trial_jobs:
            located = locate_job(
                output_dir=output_dir,
                inherited_roots=(
                    inherited_roots
                ),
                data_root=data_root,
                schema=schema,
                job=job,
            )

            sources.append({
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
                "state": (
                    located["state"]
                ),
                "artifact_path": (
                    located["path"]
                ),
                "artifact_root": (
                    located["root"]
                ),
            })

            artifact = (
                located["artifact"]
            )

            if (
                isinstance(
                    artifact,
                    dict,
                )
                and artifact.get(
                    "status"
                )
                == "ok"
            ):
                entries.append(
                    artifact[
                        "final_cache_entry"
                    ]
                )
            else:
                entries.append(None)

        consensus = (
            orch.core_panel_consensus(
                schema=schema,
                judge_entries=entries,
            )
        )

        for result in (
            consensus[
                "fields"
            ].values()
        ):
            statuses[
                result["status"]
            ] += 1

        profile, trial_name = (
            identity
        )

        first_job = (
            trial_jobs[0]
        )

        path = (
            output_dir
            / "consensus"
            / (
                profile
                + "__"
                + trial_name
                + ".json"
            )
        )

        base.atomic_write_json(
            path,
            {
                "full_executor_version": (
                    FULL_EXECUTOR_VERSION
                ),
                "profile": profile,
                "trial_name": (
                    trial_name
                ),
                "condition": (
                    first_job[
                        "condition"
                    ]
                ),
                "placement": (
                    first_job[
                        "placement"
                    ]
                ),
                "pressure_type": (
                    first_job[
                        "pressure_type"
                    ]
                ),
                "artifact_sources": (
                    sources
                ),
                "consensus": (
                    consensus
                ),
            },
        )

    return statuses


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--full-spec",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--inherited-root",
        type=Path,
        action="append",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--max-new-jobs",
        type=int,
    )

    mode = (
        parser.add_mutually_exclusive_group(
            required=True
        )
    )

    mode.add_argument(
        "--dry-run",
        action="store_true",
    )

    mode.add_argument(
        "--execute",
        action="store_true",
    )

    args = parser.parse_args()

    manifest_path = (
        args.manifest
        .expanduser()
        .resolve()
    )

    spec_path = (
        args.full_spec
        .expanduser()
        .resolve()
    )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    data_root = (
        args.data_root
        .expanduser()
        .resolve()
    )

    inherited_roots = [
        path.expanduser().resolve()
        for path
        in args.inherited_root
    ]

    if len(
        set(inherited_roots)
    ) != len(
        inherited_roots
    ):
        raise ValueError(
            "duplicate inherited root"
        )

    if (
        output_dir
        in inherited_roots
    ):
        raise ValueError(
            "writable output cannot "
            "also be inherited"
        )

    if (
        args.max_new_jobs
        is not None
        and args.max_new_jobs < 1
    ):
        raise ValueError(
            "--max-new-jobs must "
            "be positive"
        )

    manifest = load_json(
        manifest_path
    )

    spec = load_json(
        spec_path
    )

    validate_manifest(
        manifest
    )

    validate_spec(
        manifest_path=manifest_path,
        manifest=manifest,
        spec=spec,
    )

    jobs = full_jobs(
        manifest
    )

    schema = panel.load_schema()

    before = state_counts(
        jobs=jobs,
        output_dir=output_dir,
        inherited_roots=(
            inherited_roots
        ),
        data_root=data_root,
        schema=schema,
    )

    print(
        "FROZEN FULL SEMANTIC "
        "PRODUCTION STATE"
    )
    print("=" * 72)

    print(
        "planned trajectories:",
        manifest[
            "planned_trajectories"
        ],
    )

    print(
        "semantic eligible:",
        manifest[
            "usable_trajectories"
        ],
    )

    print(
        "censored/error:",
        manifest[
            "censored_or_error"
        ],
    )

    print(
        "core judge jobs:",
        len(jobs),
    )

    print()

    print_state(before)

    if before.get(
        "partial",
        0,
    ):
        raise RuntimeError(
            "partial writable state "
            "must be inspected before "
            "provider execution"
        )

    if args.dry_run:
        print()
        print(
            "frozen full-universe "
            "lock: PASS"
        )
        print(
            "network calls: 0"
        )
        print(
            "FULL PRODUCTION "
            "DRY RUN: PASS"
        )
        return

    base_url = os.getenv(
        "LITE_LLM_URL",
        "",
    ).strip()

    if not base_url:
        raise ValueError(
            "LITE_LLM_URL is not set"
        )

    keys = base.parse_keys()

    new_jobs = 0
    provider_attempts = 0

    for index, job in enumerate(
        jobs,
        1,
    ):
        located = locate_job(
            output_dir=output_dir,
            inherited_roots=(
                inherited_roots
            ),
            data_root=data_root,
            schema=schema,
            job=job,
        )

        state = located[
            "state"
        ]

        if state in {
            "inherited_ok",
            "inherited_missing",
            "writable_ok",
            "writable_missing",
        }:
            continue

        if state == "partial":
            raise RuntimeError(
                "partial job encountered"
            )

        if (
            args.max_new_jobs
            is not None
            and new_jobs
            >= args.max_new_jobs
        ):
            break

        print(
            f"[{index:04d}/"
            f"{EXPECTED_JOBS}]",
            job["profile"],
            job["condition"],
            job["placement"],
            job["judge_family"],
            "pending",
        )

        row = base.frozen_row(
            data_root=data_root,
            job=job,
        )

        artifact = base.execute_job(
            job=job,
            row=row,
            schema=schema,
            output_dir=output_dir,
            base_url=base_url,
            keys=keys,
        )

        new_jobs += 1

        provider_attempts += int(
            artifact[
                "attempt_count"
            ]
        )

        print(
            "    ->",
            artifact["status"],
            "attempts=",
            artifact[
                "attempt_count"
            ],
        )

    after = state_counts(
        jobs=jobs,
        output_dir=output_dir,
        inherited_roots=(
            inherited_roots
        ),
        data_root=data_root,
        schema=schema,
    )

    complete = (
        after.get(
            "pending",
            0,
        )
        == 0
        and after.get(
            "partial",
            0,
        )
        == 0
    )

    consensus_status = {}

    if complete:
        consensus_status = dict(
            write_consensus(
                jobs=jobs,
                output_dir=output_dir,
                inherited_roots=(
                    inherited_roots
                ),
                data_root=data_root,
                schema=schema,
            )
        )

    summary = {
        "full_executor_version": (
            FULL_EXECUTOR_VERSION
        ),
        "mode": "full_2776",
        "complete": complete,
        "planned_trajectories": (
            EXPECTED_PLANNED
        ),
        "semantic_eligible": (
            EXPECTED_USABLE
        ),
        "censored_or_error": (
            EXPECTED_CENSORED
        ),
        "judge_job_count": (
            EXPECTED_JOBS
        ),
        "inherited_roots": [
            str(root)
            for root
            in inherited_roots
        ],
        "max_new_jobs": (
            args.max_new_jobs
        ),
        "new_jobs": new_jobs,
        "provider_attempts": (
            provider_attempts
        ),
        "before": dict(before),
        "after": dict(after),
        "consensus_field_status": (
            consensus_status
        ),
    }

    run_record = (
        base.write_run_summary(
            output_dir=output_dir,
            summary=summary,
        )
    )

    print()
    print(
        "FINAL FULL STATE"
    )
    print("=" * 72)

    print_state(after)

    print()
    print(
        "complete:",
        complete,
    )

    print(
        "new jobs:",
        new_jobs,
    )

    print(
        "provider attempts:",
        provider_attempts,
    )

    if complete:
        print(
            "consensus field status:",
            consensus_status,
        )
    else:
        print(
            "consensus:",
            "deferred until "
            "terminal coverage",
        )

    print(
        "run record:",
        run_record,
    )


if __name__ == "__main__":
    main()
