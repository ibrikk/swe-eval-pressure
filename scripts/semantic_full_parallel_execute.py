#!/usr/bin/env python3
"""Concurrent executor for the repaired historical semantic universe.

Properties:
- exact frozen 2,800 / 2,776 / 24 historical universe;
- repaired-Llama frozen rows loaded directly from manifest.frozen_root;
- legacy caches are read-only;
- legacy caches are forbidden from supplying repaired-Llama jobs;
- cache identity + request-body hash validated before inheritance;
- independent pending jobs execute concurrently;
- per-cache-key attempt journals/artifacts remain atomic and resumable;
- terminal missingness remains missing;
- no third-judge tiebreak;
- consensus is written only after terminal coverage.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from pathlib import Path
from typing import Any

import semantic_dose_execute as dose
import semantic_full_execute as full
import semantic_orchestrator as orch
import semantic_panel as panel
import semantic_pilot_execute as base


EXECUTOR_VERSION = "1.0-parallel-repaired-llama"

PROFILES = (
    "claude",
    "fable",
    "codex",
    "llama",
)


def load_frozen_rows(
    manifest: dict[str, Any],
) -> dict[
    tuple[str, str],
    dict[str, Any],
]:
    frozen_root = Path(
        str(manifest["frozen_root"])
    ).expanduser().resolve()

    rows_by_identity: dict[
        tuple[str, str],
        dict[str, Any],
    ] = {}

    for profile in PROFILES:
        path = (
            frozen_root
            / profile
            / "trials.json"
        )

        if not path.is_file():
            raise FileNotFoundError(path)

        rows = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        if (
            not isinstance(rows, list)
            or len(rows) != 700
        ):
            raise ValueError(
                f"{profile}: expected "
                "700 frozen rows"
            )

        usable = [
            row
            for row in rows
            if bool(
                row.get(
                    "substantive_usable"
                )
            )
        ]

        if len(usable) != 694:
            raise ValueError(
                f"{profile}: expected "
                "694 usable rows"
            )

        for row in usable:
            identity = (
                profile,
                str(
                    row["trial_name"]
                ),
            )

            if (
                identity
                in rows_by_identity
            ):
                raise ValueError(
                    "duplicate frozen "
                    f"identity: {identity}"
                )

            rows_by_identity[
                identity
            ] = row

    if len(
        rows_by_identity
    ) != 2776:
        raise ValueError(
            "expected exactly 2776 "
            "usable frozen rows"
        )

    return rows_by_identity


def frozen_row(
    *,
    rows_by_identity: dict[
        tuple[str, str],
        dict[str, Any],
    ],
    job: dict[str, Any],
) -> dict[str, Any]:
    identity = (
        str(job["profile"]),
        str(job["trial_name"]),
    )

    row = rows_by_identity.get(
        identity
    )

    if row is None:
        raise ValueError(
            "frozen row not found: "
            f"{identity}"
        )

    placement = str(
        row.get("placement")
        or row.get("channel")
        or ""
    )

    expected = {
        "condition": str(
            row.get(
                "condition",
                "",
            )
        ),
        "placement": placement,
        "pressure_type": str(
            row.get(
                "pressure_type",
                "",
            )
        ),
    }

    for field, actual in (
        expected.items()
    ):
        if (
            actual
            != str(job[field])
        ):
            raise ValueError(
                f"{identity}: "
                f"{field} mismatch"
            )

    return row


def request_hash(
    *,
    job: dict[str, Any],
    row: dict[str, Any],
    schema: dict[str, Any],
) -> str:
    _, _, body = (
        base.prepare_job(
            job=job,
            row=row,
            schema=schema,
        )
    )

    return (
        panel.canonical_json_hash(
            body
        )
    )


def terminal_artifact(
    *,
    path: Path,
    job: dict[str, Any],
    row: dict[str, Any],
    schema: dict[str, Any],
) -> tuple[
    str,
    dict[str, Any] | None,
]:
    artifact = (
        base.read_artifact(path)
    )

    if artifact is None:
        return "absent", None

    if not dose.artifact_identity_ok(
        artifact=artifact,
        job=job,
    ):
        raise ValueError(
            "artifact identity "
            f"mismatch: {path}"
        )

    expected_hash = request_hash(
        job=job,
        row=row,
        schema=schema,
    )

    actual_hash = str(
        artifact.get(
            "request_body_sha256",
            "",
        )
    )

    if (
        actual_hash
        != expected_hash
    ):
        raise ValueError(
            "artifact request-body "
            f"hash mismatch: {path}"
        )

    if (
        artifact.get("status")
        == "ok"
        and isinstance(
            artifact.get(
                "final_cache_entry"
            ),
            dict,
        )
        and artifact[
            "final_cache_entry"
        ].get("status")
        == "ok"
    ):
        return "ok", artifact

    if (
        artifact.get("status")
        == "missing"
        and artifact.get(
            "final_cache_entry"
        )
        is None
        and isinstance(
            artifact.get(
                "attempts"
            ),
            list,
        )
    ):
        return (
            "missing",
            artifact,
        )

    return (
        "nonterminal",
        artifact,
    )


def locate_job(
    *,
    job: dict[str, Any],
    row: dict[str, Any],
    schema: dict[str, Any],
    output_dir: Path,
    legacy_roots: list[Path],
    inherited_roots: list[Path],
) -> dict[str, Any]:
    writable = (
        dose.writable_path(
            output_dir=output_dir,
            job=job,
        )
    )

    status, artifact = (
        terminal_artifact(
            path=writable,
            job=job,
            row=row,
            schema=schema,
        )
    )

    if status == "ok":
        return {
            "state": "writable_ok",
            "artifact": artifact,
            "path": str(writable),
            "root": str(
                output_dir
            ),
        }

    if status == "missing":
        return {
            "state": (
                "writable_missing"
            ),
            "artifact": artifact,
            "path": str(writable),
            "root": str(
                output_dir
            ),
        }

    if status == "nonterminal":
        return {
            "state": "partial",
            "artifact": artifact,
            "path": str(writable),
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
            "artifact": None,
            "path": str(writable),
            "root": str(
                output_dir
            ),
        }

    hits = []

    for root_class, roots in (
        ("legacy", legacy_roots),
        (
            "inherited",
            inherited_roots,
        ),
    ):
        for root in roots:
            inherited = (
                dose.inherited_path(
                    inherited_root=root,
                    job=job,
                )
            )

            (
                inherited_status,
                inherited_artifact,
            ) = terminal_artifact(
                path=inherited,
                job=job,
                row=row,
                schema=schema,
            )

            if (
                inherited_status
                == "nonterminal"
            ):
                raise RuntimeError(
                    "nonterminal artifact "
                    "inside read-only "
                    f"cache: {inherited}"
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
                    "inside read-only "
                    f"cache: {root}"
                )

            if (
                inherited_status
                in {
                    "ok",
                    "missing",
                }
            ):
                if (
                    root_class
                    == "legacy"
                    and str(
                        job["profile"]
                    )
                    == "llama"
                ):
                    raise RuntimeError(
                        "legacy cache "
                        "attempted to supply "
                        "repaired-Llama job: "
                        f"{inherited}"
                    )

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
                    "root_class": (
                        root_class
                    ),
                })

    if len(hits) > 1:
        raise ValueError(
            "same final cache key "
            "found in multiple "
            "inherited roots: "
            f"{job['cache_key']}"
        )

    if hits:
        hit = hits[0]

        return {
            "state": (
                "inherited_ok"
                if (
                    hit["status"]
                    == "ok"
                )
                else (
                    "inherited_missing"
                )
            ),
            **hit,
        }

    return {
        "state": "pending",
        "artifact": None,
        "path": "",
        "root": "",
        "root_class": "",
    }


def state_counts(
    *,
    jobs: list[
        dict[str, Any]
    ],
    rows_by_identity,
    schema,
    output_dir,
    legacy_roots,
    inherited_roots,
) -> Counter:
    return Counter(
        locate_job(
            job=job,
            row=frozen_row(
                rows_by_identity=(
                    rows_by_identity
                ),
                job=job,
            ),
            schema=schema,
            output_dir=output_dir,
            legacy_roots=(
                legacy_roots
            ),
            inherited_roots=(
                inherited_roots
            ),
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
    jobs,
    rows_by_identity,
    schema,
    output_dir,
    legacy_roots,
    inherited_roots,
) -> Counter:
    by_trial: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = {}

    for job in jobs:
        identity = (
            str(job["profile"]),
            str(
                job["trial_name"]
            ),
        )

        by_trial.setdefault(
            identity,
            [],
        ).append(job)

    if len(by_trial) != 2776:
        raise ValueError(
            "expected 2776 "
            "trajectory groups"
        )

    statuses = Counter()

    for (
        profile,
        trial_name,
    ), trial_jobs in sorted(
        by_trial.items()
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
        ) != 2:
            raise ValueError(
                "expected two judges "
                f"for {(profile, trial_name)}"
            )

        entries = []
        sources = []

        for job in trial_jobs:
            located = locate_job(
                job=job,
                row=frozen_row(
                    rows_by_identity=(
                        rows_by_identity
                    ),
                    job=job,
                ),
                schema=schema,
                output_dir=(
                    output_dir
                ),
                legacy_roots=(
                    legacy_roots
                ),
                inherited_roots=(
                    inherited_roots
                ),
            )

            artifact = (
                located["artifact"]
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
                entries.append(
                    None
                )

        consensus = (
            orch.core_panel_consensus(
                schema=schema,
                judge_entries=(
                    entries
                ),
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

        first = trial_jobs[0]

        base.atomic_write_json(
            (
                output_dir
                / "consensus"
                / (
                    profile
                    + "__"
                    + trial_name
                    + ".json"
                )
            ),
            {
                "parallel_executor_version": (
                    EXECUTOR_VERSION
                ),
                "profile": profile,
                "trial_name": (
                    trial_name
                ),
                "condition": (
                    first[
                        "condition"
                    ]
                ),
                "placement": (
                    first[
                        "placement"
                    ]
                ),
                "pressure_type": (
                    first[
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
        "--legacy-root",
        action="append",
        type=Path,
        default=[],
    )

    parser.add_argument(
        "--inherited-root",
        action="append",
        type=Path,
        default=[],
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
        "--concurrency",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--max-new-jobs",
        type=int,
    )

    mode = (
        parser
        .add_mutually_exclusive_group(
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

    if args.concurrency < 1:
        raise ValueError(
            "--concurrency must "
            "be positive"
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

    legacy_roots = [
        p.expanduser().resolve()
        for p in args.legacy_root
    ]

    inherited_roots = [
        p.expanduser().resolve()
        for p
        in args.inherited_root
    ]

    all_roots = (
        legacy_roots
        + inherited_roots
    )

    if len(
        set(all_roots)
    ) != len(all_roots):
        raise ValueError(
            "duplicate inherited root"
        )

    if (
        output_dir
        in all_roots
    ):
        raise ValueError(
            "writable output "
            "cannot be inherited"
        )

    manifest = full.load_json(
        manifest_path
    )

    spec = full.load_json(
        spec_path
    )

    full.validate_manifest(
        manifest
    )

    full.validate_spec(
        manifest_path=(
            manifest_path
        ),
        manifest=manifest,
        spec=spec,
    )

    jobs = full.full_jobs(
        manifest
    )

    rows_by_identity = (
        load_frozen_rows(
            manifest
        )
    )

    schema = panel.load_schema()

    before = state_counts(
        jobs=jobs,
        rows_by_identity=(
            rows_by_identity
        ),
        schema=schema,
        output_dir=output_dir,
        legacy_roots=(
            legacy_roots
        ),
        inherited_roots=(
            inherited_roots
        ),
    )

    print(
        "REPAIRED HISTORICAL "
        "SEMANTIC PRODUCTION"
    )
    print("=" * 76)

    print(
        "planned trajectories:",
        2800,
    )
    print(
        "semantic eligible:",
        2776,
    )
    print(
        "censored/error:",
        24,
    )
    print(
        "core judge jobs:",
        5552,
    )
    print(
        "concurrency:",
        args.concurrency,
    )
    print()

    print_state(before)

    if before.get(
        "partial",
        0,
    ):
        raise RuntimeError(
            "partial writable "
            "state exists"
        )

    inherited_total = (
        before.get(
            "inherited_ok",
            0,
        )
        + before.get(
            "inherited_missing",
            0,
        )
    )

    print()
    print(
        "inherited terminal jobs:",
        inherited_total,
    )

    if args.dry_run:
        print()
        print(
            "network calls: 0"
        )
        print(
            "PARALLEL FULL "
            "DRY RUN: PASS"
        )
        return

    base_url = os.getenv(
        "LITE_LLM_URL",
        "",
    ).strip()

    if not base_url:
        raise ValueError(
            "LITE_LLM_URL "
            "is not set"
        )

    keys = base.parse_keys()

    pending: list[
        dict[str, Any]
    ] = []

    for job in jobs:
        located = locate_job(
            job=job,
            row=frozen_row(
                rows_by_identity=(
                    rows_by_identity
                ),
                job=job,
            ),
            schema=schema,
            output_dir=output_dir,
            legacy_roots=(
                legacy_roots
            ),
            inherited_roots=(
                inherited_roots
            ),
        )

        if (
            located["state"]
            == "pending"
        ):
            pending.append(job)

        elif (
            located["state"]
            == "partial"
        ):
            raise RuntimeError(
                "partial job "
                "encountered"
            )

    if (
        args.max_new_jobs
        is not None
    ):
        pending = pending[
            : args.max_new_jobs
        ]

    print()
    print(
        "new jobs selected:",
        len(pending),
    )

    worker_count = min(
        args.concurrency,
        max(
            len(pending),
            1,
        ),
    )

    print(
        "worker count:",
        worker_count,
    )

    completed_new = 0
    provider_attempts = 0
    worker_errors = []

    def worker(
        job: dict[str, Any],
    ):
        row = frozen_row(
            rows_by_identity=(
                rows_by_identity
            ),
            job=job,
        )

        artifact = (
            base.execute_job(
                job=job,
                row=row,
                schema=schema,
                output_dir=(
                    output_dir
                ),
                base_url=base_url,
                keys=keys,
            )
        )

        return artifact

    with ThreadPoolExecutor(
        max_workers=worker_count
    ) as pool:
        future_to_job = {
            pool.submit(
                worker,
                job,
            ): job
            for job in pending
        }

        for future in (
            as_completed(
                future_to_job
            )
        ):
            job = (
                future_to_job[
                    future
                ]
            )

            try:
                artifact = (
                    future.result()
                )

                completed_new += 1

                provider_attempts += int(
                    artifact.get(
                        "attempt_count",
                        len(
                            artifact.get(
                                "attempts",
                                [],
                            )
                        ),
                    )
                )

                if (
                    completed_new
                    % 25
                    == 0
                    or completed_new
                    == len(pending)
                ):
                    print(
                        "completed:",
                        f"{completed_new}/"
                        f"{len(pending)}",
                        "provider attempts:",
                        provider_attempts,
                    )

            except Exception as exc:
                worker_errors.append({
                    "cache_key": (
                        job[
                            "cache_key"
                        ]
                    ),
                    "profile": (
                        job["profile"]
                    ),
                    "trial_name": (
                        job[
                            "trial_name"
                        ]
                    ),
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
                    "exception_type": (
                        type(exc)
                        .__name__
                    ),
                    "exception_message": (
                        str(exc)
                    ),
                })

                print(
                    "WORKER ERROR:",
                    job["profile"],
                    job[
                        "trial_name"
                    ],
                    job[
                        "judge_family"
                    ],
                    type(exc)
                    .__name__,
                    str(exc),
                )

    after = state_counts(
        jobs=jobs,
        rows_by_identity=(
            rows_by_identity
        ),
        schema=schema,
        output_dir=output_dir,
        legacy_roots=(
            legacy_roots
        ),
        inherited_roots=(
            inherited_roots
        ),
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
                rows_by_identity=(
                    rows_by_identity
                ),
                schema=schema,
                output_dir=(
                    output_dir
                ),
                legacy_roots=(
                    legacy_roots
                ),
                inherited_roots=(
                    inherited_roots
                ),
            )
        )

    summary = {
        "parallel_executor_version": (
            EXECUTOR_VERSION
        ),
        "mode": (
            "historical_full_"
            "repaired_llama"
        ),
        "concurrency": (
            args.concurrency
        ),
        "max_new_jobs": (
            args.max_new_jobs
        ),
        "complete": complete,
        "planned_trajectories": (
            2800
        ),
        "semantic_eligible": (
            2776
        ),
        "censored_or_error": (
            24
        ),
        "judge_job_count": (
            5552
        ),
        "legacy_roots": [
            str(root)
            for root
            in legacy_roots
        ],
        "inherited_roots": [
            str(root)
            for root
            in inherited_roots
        ],
        "new_jobs": (
            completed_new
        ),
        "provider_attempts": (
            provider_attempts
        ),
        "worker_errors": (
            worker_errors
        ),
        "before": dict(
            before
        ),
        "after": dict(
            after
        ),
        "consensus_field_status": (
            consensus_status
        ),
    }

    run_record = (
        base.write_run_summary(
            output_dir=(
                output_dir
            ),
            summary=summary,
        )
    )

    print()
    print("FINAL STATE")
    print("=" * 76)

    print_state(after)

    print()
    print(
        "complete:",
        complete,
    )
    print(
        "new jobs:",
        completed_new,
    )
    print(
        "provider attempts:",
        provider_attempts,
    )
    print(
        "worker errors:",
        len(
            worker_errors
        ),
    )

    if complete:
        print(
            "consensus:",
            consensus_status,
        )

    print(
        "run record:",
        run_record,
    )

    if worker_errors:
        raise RuntimeError(
            f"{len(worker_errors)} "
            "worker errors; inspect "
            "run summary before resume"
        )


if __name__ == "__main__":
    main()
