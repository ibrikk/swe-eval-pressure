#!/usr/bin/env python3
"""Execute one frozen infrastructure-only semantic recovery stage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from pathlib import Path

import semantic_full_execute as full
import semantic_full_parallel_execute as repaired
import semantic_panel as panel
import semantic_pilot_execute as base


EXECUTOR_VERSION = "1.0"


def sha(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def load_json(path: Path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def classify(
    *,
    output_dir,
    job,
):
    path = (
        output_dir
        / "jobs"
        / (
            str(job["cache_key"])
            + ".json"
        )
    )

    artifact = (
        base.read_artifact(path)
    )

    if artifact is None:
        if base.partial_attempts(
            output_dir=output_dir,
            cache_key=str(
                job["cache_key"]
            ),
        ):
            return "partial"

        return "pending"

    if (
        artifact.get("cache_key")
        != job["cache_key"]
    ):
        raise ValueError(
            "recovery artifact "
            "cache-key mismatch"
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
    ):
        return "ok"

    if (
        artifact.get("status")
        == "missing"
    ):
        return "missing"

    return "partial"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
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

    parser.add_argument(
        "--concurrency",
        type=int,
        default=32,
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

    manifest_path = (
        args.manifest
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
        manifest_path
    )

    policy = load_json(
        policy_path
    )

    if (
        policy[
            "selection_status"
        ]
        != (
            "frozen_before_"
            "recovery_outcomes"
        )
    ):
        raise ValueError(
            "recovery policy not frozen"
        )

    if (
        sha(manifest_path)
        != policy[
            "source_manifest_sha256"
        ]
    ):
        raise ValueError(
            "manifest hash mismatch"
        )

    if int(
        policy[
            "eligible_job_count"
        ]
    ) != 867:
        raise ValueError(
            "expected 867 recovery jobs"
        )

    all_jobs = {
        str(j["cache_key"]): j
        for j in manifest["jobs"]
    }

    jobs = []

    for frozen in (
        policy["eligible_jobs"]
    ):
        key = str(
            frozen["cache_key"]
        )

        if key not in all_jobs:
            raise ValueError(
                "recovery cache key "
                "missing from manifest"
            )

        job = dict(
            all_jobs[key]
        )

        for field in (
            "profile",
            "trial_name",
            "condition",
            "placement",
            "pressure_type",
            "judge_family",
            "judge_model",
            "trajectory_hash",
        ):
            if (
                str(job[field])
                != str(
                    frozen[field]
                )
            ):
                raise ValueError(
                    f"{key}: frozen "
                    f"{field} mismatch"
                )

        jobs.append(job)

    if len({
        j["cache_key"]
        for j in jobs
    }) != 867:
        raise ValueError(
            "duplicate recovery jobs"
        )

    jobs.sort(
        key=lambda j: (
            str(j["profile"]),
            str(j["trial_name"]),
            str(
                j["judge_family"]
            ),
        )
    )

    rows = (
        repaired.load_frozen_rows(
            manifest
        )
    )

    schema = (
        panel.load_schema()
    )

    before = Counter(
        classify(
            output_dir=output_dir,
            job=job,
        )
        for job in jobs
    )

    print(
        "FROZEN SEMANTIC "
        "INFRA RECOVERY"
    )
    print("=" * 76)

    print(
        "eligible jobs:",
        len(jobs),
    )
    print(
        "concurrency:",
        args.concurrency,
    )
    print(
        "policy sha256:",
        sha(policy_path),
    )
    print()

    for state in (
        "ok",
        "missing",
        "partial",
        "pending",
    ):
        print(
            f"{state}:",
            before.get(
                state,
                0,
            ),
        )

    if before.get(
        "partial",
        0,
    ):
        raise RuntimeError(
            "partial recovery state"
        )

    if args.dry_run:
        print()
        print("network calls: 0")
        print("RECOVERY DRY RUN: PASS")
        return

    base_url = os.getenv(
        "LITE_LLM_URL",
        "",
    ).strip()

    if not base_url:
        raise ValueError(
            "LITE_LLM_URL not set"
        )

    keys = base.parse_keys()

    pending = [
        job
        for job in jobs
        if classify(
            output_dir=output_dir,
            job=job,
        )
        == "pending"
    ]

    print()
    print(
        "jobs selected:",
        len(pending),
    )

    completed = 0
    provider_attempts = 0
    errors = []

    def worker(job):
        row = repaired.frozen_row(
            rows_by_identity=rows,
            job=job,
        )

        return base.execute_job(
            job=job,
            row=row,
            schema=schema,
            output_dir=output_dir,
            base_url=base_url,
            keys=keys,
        )

    with ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as pool:
        futures = {
            pool.submit(
                worker,
                job,
            ): job
            for job in pending
        }

        for future in (
            as_completed(futures)
        ):
            job = futures[future]

            try:
                artifact = (
                    future.result()
                )

                completed += 1

                provider_attempts += int(
                    artifact.get(
                        "attempt_count",
                        0,
                    )
                )

                if (
                    completed % 25 == 0
                    or completed
                    == len(pending)
                ):
                    print(
                        f"completed "
                        f"{completed}/"
                        f"{len(pending)} "
                        f"attempts="
                        f"{provider_attempts}"
                    )

            except Exception as exc:
                errors.append({
                    "cache_key": (
                        job["cache_key"]
                    ),
                    "profile": (
                        job["profile"]
                    ),
                    "trial_name": (
                        job["trial_name"]
                    ),
                    "judge": (
                        job[
                            "judge_model"
                        ]
                    ),
                    "exception_type": (
                        type(exc).__name__
                    ),
                    "message": str(exc),
                })

                print(
                    "WORKER ERROR:",
                    job["profile"],
                    job["trial_name"],
                    job["judge_family"],
                    type(exc).__name__,
                    str(exc),
                )

    after = Counter(
        classify(
            output_dir=output_dir,
            job=job,
        )
        for job in jobs
    )

    summary = {
        "recovery_executor_version": (
            EXECUTOR_VERSION
        ),
        "policy": str(
            policy_path
        ),
        "policy_sha256": sha(
            policy_path
        ),
        "eligible_jobs": (
            len(jobs)
        ),
        "concurrency": (
            args.concurrency
        ),
        "new_jobs": completed,
        "provider_attempts": (
            provider_attempts
        ),
        "worker_errors": errors,
        "before": dict(before),
        "after": dict(after),
    }

    record = (
        base.write_run_summary(
            output_dir=output_dir,
            summary=summary,
        )
    )

    print()
    print("FINAL RECOVERY STATE")
    print("=" * 76)

    for state in (
        "ok",
        "missing",
        "partial",
        "pending",
    ):
        print(
            f"{state}:",
            after.get(
                state,
                0,
            ),
        )

    print()
    print(
        "provider attempts:",
        provider_attempts,
    )
    print(
        "worker errors:",
        len(errors),
    )
    print(
        "run record:",
        record,
    )

    if errors:
        raise RuntimeError(
            "worker errors detected"
        )


if __name__ == "__main__":
    main()
