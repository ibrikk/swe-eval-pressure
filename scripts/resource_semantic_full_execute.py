#!/usr/bin/env python3
"""Full resource semantic production executor.

Frozen universe:
- 618 eligible trajectories
- Claude, Fable, Codex only
- 2 independent core judges
- 1,236 judge jobs
- resource semantic rubric 1.1

The runner:
- verifies frozen manifest/spec/schema hashes;
- reuses the validated blinded resource pilot transport;
- never exposes condition/outcome metadata to judges;
- validates cached request-body compatibility;
- preserves terminal missingness;
- refuses to continue through partial state;
- supports deterministic resumable batches;
- makes zero calls in dry-run mode.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import semantic_bulk_runner as planner
import semantic_panel as panel
import semantic_pilot_execute as base
import resource_semantic_pilot_execute as resource


RUNNER_VERSION = "resource-full-executor-1.0"

EXPECTED_MANIFEST_SHA256 = (
    "1ab4cf3074d3e4e57232adf0f965d479"
    "c8f23b3ecdc58c907eadfce442c5a2ca"
)

EXPECTED_SCHEMA_SHA256 = (
    "4083d8075889bb84440847e6215a0064"
    "e55262884c048b242ef2fd3f3498fcfa"
)

EXPECTED_SPEC_SHA256 = (
    "abd67416f2637a3853596a64703346b6"
    "73babcda409305d0567db7e83fd444d1"
)

EXPECTED_TRAJECTORIES = 618
EXPECTED_JOBS = 1236

PROFILES = (
    "claude",
    "fable",
    "codex",
)


def load_json(
    path: Path,
) -> dict[str, Any]:
    value = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        value,
        dict,
    ):
        raise ValueError(
            f"{path}: expected JSON object"
        )

    return value


def sha256_file(
    path: Path,
) -> str:
    return (
        resource.sha256_file(
            path
        )
    )


def require_hash(
    *,
    path: Path,
    expected: str,
    label: str,
) -> None:
    actual = sha256_file(
        path
    )

    if actual != expected:
        raise ValueError(
            f"{label} hash mismatch\n"
            f"expected={expected}\n"
            f"actual={actual}"
        )


def validate_frozen_inputs(
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    schema_path: Path,
    schema: dict[str, Any],
    spec_path: Path,
    spec: dict[str, Any],
) -> None:
    require_hash(
        path=manifest_path,
        expected=(
            EXPECTED_MANIFEST_SHA256
        ),
        label="Stage A manifest",
    )

    require_hash(
        path=schema_path,
        expected=(
            EXPECTED_SCHEMA_SHA256
        ),
        label="resource semantic schema",
    )

    require_hash(
        path=spec_path,
        expected=(
            EXPECTED_SPEC_SHA256
        ),
        label="full production spec",
    )

    if (
        spec[
            "selection_status"
        ]
        != (
            "frozen_full_resource_"
            "semantic_universe_before_"
            "production_outputs"
        )
    ):
        raise ValueError(
            "full production universe "
            "is not marked frozen"
        )

    if (
        spec[
            "source_manifest_sha256"
        ]
        != EXPECTED_MANIFEST_SHA256
    ):
        raise ValueError(
            "spec manifest hash mismatch"
        )

    if (
        spec[
            "semantic_schema_sha256"
        ]
        != EXPECTED_SCHEMA_SHA256
    ):
        raise ValueError(
            "spec schema hash mismatch"
        )

    if (
        schema["schema_version"]
        != "resource-1.1"
        or schema[
            "rubric_version"
        ]
        != "resource-1.1"
    ):
        raise ValueError(
            "unexpected schema/rubric identity"
        )

    if (
        manifest[
            "semantic_schema_version"
        ]
        != "resource-1.1"
        or manifest[
            "rubric_version"
        ]
        != "resource-1.1"
    ):
        raise ValueError(
            "manifest semantic identity mismatch"
        )

    checks = {
        "semantic_eligible_trajectories": (
            EXPECTED_TRAJECTORIES
        ),
        "core_judge_jobs": (
            EXPECTED_JOBS
        ),
        "unique_cache_keys": (
            EXPECTED_JOBS
        ),
        "primary_panel_size": 2,
    }

    for field, expected in (
        checks.items()
    ):
        if int(
            spec[field]
        ) != expected:
            raise ValueError(
                f"spec {field}: "
                f"expected {expected}, "
                f"found {spec[field]}"
            )

    if int(
        manifest[
            "semantic_eligible_trajectories"
        ]
    ) != EXPECTED_TRAJECTORIES:
        raise ValueError(
            "manifest trajectory denominator "
            "mismatch"
        )

    if int(
        manifest[
            "core_judge_jobs"
        ]
    ) != EXPECTED_JOBS:
        raise ValueError(
            "manifest job denominator mismatch"
        )

    if int(
        manifest[
            "unique_cache_keys"
        ]
    ) != EXPECTED_JOBS:
        raise ValueError(
            "manifest cache-key denominator "
            "mismatch"
        )

    if (
        spec[
            "pilot_outputs_reused_as_"
            "production_outputs"
        ]
        is not False
    ):
        raise ValueError(
            "pilot outputs must not be "
            "production outputs"
        )

    if (
        spec[
            "resource_1_0_outputs_reused"
        ]
        is not False
    ):
        raise ValueError(
            "resource-1.0 outputs must not "
            "be reused"
        )


def full_jobs(
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    jobs = [
        dict(job)
        for job
        in manifest["jobs"]
    ]

    if len(jobs) != EXPECTED_JOBS:
        raise ValueError(
            f"expected {EXPECTED_JOBS} jobs, "
            f"found {len(jobs)}"
        )

    cache_keys = [
        str(
            job["cache_key"]
        )
        for job in jobs
    ]

    if len(
        set(cache_keys)
    ) != EXPECTED_JOBS:
        raise ValueError(
            "cache keys are not unique"
        )

    profiles = {
        str(
            job["profile"]
        )
        for job in jobs
    }

    if profiles != set(
        PROFILES
    ):
        raise ValueError(
            f"unexpected profiles: {profiles}"
        )

    if any(
        str(
            job["profile"]
        )
        == "llama"
        for job in jobs
    ):
        raise ValueError(
            "Llama must have zero primary "
            "resource semantic jobs"
        )

    identities = {
        (
            str(job["profile"]),
            str(job["trial_name"]),
        )
        for job in jobs
    }

    if (
        len(identities)
        != EXPECTED_TRAJECTORIES
    ):
        raise ValueError(
            "trajectory identity denominator "
            "mismatch"
        )

    for identity in identities:
        matches = [
            job
            for job in jobs
            if (
                str(
                    job["profile"]
                ),
                str(
                    job["trial_name"]
                ),
            )
            == identity
        ]

        if len(matches) != 2:
            raise ValueError(
                f"{identity}: expected "
                "exactly two judge jobs"
            )

        families = {
            str(
                job[
                    "judge_family"
                ]
            )
            for job in matches
        }

        if families != {
            "deepseek",
            "gemini",
        }:
            raise ValueError(
                f"{identity}: unexpected "
                f"judge families {families}"
            )

    jobs.sort(
        key=lambda job: (
            str(job["profile"]),
            str(job["condition"]),
            str(job["trial_name"]),
            str(
                job["judge_family"]
            ),
            str(job["judge_model"]),
        )
    )

    return jobs


def current_request_hash(
    *,
    job: dict[str, Any],
    schema: dict[str, Any],
) -> tuple[
    str,
    int,
]:
    agent_blocks, body = (
        resource.prepare_blinded_job(
            job=job,
            schema=schema,
        )
    )

    return (
        panel.canonical_json_hash(
            body
        ),
        len(agent_blocks),
    )


def job_state(
    *,
    job: dict[str, Any],
    schema: dict[str, Any],
    output_dir: Path,
) -> str:
    path = (
        planner.job_artifact_path(
            output_dir,
            job,
        )
    )

    partials = (
        base.partial_attempts(
            output_dir=output_dir,
            cache_key=str(
                job["cache_key"]
            ),
        )
    )

    if not path.is_file():
        if partials:
            return "partial"

        return "pending"

    try:
        artifact = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except Exception as exc:
        raise RuntimeError(
            f"invalid artifact JSON: {path}"
        ) from exc

    if not isinstance(
        artifact,
        dict,
    ):
        raise RuntimeError(
            f"invalid artifact object: {path}"
        )

    if str(
        artifact.get(
            "cache_key",
            "",
        )
    ) != str(
        job["cache_key"]
    ):
        raise RuntimeError(
            f"cache-key mismatch: {path}"
        )

    expected_request_hash, _ = (
        current_request_hash(
            job=job,
            schema=schema,
        )
    )

    actual_request_hash = str(
        artifact.get(
            "request_body_sha256",
            "",
        )
    )

    if (
        actual_request_hash
        != expected_request_hash
    ):
        raise RuntimeError(
            "cached request-body mismatch:\n"
            f"{path}\n"
            f"expected={expected_request_hash}\n"
            f"actual={actual_request_hash}"
        )

    if (
        artifact.get(
            "semantic_schema_version"
        )
        != "resource-1.1"
        or artifact.get(
            "rubric_version"
        )
        != "resource-1.1"
    ):
        raise RuntimeError(
            f"cached semantic identity "
            f"mismatch: {path}"
        )

    status = artifact.get(
        "status"
    )

    if status == "ok":
        entry = artifact.get(
            "final_cache_entry"
        )

        if not (
            isinstance(
                entry,
                dict,
            )
            and entry.get(
                "status"
            )
            == "ok"
        ):
            raise RuntimeError(
                "artifact claims ok without "
                f"valid final cache entry: {path}"
            )

        return "cached_ok"

    if status == "missing":
        return "cached_missing"

    return "partial"


def state_counts(
    *,
    jobs: list[dict[str, Any]],
    schema: dict[str, Any],
    output_dir: Path,
) -> Counter:
    return Counter(
        job_state(
            job=job,
            schema=schema,
            output_dir=output_dir,
        )
        for job in jobs
    )


def print_state(
    *,
    title: str,
    counts: Counter,
) -> None:
    print(title)
    print("=" * 72)

    for name in (
        "cached_ok",
        "cached_missing",
        "partial",
        "pending",
    ):
        print(
            f"{name}:",
            counts.get(
                name,
                0,
            ),
        )


def request_audit(
    *,
    jobs: list[dict[str, Any]],
    schema: dict[str, Any],
    data_root: Path,
) -> Counter:
    counts = Counter()

    for job in jobs:
        # Verify frozen trial metadata internally.
        # This information is NOT sent to judges.
        resource.load_resource_row(
            data_root=data_root,
            job=job,
        )

        _, agent_count = (
            current_request_hash(
                job=job,
                schema=schema,
            )
        )

        counts[
            "requests_prepared"
        ] += 1

        if agent_count == 0:
            counts[
                "no_agent_evidence"
            ] += 1

    return counts


def select_batch(
    *,
    jobs: list[dict[str, Any]],
    start: int,
    size: int,
) -> list[dict[str, Any]]:
    if start < 0:
        raise ValueError(
            "--batch-start must be >= 0"
        )

    if size < 0:
        raise ValueError(
            "--batch-size must be >= 0"
        )

    if start > len(jobs):
        raise ValueError(
            "--batch-start exceeds job count"
        )

    if size == 0:
        return jobs[start:]

    return jobs[
        start:
        start + size
    ]


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
        "--schema",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--spec",
        type=Path,
        default=(
            Path("config")
            / "resource_semantic_full_v1.1.json"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--batch-start",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
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

    data_root = (
        args.data_root
        .expanduser()
        .resolve()
    )

    manifest_path = (
        args.manifest
        if args.manifest
        else (
            data_root
            / "analysis"
            / "semantic-resource-v1"
            / "manifests"
            / "resource-stage-a-v1.1.json"
        )
    ).expanduser().resolve()

    schema_path = (
        args.schema
        if args.schema
        else (
            data_root
            / "config"
            / "resource_semantic_schema_v1.json"
        )
    ).expanduser().resolve()

    spec_path = (
        args.spec
        .expanduser()
        .resolve()
    )

    output_dir = (
        args.output_dir
        if args.output_dir
        else (
            data_root
            / "analysis"
            / "semantic-resource-v1"
            / "full"
            / "production-v1.1"
        )
    ).expanduser().resolve()

    manifest = load_json(
        manifest_path
    )

    schema = panel.load_schema(
        schema_path
    )

    spec = load_json(
        spec_path
    )

    validate_frozen_inputs(
        manifest_path=manifest_path,
        manifest=manifest,
        schema_path=schema_path,
        schema=schema,
        spec_path=spec_path,
        spec=spec,
    )

    jobs = full_jobs(
        manifest
    )

    batch = select_batch(
        jobs=jobs,
        start=args.batch_start,
        size=args.batch_size,
    )

    audit = request_audit(
        jobs=jobs,
        schema=schema,
        data_root=data_root,
    )

    universe_before = (
        state_counts(
            jobs=jobs,
            schema=schema,
            output_dir=output_dir,
        )
    )

    batch_before = (
        state_counts(
            jobs=batch,
            schema=schema,
            output_dir=output_dir,
        )
    )

    print(
        "RESOURCE SEMANTIC FULL PRODUCTION"
    )
    print("=" * 72)

    print(
        "frozen manifest: PASS"
    )
    print(
        "frozen schema: PASS"
    )
    print(
        "frozen production spec: PASS"
    )

    print()
    print(
        "eligible trajectories:",
        EXPECTED_TRAJECTORIES,
    )
    print(
        "core judge jobs:",
        EXPECTED_JOBS,
    )
    print(
        "requests prepared:",
        audit[
            "requests_prepared"
        ],
    )
    print(
        "jobs with no agent evidence:",
        audit[
            "no_agent_evidence"
        ],
    )

    print()
    print(
        "batch start:",
        args.batch_start,
    )
    print(
        "batch size requested:",
        args.batch_size,
    )
    print(
        "batch jobs selected:",
        len(batch),
    )

    print()
    print_state(
        title="FULL UNIVERSE STATE",
        counts=universe_before,
    )

    print()
    print_state(
        title="SELECTED BATCH STATE",
        counts=batch_before,
    )

    if universe_before.get(
        "partial",
        0,
    ):
        raise RuntimeError(
            "partial semantic-production state "
            "exists; inspect before making "
            "any further provider calls"
        )

    if args.dry_run:
        print()
        print(
            "structured treatment metadata "
            "supplied to judges: 0"
        )
        print(
            "condition labels supplied "
            "to judges: 0"
        )
        print(
            "verifier outcomes supplied "
            "to judges: 0"
        )
        print(
            "task-success outcomes supplied "
            "to judges: 0"
        )
        print()
        print("network calls: 0")
        print("judge calls: 0")
        print(
            "RESOURCE SEMANTIC FULL "
            "DRY RUN: PASS"
        )
        return

    base_url = (
        __import__("os")
        .environ.get(
            "LITE_LLM_URL",
            "",
        )
        .strip()
    )

    if not base_url:
        raise ValueError(
            "LITE_LLM_URL is not set"
        )

    keys = base.parse_keys()

    new_jobs = 0
    provider_attempts = 0

    for offset, job in enumerate(
        batch,
        1,
    ):
        absolute_index = (
            args.batch_start
            + offset
        )

        state = job_state(
            job=job,
            schema=schema,
            output_dir=output_dir,
        )

        print(
            f"[{absolute_index:04d}/"
            f"{EXPECTED_JOBS}]",
            job["profile"],
            job["condition"],
            job["judge_family"],
            state,
        )

        if state in {
            "cached_ok",
            "cached_missing",
        }:
            continue

        if state == "partial":
            raise RuntimeError(
                "partial job encountered "
                "during execution"
            )

        artifact = (
            resource.execute_job(
                job=job,
                schema=schema,
                output_dir=output_dir,
                base_url=base_url,
                keys=keys,
            )
        )

        # Add full-production provenance
        # without altering the judgment.
        artifact[
            "production_runner_version"
        ] = RUNNER_VERSION

        artifact[
            "production_spec_sha256"
        ] = EXPECTED_SPEC_SHA256

        artifact[
            "source_manifest_sha256"
        ] = EXPECTED_MANIFEST_SHA256

        path = (
            planner.job_artifact_path(
                output_dir,
                job,
            )
        )

        base.atomic_write_json(
            path,
            artifact,
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

    universe_after = (
        state_counts(
            jobs=jobs,
            schema=schema,
            output_dir=output_dir,
        )
    )

    batch_after = (
        state_counts(
            jobs=batch,
            schema=schema,
            output_dir=output_dir,
        )
    )

    summary = {
        "production_runner_version": (
            RUNNER_VERSION
        ),
        "production_spec_sha256": (
            EXPECTED_SPEC_SHA256
        ),
        "source_manifest_sha256": (
            EXPECTED_MANIFEST_SHA256
        ),
        "batch_start": (
            args.batch_start
        ),
        "batch_size_requested": (
            args.batch_size
        ),
        "batch_jobs_selected": (
            len(batch)
        ),
        "new_jobs": new_jobs,
        "provider_attempts": (
            provider_attempts
        ),
        "universe_before": (
            dict(
                universe_before
            )
        ),
        "universe_after": (
            dict(
                universe_after
            )
        ),
        "batch_before": (
            dict(
                batch_before
            )
        ),
        "batch_after": (
            dict(
                batch_after
            )
        ),
    }

    run_record = (
        base.write_run_summary(
            output_dir=output_dir,
            summary=summary,
        )
    )

    print()
    print_state(
        title="FINAL FULL UNIVERSE STATE",
        counts=universe_after,
    )

    print()
    print_state(
        title="FINAL SELECTED BATCH STATE",
        counts=batch_after,
    )

    print()
    print(
        "new jobs:",
        new_jobs,
    )
    print(
        "provider attempts:",
        provider_attempts,
    )
    print(
        "run record:",
        run_record,
    )

    print()
    print(
        "RESOURCE SEMANTIC FULL "
        "BATCH: COMPLETE"
    )


if __name__ == "__main__":
    main()
