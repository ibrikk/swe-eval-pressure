#!/usr/bin/env python3
"""Execute the frozen 100-trajectory semantic dose.

The frozen pilot archive is treated as a read-only inherited cache.
New dose artifacts are written only to the dose output directory.

Dry-run mode performs NO network/API calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import semantic_bulk_runner as planner
import semantic_dose as selector
import semantic_orchestrator as orch
import semantic_panel as panel
import semantic_pilot_execute as base


DOSE_EXECUTOR_VERSION = "1.0"

IDENTITY_FIELDS = (
    "profile",
    "condition",
    "placement",
    "pressure_type",
    "judge_family",
    "judge_model",
    "trial_name",
    "trajectory_hash",
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


def frozen_signature(
    trial: dict[str, Any],
) -> tuple[Any, ...]:
    return (
        str(trial["profile"]),
        str(trial["condition"]),
        str(trial["placement"]),
        str(trial["pressure_type"]),
        str(trial["trial_name"]),
        str(trial["trajectory_hash"]),
        str(trial["dose_hash"]),
        bool(
            trial[
                "from_frozen_pilot"
            ]
        ),
    )


def validate_frozen_dose(
    *,
    manifest_path: Path,
    pilot_spec_path: Path,
    manifest: dict[str, Any],
    pilot_spec: dict[str, Any],
    dose_spec: dict[str, Any],
    generated: dict[str, Any],
) -> None:
    if (
        dose_spec.get(
            "selection_status"
        )
        != "frozen_before_dose_outcomes"
    ):
        raise ValueError(
            "dose is not marked frozen "
            "before dose outcomes"
        )

    if (
        sha256_file(
            manifest_path
        )
        != dose_spec[
            "source_manifest_sha256"
        ]
    ):
        raise ValueError(
            "historical manifest hash mismatch"
        )

    if (
        sha256_file(
            pilot_spec_path
        )
        != dose_spec[
            "source_pilot_spec_sha256"
        ]
    ):
        raise ValueError(
            "pilot spec hash mismatch"
        )

    for field in (
        "semantic_schema_version",
        "rubric_version",
        "semantic_view_version",
    ):
        if (
            str(
                dose_spec[field]
            )
            != str(
                manifest[field]
            )
        ):
            raise ValueError(
                f"{field} mismatch"
            )

    if (
        int(
            dose_spec[
                "trajectory_count"
            ]
        )
        != 100
        or int(
            dose_spec[
                "core_judge_jobs"
            ]
        )
        != 200
    ):
        raise ValueError(
            "unexpected dose size"
        )

    frozen = [
        frozen_signature(x)
        for x
        in dose_spec["trials"]
    ]

    regenerated = [
        frozen_signature(x)
        for x
        in generated["trials"]
    ]

    if frozen != regenerated:
        raise ValueError(
            "regenerated dose does not "
            "exactly match frozen dose spec"
        )

    expected_judges = {
        (
            str(j["family"]),
            str(j["model"]),
        )
        for j
        in dose_spec[
            "primary_judges"
        ]
    }

    actual_judges = {
        (
            str(j["judge_family"]),
            str(j["judge_model"]),
        )
        for trial
        in generated["trials"]
        for j in trial["jobs"]
    }

    if (
        expected_judges
        != actual_judges
    ):
        raise ValueError(
            "core judge panel mismatch"
        )


def dose_jobs(
    dose: dict[str, Any],
) -> list[dict[str, Any]]:
    jobs = []

    for trial in dose["trials"]:
        for source in trial["jobs"]:
            job = dict(source)

            job[
                "from_frozen_pilot"
            ] = bool(
                trial[
                    "from_frozen_pilot"
                ]
            )

            jobs.append(job)

    jobs.sort(
        key=lambda job: (
            str(job["profile"]),
            str(job["condition"]),
            str(job["placement"]),
            str(job["trial_name"]),
            str(
                job["judge_family"]
            ),
            str(
                job["judge_model"]
            ),
        )
    )

    if len(jobs) != 200:
        raise ValueError(
            "expected 200 dose jobs"
        )

    if len({
        str(j["cache_key"])
        for j in jobs
    }) != 200:
        raise ValueError(
            "dose job cache keys "
            "are not unique"
        )

    return jobs


def artifact_identity_ok(
    *,
    artifact: dict[str, Any],
    job: dict[str, Any],
) -> bool:
    if (
        artifact.get(
            "cache_key"
        )
        != job["cache_key"]
    ):
        return False

    for field in IDENTITY_FIELDS:
        if (
            str(
                artifact.get(
                    field,
                    "",
                )
            )
            != str(
                job.get(
                    field,
                    "",
                )
            )
        ):
            return False

    return True


def request_hash_for_job(
    *,
    data_root: Path,
    schema: dict[str, Any],
    job: dict[str, Any],
) -> str:
    row = base.frozen_row(
        data_root=data_root,
        job=job,
    )

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
    data_root: Path,
    schema: dict[str, Any],
) -> tuple[
    str,
    dict[str, Any] | None,
]:
    artifact = base.read_artifact(
        path
    )

    if artifact is None:
        return "absent", None

    if not artifact_identity_ok(
        artifact=artifact,
        job=job,
    ):
        raise ValueError(
            "artifact identity mismatch: "
            f"{path}"
        )

    stored_request_hash = str(
        artifact.get(
            "request_body_sha256",
            "",
        )
    )

    expected_request_hash = (
        request_hash_for_job(
            data_root=data_root,
            schema=schema,
            job=job,
        )
    )

    if (
        stored_request_hash
        != expected_request_hash
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
        return "missing", artifact

    return "nonterminal", artifact


def writable_path(
    *,
    output_dir: Path,
    job: dict[str, Any],
) -> Path:
    return planner.job_artifact_path(
        output_dir,
        job,
    )


def inherited_path(
    *,
    inherited_root: Path,
    job: dict[str, Any],
) -> Path:
    return planner.job_artifact_path(
        inherited_root,
        job,
    )


def locate_job(
    *,
    output_dir: Path,
    inherited_root: Path,
    data_root: Path,
    schema: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any]:
    writable = writable_path(
        output_dir=output_dir,
        job=job,
    )

    status, artifact = (
        terminal_artifact(
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
        }

    if status == "missing":
        return {
            "state": (
                "writable_missing"
            ),
            "path": str(writable),
            "artifact": artifact,
        }

    if status == "nonterminal":
        return {
            "state": "partial",
            "path": str(writable),
            "artifact": artifact,
        }

    writable_partials = (
        base.partial_attempts(
            output_dir=output_dir,
            cache_key=str(
                job["cache_key"]
            ),
        )
    )

    if writable_partials:
        return {
            "state": "partial",
            "path": str(writable),
            "artifact": None,
        }

    inherited = inherited_path(
        inherited_root=(
            inherited_root
        ),
        job=job,
    )

    inherited_status, inherited_artifact = (
        terminal_artifact(
            path=inherited,
            job=job,
            data_root=data_root,
            schema=schema,
        )
    )

    if inherited_status == "ok":
        return {
            "state": "inherited_ok",
            "path": str(inherited),
            "artifact": (
                inherited_artifact
            ),
        }

    if inherited_status == "missing":
        return {
            "state": (
                "inherited_missing"
            ),
            "path": str(inherited),
            "artifact": (
                inherited_artifact
            ),
        }

    if inherited_status == "nonterminal":
        raise RuntimeError(
            "nonterminal artifact found "
            "inside read-only inherited cache: "
            f"{inherited}"
        )

    inherited_partials = (
        base.partial_attempts(
            output_dir=(
                inherited_root
            ),
            cache_key=str(
                job["cache_key"]
            ),
        )
    )

    if inherited_partials:
        raise RuntimeError(
            "partial attempt state found "
            "inside read-only inherited cache"
        )

    return {
        "state": "pending",
        "path": "",
        "artifact": None,
    }


def state_counts(
    *,
    jobs: list[dict[str, Any]],
    output_dir: Path,
    inherited_root: Path,
    data_root: Path,
    schema: dict[str, Any],
) -> Counter:
    return Counter(
        locate_job(
            output_dir=output_dir,
            inherited_root=(
                inherited_root
            ),
            data_root=data_root,
            schema=schema,
            job=job,
        )["state"]
        for job in jobs
    )


def effective_artifact(
    *,
    job: dict[str, Any],
    output_dir: Path,
    inherited_root: Path,
    data_root: Path,
    schema: dict[str, Any],
) -> dict[str, Any] | None:
    located = locate_job(
        output_dir=output_dir,
        inherited_root=(
            inherited_root
        ),
        data_root=data_root,
        schema=schema,
        job=job,
    )

    if located["state"] in {
        "writable_ok",
        "inherited_ok",
    }:
        return located["artifact"]

    return None


def write_consensus(
    *,
    dose: dict[str, Any],
    jobs: list[dict[str, Any]],
    output_dir: Path,
    inherited_root: Path,
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

    statuses = Counter()

    for trial in dose["trials"]:
        identity = (
            str(trial["profile"]),
            str(trial["trial_name"]),
        )

        trial_jobs = sorted(
            by_trial[identity],
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

        entries = []

        sources = []

        for job in trial_jobs:
            located = locate_job(
                output_dir=output_dir,
                inherited_root=(
                    inherited_root
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
                    artifact.get(
                        "final_cache_entry"
                    )
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

        path = (
            output_dir
            / "consensus"
            / (
                str(
                    trial["profile"]
                )
                + "__"
                + str(
                    trial[
                        "trial_name"
                    ]
                )
                + ".json"
            )
        )

        base.atomic_write_json(
            path,
            {
                "dose_executor_version": (
                    DOSE_EXECUTOR_VERSION
                ),
                "profile": (
                    trial["profile"]
                ),
                "trial_name": (
                    trial[
                        "trial_name"
                    ]
                ),
                "condition": (
                    trial[
                        "condition"
                    ]
                ),
                "placement": (
                    trial[
                        "placement"
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


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--pilot-spec",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--dose-spec",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--inherited-root",
        type=Path,
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

    pilot_spec_path = (
        args.pilot_spec
        .expanduser()
        .resolve()
    )

    dose_spec_path = (
        args.dose_spec
        .expanduser()
        .resolve()
    )

    inherited_root = (
        args.inherited_root
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

    manifest = load_json(
        manifest_path
    )

    pilot_spec = load_json(
        pilot_spec_path
    )

    dose_spec = load_json(
        dose_spec_path
    )

    generated = selector.build_dose(
        manifest=manifest,
        pilot_spec=pilot_spec,
    )

    validate_frozen_dose(
        manifest_path=manifest_path,
        pilot_spec_path=(
            pilot_spec_path
        ),
        manifest=manifest,
        pilot_spec=pilot_spec,
        dose_spec=dose_spec,
        generated=generated,
    )

    jobs = dose_jobs(
        generated
    )

    schema = panel.load_schema()

    before = state_counts(
        jobs=jobs,
        output_dir=output_dir,
        inherited_root=(
            inherited_root
        ),
        data_root=data_root,
        schema=schema,
    )

    print(
        "FROZEN SEMANTIC DOSE "
        "EXECUTION STATE"
    )
    print("=" * 72)

    print_state(before)

    if before.get(
        "partial",
        0,
    ):
        raise RuntimeError(
            "partial writable jobs "
            "must be inspected before "
            "provider execution"
        )

    if args.dry_run:
        print()
        print(
            "frozen dose lock: PASS"
        )
        print(
            "network calls: 0"
        )
        print(
            "DOSE EXECUTION DRY RUN: PASS"
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
            inherited_root=(
                inherited_root
            ),
            data_root=data_root,
            schema=schema,
            job=job,
        )

        state = located[
            "state"
        ]

        print(
            f"[{index:03d}/200]",
            job["profile"],
            job["condition"],
            job["placement"],
            job["judge_family"],
            state,
        )

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
        inherited_root=(
            inherited_root
        ),
        data_root=data_root,
        schema=schema,
    )

    consensus_status = (
        write_consensus(
            dose=generated,
            jobs=jobs,
            output_dir=output_dir,
            inherited_root=(
                inherited_root
            ),
            data_root=data_root,
            schema=schema,
        )
    )

    summary = {
        "dose_executor_version": (
            DOSE_EXECUTOR_VERSION
        ),
        "mode": "dose_100",
        "trajectory_count": 100,
        "judge_job_count": 200,
        "inherited_root": str(
            inherited_root
        ),
        "new_jobs": new_jobs,
        "provider_attempts": (
            provider_attempts
        ),
        "before": dict(before),
        "after": dict(after),
        "consensus_field_status": (
            dict(
                consensus_status
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
    print(
        "FINAL DOSE STATE"
    )
    print("=" * 72)

    print_state(after)

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
        "consensus field status:",
        dict(
            consensus_status
        ),
    )
    print(
        "run record:",
        run_record,
    )


if __name__ == "__main__":
    main()
