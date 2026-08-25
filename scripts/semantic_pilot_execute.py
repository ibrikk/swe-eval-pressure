#!/usr/bin/env python3
"""Execute the frozen semantic judging pilot.

Properties:
- exact frozen pilot selection is verified before execution;
- sequential execution only;
- existing valid/missing terminal artifacts are not rerun;
- partial interrupted jobs require manual review;
- raw gateway responses are journaled before parsing;
- API keys are never written to disk;
- final job artifacts are written atomically;
- consensus never resolves judge disagreement via a third model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import semantic_bulk as bulk
import semantic_bulk_runner as planner
import semantic_orchestrator as orch
import semantic_panel as panel
from semantic_view import (
    SEMANTIC_VIEW_SCHEMA_VERSION,
    render_semantic_view,
    semantic_excerpt,
)


EXECUTOR_VERSION = "1.0"
MAX_CONTEXT_CHARS = 60000
MAX_TOKENS = 8192
TIMEOUT_SECONDS = 90


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(data, dict):
        raise ValueError(
            f"{path}: expected JSON object"
        )

    return data


def file_sha256(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def atomic_write_json(
    path: Path,
    value: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )

    try:
        with os.fdopen(
            fd,
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                value,
                handle,
                indent=2,
                ensure_ascii=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(
                handle.fileno()
            )

        os.replace(
            temp_name,
            path,
        )

    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass

        raise


def trial_signature(
    trial: dict[str, Any],
) -> tuple[str, ...]:
    return (
        str(trial["profile"]),
        str(trial["condition"]),
        str(trial["placement"]),
        str(trial["pressure_type"]),
        str(trial["trial_name"]),
        str(trial["trajectory_hash"]),
        str(trial["selection_hash"]),
    )


def validate_frozen_spec(
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    plan: dict[str, Any],
    spec: dict[str, Any],
) -> None:
    if (
        spec.get("selection_status")
        != "frozen_before_semantic_outcomes"
    ):
        raise ValueError(
            "pilot spec is not marked "
            "frozen before outcomes"
        )

    expected_manifest_hash = str(
        spec.get(
            "source_manifest_sha256",
            "",
        )
    )

    actual_manifest_hash = (
        file_sha256(
            manifest_path
        )
    )

    if (
        expected_manifest_hash
        != actual_manifest_hash
    ):
        raise ValueError(
            "pilot source manifest hash "
            "does not match frozen spec"
        )

    for field in (
        "semantic_schema_version",
        "rubric_version",
        "semantic_view_version",
    ):
        if (
            str(spec.get(field))
            != str(
                manifest.get(field)
            )
        ):
            raise ValueError(
                f"pilot {field} mismatch"
            )

    if (
        int(spec["trajectory_count"])
        != 10
        or int(
            spec["core_judge_jobs"]
        )
        != 20
    ):
        raise ValueError(
            "unexpected frozen pilot size"
        )

    frozen = [
        trial_signature(trial)
        for trial in spec["trials"]
    ]

    generated = [
        trial_signature(trial)
        for trial in plan["trials"]
    ]

    if frozen != generated:
        raise ValueError(
            "generated pilot does not "
            "exactly match frozen spec"
        )

    spec_judges = {
        (
            str(judge["family"]),
            str(judge["model"]),
        )
        for judge
        in spec["primary_judges"]
    }

    manifest_judges = {
        (
            str(job["judge_family"]),
            str(job["judge_model"]),
        )
        for job in manifest["jobs"]
    }

    if spec_judges != manifest_judges:
        raise ValueError(
            "core judge panel mismatch"
        )


def read_artifact(
    path: Path,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None

    try:
        data = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return None

    return (
        data
        if isinstance(data, dict)
        else None
    )


def artifact_is_terminal(
    path: Path,
    *,
    expected_cache_key: str,
) -> bool:
    data = read_artifact(path)

    if data is None:
        return False

    if (
        data.get("cache_key")
        != expected_cache_key
    ):
        return False

    status = data.get("status")

    if status == "ok":
        entry = data.get(
            "final_cache_entry"
        )

        return (
            isinstance(entry, dict)
            and entry.get("status")
            == "ok"
        )

    if status == "missing":
        return (
            data.get(
                "final_cache_entry"
            )
            is None
            and isinstance(
                data.get("attempts"),
                list,
            )
        )

    return False


def attempt_directory(
    output_dir: Path,
    cache_key: str,
) -> Path:
    return (
        output_dir
        / "attempts"
        / cache_key
    )


def partial_attempts(
    *,
    output_dir: Path,
    cache_key: str,
) -> list[Path]:
    directory = attempt_directory(
        output_dir,
        cache_key,
    )

    if not directory.is_dir():
        return []

    return sorted(
        directory.glob(
            "attempt-*.json"
        )
    )


def classify_job(
    *,
    output_dir: Path,
    job: dict[str, Any],
) -> str:
    artifact = planner.job_artifact_path(
        output_dir,
        job,
    )

    cache_key = str(
        job["cache_key"]
    )

    if planner.artifact_is_complete(
        artifact,
        expected_cache_key=cache_key,
    ):
        return "cached_ok"

    if artifact_is_terminal(
        artifact,
        expected_cache_key=cache_key,
    ):
        return "cached_missing"

    if partial_attempts(
        output_dir=output_dir,
        cache_key=cache_key,
    ):
        return "partial"

    return "pending"


def frozen_row(
    *,
    data_root: Path,
    job: dict[str, Any],
) -> dict[str, Any]:
    rows = bulk.load_profile_rows(
        data_root=data_root,
        profile=str(
            job["profile"]
        ),
    )

    matches = [
        row
        for row in rows
        if str(
            row.get(
                "trial_name",
                "",
            )
        )
        == str(
            job["trial_name"]
        )
    ]

    if len(matches) != 1:
        raise ValueError(
            f"{job['trial_name']}: "
            "expected exactly one frozen row"
        )

    row = matches[0]

    if not bool(
        row.get(
            "substantive_usable"
        )
    ):
        raise ValueError(
            f"{job['trial_name']}: "
            "frozen row is not usable"
        )

    placement = str(
        row.get("placement")
        or row.get("channel")
        or ""
    )

    checks = {
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
        checks.items()
    ):
        if (
            actual
            != str(job[field])
        ):
            raise ValueError(
                f"{job['trial_name']}: "
                f"{field} mismatch"
            )

    return row


def trajectory_object(
    job: dict[str, Any],
) -> dict[str, Any]:
    path = Path(
        str(
            job[
                "trajectory_path"
            ]
        )
    )

    if not path.is_file():
        raise FileNotFoundError(path)

    actual_hash = bulk.sha256_file(
        path
    )

    if (
        actual_hash
        != str(
            job[
                "trajectory_hash"
            ]
        )
    ):
        raise ValueError(
            f"{job['trial_name']}: "
            "trajectory hash mismatch"
        )

    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(data, dict):
        raise ValueError(
            "trajectory is not "
            "a JSON object"
        )

    return data


def prepare_job(
    *,
    job: dict[str, Any],
    row: dict[str, Any],
    schema: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    trajectory = (
        trajectory_object(job)
    )

    agent_blocks = (
        panel.agent_evidence_blocks(
            trajectory
        )
    )

    eval_text = str(
        row.get(
            "eval_cue_text",
            "",
        )
        or ""
    )

    pressure_type = str(
        row.get(
            "pressure_type",
            "",
        )
        or ""
    )

    semantic_context = (
        render_semantic_view(
            trajectory,
            evaluation_cue_text=(
                eval_text
            ),
            pressure_type=(
                pressure_type
            ),
        )
    )

    semantic_context = (
        semantic_excerpt(
            semantic_context,
            MAX_CONTEXT_CHARS,
        )
    )

    metadata = {
        "profile": str(
            job["profile"]
        ),
        "condition": str(
            job["condition"]
        ),
        "placement": str(
            job["placement"]
        ),
        "pressure_type": (
            pressure_type
        ),
        "eval_cue_id": str(
            row.get(
                "eval_cue_id",
                "",
            )
            or ""
        ),
        "eval_cue_text": (
            eval_text
        ),
    }

    body = panel.request_body(
        model=str(
            job["judge_model"]
        ),
        schema=schema,
        semantic_context=(
            semantic_context
        ),
        agent_blocks=(
            agent_blocks
        ),
        treatment_metadata=(
            metadata
        ),
        max_tokens=MAX_TOKENS,
    )

    return (
        agent_blocks,
        metadata,
        body,
    )


def parse_keys() -> list[str]:
    raw = (
        os.getenv(
            "LITE_LLM_KEYS"
        )
        or os.getenv(
            "LITE_LLM_KEY"
        )
        or ""
    )

    keys = [
        value.strip()
        for value in raw.split(",")
        if value.strip()
    ]

    if not keys:
        raise ValueError(
            "no LiteLLM keys loaded"
        )

    return keys


def primary_judge_index(
    *,
    schema: dict[str, Any],
    job: dict[str, Any],
) -> int:
    target = (
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

    for index, judge in enumerate(
        schema["primary_judges"]
    ):
        candidate = (
            str(judge["family"]),
            str(judge["model"]),
        )

        if candidate == target:
            return index

    raise ValueError(
        "job judge is not in "
        "the primary panel"
    )


def execute_job(
    *,
    job: dict[str, Any],
    row: dict[str, Any],
    schema: dict[str, Any],
    output_dir: Path,
    base_url: str,
    keys: list[str],
    invoke_raw: Callable[..., Any] = (
        panel.invoke_judge_raw
    ),
    timeout: int = TIMEOUT_SECONDS,
    retry_delays: tuple[
        float,
        ...,
    ] = (
        0.0,
        2.0,
        5.0,
    ),
) -> dict[str, Any]:
    agent_blocks, metadata, body = (
        prepare_job(
            job=job,
            row=row,
            schema=schema,
        )
    )

    judge_index = (
        primary_judge_index(
            schema=schema,
            job=job,
        )
    )

    call_counter = {"n": 0}
    journal_paths: list[str] = []

    def invoke_once():
        call_counter["n"] += 1

        attempt_number = (
            call_counter["n"]
        )

        key_slot = (
            judge_index
            + attempt_number
            - 1
        ) % len(keys)

        journal = (
            attempt_directory(
                output_dir,
                str(
                    job[
                        "cache_key"
                    ]
                ),
            )
            / (
                "attempt-"
                f"{attempt_number:02d}"
                ".json"
            )
        )

        try:
            raw, finish_reason = (
                invoke_raw(
                    base_url=base_url,
                    api_key=keys[
                        key_slot
                    ],
                    body=body,
                    timeout=timeout,
                )
            )

            atomic_write_json(
                journal,
                {
                    "executor_version": (
                        EXECUTOR_VERSION
                    ),
                    "cache_key": str(
                        job[
                            "cache_key"
                        ]
                    ),
                    "attempt": (
                        attempt_number
                    ),
                    "key_slot": (
                        key_slot
                    ),
                    "status": (
                        "gateway_response"
                    ),
                    "finish_reason": (
                        finish_reason
                    ),
                    "raw_response": raw,
                },
            )

            journal_paths.append(
                str(journal)
            )

            return {
                "raw_response": raw,
                "finish_reason": (
                    finish_reason
                ),
            }

        except Exception as exc:
            atomic_write_json(
                journal,
                {
                    "executor_version": (
                        EXECUTOR_VERSION
                    ),
                    "cache_key": str(
                        job[
                            "cache_key"
                        ]
                    ),
                    "attempt": (
                        attempt_number
                    ),
                    "key_slot": (
                        key_slot
                    ),
                    "status": (
                        "exception"
                    ),
                    "exception_type": (
                        type(exc).__name__
                    ),
                    "exception_message": (
                        str(exc)
                    ),
                },
            )

            journal_paths.append(
                str(journal)
            )

            raise

    judge = {
        "family": str(
            job["judge_family"]
        ),
        "model": str(
            job["judge_model"]
        ),
    }

    result = (
        orch.run_judge_with_retries(
            trial_name=str(
                job["trial_name"]
            ),
            trajectory_hash=str(
                job[
                    "trajectory_hash"
                ]
            ),
            judge=judge,
            schema=schema,
            semantic_view_version=(
                SEMANTIC_VIEW_SCHEMA_VERSION
            ),
            agent_blocks=(
                agent_blocks
            ),
            invoke_once=invoke_once,
            max_attempts=3,
            retry_delays=(
                retry_delays
            ),
        )
    )

    artifact = {
        "executor_version": (
            EXECUTOR_VERSION
        ),
        "cache_key": str(
            job["cache_key"]
        ),
        "profile": str(
            job["profile"]
        ),
        "condition": str(
            job["condition"]
        ),
        "placement": str(
            job["placement"]
        ),
        "pressure_type": str(
            job["pressure_type"]
        ),
        "judge_family": str(
            job["judge_family"]
        ),
        "judge_model": str(
            job["judge_model"]
        ),
        "request_body_sha256": (
            panel.canonical_json_hash(
                body
            )
        ),
        "treatment_metadata": (
            metadata
        ),
        "attempt_journals": (
            journal_paths
        ),
        **result,
    }

    artifact_path = (
        planner.job_artifact_path(
            output_dir,
            job,
        )
    )

    atomic_write_json(
        artifact_path,
        artifact,
    )

    return artifact


def state_counts(
    *,
    plan: dict[str, Any],
    output_dir: Path,
) -> Counter:
    return Counter(
        classify_job(
            output_dir=output_dir,
            job=job,
        )
        for job in plan["jobs"]
    )


def write_consensus(
    *,
    plan: dict[str, Any],
    output_dir: Path,
    schema: dict[str, Any],
) -> Counter:
    jobs_by_trial: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = {}

    for job in plan["jobs"]:
        key = (
            str(job["profile"]),
            str(job["trial_name"]),
        )

        jobs_by_trial.setdefault(
            key,
            [],
        ).append(job)

    status_counts: Counter = (
        Counter()
    )

    for trial in plan["trials"]:
        key = (
            str(trial["profile"]),
            str(
                trial["trial_name"]
            ),
        )

        jobs = jobs_by_trial[key]

        jobs.sort(
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

        entries = []

        for job in jobs:
            artifact_path = (
                planner.job_artifact_path(
                    output_dir,
                    job,
                )
            )

            artifact = (
                read_artifact(
                    artifact_path
                )
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
                judge_entries=(
                    entries
                ),
            )
        )

        for value in (
            consensus[
                "fields"
            ].values()
        ):
            status_counts[
                value["status"]
            ] += 1

        out = (
            output_dir
            / "consensus"
            / (
                str(
                    trial[
                        "profile"
                    ]
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

        atomic_write_json(
            out,
            {
                "executor_version": (
                    EXECUTOR_VERSION
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
                "consensus": (
                    consensus
                ),
            },
        )

    return status_counts


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
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--data-root",
        type=Path,
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
        args.pilot_spec
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

    spec = load_json(
        spec_path
    )

    plan = planner.build_pilot_plan(
        manifest=manifest,
        output_dir=output_dir,
    )

    validate_frozen_spec(
        manifest_path=manifest_path,
        manifest=manifest,
        plan=plan,
        spec=spec,
    )

    before = state_counts(
        plan=plan,
        output_dir=output_dir,
    )

    print_state(
        title=(
            "FROZEN SEMANTIC PILOT "
            "EXECUTION STATE"
        ),
        counts=before,
    )

    if before.get(
        "partial",
        0,
    ):
        raise RuntimeError(
            "partial attempt journals "
            "exist without a terminal "
            "artifact; inspect before "
            "any further provider calls"
        )

    if args.dry_run:
        print()
        print(
            "frozen pilot lock: PASS"
        )
        print(
            "network calls: 0"
        )
        print(
            "EXECUTION DRY RUN: PASS"
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

    keys = parse_keys()
    schema = panel.load_schema()

    data_root = (
        args.data_root
        .expanduser()
        .resolve()
        if args.data_root
        else Path(
            str(
                manifest[
                    "data_root"
                ]
            )
        )
    )

    new_jobs = 0
    provider_attempts = 0

    for index, job in enumerate(
        plan["jobs"],
        1,
    ):
        state = classify_job(
            output_dir=output_dir,
            job=job,
        )

        print(
            f"[{index:02d}/20]",
            job["profile"],
            job["condition"],
            job["placement"],
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

        row = frozen_row(
            data_root=data_root,
            job=job,
        )

        artifact = execute_job(
            job=job,
            row=row,
            schema=schema,
            output_dir=(
                output_dir
            ),
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
        plan=plan,
        output_dir=output_dir,
    )

    consensus_counts = (
        write_consensus(
            plan=plan,
            output_dir=output_dir,
            schema=schema,
        )
    )

    summary = {
        "executor_version": (
            EXECUTOR_VERSION
        ),
        "mode": "pilot_10",
        "new_jobs": new_jobs,
        "provider_attempts": (
            provider_attempts
        ),
        "before": dict(before),
        "after": dict(after),
        "consensus_field_status": (
            dict(
                consensus_counts
            )
        ),
    }

    atomic_write_json(
        output_dir
        / "run_summary.json",
        summary,
    )

    print()
    print_state(
        title="FINAL EXECUTION STATE",
        counts=after,
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
        "consensus field status:",
        dict(
            consensus_counts
        ),
    )
    print(
        "summary:",
        output_dir
        / "run_summary.json",
    )


if __name__ == "__main__":
    main()
