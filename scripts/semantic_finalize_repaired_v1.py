#!/usr/bin/env python3
"""Materialize the final repaired historical semantic dataset.

No network calls.

Final precedence:
1. Frozen infra-recovery artifact for every recovery-eligible key,
   EVEN if that recovery artifact is terminal missing.
2. Original repaired full-production artifact.
3. Repaired-Llama pilot artifact.
4. Legacy dose artifact.
5. Legacy pilot artifact.

This prevents outcome-dependent fallback after recovery.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import semantic_dose_execute as dose
import semantic_full_execute as full
import semantic_full_parallel_execute as repaired
import semantic_orchestrator as orch
import semantic_panel as panel
import semantic_pilot_execute as base


FINALIZER_VERSION = "1.0"


def load_json(path: Path) -> Any:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(
        payload
    ).hexdigest()


def artifact_path(
    root: Path,
    job: dict[str, Any],
) -> Path:
    return dose.inherited_path(
        inherited_root=root,
        job=job,
    )


def validated_status(
    *,
    root: Path,
    job: dict[str, Any],
    row: dict[str, Any],
    schema: dict[str, Any],
) -> tuple[
    str,
    dict[str, Any] | None,
    Path,
]:
    path = artifact_path(
        root,
        job,
    )

    status, artifact = (
        repaired.terminal_artifact(
            path=path,
            job=job,
            row=row,
            schema=schema,
        )
    )

    return (
        status,
        artifact,
        path,
    )


def final_reason(
    artifact: dict[str, Any],
) -> str:
    attempts = (
        artifact.get("attempts")
        or []
    )

    if not attempts:
        return "no_attempt_record"

    last = attempts[-1]

    validation = str(
        last.get(
            "validation_error",
            "",
        )
        or ""
    )

    etype = str(
        last.get(
            "exception_type",
            "",
        )
        or ""
    )

    message = str(
        last.get(
            "exception_message",
            "",
        )
        or ""
    )

    if validation:
        return (
            "validation:"
            + validation
        )

    if etype:
        return (
            "exception:"
            + etype
            + ":"
            + message
        )

    return (
        "status:"
        + str(
            last.get(
                "status",
                "",
            )
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--recovery-policy",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--recovery-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--full-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--llama-pilot-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--old-dose-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--old-pilot-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
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
        args.recovery_policy
        .expanduser()
        .resolve()
    )

    recovery_root = (
        args.recovery_root
        .expanduser()
        .resolve()
    )

    full_root = (
        args.full_root
        .expanduser()
        .resolve()
    )

    llama_pilot_root = (
        args.llama_pilot_root
        .expanduser()
        .resolve()
    )

    old_dose_root = (
        args.old_dose_root
        .expanduser()
        .resolve()
    )

    old_pilot_root = (
        args.old_pilot_root
        .expanduser()
        .resolve()
    )

    output_root = (
        args.output_root
        .expanduser()
        .resolve()
    )

    manifest = full.load_json(
        manifest_path
    )

    full.validate_manifest(
        manifest
    )

    policy = load_json(
        policy_path
    )

    if (
        policy.get(
            "selection_status"
        )
        != (
            "frozen_before_"
            "recovery_outcomes"
        )
    ):
        raise ValueError(
            "recovery policy is "
            "not frozen"
        )

    if (
        sha256_file(
            manifest_path
        )
        != policy[
            "source_manifest_sha256"
        ]
    ):
        raise ValueError(
            "manifest/policy "
            "hash mismatch"
        )

    if int(
        policy[
            "eligible_job_count"
        ]
    ) != 867:
        raise ValueError(
            "expected 867 frozen "
            "recovery keys"
        )

    recovery_keys = {
        str(x["cache_key"])
        for x
        in policy[
            "eligible_jobs"
        ]
    }

    if len(
        recovery_keys
    ) != 867:
        raise ValueError(
            "duplicate recovery keys"
        )

    jobs = full.full_jobs(
        manifest
    )

    if len(jobs) != 5552:
        raise ValueError(
            "expected 5552 final jobs"
        )

    rows = (
        repaired.load_frozen_rows(
            manifest
        )
    )

    schema = panel.load_schema()

    roots = {
        "recovery": recovery_root,
        "full": full_root,
        "llama_pilot": (
            llama_pilot_root
        ),
        "old_dose": old_dose_root,
        "old_pilot": (
            old_pilot_root
        ),
    }

    selections = []
    source_counts = Counter()
    status_counts = Counter()
    profile_status = Counter()
    judge_status = Counter()
    missing_rows = []

    for job in jobs:
        key = str(
            job["cache_key"]
        )

        row = repaired.frozen_row(
            rows_by_identity=rows,
            job=job,
        )

        if key in recovery_keys:
            candidate_names = [
                "recovery"
            ]

        else:
            candidate_names = [
                "full",
                "llama_pilot",
                "old_dose",
                "old_pilot",
            ]

        hits = []

        for name in (
            candidate_names
        ):
            (
                status,
                artifact,
                path,
            ) = validated_status(
                root=roots[name],
                job=job,
                row=row,
                schema=schema,
            )

            if status in {
                "ok",
                "missing",
            }:
                hits.append(
                    (
                        name,
                        status,
                        artifact,
                        path,
                    )
                )

            elif (
                status
                == "nonterminal"
            ):
                raise RuntimeError(
                    "nonterminal source "
                    f"artifact: {path}"
                )

        if len(hits) != 1:
            raise ValueError(
                f"{key}: expected "
                "exactly one selected "
                "terminal source; "
                f"found {len(hits)}"
            )

        (
            source_name,
            status,
            artifact,
            source_path,
        ) = hits[0]

        assert artifact is not None

        source_counts[
            source_name
        ] += 1

        status_counts[
            status
        ] += 1

        profile_status[
            (
                str(
                    job["profile"]
                ),
                status,
            )
        ] += 1

        judge_status[
            (
                str(
                    job[
                        "judge_family"
                    ]
                ),
                status,
            )
        ] += 1

        record = {
            "cache_key": key,
            "profile": (
                job["profile"]
            ),
            "trial_name": (
                job["trial_name"]
            ),
            "condition": (
                job["condition"]
            ),
            "placement": (
                job["placement"]
            ),
            "pressure_type": (
                job["pressure_type"]
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
            "trajectory_hash": (
                job[
                    "trajectory_hash"
                ]
            ),
            "selected_source": (
                source_name
            ),
            "selected_status": (
                status
            ),
            "source_artifact": str(
                source_path
            ),
            "source_artifact_sha256": (
                sha256_file(
                    source_path
                )
            ),
            "recovery_eligible": (
                key
                in recovery_keys
            ),
        }

        selections.append(
            record
        )

        if status == "missing":
            missing_rows.append({
                **record,
                "final_reason": (
                    final_reason(
                        artifact
                    )
                ),
                "attempt_count": (
                    artifact.get(
                        "attempt_count"
                    )
                ),
            })

    # These source counts are implied by the
    # already-audited cache topology.
    expected_sources = {
        "recovery": 867,
        "full": 4515,
        "llama_pilot": 20,
        "old_dose": 134,
        "old_pilot": 16,
    }

    if (
        dict(source_counts)
        != expected_sources
    ):
        raise ValueError(
            "unexpected final source "
            f"counts: {dict(source_counts)}"
        )

    if sum(
        status_counts.values()
    ) != 5552:
        raise ValueError(
            "final status count "
            "does not sum to 5552"
        )

    print(
        "FINAL HISTORICAL "
        "SEMANTIC MATERIALIZATION"
    )
    print("=" * 80)

    print(
        "source counts:",
        dict(source_counts),
    )

    print(
        "status counts:",
        dict(status_counts),
    )

    print(
        "missing jobs:",
        len(missing_rows),
    )

    print()
    print(
        "BY PROFILE"
    )

    for profile in (
        "claude",
        "fable",
        "codex",
        "llama",
    ):
        print(
            f"  {profile:7s}",
            "ok=",
            profile_status[
                (
                    profile,
                    "ok",
                )
            ],
            "missing=",
            profile_status[
                (
                    profile,
                    "missing",
                )
            ],
        )

    print()
    print("BY JUDGE")

    for judge in (
        "deepseek",
        "gemini",
    ):
        print(
            f"  {judge:8s}",
            "ok=",
            judge_status[
                (
                    judge,
                    "ok",
                )
            ],
            "missing=",
            judge_status[
                (
                    judge,
                    "missing",
                )
            ],
        )

    print()
    print("FINAL MISSING JOBS")
    print("-" * 80)

    for x in missing_rows:
        reason = str(
            x["final_reason"]
        ).replace(
            "\n",
            " ",
        )

        if len(reason) > 180:
            reason = (
                reason[:177]
                + "..."
            )

        print(
            x["profile"],
            x["condition"],
            x["placement"],
            x["judge_family"],
            "source=",
            x["selected_source"],
            "attempts=",
            x["attempt_count"],
            "reason=",
            reason,
        )

    if args.dry_run:
        print()
        print(
            "network calls: 0"
        )
        print(
            "FINALIZATION DRY RUN: PASS"
        )
        return

    if output_root.exists():
        raise RuntimeError(
            "final output root "
            "already exists: "
            f"{output_root}"
        )

    tmp_root = Path(
        str(output_root)
        + ".tmp"
    )

    if tmp_root.exists():
        shutil.rmtree(
            tmp_root
        )

    (
        tmp_root
        / "jobs"
    ).mkdir(
        parents=True,
        exist_ok=False,
    )

    (
        tmp_root
        / "consensus"
    ).mkdir(
        parents=True,
        exist_ok=False,
    )

    # Copy selected terminal job artifacts
    # byte-for-byte.
    for record in selections:
        source = Path(
            record[
                "source_artifact"
            ]
        )

        destination = (
            tmp_root
            / "jobs"
            / (
                record[
                    "cache_key"
                ]
                + ".json"
            )
        )

        shutil.copy2(
            source,
            destination,
        )

        copied_hash = (
            sha256_file(
                destination
            )
        )

        if (
            copied_hash
            != record[
                "source_artifact_sha256"
            ]
        ):
            raise RuntimeError(
                "copied artifact "
                "hash mismatch"
            )

        record[
            "final_artifact_sha256"
        ] = copied_hash

    # Regenerate final two-judge consensus.
    by_trial = defaultdict(
        list
    )

    jobs_by_key = {
        str(j["cache_key"]): j
        for j in jobs
    }

    for record in selections:
        identity = (
            str(
                record["profile"]
            ),
            str(
                record[
                    "trial_name"
                ]
            ),
        )

        by_trial[
            identity
        ].append(
            record
        )

    if len(by_trial) != 2776:
        raise ValueError(
            "expected 2776 "
            "trajectory groups"
        )

    field_status = Counter()
    pair_status = Counter()

    for (
        profile,
        trial_name,
    ), records in sorted(
        by_trial.items()
    ):
        if len(records) != 2:
            raise ValueError(
                "trajectory does not "
                "have exactly two "
                f"judge jobs: "
                f"{profile}/"
                f"{trial_name}"
            )

        records = sorted(
            records,
            key=lambda x: (
                str(
                    x[
                        "judge_family"
                    ]
                )
            ),
        )

        entries = []
        source_info = []

        ok_count = 0

        for record in records:
            path = (
                tmp_root
                / "jobs"
                / (
                    record[
                        "cache_key"
                    ]
                    + ".json"
                )
            )

            artifact = load_json(
                path
            )

            if (
                artifact.get(
                    "status"
                )
                == "ok"
            ):
                entries.append(
                    artifact[
                        "final_cache_entry"
                    ]
                )
                ok_count += 1
            else:
                entries.append(
                    None
                )

            source_info.append({
                "cache_key": (
                    record[
                        "cache_key"
                    ]
                ),
                "judge_family": (
                    record[
                        "judge_family"
                    ]
                ),
                "judge_model": (
                    record[
                        "judge_model"
                    ]
                ),
                "selected_source": (
                    record[
                        "selected_source"
                    ]
                ),
                "selected_status": (
                    record[
                        "selected_status"
                    ]
                ),
            })

        if ok_count == 2:
            pair_status[
                "both_ok"
            ] += 1
        elif ok_count == 1:
            pair_status[
                "one_missing"
            ] += 1
        elif ok_count == 0:
            pair_status[
                "both_missing"
            ] += 1
        else:
            raise AssertionError

        consensus = (
            orch.core_panel_consensus(
                schema=schema,
                judge_entries=(
                    entries
                ),
            )
        )

        for field in (
            consensus[
                "fields"
            ].values()
        ):
            field_status[
                field["status"]
            ] += 1

        first_job = jobs_by_key[
            records[0][
                "cache_key"
            ]
        ]

        base.atomic_write_json(
            (
                tmp_root
                / "consensus"
                / (
                    profile
                    + "__"
                    + trial_name
                    + ".json"
                )
            ),
            {
                "finalizer_version": (
                    FINALIZER_VERSION
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
                    source_info
                ),
                "consensus": (
                    consensus
                ),
            },
        )

    selection_ledger = {
        "finalizer_version": (
            FINALIZER_VERSION
        ),
        "selection_rule": {
            "recovery_eligible": (
                "recovery artifact "
                "always selected, "
                "including terminal "
                "missing recovery "
                "artifacts"
            ),
            "otherwise_precedence": [
                "full",
                "llama_pilot",
                "old_dose",
                "old_pilot",
            ],
            "no_outcome_dependent_fallback": (
                True
            ),
        },
        "source_counts": dict(
            source_counts
        ),
        "status_counts": dict(
            status_counts
        ),
        "jobs": selections,
    }

    base.atomic_write_json(
        (
            tmp_root
            / "selection_ledger.json"
        ),
        selection_ledger,
    )

    base.atomic_write_json(
        (
            tmp_root
            / "missing_jobs.json"
        ),
        {
            "missing_job_count": (
                len(missing_rows)
            ),
            "jobs": (
                missing_rows
            ),
        },
    )

    summary = {
        "finalizer_version": (
            FINALIZER_VERSION
        ),
        "planned_trajectories": (
            2800
        ),
        "semantic_eligible_trajectories": (
            2776
        ),
        "censored_or_error": (
            24
        ),
        "judge_jobs": (
            5552
        ),
        "source_counts": dict(
            source_counts
        ),
        "job_status": dict(
            status_counts
        ),
        "missing_job_count": (
            len(missing_rows)
        ),
        "pair_status": dict(
            pair_status
        ),
        "consensus_field_status": dict(
            field_status
        ),
    }

    base.atomic_write_json(
        (
            tmp_root
            / "final_summary.json"
        ),
        summary,
    )

    # Compact deterministic tree digests.
    job_hashes = [
        (
            p.name,
            sha256_file(p),
        )
        for p in sorted(
            (
                tmp_root
                / "jobs"
            ).glob("*.json")
        )
    ]

    consensus_hashes = [
        (
            p.name,
            sha256_file(p),
        )
        for p in sorted(
            (
                tmp_root
                / "consensus"
            ).glob("*.json")
        )
    ]

    freeze = {
        "freeze_version": "1.0",
        "finalizer_version": (
            FINALIZER_VERSION
        ),
        "source_manifest": str(
            manifest_path
        ),
        "source_manifest_sha256": (
            sha256_file(
                manifest_path
            )
        ),
        "recovery_policy": str(
            policy_path
        ),
        "recovery_policy_sha256": (
            sha256_file(
                policy_path
            )
        ),
        "selection_ledger_sha256": (
            sha256_file(
                tmp_root
                / "selection_ledger.json"
            )
        ),
        "missing_jobs_sha256": (
            sha256_file(
                tmp_root
                / "missing_jobs.json"
            )
        ),
        "final_summary_sha256": (
            sha256_file(
                tmp_root
                / "final_summary.json"
            )
        ),
        "jobs_tree_sha256": (
            canonical_hash(
                job_hashes
            )
        ),
        "consensus_tree_sha256": (
            canonical_hash(
                consensus_hashes
            )
        ),
        "job_file_count": (
            len(job_hashes)
        ),
        "consensus_file_count": (
            len(
                consensus_hashes
            )
        ),
        "job_status": dict(
            status_counts
        ),
        "pair_status": dict(
            pair_status
        ),
        "consensus_field_status": dict(
            field_status
        ),
    }

    base.atomic_write_json(
        (
            tmp_root
            / "freeze_manifest.json"
        ),
        freeze,
    )

    tmp_root.rename(
        output_root
    )

    final_freeze = (
        output_root
        / "freeze_manifest.json"
    )

    print()
    print("FINALIZED")
    print("=" * 80)

    print(
        "output:",
        output_root,
    )

    print(
        "jobs:",
        5552,
    )

    print(
        "consensus:",
        2776,
    )

    print(
        "job status:",
        dict(status_counts),
    )

    print(
        "pair status:",
        dict(pair_status),
    )

    print(
        "consensus fields:",
        dict(field_status),
    )

    print(
        "freeze manifest sha256:",
        sha256_file(
            final_freeze
        ),
    )

    print()
    print("network calls: 0")
    print(
        "FINAL MATERIALIZATION: PASS"
    )


if __name__ == "__main__":
    main()
