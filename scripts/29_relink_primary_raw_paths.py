#!/usr/bin/env python3
"""Relink stale PRIMARY COMPLETE result paths to moved raw artifacts.

This script:
- indexes raw Harbor result.json files by exact trial_name;
- validates candidate identity against stored task/result metadata;
- updates only analysis/current/source/primary/*/trials.json;
- preserves backups under analysis/current/audit/primary_pre_relink/;
- never modifies raw Harbor outputs or frozen historical sources;
- fails closed before modifying anything if a trial is unresolved or ambiguous.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    ROOT
    / "analysis"
    / "current"
    / "source"
    / "primary"
)

AUDIT = (
    ROOT
    / "analysis"
    / "current"
    / "audit"
)

BACKUP = (
    AUDIT
    / "primary_pre_relink"
)

PROFILES = (
    "claude",
    "fable",
    "codex",
    "llama",
)

SEARCH_ROOTS = (
    (
        Path.home()
        / "Documents"
        / "swe_atlas_eval_awareness_benchmark_v2_1_resource"
        / "results"
        / "full"
    ),
    (
        ROOT
        / "results"
        / "archive"
        / "historical-primary-repairs-20260819_22"
    ),
    (
        ROOT
        / "results"
        / "archive"
    ),
    (
        Path.home()
        / "Documents"
        / "swe-eval-pressure-llama-audit"
        / "results"
        / "full"
    ),
    (
        ROOT
        / "results"
        / "full"
    ),
)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def load_json(
    path: Path,
) -> Any:
    try:
        return json.loads(
            path.read_text(
                encoding="utf-8",
            )
        )
    except Exception:
        return None


def write_json(
    path: Path,
    value: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary.replace(path)


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields: list[str] = []
    seen: set[str] = set()

    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(
    path: Path,
) -> str:
    h = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def numeric(
    value: Any,
) -> float | None:
    if value in (
        None,
        "",
    ):
        return None

    try:
        return float(value)
    except Exception:
        return None


def same_numeric(
    left: Any,
    right: Any,
) -> bool:
    a = numeric(left)
    b = numeric(right)

    if (
        a is None
        or b is None
    ):
        return True

    return math.isclose(
        a,
        b,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def truthy(
    value: Any,
) -> bool:
    if value is True:
        return True

    return (
        str(value)
        .strip()
        .lower()
        in {
            "1",
            "true",
            "yes",
        }
    )


def substantive(
    row: dict[str, Any],
) -> bool:
    return (
        row.get(
            "substantive_usable"
        )
        is True
        or truthy(
            row.get(
                "substantive_usable"
            )
        )
    )


def resolve_path(
    value: Any,
) -> Path | None:
    if value in (
        None,
        "",
    ):
        return None

    path = Path(
        str(value)
    )

    if not path.is_absolute():
        path = ROOT / path

    return path


# ---------------------------------------------------------------------------
# Raw-result identity
# ---------------------------------------------------------------------------


def raw_trial_name(
    result: dict[str, Any],
    result_path: Path,
) -> str:
    value = result.get(
        "trial_name"
    )

    return (
        str(value)
        if value
        else result_path.parent.name
    )


def raw_task_name(
    result: dict[str, Any],
) -> str:
    value = result.get(
        "task_name"
    )

    if value:
        return str(value)

    try:
        return str(
            result[
                "config"
            ][
                "task"
            ][
                "path"
            ]
        ).rstrip("/").split("/")[-1]

    except Exception:
        return ""


def raw_agent_version(
    result: dict[str, Any],
) -> str:
    info = (
        result.get(
            "agent_info"
        )
        or {}
    )

    if isinstance(
        info,
        dict,
    ):
        value = info.get(
            "version"
        )

        if value not in (
            None,
            "",
        ):
            return str(value)

    return ""


def raw_rewards(
    result: dict[str, Any],
) -> dict[str, Any]:
    verifier = (
        result.get(
            "verifier_result"
        )
        or {}
    )

    if not isinstance(
        verifier,
        dict,
    ):
        return {}

    rewards = (
        verifier.get(
            "rewards"
        )
        or {}
    )

    return (
        rewards
        if isinstance(
            rewards,
            dict,
        )
        else {}
    )


def run_root_for(
    result_path: Path,
) -> Path | None:
    """Recover a usable raw-result container.

    Preferred:
        nearest ancestor carrying normal
        SWE-EvalPressure/Harbor provenance.

    Legacy archive fallback:
        top-level child beneath a configured
        SEARCH_ROOT. Some archived repair jobs
        preserved all trial artifacts but lost
        their run-level metadata files.
    """

    resolved_result = (
        result_path.resolve()
    )

    current = (
        resolved_result.parent
    )

    # Normal modern/provenance-preserving layout.
    for directory in (
        current,
        *current.parents,
    ):
        if any(
            (
                directory / name
            ).is_file()
            for name in (
                "run_metadata.json",
                "study_manifest.json",
                "dataset_manifest.json",
            )
        ):
            return directory

    # Backward-compatible archived-run layout.
    #
    # Example:
    # SEARCH_ROOT/
    #   swe-eval-pressure-full-fable-single-repair-.../
    #     swe-eval-pressure-full-fable-single-repair-.../
    #       ea-.../
    #         result.json
    #
    # The outer first-level directory is the
    # stable result container even when the
    # run metadata files were not retained.
    for search_root in SEARCH_ROOTS:
        if not search_root.exists():
            continue

        root = search_root.resolve()

        try:
            relative = (
                resolved_result.relative_to(
                    root
                )
            )
        except ValueError:
            continue

        if not relative.parts:
            continue

        candidate = (
            root / relative.parts[0]
        )

        if candidate.is_dir():
            return candidate

    return None


def candidate_mismatches(
    stored: dict[str, Any],
    result: dict[str, Any],
    path: Path,
) -> list[str]:
    mismatches = []

    expected_trial = str(
        stored.get(
            "trial_name"
        )
        or ""
    )

    observed_trial = raw_trial_name(
        result,
        path,
    )

    if (
        expected_trial
        and observed_trial
        != expected_trial
    ):
        mismatches.append(
            "trial_name"
        )

    expected_task = str(
        stored.get(
            "task_name"
        )
        or ""
    )

    observed_task = raw_task_name(
        result
    )

    if (
        expected_task
        and observed_task
        and expected_task
        != observed_task
    ):
        mismatches.append(
            "task_name"
        )

    expected_version = str(
        stored.get(
            "agent_version"
        )
        or ""
    )

    observed_version = (
        raw_agent_version(
            result
        )
    )

    if (
        expected_version
        and observed_version
        and expected_version
        != observed_version
    ):
        mismatches.append(
            "agent_version"
        )

    rewards = raw_rewards(
        result
    )

    if not same_numeric(
        stored.get(
            "overall_pass"
        ),
        rewards.get(
            "overall_pass",
            rewards.get(
                "reward"
            ),
        ),
    ):
        mismatches.append(
            "overall_pass"
        )

    if not same_numeric(
        stored.get(
            "tests_reward"
        ),
        rewards.get(
            "tests_reward"
        ),
    ):
        mismatches.append(
            "tests_reward"
        )

    if not same_numeric(
        stored.get(
            "rubrics_agg_score"
        ),
        rewards.get(
            "rubrics_agg_score"
        ),
    ):
        mismatches.append(
            "rubrics_agg_score"
        )

    return mismatches


# ---------------------------------------------------------------------------
# Build raw index
# ---------------------------------------------------------------------------


def build_index() -> tuple[
    dict[str, list[tuple[Path, dict[str, Any]]]],
    list[dict[str, Any]],
]:
    index: dict[
        str,
        list[
            tuple[
                Path,
                dict[str, Any],
            ]
        ],
    ] = defaultdict(list)

    roots_report = []
    seen_paths: set[Path] = set()

    for root in SEARCH_ROOTS:
        count = 0

        if not root.exists():
            roots_report.append(
                {
                    "search_root": str(
                        root
                    ),
                    "exists": 0,
                    "result_files": 0,
                }
            )
            continue

        for path in root.rglob(
            "result.json"
        ):
            resolved = path.resolve()

            if resolved in seen_paths:
                continue

            seen_paths.add(
                resolved
            )

            result = load_json(
                path
            )

            if not isinstance(
                result,
                dict,
            ):
                continue

            trial = raw_trial_name(
                result,
                path,
            )

            if not trial:
                continue

            index[trial].append(
                (
                    path,
                    result,
                )
            )

            count += 1

        roots_report.append(
            {
                "search_root": str(
                    root
                ),
                "exists": 1,
                "result_files": count,
            }
        )

    return index, roots_report


# ---------------------------------------------------------------------------
# Recovery planning and application
# ---------------------------------------------------------------------------


def main() -> None:
    AUDIT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 88)
    print("PRIMARY COMPLETE — RAW PATH RELINK")
    print("=" * 88)

    index, roots_report = (
        build_index()
    )

    for row in roots_report:
        print(
            row["search_root"],
            "exists=",
            row["exists"],
            "results=",
            row["result_files"],
        )

    mapping_rows = []
    planned_updates: dict[
        str,
        list[
            tuple[
                int,
                Path,
                Path,
                str,
            ]
        ],
    ] = defaultdict(list)

    profile_data: dict[
        str,
        list[
            dict[str, Any]
        ],
    ] = {}

    unresolved = 0

    print()
    print("RECOVERY PLAN")
    print("-" * 88)

    for profile in PROFILES:
        trials_path = (
            SOURCE
            / profile
            / "trials.json"
        )

        rows = load_json(
            trials_path
        )

        if not isinstance(
            rows,
            list,
        ):
            raise SystemExit(
                f"{trials_path}: "
                "expected JSON list"
            )

        profile_data[
            profile
        ] = rows

        counts = Counter()

        for index_in_file, row in enumerate(
            rows
        ):
            if (
                not isinstance(
                    row,
                    dict,
                )
                or not substantive(
                    row
                )
            ):
                continue

            old_path = resolve_path(
                row.get(
                    "result_path"
                )
            )

            trial = str(
                row.get(
                    "trial_name"
                )
                or ""
            )

            if (
                old_path
                is not None
                and old_path.is_file()
            ):
                counts[
                    "already_available"
                ] += 1

                mapping_rows.append(
                    {
                        "profile": profile,
                        "trial_name": trial,
                        "status": (
                            "already_available"
                        ),
                        "old_result_path": str(
                            old_path
                        ),
                        "new_result_path": str(
                            old_path
                        ),
                        "candidate_count": 1,
                        "candidate_paths_json": (
                            json.dumps(
                                [
                                    str(
                                        old_path
                                    )
                                ]
                            )
                        ),
                        "mismatches_json": "[]",
                    }
                )

                continue

            raw_candidates = (
                index.get(
                    trial,
                    [],
                )
            )

            valid = []
            candidate_diagnostics = []

            for (
                candidate_path,
                candidate_result,
            ) in raw_candidates:
                mismatches = (
                    candidate_mismatches(
                        row,
                        candidate_result,
                        candidate_path,
                    )
                )

                candidate_diagnostics.append(
                    {
                        "path": str(
                            candidate_path
                        ),
                        "mismatches": (
                            mismatches
                        ),
                    }
                )

                if not mismatches:
                    valid.append(
                        (
                            candidate_path,
                            candidate_result,
                        )
                    )

            status = ""
            selected_path: Path | None = None

            if len(valid) == 1:
                status = "unique_recovery"
                selected_path = valid[0][0]

            elif len(valid) > 1:
                digests = {
                    sha256_file(path)
                    for path, _
                    in valid
                }

                if len(digests) == 1:
                    status = (
                        "identical_duplicate_recovery"
                    )

                    selected_path = sorted(
                        path
                        for path, _
                        in valid
                    )[0]

                else:
                    status = "ambiguous"
                    unresolved += 1

            elif raw_candidates:
                status = (
                    "candidates_failed_identity_check"
                )
                unresolved += 1

            else:
                status = "not_found"
                unresolved += 1

            if selected_path is not None:
                run_root = run_root_for(
                    selected_path
                )

                if run_root is None:
                    status = (
                        "run_root_not_found"
                    )
                    unresolved += 1

                else:
                    planned_updates[
                        profile
                    ].append(
                        (
                            index_in_file,
                            selected_path,
                            run_root,
                            status,
                        )
                    )

                    counts[
                        status
                    ] += 1

            else:
                counts[
                    status
                ] += 1

            mapping_rows.append(
                {
                    "profile": profile,
                    "trial_name": trial,
                    "status": status,
                    "old_result_path": (
                        str(old_path)
                        if old_path
                        is not None
                        else ""
                    ),
                    "new_result_path": (
                        str(
                            selected_path
                        )
                        if selected_path
                        is not None
                        else ""
                    ),
                    "candidate_count": len(
                        raw_candidates
                    ),
                    "valid_candidate_count": len(
                        valid
                    ),
                    "candidate_paths_json": (
                        json.dumps(
                            [
                                str(path)
                                for path, _
                                in raw_candidates
                            ],
                            ensure_ascii=False,
                        )
                    ),
                    "candidate_diagnostics_json": (
                        json.dumps(
                            candidate_diagnostics,
                            ensure_ascii=False,
                        )
                    ),
                }
            )

        print()
        print(profile.upper())

        for key, value in sorted(
            counts.items()
        ):
            print(
                f"  {key:36s}",
                value,
            )

    mapping_path = (
        AUDIT
        / "primary_raw_path_recovery.csv"
    )

    write_csv(
        mapping_path,
        mapping_rows,
    )

    if unresolved:
        print()
        print(
            "ABORT: unresolved or ambiguous "
            f"trials = {unresolved}"
        )
        print(
            "No PRIMARY COMPLETE files "
            "were modified."
        )
        print(
            "Inspect:",
            mapping_path,
        )

        raise SystemExit(1)

    # ------------------------------------------------------------------
    # Apply only after the complete plan has passed
    # ------------------------------------------------------------------

    BACKUP.mkdir(
        parents=True,
        exist_ok=True,
    )

    applied = 0

    for profile in PROFILES:
        trials_path = (
            SOURCE
            / profile
            / "trials.json"
        )

        rows = profile_data[
            profile
        ]

        backup_path = (
            BACKUP
            / f"{profile}.trials.json"
        )

        if not backup_path.exists():
            write_json(
                backup_path,
                rows,
            )

        for (
            row_index,
            result_path,
            run_root,
            recovery_status,
        ) in planned_updates[
            profile
        ]:
            row = rows[
                row_index
            ]

            row[
                "result_path"
            ] = str(
                result_path
            )

            row[
                "run_root"
            ] = str(
                run_root
            )

            row[
                "raw_path_recovery_status"
            ] = recovery_status

            applied += 1

        write_json(
            trials_path,
            rows,
        )

    # ------------------------------------------------------------------
    # Final exact availability check
    # ------------------------------------------------------------------

    availability = {}

    for profile in PROFILES:
        path = (
            SOURCE
            / profile
            / "trials.json"
        )

        rows = load_json(path)

        usable = [
            row
            for row in rows
            if (
                isinstance(
                    row,
                    dict,
                )
                and substantive(
                    row
                )
            )
        ]

        available = sum(
            (
                candidate
                := resolve_path(
                    row.get(
                        "result_path"
                    )
                )
            )
            is not None
            and candidate.is_file()
            for row in usable
        )

        availability[
            profile
        ] = {
            "substantive": len(
                usable
            ),
            "raw_available": (
                available
            ),
            "complete": (
                available
                == len(usable)
            ),
        }

        if available != len(
            usable
        ):
            raise RuntimeError(
                f"{profile}: final raw "
                "availability mismatch "
                f"{available}/{len(usable)}"
            )

    manifest = {
        "relink_version": (
            "primary-raw-relink-1.0"
        ),
        "search_roots": [
            str(root)
            for root in SEARCH_ROOTS
        ],
        "mapping_csv": str(
            mapping_path
        ),
        "updated_rows": applied,
        "availability": (
            availability
        ),
        "raw_artifacts_modified": False,
        "frozen_sources_modified": False,
        "current_primary_source_modified": True,
    }

    write_json(
        AUDIT
        / "primary_raw_path_recovery_manifest.json",
        manifest,
    )

    print()
    print("=" * 88)
    print("PRIMARY RAW PATH RELINK: PASS")
    print("=" * 88)
    print(
        "updated rows:",
        applied,
    )

    for profile, row in (
        availability.items()
    ):
        print(
            f"{profile:7s}",
            f"{row['raw_available']}/"
            f"{row['substantive']}",
            "raw available",
        )

    print()
    print("mapping:", mapping_path)
    print(
        "backup:",
        BACKUP,
    )
    print(
        "raw artifacts modified: 0"
    )
    print(
        "frozen sources modified: 0"
    )


if __name__ == "__main__":
    main()
