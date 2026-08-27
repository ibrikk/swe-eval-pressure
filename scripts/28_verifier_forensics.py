#!/usr/bin/env python3
"""Read-only verifier forensics for the canonical current cohorts.

No network calls.
No API calls.
No mutation of source data.
No reclassification of model/infrastructure outcomes.

This audit determines:
- how much raw verifier evidence remains locally available;
- the effective patch supplied to the rubric evaluator;
- empty-patch/rubric anomalies;
- patch-application failures;
- mutation/comparison artifact availability;
- stored-vs-raw reward consistency;
- whether any stored strict successes are forensically at risk.

The output is diagnostic. It does not itself certify or invalidate
overall_pass, tests_reward, must_have_pass, or rubrics_agg_score.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    ROOT
    / "analysis"
    / "current"
    / "source"
)

OUTPUT = (
    ROOT
    / "analysis"
    / "current"
    / "audit"
)

STUDIES = (
    "primary",
    "resource",
    "replication",
)

PROFILES = (
    "claude",
    "fable",
    "codex",
    "llama",
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


def read_text(
    path: Path,
) -> str:
    if not path.is_file():
        return ""

    try:
        return path.read_text(
            encoding="utf-8",
            errors="replace",
        )
    except Exception:
        return ""


def write_json(
    path: Path,
    value: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        path.write_text(
            "",
            encoding="utf-8",
        )
        return

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


def threshold_pass(
    value: Any,
) -> bool | None:
    number = numeric(value)

    if number is None:
        return None

    return number >= 1.0


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


def resolve_result_path(
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


def boolean_json_value(
    value: Any,
) -> bool | None:
    if isinstance(
        value,
        bool,
    ):
        return value

    number = numeric(value)

    if number is None:
        return None

    return number > 0


# ---------------------------------------------------------------------------
# Raw verifier inspection
# ---------------------------------------------------------------------------


def patch_record(
    verifier: Path,
) -> dict[str, Any]:
    """Recover the exact effective patch read by evaluate_rubrics.py.

    Verifier behavior:
        read(agent.patch) or read(agent_source_only.patch)

    Therefore an empty agent.patch causes fallback to the source-only patch.
    """

    agent_path = (
        verifier / "agent.patch"
    )

    source_path = (
        verifier
        / "agent_source_only.patch"
    )

    agent_exists = (
        agent_path.is_file()
    )

    source_exists = (
        source_path.is_file()
    )

    agent_text = read_text(
        agent_path
    )

    source_text = read_text(
        source_path
    )

    if agent_text:
        effective_text = agent_text
        effective_source = (
            "agent.patch"
        )

    elif source_text:
        effective_text = source_text
        effective_source = (
            "agent_source_only.patch"
        )

    elif agent_exists:
        effective_text = ""
        effective_source = (
            "agent.patch_empty"
        )

    elif source_exists:
        effective_text = ""
        effective_source = (
            "agent_source_only.patch_empty"
        )

    else:
        effective_text = None
        effective_source = (
            "missing"
        )

    return {
        "agent_patch_present": int(
            agent_exists
        ),
        "agent_patch_bytes": (
            agent_path.stat().st_size
            if agent_exists
            else ""
        ),
        "agent_patch_sha256": (
            sha256_file(agent_path)
            if agent_exists
            else ""
        ),
        "source_only_patch_present": int(
            source_exists
        ),
        "source_only_patch_bytes": (
            source_path.stat().st_size
            if source_exists
            else ""
        ),
        "source_only_patch_sha256": (
            sha256_file(source_path)
            if source_exists
            else ""
        ),
        "rubric_input_patch_source": (
            effective_source
        ),
        "rubric_input_patch_available": int(
            effective_text
            is not None
        ),
        "rubric_input_patch_bytes": (
            len(
                effective_text.encode(
                    "utf-8"
                )
            )
            if effective_text
            is not None
            else ""
        ),
        "rubric_input_patch_empty": int(
            effective_text
            is not None
            and not effective_text.strip()
        ),
    }


def verifier_stdout(
    trial: Path,
) -> str:
    candidates = (
        trial
        / "verifier"
        / "test-stdout.txt",
        trial
        / "verifier"
        / "test_stdout.txt",
        trial
        / "trial.log",
    )

    pieces = []

    for path in candidates:
        text = read_text(path)

        if text:
            pieces.append(
                f"\n===== {path.name} =====\n"
                + text
            )

    return "\n".join(pieces)


def raw_reward_values(
    verifier: Path,
) -> dict[str, Any]:
    reward = load_json(
        verifier
        / "reward.json"
    )

    rubrics = load_json(
        verifier
        / "rubrics_results.json"
    )

    reward = (
        reward
        if isinstance(
            reward,
            dict,
        )
        else {}
    )

    rubrics = (
        rubrics
        if isinstance(
            rubrics,
            dict,
        )
        else {}
    )

    must_have = (
        reward.get(
            "must_have_pass"
        )
    )

    if must_have is None:
        must_have = (
            rubrics.get(
                "must_have_pass"
            )
        )

    rubric_score = (
        reward.get(
            "rubrics_agg_score"
        )
    )

    if rubric_score is None:
        rubric_score = (
            rubrics.get(
                "agg_score"
            )
        )

    return {
        "raw_reward_json_present": int(
            bool(reward)
        ),
        "raw_rubrics_json_present": int(
            bool(rubrics)
        ),
        "raw_tests_reward": (
            reward.get(
                "tests_reward"
            )
        ),
        "raw_tests_pass": (
            threshold_pass(
                reward.get(
                    "tests_reward"
                )
            )
        ),
        "raw_must_have_pass": (
            boolean_json_value(
                must_have
            )
        ),
        "raw_rubrics_agg_score": (
            numeric(
                rubric_score
            )
        ),
        "raw_overall_pass": (
            threshold_pass(
                reward.get(
                    "overall_pass"
                )
            )
        ),
        "raw_evaluable_rubric_count": (
            rubrics.get(
                "evaluable_count",
                "",
            )
        ),
        "raw_must_have_count": (
            rubrics.get(
                "must_have_count",
                "",
            )
        ),
        "raw_rubric_count": (
            rubrics.get(
                "rubric_count",
                "",
            )
        ),
    }


def comparison_record(
    verifier: Path,
    stdout: str,
) -> dict[str, Any]:
    comparison_path = (
        verifier
        / "comparison_results.json"
    )

    mutation_path = (
        verifier
        / "agent_mutation_results.json"
    )

    agent_comparison_path = (
        verifier
        / "agent_comparison_results.json"
    )

    comparison = load_json(
        comparison_path
    )

    comparison = (
        comparison
        if isinstance(
            comparison,
            dict,
        )
        else {}
    )

    error_value = (
        comparison.get("error")
    )

    error_text = (
        str(error_value)
        if error_value
        not in (
            None,
            "",
        )
        else ""
    )

    low_error = (
        error_text.lower()
    )

    low_stdout = (
        stdout.lower()
    )

    failed_apply = (
        "failed to apply agent.patch"
        in low_stdout
        or "patch does not apply"
        in low_stdout
        or "cannot apply binary patch"
        in low_stdout
    )

    empty_warning = (
        "empty or near-empty patch"
        in low_stdout
        or "empty agent patch"
        in low_stdout
    )

    rubric_empty_warning = (
        "warning: empty agent patch; evaluating anyway"
        in low_stdout
    )

    return {
        "comparison_results_present": int(
            comparison_path.is_file()
        ),
        "comparison_error": int(
            bool(error_text)
        ),
        "comparison_error_text": (
            error_text[:2000]
        ),
        "comparison_error_missing_agent_results": int(
            bool(error_text)
            and (
                "agent_comparison_results.json"
                in low_error
            )
        ),
        "agent_mutation_results_present": int(
            mutation_path.is_file()
        ),
        "agent_comparison_results_present": int(
            agent_comparison_path.is_file()
        ),
        "failed_to_apply_agent_patch": int(
            failed_apply
        ),
        "empty_patch_warning": int(
            empty_warning
        ),
        "rubric_empty_patch_warning": int(
            rubric_empty_warning
        ),
    }


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------


def main() -> None:
    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    trial_rows: list[
        dict[str, Any]
    ] = []

    for study in STUDIES:
        for profile in PROFILES:
            source_path = (
                SOURCE
                / study
                / profile
                / "trials.json"
            )

            rows = load_json(
                source_path
            )

            if not isinstance(
                rows,
                list,
            ):
                raise SystemExit(
                    f"{source_path}: "
                    "expected JSON list"
                )

            for source in rows:
                if (
                    not isinstance(
                        source,
                        dict,
                    )
                    or not substantive(
                        source
                    )
                ):
                    continue

                stored_tests_pass = (
                    threshold_pass(
                        source.get(
                            "tests_reward"
                        )
                    )
                )

                stored_strict_pass = (
                    threshold_pass(
                        source.get(
                            "overall_pass"
                        )
                    )
                )

                result_path = (
                    resolve_result_path(
                        source.get(
                            "result_path"
                        )
                    )
                )

                raw_available = (
                    result_path
                    is not None
                    and result_path.is_file()
                )

                row: dict[str, Any] = {
                    "study": study,
                    "profile": profile,
                    "trial_name": (
                        source.get(
                            "trial_name"
                        )
                        or ""
                    ),
                    "task_name": (
                        source.get(
                            "task_name"
                        )
                        or ""
                    ),
                    "base_task_id": (
                        source.get(
                            "base_task_id"
                        )
                        or ""
                    ),
                    "condition": (
                        source.get(
                            "condition"
                        )
                        or ""
                    ),
                    "channel": (
                        source.get(
                            "channel"
                        )
                        or ""
                    ),
                    "pressure_type": (
                        source.get(
                            "pressure_type"
                        )
                        or ""
                    ),
                    "stored_tests_reward": (
                        source.get(
                            "tests_reward"
                        )
                    ),
                    "stored_tests_pass": (
                        stored_tests_pass
                    ),
                    "stored_overall_pass": (
                        stored_strict_pass
                    ),
                    "stored_rubrics_agg_score": (
                        source.get(
                            "rubrics_agg_score"
                        )
                    ),
                    "stored_changed_files": (
                        source.get(
                            "changed_files"
                        )
                    ),
                    "result_path": (
                        str(result_path)
                        if result_path
                        is not None
                        else ""
                    ),
                    "raw_trial_available": int(
                        raw_available
                    ),
                }

                if not raw_available:
                    row.update(
                        {
                            "strict_raw_unavailable": int(
                                stored_strict_pass
                                is True
                            ),
                            "requires_manual_or_archive_recovery": 1,
                        }
                    )

                    trial_rows.append(row)
                    continue

                assert result_path is not None

                trial = (
                    result_path.parent
                )

                verifier = (
                    trial / "verifier"
                )

                stdout = verifier_stdout(
                    trial
                )

                row.update(
                    patch_record(
                        verifier
                    )
                )

                row.update(
                    raw_reward_values(
                        verifier
                    )
                )

                row.update(
                    comparison_record(
                        verifier,
                        stdout,
                    )
                )

                raw_tests_pass = (
                    row.get(
                        "raw_tests_pass"
                    )
                )

                raw_strict_pass = (
                    row.get(
                        "raw_overall_pass"
                    )
                )

                raw_must_have = (
                    row.get(
                        "raw_must_have_pass"
                    )
                )

                effective_empty = bool(
                    row.get(
                        "rubric_input_patch_empty"
                    )
                )

                effective_bytes = numeric(
                    row.get(
                        "rubric_input_patch_bytes"
                    )
                )

                failed_apply = bool(
                    row.get(
                        "failed_to_apply_agent_patch"
                    )
                )

                comparison_error = bool(
                    row.get(
                        "comparison_error"
                    )
                )

                row.update(
                    {
                        "stored_raw_tests_mismatch": int(
                            raw_tests_pass
                            is not None
                            and stored_tests_pass
                            is not None
                            and raw_tests_pass
                            != stored_tests_pass
                        ),
                        "stored_raw_strict_mismatch": int(
                            raw_strict_pass
                            is not None
                            and stored_strict_pass
                            is not None
                            and raw_strict_pass
                            != stored_strict_pass
                        ),
                        "rubric_true_on_empty_input": int(
                            raw_must_have
                            is True
                            and effective_empty
                        ),
                        "nonempty_patch_failed_apply": int(
                            failed_apply
                            and effective_bytes
                            is not None
                            and effective_bytes > 0
                        ),
                        "empty_patch_comparison_error": int(
                            effective_empty
                            and comparison_error
                        ),
                        "tests_pass_with_comparison_error": int(
                            raw_tests_pass
                            is True
                            and comparison_error
                        ),
                        "strict_with_comparison_error": int(
                            raw_strict_pass
                            is True
                            and comparison_error
                        ),
                        "strict_with_failed_patch_apply": int(
                            raw_strict_pass
                            is True
                            and failed_apply
                        ),
                        "strict_with_empty_rubric_input": int(
                            raw_strict_pass
                            is True
                            and effective_empty
                        ),
                        "strict_without_raw_must_have": int(
                            raw_strict_pass
                            is True
                            and raw_must_have
                            is not True
                        ),
                    }
                )

                row[
                    "stored_strict_at_forensic_risk"
                ] = int(
                    stored_strict_pass
                    is True
                    and (
                        comparison_error
                        or failed_apply
                        or effective_empty
                        or raw_must_have
                        is not True
                    )
                )

                row[
                    "requires_manual_or_archive_recovery"
                ] = int(
                    not raw_available
                )

                trial_rows.append(row)

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    summary_rows: list[
        dict[str, Any]
    ] = []

    groups: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in trial_rows:
        groups[
            (
                str(row["study"]),
                str(row["profile"]),
            )
        ].append(row)

    count_fields = (
        "raw_trial_available",
        "strict_raw_unavailable",
        "stored_tests_pass",
        "stored_overall_pass",
        "rubric_input_patch_available",
        "rubric_input_patch_empty",
        "raw_must_have_pass",
        "rubric_true_on_empty_input",
        "comparison_error",
        "comparison_error_missing_agent_results",
        "agent_mutation_results_present",
        "agent_comparison_results_present",
        "failed_to_apply_agent_patch",
        "nonempty_patch_failed_apply",
        "empty_patch_comparison_error",
        "tests_pass_with_comparison_error",
        "strict_with_comparison_error",
        "strict_with_failed_patch_apply",
        "strict_with_empty_rubric_input",
        "strict_without_raw_must_have",
        "stored_raw_tests_mismatch",
        "stored_raw_strict_mismatch",
        "stored_strict_at_forensic_risk",
    )

    for (
        study,
        profile,
    ), rows in sorted(
        groups.items()
    ):
        summary: dict[
            str,
            Any,
        ] = {
            "study": study,
            "profile": profile,
            "substantive_n": len(rows),
        }

        for field in count_fields:
            summary[
                field + "_n"
            ] = sum(
                value is True
                or value == 1
                for value in (
                    row.get(field)
                    for row in rows
                )
            )

        summary_rows.append(
            summary
        )

    suspicious_fields = (
        "rubric_true_on_empty_input",
        "nonempty_patch_failed_apply",
        "tests_pass_with_comparison_error",
        "strict_with_comparison_error",
        "strict_with_failed_patch_apply",
        "strict_with_empty_rubric_input",
        "strict_without_raw_must_have",
        "stored_raw_tests_mismatch",
        "stored_raw_strict_mismatch",
        "stored_strict_at_forensic_risk",
        "strict_raw_unavailable",
    )

    suspicious_rows = [
        row
        for row in trial_rows
        if any(
            row.get(field)
            in (
                1,
                True,
            )
            for field
            in suspicious_fields
        )
    ]

    write_csv(
        OUTPUT
        / "verifier_forensics_trials.csv",
        trial_rows,
    )

    write_csv(
        OUTPUT
        / "verifier_forensics_summary.csv",
        summary_rows,
    )

    write_csv(
        OUTPUT
        / "verifier_forensics_suspicious.csv",
        suspicious_rows,
    )

    manifest = {
        "audit_version": (
            "verifier-forensics-1.0"
        ),
        "source_root": str(
            SOURCE
        ),
        "output_root": str(
            OUTPUT
        ),
        "substantive_trials": len(
            trial_rows
        ),
        "suspicious_trials": len(
            suspicious_rows
        ),
        "network_calls": 0,
        "api_calls": 0,
        "source_mutations": 0,
        "tests_pass_definition": (
            "tests_reward >= 1.0"
        ),
        "strict_pass_definition": (
            "overall_pass >= 1.0"
        ),
        "rubric_input_reconstruction": (
            "agent.patch text if nonempty, "
            "else agent_source_only.patch text"
        ),
        "interpretation_policy": {
            "comparison_errors_are_not_automatically_censored": True,
            "failed_nonempty_patch_apply_is_model_failure_candidate": True,
            "rubric_true_on_empty_input_is_verifier_anomaly_candidate": True,
            "stored_strict_success_is_not_certified_by_this_audit": True,
        },
    }

    write_json(
        OUTPUT
        / "verifier_forensics_manifest.json",
        manifest,
    )

    # ------------------------------------------------------------------
    # Terminal report
    # ------------------------------------------------------------------

    print("=" * 108)
    print("CURRENT VERIFIER FORENSICS")
    print("=" * 108)

    for row in summary_rows:
        print()
        print(
            f"{row['study'].upper():11s} "
            f"{row['profile'].upper()}"
        )

        print(
            "  substantive                         ",
            row["substantive_n"],
        )
        print(
            "  raw available                       ",
            row["raw_trial_available_n"],
        )
        print(
            "  strict raw unavailable              ",
            row["strict_raw_unavailable_n"],
        )
        print(
            "  stored test passes                  ",
            row["stored_tests_pass_n"],
        )
        print(
            "  stored strict passes                ",
            row["stored_overall_pass_n"],
        )
        print(
            "  rubric input empty                  ",
            row["rubric_input_patch_empty_n"],
        )
        print(
            "  must-have true on empty rubric input",
            row["rubric_true_on_empty_input_n"],
        )
        print(
            "  comparison errors                   ",
            row["comparison_error_n"],
        )
        print(
            "  failed nonempty patch applies       ",
            row["nonempty_patch_failed_apply_n"],
        )
        print(
            "  tests pass + comparison error       ",
            row["tests_pass_with_comparison_error_n"],
        )
        print(
            "  strict + comparison error           ",
            row["strict_with_comparison_error_n"],
        )
        print(
            "  strict + failed patch apply         ",
            row["strict_with_failed_patch_apply_n"],
        )
        print(
            "  strict + empty rubric input         ",
            row["strict_with_empty_rubric_input_n"],
        )
        print(
            "  stored/raw test mismatches          ",
            row["stored_raw_tests_mismatch_n"],
        )
        print(
            "  stored/raw strict mismatches        ",
            row["stored_raw_strict_mismatch_n"],
        )
        print(
            "  STORED STRICT AT FORENSIC RISK      ",
            row["stored_strict_at_forensic_risk_n"],
        )

    print()
    print("=" * 108)
    print("OUTPUTS")
    print("=" * 108)

    for name in (
        "verifier_forensics_trials.csv",
        "verifier_forensics_summary.csv",
        "verifier_forensics_suspicious.csv",
        "verifier_forensics_manifest.json",
    ):
        print(
            " ",
            OUTPUT / name,
        )

    print()
    print("network calls: 0")
    print("API calls: 0")
    print("source mutations: 0")
    print()
    print("VERIFIER FORENSICS: PASS")


if __name__ == "__main__":
    main()
