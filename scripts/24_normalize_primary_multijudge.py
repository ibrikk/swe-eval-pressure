#!/usr/bin/env python3
"""Normalize frozen SWE-EvalPressure primary multijudge artifacts.

No network calls.
No LLM calls.
No re-judging.

Consumes:
  jobs/*.json
  consensus/*.json
  config/semantic_judge_schema.json

Produces:
  judge_labels.csv
  deepseek.csv
  gemini.csv
  consensus_labels.csv
  consensus_coverage.csv
  pair_completeness.csv
  manifest.json

Missing judge output remains missing. It is never converted to a negative label.
Strict consensus exists only when both valid primary judges provide the same
valid label for a field.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VERSION = "1.0"


def load_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    fields: list[str] = []
    seen: set[str] = set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)


def load_schema(
    path: Path,
) -> tuple[
    dict[str, dict[str, Any]],
    tuple[str, ...],
]:
    obj = load_json(path)

    fields = obj.get("fields")

    if not isinstance(fields, dict) or not fields:
        raise ValueError(
            "semantic schema has no fields"
        )

    primary = obj.get("primary_judges")

    if not isinstance(primary, list):
        raise ValueError(
            "semantic schema has no primary_judges"
        )

    families = tuple(
        str(row["family"])
        for row in primary
    )

    if len(families) != 2:
        raise ValueError(
            "expected exactly two primary judge families"
        )

    return fields, families


def valid_job(
    obj: dict[str, Any],
) -> bool:
    if obj.get("status") != "ok":
        return False

    final = obj.get("final_cache_entry")

    return (
        isinstance(final, dict)
        and final.get("status") == "ok"
        and isinstance(
            final.get("judgment"),
            dict,
        )
    )


def normalize_evidence(
    evidence: Any,
) -> list[dict[str, Any]]:
    if evidence is None:
        return []

    if not isinstance(evidence, list):
        raise ValueError(
            "evidence must be a list"
        )

    out = []

    for item in evidence:
        if not isinstance(item, dict):
            raise ValueError(
                "evidence item must be an object"
            )

        step = item.get("step_index")
        quote = item.get("quote")

        if (
            not isinstance(step, int)
            or step < 0
        ):
            raise ValueError(
                "invalid evidence step_index"
            )

        if (
            not isinstance(quote, str)
            or not quote.strip()
        ):
            raise ValueError(
                "invalid evidence quote"
            )

        out.append(
            {
                "step_index": step,
                "quote": quote,
            }
        )

    return out


def normalize_valid_judgment(
    obj: dict[str, Any],
    fields: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    final = obj["final_cache_entry"]
    judgment = final["judgment"]

    out = {}

    for field, spec in fields.items():
        value = judgment.get(field)

        if not isinstance(value, dict):
            raise ValueError(
                f"missing judgment field {field}"
            )

        label = value.get("label")

        allowed = set(
            spec.get("labels", [])
        )

        if label not in allowed:
            raise ValueError(
                f"{field}: invalid label {label!r}"
            )

        evidence = normalize_evidence(
            value.get("evidence", [])
        )

        required = set(
            spec.get(
                "evidence_required_for",
                [],
            )
        )

        if label in required and not evidence:
            raise ValueError(
                f"{field}: {label!r} requires evidence"
            )

        out[field] = {
            "label": label,
            "evidence": evidence,
        }

    return out


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--semantic-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--schema",
        type=Path,
        default=Path(
            "config/semantic_judge_schema.json"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    root = args.semantic_root.resolve()
    output = args.output_dir.resolve()

    jobs_dir = root / "jobs"
    consensus_dir = root / "consensus"

    if not jobs_dir.is_dir():
        raise SystemExit(
            f"Missing jobs directory: {jobs_dir}"
        )

    if not consensus_dir.is_dir():
        raise SystemExit(
            f"Missing consensus directory: {consensus_dir}"
        )

    fields, primary_families = (
        load_schema(args.schema)
    )

    job_index: dict[
        tuple[str, str, str],
        dict[str, Any],
    ] = {}

    judge_rows = []
    job_status = Counter()
    profile_job_counts = Counter()

    for path in sorted(
        jobs_dir.glob("*.json")
    ):
        obj = load_json(path)

        profile = str(
            obj.get("profile") or ""
        )
        trial = str(
            obj.get("trial_name") or ""
        )
        family = str(
            obj.get("judge_family") or ""
        )

        if not profile or not trial or not family:
            raise ValueError(
                f"{path}: missing job identity"
            )

        if family not in primary_families:
            raise ValueError(
                f"{path}: unexpected judge family {family}"
            )

        key = (
            profile,
            trial,
            family,
        )

        if key in job_index:
            raise ValueError(
                f"duplicate judge identity: {key}"
            )

        ok = valid_job(obj)

        status = (
            "ok"
            if ok
            else str(
                obj.get("status")
                or "missing"
            )
        )

        row: dict[str, Any] = {
            "sample_id": (
                f"{profile}::{trial}"
            ),
            "profile": profile,
            "trial_name": trial,
            "condition": str(
                obj.get("condition") or ""
            ),
            "placement": str(
                obj.get("placement") or ""
            ),
            "pressure_type": str(
                obj.get("pressure_type") or ""
            ),
            "judge_family": family,
            "judge_model": str(
                obj.get("judge_model") or ""
            ),
            "judge_status": status,
            "trajectory_hash": str(
                obj.get("trajectory_hash") or ""
            ),
            "semantic_schema_version": str(
                obj.get(
                    "semantic_schema_version"
                )
                or ""
            ),
            "rubric_version": str(
                obj.get("rubric_version")
                or ""
            ),
            "semantic_view_version": str(
                obj.get(
                    "semantic_view_version"
                )
                or ""
            ),
            "job_path": str(path),
        }

        normalized = {}

        if ok:
            normalized = (
                normalize_valid_judgment(
                    obj,
                    fields,
                )
            )

        for field in fields:
            if ok:
                row[field] = (
                    normalized[field]["label"]
                )
                row[
                    f"{field}__evidence_json"
                ] = json.dumps(
                    normalized[field]["evidence"],
                    ensure_ascii=False,
                    sort_keys=True,
                )
            else:
                row[field] = ""
                row[
                    f"{field}__evidence_json"
                ] = ""

        job_index[key] = {
            "artifact": obj,
            "row": row,
            "normalized": normalized,
            "valid": ok,
        }

        judge_rows.append(row)
        job_status[status] += 1
        profile_job_counts[profile] += 1

    consensus_index = {}
    consensus_rows = []

    field_accounting = {
        field: Counter()
        for field in fields
    }

    profile_consensus_counts = Counter()

    for path in sorted(
        consensus_dir.glob("*.json")
    ):
        obj = load_json(path)

        profile = str(
            obj.get("profile") or ""
        )
        trial = str(
            obj.get("trial_name") or ""
        )

        if not profile or not trial:
            raise ValueError(
                f"{path}: missing consensus identity"
            )

        key = (
            profile,
            trial,
        )

        if key in consensus_index:
            raise ValueError(
                f"duplicate consensus identity: {key}"
            )

        consensus = obj.get("consensus")

        if not isinstance(consensus, dict):
            raise ValueError(
                f"{path}: invalid consensus object"
            )

        consensus_fields = consensus.get(
            "fields"
        )

        if not isinstance(
            consensus_fields,
            dict,
        ):
            raise ValueError(
                f"{path}: no consensus fields"
            )

        row: dict[str, Any] = {
            "sample_id": (
                f"{profile}::{trial}"
            ),
            "profile": profile,
            "trial_name": trial,
            "condition": str(
                obj.get("condition") or ""
            ),
            "placement": str(
                obj.get("placement") or ""
            ),
            "pressure_type": str(
                obj.get("pressure_type") or ""
            ),
            "consensus_path": str(path),
        }

        field_agreements = 0

        for field, spec in fields.items():
            value = consensus_fields.get(
                field
            )

            if not isinstance(value, dict):
                raise ValueError(
                    f"{path}: missing consensus field {field}"
                )

            artifact_valid_judges = int(
                value.get(
                    "valid_judges",
                    0,
                )
            )

            actual = []

            for family in primary_families:
                job = job_index.get(
                    (
                        profile,
                        trial,
                        family,
                    )
                )

                if job and job["valid"]:
                    actual.append(
                        job[
                            "normalized"
                        ][field]["label"]
                    )

            if (
                artifact_valid_judges
                != len(actual)
            ):
                raise ValueError(
                    f"{profile}/{trial}/{field}: "
                    "consensus valid_judges mismatch "
                    f"{artifact_valid_judges} != {len(actual)}"
                )

            should_consensus = (
                len(actual) == 2
                and actual[0] == actual[1]
            )

            artifact_exists = bool(
                value.get(
                    "consensus_exists"
                )
            )

            label = value.get("label")

            if should_consensus:
                if not artifact_exists:
                    raise ValueError(
                        f"{profile}/{trial}/{field}: "
                        "missing expected strict consensus"
                    )

                if label != actual[0]:
                    raise ValueError(
                        f"{profile}/{trial}/{field}: "
                        "consensus label mismatch"
                    )

                if label not in set(
                    spec.get("labels", [])
                ):
                    raise ValueError(
                        f"{profile}/{trial}/{field}: "
                        "invalid consensus label"
                    )

                field_agreements += 1

            else:
                if artifact_exists:
                    raise ValueError(
                        f"{profile}/{trial}/{field}: "
                        "consensus exists without "
                        "two matching valid judges"
                    )

                if label is not None:
                    raise ValueError(
                        f"{profile}/{trial}/{field}: "
                        "unresolved field has label"
                    )

            status = str(
                value.get("status")
                or "unknown"
            )

            row[field] = (
                label
                if should_consensus
                else ""
            )
            row[
                f"{field}__status"
            ] = status
            row[
                f"{field}__valid_judges"
            ] = len(actual)
            row[
                f"{field}__consensus_exists"
            ] = int(
                should_consensus
            )

            field_accounting[field][
                status
            ] += 1

        row["fields_in_strict_consensus"] = (
            field_agreements
        )
        row[
            "all_fields_strict_consensus"
        ] = int(
            field_agreements
            == len(fields)
        )

        consensus_rows.append(row)
        consensus_index[key] = obj
        profile_consensus_counts[
            profile
        ] += 1

    # Every consensus trajectory should have exactly
    # one planned job from each primary judge.
    pair_rows = []

    for (
        profile,
        trial,
    ) in sorted(consensus_index):
        family_status = {}

        for family in primary_families:
            job = job_index.get(
                (
                    profile,
                    trial,
                    family,
                )
            )

            family_status[family] = (
                "ok"
                if job and job["valid"]
                else (
                    str(
                        job["row"][
                            "judge_status"
                        ]
                    )
                    if job
                    else "absent"
                )
            )

        valid_n = sum(
            value == "ok"
            for value in (
                family_status.values()
            )
        )

        pair_rows.append(
            {
                "sample_id": (
                    f"{profile}::{trial}"
                ),
                "profile": profile,
                "trial_name": trial,
                **{
                    f"{family}_status": status
                    for family, status
                    in family_status.items()
                },
                "valid_judges": valid_n,
                "both_ok": int(
                    valid_n == 2
                ),
                "one_ok": int(
                    valid_n == 1
                ),
                "neither_ok": int(
                    valid_n == 0
                ),
            }
        )

    expected_job_keys = {
        (
            row["profile"],
            row["trial_name"],
            family,
        )
        for row in consensus_rows
        for family in primary_families
    }

    actual_job_keys = set(
        job_index
    )

    if actual_job_keys != expected_job_keys:
        extra = actual_job_keys - expected_job_keys
        missing = (
            expected_job_keys
            - actual_job_keys
        )

        raise ValueError(
            "job/consensus universe mismatch: "
            f"extra={len(extra)} "
            f"missing={len(missing)}"
        )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        output / "judge_labels.csv",
        judge_rows,
    )

    for family in primary_families:
        rows = []

        for source in judge_rows:
            if (
                source["judge_family"]
                != family
            ):
                continue

            row = {
                "sample_id": source["sample_id"],
                "profile": source["profile"],
                "trial_name": (
                    source["trial_name"]
                ),
                "judge_status": (
                    source["judge_status"]
                ),
            }

            for field in fields:
                row[field] = (
                    source[field]
                )

            rows.append(row)

        write_csv(
            output / f"{family}.csv",
            rows,
        )

    write_csv(
        output / "consensus_labels.csv",
        consensus_rows,
    )

    coverage_rows = []

    for field in fields:
        counts = field_accounting[field]

        n = sum(counts.values())
        agreement = counts.get(
            "agreement",
            0,
        )

        coverage_rows.append(
            {
                "field": field,
                "n_trajectories": n,
                "strict_consensus_n": (
                    agreement
                ),
                "strict_consensus_rate": (
                    agreement / n
                    if n
                    else ""
                ),
                "status_counts_json": (
                    json.dumps(
                        dict(
                            sorted(
                                counts.items()
                            )
                        ),
                        sort_keys=True,
                    )
                ),
            }
        )

    write_csv(
        output / "consensus_coverage.csv",
        coverage_rows,
    )

    write_csv(
        output / "pair_completeness.csv",
        pair_rows,
    )

    pair_counts = Counter()

    for row in pair_rows:
        if row["both_ok"]:
            pair_counts["both_ok"] += 1
        elif row["one_ok"]:
            pair_counts["one_ok"] += 1
        else:
            pair_counts[
                "neither_ok"
            ] += 1

    manifest = {
        "version": VERSION,
        "semantic_root": str(root),
        "schema": str(
            args.schema.resolve()
        ),
        "schema_sha256": sha256_file(
            args.schema
        ),
        "primary_judges": list(
            primary_families
        ),
        "fields": list(fields),
        "job_count": len(judge_rows),
        "job_status": dict(
            sorted(
                job_status.items()
            )
        ),
        "profile_job_counts": dict(
            sorted(
                profile_job_counts.items()
            )
        ),
        "consensus_count": len(
            consensus_rows
        ),
        "profile_consensus_counts": (
            dict(
                sorted(
                    profile_consensus_counts.items()
                )
            )
        ),
        "pair_completeness": dict(
            sorted(
                pair_counts.items()
            )
        ),
        "missing_is_not_negative": True,
        "consensus_rule": (
            "2 valid primary judges with "
            "exact matching field label"
        ),
        "semantic_states_are_post_treatment": True,
    }

    (
        output / "manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "PRIMARY MULTIJUDGE NORMALIZATION: PASS"
    )
    print(
        "jobs:",
        len(judge_rows),
    )
    print(
        "job status:",
        dict(job_status),
    )
    print(
        "consensus:",
        len(consensus_rows),
    )
    print(
        "profiles:",
        dict(profile_consensus_counts),
    )
    print(
        "pair completeness:",
        dict(pair_counts),
    )

    for row in coverage_rows:
        print(
            row["field"],
            "strict=",
            row["strict_consensus_n"],
            "/",
            row["n_trajectories"],
            f"({100 * row['strict_consensus_rate']:.1f}%)",
        )

    print(
        "output:",
        output,
    )


if __name__ == "__main__":
    main()
