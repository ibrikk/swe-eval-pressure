#!/usr/bin/env python3
"""Synthesize deterministic SWE-EvalPressure behavioral result outputs.

This script performs reporting/synthesis only. It does not recompute
behavioral endpoints, matched effects, confidence intervals, p-values,
or multiplicity adjustments.

Expected inputs are profile directories produced by scripts/07_analyze.py.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Iterable

from behavior_metrics import PRIMARY_BINARY_ENDPOINTS


REPORT_SCHEMA_VERSION = "1.0"

REQUIRED_FILES = (
    "behavior_trials.csv",
    "behavior_prevalence.csv",
    "matched_behavior_pairs.csv",
    "behavior_binary_effects.csv",
    "behavior_secondary_effects.csv",
    "behavior_multiplicity.csv",
)

PRIMARY_MATRIX_BASE_FIELDS = [
    "report_schema_version",
    "analysis_schema_version",
    "analysis_mode",
    "study_signature",
    "profile",
    "pair_type",
    "channel",
    "baseline_condition",
    "baseline_channel",
    "treatment_condition",
    "treatment_channel",
    "n_pairs",
]

PRIMARY_MATRIX_ENDPOINT_FIELDS = [
    field
    for endpoint in PRIMARY_BINARY_ENDPOINTS
    for field in (
        f"{endpoint}__baseline_pct",
        f"{endpoint}__treatment_pct",
        f"{endpoint}__delta_pp",
        f"{endpoint}__ci_low_pp",
        f"{endpoint}__ci_high_pp",
        f"{endpoint}__mcnemar_p",
        f"{endpoint}__holm_p",
        f"{endpoint}__adjusted_reject",
    )
]

PRIMARY_MATRIX_FIELDS = (
    PRIMARY_MATRIX_BASE_FIELDS
    + PRIMARY_MATRIX_ENDPOINT_FIELDS
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        return list(csv.DictReader(handle))


def write_csv(
    path: Path,
    rows: Iterable[dict[str, Any]],
    *,
    fieldnames: list[str] | None = None,
) -> None:
    rows = list(rows)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if fieldnames is None:
        fieldnames = []
        seen: set[str] = set()

        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def discover_profile_dirs(
    analysis_root: Path,
) -> list[Path]:
    if all(
        (analysis_root / name).is_file()
        for name in REQUIRED_FILES
    ):
        return [analysis_root]

    profiles = [
        path
        for path in sorted(
            analysis_root.iterdir()
        )
        if path.is_dir()
        and all(
            (path / name).is_file()
            for name in REQUIRED_FILES
        )
    ]

    if not profiles:
        raise SystemExit(
            "No complete behavioral analyzer profile "
            f"directories found under {analysis_root}"
        )

    return profiles


def load_profile_tables(
    profile_dir: Path,
) -> dict[str, list[dict[str, str]]]:
    missing = [
        name
        for name in REQUIRED_FILES
        if not (profile_dir / name).is_file()
    ]

    if missing:
        raise ValueError(
            f"{profile_dir}: missing required files: "
            + ", ".join(missing)
        )

    return {
        name.removesuffix(".csv"): read_csv(
            profile_dir / name
        )
        for name in REQUIRED_FILES
    }


def profile_identity(
    profile_dir: Path,
    tables: dict[
        str,
        list[dict[str, str]],
    ],
) -> tuple[str, str, str]:
    candidates: list[dict[str, str]] = []

    for name in (
        "behavior_binary_effects",
        "behavior_secondary_effects",
        "behavior_prevalence",
        "behavior_trials",
    ):
        candidates.extend(
            tables.get(name, [])
        )

    if not candidates:
        raise ValueError(
            f"{profile_dir}: no behavioral rows"
        )

    profiles = {
        row.get("profile", "")
        for row in candidates
        if row.get("profile", "")
    }
    schemas = {
        row.get(
            "analysis_schema_version",
            "",
        )
        for row in candidates
        if row.get(
            "analysis_schema_version",
            "",
        )
    }
    modes = {
        row.get("analysis_mode", "")
        for row in candidates
        if row.get("analysis_mode", "")
    }

    if len(profiles) != 1:
        raise ValueError(
            f"{profile_dir}: inconsistent profile IDs "
            f"{sorted(profiles)}"
        )

    if len(schemas) != 1:
        raise ValueError(
            f"{profile_dir}: inconsistent analyzer schemas "
            f"{sorted(schemas)}"
        )

    if len(modes) != 1:
        raise ValueError(
            f"{profile_dir}: inconsistent analysis modes "
            f"{sorted(modes)}"
        )

    return (
        next(iter(profiles)),
        next(iter(schemas)),
        next(iter(modes)),
    )


def add_report_metadata(
    rows: Iterable[dict[str, str]],
) -> list[dict[str, str]]:
    return [
        {
            "report_schema_version": (
                REPORT_SCHEMA_VERSION
            ),
            **row,
        }
        for row in rows
    ]


def primary_effect_matrix(
    rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    groups: dict[
        tuple[str, ...],
        dict[str, Any],
    ] = {}

    base_fields = (
        "analysis_schema_version",
        "analysis_mode",
        "study_signature",
        "profile",
        "pair_type",
        "channel",
        "baseline_condition",
        "baseline_channel",
        "treatment_condition",
        "treatment_channel",
    )

    for row in rows:
        endpoint = row.get(
            "endpoint",
            "",
        )

        if endpoint not in PRIMARY_BINARY_ENDPOINTS:
            raise ValueError(
                "Unknown primary endpoint in "
                "behavior_binary_effects.csv: "
                f"{endpoint!r}"
            )

        key = tuple(
            row.get(field, "")
            for field in base_fields
        )

        if key not in groups:
            groups[key] = {
                "report_schema_version": (
                    REPORT_SCHEMA_VERSION
                ),
                **{
                    field: row.get(field, "")
                    for field in base_fields
                },
                "n_pairs": row.get(
                    "n_pairs",
                    "",
                ),
            }

        output = groups[key]

        if output["n_pairs"] != row.get(
            "n_pairs",
            "",
        ):
            raise ValueError(
                "Primary endpoints within one contrast "
                "have inconsistent matched denominators: "
                f"{key!r}"
            )

        prefix = endpoint

        output[
            f"{prefix}__baseline_pct"
        ] = row.get(
            "baseline_prevalence_pct",
            "",
        )
        output[
            f"{prefix}__treatment_pct"
        ] = row.get(
            "treatment_prevalence_pct",
            "",
        )
        output[
            f"{prefix}__delta_pp"
        ] = row.get(
            "risk_difference_pp",
            "",
        )
        output[
            f"{prefix}__ci_low_pp"
        ] = row.get(
            "bootstrap_ci_low_pp",
            "",
        )
        output[
            f"{prefix}__ci_high_pp"
        ] = row.get(
            "bootstrap_ci_high_pp",
            "",
        )
        output[
            f"{prefix}__mcnemar_p"
        ] = row.get(
            "mcnemar_exact_p",
            "",
        )
        output[
            f"{prefix}__holm_p"
        ] = row.get(
            "holm_adjusted_p",
            "",
        )
        output[
            f"{prefix}__adjusted_reject"
        ] = row.get(
            "adjusted_reject",
            "",
        )

    output_rows = []

    for key in sorted(groups):
        row = groups[key]

        missing = [
            endpoint
            for endpoint
            in PRIMARY_BINARY_ENDPOINTS
            if (
                f"{endpoint}__delta_pp"
                not in row
            )
        ]

        if missing:
            raise ValueError(
                "Primary contrast missing endpoints: "
                f"{key!r}: {missing}"
            )

        output_rows.append(row)

    return output_rows


def synthesize(
    analysis_root: Path,
    output_dir: Path,
) -> dict[str, int]:
    profile_dirs = discover_profile_dirs(
        analysis_root
    )

    combined: dict[
        str,
        list[dict[str, str]],
    ] = {
        name.removesuffix(".csv"): []
        for name in REQUIRED_FILES
    }

    inventory: list[dict[str, Any]] = []

    seen_profiles: set[str] = set()

    for profile_dir in profile_dirs:
        tables = load_profile_tables(
            profile_dir
        )

        (
            profile,
            schema,
            mode,
        ) = profile_identity(
            profile_dir,
            tables,
        )

        if profile in seen_profiles:
            raise ValueError(
                "Duplicate profile across analysis "
                f"directories: {profile}"
            )

        seen_profiles.add(profile)

        signatures = {
            row.get(
                "study_signature",
                "",
            )
            for rows in tables.values()
            for row in rows
            if row.get(
                "study_signature",
                "",
            )
        }

        inventory.append({
            "report_schema_version": (
                REPORT_SCHEMA_VERSION
            ),
            "profile": profile,
            "analysis_schema_version": (
                schema
            ),
            "analysis_mode": mode,
            "profile_dir": str(
                profile_dir
            ),
            "study_signatures": "|".join(
                sorted(signatures)
            ),
            **{
                f"{name}_rows": len(rows)
                for name, rows
                in tables.items()
            },
        })

        for name, rows in tables.items():
            combined[name].extend(
                add_report_metadata(rows)
            )

    primary_matrix = primary_effect_matrix(
        combined[
            "behavior_binary_effects"
        ]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        output_dir
        / "behavior_report_inventory.csv",
        inventory,
    )

    write_csv(
        output_dir
        / "behavior_primary_effects_all.csv",
        combined[
            "behavior_binary_effects"
        ],
    )

    write_csv(
        output_dir
        / "behavior_primary_matrix.csv",
        primary_matrix,
        fieldnames=PRIMARY_MATRIX_FIELDS,
    )

    write_csv(
        output_dir
        / "behavior_secondary_effects_all.csv",
        combined[
            "behavior_secondary_effects"
        ],
    )

    write_csv(
        output_dir
        / "behavior_multiplicity_all.csv",
        combined[
            "behavior_multiplicity"
        ],
    )

    write_csv(
        output_dir
        / "behavior_prevalence_all.csv",
        combined[
            "behavior_prevalence"
        ],
    )

    write_csv(
        output_dir
        / "matched_behavior_pairs_all.csv",
        combined[
            "matched_behavior_pairs"
        ],
    )

    return {
        "profiles": len(profile_dirs),
        "primary_effect_rows": len(
            combined[
                "behavior_binary_effects"
            ]
        ),
        "primary_matrix_rows": len(
            primary_matrix
        ),
        "secondary_effect_rows": len(
            combined[
                "behavior_secondary_effects"
            ]
        ),
        "multiplicity_rows": len(
            combined[
                "behavior_multiplicity"
            ]
        ),
        "prevalence_rows": len(
            combined[
                "behavior_prevalence"
            ]
        ),
        "matched_pair_rows": len(
            combined[
                "matched_behavior_pairs"
            ]
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--analysis-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary = synthesize(
        args.analysis_root,
        args.output_dir,
    )

    print("Behavior report synthesis complete")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(args.output_dir)


if __name__ == "__main__":
    main()
