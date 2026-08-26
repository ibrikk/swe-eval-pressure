#!/usr/bin/env python3
"""Matched inference for the frozen historical semantic study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np


VERSION = "1.0"


def load(path: Path) -> Any:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def exact_mcnemar(
    up: int,
    down: int,
) -> float:
    """Exact two-sided McNemar test."""
    n = up + down

    if n == 0:
        return 1.0

    k = min(up, down)

    lower_tail = sum(
        math.comb(n, i)
        for i in range(k + 1)
    ) / (2 ** n)

    return min(
        1.0,
        2.0 * lower_tail,
    )


def holm_adjust(
    rows: list[dict[str, Any]],
    output_field: str,
) -> None:
    """Holm step-down adjusted p-values."""
    ordering = sorted(
        range(len(rows)),
        key=lambda i: rows[i]["p_mcnemar"],
    )

    m = len(rows)
    adjusted = [0.0] * m
    running = 0.0

    for rank, index in enumerate(ordering):
        candidate = min(
            1.0,
            (m - rank)
            * rows[index]["p_mcnemar"],
        )

        running = max(
            running,
            candidate,
        )

        adjusted[index] = min(
            1.0,
            running,
        )

    for row, value in zip(
        rows,
        adjusted,
        strict=True,
    ):
        row[output_field] = value


def binary_value(
    result: dict[str, Any],
    spec: dict[str, Any],
) -> int | None:
    """Convert resolved consensus into the prespecified binary endpoint."""
    if (
        result.get("status")
        != "agreement"
    ):
        return None

    label = result.get("label")

    if label in spec["positive_labels"]:
        return 1

    if label in spec["negative_labels"]:
        return 0

    # Ambiguous or any unanticipated
    # non-binary label remains unresolved.
    return None


def paired_bootstrap_ci(
    differences: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    n = len(differences)

    if n == 0:
        return (
            float("nan"),
            float("nan"),
        )

    rng = np.random.default_rng(
        seed
    )

    indices = rng.integers(
        0,
        n,
        size=(
            replicates,
            n,
        ),
    )

    estimates = (
        differences[indices]
        .mean(axis=1)
    )

    low, high = np.quantile(
        estimates,
        [
            0.025,
            0.975,
        ],
    )

    return (
        float(low),
        float(high),
    )


def compute_contrast(
    task: dict[str, Any],
) -> dict[str, Any]:
    pairs = task["pairs"]

    baseline = np.asarray(
        [
            pair[0]
            for pair in pairs
        ],
        dtype=np.int8,
    )

    pressure = np.asarray(
        [
            pair[1]
            for pair in pairs
        ],
        dtype=np.int8,
    )

    differences = (
        pressure.astype(float)
        - baseline.astype(float)
    )

    n = len(differences)

    baseline_positive = int(
        baseline.sum()
    )

    pressure_positive = int(
        pressure.sum()
    )

    discordant_up = int(
        np.sum(
            (baseline == 0)
            & (pressure == 1)
        )
    )

    discordant_down = int(
        np.sum(
            (baseline == 1)
            & (pressure == 0)
        )
    )

    seed_material = "|".join([
        "semantic-matched-v1",
        task["profile"],
        task["placement"],
        task["pressure_type"],
        task["endpoint"],
    ]).encode("utf-8")

    seed = int.from_bytes(
        hashlib.sha256(
            seed_material
        ).digest()[:8],
        byteorder="big",
        signed=False,
    )

    ci_low, ci_high = (
        paired_bootstrap_ci(
            differences,
            replicates=task[
                "bootstrap_replicates"
            ],
            seed=seed,
        )
    )

    result = {
        key: value
        for key, value in task.items()
        if key != "pairs"
    }

    result.update({
        "paired_n": n,
        "eval_only_positive": (
            baseline_positive
        ),
        "pressure_positive": (
            pressure_positive
        ),
        "eval_only_rate": (
            baseline_positive / n
            if n
            else None
        ),
        "pressure_rate": (
            pressure_positive / n
            if n
            else None
        ),
        "delta_pp": (
            100.0
            * float(
                differences.mean()
            )
            if n
            else None
        ),
        "ci_low_pp": (
            100.0 * ci_low
            if n
            else None
        ),
        "ci_high_pp": (
            100.0 * ci_high
            if n
            else None
        ),
        "discordant_up": (
            discordant_up
        ),
        "discordant_down": (
            discordant_down
        ),
        "discordant_total": (
            discordant_up
            + discordant_down
        ),
        "p_mcnemar": exact_mcnemar(
            discordant_up,
            discordant_down,
        ),
        "bootstrap_seed": seed,
    })

    return result


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--plan",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--final-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--frozen-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--freeze-ledger",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=min(
            8,
            os.cpu_count() or 4,
        ),
    )

    args = parser.parse_args()

    plan_path = (
        args.plan
        .expanduser()
        .resolve()
    )

    final_root = (
        args.final_root
        .expanduser()
        .resolve()
    )

    frozen_root = (
        args.frozen_root
        .expanduser()
        .resolve()
    )

    ledger_path = (
        args.freeze_ledger
        .expanduser()
        .resolve()
    )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    plan = load(
        plan_path
    )

    ledger = load(
        ledger_path
    )

    freeze_path = (
        final_root
        / "freeze_manifest.json"
    )

    if (
        sha256(freeze_path)
        != ledger[
            "freeze_manifest_sha256"
        ]
    ):
        raise ValueError(
            "final semantic freeze "
            "hash mismatch"
        )

    if (
        plan["pairing_key"]
        != "base_task_id"
    ):
        raise ValueError(
            "unexpected pairing key"
        )

    if (
        len(plan["profiles"]) != 4
        or len(
            plan["placements"]
        ) != 3
        or len(
            plan["contrasts"]
        ) != 2
        or len(
            plan["endpoints"]
        ) != 5
    ):
        raise ValueError(
            "inference plan shape "
            "does not imply 120 tests"
        )

    # ------------------------------------------------
    # Resolve each semantic trajectory to its
    # original base_task_id.
    # ------------------------------------------------

    trial_to_base = {}

    for profile in (
        plan["profiles"]
    ):
        path = (
            frozen_root
            / profile
            / "trials.json"
        )

        rows = load(path)

        if len(rows) != 700:
            raise ValueError(
                f"{profile}: "
                "expected 700 rows"
            )

        counts = defaultdict(int)

        for row in rows:
            base_task_id = str(
                row["base_task_id"]
            )

            counts[
                base_task_id
            ] += 1

            key = (
                profile,
                str(
                    row["trial_name"]
                ),
            )

            if (
                key
                in trial_to_base
            ):
                raise ValueError(
                    f"duplicate trial {key}"
                )

            trial_to_base[
                key
            ] = base_task_id

        if (
            len(counts) != 70
            or set(
                counts.values()
            ) != {10}
        ):
            raise ValueError(
                f"{profile}: "
                "base_task_id is not "
                "exact 70x10 pairing"
            )

    # ------------------------------------------------
    # Index final consensus by:
    # profile × task × condition × placement.
    # ------------------------------------------------

    consensus_paths = sorted(
        (
            final_root
            / "consensus"
        ).glob("*.json")
    )

    if (
        len(consensus_paths)
        != 2776
    ):
        raise ValueError(
            "expected 2776 final "
            "consensus files"
        )

    cells = {}

    for path in consensus_paths:
        value = load(path)

        profile = str(
            value["profile"]
        )

        trial_name = str(
            value["trial_name"]
        )

        base_task_id = (
            trial_to_base[
                (
                    profile,
                    trial_name,
                )
            ]
        )

        key = (
            profile,
            base_task_id,
            str(
                value["condition"]
            ),
            str(
                value["placement"]
            ),
        )

        if key in cells:
            raise ValueError(
                f"duplicate semantic "
                f"cell {key}"
            )

        cells[key] = (
            value[
                "consensus"
            ]["fields"]
        )

    # ------------------------------------------------
    # Build the complete frozen 120-test family.
    # ------------------------------------------------

    tasks = []

    all_base_ids = {
        profile: sorted({
            key[1]
            for key in cells
            if key[0] == profile
        })
        for profile
        in plan["profiles"]
    }

    for profile in (
        plan["profiles"]
    ):
        for placement in (
            plan["placements"]
        ):
            for contrast in (
                plan["contrasts"]
            ):
                reference_condition = (
                    contrast[
                        "reference_condition"
                    ]
                )

                pressure_condition = (
                    contrast[
                        "pressure_condition"
                    ]
                )

                for (
                    endpoint,
                    spec,
                ) in (
                    plan[
                        "endpoints"
                    ].items()
                ):
                    pairs = []

                    possible_pairs = 0
                    unresolved_reference = 0
                    unresolved_pressure = 0
                    unresolved_either = 0

                    for base_task_id in (
                        all_base_ids[
                            profile
                        ]
                    ):
                        ref_key = (
                            profile,
                            base_task_id,
                            reference_condition,
                            placement,
                        )

                        pressure_key = (
                            profile,
                            base_task_id,
                            pressure_condition,
                            placement,
                        )

                        # Common infra-censored source
                        # cells may make a pair unavailable.
                        if (
                            ref_key
                            not in cells
                            or pressure_key
                            not in cells
                        ):
                            continue

                        possible_pairs += 1

                        ref_result = (
                            cells[
                                ref_key
                            ][
                                spec[
                                    "field"
                                ]
                            ]
                        )

                        pressure_result = (
                            cells[
                                pressure_key
                            ][
                                spec[
                                    "field"
                                ]
                            ]
                        )

                        ref_value = (
                            binary_value(
                                ref_result,
                                spec,
                            )
                        )

                        pressure_value = (
                            binary_value(
                                pressure_result,
                                spec,
                            )
                        )

                        if (
                            ref_value
                            is None
                        ):
                            unresolved_reference += 1

                        if (
                            pressure_value
                            is None
                        ):
                            unresolved_pressure += 1

                        if (
                            ref_value
                            is None
                            or pressure_value
                            is None
                        ):
                            unresolved_either += 1
                            continue

                        pairs.append(
                            (
                                ref_value,
                                pressure_value,
                            )
                        )

                    tasks.append({
                        "profile": profile,
                        "placement": (
                            placement
                        ),
                        "pressure_type": (
                            contrast[
                                "pressure_type"
                            ]
                        ),
                        "pressure_condition": (
                            pressure_condition
                        ),
                        "reference_condition": (
                            reference_condition
                        ),
                        "endpoint": endpoint,
                        "semantic_field": (
                            spec["field"]
                        ),
                        "possible_matched_pairs": (
                            possible_pairs
                        ),
                        "unresolved_eval_only": (
                            unresolved_reference
                        ),
                        "unresolved_pressure": (
                            unresolved_pressure
                        ),
                        "unresolved_either": (
                            unresolved_either
                        ),
                        "bootstrap_replicates": int(
                            plan[
                                "inference"
                            ][
                                "bootstrap_replicates"
                            ]
                        ),
                        "pairs": pairs,
                    })

    if len(tasks) != 120:
        raise ValueError(
            f"expected 120 tests; "
            f"found {len(tasks)}"
        )

    # Independent bootstrap jobs can run
    # in parallel.
    with ProcessPoolExecutor(
        max_workers=args.workers
    ) as pool:
        results = list(
            pool.map(
                compute_contrast,
                tasks,
            )
        )

    # ------------------------------------------------
    # Multiplicity.
    # ------------------------------------------------

    by_profile = defaultdict(
        list
    )

    for row in results:
        by_profile[
            row["profile"]
        ].append(row)

    for profile in (
        plan["profiles"]
    ):
        rows = by_profile[
            profile
        ]

        if len(rows) != 30:
            raise ValueError(
                f"{profile}: "
                "expected 30 tests"
            )

        holm_adjust(
            rows,
            "p_holm_within_profile",
        )

    holm_adjust(
        results,
        "p_holm_global_120",
    )

    results.sort(
        key=lambda row: (
            row["profile"],
            row["placement"],
            row["pressure_type"],
            row["endpoint"],
        )
    )

    # ------------------------------------------------
    # Write CSV.
    # ------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    columns = [
        "profile",
        "placement",
        "pressure_type",
        "pressure_condition",
        "reference_condition",
        "endpoint",
        "semantic_field",
        "possible_matched_pairs",
        "paired_n",
        "unresolved_eval_only",
        "unresolved_pressure",
        "unresolved_either",
        "eval_only_positive",
        "pressure_positive",
        "eval_only_rate",
        "pressure_rate",
        "delta_pp",
        "ci_low_pp",
        "ci_high_pp",
        "discordant_up",
        "discordant_down",
        "discordant_total",
        "p_mcnemar",
        "p_holm_within_profile",
        "p_holm_global_120",
        "bootstrap_replicates",
        "bootstrap_seed",
    ]

    csv_path = (
        output_dir
        / "matched_semantic_contrasts.csv"
    )

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
        )

        writer.writeheader()

        for row in results:
            writer.writerow({
                column: row.get(
                    column
                )
                for column
                in columns
            })

    within_sig = sum(
        row[
            "p_holm_within_profile"
        ] < 0.05
        for row in results
    )

    global_sig = sum(
        row[
            "p_holm_global_120"
        ] < 0.05
        for row in results
    )

    summary = {
        "analyzer_version": VERSION,
        "plan": str(
            plan_path
        ),
        "plan_sha256": sha256(
            plan_path
        ),
        "final_freeze_sha256": (
            sha256(
                freeze_path
            )
        ),
        "contrast_count": (
            len(results)
        ),
        "workers": (
            args.workers
        ),
        "within_profile_holm_significant": (
            within_sig
        ),
        "global_holm_significant": (
            global_sig
        ),
    }

    (
        output_dir
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    # ------------------------------------------------
    # Human-readable result order:
    # strongest multiplicity-adjusted results first.
    # ------------------------------------------------

    ordered = sorted(
        results,
        key=lambda row: (
            row[
                "p_holm_within_profile"
            ],
            row[
                "p_holm_global_120"
            ],
            row["p_mcnemar"],
            -abs(
                row["delta_pp"]
            ),
        ),
    )

    lines = [
        "# Matched Historical Semantic Inference",
        "",
        "Pressure conditions are compared with eval-only "
        "within the same profile, placement, and base_task_id.",
        "",
        "Consensus disagreement, missingness, and ambiguous "
        "labels are unresolved rather than negative. "
        "Each endpoint therefore uses endpoint-specific "
        "complete matched pairs.",
        "",
        "## Multiplicity",
        "",
        f"- Total tests: {len(results)}",
        "- Primary: Holm within each profile (30 tests/model)",
        "- Sensitivity: Holm across all 120 tests",
        f"- Holm/model significant: {within_sig}",
        f"- Holm/120 significant: {global_sig}",
        "",
        "## Results",
        "",
        "| Model | Place | Pressure | Endpoint | N | Eval-only | Pressure | Delta pp | 95% CI | McNemar p | Holm/model | Holm/120 |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in ordered:
        lines.append(
            "| "
            f"{row['profile']} | "
            f"{row['placement']} | "
            f"{row['pressure_type']} | "
            f"{row['endpoint']} | "
            f"{row['paired_n']} | "
            f"{100 * row['eval_only_rate']:.1f}% | "
            f"{100 * row['pressure_rate']:.1f}% | "
            f"{row['delta_pp']:+.1f} | "
            f"[{row['ci_low_pp']:+.1f}, "
            f"{row['ci_high_pp']:+.1f}] | "
            f"{row['p_mcnemar']:.4g} | "
            f"{row['p_holm_within_profile']:.4g} | "
            f"{row['p_holm_global_120']:.4g} |"
        )

    (
        output_dir
        / "report.md"
    ).write_text(
        "\n".join(lines)
        + "\n",
        encoding="utf-8",
    )

    print(
        "MATCHED SEMANTIC INFERENCE V1"
    )
    print("=" * 80)

    print(
        "plan sha256:",
        sha256(plan_path),
    )

    print(
        "contrasts:",
        len(results),
    )

    print(
        "within-profile Holm significant:",
        within_sig,
    )

    print(
        "global Holm significant:",
        global_sig,
    )

    print(
        "workers:",
        args.workers,
    )

    print()
    print("TOP 30 CONTRASTS")
    print("-" * 80)

    for row in ordered[:30]:
        print(
            f"{row['profile']:7s} "
            f"{row['placement']:8s} "
            f"{row['pressure_type']:18s} "
            f"{row['endpoint']:34s} "
            f"N={row['paired_n']:2d} "
            f"d={row['delta_pp']:+6.1f}pp "
            f"CI=["
            f"{row['ci_low_pp']:+5.1f},"
            f"{row['ci_high_pp']:+5.1f}] "
            f"p={row['p_mcnemar']:.4g} "
            f"HolmM="
            f"{row['p_holm_within_profile']:.4g} "
            f"Holm120="
            f"{row['p_holm_global_120']:.4g}"
        )

    print()
    print("network calls: 0")
    print(
        "MATCHED SEMANTIC INFERENCE: PASS"
    )


if __name__ == "__main__":
    main()
