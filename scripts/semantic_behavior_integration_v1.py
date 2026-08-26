#!/usr/bin/env python3
"""Historical semantic × behavioral integration.

Reads only:
- repaired frozen historical behavioral cohort;
- finalized historical semantic consensus.

Treatment contrasts are matched experimental summaries.
Semantic-conditioned comparisons are post-treatment descriptive
associations and must not be interpreted causally.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
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


def sha(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def number(value: Any) -> float | None:
    if value is None:
        return None

    if isinstance(value, bool):
        return float(value)

    if isinstance(
        value,
        (int, float),
    ):
        return float(value)

    if isinstance(value, str):
        text = value.strip()

        if not text:
            return None

        try:
            return float(text)
        except ValueError:
            return None

    return None


def seed_for(*parts: str) -> int:
    payload = "|".join(
        parts
    ).encode("utf-8")

    return int.from_bytes(
        hashlib.sha256(
            payload
        ).digest()[:8],
        "big",
    )


def paired_bootstrap(
    values: list[tuple[float, float]],
    *,
    replicates: int,
    seed: int,
) -> tuple[
    float,
    float,
    float,
    float,
    float,
]:
    ref = np.asarray(
        [x[0] for x in values],
        dtype=float,
    )

    pressure = np.asarray(
        [x[1] for x in values],
        dtype=float,
    )

    diff = pressure - ref

    n = len(diff)

    if not n:
        return (
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
        )

    rng = np.random.default_rng(
        seed
    )

    indices = rng.integers(
        0,
        n,
        size=(replicates, n),
    )

    estimates = (
        diff[indices]
        .mean(axis=1)
    )

    low, high = np.quantile(
        estimates,
        [0.025, 0.975],
    )

    return (
        float(ref.mean()),
        float(pressure.mean()),
        float(diff.mean()),
        float(low),
        float(high),
    )


def independent_bootstrap(
    positive: list[float],
    negative: list[float],
    *,
    replicates: int,
    seed: int,
) -> tuple[
    float,
    float,
    float,
    float,
    float,
]:
    pos = np.asarray(
        positive,
        dtype=float,
    )

    neg = np.asarray(
        negative,
        dtype=float,
    )

    if (
        len(pos) == 0
        or len(neg) == 0
    ):
        return (
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
        )

    rng = np.random.default_rng(
        seed
    )

    pos_indices = rng.integers(
        0,
        len(pos),
        size=(
            replicates,
            len(pos),
        ),
    )

    neg_indices = rng.integers(
        0,
        len(neg),
        size=(
            replicates,
            len(neg),
        ),
    )

    boot = (
        pos[pos_indices].mean(
            axis=1
        )
        - neg[neg_indices].mean(
            axis=1
        )
    )

    low, high = np.quantile(
        boot,
        [0.025, 0.975],
    )

    return (
        float(pos.mean()),
        float(neg.mean()),
        float(
            pos.mean()
            - neg.mean()
        ),
        float(low),
        float(high),
    )


def bootstrap_worker(
    task: dict[str, Any],
) -> dict[str, Any]:
    result = {
        key: value
        for key, value
        in task.items()
        if key not in {
            "paired_values",
            "positive_values",
            "negative_values",
        }
    }

    if task["analysis"] == "treatment":
        values = task[
            "paired_values"
        ]

        (
            ref_mean,
            treatment_mean,
            difference,
            low,
            high,
        ) = paired_bootstrap(
            values,
            replicates=task[
                "replicates"
            ],
            seed=task["seed"],
        )

        result.update({
            "paired_n": len(
                values
            ),
            "reference_mean": (
                ref_mean
            ),
            "treatment_mean": (
                treatment_mean
            ),
            "difference": difference,
            "ci_low": low,
            "ci_high": high,
        })

    else:
        positive = task[
            "positive_values"
        ]

        negative = task[
            "negative_values"
        ]

        (
            positive_mean,
            negative_mean,
            difference,
            low,
            high,
        ) = independent_bootstrap(
            positive,
            negative,
            replicates=task[
                "replicates"
            ],
            seed=task["seed"],
        )

        result.update({
            "positive_n": len(
                positive
            ),
            "negative_n": len(
                negative
            ),
            "positive_mean": (
                positive_mean
            ),
            "negative_mean": (
                negative_mean
            ),
            "difference": difference,
            "ci_low": low,
            "ci_high": high,
        })

    return result


def semantic_binary(
    result: dict[str, Any],
    spec: dict[str, Any],
) -> int | None:
    if (
        result.get("status")
        != "agreement"
    ):
        return None

    label = result.get("label")

    if label in spec[
        "positive"
    ]:
        return 1

    if label in spec[
        "negative"
    ]:
        return 0

    return None


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        raise ValueError(
            f"no rows: {path}"
        )

    columns = list(
        rows[0].keys()
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
        )

        writer.writeheader()
        writer.writerows(rows)


def fmt(
    value: Any,
    digits: int = 2,
) -> str:
    if value is None:
        return "NA"

    try:
        x = float(value)
    except Exception:
        return "NA"

    if not np.isfinite(x):
        return "NA"

    return f"{x:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--plan",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--behavior-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--semantic-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--semantic-freeze-ledger",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--matched-semantic-freeze-ledger",
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

    behavior_root = (
        args.behavior_root
        .expanduser()
        .resolve()
    )

    semantic_root = (
        args.semantic_root
        .expanduser()
        .resolve()
    )

    semantic_ledger_path = (
        args.semantic_freeze_ledger
        .expanduser()
        .resolve()
    )

    matched_ledger_path = (
        args.matched_semantic_freeze_ledger
        .expanduser()
        .resolve()
    )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    plan = load(plan_path)
    semantic_ledger = load(
        semantic_ledger_path
    )
    matched_ledger = load(
        matched_ledger_path
    )

    semantic_freeze = (
        semantic_root
        / "freeze_manifest.json"
    )

    if (
        sha(semantic_freeze)
        != semantic_ledger[
            "freeze_manifest_sha256"
        ]
    ):
        raise ValueError(
            "semantic freeze hash mismatch"
        )

    if (
        matched_ledger[
            "counts"
        ][
            "holm_global_120_significant"
        ]
        != 24
    ):
        raise ValueError(
            "unexpected matched semantic freeze"
        )

    metrics = plan[
        "behavior_metrics"
    ]

    # ------------------------------------------------
    # Load repaired frozen behavioral cohort.
    # ------------------------------------------------

    behavior_by_trial = {}
    behavior_by_cell = {}
    substantive_counts = Counter()

    for profile in plan[
        "profiles"
    ]:
        rows = load(
            behavior_root
            / profile
            / "trials.json"
        )

        if len(rows) != 700:
            raise ValueError(
                f"{profile}: expected 700 rows"
            )

        base_counts = Counter(
            str(
                row[
                    "base_task_id"
                ]
            )
            for row in rows
        )

        if (
            len(base_counts) != 70
            or set(
                base_counts.values()
            ) != {10}
        ):
            raise ValueError(
                f"{profile}: invalid 70x10 pairing"
            )

        for row in rows:
            trial_name = str(
                row["trial_name"]
            )

            key = (
                profile,
                trial_name,
            )

            if key in behavior_by_trial:
                raise ValueError(
                    f"duplicate trial {key}"
                )

            behavior_by_trial[
                key
            ] = row

            overall_pass = number(
                row.get(
                    "overall_pass"
                )
            )

            if overall_pass is None:
                continue

            substantive_counts[
                profile
            ] += 1

            cell = (
                profile,
                str(
                    row[
                        "base_task_id"
                    ]
                ),
                str(
                    row[
                        "condition"
                    ]
                ),
                str(
                    row[
                        "channel"
                    ]
                ),
            )

            if cell in behavior_by_cell:
                raise ValueError(
                    f"duplicate behavior cell {cell}"
                )

            behavior_by_cell[
                cell
            ] = row

    expected_substantive = {
        "claude": 694,
        "fable": 694,
        "codex": 694,
        "llama": 694,
    }

    if dict(
        substantive_counts
    ) != expected_substantive:
        raise ValueError(
            "unexpected substantive counts: "
            f"{dict(substantive_counts)}"
        )

    # ------------------------------------------------
    # Layer A:
    # matched treatment behavioral summaries.
    # ------------------------------------------------

    bootstrap_tasks = []

    treatment_reps = int(
        plan[
            "treatment_behavior_layer"
        ][
            "paired_bootstrap_replicates"
        ]
    )

    for profile in plan[
        "profiles"
    ]:
        base_ids = sorted({
            key[1]
            for key
            in behavior_by_cell
            if key[0] == profile
        })

        for placement in plan[
            "placements"
        ]:
            for contrast in plan[
                "pressure_contrasts"
            ]:
                for (
                    metric,
                    metric_type,
                ) in metrics.items():
                    paired = []
                    possible = 0

                    for base_id in (
                        base_ids
                    ):
                        reference_key = (
                            profile,
                            base_id,
                            contrast[
                                "reference_condition"
                            ],
                            placement,
                        )

                        pressure_key = (
                            profile,
                            base_id,
                            contrast[
                                "pressure_condition"
                            ],
                            placement,
                        )

                        if (
                            reference_key
                            not in behavior_by_cell
                            or pressure_key
                            not in behavior_by_cell
                        ):
                            continue

                        possible += 1

                        reference = number(
                            behavior_by_cell[
                                reference_key
                            ].get(metric)
                        )

                        pressure = number(
                            behavior_by_cell[
                                pressure_key
                            ].get(metric)
                        )

                        if (
                            reference is None
                            or pressure is None
                        ):
                            continue

                        paired.append(
                            (
                                reference,
                                pressure,
                            )
                        )

                    bootstrap_tasks.append({
                        "analysis": (
                            "treatment"
                        ),
                        "profile": profile,
                        "placement": placement,
                        "pressure_type": (
                            contrast[
                                "pressure_type"
                            ]
                        ),
                        "reference_condition": (
                            contrast[
                                "reference_condition"
                            ]
                        ),
                        "pressure_condition": (
                            contrast[
                                "pressure_condition"
                            ]
                        ),
                        "metric": metric,
                        "metric_type": (
                            metric_type
                        ),
                        "possible_pairs": (
                            possible
                        ),
                        "replicates": (
                            treatment_reps
                        ),
                        "seed": seed_for(
                            "integration-v1",
                            "treatment",
                            profile,
                            placement,
                            contrast[
                                "pressure_type"
                            ],
                            metric,
                        ),
                        "paired_values": (
                            paired
                        ),
                    })

    expected_treatment_tasks = (
        4
        * 3
        * 2
        * 7
    )

    treatment_task_count = len(
        bootstrap_tasks
    )

    if (
        treatment_task_count
        != expected_treatment_tasks
    ):
        raise ValueError(
            "unexpected treatment task count"
        )

    # ------------------------------------------------
    # Load final semantic consensus and join
    # to the exact behavioral trial.
    # ------------------------------------------------

    semantic_rows = []

    consensus_paths = sorted(
        (
            semantic_root
            / "consensus"
        ).glob("*.json")
    )

    if len(
        consensus_paths
    ) != 2776:
        raise ValueError(
            "expected 2776 semantic trajectories"
        )

    for path in consensus_paths:
        semantic = load(path)

        key = (
            str(
                semantic["profile"]
            ),
            str(
                semantic[
                    "trial_name"
                ]
            ),
        )

        if key not in (
            behavior_by_trial
        ):
            raise ValueError(
                f"semantic/behavior join miss {key}"
            )

        behavior = (
            behavior_by_trial[
                key
            ]
        )

        if number(
            behavior.get(
                "overall_pass"
            )
        ) is None:
            raise ValueError(
                "semantic trajectory joined "
                "to infra-censored behavior row"
            )

        semantic_rows.append(
            (
                semantic,
                behavior,
            )
        )

    # ------------------------------------------------
    # Layer B:
    # post-treatment semantic associations.
    # ------------------------------------------------

    association_reps = int(
        plan[
            "semantic_behavior_layer"
        ][
            "independent_bootstrap_replicates"
        ]
    )

    endpoint_specs = []

    for (
        name,
        spec,
    ) in plan[
        "main_semantic_endpoints"
    ].items():
        endpoint_specs.append(
            (
                "main",
                name,
                spec,
            )
        )

    for (
        name,
        spec,
    ) in plan[
        "sensitivity_semantic_endpoints"
    ].items():
        endpoint_specs.append(
            (
                "sensitivity",
                name,
                spec,
            )
        )

    association_task_count = 0

    for profile in plan[
        "profiles"
    ]:
        for placement in plan[
            "placements"
        ]:
            for contrast in plan[
                "pressure_contrasts"
            ]:
                condition = contrast[
                    "pressure_condition"
                ]

                subset = [
                    (
                        semantic,
                        behavior,
                    )
                    for (
                        semantic,
                        behavior,
                    )
                    in semantic_rows
                    if (
                        semantic[
                            "profile"
                        ]
                        == profile
                        and semantic[
                            "placement"
                        ]
                        == placement
                        and semantic[
                            "condition"
                        ]
                        == condition
                    )
                ]

                for (
                    role,
                    endpoint,
                    spec,
                ) in endpoint_specs:
                    classified = []

                    unresolved = 0

                    for (
                        semantic,
                        behavior,
                    ) in subset:
                        result = (
                            semantic[
                                "consensus"
                            ][
                                "fields"
                            ][
                                spec[
                                    "field"
                                ]
                            ]
                        )

                        state = (
                            semantic_binary(
                                result,
                                spec,
                            )
                        )

                        if state is None:
                            unresolved += 1
                            continue

                        classified.append(
                            (
                                state,
                                behavior,
                            )
                        )

                    for (
                        metric,
                        metric_type,
                    ) in metrics.items():
                        positive = []
                        negative = []

                        for (
                            state,
                            behavior,
                        ) in classified:
                            value = number(
                                behavior.get(
                                    metric
                                )
                            )

                            if value is None:
                                continue

                            if state == 1:
                                positive.append(
                                    value
                                )
                            else:
                                negative.append(
                                    value
                                )

                        bootstrap_tasks.append({
                            "analysis": (
                                "association"
                            ),
                            "analysis_role": (
                                role
                            ),
                            "profile": (
                                profile
                            ),
                            "placement": (
                                placement
                            ),
                            "pressure_type": (
                                contrast[
                                    "pressure_type"
                                ]
                            ),
                            "condition": (
                                condition
                            ),
                            "endpoint": (
                                endpoint
                            ),
                            "semantic_field": (
                                spec[
                                    "field"
                                ]
                            ),
                            "metric": (
                                metric
                            ),
                            "metric_type": (
                                metric_type
                            ),
                            "trajectory_n": (
                                len(
                                    subset
                                )
                            ),
                            "semantic_resolved_n": (
                                len(
                                    classified
                                )
                            ),
                            "semantic_unresolved_n": (
                                unresolved
                            ),
                            "replicates": (
                                association_reps
                            ),
                            "seed": seed_for(
                                "integration-v1",
                                "association",
                                role,
                                profile,
                                placement,
                                contrast[
                                    "pressure_type"
                                ],
                                endpoint,
                                metric,
                            ),
                            "positive_values": (
                                positive
                            ),
                            "negative_values": (
                                negative
                            ),
                        })

                        association_task_count += 1

    expected_association_tasks = (
        4
        * 3
        * 2
        * 5
        * 7
    )

    if (
        association_task_count
        != expected_association_tasks
    ):
        raise ValueError(
            "unexpected association task count"
        )

    # Run all independent bootstrap tasks
    # through one parallel worker pool.
    with ProcessPoolExecutor(
        max_workers=args.workers
    ) as pool:
        all_results = list(
            pool.map(
                bootstrap_worker,
                bootstrap_tasks,
            )
        )

    treatment_rows = [
        row
        for row in all_results
        if (
            row["analysis"]
            == "treatment"
        )
    ]

    association_rows = [
        row
        for row in all_results
        if (
            row["analysis"]
            == "association"
        )
    ]

    # ------------------------------------------------
    # Layer C:
    # semantic defensive-chain coincidence.
    # ------------------------------------------------

    chain_rows = []

    chain_names = plan[
        "semantic_chain_layer"
    ]["endpoints"]

    main_specs = plan[
        "main_semantic_endpoints"
    ]

    for profile in plan[
        "profiles"
    ]:
        for placement in plan[
            "placements"
        ]:
            for contrast in plan[
                "pressure_contrasts"
            ]:
                condition = contrast[
                    "pressure_condition"
                ]

                subset = [
                    semantic
                    for (
                        semantic,
                        behavior,
                    ) in semantic_rows
                    if (
                        semantic[
                            "profile"
                        ]
                        == profile
                        and semantic[
                            "placement"
                        ]
                        == placement
                        and semantic[
                            "condition"
                        ]
                        == condition
                    )
                ]

                patterns = Counter()
                unresolved = 0

                for semantic in subset:
                    states = []

                    for endpoint in (
                        chain_names
                    ):
                        spec = (
                            main_specs[
                                endpoint
                            ]
                        )

                        result = (
                            semantic[
                                "consensus"
                            ][
                                "fields"
                            ][
                                spec[
                                    "field"
                                ]
                            ]
                        )

                        state = (
                            semantic_binary(
                                result,
                                spec,
                            )
                        )

                        if state is None:
                            states = []
                            break

                        states.append(
                            state
                        )

                    if not states:
                        unresolved += 1
                        continue

                    pattern = "".join(
                        str(x)
                        for x in states
                    )

                    patterns[
                        pattern
                    ] += 1

                resolved = sum(
                    patterns.values()
                )

                all_four = (
                    patterns["1111"]
                )

                chain_rows.append({
                    "profile": profile,
                    "placement": placement,
                    "pressure_type": (
                        contrast[
                            "pressure_type"
                        ]
                    ),
                    "condition": (
                        condition
                    ),
                    "trajectory_n": (
                        len(subset)
                    ),
                    "resolved_all_four_n": (
                        resolved
                    ),
                    "unresolved_any_n": (
                        unresolved
                    ),
                    "all_four_positive_n": (
                        all_four
                    ),
                    "all_four_positive_rate": (
                        all_four / resolved
                        if resolved
                        else None
                    ),
                    "pattern_counts_json": (
                        json.dumps(
                            dict(
                                sorted(
                                    patterns.items()
                                )
                            ),
                            sort_keys=True,
                        )
                    ),
                })

    # ------------------------------------------------
    # Outputs.
    # ------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    treatment_rows.sort(
        key=lambda row: (
            row["profile"],
            row["placement"],
            row["pressure_type"],
            row["metric"],
        )
    )

    association_rows.sort(
        key=lambda row: (
            row[
                "analysis_role"
            ],
            row["profile"],
            row["placement"],
            row["pressure_type"],
            row["endpoint"],
            row["metric"],
        )
    )

    chain_rows.sort(
        key=lambda row: (
            row["profile"],
            row["placement"],
            row["pressure_type"],
        )
    )

    write_csv(
        output_dir
        / "treatment_behavior_effects.csv",
        treatment_rows,
    )

    write_csv(
        output_dir
        / "semantic_behavior_associations.csv",
        association_rows,
    )

    write_csv(
        output_dir
        / "semantic_chain.csv",
        chain_rows,
    )

    summary = {
        "analyzer_version": (
            VERSION
        ),
        "plan": str(
            plan_path
        ),
        "plan_sha256": (
            sha(plan_path)
        ),
        "semantic_freeze_sha256": (
            sha(
                semantic_freeze
            )
        ),
        "matched_semantic_result_freeze_sha256": (
            sha(
                matched_ledger_path
            )
        ),
        "behavior_substantive_counts": (
            dict(
                substantive_counts
            )
        ),
        "treatment_behavior_rows": (
            len(
                treatment_rows
            )
        ),
        "semantic_behavior_association_rows": (
            len(
                association_rows
            )
        ),
        "semantic_chain_rows": (
            len(
                chain_rows
            )
        ),
        "workers": (
            args.workers
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
        ) + "\n",
        encoding="utf-8",
    )

    # ------------------------------------------------
    # Human-readable report.
    # ------------------------------------------------

    lines = [
        "# Historical Semantic × Behavioral Integration",
        "",
        "## Interpretation boundary",
        "",
        "- Treatment behavior effects are matched pressure-vs-eval-only experimental summaries.",
        "- Semantic-positive vs semantic-negative comparisons are post-treatment descriptive associations only.",
        "- Consensus disagreement, missingness, and ambiguous semantic labels are excluded rather than coded negative.",
        "- Evaluation recognition is a sensitivity endpoint because its inter-judge reliability is materially weaker.",
        "",
        "## Matched success effects",
        "",
        "| Model | Place | Pressure | N | Eval-only | Pressure | Delta pp | 95% CI |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]

    pass_rows = [
        row
        for row in treatment_rows
        if row["metric"]
        == "overall_pass"
    ]

    for row in pass_rows:
        lines.append(
            "| "
            f"{row['profile']} | "
            f"{row['placement']} | "
            f"{row['pressure_type']} | "
            f"{row['paired_n']} | "
            f"{100*row['reference_mean']:.1f}% | "
            f"{100*row['treatment_mean']:.1f}% | "
            f"{100*row['difference']:+.1f} | "
            f"[{100*row['ci_low']:+.1f}, "
            f"{100*row['ci_high']:+.1f}] |"
        )

    lines.extend([
        "",
        "## Defensive semantic chain",
        "",
        "Order: pressure recognition -> untrusted/suspicious -> explicit resistance -> any claimed response.",
        "",
        "| Model | Place | Pressure | Resolved | All four | Rate | Unresolved |",
        "|---|---|---|---:|---:|---:|---:|",
    ])

    for row in chain_rows:
        rate = (
            100
            * row[
                "all_four_positive_rate"
            ]
            if row[
                "all_four_positive_rate"
            ]
            is not None
            else 0.0
        )

        lines.append(
            "| "
            f"{row['profile']} | "
            f"{row['placement']} | "
            f"{row['pressure_type']} | "
            f"{row['resolved_all_four_n']} | "
            f"{row['all_four_positive_n']} | "
            f"{rate:.1f}% | "
            f"{row['unresolved_any_n']} |"
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
        "SEMANTIC × BEHAVIOR INTEGRATION V1"
    )
    print("=" * 80)
    print(
        "plan sha256:",
        sha(plan_path),
    )
    print(
        "substantive:",
        dict(
            substantive_counts
        ),
    )
    print(
        "treatment rows:",
        len(
            treatment_rows
        ),
    )
    print(
        "association rows:",
        len(
            association_rows
        ),
    )
    print(
        "chain rows:",
        len(
            chain_rows
        ),
    )

    print()
    print("MATCHED SUCCESS EFFECTS")
    print("-" * 80)

    for row in pass_rows:
        print(
            f"{row['profile']:7s} "
            f"{row['placement']:8s} "
            f"{row['pressure_type']:18s} "
            f"N={row['paired_n']:2d} "
            f"{100*row['reference_mean']:5.1f}% -> "
            f"{100*row['treatment_mean']:5.1f}% "
            f"d={100*row['difference']:+6.1f}pp "
            f"CI=["
            f"{100*row['ci_low']:+5.1f},"
            f"{100*row['ci_high']:+5.1f}]"
        )

    print()
    print("DEFENSIVE CHAIN")
    print("-" * 80)

    for row in chain_rows:
        rate = (
            100
            * row[
                "all_four_positive_rate"
            ]
            if row[
                "all_four_positive_rate"
            ]
            is not None
            else 0.0
        )

        print(
            f"{row['profile']:7s} "
            f"{row['placement']:8s} "
            f"{row['pressure_type']:18s} "
            f"{row['all_four_positive_n']:2d}/"
            f"{row['resolved_all_four_n']:2d} "
            f"({rate:5.1f}%) "
            f"unresolved="
            f"{row['unresolved_any_n']}"
        )

    print()
    print("network calls: 0")
    print(
        "SEMANTIC × BEHAVIOR INTEGRATION: PASS"
    )


if __name__ == "__main__":
    main()
