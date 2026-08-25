#!/usr/bin/env python3
"""Secondary resource-deprivation vs clean success contrast.

This supplements, but does not modify, the frozen primary resource
inference where eval_resource_deprivation vs eval_only is primary.
"""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Any


VERSION = "resource-vs-clean-1.0"
SEED = 20260825
BOOTSTRAP_REPS = 20_000

PROFILES = (
    "claude",
    "fable",
    "codex",
    "llama",
)

CAPABLE = (
    "claude",
    "fable",
    "codex",
)


def outcome(value: Any) -> int | None:
    if value is None:
        return None

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, float)):
        return int(float(value) > 0)

    text = str(value).strip().lower()

    if text in {
        "true",
        "yes",
        "1",
        "1.0",
    }:
        return 1

    if text in {
        "false",
        "no",
        "0",
        "0.0",
    }:
        return 0

    return None


def percentile(
    values: list[float],
    q: float,
) -> float:
    values = sorted(values)

    if not values:
        return float("nan")

    pos = q * (
        len(values) - 1
    )

    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))

    if lo == hi:
        return values[lo]

    frac = pos - lo

    return (
        values[lo]
        * (1.0 - frac)
        + values[hi]
        * frac
    )


def bootstrap_ci(
    diffs: list[float],
    seed: int,
) -> tuple[float, float]:
    rng = random.Random(seed)

    n = len(diffs)

    if n == 0:
        return (
            float("nan"),
            float("nan"),
        )

    boot = []

    for _ in range(
        BOOTSTRAP_REPS
    ):
        boot.append(
            sum(
                diffs[
                    rng.randrange(n)
                ]
                for _ in range(n)
            )
            / n
        )

    return (
        percentile(
            boot,
            0.025,
        ),
        percentile(
            boot,
            0.975,
        ),
    )


def mcnemar_exact(
    fail_to_pass: int,
    pass_to_fail: int,
) -> float:
    n = (
        fail_to_pass
        + pass_to_fail
    )

    if n == 0:
        return 1.0

    k = min(
        fail_to_pass,
        pass_to_fail,
    )

    tail = sum(
        math.comb(n, i)
        for i in range(
            k + 1
        )
    ) / (
        2 ** n
    )

    return min(
        1.0,
        2.0 * tail,
    )


def holm(
    pvalues: list[float],
) -> list[float]:
    n = len(pvalues)

    order = sorted(
        range(n),
        key=lambda i: (
            pvalues[i]
        ),
    )

    adjusted = [
        1.0
        for _ in pvalues
    ]

    running = 0.0

    for rank, idx in enumerate(
        order
    ):
        value = min(
            1.0,
            (
                n - rank
            )
            * pvalues[idx],
        )

        running = max(
            running,
            value,
        )

        adjusted[idx] = running

    return adjusted


def matched(
    path: Path,
) -> list[tuple[int, int]]:
    rows = json.loads(
        path.read_text()
    )

    by_task = {}

    for row in rows:
        if not row.get(
            "substantive_usable"
        ):
            continue

        condition = str(
            row.get(
                "condition",
                "",
            )
        )

        if condition not in {
            "clean",
            "eval_resource_deprivation",
        }:
            continue

        base = str(
            row.get(
                "base_task_id",
                "",
            )
        )

        if not base:
            raise RuntimeError(
                "missing base_task_id"
            )

        value = outcome(
            row.get(
                "overall_pass"
            )
        )

        if value is None:
            raise RuntimeError(
                f"{base} {condition}: "
                "missing overall_pass"
            )

        by_task.setdefault(
            base,
            {},
        )

        if condition in by_task[
            base
        ]:
            raise RuntimeError(
                f"duplicate {base} "
                f"{condition}"
            )

        by_task[
            base
        ][condition] = value

    pairs = []

    for base in sorted(
        by_task
    ):
        cells = by_task[
            base
        ]

        if (
            "clean" in cells
            and (
                "eval_resource_deprivation"
                in cells
            )
        ):
            pairs.append(
                (
                    cells["clean"],
                    cells[
                        "eval_resource_deprivation"
                    ],
                )
            )

    return pairs


def main():
    data = (
        Path.home()
        / "Documents"
        / "swe-eval-pressure"
    )

    out_dir = (
        data
        / "analysis"
        / "resource"
        / "inference"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    for i, profile in enumerate(
        PROFILES
    ):
        path = (
            data
            / "analysis"
            / "resource"
            / profile
            / "trials.json"
        )

        pairs = matched(path)

        n = len(pairs)

        clean_pass = sum(
            a
            for a, _ in pairs
        )

        resource_pass = sum(
            b
            for _, b in pairs
        )

        diffs = [
            100.0 * (
                b - a
            )
            for a, b in pairs
        ]

        delta = (
            sum(diffs) / n
            if n
            else float("nan")
        )

        ci_lo, ci_hi = (
            bootstrap_ci(
                diffs,
                SEED + i,
            )
        )

        fail_to_pass = sum(
            a == 0 and b == 1
            for a, b in pairs
        )

        pass_to_fail = sum(
            a == 1 and b == 0
            for a, b in pairs
        )

        p = mcnemar_exact(
            fail_to_pass,
            pass_to_fail,
        )

        results.append(
            {
                "profile": profile,
                "contrast": (
                    "resource_vs_clean"
                ),
                "n_pairs": n,
                "clean_pass": clean_pass,
                "resource_pass": (
                    resource_pass
                ),
                "clean_pass_pct": (
                    100.0
                    * clean_pass
                    / n
                    if n
                    else float("nan")
                ),
                "resource_pass_pct": (
                    100.0
                    * resource_pass
                    / n
                    if n
                    else float("nan")
                ),
                "delta_pp": delta,
                "bootstrap_ci_low_pp": (
                    ci_lo
                ),
                "bootstrap_ci_high_pp": (
                    ci_hi
                ),
                "fail_to_pass": (
                    fail_to_pass
                ),
                "pass_to_fail": (
                    pass_to_fail
                ),
                "discordant_pairs": (
                    fail_to_pass
                    + pass_to_fail
                ),
                "mcnemar_exact_p": p,
                "mcnemar_holm_p": "",
                "inferential_role": (
                    "secondary_inferential"
                    if profile
                    in CAPABLE
                    else (
                        "descriptive_"
                        "exposure_unverified"
                    )
                ),
            }
        )

    capable_indices = [
        i
        for i, row
        in enumerate(results)
        if row["profile"]
        in CAPABLE
    ]

    adjusted = holm(
        [
            results[i][
                "mcnemar_exact_p"
            ]
            for i in capable_indices
        ]
    )

    for idx, p_adj in zip(
        capable_indices,
        adjusted,
    ):
        results[idx][
            "mcnemar_holm_p"
        ] = p_adj

    csv_path = (
        out_dir
        / "resource_vs_clean.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                results[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            results
        )

    report = [
        "# Resource deprivation vs clean",
        "",
        (
            "Secondary matched success "
            "contrast. The frozen primary "
            "contrast remains resource vs "
            "eval-only."
        ),
        "",
        (
            "| Model | N | Clean | Resource | "
            "Δ pp | 95% bootstrap CI | "
            "McNemar p | Holm p |"
        ),
        (
            "|---|---:|---:|---:|---:|---:|"
            "---:|---:|"
        ),
    ]

    for row in results:
        holm_value = (
            row[
                "mcnemar_holm_p"
            ]
        )

        holm_text = (
            f"{holm_value:.6g}"
            if holm_value != ""
            else "descriptive"
        )

        report.append(
            "| "
            f"{row['profile']} | "
            f"{row['n_pairs']} | "
            f"{row['clean_pass_pct']:.1f}% | "
            f"{row['resource_pass_pct']:.1f}% | "
            f"{row['delta_pp']:+.2f} | "
            f"[{row['bootstrap_ci_low_pp']:.2f}, "
            f"{row['bootstrap_ci_high_pp']:.2f}] | "
            f"{row['mcnemar_exact_p']:.6g} | "
            f"{holm_text} |"
        )

    report.extend(
        [
            "",
            (
                "Claude/Fable/Codex form the "
                "three-test Holm family. "
                "Llama is descriptive only."
            ),
            "",
            f"Bootstrap reps: {BOOTSTRAP_REPS}.",
            f"Seed: {SEED}.",
        ]
    )

    report_path = (
        out_dir
        / "resource_vs_clean.md"
    )

    report_path.write_text(
        "\n".join(report)
        + "\n",
        encoding="utf-8",
    )

    print(
        "RESOURCE VS CLEAN SECONDARY INFERENCE"
    )
    print("=" * 78)

    for row in results:
        print()
        print(row["profile"])
        print(
            " N:",
            row["n_pairs"],
        )
        print(
            " clean:",
            f"{row['clean_pass_pct']:.2f}%",
        )
        print(
            " resource:",
            f"{row['resource_pass_pct']:.2f}%",
        )
        print(
            " delta:",
            f"{row['delta_pp']:+.2f} pp",
        )
        print(
            " CI:",
            (
                f"[{row['bootstrap_ci_low_pp']:.2f}, "
                f"{row['bootstrap_ci_high_pp']:.2f}]"
            ),
        )
        print(
            " McNemar p:",
            row[
                "mcnemar_exact_p"
            ],
        )
        print(
            " Holm p:",
            row[
                "mcnemar_holm_p"
            ],
        )

    print()
    print(
        "csv:",
        csv_path,
    )
    print(
        "report:",
        report_path,
    )
    print(
        "RESOURCE VS CLEAN "
        "SECONDARY INFERENCE: PASS"
    )


if __name__ == "__main__":
    main()
