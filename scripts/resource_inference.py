#!/usr/bin/env python3
"""Paired inference for the frozen resource-deprivation study.

Primary causal contrast:
    eval_resource_deprivation vs eval_only

Primary success family:
    exact McNemar tests across capable models
    (Claude, Fable, Codex), Holm corrected.

Secondary:
    eval_only vs clean;
    Cochran Q across clean/eval_only/eval_resource_deprivation.

Exploratory process family:
    paired sign-flip tests for raw tool calls,
    input tokens, and validation commands on the
    primary resource contrast; Holm corrected across
    3 capable models x 3 metrics.

Llama is reported descriptively because of its
task-success capability floor.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import subprocess
from pathlib import Path
from typing import Any


VERSION = "1.0"
BOOTSTRAP_REPS = 20_000
SIGNFLIP_REPS = 50_000
SEED = 20260825

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

PRIMARY_BASELINE = "eval_only"
PRIMARY_TREATMENT = "eval_resource_deprivation"

PROCESS_METRICS = {
    "raw_tool_calls": "delta_raw_tool_calls",
    "input_tokens": "delta_input_tokens",
    "validation_command_calls": "delta_validation_command_calls",
}


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {
        "1",
        "true",
        "yes",
    }


def number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)
    return h.hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def holm(pvalues: list[float]) -> list[float]:
    n = len(pvalues)
    if n == 0:
        return []

    order = sorted(
        range(n),
        key=lambda i: pvalues[i],
    )

    out = [1.0] * n
    running = 0.0

    for rank, idx in enumerate(order):
        adjusted = min(
            1.0,
            (n - rank) * pvalues[idx],
        )
        running = max(
            running,
            adjusted,
        )
        out[idx] = running

    return out


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
        for i in range(k + 1)
    ) / (2 ** n)

    return min(
        1.0,
        2.0 * tail,
    )


def percentile(
    values: list[float],
    q: float,
) -> float:
    if not values:
        return float("nan")

    values = sorted(values)

    index = min(
        len(values) - 1,
        max(
            0,
            int(
                q * len(values)
            ),
        ),
    )

    return values[index]


def bootstrap_mean_ci(
    diffs: list[float],
    *,
    seed: int,
) -> tuple[float, float]:
    if not diffs:
        return (
            float("nan"),
            float("nan"),
        )

    rng = random.Random(seed)
    n = len(diffs)

    samples = []

    for _ in range(
        BOOTSTRAP_REPS
    ):
        total = 0.0

        for _ in range(n):
            total += diffs[
                rng.randrange(n)
            ]

        samples.append(
            total / n
        )

    return (
        percentile(
            samples,
            0.025,
        ),
        percentile(
            samples,
            0.975,
        ),
    )


def signflip_p(
    diffs: list[float],
    *,
    seed: int,
) -> float:
    vals = [
        float(x)
        for x in diffs
        if math.isfinite(
            float(x)
        )
    ]

    if not vals:
        return 1.0

    observed = abs(
        sum(vals) / len(vals)
    )

    if observed == 0:
        return 1.0

    rng = random.Random(seed)

    extreme = 0

    for _ in range(
        SIGNFLIP_REPS
    ):
        total = 0.0

        for value in vals:
            total += (
                value
                if rng.getrandbits(1)
                else -value
            )

        statistic = abs(
            total / len(vals)
        )

        if (
            statistic
            >= observed - 1e-12
        ):
            extreme += 1

    return (
        extreme + 1
    ) / (
        SIGNFLIP_REPS + 1
    )


def read_csv(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        newline="",
        encoding="utf-8",
    ) as f:
        return list(
            csv.DictReader(f)
        )


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        path.write_text("")
        return

    fields = list(
        rows[0].keys()
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)


def matched_rows(
    rows: list[dict[str, str]],
    baseline: str,
    treatment: str,
) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if (
            row.get(
                "baseline_condition"
            )
            == baseline
            and row.get(
                "treatment_condition"
            )
            == treatment
            and as_bool(
                row.get(
                    "pair_usable"
                )
            )
        )
    ]


def success_effect(
    *,
    profile: str,
    contrast: str,
    rows: list[dict[str, str]],
    seed: int,
) -> dict[str, Any]:
    pairs = []

    for row in rows:
        a = number(
            row.get(
                "baseline_overall_pass"
            )
        )
        b = number(
            row.get(
                "treatment_overall_pass"
            )
        )

        if (
            a is None
            or b is None
        ):
            continue

        a_pass = int(
            a > 0
        )
        b_pass = int(
            b > 0
        )

        pairs.append(
            (
                a_pass,
                b_pass,
            )
        )

    diffs = [
        b - a
        for a, b in pairs
    ]

    n = len(pairs)

    baseline_pass = sum(
        a
        for a, _ in pairs
    )

    treatment_pass = sum(
        b
        for _, b in pairs
    )

    fail_to_pass = sum(
        a == 0 and b == 1
        for a, b in pairs
    )

    pass_to_fail = sum(
        a == 1 and b == 0
        for a, b in pairs
    )

    effect = (
        100.0
        * sum(diffs)
        / n
        if n
        else float("nan")
    )

    lo, hi = bootstrap_mean_ci(
        [
            100.0 * x
            for x in diffs
        ],
        seed=seed,
    )

    return {
        "profile": profile,
        "contrast": contrast,
        "n_pairs": n,
        "baseline_pass": baseline_pass,
        "treatment_pass": treatment_pass,
        "baseline_pass_pct": (
            100.0
            * baseline_pass
            / n
            if n
            else float("nan")
        ),
        "treatment_pass_pct": (
            100.0
            * treatment_pass
            / n
            if n
            else float("nan")
        ),
        "delta_pp": effect,
        "bootstrap_ci_low_pp": lo,
        "bootstrap_ci_high_pp": hi,
        "fail_to_pass": fail_to_pass,
        "pass_to_fail": pass_to_fail,
        "discordant_pairs": (
            fail_to_pass
            + pass_to_fail
        ),
        "mcnemar_exact_p": (
            mcnemar_exact(
                fail_to_pass,
                pass_to_fail,
            )
        ),
        "mcnemar_holm_p": "",
        "inferential_role": (
            "descriptive_capability_floor"
            if profile == "llama"
            else "inferential"
        ),
    }


def process_effect(
    *,
    profile: str,
    metric: str,
    field: str,
    rows: list[dict[str, str]],
    seed: int,
) -> dict[str, Any]:
    diffs = []

    for row in rows:
        value = number(
            row.get(field)
        )

        if value is not None:
            diffs.append(
                value
            )

    mean_delta = (
        sum(diffs)
        / len(diffs)
        if diffs
        else float("nan")
    )

    lo, hi = bootstrap_mean_ci(
        diffs,
        seed=seed,
    )

    return {
        "profile": profile,
        "contrast": (
            "resource_vs_eval"
        ),
        "metric": metric,
        "n_pairs": len(diffs),
        "mean_delta": mean_delta,
        "bootstrap_ci_low": lo,
        "bootstrap_ci_high": hi,
        "signflip_p": signflip_p(
            diffs,
            seed=(
                seed
                + 10_000
            ),
        ),
        "holm_p": "",
        "inferential_role": (
            "descriptive_capability_floor"
            if profile == "llama"
            else "exploratory_inferential"
        ),
    }


def cochran_q(
    values: list[
        tuple[int, int, int]
    ],
) -> tuple[float, float]:
    """Cochran Q for three matched binary conditions.

    Returns statistic and chi-square(df=2) p.
    For df=2, survival function is exp(-Q/2).
    """

    if not values:
        return (
            float("nan"),
            float("nan"),
        )

    k = 3
    n = len(values)

    column_sums = [
        sum(row[j] for row in values)
        for j in range(k)
    ]

    row_sums = [
        sum(row)
        for row in values
    ]

    total = sum(
        column_sums
    )

    numerator = (
        (k - 1)
        * (
            k
            * sum(
                x * x
                for x
                in column_sums
            )
            - total * total
        )
    )

    denominator = (
        k * total
        - sum(
            x * x
            for x
            in row_sums
        )
    )

    if denominator <= 0:
        return (
            0.0,
            1.0,
        )

    q = (
        numerator
        / denominator
    )

    p = math.exp(
        -q / 2.0
    )

    return (
        q,
        min(
            1.0,
            max(
                0.0,
                p,
            ),
        ),
    )


def triple_test(
    profile: str,
    trials_path: Path,
) -> dict[str, Any]:
    rows = json.loads(
        trials_path.read_text(
            encoding="utf-8"
        )
    )

    by_task: dict[
        str,
        dict[str, int],
    ] = {}

    for row in rows:
        if not bool(
            row.get(
                "substantive_usable"
            )
        ):
            continue

        condition = str(
            row.get(
                "condition"
            )
        )

        if condition not in {
            "clean",
            "eval_only",
            "eval_resource_deprivation",
        }:
            continue

        value = number(
            row.get(
                "overall_pass"
            )
        )

        if value is None:
            continue

        base = str(
            row.get(
                "base_task_id"
            )
        )

        by_task.setdefault(
            base,
            {},
        )[condition] = int(
            value > 0
        )

    triples = []

    for base in sorted(
        by_task
    ):
        cells = by_task[base]

        if all(
            condition in cells
            for condition in (
                "clean",
                "eval_only",
                "eval_resource_deprivation",
            )
        ):
            triples.append(
                (
                    cells["clean"],
                    cells["eval_only"],
                    cells[
                        "eval_resource_deprivation"
                    ],
                )
            )

    q, p = cochran_q(
        triples
    )

    return {
        "profile": profile,
        "n_complete_triples": len(
            triples
        ),
        "cochran_q": q,
        "df": 2,
        "p_value": p,
        "holm_p": "",
        "inferential_role": (
            "descriptive_capability_floor"
            if profile == "llama"
            else "secondary_inferential"
        ),
    }


def main() -> None:
    root = Path(
        "analysis/resource"
    )

    output = (
        root
        / "inference"
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    primary = []
    secondary = []
    process = []
    omnibus = []

    input_hashes = {}

    for profile_index, profile in enumerate(
        PROFILES
    ):
        pair_path = (
            root
            / profile
            / "matched_pairs.csv"
        )

        trials_path = (
            root
            / profile
            / "trials.json"
        )

        if not pair_path.is_file():
            raise SystemExit(
                f"Missing {pair_path}"
            )

        if not trials_path.is_file():
            raise SystemExit(
                f"Missing {trials_path}"
            )

        input_hashes[
            f"{profile}:matched_pairs"
        ] = sha256(
            pair_path
        )

        input_hashes[
            f"{profile}:trials"
        ] = sha256(
            trials_path
        )

        rows = read_csv(
            pair_path
        )

        primary_rows = matched_rows(
            rows,
            PRIMARY_BASELINE,
            PRIMARY_TREATMENT,
        )

        eval_rows = matched_rows(
            rows,
            "clean",
            "eval_only",
        )

        primary.append(
            success_effect(
                profile=profile,
                contrast=(
                    "resource_vs_eval"
                ),
                rows=primary_rows,
                seed=(
                    SEED
                    + profile_index
                ),
            )
        )

        secondary.append(
            success_effect(
                profile=profile,
                contrast=(
                    "eval_vs_clean"
                ),
                rows=eval_rows,
                seed=(
                    SEED
                    + 100
                    + profile_index
                ),
            )
        )

        for metric_index, (
            metric,
            field,
        ) in enumerate(
            PROCESS_METRICS.items()
        ):
            process.append(
                process_effect(
                    profile=profile,
                    metric=metric,
                    field=field,
                    rows=primary_rows,
                    seed=(
                        SEED
                        + 1000
                        + profile_index
                        * 100
                        + metric_index
                    ),
                )
            )

        omnibus.append(
            triple_test(
                profile,
                trials_path,
            )
        )

    # Primary success multiplicity:
    # one resource-vs-eval test
    # for each capable model.
    capable_primary = [
        row
        for row in primary
        if row["profile"]
        in CAPABLE
    ]

    adjusted = holm(
        [
            float(
                row[
                    "mcnemar_exact_p"
                ]
            )
            for row
            in capable_primary
        ]
    )

    for row, adj in zip(
        capable_primary,
        adjusted,
    ):
        row[
            "mcnemar_holm_p"
        ] = adj

    # Secondary eval-vs-clean
    # family across capable models.
    capable_secondary = [
        row
        for row in secondary
        if row["profile"]
        in CAPABLE
    ]

    adjusted = holm(
        [
            float(
                row[
                    "mcnemar_exact_p"
                ]
            )
            for row
            in capable_secondary
        ]
    )

    for row, adj in zip(
        capable_secondary,
        adjusted,
    ):
        row[
            "mcnemar_holm_p"
        ] = adj

    # Exploratory process family:
    # 3 capable models x 3 metrics.
    capable_process = [
        row
        for row in process
        if row["profile"]
        in CAPABLE
    ]

    adjusted = holm(
        [
            float(
                row["signflip_p"]
            )
            for row
            in capable_process
        ]
    )

    for row, adj in zip(
        capable_process,
        adjusted,
    ):
        row["holm_p"] = adj

    # Secondary Cochran-Q family
    # across capable models.
    capable_q = [
        row
        for row in omnibus
        if row["profile"]
        in CAPABLE
    ]

    adjusted = holm(
        [
            float(
                row["p_value"]
            )
            for row
            in capable_q
        ]
    )

    for row, adj in zip(
        capable_q,
        adjusted,
    ):
        row["holm_p"] = adj

    write_csv(
        output
        / "primary_success.csv",
        primary,
    )

    write_csv(
        output
        / "secondary_eval_success.csv",
        secondary,
    )

    write_csv(
        output
        / "primary_process.csv",
        process,
    )

    write_csv(
        output
        / "cochran_q.csv",
        omnibus,
    )

    manifest = {
        "version": VERSION,
        "status": (
            "resource_inference_v1"
        ),
        "primary_contrast": (
            "eval_resource_deprivation "
            "vs eval_only"
        ),
        "primary_success_family": (
            "Claude, Fable, Codex "
            "exact McNemar; Holm across "
            "3 tests"
        ),
        "llama_success_role": (
            "descriptive capability floor"
        ),
        "bootstrap_reps": (
            BOOTSTRAP_REPS
        ),
        "signflip_reps": (
            SIGNFLIP_REPS
        ),
        "seed": SEED,
        "process_metrics": list(
            PROCESS_METRICS
        ),
        "input_hashes": (
            input_hashes
        ),
        "git_head": git_head(),
        "network_calls": 0,
        "semantic_calls": 0,
    }

    (
        output
        / "manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Resource-deprivation inference v1",
        "",
        (
            "Primary contrast: "
            "`eval_resource_deprivation` "
            "vs `eval_only`."
        ),
        "",
        "## Primary task-success contrast",
        "",
        (
            "| Model | N | Eval-only pass | "
            "Resource pass | Δ pp | 95% bootstrap CI | "
            "Fail→pass | Pass→fail | McNemar p | Holm p |"
        ),
        (
            "|---|---:|---:|---:|---:|---:|"
            "---:|---:|---:|---:|"
        ),
    ]

    for row in primary:
        holm_value = (
            f"{float(row['mcnemar_holm_p']):.6g}"
            if row[
                "mcnemar_holm_p"
            ] != ""
            else "descriptive"
        )

        lines.append(
            "| "
            + str(
                row["profile"]
            )
            + " | "
            + str(
                row["n_pairs"]
            )
            + " | "
            + f"{row['baseline_pass_pct']:.1f}%"
            + " | "
            + f"{row['treatment_pass_pct']:.1f}%"
            + " | "
            + f"{row['delta_pp']:+.2f}"
            + " | "
            + (
                f"[{row['bootstrap_ci_low_pp']:+.2f}, "
                f"{row['bootstrap_ci_high_pp']:+.2f}]"
            )
            + " | "
            + str(
                row["fail_to_pass"]
            )
            + " | "
            + str(
                row["pass_to_fail"]
            )
            + " | "
            + f"{row['mcnemar_exact_p']:.6g}"
            + " | "
            + holm_value
            + " |"
        )

    lines.extend(
        [
            "",
            "## Exploratory process contrast",
            "",
            (
                "| Model | Metric | N | Mean Δ | "
                "95% bootstrap CI | Sign-flip p | Holm p |"
            ),
            (
                "|---|---|---:|---:|---:|---:|---:|"
            ),
        ]
    )

    for row in process:
        holm_value = (
            f"{float(row['holm_p']):.6g}"
            if row["holm_p"] != ""
            else "descriptive"
        )

        lines.append(
            "| "
            + str(
                row["profile"]
            )
            + " | "
            + str(
                row["metric"]
            )
            + " | "
            + str(
                row["n_pairs"]
            )
            + " | "
            + f"{row['mean_delta']:+.2f}"
            + " | "
            + (
                f"[{row['bootstrap_ci_low']:+.2f}, "
                f"{row['bootstrap_ci_high']:+.2f}]"
            )
            + " | "
            + f"{row['signflip_p']:.6g}"
            + " | "
            + holm_value
            + " |"
        )

    lines.extend(
        [
            "",
            "## Complete-triple omnibus",
            "",
            (
                "| Model | N triples | Cochran Q | "
                "p | Holm p |"
            ),
            "|---|---:|---:|---:|---:|",
        ]
    )

    for row in omnibus:
        holm_value = (
            f"{float(row['holm_p']):.6g}"
            if row["holm_p"] != ""
            else "descriptive"
        )

        lines.append(
            "| "
            + str(
                row["profile"]
            )
            + " | "
            + str(
                row[
                    "n_complete_triples"
                ]
            )
            + " | "
            + f"{row['cochran_q']:.4f}"
            + " | "
            + f"{row['p_value']:.6g}"
            + " | "
            + holm_value
            + " |"
        )

    lines.extend(
        [
            "",
            (
                "Llama task-success inference is "
                "descriptive because its success "
                "outcome is at the capability floor."
            ),
            "",
            (
                "No semantic labels are used in "
                "this analysis."
            ),
            "",
        ]
    )

    (
        output
        / "report.md"
    ).write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(
        "RESOURCE INFERENCE V1"
    )
    print("=" * 72)

    print()
    print(
        "PRIMARY SUCCESS"
    )

    for row in primary:
        holm_value = (
            row["mcnemar_holm_p"]
            if row[
                "mcnemar_holm_p"
            ] != ""
            else "descriptive"
        )

        print(
            row["profile"],
            "N=",
            row["n_pairs"],
            "delta_pp=",
            round(
                row["delta_pp"],
                3,
            ),
            "CI=",
            (
                round(
                    row[
                        "bootstrap_ci_low_pp"
                    ],
                    3,
                ),
                round(
                    row[
                        "bootstrap_ci_high_pp"
                    ],
                    3,
                ),
            ),
            "discordant=",
            (
                row["fail_to_pass"],
                row["pass_to_fail"],
            ),
            "p=",
            row[
                "mcnemar_exact_p"
            ],
            "holm=",
            holm_value,
        )

    print()
    print(
        "EXPLORATORY PROCESS"
    )

    for row in process:
        print(
            row["profile"],
            row["metric"],
            "N=",
            row["n_pairs"],
            "mean_delta=",
            round(
                row["mean_delta"],
                3,
            ),
            "p=",
            row["signflip_p"],
            "holm=",
            (
                row["holm_p"]
                if row[
                    "holm_p"
                ] != ""
                else "descriptive"
            ),
        )

    print()
    print(
        "COCHRAN Q"
    )

    for row in omnibus:
        print(
            row["profile"],
            "N=",
            row[
                "n_complete_triples"
            ],
            "Q=",
            round(
                row[
                    "cochran_q"
                ],
                4,
            ),
            "p=",
            row["p_value"],
            "holm=",
            (
                row["holm_p"]
                if row[
                    "holm_p"
                ] != ""
                else "descriptive"
            ),
        )

    print()
    print(
        "outputs:",
        output,
    )
    print(
        "network calls: 0"
    )
    print(
        "semantic calls: 0"
    )
    print(
        "RESOURCE INFERENCE V1: PASS"
    )


if __name__ == "__main__":
    main()
