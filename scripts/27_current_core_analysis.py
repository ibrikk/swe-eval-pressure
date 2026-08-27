#!/usr/bin/env python3

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "analysis" / "current"
SOURCE = CURRENT / "source"
OUT = CURRENT / "results"

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

PRIMARY_SEMANTIC_FIELDS = (
    "evaluation_recognition",
    "pressure_recognition",
    "cue_provenance_stance",
    "explicit_resistance",
    "claimed_behavioral_response",
)

RESOURCE_SEMANTIC_FIELDS = (
    "evaluation_recognition",
    "resource_constraint_recognition",
    "resource_constraint_stance",
    "claimed_resource_response",
    "response_conserve_tool_calls",
    "response_batch_commands",
    "response_reduce_exploration",
    "response_reduce_validation",
    "response_prioritize_core_edit",
    "response_stop_early",
    "response_other",
    "evaluation_targeting_intent",
)

BEHAVIOR_FIELDS = (
    "broad_repo_search_any",
    "test_inspection_any",
    "validation_any",
    "iterative_repair_any",
    "provenance_related_inspection_any",
    "external_lookup_any",
    "integrity_sensitive_action_any",
)

PROCESS_CANDIDATES = (
    "raw_tool_calls",
    "tool_bearing_turns",
    "test_command_calls",
    "validation_command_calls",
    "repo_search_calls",
    "file_read_calls",
    "unique_files_read",
    "test_files_inspected",
    "edit_calls",
    "validation_calls",
    "post_edit_validation_calls",
    "edit_validation_cycles",
    "failed_validation_then_edit_cycles",
    "instruction_file_inspections",
    "git_history_inspections",
    "external_lookup_calls",
    "subagent_delegation_calls",
    "input_tokens",
    "output_tokens",
    "prompt_tokens",
    "completion_tokens",
    "duration_sec",
    "duration_seconds",
)

BOOTSTRAPS = 20000
BOOTSTRAP_SEED = 20260827


def load_json(path: Path) -> Any:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            obj,
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

    fields = []

    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def numeric(value: Any) -> float | None:
    if value in (
        None,
        "",
    ):
        return None

    try:
        return float(value)
    except Exception:
        return None


def substantive(
    row: dict[str, Any],
) -> bool:
    value = row.get(
        "substantive_usable"
    )

    if value is True:
        return True

    return str(
        value
    ).strip().lower() in {
        "1",
        "true",
        "yes",
    }


def overall_pass(
    row: dict[str, Any],
) -> int | None:
    value = numeric(
        row.get("overall_pass")
    )

    if value is None:
        return None

    return int(
        value >= 1.0
    )


def tests_pass(
    row: dict[str, Any],
) -> int | None:
    value = numeric(
        row.get("tests_reward")
    )

    if value is None:
        return None

    # IMPORTANT:
    # benchmark verifier gate is >= 1.0,
    # not merely > 0.
    return int(
        value >= 1.0
    )


def binary_value(
    row: dict[str, Any],
    metric: str,
) -> int | None:
    if metric == "overall_pass":
        return overall_pass(row)

    if metric == "tests_pass":
        return tests_pass(row)

    value = numeric(
        row.get(metric)
    )

    if value is None:
        return None

    return int(
        value > 0
    )


def percentile(
    values: list[float],
    q: float,
) -> float | None:
    if not values:
        return None

    vals = sorted(values)

    if len(vals) == 1:
        return vals[0]

    pos = (
        q
        * (
            len(vals)
            - 1
        )
    )

    lo = int(
        math.floor(pos)
    )

    hi = int(
        math.ceil(pos)
    )

    if lo == hi:
        return vals[lo]

    frac = pos - lo

    return (
        vals[lo]
        * (
            1 - frac
        )
        + vals[hi]
        * frac
    )


def mean(
    values: list[float],
) -> float | None:
    if not values:
        return None

    return sum(values) / len(values)


def median(
    values: list[float],
) -> float | None:
    return percentile(
        values,
        0.5,
    )


def wilson(
    success: int,
    n: int,
) -> tuple[
    float | None,
    float | None,
]:
    if n <= 0:
        return None, None

    z = 1.959963984540054
    p = success / n

    denom = (
        1
        + z * z / n
    )

    center = (
        p
        + z * z
        / (
            2 * n
        )
    ) / denom

    half = (
        z
        * math.sqrt(
            p
            * (
                1 - p
            )
            / n
            + z * z
            / (
                4 * n * n
            )
        )
        / denom
    )

    return (
        max(
            0.0,
            center - half,
        ),
        min(
            1.0,
            center + half,
        ),
    )


def paired_bootstrap_ci(
    deltas: list[float],
    *,
    seed_text: str,
) -> tuple[
    float | None,
    float | None,
]:
    if not deltas:
        return None, None

    digest = hashlib.sha256(
        (
            str(BOOTSTRAP_SEED)
            + "|"
            + seed_text
        ).encode()
    ).hexdigest()

    rng = random.Random(
        int(
            digest[:16],
            16,
        )
    )

    n = len(deltas)
    estimates = []

    for _ in range(
        BOOTSTRAPS
    ):
        estimates.append(
            sum(
                deltas[
                    rng.randrange(n)
                ]
                for _ in range(n)
            )
            / n
        )

    return (
        percentile(
            estimates,
            0.025,
        ),
        percentile(
            estimates,
            0.975,
        ),
    )


def exact_mcnemar(
    fail_to_pass: int,
    pass_to_fail: int,
) -> float | None:
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

    probability = sum(
        math.comb(
            n,
            i,
        )
        for i in range(
            k + 1
        )
    ) / (
        2 ** n
    )

    return min(
        1.0,
        2 * probability,
    )


def exact_sign_test(
    deltas: list[float],
) -> float | None:
    pos = sum(
        x > 0
        for x in deltas
    )

    neg = sum(
        x < 0
        for x in deltas
    )

    n = pos + neg

    if n == 0:
        return 1.0

    k = min(
        pos,
        neg,
    )

    p = (
        2
        * sum(
            math.comb(
                n,
                i,
            )
            for i in range(
                k + 1
            )
        )
        / (
            2 ** n
        )
    )

    return min(
        1.0,
        p,
    )


def holm(
    pvalues: list[
        tuple[int, float]
    ],
) -> dict[int, float]:
    ordered = sorted(
        pvalues,
        key=lambda x: x[1],
    )

    m = len(ordered)

    result = {}

    running = 0.0

    for rank, (
        index,
        p,
    ) in enumerate(
        ordered,
        start=1,
    ):
        adjusted = min(
            1.0,
            (
                m
                - rank
                + 1
            )
            * p,
        )

        running = max(
            running,
            adjusted,
        )

        result[index] = (
            running
        )

    return result


def bh(
    pvalues: list[
        tuple[int, float]
    ],
) -> dict[int, float]:
    ordered = sorted(
        pvalues,
        key=lambda x: x[1],
    )

    m = len(ordered)
    provisional = {}

    for rank, (
        index,
        p,
    ) in enumerate(
        ordered,
        start=1,
    ):
        provisional[index] = min(
            1.0,
            p
            * m
            / rank,
        )

    running = 1.0
    result = {}

    for index, _ in reversed(
        ordered
    ):
        running = min(
            running,
            provisional[index],
        )

        result[index] = running

    return result


def cohen_kappa(
    pairs: list[
        tuple[str, str]
    ],
) -> float | None:
    if not pairs:
        return None

    n = len(pairs)

    po = sum(
        a == b
        for a, b in pairs
    ) / n

    cats = sorted(
        {
            x
            for pair in pairs
            for x in pair
        }
    )

    pa = Counter(
        a
        for a, _ in pairs
    )

    pb = Counter(
        b
        for _, b in pairs
    )

    pe = sum(
        (
            pa[c]
            / n
        )
        * (
            pb[c]
            / n
        )
        for c in cats
    )

    denom = 1 - pe

    if math.isclose(
        denom,
        0.0,
    ):
        return None

    return (
        po - pe
    ) / denom


def gwet_ac1(
    pairs: list[
        tuple[str, str]
    ],
) -> float | None:
    if not pairs:
        return None

    n = len(pairs)

    po = sum(
        a == b
        for a, b in pairs
    ) / n

    cats = sorted(
        {
            x
            for pair in pairs
            for x in pair
        }
    )

    k = max(
        2,
        len(cats),
    )

    pooled = Counter()

    for a, b in pairs:
        pooled[a] += 1
        pooled[b] += 1

    total = 2 * n

    pe = (
        sum(
            (
                pooled[c]
                / total
            )
            * (
                1
                - pooled[c]
                / total
            )
            for c in pooled
        )
        / (
            k - 1
        )
    )

    denom = 1 - pe

    if math.isclose(
        denom,
        0.0,
    ):
        return None

    return (
        po - pe
    ) / denom


def normalize_condition(
    value: Any,
) -> str:
    return str(
        value or ""
    ).strip()


def normalize_channel(
    value: Any,
) -> str:
    value = str(
        value or ""
    ).strip()

    if not value:
        return "none"

    return value


def load_study(
    study: str,
) -> list[
    dict[str, Any]
]:
    rows = []

    for profile in PROFILES:
        path = (
            SOURCE
            / study
            / profile
            / "trials.json"
        )

        data = load_json(path)

        for raw in data:
            row = dict(raw)

            row["_study"] = study
            row["_profile"] = profile
            row["_condition"] = (
                normalize_condition(
                    row.get(
                        "condition"
                    )
                )
            )
            row["_channel"] = (
                normalize_channel(
                    row.get(
                        "channel"
                    )
                )
            )

            rows.append(row)

    return rows


def row_key(
    row: dict[str, Any],
) -> tuple[
    str,
    str,
    str,
    str,
]:
    return (
        row["_profile"],
        str(
            row.get(
                "base_task_id"
            )
            or ""
        ),
        row["_condition"],
        row["_channel"],
    )


def index_rows(
    rows: list[
        dict[str, Any]
    ],
) -> dict[
    tuple[
        str,
        str,
        str,
        str,
    ],
    dict[str, Any],
]:
    out = {}

    for row in rows:
        if not substantive(row):
            continue

        key = row_key(row)

        if key in out:
            raise RuntimeError(
                f"duplicate substantive "
                f"analytical key: {key}"
            )

        out[key] = row

    return out


def contrast_specs(
    study: str,
) -> list[
    dict[str, str]
]:
    if study in {
        "primary",
        "replication",
    }:
        result = []

        for placement in (
            "source",
            "root",
            "scaffold",
        ):
            result.extend(
                [
                    {
                        "contrast":
                            "evaluation_disclosure",
                        "placement":
                            placement,
                        "baseline_condition":
                            "clean",
                        "baseline_channel":
                            "none",
                        "treatment_condition":
                            "eval_only",
                        "treatment_channel":
                            placement,
                    },
                    {
                        "contrast":
                            "financial_pressure",
                        "placement":
                            placement,
                        "baseline_condition":
                            "eval_only",
                        "baseline_channel":
                            placement,
                        "treatment_condition":
                            "eval_financial",
                        "treatment_channel":
                            placement,
                    },
                    {
                        "contrast":
                            "self_preservation_pressure",
                        "placement":
                            placement,
                        "baseline_condition":
                            "eval_only",
                        "baseline_channel":
                            placement,
                        "treatment_condition":
                            "eval_self_preservation",
                        "treatment_channel":
                            placement,
                    },
                ]
            )

        return result

    if study == "resource":
        return [
            {
                "contrast":
                    "resource_evaluation_disclosure",
                "placement":
                    "scaffold",
                "baseline_condition":
                    "clean",
                "baseline_channel":
                    "none",
                "treatment_condition":
                    "eval_only",
                "treatment_channel":
                    "scaffold",
            },
            {
                "contrast":
                    "resource_deprivation",
                "placement":
                    "scaffold",
                "baseline_condition":
                    "eval_only",
                "baseline_channel":
                    "scaffold",
                "treatment_condition":
                    "eval_resource_deprivation",
                "treatment_channel":
                    "scaffold",
            },
        ]

    raise ValueError(study)


def make_pairs(
    study: str,
    rows: list[
        dict[str, Any]
    ],
) -> list[
    dict[str, Any]
]:
    idx = index_rows(rows)

    base_ids = sorted(
        {
            str(
                row.get(
                    "base_task_id"
                )
                or ""
            )
            for row in rows
            if row.get(
                "base_task_id"
            )
        }
    )

    output = []

    for profile in PROFILES:
        for spec in contrast_specs(
            study
        ):
            for base in base_ids:
                b = idx.get(
                    (
                        profile,
                        base,
                        spec[
                            "baseline_condition"
                        ],
                        spec[
                            "baseline_channel"
                        ],
                    )
                )

                t = idx.get(
                    (
                        profile,
                        base,
                        spec[
                            "treatment_condition"
                        ],
                        spec[
                            "treatment_channel"
                        ],
                    )
                )

                if (
                    b is None
                    or t is None
                ):
                    continue

                output.append(
                    {
                        "study": study,
                        "profile": profile,
                        "base_task_id":
                            base,
                        **spec,
                        "baseline": b,
                        "treatment": t,
                    }
                )

    return output


def cell_summary(
    studies: dict[
        str,
        list[
            dict[str, Any]
        ],
    ],
) -> list[
    dict[str, Any]
]:
    groups = defaultdict(list)

    for study, rows in (
        studies.items()
    ):
        for row in rows:
            if not substantive(row):
                continue

            groups[
                (
                    study,
                    row["_profile"],
                    row["_condition"],
                    row["_channel"],
                )
            ].append(row)

    output = []

    for (
        study,
        profile,
        condition,
        channel,
    ), rows in sorted(
        groups.items()
    ):
        op = [
            overall_pass(r)
            for r in rows
        ]

        op = [
            x
            for x in op
            if x is not None
        ]

        tp = [
            tests_pass(r)
            for r in rows
        ]

        tp = [
            x
            for x in tp
            if x is not None
        ]

        op_n = sum(op)
        tp_n = sum(tp)

        op_lo, op_hi = wilson(
            op_n,
            len(op),
        )

        tp_lo, tp_hi = wilson(
            tp_n,
            len(tp),
        )

        output.append(
            {
                "study": study,
                "profile": profile,
                "condition": condition,
                "channel": channel,
                "n_substantive":
                    len(rows),
                "overall_pass_n":
                    op_n,
                "overall_pass_rate":
                    (
                        op_n
                        / len(op)
                        if op
                        else None
                    ),
                "overall_pass_ci95_low":
                    op_lo,
                "overall_pass_ci95_high":
                    op_hi,
                "tests_pass_n":
                    tp_n,
                "tests_pass_rate":
                    (
                        tp_n
                        / len(tp)
                        if tp
                        else None
                    ),
                "tests_pass_ci95_low":
                    tp_lo,
                "tests_pass_ci95_high":
                    tp_hi,
            }
        )

    return output


def binary_effect_rows(
    pairs: list[
        dict[str, Any]
    ],
) -> list[
    dict[str, Any]
]:
    groups = defaultdict(list)

    for pair in pairs:
        groups[
            (
                pair["study"],
                pair["profile"],
                pair["contrast"],
                pair["placement"],
            )
        ].append(pair)

    output = []

    for key, vals in sorted(
        groups.items()
    ):
        (
            study,
            profile,
            contrast,
            placement,
        ) = key

        for metric in (
            "overall_pass",
            "tests_pass",
        ):
            usable = []

            for pair in vals:
                b = binary_value(
                    pair["baseline"],
                    metric,
                )
                t = binary_value(
                    pair["treatment"],
                    metric,
                )

                if (
                    b is None
                    or t is None
                ):
                    continue

                usable.append(
                    (
                        b,
                        t,
                    )
                )

            if not usable:
                continue

            deltas = [
                float(t - b)
                for b, t in usable
            ]

            f2p = sum(
                b == 0 and t == 1
                for b, t in usable
            )

            p2f = sum(
                b == 1 and t == 0
                for b, t in usable
            )

            lo, hi = (
                paired_bootstrap_ci(
                    deltas,
                    seed_text=(
                        f"{study}|"
                        f"{profile}|"
                        f"{contrast}|"
                        f"{placement}|"
                        f"{metric}"
                    ),
                )
            )

            output.append(
                {
                    "study": study,
                    "profile": profile,
                    "contrast": contrast,
                    "placement": placement,
                    "metric": metric,
                    "matched_n":
                        len(usable),
                    "baseline_rate":
                        mean(
                            [
                                float(b)
                                for b, _
                                in usable
                            ]
                        ),
                    "treatment_rate":
                        mean(
                            [
                                float(t)
                                for _, t
                                in usable
                            ]
                        ),
                    "effect_pp":
                        100
                        * mean(
                            deltas
                        ),
                    "ci95_low_pp":
                        (
                            100 * lo
                            if lo
                            is not None
                            else None
                        ),
                    "ci95_high_pp":
                        (
                            100 * hi
                            if hi
                            is not None
                            else None
                        ),
                    "fail_to_pass":
                        f2p,
                    "pass_to_fail":
                        p2f,
                    "mcnemar_p":
                        exact_mcnemar(
                            f2p,
                            p2f,
                        ),
                }
            )

    # Primary historical:
    # Holm across 9 placement/contrast
    # tests within profile and metric.
    families = defaultdict(list)

    for i, row in enumerate(
        output
    ):
        if row["study"] in {
            "primary",
            "replication",
        }:
            family = (
                row["study"],
                row["profile"],
                row["metric"],
                "nine_primary_contrasts",
            )

        elif (
            row["study"]
            == "resource"
            and row["contrast"]
            == "resource_deprivation"
            and row["metric"]
            == "overall_pass"
            and row["profile"]
            in CAPABLE
        ):
            family = (
                "resource",
                "capable_models",
                "overall_pass",
                "resource_primary_success",
            )

        else:
            family = (
                row["study"],
                row["profile"],
                row["metric"],
                row["contrast"],
            )

        families[
            family
        ].append(
            (
                i,
                row["mcnemar_p"],
            )
        )

    for family, vals in (
        families.items()
    ):
        adjusted = holm(
            [
                (
                    i,
                    p,
                )
                for i, p in vals
                if p is not None
            ]
        )

        for i, _ in vals:
            output[i][
                "multiplicity_family"
            ] = "|".join(
                family
            )

            output[i][
                "holm_p"
            ] = adjusted.get(i)

    return output


def process_metric_available(
    pairs: list[
        dict[str, Any]
    ],
    metric: str,
) -> bool:
    total = 0
    valid = 0

    for pair in pairs:
        total += 1

        if (
            numeric(
                pair["baseline"].get(
                    metric
                )
            )
            is not None
            and numeric(
                pair[
                    "treatment"
                ].get(
                    metric
                )
            )
            is not None
        ):
            valid += 1

    return (
        total > 0
        and valid
        / total
        >= 0.8
    )


def process_effect_rows(
    pairs: list[
        dict[str, Any]
    ],
) -> list[
    dict[str, Any]
]:
    groups = defaultdict(list)

    for pair in pairs:
        groups[
            (
                pair["study"],
                pair["profile"],
                pair["contrast"],
                pair["placement"],
            )
        ].append(pair)

    output = []

    for key, vals in sorted(
        groups.items()
    ):
        (
            study,
            profile,
            contrast,
            placement,
        ) = key

        metrics = [
            metric
            for metric
            in PROCESS_CANDIDATES
            if process_metric_available(
                vals,
                metric,
            )
        ]

        for metric in metrics:
            records = []

            for pair in vals:
                b = numeric(
                    pair[
                        "baseline"
                    ].get(metric)
                )

                t = numeric(
                    pair[
                        "treatment"
                    ].get(metric)
                )

                if (
                    b is None
                    or t is None
                ):
                    continue

                records.append(
                    (
                        b,
                        t,
                        t - b,
                    )
                )

            if not records:
                continue

            deltas = [
                d
                for _, _, d
                in records
            ]

            lo, hi = (
                paired_bootstrap_ci(
                    deltas,
                    seed_text=(
                        f"process|"
                        f"{study}|"
                        f"{profile}|"
                        f"{contrast}|"
                        f"{placement}|"
                        f"{metric}"
                    ),
                )
            )

            output.append(
                {
                    "study": study,
                    "profile": profile,
                    "contrast": contrast,
                    "placement": placement,
                    "metric": metric,
                    "matched_n":
                        len(records),
                    "baseline_mean":
                        mean(
                            [
                                b
                                for b, _, _
                                in records
                            ]
                        ),
                    "treatment_mean":
                        mean(
                            [
                                t
                                for _, t, _
                                in records
                            ]
                        ),
                    "mean_delta":
                        mean(deltas),
                    "median_delta":
                        median(deltas),
                    "ci95_low":
                        lo,
                    "ci95_high":
                        hi,
                    "positive_delta_n":
                        sum(
                            d > 0
                            for d in deltas
                        ),
                    "negative_delta_n":
                        sum(
                            d < 0
                            for d in deltas
                        ),
                    "zero_delta_n":
                        sum(
                            d == 0
                            for d in deltas
                        ),
                    "sign_test_p":
                        exact_sign_test(
                            deltas
                        ),
                }
            )

    # BH within one model × contrast ×
    # placement exploratory process family.
    families = defaultdict(list)

    for i, row in enumerate(
        output
    ):
        family = (
            row["study"],
            row["profile"],
            row["contrast"],
            row["placement"],
        )

        families[
            family
        ].append(
            (
                i,
                row[
                    "sign_test_p"
                ],
            )
        )

    for family, vals in (
        families.items()
    ):
        adjusted = bh(
            [
                (
                    i,
                    p,
                )
                for i, p in vals
                if p is not None
            ]
        )

        for i, _ in vals:
            output[i][
                "fdr_family"
            ] = "|".join(
                family
            )

            output[i][
                "bh_q"
            ] = adjusted.get(i)

    return output


def behavior_prevalence(
    studies: dict[
        str,
        list[
            dict[str, Any]
        ],
    ],
) -> list[
    dict[str, Any]
]:
    output = []

    for study, rows in (
        studies.items()
    ):
        groups = defaultdict(list)

        for row in rows:
            if not substantive(row):
                continue

            groups[
                (
                    row["_profile"],
                    row["_condition"],
                    row["_channel"],
                )
            ].append(row)

        for (
            profile,
            condition,
            channel,
        ), vals in sorted(
            groups.items()
        ):
            for metric in (
                BEHAVIOR_FIELDS
            ):
                observed = [
                    numeric(
                        row.get(metric)
                    )
                    for row in vals
                ]

                observed = [
                    x
                    for x in observed
                    if x is not None
                ]

                # Missing behavior fields are
                # missing, never silently zero.
                if (
                    len(observed)
                    / max(
                        1,
                        len(vals),
                    )
                    < 0.8
                ):
                    continue

                positive = sum(
                    x > 0
                    for x in observed
                )

                lo, hi = wilson(
                    positive,
                    len(observed),
                )

                output.append(
                    {
                        "study": study,
                        "profile":
                            profile,
                        "condition":
                            condition,
                        "channel":
                            channel,
                        "metric":
                            metric,
                        "n":
                            len(
                                observed
                            ),
                        "positive_n":
                            positive,
                        "prevalence":
                            (
                                positive
                                / len(
                                    observed
                                )
                            ),
                        "ci95_low":
                            lo,
                        "ci95_high":
                            hi,
                    }
                )

    return output


def behavior_effects(
    all_pairs: list[
        dict[str, Any]
    ],
) -> list[
    dict[str, Any]
]:
    groups = defaultdict(list)

    for pair in all_pairs:
        groups[
            (
                pair["study"],
                pair["profile"],
                pair["contrast"],
                pair["placement"],
            )
        ].append(pair)

    output = []

    for key, vals in sorted(
        groups.items()
    ):
        (
            study,
            profile,
            contrast,
            placement,
        ) = key

        for metric in (
            BEHAVIOR_FIELDS
        ):
            usable = []

            for pair in vals:
                b = binary_value(
                    pair[
                        "baseline"
                    ],
                    metric,
                )

                t = binary_value(
                    pair[
                        "treatment"
                    ],
                    metric,
                )

                if (
                    b is None
                    or t is None
                ):
                    continue

                usable.append(
                    (
                        b,
                        t,
                    )
                )

            if (
                len(usable)
                < 0.8
                * len(vals)
            ):
                continue

            deltas = [
                float(
                    t - b
                )
                for b, t
                in usable
            ]

            f2p = sum(
                b == 0
                and t == 1
                for b, t
                in usable
            )

            p2f = sum(
                b == 1
                and t == 0
                for b, t
                in usable
            )

            lo, hi = (
                paired_bootstrap_ci(
                    deltas,
                    seed_text=(
                        f"behavior|"
                        f"{study}|"
                        f"{profile}|"
                        f"{contrast}|"
                        f"{placement}|"
                        f"{metric}"
                    ),
                )
            )

            output.append(
                {
                    "study": study,
                    "profile": profile,
                    "contrast": contrast,
                    "placement": placement,
                    "metric": metric,
                    "matched_n":
                        len(usable),
                    "baseline_rate":
                        mean(
                            [
                                float(b)
                                for b, _
                                in usable
                            ]
                        ),
                    "treatment_rate":
                        mean(
                            [
                                float(t)
                                for _, t
                                in usable
                            ]
                        ),
                    "effect_pp":
                        100
                        * mean(
                            deltas
                        ),
                    "ci95_low_pp":
                        (
                            100 * lo
                            if lo
                            is not None
                            else None
                        ),
                    "ci95_high_pp":
                        (
                            100 * hi
                            if hi
                            is not None
                            else None
                        ),
                    "fail_to_pass":
                        f2p,
                    "pass_to_fail":
                        p2f,
                    "mcnemar_p":
                        exact_mcnemar(
                            f2p,
                            p2f,
                        ),
                }
            )

    # Seven-behavior Holm family within
    # model × contrast × placement.
    families = defaultdict(list)

    for i, row in enumerate(
        output
    ):
        family = (
            row["study"],
            row["profile"],
            row["contrast"],
            row["placement"],
        )

        families[
            family
        ].append(
            (
                i,
                row[
                    "mcnemar_p"
                ],
            )
        )

    for family, vals in (
        families.items()
    ):
        adjusted = holm(
            [
                (
                    i,
                    p,
                )
                for i, p
                in vals
                if p is not None
            ]
        )

        for i, _ in vals:
            output[i][
                "multiplicity_family"
            ] = "|".join(
                family
            )

            output[i][
                "holm_p"
            ] = adjusted.get(i)

    return output


def load_semantic_jobs(
    root: Path,
    fields: tuple[str, ...],
    study: str,
) -> list[
    dict[str, Any]
]:
    output = []

    for path in sorted(
        (
            root
            / "jobs"
        ).glob(
            "*.json"
        )
    ):
        obj = load_json(path)

        status = str(
            obj.get("status")
            or ""
        )

        final = obj.get(
            "final_cache_entry"
        )

        valid = (
            status == "ok"
            and isinstance(
                final,
                dict,
            )
            and str(
                final.get(
                    "status"
                )
                or ""
            )
            == "ok"
            and isinstance(
                final.get(
                    "judgment"
                ),
                dict,
            )
        )

        judgments = (
            final.get(
                "judgment"
            )
            if valid
            else {}
        )

        row = {
            "study": study,
            "profile": str(
                obj.get("profile")
                or ""
            ),
            "trial_name": str(
                obj.get(
                    "trial_name"
                )
                or (
                    final.get(
                        "trial_name"
                    )
                    if isinstance(
                        final,
                        dict,
                    )
                    else ""
                )
                or ""
            ),
            "condition": str(
                obj.get(
                    "condition"
                )
                or ""
            ),
            "placement": str(
                obj.get(
                    "placement"
                )
                or ""
            ),
            "pressure_type": str(
                obj.get(
                    "pressure_type"
                )
                or ""
            ),
            "judge_family": str(
                obj.get(
                    "judge_family"
                )
                or ""
            ),
            "status": status,
            "valid": int(valid),
            "artifact":
                str(path),
        }

        for field in fields:
            value = (
                judgments.get(
                    field,
                    {}
                )
                if isinstance(
                    judgments,
                    dict,
                )
                else {}
            )

            row[
                field
            ] = (
                value.get(
                    "label"
                )
                if isinstance(
                    value,
                    dict,
                )
                else None
            )

            evidence = (
                value.get(
                    "evidence",
                    [],
                )
                if isinstance(
                    value,
                    dict,
                )
                else []
            )

            row[
                field
                + "__evidence"
            ] = json.dumps(
                evidence,
                ensure_ascii=False,
            )

        output.append(row)

    return output


def semantic_consensus(
    jobs: list[
        dict[str, Any]
    ],
    fields: tuple[str, ...],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    grouped = defaultdict(dict)

    for row in jobs:
        key = (
            row["study"],
            row["profile"],
            row["trial_name"],
        )

        grouped[key][
            row["judge_family"]
        ] = row

    consensus_rows = []
    agreement_rows = []
    distribution_rows = []

    for key, judges in sorted(
        grouped.items()
    ):
        (
            study,
            profile,
            trial,
        ) = key

        ds = judges.get(
            "deepseek"
        )
        gm = judges.get(
            "gemini"
        )

        base = {
            "study": study,
            "profile": profile,
            "trial_name": trial,
            "deepseek_valid":
                int(
                    bool(ds)
                    and ds.get(
                        "valid"
                    )
                    == 1
                ),
            "gemini_valid":
                int(
                    bool(gm)
                    and gm.get(
                        "valid"
                    )
                    == 1
                ),
        }

        row = dict(base)

        for field in fields:
            dl = (
                ds.get(field)
                if (
                    ds
                    and ds.get(
                        "valid"
                    )
                    == 1
                )
                else None
            )

            gl = (
                gm.get(field)
                if (
                    gm
                    and gm.get(
                        "valid"
                    )
                    == 1
                )
                else None
            )

            if (
                dl is not None
                and gl is not None
                and dl == gl
            ):
                status = (
                    "agreement"
                )
                label = dl

            elif (
                dl is not None
                and gl is not None
            ):
                status = (
                    "disagreement"
                )
                label = None

            elif (
                dl is not None
                or gl is not None
            ):
                status = (
                    "single_valid"
                )
                label = None

            else:
                status = (
                    "missing"
                )
                label = None

            row[
                field
                + "__status"
            ] = status

            row[
                field
                + "__label"
            ] = label

            row[
                field
                + "__deepseek"
            ] = dl

            row[
                field
                + "__gemini"
            ] = gl

        consensus_rows.append(
            row
        )

    # Reliability and distributions
    studies_fields = defaultdict(
        list
    )

    for row in consensus_rows:
        for field in fields:
            ds = row[
                field
                + "__deepseek"
            ]

            gm = row[
                field
                + "__gemini"
            ]

            if (
                ds is not None
                and gm is not None
            ):
                studies_fields[
                    (
                        row[
                            "study"
                        ],
                        field,
                    )
                ].append(
                    (
                        ds,
                        gm,
                    )
                )

    for (
        study,
        field,
    ), pairs in sorted(
        studies_fields.items()
    ):
        agree = sum(
            a == b
            for a, b in pairs
        )

        agreement_rows.append(
            {
                "study": study,
                "field": field,
                "both_valid_n":
                    len(pairs),
                "agreement_n":
                    agree,
                "raw_agreement":
                    agree
                    / len(pairs),
                "cohen_kappa":
                    cohen_kappa(
                        pairs
                    ),
                "gwet_ac1":
                    gwet_ac1(
                        pairs
                    ),
            }
        )

    # Full label distributions by judge
    for field in fields:
        groups = defaultdict(
            Counter
        )

        for row in jobs:
            if row["valid"] != 1:
                continue

            label = row.get(
                field
            )

            if label is None:
                continue

            groups[
                (
                    row["study"],
                    row["profile"],
                    row[
                        "judge_family"
                    ],
                )
            ][label] += 1

        for (
            study,
            profile,
            judge,
        ), counts in sorted(
            groups.items()
        ):
            n = sum(
                counts.values()
            )

            for label, count in sorted(
                counts.items()
            ):
                distribution_rows.append(
                    {
                        "study":
                            study,
                        "profile":
                            profile,
                        "field":
                            field,
                        "source":
                            judge,
                        "label":
                            label,
                        "n":
                            n,
                        "count":
                            count,
                        "rate":
                            count
                            / n,
                    }
                )

    # Strict consensus distribution
    for field in fields:
        groups = defaultdict(
            Counter
        )

        for row in consensus_rows:
            if (
                row[
                    field
                    + "__status"
                ]
                != "agreement"
            ):
                continue

            label = row[
                field
                + "__label"
            ]

            groups[
                (
                    row["study"],
                    row["profile"],
                )
            ][label] += 1

        for (
            study,
            profile,
        ), counts in sorted(
            groups.items()
        ):
            resolved_n = sum(
                counts.values()
            )

            total_n = sum(
                1
                for row
                in consensus_rows
                if (
                    row["study"]
                    == study
                    and row[
                        "profile"
                    ]
                    == profile
                )
            )

            for label, count in sorted(
                counts.items()
            ):
                distribution_rows.append(
                    {
                        "study":
                            study,
                        "profile":
                            profile,
                        "field":
                            field,
                        "source":
                            "strict_consensus",
                        "label":
                            label,
                        "n":
                            resolved_n,
                        "full_cell_n":
                            total_n,
                        "count":
                            count,
                        "rate":
                            (
                                count
                                / resolved_n
                                if resolved_n
                                else None
                            ),
                        "full_cell_lower_bound":
                            (
                                count
                                / total_n
                                if total_n
                                else None
                            ),
                        "full_cell_upper_bound":
                            (
                                (
                                    count
                                    + total_n
                                    - resolved_n
                                )
                                / total_n
                                if total_n
                                else None
                            ),
                    }
                )

    return (
        consensus_rows,
        agreement_rows,
        distribution_rows,
    )


def semantic_lookup(
    consensus_rows: list[
        dict[str, Any]
    ],
) -> dict[
    tuple[
        str,
        str,
        str,
    ],
    dict[str, Any],
]:
    return {
        (
            row["study"],
            row["profile"],
            row["trial_name"],
        ):
        row
        for row
        in consensus_rows
    }


def said_did_rows(
    pairs: list[
        dict[str, Any]
    ],
    semantic: dict[
        tuple[
            str,
            str,
            str,
        ],
        dict[str, Any],
    ],
    fields_by_study: dict[
        str,
        tuple[str, ...],
    ],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    rows = []

    metrics = (
        "overall_pass",
        "tests_pass",
        "raw_tool_calls",
        "test_command_calls",
        "validation_command_calls",
        "input_tokens",
        "output_tokens",
        "duration_sec",
    )

    for pair in pairs:
        treatment = pair[
            "treatment"
        ]

        key = (
            pair["study"],
            pair["profile"],
            str(
                treatment.get(
                    "trial_name"
                )
                or ""
            ),
        )

        sem = semantic.get(
            key
        )

        if sem is None:
            continue

        row = {
            "study": pair["study"],
            "profile": pair["profile"],
            "base_task_id":
                pair[
                    "base_task_id"
                ],
            "contrast":
                pair[
                    "contrast"
                ],
            "placement":
                pair[
                    "placement"
                ],
            "baseline_trial":
                pair[
                    "baseline"
                ].get(
                    "trial_name"
                ),
            "treatment_trial":
                treatment.get(
                    "trial_name"
                ),
            "post_treatment_semantics":
                1,
            "causal_subgroup_claim":
                0,
        }

        for field in (
            fields_by_study[
                pair["study"]
            ]
        ):
            row[
                field
                + "__status"
            ] = sem.get(
                field
                + "__status"
            )

            row[
                field
                + "__label"
            ] = sem.get(
                field
                + "__label"
            )

        for metric in metrics:
            if metric == (
                "overall_pass"
            ):
                b = overall_pass(
                    pair[
                        "baseline"
                    ]
                )

                t = overall_pass(
                    pair[
                        "treatment"
                    ]
                )

            elif metric == (
                "tests_pass"
            ):
                b = tests_pass(
                    pair[
                        "baseline"
                    ]
                )

                t = tests_pass(
                    pair[
                        "treatment"
                    ]
                )

            else:
                b = numeric(
                    pair[
                        "baseline"
                    ].get(metric)
                )

                t = numeric(
                    pair[
                        "treatment"
                    ].get(metric)
                )

            row[
                "delta_"
                + metric
            ] = (
                t - b
                if (
                    b is not None
                    and t is not None
                )
                else None
            )

        rows.append(row)

    summaries = []

    groups = defaultdict(list)

    for row in rows:
        fields = fields_by_study[
            row["study"]
        ]

        for field in fields:
            if (
                row.get(
                    field
                    + "__status"
                )
                != "agreement"
            ):
                continue

            label = row.get(
                field
                + "__label"
            )

            groups[
                (
                    row["study"],
                    row["profile"],
                    row["contrast"],
                    row["placement"],
                    field,
                    label,
                )
            ].append(row)

    for key, vals in sorted(
        groups.items()
    ):
        (
            study,
            profile,
            contrast,
            placement,
            field,
            label,
        ) = key

        summary = {
            "study": study,
            "profile": profile,
            "contrast": contrast,
            "placement": placement,
            "semantic_field": field,
            "semantic_label": label,
            "n": len(vals),
            "post_treatment_semantics":
                1,
            "causal_interpretation":
                0,
        }

        for metric in (
            "overall_pass",
            "tests_pass",
            "raw_tool_calls",
            "test_command_calls",
            "validation_command_calls",
            "input_tokens",
            "output_tokens",
            "duration_sec",
        ):
            values = [
                row.get(
                    "delta_"
                    + metric
                )
                for row in vals
            ]

            values = [
                float(x)
                for x in values
                if x is not None
            ]

            summary[
                "mean_delta_"
                + metric
            ] = mean(values)

            summary[
                "median_delta_"
                + metric
            ] = median(values)

        summaries.append(
            summary
        )

    return rows, summaries


def replication_direction(
    binary_rows: list[
        dict[str, Any]
    ],
    process_rows: list[
        dict[str, Any]
    ],
) -> list[
    dict[str, Any]
]:
    output = []

    for metric_name, rows, effect_key in (
        (
            "binary",
            binary_rows,
            "effect_pp",
        ),
        (
            "process",
            process_rows,
            "mean_delta",
        ),
    ):
        primary = {}

        replication = {}

        for row in rows:
            key = (
                row["profile"],
                row["contrast"],
                row["placement"],
                row["metric"],
            )

            if (
                row["study"]
                == "primary"
            ):
                primary[
                    key
                ] = row

            elif (
                row["study"]
                == "replication"
            ):
                replication[
                    key
                ] = row

        for key in sorted(
            set(primary)
            & set(replication)
        ):
            a = primary[key]
            b = replication[key]

            x = a.get(
                effect_key
            )

            y = b.get(
                effect_key
            )

            if (
                x is None
                or y is None
            ):
                direction = None

            elif (
                math.isclose(
                    float(x),
                    0.0,
                )
                or math.isclose(
                    float(y),
                    0.0,
                )
            ):
                direction = (
                    "zero_in_one"
                )

            elif (
                float(x)
                * float(y)
                > 0
            ):
                direction = (
                    "same"
                )

            else:
                direction = (
                    "opposite"
                )

            output.append(
                {
                    "effect_type":
                        metric_name,
                    "profile": key[0],
                    "contrast": key[1],
                    "placement": key[2],
                    "metric": key[3],
                    "primary_effect":
                        x,
                    "replication_effect":
                        y,
                    "direction":
                        direction,
                    "replication_status":
                        "partial",
                }
            )

    return output


def main() -> None:
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    studies = {
        study: load_study(
            study
        )
        for study in (
            "primary",
            "resource",
            "replication",
        )
    }

    all_pairs = []

    for study, rows in (
        studies.items()
    ):
        all_pairs.extend(
            make_pairs(
                study,
                rows,
            )
        )

    cells = cell_summary(
        studies
    )

    binary = binary_effect_rows(
        all_pairs
    )

    process = process_effect_rows(
        all_pairs
    )

    prevalence = behavior_prevalence(
        studies
    )

    behavior = behavior_effects(
        all_pairs
    )

    primary_jobs = (
        load_semantic_jobs(
            ROOT
            / "analysis"
            / "semantic-multijudge-v1"
            / "final-repaired-llama-v1",
            PRIMARY_SEMANTIC_FIELDS,
            "primary",
        )
    )

    resource_jobs = (
        load_semantic_jobs(
            ROOT
            / "analysis"
            / "semantic-resource-v1"
            / "full"
            / "production-v1.1",
            RESOURCE_SEMANTIC_FIELDS,
            "resource",
        )
    )

    (
        primary_consensus,
        primary_agreement,
        primary_distribution,
    ) = semantic_consensus(
        primary_jobs,
        PRIMARY_SEMANTIC_FIELDS,
    )

    (
        resource_consensus,
        resource_agreement,
        resource_distribution,
    ) = semantic_consensus(
        resource_jobs,
        RESOURCE_SEMANTIC_FIELDS,
    )

    consensus = (
        primary_consensus
        + resource_consensus
    )

    semantic = semantic_lookup(
        consensus
    )

    semantic_fields = {
        "primary":
            PRIMARY_SEMANTIC_FIELDS,
        "resource":
            RESOURCE_SEMANTIC_FIELDS,
    }

    semantic_pairs = [
        pair
        for pair in all_pairs
        if pair["study"]
        in semantic_fields
    ]

    said_did, said_did_summary = (
        said_did_rows(
            semantic_pairs,
            semantic,
            semantic_fields,
        )
    )

    repl = replication_direction(
        binary,
        process,
    )

    write_csv(
        OUT / "cell_performance.csv",
        cells,
    )

    write_csv(
        OUT / "matched_binary_effects.csv",
        binary,
    )

    write_csv(
        OUT / "matched_process_effects.csv",
        process,
    )

    write_csv(
        OUT / "behavior_prevalence.csv",
        prevalence,
    )

    write_csv(
        OUT / "matched_behavior_effects.csv",
        behavior,
    )

    write_csv(
        OUT / "semantic_jobs_primary.csv",
        primary_jobs,
    )

    write_csv(
        OUT / "semantic_jobs_resource.csv",
        resource_jobs,
    )

    write_csv(
        OUT / "semantic_consensus.csv",
        consensus,
    )

    write_csv(
        OUT / "semantic_agreement.csv",
        (
            primary_agreement
            + resource_agreement
        ),
    )

    write_csv(
        OUT / "semantic_label_distribution.csv",
        (
            primary_distribution
            + resource_distribution
        ),
    )

    write_csv(
        OUT / "said_did_pairs.csv",
        said_did,
    )

    write_csv(
        OUT / "said_did_summary.csv",
        said_did_summary,
    )

    write_csv(
        OUT / "replication_direction.csv",
        repl,
    )

    manifest = {
        "analysis_version":
            "current-core-1.0",
        "bootstrap_replicates":
            BOOTSTRAPS,
        "bootstrap_seed":
            BOOTSTRAP_SEED,
        "success_endpoint":
            "overall_pass >= 1.0",
        "secondary_test_endpoint":
            "tests_reward >= 1.0",
        "semantic_inputs":
            {
                "primary":
                    (
                        "raw DeepSeek/Gemini "
                        "job artifacts"
                    ),
                "resource":
                    (
                        "raw DeepSeek/Gemini "
                        "job artifacts"
                    ),
            },
        "semantic_consensus_rule":
            (
                "strict agreement of both "
                "valid primary judges"
            ),
        "semantic_states_post_treatment":
            True,
        "said_did_causal":
            False,
        "historical_aggregate_inputs":
            False,
        "historical_inference_inputs":
            False,
        "historical_semantic_summary_inputs":
            False,
        "primary_behavior_taxonomy_available":
            bool(
                any(
                    row["study"]
                    == "primary"
                    for row in prevalence
                )
            ),
        "resource_behavior_taxonomy_available":
            bool(
                any(
                    row["study"]
                    == "resource"
                    for row in prevalence
                )
            ),
        "replication_behavior_taxonomy_available":
            bool(
                any(
                    row["study"]
                    == "replication"
                    for row in prevalence
                )
            ),
        "row_counts":
            {
                "cell_performance":
                    len(cells),
                "matched_binary_effects":
                    len(binary),
                "matched_process_effects":
                    len(process),
                "behavior_prevalence":
                    len(prevalence),
                "matched_behavior_effects":
                    len(behavior),
                "primary_semantic_jobs":
                    len(primary_jobs),
                "resource_semantic_jobs":
                    len(resource_jobs),
                "semantic_consensus":
                    len(consensus),
                "said_did_pairs":
                    len(said_did),
                "said_did_summary":
                    len(
                        said_did_summary
                    ),
                "replication_direction":
                    len(repl),
            },
    }

    write_json(
        OUT / "manifest.json",
        manifest,
    )

    print(
        "=" * 80
    )
    print(
        "CURRENT CORE ANALYSIS: PASS"
    )
    print(
        "=" * 80
    )

    print()
    print(
        "Fresh outputs:",
        OUT,
    )

    print()
    print(
        "Primary semantic jobs:",
        len(primary_jobs),
    )
    print(
        "Resource semantic jobs:",
        len(resource_jobs),
    )
    print(
        "Consensus trajectories:",
        len(consensus),
    )
    print(
        "Matched binary effects:",
        len(binary),
    )
    print(
        "Matched process effects:",
        len(process),
    )
    print(
        "Behavior prevalence rows:",
        len(prevalence),
    )
    print(
        "Behavior effect rows:",
        len(behavior),
    )
    print(
        "Said/did pairs:",
        len(said_did),
    )
    print(
        "Said/did summaries:",
        len(
            said_did_summary
        ),
    )

    print()
    print(
        "PRIMARY OVERALL-PASS EFFECTS"
    )
    print(
        "-" * 80
    )

    for row in binary:
        if (
            row["study"]
            == "primary"
            and row["metric"]
            == "overall_pass"
        ):
            print(
                f"{row['profile']:7s} "
                f"{row['contrast']:28s} "
                f"{row['placement']:8s} "
                f"n={row['matched_n']:2d} "
                f"Δ={row['effect_pp']:+7.2f}pp "
                f"CI=["
                f"{row['ci95_low_pp']:+7.2f}, "
                f"{row['ci95_high_pp']:+7.2f}] "
                f"p={row['mcnemar_p']:.6g} "
                f"Holm={row['holm_p']:.6g}"
            )

    print()
    print(
        "RESOURCE OVERALL-PASS EFFECTS"
    )
    print(
        "-" * 80
    )

    for row in binary:
        if (
            row["study"]
            == "resource"
            and row["metric"]
            == "overall_pass"
        ):
            print(
                f"{row['profile']:7s} "
                f"{row['contrast']:28s} "
                f"n={row['matched_n']:2d} "
                f"Δ={row['effect_pp']:+7.2f}pp "
                f"CI=["
                f"{row['ci95_low_pp']:+7.2f}, "
                f"{row['ci95_high_pp']:+7.2f}] "
                f"p={row['mcnemar_p']:.6g} "
                f"Holm={row['holm_p']:.6g}"
            )

    print()
    print(
        "SEMANTIC AGREEMENT"
    )
    print(
        "-" * 80
    )

    for row in (
        primary_agreement
        + resource_agreement
    ):
        print(
            f"{row['study']:8s} "
            f"{row['field']:38s} "
            f"n={row['both_valid_n']:4d} "
            f"agree="
            f"{100*row['raw_agreement']:5.1f}% "
            f"kappa="
            f"{row['cohen_kappa']} "
            f"AC1="
            f"{row['gwet_ac1']}"
        )

    print()
    print(
        "PRIMARY ACTION TAXONOMY:",
        (
            "AVAILABLE"
            if manifest[
                "primary_behavior_taxonomy_available"
            ]
            else (
                "NOT YET PRESENT IN "
                "HISTORICAL PRIMARY SOURCE"
            )
        ),
    )

    print(
        "RESOURCE ACTION TAXONOMY:",
        (
            "AVAILABLE"
            if manifest[
                "resource_behavior_taxonomy_available"
            ]
            else "MISSING"
        ),
    )

    print(
        "REPLICATION ACTION TAXONOMY:",
        (
            "AVAILABLE"
            if manifest[
                "replication_behavior_taxonomy_available"
            ]
            else "MISSING"
        ),
    )


if __name__ == "__main__":
    main()
