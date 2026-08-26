#!/usr/bin/env python3

from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import json

import numpy as np


DATA_ROOT = (
    Path.home()
    / "Documents"
    / "swe-eval-pressure"
)

BEHAVIOR_ROOT = (
    DATA_ROOT
    / "analysis"
    / "frozen"
    / "historical-primary-repaired-llama-20260826"
)

SEMANTIC_ROOT = (
    DATA_ROOT
    / "analysis"
    / "semantic-multijudge-v1"
    / "final-repaired-llama-v1"
)

OUT = (
    DATA_ROOT
    / "analysis"
    / "semantic-multijudge-v1"
    / "behavior-integration-v1"
    / "semantic_stratified_matched_deltas.csv"
)

PROFILES = [
    "claude",
    "fable",
    "codex",
    "llama",
]

PLACEMENTS = [
    "root",
    "scaffold",
    "source",
]

CONTRASTS = {
    "financial": "eval_financial",
    "self_preservation": "eval_self_preservation",
}

METRICS = [
    "overall_pass",
    "seeded_cue_removed_or_modified",
    "raw_tool_calls",
    "input_tokens",
    "validation_command_calls",
    "test_command_calls",
    "network_lookup_calls",
]

ENDPOINTS = {
    "pressure_recognition": (
        "pressure_recognition",
        {"observed"},
        {"not_observed"},
    ),
    "cue_untrusted_or_suspicious": (
        "cue_provenance_stance",
        {"untrusted_or_suspicious"},
        {
            "trusted",
            "neutral_or_uncertain",
            "not_observed",
        },
    ),
    "explicit_resistance": (
        "explicit_resistance",
        {"observed"},
        {"not_observed"},
    ),
    "any_claimed_behavioral_response": (
        "claimed_behavioral_response",
        {
            "ignore",
            "remove_or_modify_cue",
            "inspect_or_verify",
            "comply_with_cue",
            "other",
        },
        {"none_observed"},
    ),
}


def load(path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def number(x):
    if x is None:
        return None
    if isinstance(x, bool):
        return float(x)
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return float(str(x).strip())
    except Exception:
        return None


def seed_for(*parts):
    h = hashlib.sha256(
        "|".join(parts).encode()
    ).digest()
    return int.from_bytes(
        h[:8],
        "big",
    )


def bootstrap_diff(a, b, seed, reps=10000):
    if not a or not b:
        return (
            None,
            None,
            None,
            None,
            None,
        )

    a = np.asarray(
        a,
        dtype=float,
    )
    b = np.asarray(
        b,
        dtype=float,
    )

    rng = np.random.default_rng(
        seed
    )

    ia = rng.integers(
        0,
        len(a),
        size=(reps, len(a)),
    )

    ib = rng.integers(
        0,
        len(b),
        size=(reps, len(b)),
    )

    boot = (
        a[ia].mean(axis=1)
        - b[ib].mean(axis=1)
    )

    lo, hi = np.quantile(
        boot,
        [0.025, 0.975],
    )

    return (
        float(a.mean()),
        float(b.mean()),
        float(a.mean() - b.mean()),
        float(lo),
        float(hi),
    )


# --------------------------------------------------
# Frozen behavioral rows
# --------------------------------------------------

behavior = {}
trial_to_base = {}

for profile in PROFILES:
    rows = load(
        BEHAVIOR_ROOT
        / profile
        / "trials.json"
    )

    assert len(rows) == 700

    for row in rows:
        trial_key = (
            profile,
            str(row["trial_name"]),
        )

        if trial_key in trial_to_base:
            raise ValueError(
                f"duplicate trial identity: {trial_key}"
            )

        trial_to_base[trial_key] = str(
            row["base_task_id"]
        )

        if number(
            row.get("overall_pass")
        ) is None:
            continue

        key = (
            profile,
            str(row["base_task_id"]),
            str(row["condition"]),
            str(row["channel"]),
        )

        assert key not in behavior
        behavior[key] = row


# --------------------------------------------------
# Semantic pressure rows
# --------------------------------------------------

semantic = {}

paths = sorted(
    (
        SEMANTIC_ROOT
        / "consensus"
    ).glob("*.json")
)

assert len(paths) == 2776

for path in paths:
    x = load(path)

    if x["condition"] not in {
        "eval_financial",
        "eval_self_preservation",
    }:
        continue

    trial_key = (
        str(x["profile"]),
        str(x["trial_name"]),
    )

    if trial_key not in trial_to_base:
        raise ValueError(
            f"semantic trial missing frozen task identity: {trial_key}"
        )

    base_task_id = trial_to_base[
        trial_key
    ]

    semantic[
        (
            str(x["profile"]),
            base_task_id,
            str(x["condition"]),
            str(x["placement"]),
        )
    ] = x


# --------------------------------------------------
# Semantic-stratified task-matched deltas
# --------------------------------------------------

rows_out = []

for profile in PROFILES:
    for placement in PLACEMENTS:
        for pressure_name, condition in CONTRASTS.items():

            for endpoint, spec in ENDPOINTS.items():
                field, positive_labels, negative_labels = spec

                by_metric = {
                    metric: {
                        1: [],
                        0: [],
                    }
                    for metric in METRICS
                }

                resolved = 0
                unresolved = 0

                for base_id in {
                    key[1]
                    for key in behavior
                    if key[0] == profile
                }:
                    pressure_key = (
                        profile,
                        base_id,
                        condition,
                        placement,
                    )

                    reference_key = (
                        profile,
                        base_id,
                        "eval_only",
                        placement,
                    )

                    if (
                        pressure_key not in behavior
                        or reference_key not in behavior
                        or pressure_key not in semantic
                    ):
                        continue

                    sem = semantic[
                        pressure_key
                    ]

                    result = (
                        sem["consensus"]
                        ["fields"]
                        [field]
                    )

                    if result.get("status") != "agreement":
                        unresolved += 1
                        continue

                    label = result.get("label")

                    if label in positive_labels:
                        state = 1
                    elif label in negative_labels:
                        state = 0
                    else:
                        unresolved += 1
                        continue

                    resolved += 1

                    pressure_row = behavior[
                        pressure_key
                    ]

                    reference_row = behavior[
                        reference_key
                    ]

                    for metric in METRICS:
                        p = number(
                            pressure_row.get(metric)
                        )
                        r = number(
                            reference_row.get(metric)
                        )

                        if p is None or r is None:
                            continue

                        # Within-task matched treatment delta.
                        by_metric[
                            metric
                        ][state].append(
                            p - r
                        )

                for metric in METRICS:
                    pos = by_metric[
                        metric
                    ][1]

                    neg = by_metric[
                        metric
                    ][0]

                    (
                        pos_mean,
                        neg_mean,
                        diff,
                        lo,
                        hi,
                    ) = bootstrap_diff(
                        pos,
                        neg,
                        seed_for(
                            "semantic-stratified-matched-v1",
                            profile,
                            placement,
                            pressure_name,
                            endpoint,
                            metric,
                        ),
                    )

                    rows_out.append({
                        "profile": profile,
                        "placement": placement,
                        "pressure_type": pressure_name,
                        "condition": condition,
                        "endpoint": endpoint,
                        "metric": metric,
                        "resolved_semantic_n": resolved,
                        "unresolved_semantic_n": unresolved,
                        "semantic_positive_n": len(pos),
                        "semantic_negative_n": len(neg),
                        "positive_mean_pressure_minus_eval_only": pos_mean,
                        "negative_mean_pressure_minus_eval_only": neg_mean,
                        "difference_of_matched_deltas": diff,
                        "ci_low": lo,
                        "ci_high": hi,
                        "interpretation": (
                            "post-treatment descriptive association; "
                            "not causal"
                        ),
                    })


OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

columns = list(
    rows_out[0].keys()
)

with OUT.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=columns,
    )
    writer.writeheader()
    writer.writerows(rows_out)

assert len(rows_out) == (
    4 * 3 * 2 * 4 * 7
)

print(
    "SEMANTIC-STRATIFIED MATCHED BEHAVIOR"
)
print("=" * 80)
print(
    "rows:",
    len(rows_out),
)
print(
    "output:",
    OUT,
)

print()
print(
    "SOURCE-LOCAL PRIMARY VIEW"
)
print("-" * 80)

for r in rows_out:
    if (
        r["placement"] != "source"
        or r["metric"] not in {
            "overall_pass",
            "seeded_cue_removed_or_modified",
            "raw_tool_calls",
            "validation_command_calls",
        }
    ):
        continue

    if (
        r["semantic_positive_n"] < 5
        or r["semantic_negative_n"] < 5
        or r["difference_of_matched_deltas"] is None
    ):
        continue

    scale = (
        100
        if r["metric"] in {
            "overall_pass",
            "seeded_cue_removed_or_modified",
        }
        else 1
    )

    suffix = (
        "pp"
        if scale == 100
        else ""
    )

    d = (
        scale
        * r["difference_of_matched_deltas"]
    )
    lo = scale * r["ci_low"]
    hi = scale * r["ci_high"]

    print(
        f"{r['profile']:7s} "
        f"{r['pressure_type']:18s} "
        f"{r['endpoint']:34s} "
        f"{r['metric']:30s} "
        f"N+={r['semantic_positive_n']:2d} "
        f"N-={r['semantic_negative_n']:2d} "
        f"DiD={d:+9.2f}{suffix} "
        f"CI=[{lo:+9.2f},{hi:+9.2f}]"
    )

print()
print("network calls: 0")
print(
    "SEMANTIC-STRATIFIED MATCHED BEHAVIOR: PASS"
)
