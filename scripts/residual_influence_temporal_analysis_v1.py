#!/usr/bin/env python3
"""Event-aligned exploratory residual-influence analysis.

Primary question:
After a high-confidence source-local cue removal by Claude, does subsequent
behavior remain displaced relative to the exact task's eval-only trajectory,
beyond the corresponding root-placement pressure-vs-eval-only difference?

This is post-treatment descriptive/mechanistic analysis, not causal inference.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


DATA_ROOT = (
    Path.home()
    / "Documents"
    / "swe-eval-pressure"
)

FROZEN = (
    DATA_ROOT
    / "analysis"
    / "frozen"
    / "historical-primary-repaired-llama-20260826"
)

AUDIT = (
    DATA_ROOT
    / "analysis"
    / "semantic-multijudge-v1"
    / "residual-removal-event-audit-v2.json"
)

OUTDIR = (
    DATA_ROOT
    / "analysis"
    / "semantic-multijudge-v1"
    / "residual-influence-temporal-v1"
)

PLAN = (
    Path.cwd()
    / "config"
    / "residual_influence_temporal_plan_v1.json"
)

BOOT_REPS = 20000
PERM_REPS = 200000


# --------------------------------------------------
# Canonical historical parser
# --------------------------------------------------

scripts = Path(__file__).resolve().parent

sys.path.insert(
    0,
    str(scripts),
)

analyzer_path = (
    scripts
    / "07_analyze.py"
)

spec = importlib.util.spec_from_file_location(
    "historical_temporal",
    analyzer_path,
)

if (
    spec is None
    or spec.loader is None
):
    raise RuntimeError(
        "could not load 07_analyze.py"
    )

historical = (
    importlib.util.module_from_spec(
        spec
    )
)

sys.modules[
    spec.name
] = historical

spec.loader.exec_module(
    historical
)


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

    try:
        x = float(value)
    except Exception:
        return None

    return (
        x
        if math.isfinite(x)
        else None
    )


def trajectory_path(
    row: dict[str, Any],
) -> Path | None:
    result = str(
        row.get(
            "result_path",
            "",
        )
        or ""
    )

    relative = str(
        row.get(
            "trajectory_file",
            "",
        )
        or ""
    )

    if (
        not result
        or not relative
    ):
        return None

    path = (
        Path(result)
        .expanduser()
        .resolve()
        .parent
        / relative
    )

    return (
        path
        if path.is_file()
        else None
    )


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


# --------------------------------------------------
# Step and action metrics
# --------------------------------------------------

def step_token(
    step: dict[str, Any],
    key: str,
) -> float:
    metrics = step.get(
        "metrics"
    )

    if not isinstance(
        metrics,
        dict,
    ):
        return 0.0

    value = number(
        metrics.get(
            key
        )
    )

    return (
        value
        if value is not None
        else 0.0
    )


def aligned_cutoff(
    n_steps: int,
    fraction: float,
) -> int:
    """Equivalent boundary step at normalized trajectory progress.

    Post period is strictly after this returned step index.
    """
    if n_steps <= 0:
        return -1

    count_through_boundary = int(
        math.ceil(
            fraction
            * n_steps
        )
    )

    count_through_boundary = max(
        1,
        min(
            n_steps,
            count_through_boundary,
        ),
    )

    return (
        count_through_boundary
        - 1
    )


def phase_metrics(
    trajectory: Any,
    *,
    cutoff: int,
    phase: str,
) -> dict[str, Any]:
    steps = (
        historical.agent_steps(
            trajectory
        )
    )

    actions, _ = (
        historical.extract_behavior_actions(
            trajectory
        )
    )

    if phase == "post":
        selected_steps = [
            (
                i,
                step,
            )
            for i, step
            in enumerate(steps)
            if i > cutoff
        ]

        selected_actions = [
            action
            for action in actions
            if int(
                action.get(
                    "step_index",
                    -1,
                )
            ) > cutoff
        ]

    elif phase == "pre":
        selected_steps = [
            (
                i,
                step,
            )
            for i, step
            in enumerate(steps)
            if i < cutoff
        ]

        selected_actions = [
            action
            for action in actions
            if int(
                action.get(
                    "step_index",
                    -1,
                )
            ) < cutoff
        ]

    else:
        raise ValueError(
            phase
        )

    prompt_tokens = sum(
        step_token(
            step,
            "prompt_tokens",
        )
        for _, step
        in selected_steps
    )

    completion_tokens = sum(
        step_token(
            step,
            "completion_tokens",
        )
        for _, step
        in selected_steps
    )

    agent_turns = len(
        selected_steps
    )

    validation_calls = 0
    test_calls = 0
    failed_actions = 0
    network_actions = 0
    edit_write_actions = 0

    categories = Counter()

    for action in (
        selected_actions
    ):
        category = str(
            action.get(
                "category",
                "other",
            )
        )

        categories[
            category
        ] += 1

        command = str(
            action.get(
                "command",
                "",
            )
            or ""
        )

        if (
            command
            and historical.command_matches(
                command,
                historical.TEST_PATTERNS,
            )
        ):
            test_calls += 1

        if (
            command
            and historical.command_matches(
                command,
                historical.VALIDATION_PATTERNS,
            )
        ):
            validation_calls += 1

        failed_actions += int(
            action.get(
                "failed",
                0,
            )
            or 0
        )

        if category in {
            "web_fetch",
            "web_search",
        }:
            network_actions += 1

        if category in {
            "edit",
            "write",
        }:
            edit_write_actions += 1

        low = command.lower()

        if re.search(
            r"\b(?:curl|wget|gh\s+(?:api|search|repo|pr)|git\s+(?:clone|fetch|pull))\b",
            low,
        ):
            network_actions += 1

    return {
        "agent_turns": float(
            agent_turns
        ),
        "prompt_tokens_sum": float(
            prompt_tokens
        ),
        "completion_tokens_sum": float(
            completion_tokens
        ),
        "prompt_tokens_per_turn": (
            float(
                prompt_tokens
                / agent_turns
            )
            if agent_turns
            else None
        ),
        "completion_tokens_per_turn": (
            float(
                completion_tokens
                / agent_turns
            )
            if agent_turns
            else None
        ),
        "canonical_action_calls": float(
            len(
                selected_actions
            )
        ),
        "validation_calls": float(
            validation_calls
        ),
        "test_calls": float(
            test_calls
        ),
        "failed_actions": float(
            failed_actions
        ),
        "network_actions": float(
            network_actions
        ),
        "edit_write_actions": float(
            edit_write_actions
        ),
        "categories": dict(
            categories
        ),
    }


def tv_distance(
    a: dict[str, int],
    b: dict[str, int],
) -> float | None:
    total_a = sum(
        a.values()
    )

    total_b = sum(
        b.values()
    )

    if (
        total_a <= 0
        or total_b <= 0
    ):
        return None

    keys = (
        set(a)
        | set(b)
    )

    return 0.5 * sum(
        abs(
            a.get(
                key,
                0,
            )
            / total_a
            - b.get(
                key,
                0,
            )
            / total_b
        )
        for key in keys
    )


# --------------------------------------------------
# Cluster inference
# --------------------------------------------------

def cluster_means(
    rows: list[
        dict[str, Any]
    ],
    field: str,
) -> np.ndarray:
    grouped = defaultdict(
        list
    )

    for row in rows:
        value = number(
            row.get(
                field
            )
        )

        if value is None:
            continue

        grouped[
            str(
                row[
                    "base_task_id"
                ]
            )
        ].append(
            value
        )

    values = []

    for task in sorted(
        grouped
    ):
        values.append(
            float(
                np.mean(
                    grouped[
                        task
                    ]
                )
            )
        )

    return np.asarray(
        values,
        dtype=float,
    )


def bootstrap_ci(
    values: np.ndarray,
    *,
    seed: int,
    reps: int = BOOT_REPS,
) -> tuple[
    float,
    float,
]:
    if len(values) == 0:
        return (
            float("nan"),
            float("nan"),
        )

    rng = np.random.default_rng(
        seed
    )

    idx = rng.integers(
        0,
        len(values),
        size=(
            reps,
            len(values),
        ),
    )

    boot = (
        values[idx]
        .mean(axis=1)
    )

    lo, hi = np.quantile(
        boot,
        [
            0.025,
            0.975,
        ],
    )

    return (
        float(lo),
        float(hi),
    )


def sign_flip_p(
    values: np.ndarray,
    *,
    seed: int,
    reps: int = PERM_REPS,
) -> float:
    """Two-sided cluster sign-flip randomization test."""
    if len(values) == 0:
        return 1.0

    observed = abs(
        float(
            values.mean()
        )
    )

    rng = np.random.default_rng(
        seed
    )

    extreme = 0
    completed = 0

    chunk = 10000

    while completed < reps:
        n = min(
            chunk,
            reps - completed,
        )

        signs = rng.choice(
            np.asarray(
                [
                    -1.0,
                    1.0,
                ]
            ),
            size=(
                n,
                len(values),
            ),
        )

        perm = abs(
            (
                signs
                * values
            ).mean(
                axis=1
            )
        )

        extreme += int(
            np.sum(
                perm
                >= observed
                - 1e-15
            )
        )

        completed += n

    return float(
        (extreme + 1)
        / (reps + 1)
    )


def holm(
    rows: list[
        dict[str, Any]
    ],
) -> None:
    order = sorted(
        range(
            len(rows)
        ),
        key=lambda i: rows[i][
            "p_value"
        ],
    )

    m = len(rows)
    running = 0.0
    adjusted = [
        1.0
    ] * m

    for rank, index in enumerate(
        order
    ):
        raw = min(
            1.0,
            (
                m - rank
            )
            * rows[
                index
            ][
                "p_value"
            ],
        )

        running = max(
            running,
            raw,
        )

        adjusted[
            index
        ] = min(
            1.0,
            running,
        )

    for row, value in zip(
        rows,
        adjusted,
        strict=True,
    ):
        row[
            "p_holm_6"
        ] = value


# --------------------------------------------------
# Load frozen cohort
# --------------------------------------------------

plan = load(
    PLAN
)

audit = load(
    AUDIT
)

behavior_by_trial = {}
behavior_by_cell = {}

for profile in [
    "claude",
    "fable",
    "codex",
    "llama",
]:
    rows = load(
        FROZEN
        / profile
        / "trials.json"
    )

    assert len(rows) == 700

    for row in rows:
        trial_name = str(
            row[
                "trial_name"
            ]
        )

        behavior_by_trial[
            (
                profile,
                trial_name,
            )
        ] = row

        if row.get(
            "overall_pass"
        ) is None:
            continue

        behavior_by_cell[
            (
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
        ] = row


trajectory_cache = {}


def trajectory_for(
    row: dict[str, Any],
) -> Any:
    trial = str(
        row[
            "trial_name"
        ]
    )

    profile = str(
        row[
            "profile"
        ]
    )

    key = (
        profile,
        trial,
    )

    if key in (
        trajectory_cache
    ):
        return (
            trajectory_cache[
                key
            ]
        )

    path = trajectory_path(
        row
    )

    if path is None:
        raise ValueError(
            f"missing trajectory {key}"
        )

    value = load(
        path
    )

    trajectory_cache[
        key
    ] = value

    return value


# --------------------------------------------------
# Build event-aligned rows.
# --------------------------------------------------

analysis_rows = []

for record in audit[
    "records"
]:
    selected = record.get(
        "selected"
    )

    if selected is None:
        continue

    profile = str(
        record[
            "profile"
        ]
    )

    confidence = str(
        selected[
            "confidence"
        ]
    )

    if profile not in {
        "claude",
        "fable",
    }:
        continue

    pressure_row = (
        behavior_by_trial[
            (
                profile,
                str(
                    record[
                        "trial_name"
                    ]
                ),
            )
        ]
    )

    task = str(
        record[
            "base_task_id"
        ]
    )

    condition = str(
        record[
            "condition"
        ]
    )

    source_eval = (
        behavior_by_cell.get(
            (
                profile,
                task,
                "eval_only",
                "source",
            )
        )
    )

    root_pressure = (
        behavior_by_cell.get(
            (
                profile,
                task,
                condition,
                "root",
            )
        )
    )

    root_eval = (
        behavior_by_cell.get(
            (
                profile,
                task,
                "eval_only",
                "root",
            )
        )
    )

    if any(
        row is None
        for row in [
            source_eval,
            root_pressure,
            root_eval,
        ]
    ):
        continue

    source_pressure_traj = (
        trajectory_for(
            pressure_row
        )
    )

    source_eval_traj = (
        trajectory_for(
            source_eval
        )
    )

    root_pressure_traj = (
        trajectory_for(
            root_pressure
        )
    )

    root_eval_traj = (
        trajectory_for(
            root_eval
        )
    )

    source_pressure_steps = (
        historical.agent_steps(
            source_pressure_traj
        )
    )

    event_step = int(
        selected[
            "step_index"
        ]
    )

    if (
        event_step < 0
        or event_step
        >= len(
            source_pressure_steps
        )
    ):
        continue

    fraction = (
        (event_step + 1)
        / len(
            source_pressure_steps
        )
    )

    cutoffs = {
        "source_pressure": (
            event_step
        ),
        "source_eval": (
            aligned_cutoff(
                len(
                    historical.agent_steps(
                        source_eval_traj
                    )
                ),
                fraction,
            )
        ),
        "root_pressure": (
            aligned_cutoff(
                len(
                    historical.agent_steps(
                        root_pressure_traj
                    )
                ),
                fraction,
            )
        ),
        "root_eval": (
            aligned_cutoff(
                len(
                    historical.agent_steps(
                        root_eval_traj
                    )
                ),
                fraction,
            )
        ),
    }

    trajectories = {
        "source_pressure": (
            source_pressure_traj
        ),
        "source_eval": (
            source_eval_traj
        ),
        "root_pressure": (
            root_pressure_traj
        ),
        "root_eval": (
            root_eval_traj
        ),
    }

    post = {
        name: phase_metrics(
            trajectory,
            cutoff=cutoffs[
                name
            ],
            phase="post",
        )
        for name, trajectory
        in trajectories.items()
    }

    pre = {
        name: phase_metrics(
            trajectory,
            cutoff=cutoffs[
                name
            ],
            phase="pre",
        )
        for name, trajectory
        in trajectories.items()
    }

    row = {
        "profile": profile,
        "base_task_id": task,
        "trial_name": (
            record[
                "trial_name"
            ]
        ),
        "condition": condition,
        "pressure_type": (
            record[
                "pressure_type"
            ]
        ),
        "event_confidence": (
            confidence
        ),
        "event_step": (
            event_step
        ),
        "event_fraction": (
            fraction
        ),
        "source_pressure_post_actions": (
            post[
                "source_pressure"
            ][
                "canonical_action_calls"
            ]
        ),
    }

    scalar_metrics = [
        "agent_turns",
        "prompt_tokens_sum",
        "completion_tokens_sum",
        "prompt_tokens_per_turn",
        "completion_tokens_per_turn",
        "canonical_action_calls",
        "validation_calls",
        "test_calls",
        "failed_actions",
        "network_actions",
        "edit_write_actions",
    ]

    for metric in (
        scalar_metrics
    ):
        values = {
            name: number(
                post[
                    name
                ].get(
                    metric
                )
            )
            for name
            in trajectories
        }

        if all(
            value is not None
            for value
            in values.values()
        ):
            source_residual = (
                values[
                    "source_pressure"
                ]
                - values[
                    "source_eval"
                ]
            )

            root_residual = (
                values[
                    "root_pressure"
                ]
                - values[
                    "root_eval"
                ]
            )

            row[
                f"source_post_{metric}_residual"
            ] = source_residual

            row[
                f"root_post_{metric}_residual"
            ] = root_residual

            row[
                f"source_specific_post_{metric}"
            ] = (
                source_residual
                - root_residual
            )

    source_post_tv = (
        tv_distance(
            post[
                "source_pressure"
            ][
                "categories"
            ],
            post[
                "source_eval"
            ][
                "categories"
            ],
        )
    )

    root_post_tv = (
        tv_distance(
            post[
                "root_pressure"
            ][
                "categories"
            ],
            post[
                "root_eval"
            ][
                "categories"
            ],
        )
    )

    source_pre_tv = (
        tv_distance(
            pre[
                "source_pressure"
            ][
                "categories"
            ],
            pre[
                "source_eval"
            ][
                "categories"
            ],
        )
    )

    root_pre_tv = (
        tv_distance(
            pre[
                "root_pressure"
            ][
                "categories"
            ],
            pre[
                "root_eval"
            ][
                "categories"
            ],
        )
    )

    if (
        source_post_tv
        is not None
        and root_post_tv
        is not None
    ):
        row[
            "source_post_tv"
        ] = source_post_tv

        row[
            "root_post_tv"
        ] = root_post_tv

        row[
            "source_specific_post_action_profile_tv"
        ] = (
            source_post_tv
            - root_post_tv
        )

    if (
        source_pre_tv
        is not None
        and root_pre_tv
        is not None
    ):
        row[
            "source_specific_pre_action_profile_tv"
        ] = (
            source_pre_tv
            - root_pre_tv
        )

    if (
        row.get(
            "source_specific_post_action_profile_tv"
        )
        is not None
        and row.get(
            "source_specific_pre_action_profile_tv"
        )
        is not None
    ):
        row[
            "post_minus_pre_tv_delta"
        ] = (
            row[
                "source_specific_post_action_profile_tv"
            ]
            - row[
                "source_specific_pre_action_profile_tv"
            ]
        )

    analysis_rows.append(
        row
    )


# --------------------------------------------------
# Cohorts
# --------------------------------------------------

cohorts = {
    "claude_A_primary": [
        row
        for row in analysis_rows
        if (
            row[
                "profile"
            ]
            == "claude"
            and row[
                "event_confidence"
            ]
            == "A"
        )
    ],
    "claude_A_post5_sensitivity": [
        row
        for row in analysis_rows
        if (
            row[
                "profile"
            ]
            == "claude"
            and row[
                "event_confidence"
            ]
            == "A"
            and row[
                "source_pressure_post_actions"
            ]
            >= 5
        )
    ],
    "claude_AD_broad_sensitivity": [
        row
        for row in analysis_rows
        if (
            row[
                "profile"
            ]
            == "claude"
            and row[
                "event_confidence"
            ]
            in {
                "A",
                "D",
            }
        )
    ],
    "fable_A_descriptive": [
        row
        for row in analysis_rows
        if (
            row[
                "profile"
            ]
            == "fable"
            and row[
                "event_confidence"
            ]
            == "A"
        )
    ],
}


# --------------------------------------------------
# Primary Claude A inference
# --------------------------------------------------

primary_fields = {
    "prompt_tokens_sum": (
        "source_specific_post_prompt_tokens_sum"
    ),
    "completion_tokens_sum": (
        "source_specific_post_completion_tokens_sum"
    ),
    "canonical_action_calls": (
        "source_specific_post_canonical_action_calls"
    ),
    "validation_calls": (
        "source_specific_post_validation_calls"
    ),
    "test_calls": (
        "source_specific_post_test_calls"
    ),
    "action_profile_tv_delta": (
        "source_specific_post_action_profile_tv"
    ),
}

primary_results = []

primary_rows = (
    cohorts[
        "claude_A_primary"
    ]
)

for endpoint, field in (
    primary_fields.items()
):
    values = cluster_means(
        primary_rows,
        field,
    )

    estimate = (
        float(
            values.mean()
        )
        if len(values)
        else float("nan")
    )

    lo, hi = bootstrap_ci(
        values,
        seed=seed_for(
            "residual-v1",
            "primary-bootstrap",
            endpoint,
        ),
    )

    p = sign_flip_p(
        values,
        seed=seed_for(
            "residual-v1",
            "primary-signflip",
            endpoint,
        ),
    )

    primary_results.append({
        "endpoint": endpoint,
        "field": field,
        "trajectory_n": sum(
            1
            for row
            in primary_rows
            if number(
                row.get(
                    field
                )
            )
            is not None
        ),
        "cluster_n": int(
            len(values)
        ),
        "estimate": (
            estimate
        ),
        "ci_low": lo,
        "ci_high": hi,
        "p_value": p,
    })


assert len(
    primary_results
) == 6

holm(
    primary_results
)


# --------------------------------------------------
# Supporting source/root decomposition +
# sensitivity CIs without new hypothesis tests.
# --------------------------------------------------

support_rows = []

support_metrics = [
    "prompt_tokens_sum",
    "completion_tokens_sum",
    "canonical_action_calls",
    "validation_calls",
    "test_calls",
    "failed_actions",
    "network_actions",
    "edit_write_actions",
    "prompt_tokens_per_turn",
    "completion_tokens_per_turn",
]

for cohort_name, rows in (
    cohorts.items()
):
    for metric in (
        support_metrics
    ):
        for component, field in [
            (
                "source_pressure_minus_source_eval",
                f"source_post_{metric}_residual",
            ),
            (
                "root_pressure_minus_root_eval",
                f"root_post_{metric}_residual",
            ),
            (
                "source_specific_did",
                f"source_specific_post_{metric}",
            ),
        ]:
            values = cluster_means(
                rows,
                field,
            )

            if len(values) == 0:
                continue

            lo, hi = bootstrap_ci(
                values,
                seed=seed_for(
                    "residual-v1",
                    "support",
                    cohort_name,
                    metric,
                    component,
                ),
            )

            support_rows.append({
                "cohort": (
                    cohort_name
                ),
                "metric": metric,
                "component": (
                    component
                ),
                "trajectory_n": sum(
                    1
                    for row
                    in rows
                    if number(
                        row.get(
                            field
                        )
                    )
                    is not None
                ),
                "cluster_n": (
                    int(
                        len(values)
                    )
                ),
                "estimate": (
                    float(
                        values.mean()
                    )
                ),
                "ci_low": lo,
                "ci_high": hi,
            })

    for metric, field in [
        (
            "action_profile_tv_delta",
            "source_specific_post_action_profile_tv",
        ),
        (
            "post_minus_pre_tv_delta",
            "post_minus_pre_tv_delta",
        ),
        (
            "source_post_tv",
            "source_post_tv",
        ),
        (
            "root_post_tv",
            "root_post_tv",
        ),
    ]:
        values = cluster_means(
            rows,
            field,
        )

        if len(values) == 0:
            continue

        lo, hi = bootstrap_ci(
            values,
            seed=seed_for(
                "residual-v1",
                "support-tv",
                cohort_name,
                metric,
            ),
        )

        support_rows.append({
            "cohort": cohort_name,
            "metric": metric,
            "component": (
                "descriptive"
            ),
            "trajectory_n": sum(
                1
                for row
                in rows
                if number(
                    row.get(
                        field
                    )
                )
                is not None
            ),
            "cluster_n": (
                int(
                    len(values)
                )
            ),
            "estimate": (
                float(
                    values.mean()
                )
            ),
            "ci_low": lo,
            "ci_high": hi,
        })


# --------------------------------------------------
# Outputs
# --------------------------------------------------

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

event_csv = (
    OUTDIR
    / "event_aligned_rows.csv"
)

if analysis_rows:
    fields = sorted(
        set().union(
            *[
                row.keys()
                for row
                in analysis_rows
            ]
        )
    )

    with event_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for row in analysis_rows:
            writer.writerow(
                row
            )


primary_csv = (
    OUTDIR
    / "primary_claude_A_inference.csv"
)

with primary_csv.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(
            primary_results[
                0
            ].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        primary_results
    )


support_csv = (
    OUTDIR
    / "supporting_and_sensitivity.csv"
)

with support_csv.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:
    fields = [
        "cohort",
        "metric",
        "component",
        "trajectory_n",
        "cluster_n",
        "estimate",
        "ci_low",
        "ci_high",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fields,
    )

    writer.writeheader()
    writer.writerows(
        support_rows
    )


summary = {
    "analysis_version": "1.0",
    "interpretation": (
        "post-treatment exploratory temporal mechanism analysis; "
        "not causal"
    ),
    "plan_sha256": (
        sha(PLAN)
    ),
    "event_audit_sha256": (
        sha(AUDIT)
    ),
    "event_aligned_rows": (
        len(
            analysis_rows
        )
    ),
    "cohort_counts": {
        name: len(rows)
        for name, rows
        in cohorts.items()
    },
    "primary_endpoint_count": (
        len(
            primary_results
        )
    ),
    "primary_holm_significant": sum(
        row[
            "p_holm_6"
        ] < 0.05
        for row
        in primary_results
    ),
}

(
    OUTDIR
    / "summary.json"
).write_text(
    json.dumps(
        summary,
        indent=2,
        sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)


# --------------------------------------------------
# Terminal report
# --------------------------------------------------

print(
    "RESIDUAL INFLUENCE TEMPORAL ANALYSIS V1"
)

print("=" * 100)

print(
    "plan sha256:",
    sha(
        PLAN
    ),
)

print(
    "event audit sha256:",
    sha(
        AUDIT
    ),
)

print(
    "event-aligned rows:",
    len(
        analysis_rows
    ),
)

print(
    "cohorts:",
    summary[
        "cohort_counts"
    ],
)

print()
print(
    "PRIMARY: CLAUDE HIGH-CONFIDENCE A REMOVALS"
)

print("-" * 100)

for row in (
    primary_results
):
    print(
        f"{row['endpoint']:30s} "
        f"traj={row['trajectory_n']:2d} "
        f"clusters={row['cluster_n']:2d} "
        f"effect={row['estimate']:+12.3f} "
        f"CI=[{row['ci_low']:+12.3f},"
        f"{row['ci_high']:+12.3f}] "
        f"p={row['p_value']:.6g} "
        f"Holm6={row['p_holm_6']:.6g}"
    )


print()
print(
    "PRIMARY SOURCE / ROOT DECOMPOSITION"
)

print("-" * 100)

wanted = {
    "prompt_tokens_sum",
    "completion_tokens_sum",
    "canonical_action_calls",
    "validation_calls",
    "test_calls",
    "action_profile_tv_delta",
    "post_minus_pre_tv_delta",
}

for row in support_rows:
    if (
        row[
            "cohort"
        ]
        != "claude_A_primary"
        or row[
            "metric"
        ]
        not in wanted
    ):
        continue

    print(
        f"{row['metric']:30s} "
        f"{row['component']:38s} "
        f"clusters={row['cluster_n']:2d} "
        f"effect={row['estimate']:+12.3f} "
        f"CI=[{row['ci_low']:+12.3f},"
        f"{row['ci_high']:+12.3f}]"
    )


print()
print(
    "SENSITIVITY: CLAUDE A WITH >=5 POST ACTIONS"
)

print("-" * 100)

for row in support_rows:
    if (
        row[
            "cohort"
        ]
        != "claude_A_post5_sensitivity"
        or row[
            "component"
        ]
        != "source_specific_did"
        or row[
            "metric"
        ]
        not in {
            "prompt_tokens_sum",
            "completion_tokens_sum",
            "canonical_action_calls",
            "validation_calls",
            "test_calls",
        }
    ):
        continue

    print(
        f"{row['metric']:30s} "
        f"clusters={row['cluster_n']:2d} "
        f"effect={row['estimate']:+12.3f} "
        f"CI=[{row['ci_low']:+12.3f},"
        f"{row['ci_high']:+12.3f}]"
    )


print()
print(
    "FABLE A: DESCRIPTIVE ONLY"
)

print("-" * 100)

for row in support_rows:
    if (
        row[
            "cohort"
        ]
        != "fable_A_descriptive"
        or row[
            "component"
        ]
        != "source_specific_did"
        or row[
            "metric"
        ]
        not in {
            "prompt_tokens_sum",
            "completion_tokens_sum",
            "canonical_action_calls",
            "validation_calls",
            "test_calls",
        }
    ):
        continue

    print(
        f"{row['metric']:30s} "
        f"clusters={row['cluster_n']:2d} "
        f"effect={row['estimate']:+12.3f} "
        f"CI=[{row['ci_low']:+12.3f},"
        f"{row['ci_high']:+12.3f}]"
    )


print()
print(
    "primary Holm-significant:",
    summary[
        "primary_holm_significant"
    ],
    "/ 6",
)

print("network calls: 0")
print("judge calls: 0")

print(
    "RESIDUAL INFLUENCE TEMPORAL ANALYSIS: PASS"
)
