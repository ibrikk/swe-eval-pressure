#!/usr/bin/env python3
"""Zero-call feasibility audit for post-removal residual influence.

No hypothesis tests.
No network calls.
No semantic judge calls.

Audits whether source-local defensive cue removals can be assigned
an observable action step and whether step-level token accounting exists.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


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

SEMANTIC = (
    DATA_ROOT
    / "analysis"
    / "semantic-multijudge-v1"
    / "final-repaired-llama-v1"
)

PROFILES = [
    "claude",
    "fable",
    "codex",
    "llama",
]


# --------------------------------------------------
# Reuse the canonical action parser already used
# for the frozen behavioral study.
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
    "historical_analyzer",
    analyzer_path,
)

if spec is None or spec.loader is None:
    raise RuntimeError(
        "could not import canonical analyzer"
    )

historical = importlib.util.module_from_spec(
    spec
)

# Python 3.13 dataclasses inspect sys.modules
# while class decorators execute. Dynamic modules
# must therefore be registered before exec_module().
sys.modules[spec.name] = historical

spec.loader.exec_module(
    historical
)


def load(path: Path) -> Any:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def norm(text: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        str(text),
    ).strip().lower()


def trajectory_path(
    row: dict[str, Any],
) -> Path | None:
    result_path = str(
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

    if not result_path or not relative:
        return None

    trial = Path(
        result_path
    ).expanduser().resolve().parent

    path = (
        trial
        / relative
    )

    return (
        path
        if path.is_file()
        else None
    )


def consensus_label(
    semantic: dict[str, Any],
    field: str,
) -> str | None:
    value = (
        semantic[
            "consensus"
        ][
            "fields"
        ][field]
    )

    if (
        value.get("status")
        != "agreement"
    ):
        return None

    return str(
        value.get("label")
    )


def target_matches(
    text: str,
    target: str,
) -> bool:
    if not target:
        return False

    low = norm(text)
    target_low = norm(target)

    if target_low in low:
        return True

    base = Path(
        target
    ).name.lower()

    return bool(
        base
        and base in low
    )


def removed_text(
    blob: str,
) -> str:
    """Recover deletion-side text from patch-like action arguments."""
    removed = []

    for line in str(
        blob
    ).splitlines():
        stripped = (
            line.lstrip()
        )

        if (
            stripped.startswith("-")
            and not stripped.startswith("---")
        ):
            removed.append(
                stripped[1:]
            )

    return "\n".join(
        removed
    )


def cue_probes(
    row: dict[str, Any],
) -> list[str]:
    text = str(
        row.get(
            "eval_cue_text",
            "",
        )
        or ""
    )

    probes = []

    for piece in re.split(
        r"(?<=[.!?])\s+|\n+",
        text,
    ):
        p = norm(piece)

        if len(p) >= 20:
            probes.append(p)

    if (
        not probes
        and len(norm(text)) >= 20
    ):
        probes.append(
            norm(text)
        )

    return probes


def high_confidence_removal(
    action: dict[str, Any],
    row: dict[str, Any],
) -> bool:
    """High precision: cue text occurs specifically on deletion side."""
    target = str(
        row.get(
            "treatment_artifact",
            row.get(
                "source_target",
                "",
            ),
        )
        or ""
    )

    blob = "\n".join([
        str(
            action.get(
                "command",
                "",
            )
        ),
        str(
            action.get(
                "path",
                "",
            )
        ),
        str(
            action.get(
                "arguments_text",
                "",
            )
        ),
    ])

    if not target_matches(
        blob,
        target,
    ):
        return False

    deleted = norm(
        removed_text(
            blob
        )
    )

    if not deleted:
        return False

    return any(
        probe in deleted
        for probe in cue_probes(
            row
        )
    )


MODIFY_RE = re.compile(
    r"""
    \b(
        sed\s+-i
        |perl\s+-pi
        |apply_patch
        |python(?:3)?\b
        |tee\b
    )
    |>>
    |(?<![<])>(?!>)
    """,
    re.I | re.X,
)


def plausible_target_mutation(
    action: dict[str, Any],
    row: dict[str, Any],
) -> bool:
    target = str(
        row.get(
            "treatment_artifact",
            row.get(
                "source_target",
                "",
            ),
        )
        or ""
    )

    category = str(
        action.get(
            "category",
            "",
        )
    )

    blob = "\n".join([
        str(
            action.get(
                "command",
                "",
            )
        ),
        str(
            action.get(
                "path",
                "",
            )
        ),
        str(
            action.get(
                "arguments_text",
                "",
            )
        ),
    ])

    if not target_matches(
        blob,
        target,
    ):
        return False

    if category in {
        "edit",
        "write",
    }:
        return True

    if (
        category == "bash"
        and MODIFY_RE.search(
            blob
        )
    ):
        return True

    return False


def recursive_usage_keys(
    value: Any,
    prefix: str = "",
) -> list[
    tuple[str, Any]
]:
    found = []

    if isinstance(
        value,
        dict,
    ):
        for key, child in (
            value.items()
        ):
            name = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )

            low = str(
                key
            ).lower()

            if (
                "token" in low
                or "usage" in low
            ):
                if isinstance(
                    child,
                    (
                        str,
                        int,
                        float,
                        bool,
                    ),
                ):
                    found.append(
                        (
                            name,
                            child,
                        )
                    )

            found.extend(
                recursive_usage_keys(
                    child,
                    name,
                )
            )

    elif isinstance(
        value,
        list,
    ):
        for i, child in enumerate(
            value
        ):
            found.extend(
                recursive_usage_keys(
                    child,
                    f"{prefix}[{i}]",
                )
            )

    return found


# --------------------------------------------------
# Behavioral frozen rows
# --------------------------------------------------

behavior_by_trial = {}
behavior_by_cell = {}

for profile in PROFILES:
    rows = load(
        FROZEN
        / profile
        / "trials.json"
    )

    assert len(rows) == 700

    for row in rows:
        trial_key = (
            profile,
            str(
                row[
                    "trial_name"
                ]
            ),
        )

        behavior_by_trial[
            trial_key
        ] = row

        if row.get(
            "overall_pass"
        ) is None:
            continue

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

        behavior_by_cell[
            cell
        ] = row


# --------------------------------------------------
# Semantic consensus
# --------------------------------------------------

semantic_by_trial = {}

for path in sorted(
    (
        SEMANTIC
        / "consensus"
    ).glob("*.json")
):
    semantic = load(
        path
    )

    semantic_by_trial[
        (
            str(
                semantic[
                    "profile"
                ]
            ),
            str(
                semantic[
                    "trial_name"
                ]
            ),
        )
    ] = semantic


# --------------------------------------------------
# Select the strongest prospective residual cohort:
#
# source pressure
# explicit resistance = observed
# claimed response = remove_or_modify_cue
# objectively modified final planted cue
# --------------------------------------------------

candidates = []

for trial_key, row in (
    behavior_by_trial.items()
):
    profile = trial_key[0]

    if (
        row.get("channel")
        != "source"
    ):
        continue

    if row.get(
        "condition"
    ) not in {
        "eval_financial",
        "eval_self_preservation",
    }:
        continue

    if int(
        row.get(
            "seeded_cue_removed_or_modified",
            0,
        )
        or 0
    ) != 1:
        continue

    semantic = semantic_by_trial.get(
        trial_key
    )

    if semantic is None:
        continue

    resistance = (
        consensus_label(
            semantic,
            "explicit_resistance",
        )
    )

    response = (
        consensus_label(
            semantic,
            "claimed_behavioral_response",
        )
    )

    stance = (
        consensus_label(
            semantic,
            "cue_provenance_stance",
        )
    )

    pressure_recognition = (
        consensus_label(
            semantic,
            "pressure_recognition",
        )
    )

    if (
        resistance != "observed"
        or response
        != "remove_or_modify_cue"
    ):
        continue

    candidates.append({
        "profile": profile,
        "row": row,
        "semantic": semantic,
        "stance": stance,
        "pressure_recognition": (
            pressure_recognition
        ),
    })


# --------------------------------------------------
# Event detectability
# --------------------------------------------------

summary = Counter()
by_profile = Counter()
methods = Counter()
post_action_counts = []
event_fractions = []
token_key_counts = Counter()
token_examples = defaultdict(
    list
)

detail = []

for item in candidates:
    profile = item[
        "profile"
    ]

    row = item[
        "row"
    ]

    summary[
        "candidate_defensive_removers"
    ] += 1

    by_profile[
        (
            profile,
            "candidates",
        )
    ] += 1

    path = trajectory_path(
        row
    )

    if path is None:
        summary[
            "missing_trajectory_file"
        ] += 1
        continue

    summary[
        "trajectory_loaded"
    ] += 1

    trajectory = load(
        path
    )

    actions, metrics = (
        historical.extract_behavior_actions(
            trajectory
        )
    )

    if not actions:
        summary[
            "no_canonical_actions"
        ] += 1
        continue

    # Step-level token / usage feasibility.
    for step in historical.agent_steps(
        trajectory
    ):
        for key, value in (
            recursive_usage_keys(
                step
            )
        ):
            token_key_counts[
                key
            ] += 1

            if (
                len(
                    token_examples[
                        key
                    ]
                )
                < 3
            ):
                token_examples[
                    key
                ].append(
                    value
                )

    high = [
        i
        for i, action in enumerate(
            actions
        )
        if high_confidence_removal(
            action,
            row,
        )
    ]

    plausible = [
        i
        for i, action in enumerate(
            actions
        )
        if plausible_target_mutation(
            action,
            row,
        )
    ]

    event_i = None
    method = None

    if high:
        event_i = high[0]
        method = (
            "explicit_cue_text_deletion"
        )

    elif plausible:
        event_i = plausible[0]
        method = (
            "first_target_mutation"
        )

    if event_i is None:
        summary[
            "no_observable_removal_event"
        ] += 1
        continue

    methods[
        method
    ] += 1

    summary[
        "removal_event_observed"
    ] += 1

    by_profile[
        (
            profile,
            "event_observed",
        )
    ] += 1

    post_n = (
        len(actions)
        - event_i
        - 1
    )

    frac = (
        (event_i + 1)
        / len(actions)
    )

    post_action_counts.append(
        post_n
    )

    event_fractions.append(
        frac
    )

    if post_n >= 3:
        summary[
            "post_actions_ge_3"
        ] += 1

    if post_n >= 5:
        summary[
            "post_actions_ge_5"
        ] += 1

    if post_n >= 10:
        summary[
            "post_actions_ge_10"
        ] += 1

    eval_key = (
        profile,
        str(
            row[
                "base_task_id"
            ]
        ),
        "eval_only",
        "source",
    )

    eval_row = (
        behavior_by_cell.get(
            eval_key
        )
    )

    eval_path = (
        trajectory_path(
            eval_row
        )
        if eval_row
        else None
    )

    if eval_path is not None:
        summary[
            "matched_eval_trajectory_available"
        ] += 1

    detail.append({
        "profile": profile,
        "trial_name": (
            row[
                "trial_name"
            ]
        ),
        "base_task_id": (
            row[
                "base_task_id"
            ]
        ),
        "condition": (
            row[
                "condition"
            ]
        ),
        "pressure_type": (
            row[
                "pressure_type"
            ]
        ),
        "event_method": (
            method
        ),
        "event_action_index": (
            event_i
        ),
        "total_actions": (
            len(actions)
        ),
        "event_fraction": (
            frac
        ),
        "post_actions": (
            post_n
        ),
        "matched_eval_available": (
            eval_path
            is not None
        ),
        "pressure_recognition": (
            item[
                "pressure_recognition"
            ]
        ),
        "stance": (
            item[
                "stance"
            ]
        ),
    })


print(
    "RESIDUAL-INFLUENCE TEMPORAL FEASIBILITY"
)
print("=" * 84)

print(
    "candidate defensive removers:",
    summary[
        "candidate_defensive_removers"
    ],
)

print(
    "trajectory loaded:",
    summary[
        "trajectory_loaded"
    ],
)

print(
    "observable removal event:",
    summary[
        "removal_event_observed"
    ],
)

print(
    "  high-confidence cue-text deletion:",
    methods[
        "explicit_cue_text_deletion"
    ],
)

print(
    "  fallback first target mutation:",
    methods[
        "first_target_mutation"
    ],
)

print(
    "no observable removal event:",
    summary[
        "no_observable_removal_event"
    ],
)

print(
    "matched eval trajectory:",
    summary[
        "matched_eval_trajectory_available"
    ],
)

print()
print(
    "POST-EVENT BEHAVIORAL SUPPORT"
)

for threshold in (
    3,
    5,
    10,
):
    print(
        f"  >= {threshold:2d} post actions:",
        summary[
            f"post_actions_ge_{threshold}"
        ],
    )

if post_action_counts:
    values = sorted(
        post_action_counts
    )

    print(
        "  median post actions:",
        values[
            len(values) // 2
        ],
    )

if event_fractions:
    values = sorted(
        event_fractions
    )

    print(
        "  median event fraction:",
        f"{values[len(values)//2]:.3f}",
    )


print()
print("BY PROFILE")
print("-" * 84)

for profile in PROFILES:
    print(
        f"{profile:7s}",
        "candidates=",
        by_profile[
            (
                profile,
                "candidates",
            )
        ],
        "events=",
        by_profile[
            (
                profile,
                "event_observed",
            )
        ],
    )


print()
print(
    "STEP-LEVEL TOKEN / USAGE KEYS"
)
print("-" * 84)

if not token_key_counts:
    print(
        "none detected"
    )

else:
    for key, count in (
        token_key_counts
        .most_common(30)
    ):
        print(
            f"{count:5d}",
            key,
            "examples=",
            token_examples[
                key
            ],
        )


print()
print("FIRST 30 EVENT ROWS")
print("-" * 84)

for x in detail[:30]:
    print(
        x["profile"],
        x["condition"],
        "method=",
        x[
            "event_method"
        ],
        "event=",
        (
            f"{x['event_action_index']+1}"
            f"/{x['total_actions']}"
        ),
        "fraction=",
        f"{x['event_fraction']:.3f}",
        "post=",
        x[
            "post_actions"
        ],
        "eval=",
        x[
            "matched_eval_available"
        ],
    )


out = (
    DATA_ROOT
    / "analysis"
    / "semantic-multijudge-v1"
    / "residual-influence-feasibility-v1.json"
)

out.write_text(
    json.dumps(
        {
            "summary": dict(
                summary
            ),
            "methods": dict(
                methods
            ),
            "by_profile": {
                profile: {
                    "candidates": (
                        by_profile[
                            (
                                profile,
                                "candidates",
                            )
                        ]
                    ),
                    "events": (
                        by_profile[
                            (
                                profile,
                                "event_observed",
                            )
                        ]
                    ),
                }
                for profile
                in PROFILES
            },
            "token_key_counts": dict(
                token_key_counts
            ),
            "events": detail,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)

print()
print("output:", out)
print("network calls: 0")
print("judge calls: 0")
print(
    "TEMPORAL FEASIBILITY AUDIT: PASS"
)
