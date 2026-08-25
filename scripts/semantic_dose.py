#!/usr/bin/env python3
"""Deterministic 100-trajectory semantic dose selection.

Stage C selection only:
- 10 trajectories per treatment cell;
- 25 trajectories per evaluated-agent profile;
- exact frozen 10-trajectory pilot included as a subset;
- remaining trajectories chosen deterministically;
- no semantic outcomes consulted;
- NO network/API calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import semantic_bulk_runner as pilot


DOSE_SPEC_VERSION = "1.0"

PROFILES = (
    "claude",
    "fable",
    "codex",
    "llama",
)

CELLS = pilot.PILOT_CELLS


# Number of trajectories drawn from each profile
# within each treatment cell.
#
# Every row sums to 10.
# Every profile column sums to 25.
#
# The 3/3/2/2 allocation rotates across cells
# so no profile is systematically oversampled
# in one condition/placement family.
CELL_PROFILE_COUNTS = {
    ("clean", "none"): {
        "claude": 3,
        "fable": 3,
        "codex": 2,
        "llama": 2,
    },
    ("eval_only", "root"): {
        "claude": 3,
        "fable": 2,
        "codex": 3,
        "llama": 2,
    },
    ("eval_only", "scaffold"): {
        "claude": 3,
        "fable": 2,
        "codex": 2,
        "llama": 3,
    },
    ("eval_only", "source"): {
        "claude": 2,
        "fable": 3,
        "codex": 3,
        "llama": 2,
    },
    ("eval_financial", "root"): {
        "claude": 2,
        "fable": 3,
        "codex": 2,
        "llama": 3,
    },
    ("eval_financial", "scaffold"): {
        "claude": 2,
        "fable": 2,
        "codex": 3,
        "llama": 3,
    },
    ("eval_financial", "source"): {
        "claude": 3,
        "fable": 3,
        "codex": 2,
        "llama": 2,
    },
    (
        "eval_self_preservation",
        "root",
    ): {
        "claude": 3,
        "fable": 2,
        "codex": 3,
        "llama": 2,
    },
    (
        "eval_self_preservation",
        "scaffold",
    ): {
        "claude": 2,
        "fable": 3,
        "codex": 2,
        "llama": 3,
    },
    (
        "eval_self_preservation",
        "source",
    ): {
        "claude": 2,
        "fable": 2,
        "codex": 3,
        "llama": 3,
    },
}


def load_json(
    path: Path,
) -> dict[str, Any]:
    value = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(value, dict):
        raise ValueError(
            f"{path}: expected JSON object"
        )

    return value


def dose_hash(
    *,
    profile: str,
    condition: str,
    placement: str,
    trial_name: str,
) -> str:
    value = "|".join([
        profile,
        condition,
        placement,
        trial_name,
    ])

    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def validate_allocation() -> None:
    if set(
        CELL_PROFILE_COUNTS
    ) != set(CELLS):
        raise ValueError(
            "dose allocation cells do not "
            "match treatment cells"
        )

    totals = Counter()

    for cell in CELLS:
        counts = (
            CELL_PROFILE_COUNTS[
                cell
            ]
        )

        if set(counts) != set(
            PROFILES
        ):
            raise ValueError(
                f"{cell}: profile set mismatch"
            )

        if sum(
            counts.values()
        ) != 10:
            raise ValueError(
                f"{cell}: expected cell "
                "allocation of 10"
            )

        for profile, count in (
            counts.items()
        ):
            if count not in {
                2,
                3,
            }:
                raise ValueError(
                    f"{cell} {profile}: "
                    "allocation must be 2 or 3"
                )

            totals[
                profile
            ] += count

    if dict(totals) != {
        "claude": 25,
        "fable": 25,
        "codex": 25,
        "llama": 25,
    }:
        raise ValueError(
            "dose profile totals "
            f"are incorrect: {dict(totals)}"
        )


def pilot_identity(
    trial: dict[str, Any],
) -> tuple[str, str]:
    return (
        str(trial["profile"]),
        str(trial["trial_name"]),
    )


def build_dose(
    *,
    manifest: dict[str, Any],
    pilot_spec: dict[str, Any],
) -> dict[str, Any]:
    validate_allocation()

    all_trials = (
        pilot.collapse_manifest_jobs(
            manifest["jobs"]
        )
    )

    frozen_pilot = [
        dict(trial)
        for trial
        in pilot_spec["trials"]
    ]

    if len(
        frozen_pilot
    ) != 10:
        raise ValueError(
            "expected exactly 10 "
            "frozen pilot trajectories"
        )

    pilot_ids = {
        pilot_identity(trial)
        for trial in frozen_pilot
    }

    if len(
        pilot_ids
    ) != 10:
        raise ValueError(
            "duplicate pilot identity"
        )

    selected: list[
        dict[str, Any]
    ] = []

    selected_ids: set[
        tuple[str, str]
    ] = set()

    for cell in CELLS:
        (
            condition,
            placement,
        ) = cell

        allocation = (
            CELL_PROFILE_COUNTS[
                cell
            ]
        )

        for profile in PROFILES:
            target = int(
                allocation[
                    profile
                ]
            )

            pilot_here = [
                frozen
                for frozen
                in frozen_pilot
                if (
                    str(
                        frozen[
                            "profile"
                        ]
                    )
                    == profile
                    and str(
                        frozen[
                            "condition"
                        ]
                    )
                    == condition
                    and str(
                        frozen[
                            "placement"
                        ]
                    )
                    == placement
                )
            ]

            if len(
                pilot_here
            ) > 1:
                raise ValueError(
                    "more than one frozen "
                    "pilot trajectory in the "
                    "same profile/cell"
                )

            candidates = [
                trial
                for trial
                in all_trials.values()
                if (
                    str(
                        trial[
                            "profile"
                        ]
                    )
                    == profile
                    and str(
                        trial[
                            "condition"
                        ]
                    )
                    == condition
                    and str(
                        trial[
                            "placement"
                        ]
                    )
                    == placement
                )
            ]

            candidates.sort(
                key=lambda trial: (
                    dose_hash(
                        profile=profile,
                        condition=condition,
                        placement=placement,
                        trial_name=str(
                            trial[
                                "trial_name"
                            ]
                        ),
                    ),
                    str(
                        trial[
                            "trial_name"
                        ]
                    ),
                )
            )

            chosen: list[
                dict[str, Any]
            ] = []

            # The frozen pilot is included
            # explicitly, regardless of its
            # rank under the dose hash.
            if pilot_here:
                frozen = pilot_here[0]

                key = (
                    str(
                        frozen[
                            "profile"
                        ]
                    ),
                    str(
                        frozen[
                            "trial_name"
                        ]
                    ),
                )

                source = all_trials.get(
                    key
                )

                if source is None:
                    raise ValueError(
                        "frozen pilot trial "
                        "missing from manifest"
                    )

                if (
                    str(
                        source[
                            "trajectory_hash"
                        ]
                    )
                    != str(
                        frozen[
                            "trajectory_hash"
                        ]
                    )
                ):
                    raise ValueError(
                        "frozen pilot trajectory "
                        "hash mismatch"
                    )

                chosen.append(
                    dict(source)
                )

            for trial in candidates:
                key = (
                    str(
                        trial["profile"]
                    ),
                    str(
                        trial[
                            "trial_name"
                        ]
                    ),
                )

                if any(
                    (
                        str(
                            x["profile"]
                        ),
                        str(
                            x[
                                "trial_name"
                            ]
                        ),
                    )
                    == key
                    for x in chosen
                ):
                    continue

                chosen.append(
                    dict(trial)
                )

                if len(
                    chosen
                ) == target:
                    break

            if len(
                chosen
            ) != target:
                raise ValueError(
                    f"{profile} {cell}: "
                    f"needed {target}, "
                    f"found {len(chosen)}"
                )

            for trial in chosen:
                key = (
                    str(
                        trial["profile"]
                    ),
                    str(
                        trial[
                            "trial_name"
                        ]
                    ),
                )

                if key in selected_ids:
                    raise ValueError(
                        "duplicate dose "
                        f"trajectory: {key}"
                    )

                selected_ids.add(key)

                row = {
                    "profile": profile,
                    "condition": (
                        condition
                    ),
                    "placement": (
                        placement
                    ),
                    "pressure_type": str(
                        trial.get(
                            "pressure_type",
                            "",
                        )
                    ),
                    "trial_name": str(
                        trial[
                            "trial_name"
                        ]
                    ),
                    "trajectory_path": str(
                        trial[
                            "trajectory_path"
                        ]
                    ),
                    "trajectory_hash": str(
                        trial[
                            "trajectory_hash"
                        ]
                    ),
                    "dose_hash": (
                        dose_hash(
                            profile=profile,
                            condition=(
                                condition
                            ),
                            placement=(
                                placement
                            ),
                            trial_name=str(
                                trial[
                                    "trial_name"
                                ]
                            ),
                        )
                    ),
                    "from_frozen_pilot": (
                        key in pilot_ids
                    ),
                    "jobs": [
                        dict(job)
                        for job
                        in trial["jobs"]
                    ],
                }

                selected.append(row)

    selected.sort(
        key=lambda trial: (
            CELLS.index((
                trial["condition"],
                trial["placement"],
            )),
            PROFILES.index(
                trial["profile"]
            ),
            trial["dose_hash"],
            trial["trial_name"],
        )
    )

    if len(
        selected
    ) != 100:
        raise ValueError(
            "expected 100 dose "
            f"trajectories, found "
            f"{len(selected)}"
        )

    cell_counts = Counter(
        (
            trial["condition"],
            trial["placement"],
        )
        for trial in selected
    )

    if (
        set(cell_counts)
        != set(CELLS)
        or any(
            count != 10
            for count
            in cell_counts.values()
        )
    ):
        raise ValueError(
            "dose cell balance failed"
        )

    profile_counts = Counter(
        trial["profile"]
        for trial in selected
    )

    if dict(
        profile_counts
    ) != {
        "claude": 25,
        "fable": 25,
        "codex": 25,
        "llama": 25,
    }:
        raise ValueError(
            "dose profile balance failed"
        )

    selected_pilot_ids = {
        (
            trial["profile"],
            trial["trial_name"],
        )
        for trial in selected
        if trial[
            "from_frozen_pilot"
        ]
    }

    if (
        selected_pilot_ids
        != pilot_ids
    ):
        raise ValueError(
            "frozen pilot is not "
            "an exact dose subset"
        )

    jobs = [
        dict(job)
        for trial in selected
        for job in trial["jobs"]
    ]

    if len(jobs) != 200:
        raise ValueError(
            "expected 200 core "
            "judge jobs"
        )

    cache_keys = [
        str(
            job["cache_key"]
        )
        for job in jobs
    ]

    if len(
        set(cache_keys)
    ) != 200:
        raise ValueError(
            "dose cache keys "
            "are not unique"
        )

    return {
        "dose_spec_version": (
            DOSE_SPEC_VERSION
        ),
        "selection_status": (
            "selected_before_dose_outcomes"
        ),
        "trajectory_count": 100,
        "core_judge_jobs": 200,
        "frozen_pilot_count": 10,
        "new_trajectory_count": 90,
        "profile_counts": dict(
            profile_counts
        ),
        "cell_counts": {
            (
                f"{condition}"
                "×"
                f"{placement}"
            ): count
            for (
                condition,
                placement,
            ), count
            in sorted(
                cell_counts.items()
            )
        },
        "allocation": {
            (
                f"{condition}"
                "×"
                f"{placement}"
            ): (
                CELL_PROFILE_COUNTS[
                    (
                        condition,
                        placement,
                    )
                ]
            )
            for (
                condition,
                placement,
            )
            in CELLS
        },
        "trials": selected,
    }


def print_summary(
    dose: dict[str, Any],
) -> None:
    print(
        "SEMANTIC STAGE C "
        "DOSE SELECTION"
    )
    print("=" * 72)

    print(
        "trajectories:",
        dose[
            "trajectory_count"
        ],
    )

    print(
        "core judge jobs:",
        dose[
            "core_judge_jobs"
        ],
    )

    print(
        "frozen pilot included:",
        dose[
            "frozen_pilot_count"
        ],
    )

    print(
        "new trajectories:",
        dose[
            "new_trajectory_count"
        ],
    )

    print(
        "profile counts:",
        dose[
            "profile_counts"
        ],
    )

    print()
    print("CELL COUNTS")

    for cell, count in (
        dose[
            "cell_counts"
        ].items()
    ):
        print(
            f"  {cell}: {count}"
        )

    print()
    print("network calls: 0")
    print(
        "STAGE C DOSE "
        "SELECTION: PASS"
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--pilot-spec",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    manifest = load_json(
        args.manifest
    )

    pilot_spec = load_json(
        args.pilot_spec
    )

    dose = build_dose(
        manifest=manifest,
        pilot_spec=pilot_spec,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            dose,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print_summary(dose)

    print(
        "plan:",
        args.output,
    )


if __name__ == "__main__":
    main()
