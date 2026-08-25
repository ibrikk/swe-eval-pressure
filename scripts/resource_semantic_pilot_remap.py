#!/usr/bin/env python3
"""Remap the frozen resource-1.0 pilot identities onto resource-1.1 jobs.

No semantic selection is performed.
No network/API calls are made.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


OLD_SPEC_SHA = (
    "f2cc04068f1310a8367263d09391aa7491856ddb8e52825af3d2b71576493127"
)

OLD_PLAN_SHA = (
    "9d1d167636013fb24c07b66084c1a2c3a2858e949e2261a86a8eedcae6e7fb33"
)

NEW_MANIFEST_SHA = (
    "1ab4cf3074d3e4e57232adf0f965d479c8f23b3ecdc58c907eadfce442c5a2ca"
)

DATA = (
    Path.home()
    / "Documents"
    / "swe-eval-pressure"
)

OLD_SPEC = (
    Path("config")
    / "resource_semantic_pilot_v1.json"
)

OLD_PLAN = (
    DATA
    / "analysis"
    / "semantic-resource-v1"
    / "pilot"
    / "resource-pilot-plan.json"
)

NEW_MANIFEST = (
    DATA
    / "analysis"
    / "semantic-resource-v1"
    / "manifests"
    / "resource-stage-a-v1.1.json"
)

NEW_SPEC = (
    Path("config")
    / "resource_semantic_pilot_v1.1.json"
)

NEW_PLAN = (
    DATA
    / "analysis"
    / "semantic-resource-v1"
    / "pilot"
    / "resource-pilot-plan-v1.1.json"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def load(path: Path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


for path, expected, label in (
    (
        OLD_SPEC,
        OLD_SPEC_SHA,
        "old pilot spec",
    ),
    (
        OLD_PLAN,
        OLD_PLAN_SHA,
        "old pilot plan",
    ),
    (
        NEW_MANIFEST,
        NEW_MANIFEST_SHA,
        "new Stage A manifest",
    ),
):
    actual = sha256_file(path)

    if actual != expected:
        raise ValueError(
            f"{label} hash mismatch\n"
            f"expected={expected}\n"
            f"actual={actual}"
        )


old_spec = load(OLD_SPEC)
old_plan = load(OLD_PLAN)
manifest = load(NEW_MANIFEST)

if (
    manifest["semantic_schema_version"]
    != "resource-1.1"
    or manifest["rubric_version"]
    != "resource-1.1"
):
    raise ValueError(
        "new manifest is not resource-1.1"
    )

if int(
    manifest[
        "semantic_eligible_trajectories"
    ]
) != 618:
    raise ValueError(
        "eligible denominator changed"
    )

if int(
    manifest["core_judge_jobs"]
) != 1236:
    raise ValueError(
        "job denominator changed"
    )


jobs_by_trial = {}

for job in manifest["jobs"]:
    key = (
        str(job["profile"]),
        str(job["trial_name"]),
    )

    jobs_by_trial.setdefault(
        key,
        [],
    ).append(job)


old_plan_jobs = {
    str(job["cache_key"])
    for job in old_plan["jobs"]
}

new_trials = []
new_jobs = []


for old_trial in old_spec["trials"]:
    key = (
        str(old_trial["profile"]),
        str(old_trial["trial_name"]),
    )

    matches = jobs_by_trial.get(
        key,
        [],
    )

    if len(matches) != 2:
        raise ValueError(
            f"{key}: expected exactly "
            f"2 resource-1.1 jobs, "
            f"found {len(matches)}"
        )

    matches = sorted(
        matches,
        key=lambda job: (
            str(job["judge_family"]),
            str(job["judge_model"]),
        ),
    )

    first = matches[0]

    checks = {
        "condition": str(
            old_trial["condition"]
        ),
        "placement": str(
            old_trial["placement"]
        ),
        "pressure_type": str(
            old_trial["pressure_type"]
        ),
        "trajectory_hash": str(
            old_trial[
                "trajectory_hash"
            ]
        ),
    }

    for field, expected in (
        checks.items()
    ):
        actual = str(
            first.get(
                field,
                "",
            )
        )

        if actual != expected:
            raise ValueError(
                f"{key}: {field} changed\n"
                f"old={expected}\n"
                f"new={actual}"
            )

    remapped = {
        "profile": str(
            old_trial["profile"]
        ),
        "trial_name": str(
            old_trial["trial_name"]
        ),
        "condition": str(
            old_trial["condition"]
        ),
        "placement": str(
            old_trial["placement"]
        ),
        "pressure_type": str(
            old_trial["pressure_type"]
        ),
        "trajectory_path": str(
            first["trajectory_path"]
        ),
        "trajectory_hash": str(
            first["trajectory_hash"]
        ),
        "selection_hash": str(
            old_trial[
                "selection_hash"
            ]
        ),
        "selection_origin": (
            "frozen_resource_1.0_pilot"
        ),
        "judge_jobs": [
            {
                "judge_family": str(
                    job[
                        "judge_family"
                    ]
                ),
                "judge_model": str(
                    job[
                        "judge_model"
                    ]
                ),
                "cache_key": str(
                    job["cache_key"]
                ),
            }
            for job in matches
        ],
    }

    new_trials.append(
        remapped
    )

    new_jobs.extend(
        dict(job)
        for job in matches
    )


if len(new_trials) != 9:
    raise ValueError(
        f"expected 9 trajectories, "
        f"found {len(new_trials)}"
    )

if len(new_jobs) != 18:
    raise ValueError(
        f"expected 18 jobs, "
        f"found {len(new_jobs)}"
    )


new_cache_keys = {
    str(job["cache_key"])
    for job in new_jobs
}

if len(new_cache_keys) != 18:
    raise ValueError(
        "new cache keys are not unique"
    )

overlap = (
    old_plan_jobs
    & new_cache_keys
)

if overlap:
    raise ValueError(
        "resource-1.1 cache keys overlap "
        "resource-1.0 pilot keys"
    )


profile_counts = dict(
    Counter(
        trial["profile"]
        for trial in new_trials
    )
)

condition_counts = dict(
    Counter(
        trial["condition"]
        for trial in new_trials
    )
)

judge_counts = dict(
    Counter(
        job["judge_model"]
        for job in new_jobs
    )
)


new_spec = {
    "pilot_version": (
        "resource-pilot-1.1"
    ),
    "selection_status": (
        "frozen_identity_remap_before_"
        "resource_1_1_semantic_outcomes"
    ),
    "selection_method": (
        "Reuse exactly the nine trajectory "
        "identities frozen in the resource-1.0 "
        "pre-production pilot. No trajectory "
        "is newly selected or replaced."
    ),
    "source_resource_1_0_spec_path": str(
        OLD_SPEC.resolve()
    ),
    "source_resource_1_0_spec_sha256": (
        OLD_SPEC_SHA
    ),
    "source_resource_1_0_plan_sha256": (
        OLD_PLAN_SHA
    ),
    "source_manifest_path": str(
        NEW_MANIFEST.resolve()
    ),
    "source_manifest_sha256": (
        NEW_MANIFEST_SHA
    ),
    "semantic_schema_version": (
        "resource-1.1"
    ),
    "rubric_version": (
        "resource-1.1"
    ),
    "semantic_view_version": (
        manifest[
            "semantic_view_version"
        ]
    ),
    "trajectory_count": 9,
    "core_judge_jobs": 18,
    "profile_counts": (
        profile_counts
    ),
    "condition_counts": (
        condition_counts
    ),
    "judge_counts": (
        judge_counts
    ),
    "outcome_based_selection": False,
    "verifier_outcomes_used_for_selection": False,
    "resource_1_0_semantic_outputs_used_for_selection": False,
    "same_trajectory_identities_as_resource_1_0": True,
    "previous_judgments_reused": False,
    "cache_key_overlap_with_resource_1_0": 0,
    "network_calls": 0,
    "judge_calls": 0,
    "trials": new_trials,
}


new_plan = {
    "pilot_version": (
        "resource-pilot-1.1"
    ),
    "selection_origin": (
        "exact_identity_remap_from_"
        "resource_1.0_frozen_pilot"
    ),
    "source_manifest_sha256": (
        NEW_MANIFEST_SHA
    ),
    "trajectory_count": 9,
    "job_count": 18,
    "profile_counts": (
        profile_counts
    ),
    "condition_counts": (
        condition_counts
    ),
    "judge_counts": (
        judge_counts
    ),
    "cache_key_overlap_with_resource_1_0": 0,
    "network_calls": 0,
    "judge_calls": 0,
    "trials": new_trials,
    "jobs": new_jobs,
}


NEW_SPEC.write_text(
    json.dumps(
        new_spec,
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)

NEW_PLAN.parent.mkdir(
    parents=True,
    exist_ok=True,
)

NEW_PLAN.write_text(
    json.dumps(
        new_plan,
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)


print(
    "RESOURCE SEMANTIC PILOT 1.1 REMAP"
)
print("=" * 72)

print(
    "trajectories:",
    len(new_trials),
)
print(
    "judge jobs:",
    len(new_jobs),
)
print(
    "profile counts:",
    profile_counts,
)
print(
    "condition counts:",
    condition_counts,
)
print(
    "judge counts:",
    judge_counts,
)
print(
    "old/new cache-key overlap:",
    len(overlap),
)

print()
print(
    "SAME FROZEN TRAJECTORIES"
)

for i, trial in enumerate(
    new_trials,
    1,
):
    print(
        f"{i:02d}",
        trial["profile"],
        trial["condition"],
        trial["trial_name"],
    )

print()
print(
    "spec:",
    NEW_SPEC.resolve(),
)
print(
    "spec sha256:",
    sha256_file(
        NEW_SPEC
    ),
)

print(
    "plan:",
    NEW_PLAN.resolve(),
)
print(
    "plan sha256:",
    sha256_file(
        NEW_PLAN
    ),
)

print()
print("network calls: 0")
print("judge calls: 0")
print(
    "RESOURCE SEMANTIC PILOT 1.1 "
    "REMAP: PASS"
)
