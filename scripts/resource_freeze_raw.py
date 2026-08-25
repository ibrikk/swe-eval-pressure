#!/usr/bin/env python3
"""Freeze and validate the completed resource-deprivation raw run.

This script is outcome-blind. It verifies only:
- exact shard/profile coverage;
- exact planned trajectory counts;
- 70 matched task triples per profile;
- cross-profile task identity;
- raw generated/result tree hashes.

It does not inspect rewards, pass/fail, exceptions, or agent semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FREEZE_VERSION = "1.0"

PROFILES = (
    "claude",
    "fable",
    "codex",
    "llama",
)

CHUNKS = (
    1,
    2,
    3,
)

EXPECTED_CHUNK_TRAJECTORIES = {
    1: 90,
    2: 90,
    3: 30,
}

EXPECTED_PER_PROFILE = 210
EXPECTED_PER_CONDITION = 70
EXPECTED_BASE_TASKS = 70
EXPECTED_TOTAL = 840

BASE_RE = re.compile(
    r"^(ea-[0-9a-f]+)-"
)


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def tree_identity(
    root: Path,
) -> dict[str, Any]:
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
    )

    digest = hashlib.sha256()
    total_bytes = 0

    for path in files:
        relative = str(
            path.relative_to(root)
        )

        size = path.stat().st_size
        total_bytes += size

        file_hash = sha256_file(
            path
        )

        record = (
            relative
            + "\0"
            + str(size)
            + "\0"
            + file_hash
            + "\n"
        )

        digest.update(
            record.encode("utf-8")
        )

    return {
        "file_count": len(files),
        "total_bytes": total_bytes,
        "tree_sha256": (
            digest.hexdigest()
        ),
    }


def condition_from_name(
    name: str,
) -> str:
    if "-eval-resource-" in name:
        return "eval_resource"

    if "-eval-" in name:
        return "eval_only"

    if "-clean-" in name:
        return "clean"

    raise ValueError(
        f"unrecognized resource trial name: {name}"
    )


def base_task_from_name(
    name: str,
) -> str:
    match = BASE_RE.match(name)

    if match is None:
        raise ValueError(
            "cannot parse base task from "
            f"trial name: {name}"
        )

    return match.group(1)


def generated_trials(
    shard_root: Path,
) -> list[dict[str, str]]:
    if not shard_root.is_dir():
        raise FileNotFoundError(
            shard_root
        )

    trial_dirs = sorted(
        path
        for path in shard_root.iterdir()
        if path.is_dir()
    )

    rows = []

    for path in trial_dirs:
        rows.append({
            "trial_name": path.name,
            "base_task": (
                base_task_from_name(
                    path.name
                )
            ),
            "condition": (
                condition_from_name(
                    path.name
                )
            ),
            "path": str(
                path.resolve()
            ),
        })

    return rows


def find_result_shard(
    *,
    results_root: Path,
    profile: str,
    chunk: int,
) -> Path:
    pattern = (
        "swe-eval-pressure-resource-"
        f"{profile}-chunk-{chunk}-"
        "size-30-*"
    )

    matches = sorted(
        path
        for path
        in results_root.glob(pattern)
        if path.is_dir()
    )

    if len(matches) != 1:
        raise ValueError(
            f"{profile} chunk {chunk}: "
            "expected exactly one raw "
            f"result shard, found {len(matches)}: "
            f"{matches}"
        )

    return matches[0].resolve()


def central_result_json(
    result_root: Path,
) -> Path:
    expected = (
        result_root
        / result_root.name
        / "result.json"
    )

    if expected.is_file():
        return expected

    candidates = sorted(
        result_root.glob(
            "*/result.json"
        )
    )

    if len(candidates) == 1:
        return candidates[0]

    raise ValueError(
        f"{result_root}: could not identify "
        "one central result.json; "
        f"found {len(candidates)} candidates"
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    data_root = (
        args.data_root
        .expanduser()
        .resolve()
    )

    results_root = (
        data_root
        / "results"
        / "resource"
    )

    generated_root = (
        data_root
        / "generated"
        / "_shards"
        / "resource"
    )

    shards = []
    all_trials = []

    profile_base_sets = {}

    for profile in PROFILES:
        profile_trials = []

        for chunk in CHUNKS:
            generated = (
                generated_root
                / profile
                / (
                    f"chunk-{chunk}-"
                    "size-30"
                )
            )

            rows = generated_trials(
                generated
            )

            expected_n = (
                EXPECTED_CHUNK_TRAJECTORIES[
                    chunk
                ]
            )

            if len(rows) != expected_n:
                raise ValueError(
                    f"{profile} chunk {chunk}: "
                    f"expected {expected_n} "
                    "generated trajectories, "
                    f"found {len(rows)}"
                )

            result_root = (
                find_result_shard(
                    results_root=results_root,
                    profile=profile,
                    chunk=chunk,
                )
            )

            result_json = (
                central_result_json(
                    result_root
                )
            )

            generated_identity = (
                tree_identity(
                    generated
                )
            )

            result_identity = (
                tree_identity(
                    result_root
                )
            )

            shard = {
                "profile": profile,
                "chunk": chunk,
                "planned_trajectories": (
                    len(rows)
                ),
                "generated_root": str(
                    generated
                ),
                "generated_tree": (
                    generated_identity
                ),
                "result_root": str(
                    result_root
                ),
                "result_tree": (
                    result_identity
                ),
                "central_result_json": str(
                    result_json.resolve()
                ),
                "central_result_sha256": (
                    sha256_file(
                        result_json
                    )
                ),
            }

            shards.append(
                shard
            )

            for row in rows:
                value = dict(row)

                value["profile"] = (
                    profile
                )

                value["chunk"] = chunk

                profile_trials.append(
                    value
                )

                all_trials.append(
                    value
                )

        if (
            len(profile_trials)
            != EXPECTED_PER_PROFILE
        ):
            raise ValueError(
                f"{profile}: expected "
                f"{EXPECTED_PER_PROFILE} "
                "planned trajectories, found "
                f"{len(profile_trials)}"
            )

        counts = Counter(
            row["condition"]
            for row in profile_trials
        )

        expected_conditions = {
            "clean": (
                EXPECTED_PER_CONDITION
            ),
            "eval_only": (
                EXPECTED_PER_CONDITION
            ),
            "eval_resource": (
                EXPECTED_PER_CONDITION
            ),
        }

        if (
            dict(counts)
            != expected_conditions
        ):
            raise ValueError(
                f"{profile}: condition "
                f"counts mismatch: {counts}"
            )

        grouped: dict[
            str,
            set[str],
        ] = {}

        for row in profile_trials:
            grouped.setdefault(
                row["base_task"],
                set(),
            ).add(
                row["condition"]
            )

        if (
            len(grouped)
            != EXPECTED_BASE_TASKS
        ):
            raise ValueError(
                f"{profile}: expected "
                f"{EXPECTED_BASE_TASKS} "
                "base tasks, found "
                f"{len(grouped)}"
            )

        expected_set = {
            "clean",
            "eval_only",
            "eval_resource",
        }

        bad = {
            task: conditions
            for task, conditions
            in grouped.items()
            if conditions != expected_set
        }

        if bad:
            raise ValueError(
                f"{profile}: incomplete "
                f"matched triples: {bad}"
            )

        profile_base_sets[
            profile
        ] = set(
            grouped
        )

    if len(all_trials) != EXPECTED_TOTAL:
        raise ValueError(
            "expected exactly "
            f"{EXPECTED_TOTAL} planned "
            "resource trajectories, found "
            f"{len(all_trials)}"
        )

    reference = (
        profile_base_sets[
            PROFILES[0]
        ]
    )

    for profile in PROFILES[1:]:
        if (
            profile_base_sets[
                profile
            ]
            != reference
        ):
            raise ValueError(
                f"{profile}: base-task set "
                "differs across profiles"
            )

    manifest = {
        "freeze_version": (
            FREEZE_VERSION
        ),
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "scope": (
            "resource_deprivation_raw"
        ),
        "outcome_blind": True,
        "profiles": list(
            PROFILES
        ),
        "chunks": list(
            CHUNKS
        ),
        "planned_trajectories": (
            EXPECTED_TOTAL
        ),
        "planned_per_profile": (
            EXPECTED_PER_PROFILE
        ),
        "planned_per_condition_per_profile": (
            EXPECTED_PER_CONDITION
        ),
        "base_tasks": (
            EXPECTED_BASE_TASKS
        ),
        "conditions": [
            "clean",
            "eval_only",
            "eval_resource",
        ],
        "shard_count": len(
            shards
        ),
        "shards": shards,
        "planned_trials": (
            all_trials
        ),
    }

    output = (
        args.output
        .expanduser()
        .resolve()
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "RESOURCE RAW FREEZE"
    )
    print("=" * 72)

    print(
        "raw result shards:",
        len(shards),
    )

    print(
        "planned trajectories:",
        len(all_trials),
    )

    print(
        "profiles:",
        len(PROFILES),
    )

    print(
        "planned/profile:",
        EXPECTED_PER_PROFILE,
    )

    print(
        "base tasks/profile:",
        EXPECTED_BASE_TASKS,
    )

    print()
    print(
        "per profile / condition:"
    )

    for profile in PROFILES:
        subset = [
            row
            for row in all_trials
            if (
                row["profile"]
                == profile
            )
        ]

        counts = Counter(
            row["condition"]
            for row in subset
        )

        print(
            f"  {profile}:",
            dict(counts),
        )

    print()
    print(
        "matched task triples:",
        (
            EXPECTED_BASE_TASKS
            * len(PROFILES)
        ),
    )

    print(
        "duplicate planned identities:",
        (
            len(all_trials)
            - len({
                (
                    row["profile"],
                    row["trial_name"],
                )
                for row in all_trials
            })
        ),
    )

    print(
        "cross-profile base-task "
        "identity: PASS"
    )

    print(
        "network calls: 0"
    )

    print(
        "RESOURCE RAW FREEZE: PASS"
    )

    print(
        "manifest:",
        output,
    )


if __name__ == "__main__":
    main()
