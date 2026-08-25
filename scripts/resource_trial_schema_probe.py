#!/usr/bin/env python3
"""Inspect per-trial Harbor result structure without treatment-effect analysis."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ERROR_RE = re.compile(
    r"\b[A-Za-z][A-Za-z0-9_]*(?:Error|Exception)\b"
)


def load_json(
    path: Path,
) -> Any:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def walk(
    value: Any,
    prefix: str = "",
):
    if isinstance(value, dict):
        for key, item in value.items():
            path = (
                f"{prefix}.{key}"
                if prefix
                else key
            )

            yield (
                path,
                item,
            )

            yield from walk(
                item,
                path,
            )

    elif isinstance(value, list):
        for index, item in enumerate(
            value[:3]
        ):
            path = (
                f"{prefix}[{index}]"
            )

            yield (
                path,
                item,
            )

            yield from walk(
                item,
                path,
            )


def value_type(
    value: Any,
) -> str:
    return type(
        value
    ).__name__


def trial_result_path(
    *,
    result_root: Path,
    trial_name: str,
) -> Path:
    trial_root = (
        result_root
        / result_root.name
    )

    exact = (
        trial_root
        / trial_name
        / "result.json"
    )

    if exact.is_file():
        return exact

    matches = sorted(
        path
        for path
        in trial_root.glob(
            f"{trial_name}__*"
        )
        if (
            path.is_dir()
            and (
                path
                / "result.json"
            ).is_file()
        )
    )

    if len(matches) != 1:
        raise ValueError(
            f"{trial_name}: expected "
            "exactly one Harbor result "
            "directory after suffix "
            f"resolution, found "
            f"{len(matches)}: "
            f"{matches}"
        )

    return (
        matches[0]
        / "result.json"
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    manifest = load_json(
        args.manifest
    )

    shard_map = {
        (
            s["profile"],
            int(s["chunk"]),
        ): Path(
            s["result_root"]
        )
        for s in manifest["shards"]
    }

    key_signatures = Counter()
    path_types = defaultdict(
        Counter
    )
    interesting_paths = defaultdict(
        Counter
    )
    exception_like = Counter()

    missing = []
    invalid = []
    total = 0

    examples = {}

    interesting_tokens = (
        "exception",
        "error",
        "reward",
        "score",
        "agent",
        "verifier",
        "trajectory",
        "started",
        "finished",
    )

    for trial in (
        manifest[
            "planned_trials"
        ]
    ):
        profile = trial[
            "profile"
        ]
        chunk = int(
            trial["chunk"]
        )
        trial_name = trial[
            "trial_name"
        ]

        root = shard_map[
            (
                profile,
                chunk,
            )
        ]

        path = (
            trial_result_path(
                result_root=root,
                trial_name=trial_name,
            )
        )

        if not path.is_file():
            missing.append({
                "profile": profile,
                "chunk": chunk,
                "trial_name": (
                    trial_name
                ),
                "path": str(path),
            })
            continue

        try:
            value = load_json(
                path
            )
        except Exception as exc:
            invalid.append({
                "profile": profile,
                "trial_name": (
                    trial_name
                ),
                "error": (
                    type(exc).__name__
                ),
            })
            continue

        total += 1

        if isinstance(
            value,
            dict,
        ):
            signature = tuple(
                sorted(
                    value.keys()
                )
            )

            key_signatures[
                signature
            ] += 1

        for nested_path, item in (
            walk(value)
        ):
            path_types[
                nested_path
            ][
                value_type(item)
            ] += 1

            lower = (
                nested_path.lower()
            )

            if any(
                token in lower
                for token
                in interesting_tokens
            ):
                if isinstance(
                    item,
                    (
                        str,
                        int,
                        float,
                        bool,
                        type(None),
                    ),
                ):
                    interesting_paths[
                        nested_path
                    ][
                        repr(item)[:160]
                    ] += 1
                else:
                    interesting_paths[
                        nested_path
                    ][
                        f"<{value_type(item)}>"
                    ] += 1

            if isinstance(
                item,
                str,
            ):
                for match in (
                    ERROR_RE.findall(
                        item
                    )
                ):
                    exception_like[
                        match
                    ] += 1

        if (
            profile
            not in examples
        ):
            examples[
                profile
            ] = {
                "path": str(path),
                "top_level_type": (
                    value_type(value)
                ),
                "top_level_keys": (
                    sorted(
                        value.keys()
                    )
                    if isinstance(
                        value,
                        dict,
                    )
                    else []
                ),
            }

    print(
        "RESOURCE PER-TRIAL SCHEMA PROBE"
    )
    print("=" * 72)

    print(
        "planned:",
        len(
            manifest[
                "planned_trials"
            ]
        ),
    )

    print(
        "result.json found:",
        total,
    )

    print(
        "missing result.json:",
        len(missing),
    )

    print(
        "invalid JSON:",
        len(invalid),
    )

    print()
    print(
        "TOP-LEVEL KEY SIGNATURES"
    )

    for signature, count in (
        key_signatures.most_common()
    ):
        print(
            count,
            "×",
            list(signature),
        )

    print()
    print(
        "EXCEPTION-LIKE STRINGS"
    )

    if exception_like:
        for value, count in (
            exception_like.most_common()
        ):
            print(
                count,
                "×",
                value,
            )
    else:
        print(
            "(none discovered)"
        )

    print()
    print(
        "INTERESTING SCALAR PATHS"
    )

    for path in sorted(
        interesting_paths
    ):
        counts = (
            interesting_paths[
                path
            ]
        )

        print()
        print(
            path,
            "n=",
            sum(
                counts.values()
            ),
        )

        for value, count in (
            counts.most_common(12)
        ):
            print(
                " ",
                count,
                "×",
                value,
            )

    print()
    print(
        "FIRST RESULT PER PROFILE"
    )

    for profile in sorted(
        examples
    ):
        print(
            profile,
            examples[profile],
        )

    if missing:
        print()
        print(
            "MISSING EXAMPLES"
        )

        for item in missing[:20]:
            print(item)

    if invalid:
        print()
        print(
            "INVALID EXAMPLES"
        )

        for item in invalid[:20]:
            print(item)

    expected = len(
        manifest[
            "planned_trials"
        ]
    )

    if total != expected:
        raise ValueError(
            "per-trial result coverage "
            f"mismatch: expected "
            f"{expected}, found {total}"
        )

    if missing:
        raise ValueError(
            f"missing per-trial "
            f"result.json files: "
            f"{len(missing)}"
        )

    if invalid:
        raise ValueError(
            f"invalid per-trial "
            f"result.json files: "
            f"{len(invalid)}"
        )

    print()
    print(
        "result coverage:",
        f"{total}/{expected}",
    )
    print(
        "network calls: 0"
    )
    print(
        "RESOURCE SCHEMA PROBE: PASS"
    )


if __name__ == "__main__":
    main()
