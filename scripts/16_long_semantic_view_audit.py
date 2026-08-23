#!/usr/bin/env python3
"""Inspect the nine compact semantic views that still exceed a target length."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from semantic_view import render_semantic_view


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(row: dict[str, Any]) -> Path | None:
    rel = Path(str(row.get("trajectory_file") or ""))
    if row.get("result_path"):
        p = Path(str(row["result_path"])).parent / rel
        if p.is_file():
            return p
    if row.get("run_root") and row.get("trial_name"):
        p = Path(str(row["run_root"])) / str(row["trial_name"]) / rel
        if p.is_file():
            return p
    return None


HEADER_RE = re.compile(
    r"^\[(SYSTEM|USER|AGENT|RECORDED REASONING|TOOL CALL:[^\]]+|TOOL RESULT)\]\n",
    re.M,
)


def split_blocks(view: str) -> list[tuple[str, str]]:
    matches = list(HEADER_RE.finditer(view))
    out = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(view)
        header = m.group(1)
        body = view[start:end].strip()
        if header.startswith("TOOL CALL:"):
            kind = "tool_call"
        elif header == "TOOL RESULT":
            kind = "tool_result"
        elif header == "RECORDED REASONING":
            kind = "reasoning"
        else:
            kind = header.lower()
        out.append((kind, body))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=Path, required=True)
    ap.add_argument("--threshold", type=int, default=60000)
    args = ap.parse_args()

    found = 0
    for profile_dir in sorted(args.input_root.iterdir()):
        trials_path = profile_dir / "trials.json"
        if not trials_path.is_file():
            continue

        for row in load_json(trials_path):
            if not row.get("substantive_usable"):
                continue

            path = resolve(row)
            if path is None:
                continue

            view = render_semantic_view(
                load_json(path),
                evaluation_cue_text=str(row.get("eval_cue_text", "") or ""),
                pressure_type=str(row.get("pressure_type", "") or ""),
            )
            if len(view) <= args.threshold:
                continue

            found += 1
            blocks = split_blocks(view)
            totals = Counter()
            counts = Counter()
            largest = []

            for kind, body in blocks:
                n = len(body)
                totals[kind] += n
                counts[kind] += 1
                largest.append((n, kind, body[:180].replace("\n", " ")))

            print("=" * 110)
            print(
                profile_dir.name,
                row.get("trial_name"),
                row.get("condition"),
                row.get("channel"),
                f"chars={len(view)}",
            )
            print("component totals:")
            for kind in sorted(totals):
                print(f"  {kind:16s} count={counts[kind]:3d} chars={totals[kind]:7d}")

            print("largest blocks:")
            for n, kind, preview in sorted(largest, reverse=True)[:8]:
                print(f"  {n:7d}  {kind:16s}  {preview}")

    print(f"\nover_limit_count={found}")


if __name__ == "__main__":
    main()
