#!/usr/bin/env python3
"""Audit preservation of observable agent evidence in semantic-view 2.1.

For every substantively usable ATIF trajectory, verify that each non-empty
observable agent/assistant/model message from the raw trajectory occurs
verbatim in the rendered semantic view.

This is deliberately independent of any semantic-judge label.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from semantic_view import render_semantic_view


AGENT_SOURCES = {"agent", "assistant", "model"}


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


def stringify(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (int, float, bool)):
        return str(x)
    if isinstance(x, list):
        return "\n".join(part for part in (stringify(v) for v in x) if part)
    if isinstance(x, dict):
        for key in ("text", "content", "message", "output", "result"):
            if key in x:
                value = stringify(x.get(key))
                if value:
                    return value
        return json.dumps(x, ensure_ascii=False, sort_keys=True)
    return str(x)


def raw_agent_messages(obj: Any) -> list[str]:
    if not (isinstance(obj, dict) and isinstance(obj.get("steps"), list)):
        return []
    out = []
    for step in obj["steps"]:
        if not isinstance(step, dict):
            continue
        source = str(step.get("source") or step.get("role") or "").strip().lower()
        if source not in AGENT_SOURCES:
            continue
        msg = stringify(
            step.get("message")
            or step.get("content")
            or step.get("text")
        ).strip()
        if msg:
            out.append(msg)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=Path, required=True)
    args = ap.parse_args()

    summaries = []
    failures = []

    for profile_dir in sorted(args.input_root.iterdir()):
        trials_path = profile_dir / "trials.json"
        if not trials_path.is_file():
            continue

        trajectories = 0
        unresolved = 0
        agent_messages = 0
        preserved_messages = 0

        for row in load_json(trials_path):
            if not row.get("substantive_usable"):
                continue
            trajectories += 1

            path = resolve(row)
            if path is None:
                unresolved += 1
                continue

            obj = load_json(path)
            view = render_semantic_view(
                obj,
                evaluation_cue_text=str(row.get("eval_cue_text", "") or ""),
                pressure_type=str(row.get("pressure_type", "") or ""),
            )

            messages = raw_agent_messages(obj)
            agent_messages += len(messages)

            for idx, msg in enumerate(messages):
                if msg in view:
                    preserved_messages += 1
                elif len(failures) < 50:
                    failures.append({
                        "profile": profile_dir.name,
                        "trial_name": row.get("trial_name", ""),
                        "message_index": idx,
                        "message_chars": len(msg),
                        "preview": msg[:180].replace("\n", " "),
                    })

        summaries.append({
            "profile": profile_dir.name,
            "trajectories": trajectories,
            "unresolved": unresolved,
            "raw_agent_messages": agent_messages,
            "verbatim_preserved": preserved_messages,
            "preservation_pct": (
                round(100 * preserved_messages / agent_messages, 4)
                if agent_messages else 100.0
            ),
        })

    print("PROFILE SUMMARY")
    cols = list(summaries[0])
    print("\t".join(cols))
    for row in summaries:
        print("\t".join(str(row[c]) for c in cols))

    print("\nMISSING AGENT MESSAGES")
    if failures:
        cols2 = list(failures[0])
        print("\t".join(cols2))
        for row in failures:
            print("\t".join(str(row[c]) for c in cols2))
        raise SystemExit(
            f"FAIL: {len(failures)}+ observable agent messages were not preserved verbatim"
        )

    print("none")
    print("\nPASS: every observable agent message was preserved verbatim.")


if __name__ == "__main__":
    main()
