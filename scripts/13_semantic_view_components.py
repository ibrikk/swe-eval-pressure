#!/usr/bin/env python3
"""Decompose semantic-view size by evidence type.

Used to design a compact judge/human evidence view before semantic judge v2.4
is frozen. This script does not call any model and does not modify analysis.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from semantic_view import (
    format_observation,
    format_tool_calls,
    stringify_content,
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(row: dict[str, Any]) -> Path | None:
    rel = Path(str(row.get("trajectory_file") or ""))
    result_path = row.get("result_path")
    if result_path:
        p = Path(str(result_path)).parent / rel
        if p.is_file():
            return p
    run_root = row.get("run_root")
    trial_name = row.get("trial_name")
    if run_root and trial_name:
        p = Path(str(run_root)) / str(trial_name) / rel
        if p.is_file():
            return p
    return None


def pct(vals: list[int], q: float) -> int:
    if not vals:
        return 0
    xs = sorted(vals)
    return xs[round((len(xs) - 1) * q)]


def summarize(vals: list[int]) -> str:
    if not vals:
        return "0/0/0"
    return f"{int(statistics.median(vals))}/{pct(vals,0.90)}/{pct(vals,0.95)}"


def block_lengths(obj: Any) -> dict[str, int]:
    out = {
        "system": 0,
        "user": 0,
        "agent_message": 0,
        "reasoning": 0,
        "tool_calls": 0,
        "tool_results": 0,
        "other_visible": 0,
    }

    if not (isinstance(obj, dict) and isinstance(obj.get("steps"), list)):
        return out

    for step in obj["steps"]:
        if not isinstance(step, dict):
            continue

        source = str(
            step.get("source") or step.get("role") or "unknown"
        ).strip().lower()

        message = stringify_content(
            step.get("message")
            or step.get("content")
            or step.get("text")
        ).strip()

        if message:
            if source == "system":
                out["system"] += len(message)
            elif source == "user":
                out["user"] += len(message)
            elif source in {"agent", "assistant", "model"}:
                out["agent_message"] += len(message)
            else:
                out["other_visible"] += len(message)

        reasoning = stringify_content(step.get("reasoning_content")).strip()
        if reasoning:
            out["reasoning"] += len(reasoning)

        calls = format_tool_calls(step.get("tool_calls"))
        out["tool_calls"] += sum(len(x) for x in calls)

        results = format_observation(step.get("observation"))
        out["tool_results"] += sum(len(x) for x in results)

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=Path, required=True)
    args = ap.parse_args()

    profiles = []
    for profile_dir in sorted(args.input_root.iterdir()):
        trials_path = profile_dir / "trials.json"
        if not trials_path.is_file():
            continue

        buckets: dict[str, list[int]] = {
            "system": [],
            "user": [],
            "agent_message": [],
            "reasoning": [],
            "tool_calls": [],
            "tool_results": [],
            "other_visible": [],
            "agent_plus_reasoning": [],
            "non_tool_result": [],
            "total_components": [],
        }
        unresolved = 0
        non_atif = 0

        for row in load_json(trials_path):
            if not row.get("substantive_usable"):
                continue
            path = resolve(row)
            if path is None:
                unresolved += 1
                continue
            obj = load_json(path)
            if not (isinstance(obj, dict) and isinstance(obj.get("steps"), list)):
                non_atif += 1
                continue

            lens = block_lengths(obj)
            for k in [
                "system", "user", "agent_message", "reasoning",
                "tool_calls", "tool_results", "other_visible",
            ]:
                buckets[k].append(lens[k])

            ar = lens["agent_message"] + lens["reasoning"]
            non_tool = (
                lens["system"] + lens["user"] + lens["agent_message"]
                + lens["reasoning"] + lens["tool_calls"]
                + lens["other_visible"]
            )
            total = non_tool + lens["tool_results"]
            buckets["agent_plus_reasoning"].append(ar)
            buckets["non_tool_result"].append(non_tool)
            buckets["total_components"].append(total)

        total_chars = sum(buckets["total_components"])
        tool_chars = sum(buckets["tool_results"])
        profiles.append({
            "profile": profile_dir.name,
            "n": len(buckets["total_components"]),
            "unresolved": unresolved,
            "non_atif": non_atif,
            "system_med/p90/p95": summarize(buckets["system"]),
            "user_med/p90/p95": summarize(buckets["user"]),
            "agent_msg_med/p90/p95": summarize(buckets["agent_message"]),
            "reasoning_med/p90/p95": summarize(buckets["reasoning"]),
            "tool_calls_med/p90/p95": summarize(buckets["tool_calls"]),
            "tool_results_med/p90/p95": summarize(buckets["tool_results"]),
            "agent+reason_med/p90/p95": summarize(buckets["agent_plus_reasoning"]),
            "non_tool_result_med/p90/p95": summarize(buckets["non_tool_result"]),
            "tool_result_share_pct": round(100 * tool_chars / total_chars, 1) if total_chars else 0,
        })

    if not profiles:
        raise SystemExit("No profile data found")

    cols = list(profiles[0])
    print("\t".join(cols))
    for row in profiles:
        print("\t".join(str(row[c]) for c in cols))


if __name__ == "__main__":
    main()
