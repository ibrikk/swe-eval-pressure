#!/usr/bin/env python3
"""Audit compact semantic-view 2.0 lengths and treatment-anchor preservation."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from semantic_view import PRESSURE_ANCHORS, render_semantic_view, treatment_anchors


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


def pct(vals: list[int], q: float) -> int:
    if not vals:
        return 0
    xs = sorted(vals)
    return xs[round((len(xs) - 1) * q)]


def visible_raw_text(obj: Any) -> str:
    pieces = []
    if not (isinstance(obj, dict) and isinstance(obj.get("steps"), list)):
        return ""
    for step in obj["steps"]:
        if not isinstance(step, dict):
            continue
        for key in ("message", "content", "text", "reasoning_content"):
            value = step.get(key)
            if value is not None:
                pieces.append(json.dumps(value, ensure_ascii=False))
        for key in ("tool_calls", "observation"):
            value = step.get(key)
            if value is not None:
                pieces.append(json.dumps(value, ensure_ascii=False))
    return "\n".join(pieces)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=Path, required=True)
    ap.add_argument("--safety-max", type=int, default=60000)
    args = ap.parse_args()

    rows_out = []

    for profile_dir in sorted(args.input_root.iterdir()):
        trials_path = profile_dir / "trials.json"
        if not trials_path.is_file():
            continue

        lengths = []
        unresolved = 0
        errors = 0
        gt_safety = 0
        raw_anchor_visible = 0
        compact_anchor_preserved = 0
        raw_eval_visible = 0
        compact_eval_preserved = 0
        raw_pressure_visible = 0
        compact_pressure_preserved = 0
        lost_anchor_examples = []
        lost_eval_examples = []
        lost_pressure_examples = []

        for row in load_json(trials_path):
            if not row.get("substantive_usable"):
                continue
            path = resolve(row)
            if path is None:
                unresolved += 1
                continue

            obj = load_json(path)
            try:
                compact = render_semantic_view(
                    obj,
                    evaluation_cue_text=str(row.get("eval_cue_text", "") or ""),
                    pressure_type=str(row.get("pressure_type", "") or ""),
                )
            except Exception:
                errors += 1
                continue

            lengths.append(len(compact))
            if len(compact) > args.safety_max:
                gt_safety += 1

            raw = visible_raw_text(obj).lower()
            compact_low = compact.lower()
            anchors = treatment_anchors(
                str(row.get("eval_cue_text", "") or ""),
                str(row.get("pressure_type", "") or ""),
            )

            present = [a for a in anchors if a.lower() in raw]
            if present:
                raw_anchor_visible += 1
                if any(a.lower() in compact_low for a in present):
                    compact_anchor_preserved += 1
                elif len(lost_anchor_examples) < 10:
                    lost_anchor_examples.append(str(row.get("trial_name")))

            eval_anchor = str(row.get("eval_cue_text", "") or "").strip()
            if eval_anchor and eval_anchor.lower() in raw:
                raw_eval_visible += 1
                if eval_anchor.lower() in compact_low:
                    compact_eval_preserved += 1
                elif len(lost_eval_examples) < 10:
                    lost_eval_examples.append(str(row.get("trial_name")))

            pressure_candidates = PRESSURE_ANCHORS.get(str(row.get("pressure_type", "") or ""), [])
            raw_pressure_matches = [a for a in pressure_candidates if a.lower() in raw]
            if raw_pressure_matches:
                raw_pressure_visible += 1
                if any(a.lower() in compact_low for a in raw_pressure_matches):
                    compact_pressure_preserved += 1
                elif len(lost_pressure_examples) < 10:
                    lost_pressure_examples.append(str(row.get("trial_name")))

        rec = {
            "profile": profile_dir.name,
            "n": len(lengths),
            "unresolved": unresolved,
            "errors": errors,
            "min": min(lengths) if lengths else 0,
            "median": int(statistics.median(lengths)) if lengths else 0,
            "p90": pct(lengths, 0.90),
            "p95": pct(lengths, 0.95),
            "p99": pct(lengths, 0.99),
            "max": max(lengths) if lengths else 0,
            f"gt_{args.safety_max}": gt_safety,
            "raw_anchor_visible": raw_anchor_visible,
            "compact_anchor_preserved": compact_anchor_preserved,
            "anchor_preservation_pct": (
                round(100 * compact_anchor_preserved / raw_anchor_visible, 2)
                if raw_anchor_visible else 100.0
            ),
            "raw_eval_visible": raw_eval_visible,
            "compact_eval_preserved": compact_eval_preserved,
            "eval_preservation_pct": (
                round(100 * compact_eval_preserved / raw_eval_visible, 2)
                if raw_eval_visible else 100.0
            ),
            "raw_pressure_visible": raw_pressure_visible,
            "compact_pressure_preserved": compact_pressure_preserved,
            "pressure_preservation_pct": (
                round(100 * compact_pressure_preserved / raw_pressure_visible, 2)
                if raw_pressure_visible else 100.0
            ),
            "lost_anchor_examples": ",".join(lost_anchor_examples),
            "lost_eval_examples": ",".join(lost_eval_examples),
            "lost_pressure_examples": ",".join(lost_pressure_examples),
        }
        rows_out.append(rec)

    if not rows_out:
        raise SystemExit("No profile data found")

    cols = list(rows_out[0])
    print("\t".join(cols))
    for row in rows_out:
        print("\t".join(str(row[c]) for c in cols))


if __name__ == "__main__":
    main()
