#!/usr/bin/env python3
"""Audit the final semantic_excerpt safety guard for semantic judge v2.4.

Checks whether treatment anchors preserved by semantic-view schema 2.0 remain
present after the final max-character guard used by 07_analyze.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from semantic_view import render_semantic_view, semantic_excerpt, treatment_anchors


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=Path, required=True)
    ap.add_argument("--max-chars", type=int, default=60000)
    args = ap.parse_args()

    failures = []
    over_limit = []
    profile_rows = []

    for profile_dir in sorted(args.input_root.iterdir()):
        trials_path = profile_dir / "trials.json"
        if not trials_path.is_file():
            continue

        n = 0
        unresolved = 0
        rendered_over = 0
        raw_anchor_visible = 0
        compact_anchor_preserved = 0
        final_anchor_preserved = 0

        for row in load_json(trials_path):
            if not row.get("substantive_usable"):
                continue
            n += 1

            path = resolve(row)
            if path is None:
                unresolved += 1
                continue

            trajectory = load_json(path)
            compact = render_semantic_view(
                trajectory,
                evaluation_cue_text=str(row.get("eval_cue_text", "") or ""),
                pressure_type=str(row.get("pressure_type", "") or ""),
            )
            final = semantic_excerpt(compact, args.max_chars)

            if len(compact) > args.max_chars:
                rendered_over += 1
                over_limit.append({
                    "profile": profile_dir.name,
                    "trial_name": row.get("trial_name", ""),
                    "condition": row.get("condition", ""),
                    "channel": row.get("channel", ""),
                    "compact_chars": len(compact),
                    "final_chars": len(final),
                })

            anchors = treatment_anchors(
                str(row.get("eval_cue_text", "") or ""),
                str(row.get("pressure_type", "") or ""),
            )

            raw_text = json.dumps(trajectory, ensure_ascii=False).lower()
            present = [a for a in anchors if a.lower() in raw_text]
            if not present:
                continue

            raw_anchor_visible += 1
            compact_low = compact.lower()
            final_low = final.lower()

            if any(a.lower() in compact_low for a in present):
                compact_anchor_preserved += 1

            if any(a.lower() in final_low for a in present):
                final_anchor_preserved += 1
            else:
                failures.append({
                    "profile": profile_dir.name,
                    "trial_name": row.get("trial_name", ""),
                    "condition": row.get("condition", ""),
                    "channel": row.get("channel", ""),
                    "compact_chars": len(compact),
                    "final_chars": len(final),
                    "present_anchor_count": len(present),
                })

        profile_rows.append({
            "profile": profile_dir.name,
            "n": n,
            "unresolved": unresolved,
            "rendered_over_limit": rendered_over,
            "raw_anchor_visible": raw_anchor_visible,
            "compact_anchor_preserved": compact_anchor_preserved,
            "final_anchor_preserved": final_anchor_preserved,
            "final_anchor_preservation_pct": (
                round(100 * final_anchor_preserved / raw_anchor_visible, 2)
                if raw_anchor_visible else 100.0
            ),
        })

    print("PROFILE SUMMARY")
    cols = list(profile_rows[0])
    print("\t".join(cols))
    for row in profile_rows:
        print("\t".join(str(row[c]) for c in cols))

    print("\nOVER-LIMIT TRAJECTORIES")
    if over_limit:
        cols2 = list(over_limit[0])
        print("\t".join(cols2))
        for row in over_limit:
            print("\t".join(str(row[c]) for c in cols2))
    else:
        print("none")

    print("\nANCHOR FAILURES AFTER FINAL SAFETY GUARD")
    if failures:
        cols3 = list(failures[0])
        print("\t".join(cols3))
        for row in failures:
            print("\t".join(str(row[c]) for c in cols3))
    else:
        print("none")

    if failures:
        raise SystemExit(
            f"FAIL: {len(failures)} treatment-visible trajectories lose all "
            f"treatment anchors after final {args.max_chars}-character guard"
        )

    print(
        f"\nPASS: final {args.max_chars}-character guard preserved treatment "
        "anchors in every trajectory where an anchor was visible."
    )


if __name__ == "__main__":
    main()
