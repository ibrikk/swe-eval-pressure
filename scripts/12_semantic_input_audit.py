#!/usr/bin/env python3
"""Audit semantic-view lengths and trajectory resolvability before v2.4 re-judging."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from semantic_view import render_semantic_view


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(row: dict) -> Path | None:
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


def percentile(vals: list[int], q: float) -> int:
    if not vals:
        return 0
    xs = sorted(vals)
    idx = round((len(xs) - 1) * q)
    return xs[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=Path, required=True)
    ap.add_argument("--thresholds", default="60000,120000,200000")
    args = ap.parse_args()

    thresholds = [int(x) for x in args.thresholds.split(",") if x.strip()]
    rows_out = []

    for profile_dir in sorted(args.input_root.iterdir()):
        trials_path = profile_dir / "trials.json"
        if not trials_path.is_file():
            continue
        profile = profile_dir.name
        lengths = []
        unresolved = 0
        render_errors = 0

        for row in load_json(trials_path):
            if not row.get("substantive_usable"):
                continue
            path = resolve(row)
            if path is None:
                unresolved += 1
                continue
            try:
                view = render_semantic_view(load_json(path))
            except Exception:
                render_errors += 1
                continue
            lengths.append(len(view))

        rec = {
            "profile": profile,
            "n_rendered": len(lengths),
            "unresolved": unresolved,
            "render_errors": render_errors,
            "min_chars": min(lengths) if lengths else 0,
            "median_chars": int(statistics.median(lengths)) if lengths else 0,
            "p90_chars": percentile(lengths, 0.90),
            "p95_chars": percentile(lengths, 0.95),
            "p99_chars": percentile(lengths, 0.99),
            "max_chars": max(lengths) if lengths else 0,
        }
        for t in thresholds:
            rec[f"gt_{t}"] = sum(v > t for v in lengths)
        rows_out.append(rec)

    if not rows_out:
        raise SystemExit("No canonical profile directories found")

    cols = list(rows_out[0])
    print("\t".join(cols))
    for r in rows_out:
        print("\t".join(str(r[c]) for c in cols))


if __name__ == "__main__":
    main()
