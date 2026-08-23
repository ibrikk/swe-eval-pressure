#!/usr/bin/env python3
"""Freeze the primary 200-trajectory semantic validation sample.

Design:
  4 profiles x 10 experimental cells x 5 trajectories = 200.

Selection is deterministic and judge-independent:
- uses only profile, substantive usability, condition, channel, and trial identity;
- never reads semantic-judge labels for sampling;
- excludes the 16 semantic-codebook calibration trajectories;
- ranks candidates by SHA-256(seed | profile | cell | trial_name).

The exact semantic-view 2.1 evidence shown to future LLM judges and humans is
also frozen here, together with SHA-256 hashes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from semantic_view import SEMANTIC_VIEW_SCHEMA_VERSION, render_semantic_view


PROFILES = ("claude", "fable", "codex", "llama")
N_PER_CELL = 5

CELLS = (
    ("clean", "clean", "none"),
    ("eval_only_root", "eval_only", "root"),
    ("eval_only_scaffold", "eval_only", "scaffold"),
    ("eval_only_source", "eval_only", "source"),
    ("eval_financial_root", "eval_financial", "root"),
    ("eval_financial_scaffold", "eval_financial", "scaffold"),
    ("eval_financial_source", "eval_financial", "source"),
    ("eval_self_preservation_root", "eval_self_preservation", "root"),
    ("eval_self_preservation_scaffold", "eval_self_preservation", "scaffold"),
    ("eval_self_preservation_source", "eval_self_preservation", "source"),
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_trajectory(row: dict[str, Any]) -> Path | None:
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


def load_calibration_exclusions(path: Path) -> set[str]:
    if not path.is_file():
        raise SystemExit(f"Missing calibration results: {path}")
    excluded: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        trial_name = str(row.get("trial_name") or "")
        if not trial_name:
            raise SystemExit("Calibration JSONL contains a row without trial_name")
        excluded.add(trial_name)
    return excluded


def cell_for(row: dict[str, Any]) -> str | None:
    condition = str(row.get("condition") or "")
    channel = str(row.get("channel") or "")
    for cell, wanted_condition, wanted_channel in CELLS:
        if condition == wanted_condition and channel == wanted_channel:
            return cell
    return None


def deterministic_rank(seed: str, profile: str, cell: str, trial_name: str) -> str:
    return sha256_text(f"{seed}|{profile}|{cell}|{trial_name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-root",
        type=Path,
        default=Path("analysis/semantic/v23-frozen"),
    )
    ap.add_argument(
        "--calibration-jsonl",
        type=Path,
        default=Path("analysis/semantic/v26-smoke-c/smoke_results.jsonl"),
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/validation/semantic-v26-primary-200"),
    )
    ap.add_argument("--seed", default="20260823")
    args = ap.parse_args()

    excluded = load_calibration_exclusions(args.calibration_jsonl)
    if len(excluded) != 16:
        raise SystemExit(
            f"Expected exactly 16 calibration exclusions, found {len(excluded)}"
        )

    selected: list[dict[str, Any]] = []
    candidate_counts: dict[str, int] = {}

    for profile in PROFILES:
        trials_path = args.input_root / profile / "trials.json"
        if not trials_path.is_file():
            raise SystemExit(f"Missing {trials_path}")

        rows = load_json(trials_path)
        if not isinstance(rows, list):
            raise SystemExit(f"{trials_path} is not a JSON list")

        for cell, condition, channel in CELLS:
            candidates = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                if not row.get("substantive_usable"):
                    continue

                trial_name = str(row.get("trial_name") or "")
                if not trial_name or trial_name in excluded:
                    continue

                if str(row.get("condition") or "") != condition:
                    continue
                if str(row.get("channel") or "") != channel:
                    continue

                candidates.append(row)

            candidate_counts[f"{profile}:{cell}"] = len(candidates)
            if len(candidates) < N_PER_CELL:
                raise SystemExit(
                    f"Not enough eligible candidates for {profile}/{cell}: "
                    f"{len(candidates)} < {N_PER_CELL}"
                )

            candidates.sort(
                key=lambda r: deterministic_rank(
                    args.seed,
                    profile,
                    cell,
                    str(r.get("trial_name") or ""),
                )
            )

            for row in candidates[:N_PER_CELL]:
                selected.append(
                    {
                        "profile": profile,
                        "cell": cell,
                        "condition": condition,
                        "channel": channel,
                        "row": row,
                    }
                )

    expected = len(PROFILES) * len(CELLS) * N_PER_CELL
    if len(selected) != expected:
        raise SystemExit(f"Expected {expected} selections, got {len(selected)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    views_dir = args.output_dir / "views"
    views_dir.mkdir(parents=True, exist_ok=True)

    sample_rows = []
    judge_inputs = []
    identity_lines = []
    profile_counts = Counter()
    cell_counts = Counter()
    stratum_counts = Counter()

    for idx, item in enumerate(selected, start=1):
        sample_id = f"SV{idx:03d}"
        profile = item["profile"]
        cell = item["cell"]
        row = item["row"]
        trial_name = str(row.get("trial_name") or "")

        trajectory_path = resolve_trajectory(row)
        if trajectory_path is None:
            raise SystemExit(f"Could not resolve trajectory for {trial_name}")

        trajectory = load_json(trajectory_path)
        cue_reference = str(row.get("eval_cue_text", "") or "")
        pressure_type = str(row.get("pressure_type", "") or "")

        view = render_semantic_view(
            trajectory,
            evaluation_cue_text=cue_reference,
            pressure_type=pressure_type,
        )
        view_hash = sha256_text(view)

        view_path = views_dir / f"{sample_id}.txt"
        view_path.write_text(view, encoding="utf-8")

        sample_row = {
            "sample_id": sample_id,
            "profile": profile,
            "cell": cell,
            "condition": item["condition"],
            "channel": item["channel"],
            "trial_name": trial_name,
            "pressure_type": pressure_type,
            "evaluation_cue_reference_present": bool(cue_reference),
            "semantic_view_version": SEMANTIC_VIEW_SCHEMA_VERSION,
            "semantic_view_chars": len(view),
            "semantic_view_sha256": view_hash,
            "trajectory_file": str(trajectory_path),
        }
        sample_rows.append(sample_row)

        judge_inputs.append(
            {
                "sample_id": sample_id,
                "evaluation_cue_text_reference": cue_reference,
                "semantic_trajectory": view,
                "semantic_view_version": SEMANTIC_VIEW_SCHEMA_VERSION,
                "semantic_view_sha256": view_hash,
            }
        )

        identity_lines.append(
            "\t".join(
                [
                    sample_id,
                    profile,
                    cell,
                    trial_name,
                    view_hash,
                ]
            )
        )

        profile_counts[profile] += 1
        cell_counts[cell] += 1
        stratum_counts[f"{profile}:{cell}"] += 1

    # Private sample map with experimental metadata.
    sample_csv = args.output_dir / "sample_map.csv"
    with sample_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(sample_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sample_rows)

    # Exact blinded inputs for future LLM judges and human-packet generation.
    inputs_jsonl = args.output_dir / "blinded_inputs.jsonl"
    with inputs_jsonl.open("w", encoding="utf-8") as f:
        for row in judge_inputs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    exclusions_path = args.output_dir / "calibration_exclusions.txt"
    exclusions_path.write_text(
        "\n".join(sorted(excluded)) + "\n",
        encoding="utf-8",
    )

    identities_text = "\n".join(identity_lines) + "\n"
    identities_path = args.output_dir / "sample_identities.tsv"
    identities_path.write_text(identities_text, encoding="utf-8")

    manifest = {
        "schema_version": "1.0",
        "purpose": "primary human/independent-LLM semantic validation sample",
        "selection_is_semantic_label_independent": True,
        "selection_fields_used": [
            "profile",
            "substantive_usable",
            "condition",
            "channel",
            "trial_name",
        ],
        "input_root": str(args.input_root),
        "calibration_exclusion_source": str(args.calibration_jsonl),
        "n_calibration_excluded": len(excluded),
        "seed": args.seed,
        "selection_method": (
            "Within each profile x experimental-cell stratum, rank eligible "
            "non-calibration trials by SHA256(seed|profile|cell|trial_name) "
            "and take the first 5."
        ),
        "profiles": list(PROFILES),
        "cells": [
            {
                "cell": cell,
                "condition": condition,
                "channel": channel,
            }
            for cell, condition, channel in CELLS
        ],
        "n_per_profile_cell": N_PER_CELL,
        "n_total": len(sample_rows),
        "semantic_view_version": SEMANTIC_VIEW_SCHEMA_VERSION,
        "profile_counts": dict(sorted(profile_counts.items())),
        "cell_counts": dict(sorted(cell_counts.items())),
        "stratum_counts": dict(sorted(stratum_counts.items())),
        "candidate_counts": dict(sorted(candidate_counts.items())),
        "sample_identities_sha256": sha256_text(identities_text),
        "sample_map_sha256": sha256_file(sample_csv),
        "blinded_inputs_sha256": sha256_file(inputs_jsonl),
        "calibration_exclusions_sha256": sha256_file(exclusions_path),
    }

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("PRIMARY VALIDATION SAMPLE FROZEN")
    print(f"total: {len(sample_rows)}")
    print(f"calibration_exclusions: {len(excluded)}")
    print(f"semantic_view_version: {SEMANTIC_VIEW_SCHEMA_VERSION}")
    print("profile_counts:")
    for k, v in sorted(profile_counts.items()):
        print(f"  {k}: {v}")
    print("cell_counts:")
    for k, v in sorted(cell_counts.items()):
        print(f"  {k}: {v}")
    print("stratum_count_values:", sorted(set(stratum_counts.values())))
    print(f"sample_identities_sha256: {manifest['sample_identities_sha256']}")
    print(f"output_dir: {args.output_dir}")


if __name__ == "__main__":
    main()
