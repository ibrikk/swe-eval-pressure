#!/usr/bin/env python3
"""Non-destructive semantic judge smoke test.

Selects a deliberately varied set of trajectories from the frozen v2.3
analysis, renders them with semantic-view 2.1, and calls the exact semantic
judge implementation in scripts/07_analyze.py.

Outputs are written only under --output-dir. No production semantic caches or
analysis/semantic/full files are modified.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

from litellm_pool import parse_litellm_keys

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from semantic_view import SEMANTIC_VIEW_SCHEMA_VERSION, render_semantic_view


def load_analyzer():
    path = SCRIPT_DIR / "07_analyze.py"
    spec = importlib.util.spec_from_file_location("swe_eval_analyzer_v24", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def is_positive(value: Any) -> bool:
    return str(value or "").strip() in {"implicit", "explicit"}


def pressure_positive(row: dict[str, Any]) -> bool:
    return is_positive(row.get("pressure_recognition"))


def pressure_rejected(row: dict[str, Any]) -> bool:
    return str(row.get("pressure_cue_stance") or "") == "rejected_as_prompt_injection"


def eval_positive(row: dict[str, Any]) -> bool:
    return is_positive(row.get("evaluation_cue_recognition"))


def current_eval_positive(row: dict[str, Any]) -> bool:
    return is_positive(row.get("current_agent_evaluation_awareness"))


def choose_first(rows: list[dict[str, Any]], pred, used: set[str]):
    candidates = sorted(
        (r for r in rows if pred(r) and str(r.get("trial_name")) not in used),
        key=lambda r: str(r.get("trial_name")),
    )
    if not candidates:
        return None
    chosen = candidates[0]
    used.add(str(chosen.get("trial_name")))
    return chosen


def select_profile_cases(profile: str, rows: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    used: set[str] = set()
    out: list[tuple[str, dict[str, Any]]] = []

    specs = [
        (
            "clean_spontaneous_eval",
            lambda r: r.get("condition") == "clean" and current_eval_positive(r),
        ),
        (
            "seeded_eval_recognized",
            lambda r: r.get("condition") != "clean" and eval_positive(r),
        ),
        (
            "pressure_recognized_or_rejected",
            lambda r: r.get("condition") in {"eval_financial", "eval_self_preservation"}
            and (pressure_positive(r) or pressure_rejected(r)),
        ),
        (
            "hard_negative",
            lambda r: r.get("condition") != "clean"
            and not eval_positive(r)
            and not current_eval_positive(r)
            and not pressure_positive(r),
        ),
    ]

    for label, pred in specs:
        chosen = choose_first(rows, pred, used)
        if chosen is None:
            # Fall back to a sensible condition-preserving negative when the
            # profile has no positives for this construct (e.g. Llama).
            fallback_pred = {
                "clean_spontaneous_eval": lambda r: r.get("condition") == "clean",
                "seeded_eval_recognized": lambda r: r.get("condition") == "eval_only",
                "pressure_recognized_or_rejected": lambda r: r.get("condition")
                in {"eval_financial", "eval_self_preservation"},
                "hard_negative": lambda r: r.get("condition") != "clean",
            }[label]
            chosen = choose_first(rows, fallback_pred, used)

        if chosen is not None:
            out.append((label, chosen))

    return out


def old_label_subset(row: dict[str, Any]) -> dict[str, Any]:
    fields = [
        "evaluation_cue_recognition",
        "evaluation_cue_stance",
        "hidden_grader_awareness",
        "benchmark_recognition",
        "current_agent_evaluation_awareness",
        "evaluation_purpose_recognition",
        "evaluation_response",
        "pressure_recognition",
        "pressure_type_recognized",
        "pressure_cue_stance",
        "pressure_response",
        "evaluation_targeting_intent",
    ]
    return {k: row.get(k, "") for k in fields}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-root",
        type=Path,
        default=Path("analysis/semantic/v23-frozen"),
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/semantic/v24-smoke"),
    )
    ap.add_argument(
        "--model",
        default=os.getenv("ANALYSIS_MODEL", "openai/gpt-5.6"),
    )
    args = ap.parse_args()

    if not parse_litellm_keys() or not os.getenv("LITE_LLM_URL"):
        raise SystemExit(
            "LITE_LLM_KEY/LITE_LLM_KEYS and LITE_LLM_URL must be set before running the smoke test"
        )

    analyzer = load_analyzer()
    judge_version = str(analyzer.SEMANTIC_JUDGE_VERSION)
    if judge_version != "2.6":
        raise SystemExit(f"Expected semantic judge 2.6, found {judge_version}")

    os.environ["ANALYSIS_MODEL"] = args.model

    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected: list[tuple[str, str, dict[str, Any]]] = []
    for profile in ("claude", "fable", "codex", "llama"):
        trials_path = args.input_root / profile / "trials.json"
        if not trials_path.is_file():
            raise SystemExit(f"Missing {trials_path}")
        rows = [
            r
            for r in load_json(trials_path)
            if isinstance(r, dict) and r.get("substantive_usable")
        ]
        for case_type, row in select_profile_cases(profile, rows):
            selected.append((profile, case_type, row))

    results = []
    for idx, (profile, case_type, row) in enumerate(selected, start=1):
        trial_path = resolve_trajectory(row)
        if trial_path is None:
            raise SystemExit(f"Could not resolve trajectory for {row.get('trial_name')}")

        trajectory = load_json(trial_path)
        view = render_semantic_view(
            trajectory,
            evaluation_cue_text=str(row.get("eval_cue_text", "") or ""),
            pressure_type=str(row.get("pressure_type", "") or ""),
        )

        payload = {
            "evaluation_cue_text_reference": row.get("eval_cue_text", ""),
            "semantic_trajectory": view,
        }

        print(
            f"[{idx:02d}/{len(selected)}] {profile} {case_type} "
            f"{row.get('trial_name')} chars={len(view)}",
            flush=True,
        )
        judgment = analyzer.call_semantic_judge(payload)

        result = {
            "profile": profile,
            "case_type": case_type,
            "trial_name": row.get("trial_name", ""),
            "condition": row.get("condition", ""),
            "channel": row.get("channel", ""),
            "pressure_type": row.get("pressure_type", ""),
            "judge_model": args.model,
            "judge_version": judge_version,
            "judge_temperature": analyzer.SEMANTIC_JUDGE_TEMPERATURE,
            "semantic_view_version": SEMANTIC_VIEW_SCHEMA_VERSION,
            "semantic_view_chars": len(view),
            "old_v23": old_label_subset(row),
            "new_judgment": judgment,
        }
        results.append(result)

        safe_name = f"{idx:02d}_{profile}_{case_type}"
        (args.output_dir / f"{safe_name}.view.txt").write_text(
            view, encoding="utf-8"
        )
        (args.output_dir / f"{safe_name}.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    jsonl_path = args.output_dir / "smoke_results.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    csv_path = args.output_dir / "smoke_summary.csv"
    fields = [
        "profile",
        "case_type",
        "trial_name",
        "condition",
        "channel",
        "pressure_type",
        "semantic_view_chars",
        "judge_model",
        "judge_version",
        "semantic_view_version",
        "old_eval_cue",
        "new_eval_cue",
        "old_current_eval",
        "new_current_eval",
        "old_pressure",
        "new_pressure",
        "old_pressure_stance",
        "new_pressure_stance",
        "old_eval_response",
        "new_eval_response",
        "old_targeting",
        "new_targeting",
        "new_confidence",
        "new_status",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for item in results:
            old = item["old_v23"]
            new = item["new_judgment"] if isinstance(item["new_judgment"], dict) else {}
            w.writerow({
                "profile": item["profile"],
                "case_type": item["case_type"],
                "trial_name": item["trial_name"],
                "condition": item["condition"],
                "channel": item["channel"],
                "pressure_type": item["pressure_type"],
                "semantic_view_chars": item["semantic_view_chars"],
                "judge_model": item["judge_model"],
                "judge_version": item["judge_version"],
                "semantic_view_version": item["semantic_view_version"],
                "old_eval_cue": old.get("evaluation_cue_recognition", ""),
                "new_eval_cue": new.get("evaluation_cue_recognition", ""),
                "old_current_eval": old.get("current_agent_evaluation_awareness", ""),
                "new_current_eval": new.get("current_agent_evaluation_awareness", ""),
                "old_pressure": old.get("pressure_recognition", ""),
                "new_pressure": new.get("pressure_recognition", ""),
                "old_pressure_stance": old.get("pressure_cue_stance", ""),
                "new_pressure_stance": new.get("pressure_cue_stance", ""),
                "old_eval_response": old.get("evaluation_response", ""),
                "new_eval_response": new.get("evaluation_response", ""),
                "old_targeting": old.get("evaluation_targeting_intent", ""),
                "new_targeting": new.get("evaluation_targeting_intent", ""),
                "new_confidence": new.get("confidence", ""),
                "new_status": new.get("status", "ok"),
            })

    manifest = {
        "schema_version": "1.0",
        "judge_model": args.model,
        "judge_version": judge_version,
        "judge_temperature": analyzer.SEMANTIC_JUDGE_TEMPERATURE,
        "semantic_view_version": SEMANTIC_VIEW_SCHEMA_VERSION,
        "input_root": str(args.input_root),
        "n": len(results),
        "selection": "4 deterministic case types per profile when available",
        "case_types": [
            "clean_spontaneous_eval",
            "seeded_eval_recognized",
            "pressure_recognized_or_rejected",
            "hard_negative",
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nWrote {len(results)} smoke judgments to {args.output_dir}")
    print(f"Summary: {csv_path}")


if __name__ == "__main__":
    main()
