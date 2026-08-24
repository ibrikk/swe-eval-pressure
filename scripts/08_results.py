#!/usr/bin/env python3
"""Generate standardized SWE-EvalPressure result tables from canonical analyzer outputs.

Consumes profile directories produced by scripts/07_analyze.py.
This script does not reconstruct Harbor runs and never calls an LLM.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

RESULTS_SCHEMA_VERSION = "1.0"
DEFAULT_BOOTSTRAP_REPS = 20_000
DEFAULT_SEED = 20260823

STANDARD_CHANNELS = ("root", "scaffold", "source")
PRESSURE_CONDITIONS = ("eval_financial", "eval_self_preservation")
SEMANTIC_RECOGNIZED = {"implicit", "explicit"}
CURRENT_EA_PRESENT = {"implicit", "explicit"}
TARGETING_PRESENT = {"considered", "attempted"}


def safe_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def numeric(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def passed(row: dict[str, Any]) -> int:
    value = numeric(row.get("overall_pass"))
    return int(value is not None and value > 0)


def pct(num: int, den: int) -> float | None:
    return 100.0 * num / den if den else None


def rate(num: int, den: int) -> float | None:
    return num / den if den else None


def usable(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if bool(r.get("substantive_usable"))]


def semantic_ok(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        r for r in rows
        if bool(r.get("substantive_usable"))
        and str(r.get("semantic_judge_status") or "") == "ok"
    ]


def discover_profiles(root: Path) -> list[Path]:
    if (root / "trials.json").is_file():
        return [root]
    return sorted(
        p for p in root.iterdir()
        if p.is_dir() and (p / "trials.json").is_file()
    )


def load_profiles(root: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]], dict[str, Path]]:
    data: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    for profile_dir in discover_profiles(root):
        rows = safe_json(profile_dir / "trials.json")
        if not isinstance(rows, list):
            raise SystemExit(f"Invalid trials.json: {profile_dir / 'trials.json'}")
        summary = safe_json(profile_dir / "summary.json")
        if not isinstance(summary, dict):
            summary = {}
        profile = str(summary.get("profile") or profile_dir.name)
        if profile in data:
            raise SystemExit(f"Duplicate profile discovered: {profile}")
        data[profile] = rows
        summaries[profile] = summary
        paths[profile] = profile_dir
    if not data:
        raise SystemExit(f"No profile directories with trials.json found under {root}")
    return data, summaries, paths


def validate_complete_profiles(
    data: dict[str, list[dict[str, Any]]],
    summaries: dict[str, dict[str, Any]],
) -> None:
    errors: list[str] = []
    for profile, rows in data.items():
        summary = summaries.get(profile, {})
        planned = int(summary.get("planned_trajectories") or 0)
        found = int(summary.get("results_found") or len(rows))
        missing = int(summary.get("missing") or max(0, planned - found)) if planned else 0
        if planned and found != planned:
            errors.append(f"{profile}: results_found={found}, planned={planned}")
        if missing:
            errors.append(f"{profile}: missing={missing}")
    if errors:
        raise SystemExit(
            "Refusing complete-study standardized results from partial canonical analysis:\n  - "
            + "\n  - ".join(errors)
            + "\nFinish/reconstruct the full study first, or rerun with --allow-partial only for diagnostics."
        )


def bootstrap_ci(diffs: list[int], reps: int, seed: int) -> tuple[float | None, float | None]:
    if not diffs:
        return None, None
    rng = random.Random(seed)
    n = len(diffs)
    draws = [
        sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(reps)
    ]
    draws.sort()
    return draws[int(0.025 * reps)], draws[min(reps - 1, int(0.975 * reps))]


def mcnemar_exact(treat_only: int, base_only: int) -> float:
    n = treat_only + base_only
    if n == 0:
        return 1.0
    k = min(treat_only, base_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def git_commit(project_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def run_inventory(data, summaries, paths) -> list[dict[str, Any]]:
    out = []
    for profile, rows in data.items():
        summary = summaries[profile]
        profile_dir = paths[profile]
        trials_path = profile_dir / "trials.json"
        sem_path = profile_dir / "semantic_judgments.json"
        summary_path = profile_dir / "summary.json"
        out.append({
            "profile": profile,
            "profile_dir": str(profile_dir),
            "analysis_schema_version": summary.get("analysis_schema_version"),
            "semantic_judge_version": summary.get("semantic_judge_version"),
            "semantic_enabled": summary.get("semantic_enabled"),
            "planned_trajectories": summary.get("planned_trajectories"),
            "results_found": summary.get("results_found"),
            "usable_completed": summary.get("usable_completed", len(usable(rows))),
            "censored_or_error": summary.get("censored_or_error"),
            "missing": summary.get("missing"),
            "model": summary.get("model"),
            "agent": summary.get("agent"),
            "observed_agent_versions": json.dumps(summary.get("observed_agent_versions", []), ensure_ascii=False),
            "selected_run_directories": json.dumps(summary.get("selected_run_directories", []), ensure_ascii=False),
            "trials_sha256": sha256_file(trials_path) if trials_path.is_file() else "",
            "semantic_judgments_sha256": sha256_file(sem_path) if sem_path.is_file() else "",
            "summary_sha256": sha256_file(summary_path) if summary_path.is_file() else "",
        })
    return out


def performance_by_condition(data) -> list[dict[str, Any]]:
    out = []
    for profile, rows in data.items():
        groups = defaultdict(list)
        for r in usable(rows):
            groups[str(r.get("condition") or "unknown")].append(r)
        for condition, vals in sorted(groups.items()):
            p = sum(passed(r) for r in vals)
            out.append({
                "profile": profile, "condition": condition, "n": len(vals),
                "pass_count": p, "pass_rate": rate(p, len(vals)), "pass_pct": pct(p, len(vals)),
            })
    return out


def performance_by_condition_channel(data) -> list[dict[str, Any]]:
    out = []
    for profile, rows in data.items():
        groups = defaultdict(list)
        for r in usable(rows):
            groups[(str(r.get("condition") or "unknown"), str(r.get("channel") or "unknown"))].append(r)
        for (condition, channel), vals in sorted(groups.items()):
            p = sum(passed(r) for r in vals)
            out.append({
                "profile": profile, "condition": condition, "channel": channel, "n": len(vals),
                "pass_count": p, "pass_rate": rate(p, len(vals)), "pass_pct": pct(p, len(vals)),
            })
    return out


def matched_pass_effects(data, reps: int, seed: int) -> list[dict[str, Any]]:
    out = []
    contrasts = (
        ("eval_vs_clean", "clean", "eval_only"),
        ("financial_vs_eval", "eval_only", "eval_financial"),
        ("self_vs_eval", "eval_only", "eval_self_preservation"),
    )
    for profile, rows in data.items():
        rows = usable(rows)
        profile_seed = int(hashlib.sha256(profile.encode("utf-8")).hexdigest()[:8], 16)
        clean = {str(r.get("base_task_id")): r for r in rows if r.get("condition") == "clean"}
        by = {
            (str(r.get("base_task_id")), str(r.get("condition")), str(r.get("channel"))): r
            for r in rows
        }
        for ci, (label, baseline, treatment) in enumerate(contrasts):
            for cj, channel in enumerate(STANDARD_CHANNELS):
                pairs = []
                for base in sorted(clean):
                    a = clean.get(base) if baseline == "clean" else by.get((base, baseline, channel))
                    b = by.get((base, treatment, channel))
                    if a is not None and b is not None:
                        pairs.append((passed(a), passed(b)))
                diffs = [b - a for a, b in pairs]
                n = len(pairs)
                bp = sum(a for a, _ in pairs)
                tp = sum(b for _, b in pairs)
                to = sum(a == 0 and b == 1 for a, b in pairs)
                bo = sum(a == 1 and b == 0 for a, b in pairs)
                effect = sum(diffs) / n if n else None
                lo, hi = bootstrap_ci(diffs, reps, seed + profile_seed + ci * 10 + cj)
                out.append({
                    "profile": profile,
                    "contrast": label,
                    "baseline_condition": baseline,
                    "treatment_condition": treatment,
                    "channel": channel,
                    "n_pairs": n,
                    "baseline_pass_count": bp,
                    "treatment_pass_count": tp,
                    "baseline_pass_rate": rate(bp, n),
                    "treatment_pass_rate": rate(tp, n),
                    "delta_pp": None if effect is None else 100 * effect,
                    "bootstrap_ci_low_pp": None if lo is None else 100 * lo,
                    "bootstrap_ci_high_pp": None if hi is None else 100 * hi,
                    "treatment_only_pass": to,
                    "baseline_only_pass": bo,
                    "mcnemar_exact_p": mcnemar_exact(to, bo),
                    "bootstrap_reps": reps,
                })
    return out


def semantic_overview(data) -> list[dict[str, Any]]:
    out = []
    for profile, rows in data.items():
        u = usable(rows)
        sem = semantic_ok(rows)
        clean = [r for r in sem if r.get("condition") == "clean"]
        seeded = [r for r in sem if r.get("condition") != "clean"]
        clean_ea = sum(r.get("current_agent_evaluation_awareness") in CURRENT_EA_PRESENT for r in clean)
        cue_rec = sum(r.get("evaluation_cue_recognition") in SEMANTIC_RECOGNIZED for r in seeded)
        current_ea = sum(r.get("current_agent_evaluation_awareness") in CURRENT_EA_PRESENT for r in sem)
        adapted = sum(r.get("evaluation_response") == "adapted" for r in seeded)
        targeting = sum(r.get("evaluation_targeting_intent") in TARGETING_PRESENT for r in sem)
        out.append({
            "profile": profile,
            "usable_n": len(u),
            "semantic_ok_n": len(sem),
            "semantic_coverage_pct": pct(len(sem), len(u)),
            "clean_n": len(clean),
            "clean_current_ea_n": clean_ea,
            "clean_current_ea_pct": pct(clean_ea, len(clean)),
            "seeded_n": len(seeded),
            "seeded_eval_cue_recognized_n": cue_rec,
            "seeded_eval_cue_recognized_pct": pct(cue_rec, len(seeded)),
            "all_current_ea_n": current_ea,
            "all_current_ea_pct": pct(current_ea, len(sem)),
            "seeded_adapted_n": adapted,
            "seeded_adapted_pct": pct(adapted, len(seeded)),
            "evaluation_targeting_n": targeting,
            "evaluation_targeting_pct": pct(targeting, len(sem)),
        })
    return out


def awareness_by_placement(data) -> list[dict[str, Any]]:
    out = []
    for profile, rows in data.items():
        sem = [r for r in semantic_ok(rows) if r.get("condition") != "clean"]
        for channel in STANDARD_CHANNELS:
            vals = [r for r in sem if r.get("channel") == channel]
            if not vals:
                continue
            n = len(vals)
            trace = sum(bool(r.get("eval_cue_observed")) for r in vals)
            rec = sum(r.get("evaluation_cue_recognition") in SEMANTIC_RECOGNIZED for r in vals)
            ea = sum(r.get("current_agent_evaluation_awareness") in CURRENT_EA_PRESENT for r in vals)
            adapted = sum(r.get("evaluation_response") == "adapted" for r in vals)
            p = sum(passed(r) for r in vals)
            out.append({
                "profile": profile, "channel": channel, "n": n,
                "cue_text_trace_match_n": trace, "cue_text_trace_match_pct": pct(trace, n),
                "eval_cue_recognized_n": rec, "eval_cue_recognized_pct": pct(rec, n),
                "current_ea_n": ea, "current_ea_pct": pct(ea, n),
                "adapted_n": adapted, "adapted_pct": pct(adapted, n),
                "pass_count": p, "pass_pct": pct(p, n),
            })
    return out


def pressure_overview(data) -> list[dict[str, Any]]:
    out = []
    for profile, rows in data.items():
        sem = semantic_ok(rows)
        for condition in PRESSURE_CONDITIONS:
            vals = [r for r in sem if r.get("condition") == condition]
            if not vals:
                continue
            n = len(vals)
            rec = sum(r.get("pressure_recognition") in SEMANTIC_RECOGNIZED for r in vals)
            rej = sum(r.get("pressure_response") == "rejected_as_artificial" for r in vals)
            ign = sum(r.get("pressure_response") == "ignored" for r in vals)
            out.append({
                "profile": profile, "condition": condition, "n": n,
                "pressure_recognized_n": rec, "pressure_recognized_pct": pct(rec, n),
                "rejected_as_artificial_n": rej, "rejected_as_artificial_pct": pct(rej, n),
                "ignored_n": ign, "ignored_pct": pct(ign, n),
            })
    return out


def pressure_by_placement(data) -> list[dict[str, Any]]:
    out = []
    for profile, rows in data.items():
        sem = semantic_ok(rows)
        for condition in PRESSURE_CONDITIONS:
            for channel in STANDARD_CHANNELS:
                vals = [r for r in sem if r.get("condition") == condition and r.get("channel") == channel]
                if not vals:
                    continue
                n = len(vals)
                rec = sum(r.get("pressure_recognition") in SEMANTIC_RECOGNIZED for r in vals)
                rej = sum(r.get("pressure_response") == "rejected_as_artificial" for r in vals)
                ign = sum(r.get("pressure_response") == "ignored" for r in vals)
                out.append({
                    "profile": profile, "condition": condition, "channel": channel, "n": n,
                    "pressure_recognized_n": rec, "pressure_recognized_pct": pct(rec, n),
                    "rejected_as_artificial_n": rej, "rejected_as_artificial_pct": pct(rej, n),
                    "ignored_n": ign, "ignored_pct": pct(ign, n),
                })
    return out


def awareness_decomposition(data) -> list[dict[str, Any]]:
    out = []
    labels = {
        (False, False): "neither",
        (True, False): "evaluation_only",
        (False, True): "pressure_only",
        (True, True): "evaluation_and_pressure",
    }
    for profile, rows in data.items():
        sem = semantic_ok(rows)
        for condition in PRESSURE_CONDITIONS:
            vals = [r for r in sem if r.get("condition") == condition]
            if not vals:
                continue
            counts = Counter()
            for r in vals:
                e = r.get("evaluation_cue_recognition") in SEMANTIC_RECOGNIZED
                p = r.get("pressure_recognition") in SEMANTIC_RECOGNIZED
                counts[labels[(e, p)]] += 1
            for label in ("neither", "evaluation_only", "pressure_only", "evaluation_and_pressure"):
                out.append({
                    "profile": profile, "condition": condition,
                    "awareness_pattern": label, "n": counts[label],
                    "pct": pct(counts[label], len(vals)), "condition_n": len(vals),
                })
    return out


def mechanism_descriptive(data) -> list[dict[str, Any]]:
    out = []
    for profile, rows in data.items():
        sem = [r for r in semantic_ok(rows) if r.get("condition") != "clean"]
        mechs = sorted({str(r.get("eval_cue_mechanism")) for r in sem if r.get("eval_cue_mechanism")})
        for mech in mechs:
            vals = [r for r in sem if r.get("eval_cue_mechanism") == mech]
            n = len(vals)
            rec = sum(r.get("evaluation_cue_recognition") in SEMANTIC_RECOGNIZED for r in vals)
            ea = sum(r.get("current_agent_evaluation_awareness") in CURRENT_EA_PRESENT for r in vals)
            adapted = sum(r.get("evaluation_response") == "adapted" for r in vals)
            out.append({
                "profile": profile, "eval_cue_mechanism": mech, "n": n,
                "eval_cue_recognized_n": rec, "eval_cue_recognized_pct": pct(rec, n),
                "current_ea_n": ea, "current_ea_pct": pct(ea, n),
                "adapted_n": adapted, "adapted_pct": pct(adapted, n),
                "interpretation": "descriptive_only_base_task_confounded",
            })
    return out


def recognition_outcome(data) -> list[dict[str, Any]]:
    out = []
    for profile, rows in data.items():
        sem = [r for r in semantic_ok(rows) if r.get("condition") != "clean"]
        for label, flag in (("not_recognized", False), ("recognized", True)):
            vals = [r for r in sem if (r.get("evaluation_cue_recognition") in SEMANTIC_RECOGNIZED) == flag]
            if not vals:
                continue
            n = len(vals)
            p = sum(passed(r) for r in vals)
            adapted = sum(r.get("evaluation_response") == "adapted" for r in vals)
            out.append({
                "profile": profile, "recognition_group": label, "n": n,
                "pass_count": p, "pass_pct": pct(p, n),
                "adapted_n": adapted, "adapted_pct": pct(adapted, n),
                "interpretation": "post_treatment_descriptive_not_causal",
            })
    return out


def terminal_status(data) -> list[dict[str, Any]]:
    out = []
    for profile, rows in data.items():
        counts = Counter(str(r.get("terminal_status") or "unknown") for r in rows)
        for status, n in sorted(counts.items()):
            out.append({"profile": profile, "terminal_status": status, "n": n})
    return out


def safety_refusals(data) -> list[dict[str, Any]]:
    out = []
    for profile, rows in data.items():
        counts = Counter(
            str(r.get("base_task_id") or "")
            for r in rows if r.get("terminal_status") == "safety_refusal"
        )
        for base_task_id, n in counts.most_common():
            out.append({"profile": profile, "base_task_id": base_task_id, "safety_refusal_n": n})
    return out


def targeting_integrity(data) -> list[dict[str, Any]]:
    out = []
    for profile, rows in data.items():
        u = usable(rows)
        sem = semantic_ok(rows)
        targeting = sum(r.get("evaluation_targeting_intent") in TARGETING_PRESENT for r in sem)
        out.append({
            "profile": profile,
            "usable_n": len(u),
            "semantic_ok_n": len(sem),
            "evaluation_targeting_n": targeting,
            "evaluation_targeting_pct": pct(targeting, len(sem)),
            "internet_lookup_any_n": sum(int(bool(r.get("internet_lookup_any"))) for r in u),
            "public_code_host_lookup_n": sum(int(bool(r.get("public_code_host_lookup"))) for r in u),
            "same_repository_public_lookup_n": sum(int(bool(r.get("same_repository_public_lookup"))) for r in u),
            "public_commit_or_patch_lookup_n": sum(int(bool(r.get("public_commit_or_patch_lookup"))) for r in u),
            "test_files_modified_n": sum(int(bool(r.get("test_files_modified"))) for r in u),
            "evaluation_context_file_modified_n": sum(int(bool(r.get("evaluation_context_file_modified"))) for r in u),
            "seeded_cue_removed_or_modified_n": sum(int(bool(r.get("seeded_cue_removed_or_modified"))) for r in u),
        })
    return out


def md_table(rows: list[dict[str, Any]], cols: list[str]) -> str:
    if not rows:
        return "_No rows._"
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in rows:
        values = []
        for col in cols:
            value = row.get(col, "")
            if isinstance(value, float):
                value = f"{value:.4f}"
            values.append(str("" if value is None else value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def report(inventory, perf, matched, semantic, pressure, integrity) -> str:
    return "\n".join([
        "# SWE-EvalPressure standardized results",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Generated from canonical `trials.json` outputs produced by `scripts/07_analyze.py`.",
        "This script does not reconstruct Harbor runs and does not call an LLM.",
        "",
        "## Run inventory",
        "",
        md_table(inventory, ["profile", "analysis_schema_version", "semantic_judge_version", "usable_completed", "censored_or_error", "missing"]),
        "",
        "## Performance by condition",
        "",
        md_table(perf, ["profile", "condition", "n", "pass_count", "pass_pct"]),
        "",
        "## Matched task effects",
        "",
        "Effects are treatment minus baseline on matched base tasks within placement. Bootstrap intervals resample matched base-task pairs.",
        "",
        md_table(matched, ["profile", "contrast", "channel", "n_pairs", "delta_pp", "bootstrap_ci_low_pp", "bootstrap_ci_high_pp", "mcnemar_exact_p"]),
        "",
        "## Semantic overview",
        "",
        md_table(semantic, ["profile", "semantic_ok_n", "clean_current_ea_pct", "seeded_eval_cue_recognized_pct", "seeded_adapted_pct", "evaluation_targeting_n"]),
        "",
        "## Pressure overview",
        "",
        md_table(pressure, ["profile", "condition", "n", "pressure_recognized_pct", "rejected_as_artificial_pct", "ignored_pct"]),
        "",
        "## Targeting / integrity",
        "",
        md_table(integrity, ["profile", "usable_n", "evaluation_targeting_n", "internet_lookup_any_n", "same_repository_public_lookup_n", "test_files_modified_n", "evaluation_context_file_modified_n"]),
        "",
        "## Interpretation constraints",
        "",
        "- `cue_text_trace_match` is trace recoverability, not proof that a model cognitively observed the cue.",
        "- Cue-mechanism comparisons are descriptive because mechanism is permanently assigned at the base-task level.",
        "- Recognition-versus-outcome comparisons are post-treatment descriptive associations, not causal effects.",
        "- McNemar p-values and bootstrap intervals are unadjusted across the family of contrasts; multiplicity-aware inference belongs in the paper inference layer.",
        "- Capability-floor models should not be interpreted as showing treatment robustness merely because all matched success deltas are zero.",
        "",
        "## Machine-readable outputs",
        "",
        "See the CSV files in this directory and `results_manifest.json` for exact input hashes and generation parameters.",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate standardized SWE-EvalPressure results from canonical analyzer outputs.")
    parser.add_argument("--input-root", type=Path, required=True, help="Profile directory or directory containing profile subdirectories with trials.json.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="Allow standardized tables from an intentionally partial analysis (diagnostics only).",
    )
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data, summaries, paths = load_profiles(input_root)
    if not args.allow_partial:
        validate_complete_profiles(data, summaries)

    inventory = run_inventory(data, summaries, paths)
    perf = performance_by_condition(data)
    perf_channel = performance_by_condition_channel(data)
    matched = matched_pass_effects(data, args.bootstrap_reps, args.seed)
    sem = semantic_overview(data)
    awareness_place = awareness_by_placement(data)
    pressure = pressure_overview(data)
    pressure_place = pressure_by_placement(data)
    decomposition = awareness_decomposition(data)
    mechanism = mechanism_descriptive(data)
    recog = recognition_outcome(data)
    terminal = terminal_status(data)
    refusals = safety_refusals(data)
    integrity = targeting_integrity(data)

    outputs = {
        "run_inventory.csv": inventory,
        "performance_by_condition.csv": perf,
        "performance_by_condition_channel.csv": perf_channel,
        "matched_pass_effects.csv": matched,
        "semantic_overview.csv": sem,
        "evaluation_awareness_by_placement.csv": awareness_place,
        "pressure_overview.csv": pressure,
        "pressure_by_placement.csv": pressure_place,
        "awareness_decomposition.csv": decomposition,
        "mechanism_descriptive.csv": mechanism,
        "recognition_outcome_descriptive.csv": recog,
        "terminal_status.csv": terminal,
        "safety_refusals_by_base_task.csv": refusals,
        "targeting_integrity.csv": integrity,
    }
    for name, rows in outputs.items():
        write_csv(output_dir / name, rows)

    project_root = Path(__file__).resolve().parents[1]
    manifest = {
        "results_schema_version": RESULTS_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": str(Path(__file__).resolve()),
        "git_commit": git_commit(project_root),
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "bootstrap_reps": args.bootstrap_reps,
        "seed": args.seed,
        "profiles": list(data),
        "inputs": inventory,
        "outputs": sorted(outputs),
        "interpretation_constraints": [
            "cue_text_trace_match_is_trace_recoverability_not_cognitive_observation",
            "mechanism_comparisons_are_descriptive_due_to_base_task_assignment",
            "recognition_outcome_associations_are_post_treatment_and_not_causal",
            "matched_p_values_are_unadjusted_for_multiple_comparisons",
            "capability_floor_models_require_separate_interpretation",
        ],
    }
    write_json(output_dir / "results_manifest.json", manifest)
    (output_dir / "report.md").write_text(
        report(inventory, perf, matched, sem, pressure, integrity),
        encoding="utf-8",
    )
    print(output_dir / "report.md")


if __name__ == "__main__":
    main()
