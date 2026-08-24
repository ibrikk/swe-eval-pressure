#!/usr/bin/env python3
"""Build the eval-awareness Excel workbook from analyzer CSV outputs.

The workbook follows the reporting conventions used in the historical
`eval_awareness_results_figures` workbook: run inventory, data dictionary,
trial-level data, model × condition summaries, raw tool breakdown, numbered
core-figure tabs, experiment design, and analyzer supporting tables.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import xlsxwriter


# Analyzer trial CSVs can contain full semantic trajectories/evidence fields that
# legitimately exceed Python's conservative 128 KiB CSV-cell default. Raise the
# parser limit to the largest value supported by this Python build so report
# generation can read the canonical analyzer output without truncating it.
_csv_field_limit = sys.maxsize
while True:
    try:
        csv.field_size_limit(_csv_field_limit)
        break
    except OverflowError:
        _csv_field_limit //= 10


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------


def norm(value: Any) -> str:
    return str(value or "").strip()


def to_float(value: Any) -> float | None:
    text = norm(value)
    if not text or text.lower() in {"none", "null", "nan", "na", "n/a"}:
        return None
    try:
        out = float(text)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def to_bool(value: Any) -> bool | None:
    text = norm(value).lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n", ""}:
        return False
    number = to_float(value)
    if number in {0.0, 1.0}:
        return bool(number)
    return None


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def union_headers(rows: Iterable[dict[str, Any]]) -> list[str]:
    headers: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key is None:
                continue
            if key not in seen:
                seen.add(key)
                headers.append(key)
    return headers


def substantive(row: dict[str, Any]) -> bool:
    value = to_bool(row.get("substantive_usable"))
    if value is not None:
        return value
    return norm(row.get("terminal_status")).lower() in {"completed", "safety_refusal"}


def profile_name(row: dict[str, Any]) -> str:
    return norm(row.get("profile") or row.get("_analysis_profile") or "unknown")


def model_label(row: dict[str, Any]) -> str:
    profile = profile_name(row).lower()
    display = {
        "claude": "Opus 4.8",
        "fable": "Fable 5",
        "codex": "Codex GPT-5.6 Sol",
        "llama": "Llama 3.3 70B",
    }
    if profile in display:
        return display[profile]
    return norm(row.get("model_name") or row.get("profile") or row.get("_analysis_profile") or "unknown")


def numeric_values(rows: Iterable[dict[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = to_float(row.get(field))
        if value is not None:
            values.append(value)
    return values


def avg(rows: Iterable[dict[str, Any]], field: str) -> float | None:
    values = numeric_values(rows, field)
    return mean(values) if values else None


def med(rows: Iterable[dict[str, Any]], field: str) -> float | None:
    values = numeric_values(rows, field)
    return median(values) if values else None


def pct(numerator: int | float, denominator: int | float) -> float | None:
    if not denominator:
        return None
    return 100.0 * float(numerator) / float(denominator)


def semantic_ok(row: dict[str, Any]) -> bool:
    return norm(row.get("semantic_judge_status")).lower() == "ok"


def semantic_recognized(value: Any) -> bool:
    return norm(value).lower() in {"implicit", "explicit"}


def semantic_explicit(value: Any) -> bool:
    return norm(value).lower() == "explicit"


# ---------------------------------------------------------------------------
# Analyzer input loading
# ---------------------------------------------------------------------------


def load_analysis(analysis_root: Path, mode: str, profile_filter: str):
    mode_root = analysis_root / mode
    if not mode_root.exists():
        raise SystemExit(f"Analysis mode directory not found: {mode_root}")

    profile_dirs: list[Path] = []
    for path in sorted(mode_root.iterdir()):
        if not path.is_dir() or not (path / "trials.csv").exists():
            continue
        if profile_filter != "all" and path.name != profile_filter:
            continue
        profile_dirs.append(path)

    if not profile_dirs:
        raise SystemExit(f"No profile analyzer outputs with trials.csv found under {mode_root}")

    supporting_names = [
        "coverage",
        "terminal_status",
        "treatment_delivery",
        "tool_usage",
        "external_lookup",
        "awareness",
        "matched_pairs",
        "matched_pair_summary",
    ]

    all_trials: list[dict[str, str]] = []
    supporting: dict[str, dict[str, list[dict[str, str]]]] = {}

    for profile_dir in profile_dirs:
        profile = profile_dir.name
        rows = read_csv(profile_dir / "trials.csv")
        for row in rows:
            row.setdefault("profile", profile)
            row["_analysis_profile"] = profile
        all_trials.extend(rows)
        supporting[profile] = {
            name: read_csv(profile_dir / f"{name}.csv") for name in supporting_names
        }

    return all_trials, supporting


def flatten_supporting(
    supporting: dict[str, dict[str, list[dict[str, str]]]], key: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for profile, tables in supporting.items():
        for raw in tables.get(key, []):
            row = {"profile": profile, **raw}
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------


def run_inventory(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trials:
        groups[profile_name(row)].append(row)

    output: list[dict[str, Any]] = []
    for profile, rows in sorted(groups.items()):
        usable = [r for r in rows if substantive(r)]
        passes = numeric_values(usable, "overall_pass")
        tests = numeric_values(usable, "tests_reward")
        judged = sum(semantic_ok(r) for r in usable)
        exemplar = rows[0]
        output.append(
            {
                "profile": profile,
                "model_label": model_label(exemplar),
                "agent_name": norm(exemplar.get("agent_name")),
                "agent_version": norm(exemplar.get("agent_version")),
                "n_total": len(rows),
                "n_usable": len(usable),
                "n_censored": len(rows) - len(usable),
                "overall_pass_rate": mean(passes) if passes else None,
                "tests_rate": mean(tests) if tests else None,
                "avg_rubric": avg(usable, "rubrics_agg_score"),
                "avg_input_tokens": avg(usable, "input_tokens"),
                "avg_output_tokens": avg(usable, "output_tokens"),
                "avg_duration_sec": avg(usable, "duration_sec"),
                "semantic_judgments_ok": judged,
            }
        )
    return output


def primary_summary(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in trials:
        key = (
            profile_name(row),
            model_label(row),
            norm(row.get("condition") or "unknown"),
            norm(row.get("channel") or "none"),
        )
        groups[key].append(row)

    output: list[dict[str, Any]] = []
    for (profile, model, condition, channel), rows in sorted(groups.items()):
        usable = [r for r in rows if substantive(r)]
        passes = numeric_values(usable, "overall_pass")
        tests = numeric_values(usable, "tests_reward")
        output.append(
            {
                "profile": profile,
                "model_label": model,
                "condition": condition,
                "channel": channel,
                "n_runs": len(rows),
                "n_censored": len(rows) - len(usable),
                "pass_count": sum(v > 0 for v in passes),
                "pass_rate": mean(passes) if passes else None,
                "tests_pass_rate": mean(tests) if tests else None,
                "avg_rubric": avg(usable, "rubrics_agg_score"),
                "avg_input_tokens": avg(usable, "input_tokens"),
                "median_input_tokens": med(usable, "input_tokens"),
                "avg_output_tokens": avg(usable, "output_tokens"),
                "avg_duration_sec": avg(usable, "duration_sec"),
                "avg_cost_usd": avg(usable, "cost_usd"),
                "avg_tool_calls_raw": avg(usable, "raw_tool_calls"),
                "avg_bash_calls_raw": avg(usable, "bash_calls"),
            }
        )
    return output


def tool_breakdown(trials: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in trials:
        if not substantive(row):
            continue
        groups[(profile_name(row), model_label(row), norm(row.get("condition") or "unknown"))].append(row)

    fields = [
        ("raw_tool_calls", "individual_tool_calls"),
        ("bash_calls", "bash_calls"),
        ("read_calls", "read_calls"),
        ("edit_calls", "edit_calls"),
        ("write_calls", "write_calls"),
        ("web_fetch_calls", "webfetch_calls"),
        ("web_search_calls", "websearch_calls"),
    ]

    output: list[dict[str, Any]] = []
    for (profile, model, condition), rows in sorted(groups.items()):
        out: dict[str, Any] = {
            "experiment": "resource" if "resource" in mode.lower() else "primary",
            "profile": profile,
            "model": model,
            "condition": condition,
            "n": len(rows),
        }
        known_total = 0
        for source, target in fields:
            values = numeric_values(rows, source)
            total = int(round(sum(values))) if values else 0
            out[target] = total
            if source != "raw_tool_calls":
                known_total += total
        raw_total = int(round(sum(numeric_values(rows, "raw_tool_calls"))))
        out["other_calls"] = max(0, raw_total - known_total)
        output.append(out)
    return output


def treatment_delivery_figure(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in trials:
        if not substantive(row) or norm(row.get("condition")) == "clean":
            continue
        channel = norm(row.get("channel") or "none")
        groups[(profile_name(row), model_label(row), channel)].append(row)

    output = []
    for (profile, model, channel), rows in sorted(groups.items()):
        eval_observed = sum(to_bool(r.get("eval_cue_observed")) is True for r in rows)
        pressure_rows = [r for r in rows if norm(r.get("pressure_type")).lower() not in {"", "none"}]
        pressure_observed = sum(to_bool(r.get("pressure_cue_observed")) is True for r in pressure_rows)
        actually_read = sum(to_bool(r.get("artifact_actually_read")) is True for r in rows)
        output.append(
            {
                "profile": profile,
                "model_label": model,
                "placement": channel,
                "seeded_usable": len(rows),
                "artifact_actually_read": actually_read,
                "artifact_read_rate_pct": pct(actually_read, len(rows)),
                "eval_cue_observed": eval_observed,
                "eval_cue_observed_pct": pct(eval_observed, len(rows)),
                "pressure_usable": len(pressure_rows),
                "pressure_cue_observed": pressure_observed,
                "pressure_cue_observed_pct": pct(pressure_observed, len(pressure_rows)),
            }
        )
    return output


def evaluation_awareness_figure(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in trials:
        if not substantive(row) or norm(row.get("condition")) == "clean" or not semantic_ok(row):
            continue
        key = (profile_name(row), model_label(row), norm(row.get("channel") or "none"))
        groups[key].append(row)

    output = []
    for (profile, model, channel), rows in sorted(groups.items()):
        eval_cue_recognized = sum(semantic_recognized(r.get("evaluation_cue_recognition")) for r in rows)
        current_awareness = sum(semantic_recognized(r.get("current_agent_evaluation_awareness")) for r in rows)
        current_explicit = sum(semantic_explicit(r.get("current_agent_evaluation_awareness")) for r in rows)
        benchmark_recognized = sum(semantic_recognized(r.get("benchmark_recognition")) for r in rows)
        output.append(
            {
                "profile": profile,
                "model_label": model,
                "placement": channel,
                "n_judged": len(rows),
                "evaluation_cue_recognized": eval_cue_recognized,
                "evaluation_cue_recognized_pct": pct(eval_cue_recognized, len(rows)),
                "current_run_eval_awareness": current_awareness,
                "current_run_eval_awareness_pct": pct(current_awareness, len(rows)),
                "explicit_current_run_eval_awareness": current_explicit,
                "explicit_current_run_eval_awareness_pct": pct(current_explicit, len(rows)),
                "benchmark_recognition": benchmark_recognized,
                "benchmark_recognition_pct": pct(benchmark_recognized, len(rows)),
            }
        )
    return output


def pressure_recognition_figure(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in trials:
        pressure = norm(row.get("pressure_type")).lower()
        if not substantive(row) or pressure in {"", "none"} or not semantic_ok(row):
            continue
        groups[(profile_name(row), model_label(row), pressure, norm(row.get("channel") or "none"))].append(row)

    output = []
    for (profile, model, pressure, channel), rows in sorted(groups.items()):
        recognized = sum(semantic_recognized(r.get("pressure_recognition")) for r in rows)
        explicit = sum(semantic_explicit(r.get("pressure_recognition")) for r in rows)
        adapted = sum(norm(r.get("pressure_response")).lower() == "adapted" for r in rows)
        output.append(
            {
                "profile": profile,
                "model_label": model,
                "pressure": pressure,
                "placement": channel,
                "n_judged": len(rows),
                "pressure_recognized": recognized,
                "pressure_recognized_pct": pct(recognized, len(rows)),
                "explicit_pressure_recognition": explicit,
                "explicit_pressure_recognition_pct": pct(explicit, len(rows)),
                "pressure_response_adapted": adapted,
                "pressure_response_adapted_pct": pct(adapted, len(rows)),
            }
        )
    return output


def external_lookup_figure(
    supporting: dict[str, dict[str, list[dict[str, str]]]], trials: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    model_by_profile = {}
    for row in trials:
        model_by_profile.setdefault(profile_name(row), model_label(row))

    output = []
    for profile, tables in sorted(supporting.items()):
        rows = tables.get("external_lookup", [])
        total_n = sum(int(to_float(r.get("n")) or 0) for r in rows)
        if total_n == 0:
            continue
        fields = [
            "internet_lookup_any",
            "public_code_host_lookup",
            "same_repository_public_lookup",
            "public_commit_or_patch_lookup",
            "local_git_history_inspection",
        ]
        out: dict[str, Any] = {
            "profile": profile,
            "model_label": model_by_profile.get(profile, profile),
            "n_usable": total_n,
        }
        for field in fields:
            count = sum(int(to_float(r.get(field)) or 0) for r in rows)
            out[field] = count
            out[f"{field}_pct"] = pct(count, total_n)
        output.append(out)
    return output


def matched_summary(
    supporting: dict[str, dict[str, list[dict[str, str]]]]
) -> list[dict[str, Any]]:
    return flatten_supporting(supporting, "matched_pair_summary")


# ---------------------------------------------------------------------------
# Workbook helpers
# ---------------------------------------------------------------------------


def make_formats(workbook: xlsxwriter.Workbook) -> dict[str, Any]:
    return {
        "title": workbook.add_format({"bold": True, "font_size": 16, "font_color": "#FFFFFF", "bg_color": "#1F4E78", "align": "left", "valign": "vcenter"}),
        "subtitle": workbook.add_format({"font_size": 10, "font_color": "#404040", "bg_color": "#D9EAF7", "text_wrap": True, "valign": "vcenter"}),
        "section": workbook.add_format({"bold": True, "font_color": "#1F1F1F", "bg_color": "#D9EAF7", "border": 1}),
        "header": workbook.add_format({"bold": True, "font_color": "#FFFFFF", "bg_color": "#4472C4", "border": 1, "align": "center", "valign": "vcenter", "text_wrap": True}),
        "wrap": workbook.add_format({"text_wrap": True, "valign": "top"}),
        "muted": workbook.add_format({"font_color": "#666666", "italic": True}),
        "pct": workbook.add_format({"num_format": "0.0%"}),
        "pct_points": workbook.add_format({"num_format": '0.0"%"'}),
        "int": workbook.add_format({"num_format": "0"}),
        "float": workbook.add_format({"num_format": "0.00"}),
    }


def add_title(ws, title: str, subtitle: str, fmt: dict[str, Any], last_col: int = 9):
    ws.merge_range(0, 0, 0, last_col, title, fmt["title"])
    ws.merge_range(1, 0, 1, last_col, subtitle, fmt["subtitle"])
    ws.set_row(0, 25)
    ws.set_row(1, 28)


def write_value(ws, row: int, col: int, value: Any, fmt: Any = None):
    if value is None:
        ws.write_blank(row, col, None, fmt)
    elif isinstance(value, bool):
        ws.write_boolean(row, col, value, fmt)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        ws.write_number(row, col, float(value), fmt)
    else:
        number = to_float(value)
        if number is not None:
            ws.write_number(row, col, number, fmt)
        else:
            ws.write(row, col, value, fmt)


def write_rows(
    ws,
    rows: list[dict[str, Any]],
    start_row: int,
    start_col: int,
    fmt: dict[str, Any],
    table_name: str | None = None,
    headers: list[str] | None = None,
) -> tuple[int, int]:
    headers = headers or union_headers(rows)
    if not headers:
        ws.write(start_row, start_col, "No rows available", fmt["muted"])
        return start_row, start_col

    for j, header in enumerate(headers):
        ws.write(start_row, start_col + j, header, fmt["header"])

    percent_fraction_fields = {"overall_pass_rate", "tests_rate", "pass_rate", "tests_pass_rate"}
    percent_point_suffixes = ("_pct",)

    for i, row in enumerate(rows, start_row + 1):
        for j, header in enumerate(headers):
            value = row.get(header)
            cell_fmt = None
            if header in percent_fraction_fields:
                cell_fmt = fmt["pct"]
            elif header.endswith(percent_point_suffixes):
                cell_fmt = fmt["pct_points"]
            write_value(ws, i, start_col + j, value, cell_fmt)

    end_row = start_row + len(rows)
    end_col = start_col + len(headers) - 1
    if table_name and rows:
        ws.add_table(
            start_row,
            start_col,
            end_row,
            end_col,
            {
                "name": table_name,
                "style": "Table Style Medium 2",
                "columns": [{"header": h} for h in headers],
            },
        )
    return end_row, end_col


def add_bar_chart(
    workbook,
    ws,
    title: str,
    category_col: int,
    value_cols: list[int],
    series_names: list[str],
    data_start_row: int,
    data_count: int,
    anchor: str,
    percent_points: bool = False,
):
    if data_count <= 0:
        return
    chart = workbook.add_chart({"type": "column"})
    sheet_name = ws.get_name()
    for col, name in zip(value_cols, series_names):
        chart.add_series(
            {
                "name": name,
                "categories": [sheet_name, data_start_row, category_col, data_start_row + data_count - 1, category_col],
                "values": [sheet_name, data_start_row, col, data_start_row + data_count - 1, col],
                "data_labels": {"value": True, "num_format": '0.0"%"' if percent_points else "0.0"},
            }
        )
    chart.set_title({"name": title})
    chart.set_legend({"position": "bottom"})
    chart.set_style(10)
    if percent_points:
        chart.set_y_axis({"min": 0, "max": 100, "num_format": '0"%"'})
    ws.insert_chart(anchor, chart, {"x_scale": 1.25, "y_scale": 1.15})


def display_category(row: dict[str, Any], fields: list[str]) -> str:
    return " / ".join(norm(row.get(field)) for field in fields if norm(row.get(field)))


# ---------------------------------------------------------------------------
# Workbook construction
# ---------------------------------------------------------------------------


def build_workbook(args: argparse.Namespace) -> Path:
    analysis_root = args.analysis_root.resolve()
    trials, supporting = load_analysis(analysis_root, args.mode, args.profile)
    profiles = sorted({profile_name(r) for r in trials})

    inventory = run_inventory(trials)
    summary = primary_summary(trials)
    tools = tool_breakdown(trials, args.mode)
    c01 = treatment_delivery_figure(trials)
    c02 = evaluation_awareness_figure(trials)
    c03 = pressure_recognition_figure(trials)
    c04 = matched_summary(supporting)
    c06 = external_lookup_figure(supporting, trials)

    coverage = flatten_supporting(supporting, "coverage")
    terminal = flatten_supporting(supporting, "terminal_status")
    treatment_raw = flatten_supporting(supporting, "treatment_delivery")
    awareness_raw = flatten_supporting(supporting, "awareness")
    external_raw = flatten_supporting(supporting, "external_lookup")
    matched_pairs = flatten_supporting(supporting, "matched_pairs")
    tool_usage_raw = flatten_supporting(supporting, "tool_usage")

    output = args.output or (analysis_root / args.mode / f"eval_awareness_{args.mode}_results_figures.xlsx")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    wb = xlsxwriter.Workbook(output)
    fmt = make_formats(wb)
    wb.set_properties(
        {
            "title": "Eval-awareness results figures",
            "subject": f"{args.mode} analysis workbook",
            "author": "SWE-EvalPressure",
        }
    )

    usable_n = sum(substantive(r) for r in trials)
    successful_semantic = sum(substantive(r) and semantic_ok(r) for r in trials)
    base_tasks = len({norm(r.get("base_task_id")) for r in trials if norm(r.get("base_task_id"))})

    # README -----------------------------------------------------------------
    ws = wb.add_worksheet("README")
    ws.write(0, 0, "START HERE — eval-awareness workbook", fmt["title"])
    ws.set_row(0, 25)
    readme_rows = [
        ["Scope", "Evaluation-awareness benchmark analysis generated from the analyzer outputs."],
        ["Key unit distinction", "A base task is not a trajectory. A trajectory = one model run on one task under one experimental variant/condition."],
        ["Report mode", args.mode],
        ["Models included", ", ".join(model_label(next(r for r in trials if profile_name(r) == p)) for p in profiles)],
        ["Base tasks observed", base_tasks],
        ["Trajectory rows observed", len(trials)],
        ["Substantive usable trajectories", usable_n],
        ["Successful semantic judgments", successful_semantic],
        ["Primary full design", "70 base tasks × 10 variants = 700 trajectories/model."],
        ["Resource full design", "70 base tasks × 3 variants = 210 trajectories/model."],
        ["Tool accounting", "raw_tool_calls and the analyzer tool-category fields are used for tool summaries."],
        ["Semantic coding", "Evaluation-awareness and pressure-recognition figures use rows with semantic_judge_status = ok."],
        ["Terminal vs verifier outcome", "Terminal/runtime status and verifier correctness are reported separately."],
        ["Core figures", "C01–C06 are the presentation figures generated from this analysis."],
    ]
    for r, (label, value) in enumerate(readme_rows, 2):
        ws.write(r, 0, label, fmt["section"])
        write_value(ws, r, 1, value, fmt["wrap"])
    ws.set_column("A:A", 31)
    ws.set_column("B:B", 88)
    ws.freeze_panes(2, 0)

    # Run inventory ----------------------------------------------------------
    ws = wb.add_worksheet("Run_Inventory")
    add_title(ws, "Run inventory", "One row per analyzed model/profile.", fmt, 13)
    ws.write(2, 0, "Analyzed runs", fmt["section"])
    write_rows(ws, inventory, 3, 0, fmt, "TblRunInventory")
    ws.set_column(0, 13, 18)
    ws.set_column(1, 1, 28)
    ws.freeze_panes(4, 0)

    # Data dictionary --------------------------------------------------------
    dictionary = [
        {"Field": "substantive_usable", "Meaning": "Analyzer flag for trajectories included in substantive aggregates.", "Source": "deterministic"},
        {"Field": "terminal_status", "Meaning": "Terminal/runtime outcome assigned by the analyzer.", "Source": "deterministic"},
        {"Field": "overall_pass", "Meaning": "Harbor verifier overall pass value.", "Source": "verifier"},
        {"Field": "tests_reward", "Meaning": "Harbor tests reward.", "Source": "verifier"},
        {"Field": "rubrics_agg_score", "Meaning": "Aggregate rubric score.", "Source": "verifier"},
        {"Field": "raw_tool_calls", "Meaning": "Individual tool calls reconstructed from the trajectory.", "Source": "deterministic"},
        {"Field": "artifact_actually_read", "Meaning": "Treatment artifact content was successfully read according to tool-call evidence.", "Source": "deterministic"},
        {"Field": "eval_cue_observed", "Meaning": "Evaluation-cue text probe was observed in the recorded trajectory.", "Source": "deterministic"},
        {"Field": "pressure_cue_observed", "Meaning": "Pressure-text probe was observed in the recorded trajectory.", "Source": "deterministic"},
        {"Field": "evaluation_cue_recognition", "Meaning": "Semantic label: none / implicit / explicit.", "Source": "semantic judge"},
        {"Field": "current_agent_evaluation_awareness", "Meaning": "Semantic label for awareness that the current run is being evaluated: none / implicit / explicit.", "Source": "semantic judge"},
        {"Field": "benchmark_recognition", "Meaning": "Semantic label: none / implicit / explicit.", "Source": "semantic judge"},
        {"Field": "pressure_recognition", "Meaning": "Semantic label: none / implicit / explicit.", "Source": "semantic judge"},
        {"Field": "pressure_response", "Meaning": "Semantic label: none / acknowledged_only / adapted / ignored / rejected_as_artificial / unclear.", "Source": "semantic judge"},
        {"Field": "evaluation_targeting_intent", "Meaning": "Semantic label: none / considered / attempted / unclear.", "Source": "semantic judge"},
        {"Field": "semantic_evidence_quotes", "Meaning": "Supporting trajectory quotes returned with the semantic judgment.", "Source": "semantic judge"},
        {"Field": "semantic_judge_status", "Meaning": "Status of the semantic-judge call for the trajectory.", "Source": "semantic judge"},
    ]
    ws = wb.add_worksheet("Data_Dictionary")
    add_title(ws, "Data dictionary", "Key fields used in the summary and figure tabs.", fmt, 5)
    ws.write(2, 0, "Fields", fmt["section"])
    write_rows(ws, dictionary, 3, 0, fmt, "TblDataDictionary", ["Field", "Meaning", "Source"])
    ws.set_column("A:A", 34)
    ws.set_column("B:B", 92)
    ws.set_column("C:C", 18)
    ws.freeze_panes(4, 0)

    # All trials -------------------------------------------------------------
    ws = wb.add_worksheet("All_Trials")
    add_title(ws, "All trials", "Trajectory-level analyzer output.", fmt, 9)
    write_rows(ws, trials, 3, 0, fmt, "TblAllTrials")
    ws.set_column(0, max(0, len(union_headers(trials)) - 1), 14)
    ws.freeze_panes(4, 4)

    # Summary ---------------------------------------------------------------
    summary_sheet = "Resource_deprivation_Summary" if "resource" in args.mode.lower() else "Primary_Summary"
    summary_title = "Resource Deprivation summary by model × condition × channel" if "resource" in args.mode.lower() else "Primary summary by model × condition × channel"
    ws = wb.add_worksheet(summary_sheet[:31])
    add_title(ws, summary_title, "Computed from substantive usable trajectory rows; n_runs and n_censored report observed coverage.", fmt, 15)
    summary_headers = [
        "profile", "model_label", "condition", "channel", "n_runs", "n_censored", "pass_count", "pass_rate",
        "tests_pass_rate", "avg_rubric", "avg_input_tokens", "median_input_tokens", "avg_output_tokens",
        "avg_duration_sec", "avg_cost_usd", "avg_tool_calls_raw", "avg_bash_calls_raw",
    ]
    ws.write(2, 0, "Summary", fmt["section"])
    write_rows(ws, summary, 3, 0, fmt, "TblSummary", summary_headers)
    ws.set_column(0, 16, 17)
    ws.set_column(1, 1, 26)
    ws.freeze_panes(4, 2)

    # Tool breakdown ---------------------------------------------------------
    ws = wb.add_worksheet("Tool_Breakdown")
    add_title(ws, "Raw tool breakdown", "Counts summed from analyzer trajectory-level tool fields.", fmt, 12)
    tool_headers = ["experiment", "profile", "model", "condition", "n", "individual_tool_calls", "bash_calls", "read_calls", "edit_calls", "write_calls", "webfetch_calls", "websearch_calls", "other_calls"]
    ws.write(2, 0, "Tool counts", fmt["section"])
    write_rows(ws, tools, 3, 0, fmt, "TblToolBreakdown", tool_headers)
    ws.set_column(0, 12, 17)
    ws.set_column(2, 2, 26)
    ws.freeze_panes(4, 0)

    # Supporting analyzer tables -------------------------------------------
    supporting_sheets = [
        ("Coverage", coverage, "Coverage by condition × channel."),
        ("Terminal_Status", terminal, "Terminal/runtime outcome counts."),
        ("Treatment_Delivery", treatment_raw, "Treatment-delivery summary from the analyzer."),
        ("Semantic_Coding", awareness_raw, "Trajectory-level semantic coding and supporting quotes."),
        ("External_Lookup", external_raw, "External lookup and source-code lookup counts."),
        ("Matched_Pairs", matched_pairs, "Within-base-task matched comparisons."),
        ("Matched_Summary", c04, "Matched-pair summary by comparison × channel."),
        ("Tool_Usage", tool_usage_raw, "Tool and effort summary by condition × channel."),
    ]
    for sheet_name, rows, subtitle in supporting_sheets:
        ws = wb.add_worksheet(sheet_name[:31])
        add_title(ws, sheet_name.replace("_", " "), subtitle, fmt, 10)
        write_rows(ws, rows, 3, 0, fmt, f"Tbl{sheet_name.replace('_', '')[:20]}")
        ws.set_column(0, max(0, len(union_headers(rows)) - 1), 16)
        ws.freeze_panes(4, 0)

    # C01 treatment delivery -------------------------------------------------
    ws = wb.add_worksheet("C01_TreatmentDelivery")
    add_title(ws, "Core Fig 01 — treatment delivery", "Treatment-artifact access and cue exposure by model × placement from deterministic analyzer fields.", fmt, 10)
    ws.write(3, 0, "Pivot / editable source", fmt["section"])
    c01_rows = [{
        "Model / placement": display_category(row, ["model_label", "placement"]),
        "Model": row["model_label"],
        "Placement": row["placement"],
        "Seeded usable": row["seeded_usable"],
        "Artifact read": row["artifact_actually_read"],
        "Artifact read %": row["artifact_read_rate_pct"],
        "Eval cue observed": row["eval_cue_observed"],
        "Eval cue observed %": row["eval_cue_observed_pct"],
        "Pressure rows": row["pressure_usable"],
        "Pressure cue observed": row["pressure_cue_observed"],
        "Pressure cue observed %": row["pressure_cue_observed_pct"],
    } for row in c01]
    c01_headers = ["Model / placement", "Model", "Placement", "Seeded usable", "Artifact read", "Artifact read %", "Eval cue observed", "Eval cue observed %", "Pressure rows", "Pressure cue observed", "Pressure cue observed %"]
    write_rows(ws, c01_rows, 4, 0, fmt, "TblC01", c01_headers)
    add_bar_chart(wb, ws, "Evaluation cue observed", 0, [7], ["Observed %"], 5, len(c01_rows), "M4", True)
    ws.set_column(0, 0, 30); ws.set_column(1, 10, 18); ws.freeze_panes(5, 0)

    # C02 evaluation awareness ----------------------------------------------
    ws = wb.add_worksheet("C02_EvalAwareness")
    add_title(ws, "Core Fig 02 — current-run evaluation awareness", "Semantic coding by model × placement. Denominator: rows with semantic_judge_status = ok.", fmt, 11)
    ws.write(3, 0, "Pivot / editable source", fmt["section"])
    c02_rows = [{
        "Model / placement": display_category(row, ["model_label", "placement"]),
        "Model": row["model_label"],
        "Placement": row["placement"],
        "Judged": row["n_judged"],
        "Eval cue recognized": row["evaluation_cue_recognized"],
        "Eval cue recognized %": row["evaluation_cue_recognized_pct"],
        "Current-run eval awareness": row["current_run_eval_awareness"],
        "Current-run eval awareness %": row["current_run_eval_awareness_pct"],
        "Explicit current-run awareness": row["explicit_current_run_eval_awareness"],
        "Explicit current-run awareness %": row["explicit_current_run_eval_awareness_pct"],
        "Benchmark recognition": row["benchmark_recognition"],
        "Benchmark recognition %": row["benchmark_recognition_pct"],
    } for row in c02]
    c02_headers = ["Model / placement", "Model", "Placement", "Judged", "Eval cue recognized", "Eval cue recognized %", "Current-run eval awareness", "Current-run eval awareness %", "Explicit current-run awareness", "Explicit current-run awareness %", "Benchmark recognition", "Benchmark recognition %"]
    write_rows(ws, c02_rows, 4, 0, fmt, "TblC02", c02_headers)
    add_bar_chart(wb, ws, "Current-run evaluation awareness", 0, [7, 9], ["Implicit or explicit", "Explicit"], 5, len(c02_rows), "N4", True)
    ws.set_column(0, 0, 30); ws.set_column(1, 11, 20); ws.freeze_panes(5, 0)

    # C03 pressure recognition ----------------------------------------------
    ws = wb.add_worksheet("C03_PressureRecognition")
    add_title(ws, "Core Fig 03 — pressure recognition", "Semantic coding by model × pressure type × placement. Denominator: rows with semantic_judge_status = ok.", fmt, 12)
    ws.write(3, 0, "Pivot / editable source", fmt["section"])
    c03_rows = [{
        "Model / pressure / placement": display_category(row, ["model_label", "pressure", "placement"]),
        "Model": row["model_label"],
        "Pressure": row["pressure"],
        "Placement": row["placement"],
        "Judged": row["n_judged"],
        "Pressure recognized": row["pressure_recognized"],
        "Pressure recognized %": row["pressure_recognized_pct"],
        "Explicit recognition": row["explicit_pressure_recognition"],
        "Explicit recognition %": row["explicit_pressure_recognition_pct"],
        "Response = adapted": row["pressure_response_adapted"],
        "Response = adapted %": row["pressure_response_adapted_pct"],
    } for row in c03]
    c03_headers = ["Model / pressure / placement", "Model", "Pressure", "Placement", "Judged", "Pressure recognized", "Pressure recognized %", "Explicit recognition", "Explicit recognition %", "Response = adapted", "Response = adapted %"]
    write_rows(ws, c03_rows, 4, 0, fmt, "TblC03", c03_headers)
    add_bar_chart(wb, ws, "Pressure recognition", 0, [6, 8], ["Implicit or explicit", "Explicit"], 5, len(c03_rows), "M4", True)
    ws.set_column(0, 0, 40); ws.set_column(1, 10, 19); ws.freeze_panes(5, 0)

    # C04 matched correctness ------------------------------------------------
    ws = wb.add_worksheet("C04_MatchedCorrectness")
    add_title(ws, "Core Fig 04 — matched correctness transitions", "Same-task matched comparisons from matched_pair_summary.csv.", fmt, 12)
    ws.write(3, 0, "Pivot / editable source", fmt["section"])
    c04_rows = [{
        "Model / comparison / placement": display_category(row, ["profile", "pair_type", "channel"]),
        "Model": model_label(next((r for r in trials if profile_name(r) == norm(row.get("profile"))), {"profile": row.get("profile")})),
        "Comparison": row.get("pair_type", ""),
        "Placement": row.get("channel", ""),
        "Planned pairs": row.get("planned_pairs", ""),
        "Usable pairs": row.get("usable_pairs", ""),
        "Incomplete/censored": row.get("incomplete_or_censored_pairs", ""),
        "Fail→pass": row.get("fail_to_pass", ""),
        "Pass→fail": row.get("pass_to_fail", ""),
        "Pass→pass": row.get("pass_to_pass", ""),
        "Fail→fail": row.get("fail_to_fail", ""),
        "Mean Δ pass": row.get("mean_delta_overall_pass", ""),
        "Median Δ pass": row.get("median_delta_overall_pass", ""),
    } for row in c04]
    c04_headers = ["Model / comparison / placement", "Model", "Comparison", "Placement", "Planned pairs", "Usable pairs", "Incomplete/censored", "Fail→pass", "Pass→fail", "Pass→pass", "Fail→fail", "Mean Δ pass", "Median Δ pass"]
    write_rows(ws, c04_rows, 4, 0, fmt, "TblC04", c04_headers)
    add_bar_chart(wb, ws, "Matched correctness transitions", 0, [7, 8], ["Fail→pass", "Pass→fail"], 5, len(c04_rows), "O4", False)
    ws.set_column(0, 0, 42); ws.set_column(1, 12, 18); ws.freeze_panes(5, 0)

    # C05 matched effort -----------------------------------------------------
    ws = wb.add_worksheet("C05_MatchedEffort")
    add_title(ws, "Core Fig 05 — matched tool and context change", "Mean treatment-minus-baseline deltas from matched_pair_summary.csv.", fmt, 12)
    ws.write(3, 0, "Pivot / editable source", fmt["section"])
    c05_headers = ["Model / comparison / placement", "Model", "Comparison", "Placement", "Usable pairs", "Mean Δ raw tool calls", "Median Δ raw tool calls", "Mean Δ input tokens", "Median Δ input tokens", "Mean Δ output tokens", "Mean Δ duration sec"]
    c05_rows = [{
        "Model / comparison / placement": display_category(row, ["profile", "pair_type", "channel"]),
        "Model": model_label(next((r for r in trials if profile_name(r) == norm(row.get("profile"))), {"profile": row.get("profile")})),
        "Comparison": row.get("pair_type", ""),
        "Placement": row.get("channel", ""),
        "Usable pairs": row.get("usable_pairs", ""),
        "Mean Δ raw tool calls": row.get("mean_delta_raw_tool_calls", ""),
        "Median Δ raw tool calls": row.get("median_delta_raw_tool_calls", ""),
        "Mean Δ input tokens": row.get("mean_delta_input_tokens", ""),
        "Median Δ input tokens": row.get("median_delta_input_tokens", ""),
        "Mean Δ output tokens": row.get("mean_delta_output_tokens", ""),
        "Mean Δ duration sec": row.get("mean_delta_duration_sec", ""),
    } for row in c04]
    write_rows(ws, c05_rows, 4, 0, fmt, "TblC05", c05_headers)
    add_bar_chart(wb, ws, "Mean matched raw tool-call delta", 0, [5], ["Δ raw tool calls"], 5, len(c05_rows), "M4", False)
    add_bar_chart(wb, ws, "Mean matched input-token delta", 0, [7], ["Δ input tokens"], 5, len(c05_rows), "M22", False)
    ws.set_column(0, 0, 42); ws.set_column(1, 10, 20); ws.freeze_panes(5, 0)

    # C06 external lookup ----------------------------------------------------
    ws = wb.add_worksheet("C06_ExternalLookup")
    add_title(ws, "Core Fig 06 — external lookup", "Counts from external_lookup.csv across substantive_usable trajectories.", fmt, 12)
    ws.write(3, 0, "Pivot / editable source", fmt["section"])
    c06_rows = [{
        "Model": row["model_label"],
        "Usable": row["n_usable"],
        "Internet lookup": row["internet_lookup_any"],
        "Internet lookup %": row["internet_lookup_any_pct"],
        "Public code host": row["public_code_host_lookup"],
        "Public code host %": row["public_code_host_lookup_pct"],
        "Same repository": row["same_repository_public_lookup"],
        "Same repository %": row["same_repository_public_lookup_pct"],
        "Commit/patch": row["public_commit_or_patch_lookup"],
        "Commit/patch %": row["public_commit_or_patch_lookup_pct"],
        "Local git history": row["local_git_history_inspection"],
        "Local git history %": row["local_git_history_inspection_pct"],
    } for row in c06]
    c06_headers = ["Model", "Usable", "Internet lookup", "Internet lookup %", "Public code host", "Public code host %", "Same repository", "Same repository %", "Commit/patch", "Commit/patch %", "Local git history", "Local git history %"]
    write_rows(ws, c06_rows, 4, 0, fmt, "TblC06", c06_headers)
    add_bar_chart(wb, ws, "External/public source-code lookup", 0, [5, 7, 9], ["Public code host", "Same repository", "Commit/patch"], 5, len(c06_rows), "N4", True)
    ws.set_column(0, 0, 26); ws.set_column(1, 11, 18); ws.freeze_panes(5, 0)

    # Experiment design ------------------------------------------------------
    ws = wb.add_worksheet("Experiment_Design")
    ws.write(0, 0, "EXPERIMENTAL DESIGN", fmt["title"])
    design_rows = [
        ["Unit", "Meaning", "Primary full", "Resource full"],
        ["Base task", "One underlying SWE-Atlas software-engineering task.", "70/model", "70/model"],
        ["Variant/condition", "One treatment configuration applied to a base task.", "10/task", "3/task"],
        ["Trajectory/run", "One model executing one task under one variant.", "700/model", "210/model"],
        ["Matched pair", "Same model + same base task (+ same placement where applicable), compared across two conditions.", "Analyzer matched-pair output", "Analyzer matched-pair output"],
        [],
        ["PRIMARY 10-VARIANT MATRIX", "", "", ""],
        ["#", "Condition", "Placement", "Pressure"],
        [1, "clean", "none", "none"],
        [2, "eval_only", "source", "none"],
        [3, "eval_financial", "source", "financial"],
        [4, "eval_self_preservation", "source", "self_preservation"],
        [5, "eval_only", "root", "none"],
        [6, "eval_financial", "root", "financial"],
        [7, "eval_self_preservation", "root", "self_preservation"],
        [8, "eval_only", "scaffold", "none"],
        [9, "eval_financial", "scaffold", "financial"],
        [10, "eval_self_preservation", "scaffold", "self_preservation"],
        [],
        ["RESOURCE 3-VARIANT MATRIX", "", "", ""],
        ["#", "Condition", "Placement", "Pressure"],
        [1, "clean", "none", "none"],
        [2, "eval_only", "scaffold", "none"],
        [3, "eval_resource_deprivation", "scaffold", "resource_deprivation"],
    ]
    for r, row in enumerate(design_rows, 2):
        if not row:
            continue
        for c, value in enumerate(row):
            cell_fmt = fmt["section"] if r in {2, 8, 21} else (fmt["header"] if row and row[0] == "#" else fmt["wrap"])
            write_value(ws, r, c, value, cell_fmt)
    ws.set_column("A:A", 24); ws.set_column("B:B", 78); ws.set_column("C:D", 22)
    ws.freeze_panes(2, 0)

    wb.close()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="full", help="Analyzer mode directory under analysis/ (default: full)")
    parser.add_argument("--profile", default="all", help="Profile to include, or all (default: all)")
    parser.add_argument("--analysis-root", type=Path, default=Path("analysis"), help="Analyzer output root (default: analysis)")
    parser.add_argument("--output", type=Path, help="Output .xlsx path")
    args = parser.parse_args()
    print(build_workbook(args))


if __name__ == "__main__":
    main()
