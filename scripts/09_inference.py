#!/usr/bin/env python3
"""Paper-grade inferential analysis for SWE-EvalPressure.

Consumes canonical analyzer outputs plus the standardized matched-effect table
from scripts/08_results.py. This layer is intentionally separate from the
operational analyzer/results workflow.

Primary analyses:
1. Planned matched contrasts with Holm multiplicity correction.
2. Evaluation-salience GEE: clean vs eval-only/{root,scaffold,source}.
3. Pressure x placement GEE on seeded runs.

Semantic recognition/outcome regressions are intentionally excluded here
because those labels are post-treatment and require human-validation before
paper-grade modeling.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Binomial

INFERENCE_SCHEMA_VERSION = "1.1"

CHANNEL_ORDER = ["root", "scaffold", "source"]
PRESSURE_ORDER = ["eval_only", "eval_financial", "eval_self_preservation"]


def safe_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def git_commit(project_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def discover_profiles(root: Path) -> list[Path]:
    if (root / "trials.json").is_file():
        return [root]
    return sorted(
        p for p in root.iterdir()
        if p.is_dir() and (p / "trials.json").is_file()
    )


def load_trials(input_root: Path, allow_partial: bool = False) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    completeness_errors: list[str] = []
    for p in discover_profiles(input_root):
        rows = safe_json(p / "trials.json")
        if not isinstance(rows, list):
            raise SystemExit(f"Invalid trials.json: {p / 'trials.json'}")
        summary = safe_json(p / "summary.json") or {}
        profile = str(summary.get("profile") or p.name)
        planned = int(summary.get("planned_trajectories") or 0)
        found = int(summary.get("results_found") or len(rows))
        missing = int(summary.get("missing") or max(0, planned - found)) if planned else 0
        if not allow_partial:
            if planned and found != planned:
                completeness_errors.append(f"{profile}: results_found={found}, planned={planned}")
            if missing:
                completeness_errors.append(f"{profile}: missing={missing}")
        df = pd.DataFrame(rows)
        if "substantive_usable" in df:
            df = df[df["substantive_usable"].astype(bool)].copy()
        if "overall_pass" not in df:
            raise SystemExit(f"{profile}: trials.json lacks overall_pass")
        df["pass_binary"] = pd.to_numeric(df["overall_pass"], errors="coerce").fillna(0).gt(0).astype(int)
        df["profile"] = profile
        out[profile] = df
    if not out:
        raise SystemExit(f"No profile trials found under {input_root}")
    if completeness_errors:
        raise SystemExit(
            "Refusing paper-grade inference on incomplete canonical analysis:\n  - "
            + "\n  - ".join(completeness_errors)
            + "\nFinish/reconstruct the full study first. Use --allow-partial only for diagnostic inference."
        )
    return out


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, preserving original order."""
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted_sorted: list[float] = []
    running = 0.0
    for rank, idx in enumerate(order):
        raw = p_values[idx]
        adj = min(1.0, (m - rank) * raw)
        running = max(running, adj)
        adjusted_sorted.append(running)
    out = [1.0] * m
    for idx, adj in zip(order, adjusted_sorted):
        out[idx] = adj
    return out



def primary_omnibus_inference(
    eval_tests: pd.DataFrame,
    pressure_tests: pd.DataFrame,
) -> pd.DataFrame:
    """Build the pre-specified cross-model omnibus family.

    Primary tests per estimable model:
    1. global_eval_salience
    2. global_condition
    3. condition_by_channel_interaction

    Holm correction is applied across all estimable tests in this family.
    Reference-level/simple-effect tests remain secondary and are not included.
    """
    pieces = []
    if not eval_tests.empty:
        x = eval_tests[
            (eval_tests["test"] == "global_eval_salience")
            & (eval_tests["status"] == "ok")
        ].copy()
        if not x.empty:
            x["primary_hypothesis"] = "evaluation_salience"
            pieces.append(x)
    if not pressure_tests.empty:
        for test_name, hypothesis in (
            ("global_condition", "pressure_condition"),
            ("condition_by_channel_interaction", "pressure_by_placement_interaction"),
        ):
            x = pressure_tests[
                (pressure_tests["test"] == test_name)
                & (pressure_tests["status"] == "ok")
            ].copy()
            if not x.empty:
                x["primary_hypothesis"] = hypothesis
                pieces.append(x)

    if not pieces:
        return pd.DataFrame()

    out = pd.concat(pieces, ignore_index=True)
    out["primary_family"] = "capable_models_3_primary_omnibus_tests"
    out["primary_family_size"] = len(out)
    p = pd.to_numeric(out["p_value"], errors="coerce").fillna(1.0).tolist()
    out["holm_p"] = holm_adjust(p)
    out["reject_0_05_unadjusted"] = pd.to_numeric(out["p_value"], errors="coerce") < 0.05
    out["reject_0_05_holm"] = out["holm_p"] < 0.05
    return out


def matched_inference(standardized_results_dir: Path) -> pd.DataFrame:
    path = standardized_results_dir / "matched_pass_effects.csv"
    if not path.is_file():
        raise SystemExit(
            f"Missing standardized matched effects: {path}\n"
            "Run scripts/08_results.py (or ./lab.sh results ...) first."
        )
    df = pd.read_csv(path)
    required = {
        "profile", "contrast", "channel", "n_pairs", "delta_pp",
        "bootstrap_ci_low_pp", "bootstrap_ci_high_pp", "mcnemar_exact_p",
        "treatment_only_pass", "baseline_only_pass",
    }
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"{path} missing columns: {sorted(missing)}")

    frames = []
    for profile, g in df.groupby("profile", sort=False):
        g = g.copy()
        p = pd.to_numeric(g["mcnemar_exact_p"], errors="coerce").fillna(1.0).tolist()
        g["holm_family"] = f"{profile}:planned_9_matched_contrasts"
        g["holm_family_size"] = len(g)
        g["mcnemar_holm_p"] = holm_adjust(p)
        g["mcnemar_reject_0_05"] = g["mcnemar_exact_p"].astype(float) < 0.05
        g["holm_reject_0_05"] = g["mcnemar_holm_p"].astype(float) < 0.05
        g["discordant_pairs"] = (
            pd.to_numeric(g["treatment_only_pass"], errors="coerce").fillna(0)
            + pd.to_numeric(g["baseline_only_pass"], errors="coerce").fillna(0)
        ).astype(int)
        frames.append(g)
    return pd.concat(frames, ignore_index=True) if frames else df


def coef_table(result, profile: str, analysis: str) -> pd.DataFrame:
    ci = result.conf_int()
    rows = []
    for name in result.params.index:
        beta = float(result.params[name])
        se = float(result.bse[name])
        z = beta / se if se > 0 else float("nan")
        p = float(result.pvalues[name])
        lo = float(ci.loc[name, 0])
        hi = float(ci.loc[name, 1])
        rows.append({
            "profile": profile,
            "analysis": analysis,
            "term": name,
            "estimate_log_odds": beta,
            "robust_se": se,
            "z": z,
            "p_value": p,
            "ci_low_log_odds": lo,
            "ci_high_log_odds": hi,
            "odds_ratio": math.exp(beta),
            "or_ci_low": math.exp(lo),
            "or_ci_high": math.exp(hi),
        })
    return pd.DataFrame(rows)


def joint_wald(result, indices: list[int], label: str, profile: str, analysis: str) -> dict[str, Any]:
    k = len(result.params)
    if not indices:
        return {
            "profile": profile, "analysis": analysis, "test": label,
            "df": 0, "chi2": None, "p_value": None,
        }
    R = np.zeros((len(indices), k))
    for row_idx, param_idx in enumerate(indices):
        R[row_idx, param_idx] = 1.0
    try:
        wt = result.wald_test(R, scalar=True)
    except (ValueError, np.linalg.LinAlgError) as exc:
        return {
            "profile": profile,
            "analysis": analysis,
            "test": label,
            "df": len(indices),
            "chi2": None,
            "p_value": None,
            "status": f"not_estimable_wald:{type(exc).__name__}",
        }
    return {
        "profile": profile,
        "analysis": analysis,
        "test": label,
        "df": len(indices),
        "chi2": float(np.asarray(wt.statistic).squeeze()),
        "p_value": float(np.asarray(wt.pvalue).squeeze()),
        "status": "ok",
    }


def param_indices(result, contains_all: list[str] | None = None, contains_any: list[str] | None = None) -> list[int]:
    contains_all = contains_all or []
    contains_any = contains_any or []
    out = []
    for i, name in enumerate(result.params.index):
        if name == "Intercept":
            continue
        if contains_all and not all(token in name for token in contains_all):
            continue
        if contains_any and not any(token in name for token in contains_any):
            continue
        out.append(i)
    return out


def fit_gee(formula: str, df: pd.DataFrame):
    model = smf.gee(
        formula=formula,
        groups="base_task_id",
        data=df,
        family=Binomial(),
        cov_struct=Exchangeable(),
    )
    return model.fit(maxiter=100)


def evaluation_salience_gee(profile: str, df: pd.DataFrame):
    clean = df[df["condition"] == "clean"].copy()
    clean["variant"] = "clean"

    eval_only = df[df["condition"] == "eval_only"].copy()
    eval_only = eval_only[eval_only["channel"].isin(CHANNEL_ORDER)].copy()
    eval_only["variant"] = eval_only["channel"].map(lambda x: f"eval_only_{x}")

    use = pd.concat(
        [clean[["base_task_id", "pass_binary", "variant"]],
         eval_only[["base_task_id", "pass_binary", "variant"]]],
        ignore_index=True,
    )
    levels = ["clean", "eval_only_root", "eval_only_scaffold", "eval_only_source"]
    use["variant"] = pd.Categorical(use["variant"], categories=levels, ordered=True)

    # Capability floor / ceiling: no estimable binary regression.
    unique_outcomes = sorted(use["pass_binary"].dropna().unique().tolist())
    if len(unique_outcomes) < 2:
        return None, pd.DataFrame(), [{
            "profile": profile,
            "analysis": "evaluation_salience_gee",
            "test": "global_eval_salience",
            "df": 0,
            "chi2": None,
            "p_value": None,
            "status": f"not_estimable_outcome_values={unique_outcomes}",
        }], use

    result = fit_gee(
        "pass_binary ~ C(variant, Treatment(reference='clean'))",
        use,
    )
    coefs = coef_table(result, profile, "evaluation_salience_gee")
    indices = [i for i, n in enumerate(result.params.index) if n != "Intercept"]
    test = joint_wald(
        result, indices, "global_eval_salience", profile, "evaluation_salience_gee"
    )
    test.setdefault("status", "ok")
    return result, coefs, [test], use


def pressure_placement_gee(profile: str, df: pd.DataFrame):
    use = df[
        df["condition"].isin(PRESSURE_ORDER)
        & df["channel"].isin(CHANNEL_ORDER)
    ][["base_task_id", "pass_binary", "condition", "channel"]].copy()

    use["condition"] = pd.Categorical(use["condition"], categories=PRESSURE_ORDER, ordered=True)
    use["channel"] = pd.Categorical(use["channel"], categories=CHANNEL_ORDER, ordered=True)

    unique_outcomes = sorted(use["pass_binary"].dropna().unique().tolist())
    if len(unique_outcomes) < 2:
        return None, pd.DataFrame(), [{
            "profile": profile,
            "analysis": "pressure_placement_gee",
            "test": "global_condition",
            "df": 0,
            "chi2": None,
            "p_value": None,
            "status": f"not_estimable_outcome_values={unique_outcomes}",
        }, {
            "profile": profile,
            "analysis": "pressure_placement_gee",
            "test": "global_channel",
            "df": 0,
            "chi2": None,
            "p_value": None,
            "status": f"not_estimable_outcome_values={unique_outcomes}",
        }, {
            "profile": profile,
            "analysis": "pressure_placement_gee",
            "test": "condition_by_channel_interaction",
            "df": 0,
            "chi2": None,
            "p_value": None,
            "status": f"not_estimable_outcome_values={unique_outcomes}",
        }], use

    formula = (
        "pass_binary ~ "
        "C(condition, Treatment(reference='eval_only')) * "
        "C(channel, Treatment(reference='root'))"
    )
    result = fit_gee(formula, use)
    coefs = coef_table(result, profile, "pressure_placement_gee")

    names = list(result.params.index)
    cond_main = [
        i for i, n in enumerate(names)
        if n != "Intercept"
        and "C(condition" in n
        and ":" not in n
    ]
    channel_main = [
        i for i, n in enumerate(names)
        if n != "Intercept"
        and "C(channel" in n
        and ":" not in n
    ]
    interaction = [i for i, n in enumerate(names) if ":" in n]

    # Global "any condition difference across placements" includes condition main
    # effects plus condition×channel interactions.
    global_condition = sorted(set(cond_main + interaction))
    # Global "any placement difference across conditions" includes channel main
    # effects plus interactions.
    global_channel = sorted(set(channel_main + interaction))

    tests = []
    for indices, label in (
        (global_condition, "global_condition"),
        (global_channel, "global_channel"),
        (interaction, "condition_by_channel_interaction"),
        (cond_main, "condition_main_at_root"),
        (channel_main, "channel_main_under_eval_only"),
    ):
        row = joint_wald(result, indices, label, profile, "pressure_placement_gee")
        row.setdefault("status", "ok")
        tests.append(row)
    return result, coefs, tests, use


def sample_inventory(profile: str, analysis: str, use: pd.DataFrame) -> dict[str, Any]:
    return {
        "profile": profile,
        "analysis": analysis,
        "n_rows": int(len(use)),
        "n_base_tasks": int(use["base_task_id"].nunique()) if "base_task_id" in use else None,
        "pass_count": int(use["pass_binary"].sum()) if "pass_binary" in use else None,
        "pass_rate": float(use["pass_binary"].mean()) if len(use) and "pass_binary" in use else None,
    }


def markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
    if df.empty:
        return "_No rows._"
    x = df[cols].copy()
    for col in x.columns:
        if pd.api.types.is_float_dtype(x[col]):
            x[col] = x[col].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = [
        "| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
        for row in x.itertuples(index=False, name=None)
    ]
    return "\n".join([header, sep] + body)


def build_report(
    matched: pd.DataFrame,
    primary_omnibus: pd.DataFrame,
    eval_tests: pd.DataFrame,
    pressure_tests: pd.DataFrame,
    samples: pd.DataFrame,
) -> str:
    significant_unadjusted = int((matched["mcnemar_exact_p"] < 0.05).sum()) if not matched.empty else 0
    significant_holm = int((matched["mcnemar_holm_p"] < 0.05).sum()) if not matched.empty else 0

    return "\n".join([
        "# SWE-EvalPressure inference report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "This is the paper-specific inferential layer. It consumes canonical analyzer outputs and standardized results; it does not reconstruct Harbor runs or call an LLM.",
        "",
        "## Planned matched contrasts",
        "",
        f"- Unadjusted McNemar p < .05: **{significant_unadjusted}**",
        f"- Holm-adjusted p < .05 within each model's planned contrast family: **{significant_holm}**",
        "",
        markdown_table(
            matched,
            ["profile", "contrast", "channel", "n_pairs", "delta_pp",
             "bootstrap_ci_low_pp", "bootstrap_ci_high_pp",
             "mcnemar_exact_p", "mcnemar_holm_p"],
        ),
        "",
        "## Primary omnibus family",
        "",
        "The pre-specified primary family contains, for each estimable model: global evaluation salience, global pressure-condition effect, and pressure×placement interaction. Holm correction is applied across the full capable-model family.",
        "",
        markdown_table(
            primary_omnibus,
            ["profile", "primary_hypothesis", "test", "df", "chi2", "p_value", "holm_p", "reject_0_05_holm"],
        ),
        "",
        "## Evaluation-salience GEE",
        "",
        "Binary pass outcome; repeated observations clustered by base task using GEE with an exchangeable working correlation.",
        "",
        markdown_table(eval_tests, ["profile", "test", "df", "chi2", "p_value", "status"]),
        "",
        "## Pressure × placement GEE",
        "",
        "Seeded runs only. Formula: `pass ~ condition * channel`; base-task-clustered GEE.",
        "",
        markdown_table(pressure_tests, ["profile", "test", "df", "chi2", "p_value", "status"]),
        "",
        "## Analysis samples",
        "",
        markdown_table(samples, ["profile", "analysis", "n_rows", "n_base_tasks", "pass_count", "pass_rate"]),
        "",
        "## Interpretation constraints",
        "",
        "- Matched McNemar tests are Holm-corrected within each model's pre-specified 9-contrast family.",
        "- The three primary omnibus hypotheses per estimable model (evaluation salience, pressure condition, pressure×placement interaction) are additionally Holm-corrected across the full capable-model family.",
        "- Reference-level tests such as `condition_main_at_root`, `channel_main_under_eval_only`, and `global_channel` are secondary/exploratory and are not treated as headline primary evidence.",
        "- Bootstrap intervals resample matched base-task pairs and are not themselves multiplicity-adjusted.",
        "- GEE coefficients use robust sandwich covariance with base-task clustering and an exchangeable working correlation.",
        "- `global_condition` in the pressure model jointly tests condition main effects and condition×channel interactions; it is therefore an omnibus test for any pressure-condition difference across placements.",
        "- `global_channel` analogously tests any placement difference across conditions.",
        "- Models with a constant binary outcome are reported as not estimable rather than forced through a logistic model.",
        "- Semantic recognition/outcome regressions are intentionally excluded pending human validation because semantic labels are post-treatment measurements.",
        "- Cue-mechanism comparisons remain descriptive because mechanism is assigned at the base-task level.",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paper-grade inferential analysis for SWE-EvalPressure."
    )
    parser.add_argument(
        "--input-root", type=Path, required=True,
        help="Canonical profile directory or root containing profile subdirectories with trials.json.",
    )
    parser.add_argument(
        "--standardized-results-dir", type=Path, required=True,
        help="Directory produced by scripts/08_results.py.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="Permit diagnostic inference on incomplete canonical analysis. Paper inference is complete-only by default.",
    )
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    standardized_results_dir = args.standardized_results_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_trials(input_root, allow_partial=args.allow_partial)

    matched = matched_inference(standardized_results_dir)
    matched.to_csv(output_dir / "matched_inference.csv", index=False)

    eval_coefs = []
    eval_tests = []
    pressure_coefs = []
    pressure_tests = []
    samples = []

    for profile, df in data.items():
        _, c, t, use = evaluation_salience_gee(profile, df)
        if not c.empty:
            eval_coefs.append(c)
        eval_tests.extend(t)
        samples.append(sample_inventory(profile, "evaluation_salience_gee", use))

        _, c, t, use = pressure_placement_gee(profile, df)
        if not c.empty:
            pressure_coefs.append(c)
        pressure_tests.extend(t)
        samples.append(sample_inventory(profile, "pressure_placement_gee", use))

    eval_coef_df = pd.concat(eval_coefs, ignore_index=True) if eval_coefs else pd.DataFrame()
    eval_test_df = pd.DataFrame(eval_tests)
    pressure_coef_df = pd.concat(pressure_coefs, ignore_index=True) if pressure_coefs else pd.DataFrame()
    pressure_test_df = pd.DataFrame(pressure_tests)
    sample_df = pd.DataFrame(samples)
    primary_omnibus_df = primary_omnibus_inference(eval_test_df, pressure_test_df)

    eval_coef_df.to_csv(output_dir / "eval_salience_gee_coefficients.csv", index=False)
    eval_test_df.to_csv(output_dir / "eval_salience_gee_tests.csv", index=False)
    pressure_coef_df.to_csv(output_dir / "pressure_placement_gee_coefficients.csv", index=False)
    pressure_test_df.to_csv(output_dir / "pressure_placement_gee_tests.csv", index=False)
    primary_omnibus_df.to_csv(output_dir / "primary_omnibus_inference.csv", index=False)
    sample_df.to_csv(output_dir / "analysis_samples.csv", index=False)

    project_root = Path(__file__).resolve().parents[1]
    manifest = {
        "inference_schema_version": INFERENCE_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": str(Path(__file__).resolve()),
        "git_commit": git_commit(project_root),
        "input_root": str(input_root),
        "standardized_results_dir": str(standardized_results_dir),
        "output_dir": str(output_dir),
        "profiles": list(data),
        "primary_analyses": [
            "planned_matched_contrasts_holm_within_profile",
            "evaluation_salience_base_task_clustered_gee",
            "pressure_by_placement_base_task_clustered_gee",
            "holm_correction_across_capable_model_primary_omnibus_family",
        ],
        "excluded_from_primary_inference": [
            "semantic_recognition_to_outcome_models_pending_human_validation",
            "cue_mechanism_causal_comparisons_due_to_base_task_assignment",
        ],
        "software": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": stats.__version__ if hasattr(stats, "__version__") else None,
            "statsmodels": sm.__version__,
        },
    }
    # scipy.stats has no __version__; fill from scipy package.
    try:
        import scipy
        manifest["software"]["scipy"] = scipy.__version__
    except Exception:
        pass
    write_json(output_dir / "inference_manifest.json", manifest)

    (output_dir / "report.md").write_text(
        build_report(matched, primary_omnibus_df, eval_test_df, pressure_test_df, sample_df),
        encoding="utf-8",
    )
    print(output_dir / "report.md")


if __name__ == "__main__":
    main()
