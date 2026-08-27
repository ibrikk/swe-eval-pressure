from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "integrated_analysis",
    ROOT
    / "scripts"
    / "13_integrated_analysis.py",
)

assert spec is not None
assert spec.loader is not None

mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


class IntegratedAnalysisTests(unittest.TestCase):

    def test_wilson_interval_half(self):
        low, high = mod.wilson_interval(
            5,
            10,
        )

        self.assertIsNotNone(low)
        self.assertIsNotNone(high)

        assert low is not None
        assert high is not None

        self.assertAlmostEqual(
            low,
            0.2366,
            places=3,
        )
        self.assertAlmostEqual(
            high,
            0.7634,
            places=3,
        )

    def test_semantic_coverage_uses_only_substantive(self):
        rows = [
            {
                "profile": "claude",
                "condition": "eval_only",
                "channel": "root",
                "substantive_usable": "1",
                "semantic_judge_status": "ok",
            },
            {
                "profile": "claude",
                "condition": "eval_only",
                "channel": "root",
                "substantive_usable": "1",
                "semantic_judge_status": "error",
            },
            {
                "profile": "claude",
                "condition": "eval_only",
                "channel": "root",
                "substantive_usable": "0",
                "semantic_judge_status": "ok",
            },
        ]

        out = mod.semantic_coverage_rows(
            rows
        )

        self.assertEqual(len(out), 1)
        self.assertEqual(
            out[0]["substantive_n"],
            2,
        )
        self.assertEqual(
            out[0]["semantic_ok_n"],
            1,
        )
        self.assertEqual(
            out[0]["semantic_coverage_pct"],
            50.0,
        )

    def test_correctness_effect_is_matched(self):
        pairs = [
            {
                "profile": "claude",
                "pair_type": "eval_effect",
                "channel": "root",
                "pair_usable": "1",
                "baseline_overall_pass": "0",
                "treatment_overall_pass": "1",
            },
            {
                "profile": "claude",
                "pair_type": "eval_effect",
                "channel": "root",
                "pair_usable": "1",
                "baseline_overall_pass": "1",
                "treatment_overall_pass": "0",
            },
            {
                "profile": "claude",
                "pair_type": "eval_effect",
                "channel": "root",
                "pair_usable": "1",
                "baseline_overall_pass": "1",
                "treatment_overall_pass": "1",
            },
            {
                "profile": "claude",
                "pair_type": "eval_effect",
                "channel": "root",
                "pair_usable": "0",
                "baseline_overall_pass": "",
                "treatment_overall_pass": "",
            },
        ]

        out = mod.correctness_effect_rows(
            pairs,
            mode="full",
            bootstrap_replicates=100,
            bootstrap_seed=1,
        )

        mod.apply_correctness_holm(out)

        self.assertEqual(len(out), 1)
        row = out[0]

        self.assertEqual(
            row["planned_pairs"],
            4,
        )
        self.assertEqual(
            row["n_pairs"],
            3,
        )
        self.assertAlmostEqual(
            row["risk_difference_pp"],
            0.0,
        )
        self.assertEqual(
            row["discordant_pairs"],
            2,
        )
        self.assertEqual(
            row["mcnemar_exact_p"],
            1.0,
        )
        self.assertEqual(
            row["holm_adjusted_p"],
            1.0,
        )

    def test_said_did_uses_treatment_semantics(self):
        trials = [
            {
                "trial_name": "baseline",
                "profile": "claude",
                "semantic_judge_status": "ok",
                "evaluation_cue_stance": "accepted",
            },
            {
                "trial_name": "treatment",
                "profile": "claude",
                "semantic_judge_status": "ok",
                "evaluation_cue_stance": (
                    "rejected_as_prompt_injection"
                ),
                "evaluation_cue_recognition": "explicit",
            },
        ]

        pair = {
            "profile": "claude",
            "base_task_id": "task1",
            "pair_type": "financial_effect",
            "channel": "root",
            "pair_usable": "1",
            "baseline_trial": "baseline",
            "treatment_trial": "treatment",
            "baseline_condition": "eval_only",
            "treatment_condition": "eval_financial",
            "baseline_prompt_tokens": "100",
            "treatment_prompt_tokens": "120",
            "delta_prompt_tokens": "20",
            "baseline_overall_pass": "1",
            "treatment_overall_pass": "0",
            "delta_overall_pass": "-1",
        }

        out = mod.said_did_pair_rows(
            trials,
            [pair],
        )

        self.assertEqual(len(out), 1)

        self.assertEqual(
            out[0][
                "evaluation_cue_stance"
            ],
            "rejected_as_prompt_injection",
        )

        self.assertEqual(
            out[0]["correctness_transition"],
            "pass→fail",
        )

        summary = (
            mod.said_did_summary_rows(
                out
            )
        )

        match = [
            row
            for row in summary
            if row["semantic_field"]
            == "evaluation_cue_stance"
            and row["semantic_label"]
            == "rejected_as_prompt_injection"
            and row["metric"]
            == "prompt_tokens"
        ]

        self.assertEqual(len(match), 1)
        self.assertEqual(
            match[0]["mean_delta"],
            20.0,
        )
        self.assertEqual(
            match[0][
                "causal_interpretation_allowed"
            ],
            0,
        )

    def test_resource_is_not_classed_as_pressure(self):
        rows = mod.effect_catalog_rows(
            [
                {
                    "profile": "codex",
                    "analysis_mode": "resource",
                    "pair_type": "resource_effect",
                    "channel": "scaffold",
                    "endpoint": "validation_any",
                    "n_pairs": "70",
                    "risk_difference_pp": "-5",
                    "bootstrap_ci_low_pp": "-10",
                    "bootstrap_ci_high_pp": "0",
                    "mcnemar_exact_p": "0.2",
                    "holm_adjusted_p": "0.2",
                    "family_name": "resource",
                    "adjusted_reject": "0",
                }
            ],
            [],
            [],
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(
            rows[0]["intervention"],
            "resource_deprivation",
        )
        self.assertEqual(
            rows[0]["mechanism_class"],
            "resource_constraint",
        )


    def test_read_csv_accepts_large_fields(self):
        import csv
        import tempfile

        large_value = "x" * 200_000

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "large.csv"

            with path.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "sample_id",
                        "evidence",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "sample_id": "SV001",
                        "evidence": large_value,
                    }
                )

            rows = mod.read_csv(path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(
            rows[0]["sample_id"],
            "SV001",
        )
        self.assertEqual(
            len(rows[0]["evidence"]),
            200_000,
        )


    def test_resource_success_holm_crosses_capable_models(
        self,
    ):
        rows = [
            {
                "profile": "claude",
                "analysis_mode": "resource",
                "pair_type": "resource_effect",
                "mcnemar_exact_p": 1.0,
            },
            {
                "profile": "fable",
                "analysis_mode": "resource",
                "pair_type": "resource_effect",
                "mcnemar_exact_p": (
                    0.60723876953125
                ),
            },
            {
                "profile": "codex",
                "analysis_mode": "resource",
                "pair_type": "resource_effect",
                "mcnemar_exact_p": (
                    0.004343509674072266
                ),
            },
            {
                "profile": "llama",
                "analysis_mode": "resource",
                "pair_type": "resource_effect",
                "mcnemar_exact_p": 1.0,
            },
        ]

        mod.apply_correctness_holm(rows)

        codex = next(
            row
            for row in rows
            if row["profile"] == "codex"
        )

        self.assertAlmostEqual(
            codex["holm_adjusted_p"],
            0.013030529022216797,
        )

        self.assertEqual(
            codex["family_size"],
            3,
        )

        self.assertEqual(
            codex["family_scope"],
            "across_3_capable_models",
        )

        llama = next(
            row
            for row in rows
            if row["profile"] == "llama"
        )

        self.assertEqual(
            llama["holm_adjusted_p"],
            "",
        )

        self.assertEqual(
            llama["analysis_tier"],
            "descriptive_capability_floor",
        )


    def test_resource_canonical_inference_overrides_ci(
        self,
    ):
        import csv
        import tempfile

        rows = [
            {
                "profile": "codex",
                "analysis_mode": "resource",
                "pair_type": "resource_effect",
                "risk_difference_pp": -20.0,
                "mcnemar_exact_p": (
                    0.004343509674072266
                ),
            },
            {
                "profile": "codex",
                "analysis_mode": "resource",
                "pair_type": "eval_effect",
                "risk_difference_pp": 0.0,
                "mcnemar_exact_p": 1.0,
            },
        ]

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)

            files = {
                "primary_success.csv": {
                    "profile": "codex",
                    "n_pairs": "70",
                    "baseline_pass_pct": "75.7",
                    "treatment_pass_pct": "55.7",
                    "delta_pp": "-20.0",
                    "bootstrap_ci_low_pp": "-31.428571",
                    "bootstrap_ci_high_pp": "-8.571429",
                    "fail_to_pass": "4",
                    "pass_to_fail": "18",
                    "discordant_pairs": "22",
                    "mcnemar_exact_p": (
                        "0.004343509674072266"
                    ),
                    "mcnemar_holm_p": (
                        "0.013030529022216797"
                    ),
                    "inferential_role": "inferential",
                },
                "secondary_eval_success.csv": {
                    "profile": "codex",
                    "n_pairs": "70",
                    "baseline_pass_pct": "75",
                    "treatment_pass_pct": "75",
                    "delta_pp": "0.0",
                    "bootstrap_ci_low_pp": "-5",
                    "bootstrap_ci_high_pp": "5",
                    "fail_to_pass": "2",
                    "pass_to_fail": "2",
                    "discordant_pairs": "4",
                    "mcnemar_exact_p": "1.0",
                    "mcnemar_holm_p": "1.0",
                    "inferential_role": "inferential",
                },
            }

            for name, row in files.items():
                path = root / name

                with path.open(
                    "w",
                    newline="",
                    encoding="utf-8",
                ) as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=list(row),
                    )
                    writer.writeheader()
                    writer.writerow(row)

            loaded = (
                mod.apply_resource_canonical_inference(
                    rows,
                    root,
                )
            )

        self.assertEqual(loaded, 2)

        resource = next(
            row
            for row in rows
            if row["pair_type"]
            == "resource_effect"
        )

        self.assertEqual(
            resource["bootstrap_ci_low_pp"],
            "-31.428571",
        )
        self.assertEqual(
            resource["holm_adjusted_p"],
            "0.013030529022216797",
        )
        self.assertEqual(
            resource["ci_source"],
            "scripts/resource_inference.py",
        )


if __name__ == "__main__":
    unittest.main()
