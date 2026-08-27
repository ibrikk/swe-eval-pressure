from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "semantic_reliability",
    ROOT / "scripts" / "23_semantic_reliability.py",
)

assert spec is not None
assert spec.loader is not None

mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


class SemanticReliabilityTests(unittest.TestCase):

    def test_perfect_cohen_kappa(self):
        pairs = [
            ("a", "a"),
            ("b", "b"),
            ("a", "a"),
            ("b", "b"),
        ]

        self.assertAlmostEqual(
            mod.cohen_kappa(
                pairs,
                ["a", "b"],
            ),
            1.0,
        )

    def test_raw_agreement(self):
        pairs = [
            ("a", "a"),
            ("a", "b"),
            ("b", "b"),
            ("b", "a"),
        ]

        self.assertAlmostEqual(
            mod.raw_agreement(pairs),
            0.5,
        )

    def test_perfect_fleiss_kappa(self):
        matrix = [
            [3, 0],
            [0, 3],
            [3, 0],
            [0, 3],
        ]

        self.assertAlmostEqual(
            mod.fleiss_kappa(matrix),
            1.0,
        )

    def test_perfect_multirater_ac1(self):
        ratings = [
            ["a", "a", "a"],
            ["b", "b", "b"],
            ["a", "a", "a"],
            ["b", "b", "b"],
        ]

        self.assertAlmostEqual(
            mod.gwet_ac1_multi(
                ratings,
                ["a", "b"],
            ),
            1.0,
        )

    def test_majority_of_three(self):
        self.assertEqual(
            mod.majority_of_three(
                ["a", "a", "b"]
            ),
            "a",
        )

        self.assertIsNone(
            mod.majority_of_three(
                ["a", "b", "c"]
            )
        )

        self.assertIsNone(
            mod.majority_of_three(
                ["a", "a", None]
            )
        )

    def test_llm_consensus(self):
        self.assertEqual(
            mod.two_rater_consensus(
                "a",
                "a",
            ),
            "a",
        )

        self.assertIsNone(
            mod.two_rater_consensus(
                "a",
                "b",
            )
        )

        self.assertIsNone(
            mod.two_rater_consensus(
                "a",
                None,
            )
        )

    def test_missing_pairs_are_excluded(self):
        a = {
            "x": "yes",
            "y": None,
        }
        b = {
            "x": "yes",
            "y": "no",
        }

        self.assertEqual(
            mod.observed_pairs(
                a,
                b,
                ["x", "y"],
            ),
            [("yes", "yes")],
        )

    def test_class_metrics_perfect(self):
        pairs = [
            ("a", "a"),
            ("a", "a"),
            ("b", "b"),
        ]

        rows = mod.class_metric_rows(
            pairs,
            ["a", "b"],
            field="f",
            comparison="c",
        )

        for row in rows:
            self.assertEqual(
                row["precision"],
                1.0,
            )
            self.assertEqual(
                row["recall"],
                1.0,
            )
            self.assertEqual(
                row["f1"],
                1.0,
            )


if __name__ == "__main__":
    unittest.main()
