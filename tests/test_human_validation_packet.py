from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "human_validation_packet",
    ROOT
    / "scripts"
    / "22_build_human_validation_packet.py",
)

assert spec is not None
assert spec.loader is not None

mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


class HumanValidationPacketTests(
    unittest.TestCase
):

    def test_schema_is_current_five_field_schema(
        self,
    ):
        schema = mod.load_schema(
            ROOT
            / "config"
            / "semantic_judge_schema.json"
        )

        self.assertEqual(
            list(schema["fields"]),
            [
                "evaluation_recognition",
                "pressure_recognition",
                "cue_provenance_stance",
                "explicit_resistance",
                "claimed_behavioral_response",
            ],
        )

        self.assertEqual(
            schema["agreement"][
                "human_validation_raters"
            ],
            3,
        )

        self.assertEqual(
            schema["agreement"][
                "target_human_validation_sample"
            ],
            200,
        )

    def test_blinding_rejects_admin_metadata(
        self,
    ):
        text = "example semantic trajectory"

        row = {
            "sample_id": "SV001",
            "semantic_trajectory": text,
            "semantic_view_sha256": (
                hashlib.sha256(
                    text.encode()
                ).hexdigest()
            ),
            "condition": "eval_only",
        }

        with self.assertRaises(ValueError):
            mod.validate_inputs(
                [row],
                expected_items=1,
            )

    def test_hash_mismatch_is_rejected(self):
        row = {
            "sample_id": "SV001",
            "semantic_trajectory": "abc",
            "semantic_view_sha256": (
                "definitely-wrong"
            ),
        }

        with self.assertRaises(ValueError):
            mod.validate_inputs(
                [row],
                expected_items=1,
            )

    def test_duplicate_ids_are_rejected(self):
        text = "abc"
        digest = hashlib.sha256(
            text.encode()
        ).hexdigest()

        rows = [
            {
                "sample_id": "SV001",
                "semantic_trajectory": text,
                "semantic_view_sha256": digest,
            },
            {
                "sample_id": "SV001",
                "semantic_trajectory": text,
                "semantic_view_sha256": digest,
            },
        ]

        with self.assertRaises(ValueError):
            mod.validate_inputs(
                rows,
                expected_items=2,
            )

    def test_generated_html_contains_no_admin_fields(
        self,
    ):
        schema = mod.load_schema(
            ROOT
            / "config"
            / "semantic_judge_schema.json"
        )

        rows = [
            {
                "sample_id": "SV001",
                "semantic_trajectory": (
                    "[AGENT]\nhello"
                ),
            }
        ]

        page = mod.render_html(
            rows,
            schema["fields"],
        )

        self.assertIn(
            "SV001",
            page,
        )
        self.assertIn(
            "evaluation_recognition",
            page,
        )

        for forbidden in (
            "overall_pass",
            "deepseek",
            "gemini",
            "trial_name",
        ):
            self.assertNotIn(
                forbidden,
                page,
            )




if __name__ == "__main__":
    unittest.main()
