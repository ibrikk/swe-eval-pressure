from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "normalize_primary_multijudge",
    ROOT
    / "scripts"
    / "24_normalize_primary_multijudge.py",
)

assert spec is not None
assert spec.loader is not None

mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


FIELDS = {
    "evaluation_recognition": {
        "labels": [
            "observed",
            "not_observed",
            "ambiguous",
        ],
        "evidence_required_for": [
            "observed",
        ],
    }
}


class NormalizePrimaryMultijudgeTests(
    unittest.TestCase
):

    def test_valid_job_requires_both_statuses(self):
        good = {
            "status": "ok",
            "final_cache_entry": {
                "status": "ok",
                "judgment": {},
            },
        }

        self.assertTrue(
            mod.valid_job(good)
        )

        bad = {
            "status": "ok",
            "final_cache_entry": {
                "status": "missing",
                "judgment": {},
            },
        }

        self.assertFalse(
            mod.valid_job(bad)
        )

    def test_missing_is_not_valid(self):
        obj = {
            "status": "missing",
        }

        self.assertFalse(
            mod.valid_job(obj)
        )

    def test_valid_observed_requires_evidence(self):
        obj = {
            "final_cache_entry": {
                "judgment": {
                    "evaluation_recognition": {
                        "label": "observed",
                        "evidence": [],
                    }
                }
            }
        }

        with self.assertRaises(
            ValueError
        ):
            mod.normalize_valid_judgment(
                obj,
                FIELDS,
            )

    def test_valid_negative_may_have_no_evidence(self):
        obj = {
            "final_cache_entry": {
                "judgment": {
                    "evaluation_recognition": {
                        "label": (
                            "not_observed"
                        ),
                        "evidence": [],
                    }
                }
            }
        }

        out = (
            mod.normalize_valid_judgment(
                obj,
                FIELDS,
            )
        )

        self.assertEqual(
            out[
                "evaluation_recognition"
            ]["label"],
            "not_observed",
        )

    def test_invalid_label_is_rejected(self):
        obj = {
            "final_cache_entry": {
                "judgment": {
                    "evaluation_recognition": {
                        "label": "yes",
                        "evidence": [],
                    }
                }
            }
        }

        with self.assertRaises(
            ValueError
        ):
            mod.normalize_valid_judgment(
                obj,
                FIELDS,
            )


if __name__ == "__main__":
    unittest.main()
