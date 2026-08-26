from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

spec = importlib.util.spec_from_file_location(
    "swe_eval_pressure_analyzer_hardening",
    SCRIPTS / "07_analyze.py",
)
assert spec is not None and spec.loader is not None
analyzer = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = analyzer
spec.loader.exec_module(analyzer)


class AnalyzerReconstructionHardeningTests(unittest.TestCase):

    def test_newer_zero_step_llama_cannot_replace_valid_candidate(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)

            valid_dir = root / "valid"
            invalid_dir = root / "invalid"

            (valid_dir / "agent").mkdir(parents=True)
            (invalid_dir / "agent").mkdir(parents=True)

            (valid_dir / "agent" / "trajectory.json").write_text(
                json.dumps({
                    "steps": [
                        {"source": "system", "message": "system"},
                        {"source": "user", "message": "task"},
                        {
                            "source": "agent",
                            "message": (
                                "```mswea_bash_command\n"
                                "pwd\n"
                                "```"
                            ),
                        },
                    ]
                })
            )

            (invalid_dir / "agent" / "trajectory.json").write_text(
                json.dumps({
                    "steps": [
                        {"source": "system", "message": "system"},
                        {"source": "user", "message": "task"},
                    ]
                })
            )

            now = datetime.now(timezone.utc)

            def make_candidate(name, trial_dir, finished):
                return {
                    "path": trial_dir / "result.json",
                    "trial_dir": trial_dir,
                    "run_root": root,
                    "run_signature": "same-study",
                    "status": "completed",
                    "exception_type": "",
                    "finished_at": finished,
                    "result": {
                        "trial_name": name,
                        "agent_execution": {
                            "started_at": "2026-08-26T00:00:00Z"
                        },
                        "agent_result": {
                            "status": "finished"
                        },
                        "verifier_result": {
                            "rewards": {}
                        },
                    },
                }

            candidates = {
                "task": [
                    make_candidate(
                        "valid-older",
                        valid_dir,
                        now,
                    ),
                    make_candidate(
                        "invalid-newer",
                        invalid_dir,
                        now + timedelta(seconds=10),
                    ),
                ]
            }

            analyzer.apply_candidate_protocol_classification(
                "llama",
                candidates,
            )

            self.assertEqual(
                candidates["task"][0]["status"],
                "completed",
            )
            self.assertEqual(
                candidates["task"][1]["status"],
                "agent_protocol_error",
            )
            self.assertEqual(
                candidates["task"][1]["exception_type"],
                "NoRecordedAgentStep",
            )

            selected, _, _ = analyzer.select_candidates(
                candidates,
                repeats=1,
            )

            self.assertEqual(
                selected[0]["result"]["trial_name"],
                "valid-older",
            )

    def test_completed_llama_missing_trajectory_is_protocol_error(self):
        result = {
            "agent_execution": {
                "started_at": "2026-08-26T00:00:00Z"
            },
            "agent_result": {
                "status": "finished"
            },
        }

        status, exc_type, _ = (
            analyzer.classify_agent_protocol_status(
                "llama",
                "completed",
                "",
                "",
                None,
                result,
                {},
            )
        )

        self.assertEqual(
            status,
            "agent_protocol_error",
        )
        self.assertEqual(
            exc_type,
            "MissingTrajectory",
        )
        self.assertFalse(
            analyzer.substantive_status(status)
        )

    def test_install_only_run_is_not_discovered(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            results = root / "results" / "resource"
            run = results / "install-only"
            run.mkdir(parents=True)

            (run / "run_metadata.json").write_text(
                json.dumps({
                    "mode": "resource",
                    "profile": "llama",
                    "model": "model",
                    "agent": "agent",
                    "install_only": True,
                    "created_at": (
                        "2026-08-26T00:00:00+00:00"
                    ),
                })
            )

            (run / "study_manifest.json").write_text(
                json.dumps({
                    "mode": "resource",
                    "profile": "llama",
                    "tasks": [
                        {
                            "directory": "task-a",
                            "base_task_id": "a",
                        }
                    ],
                })
            )

            selected, excluded, warnings = (
                analyzer.discover_runs(
                    root,
                    results,
                    "resource",
                    "llama",
                    None,
                )
            )

            self.assertEqual(selected, [])
            self.assertEqual(excluded, [])
            self.assertTrue(
                any(
                    "install-only" in warning
                    for warning in warnings
                )
            )


if __name__ == "__main__":
    unittest.main()
