from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

spec = importlib.util.spec_from_file_location(
    "swe_eval_pressure_analyzer",
    SCRIPTS / "07_analyze.py",
)
assert spec is not None and spec.loader is not None
analyzer = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = analyzer
spec.loader.exec_module(analyzer)


class LlamaAnalysisRegressionTests(unittest.TestCase):
    def test_text_based_shell_command_is_recovered(self) -> None:
        step = {
            "step_id": "3",
            "source": "agent",
            "message": (
                "```mswea_bash_command\n"
                "cd /workspace && pytest -q\n"
                "```"
            ),
            "observation": {"returncode": 0, "output": "ok"},
        }

        calls = analyzer.text_based_shell_calls(step)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["category"], "bash")
        self.assertEqual(calls[0]["command"], "cd /workspace && pytest -q")
        self.assertEqual(calls[0]["failed"], 0)

    def test_no_recorded_llama_agent_step_is_protocol_error(self) -> None:
        trajectory = {
            "steps": [
                {"source": "system", "message": "system"},
                {"source": "user", "message": "task"},
            ]
        }
        result = {
            "agent_execution": {"started_at": "2026-08-16T00:00:00Z"},
            "agent_result": {"status": "finished"},
        }

        status, exc_type, message = analyzer.classify_agent_protocol_status(
            "llama",
            "completed",
            "",
            "",
            Path("/tmp/trajectory.json"),
            result,
            trajectory,
        )

        self.assertEqual(status, "agent_protocol_error")
        self.assertEqual(exc_type, "NoRecordedAgentStep")
        self.assertIn("no agent/model steps", message)
        self.assertFalse(analyzer.substantive_status(status))

    def test_real_llama_agent_step_remains_completed(self) -> None:
        trajectory = {
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
        }
        result = {
            "agent_execution": {"started_at": "2026-08-19T00:00:00Z"},
            "agent_result": {"status": "finished"},
        }

        status, exc_type, message = analyzer.classify_agent_protocol_status(
            "llama",
            "completed",
            "",
            "",
            Path("/tmp/trajectory.json"),
            result,
            trajectory,
        )

        self.assertEqual((status, exc_type, message), ("completed", "", ""))
        self.assertTrue(analyzer.substantive_status(status))

    def test_image_build_error_remains_environment_error(self) -> None:
        result = {
            "exception_info": {
                "exception_type": "ImageBuildError",
                "exception_message": "Image build for im-test failed.",
            }
        }

        status, exc_type, _ = analyzer.terminal_status(result)

        self.assertEqual(status, "environment_error")
        self.assertEqual(exc_type, "ImageBuildError")
        self.assertFalse(analyzer.substantive_status(status))


if __name__ == "__main__":
    unittest.main()
