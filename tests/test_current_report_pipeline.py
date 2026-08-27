from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_current_report_scripts_compile() -> None:
    for rel in (
        "scripts/36_iclr_team_preread_v2.py",
        "scripts/37_visual_report.py",
    ):
        subprocess.run(
            [sys.executable, "-m", "py_compile", str(ROOT / rel)],
            check=True,
        )


def test_current_pipeline_help() -> None:
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/current_analysis_pipeline.sh"), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "./lab.sh analyze current" in result.stdout
    assert "--report-only" in result.stdout


def test_lab_routes_current_analysis() -> None:
    text = (ROOT / "lab.sh").read_text(encoding="utf-8")

    assert "current_analysis_pipeline.sh" in text
    assert "analyze current" in text


def test_visual_report_is_presentation_only() -> None:
    text = (ROOT / "scripts/37_visual_report.py").read_text(encoding="utf-8")

    forbidden = (
        "requests.",
        "httpx.",
        "litellm.",
        "openai.",
        "anthropic.",
    )

    for token in forbidden:
        assert token not in text
