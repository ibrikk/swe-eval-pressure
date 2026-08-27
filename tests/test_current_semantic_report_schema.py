from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_iclr_report_scripts_compile() -> None:
    for rel in (
        "scripts/36_iclr_team_preread_v2.py",
        "scripts/37_visual_report.py",
    ):
        subprocess.run(
            [sys.executable, "-m", "py_compile", str(ROOT / rel)],
            check=True,
        )


def test_semantic_treatment_effects_match_current_consensus_schema() -> None:
    text = (ROOT / "scripts" / "36_iclr_team_preread_v2.py").read_text(
        encoding="utf-8"
    )

    assert "trial_name" in text
    assert "__status" in text
    assert "__label" in text
    assert "__deepseek" in text
    assert "__gemini" in text
    assert "strict_consensus" in text
    assert "produced 0 strict-consensus effects" in text


def test_visual_report_consumes_strict_consensus_semantic_effects() -> None:
    text = (ROOT / "scripts" / "37_visual_report.py").read_text(
        encoding="utf-8"
    )

    assert 'r.get("semantic_source")' in text
    assert "semantic_treatment_effects_forest.svg" in text


def test_current_html_generator_strips_trailing_whitespace() -> None:
    text = (ROOT / "scripts" / "35_current_html.py").read_text(
        encoding="utf-8"
    )

    assert "def clean_generated_text" in text
    assert "clean_generated_text(" in text
