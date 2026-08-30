from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_topline_visual_report_compiles() -> None:
    subprocess.run(
        [sys.executable, "-m", "py_compile", str(ROOT / "scripts" / "37_visual_report.py")],
        check=True,
    )


def test_topline_is_story_led_not_pvalue_ranked() -> None:
    text = (ROOT / "scripts" / "37_visual_report.py").read_text(encoding="utf-8")

    assert "Results at a glance" in text
    assert "Evaluation context changes <em>how agents work</em>" in text
    assert "Source-local evaluation context is recognized and resisted." in text
    assert "The resource follow-up produces large, model-specific contraction." in text
    assert "No capable-model primary success effect survives Holm" in text
    assert "Mechanism-open signal" in text


def test_dense_evidence_plot_is_moved_out_of_first_impression() -> None:
    text = (ROOT / "scripts" / "37_visual_report.py").read_text(encoding="utf-8")

    assert "Summary of supported claims" in text
    assert 'figure_html(*generated["evidence"])' in text
    assert 'for k in ("success", "tests", "behavior", "cue_removal", "root_breadth")' in text
