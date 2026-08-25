import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "scripts"),
)

import semantic_reliability_report as report


def test_percent():
    assert (
        report.percent(0.995)
        == "99.50%"
    )

    assert (
        report.percent(None)
        == "NA"
    )


def test_outcome_blind():
    source = (
        ROOT
        / "scripts"
        / "semantic_reliability_report.py"
    ).read_text()

    forbidden = [
        '["label"]',
        '.get("label"',
        '["evidence"]',
        '.get("evidence"',
    ]

    for token in forbidden:
        assert token not in source


def test_no_network():
    source = (
        ROOT
        / "scripts"
        / "semantic_reliability_report.py"
    ).read_text()

    forbidden = [
        "urlopen(",
        "invoke_judge(",
        "invoke_judge_raw(",
        "requests.",
        "httpx.",
    ]

    for token in forbidden:
        assert token not in source
