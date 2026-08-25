import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0,
    str(ROOT / "scripts"),
)

import semantic_bulk as bulk


def test_frozen_expected_denominators():
    assert bulk.PROFILES == (
        "claude",
        "fable",
        "codex",
        "llama",
    )

    assert (
        bulk.EXPECTED_USABLE_PER_PROFILE
        == 694
    )

    assert (
        bulk.EXPECTED_USABLE_TOTAL
        == 2776
    )


def test_stage_a_never_has_network_code():
    source = (
        ROOT
        / "scripts"
        / "semantic_bulk.py"
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


def test_summary_counts():
    manifest = {
        "planned_trajectories": 2800,
        "usable_trajectories": 2776,
        "censored_or_error": 24,
        "primary_panel_size": 2,
        "job_count": 5552,
        "unique_cache_keys": 5552,
        "planned_counts": {
            "claude": 700,
            "fable": 700,
            "codex": 700,
            "llama": 700,
        },
        "trial_counts": {
            "claude": 694,
            "fable": 694,
            "codex": 694,
            "llama": 694,
        },
        "censored_counts": {
            "claude": 6,
            "fable": 6,
            "codex": 6,
            "llama": 6,
        },
        "judge_counts": {
            "azure_ai/DeepSeek-V4-Pro": 2776,
            "gemini/gemini-3.6-flash": 2776,
        },
    }

    text = "\n".join(
        bulk.summary_lines(manifest)
    )

    assert (
        "planned trajectories: 2800"
        in text
    )
    assert (
        "usable trajectories: 2776"
        in text
    )
    assert "censored/error: 24" in text
    assert "judge jobs: 5552" in text
    assert "network calls: 0" in text


def test_job_cache_identity_uses_model():
    source = (
        ROOT
        / "scripts"
        / "semantic_bulk.py"
    ).read_text()

    assert (
        "panel.judge_cache_key("
        in source
    )
    assert 'model=judge["model"]' in source


def test_only_substantively_usable_rows_are_judged():
    source = (
        ROOT
        / "scripts"
        / "semantic_bulk.py"
    ).read_text()

    assert (
        '"substantive_usable" in row'
        in source
    )
    assert (
        'row.get(\n'
        '                    "substantive_usable"'
        in source
    )
