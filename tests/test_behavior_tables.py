import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from behavior_metrics import PRIMARY_BINARY_ENDPOINTS  # noqa: E402
from behavior_tables import behavior_prevalence_rows  # noqa: E402


def row(
    *,
    condition="eval_only",
    channel="source",
    status="completed",
    usable=1,
    search=0,
    validation=0,
):
    result = {
        "analysis_schema_version": "2.3",
        "analysis_mode": "full",
        "study_signature": "study-1",
        "profile": "claude",
        "condition": condition,
        "channel": channel,
        "terminal_status": status,
        "substantive_usable": usable,
    }

    for endpoint in PRIMARY_BINARY_ENDPOINTS:
        result[endpoint] = 0

    result["broad_repo_search_any"] = search
    result["validation_any"] = validation

    return result


def endpoint_row(rows, endpoint):
    matches = [
        item
        for item in rows
        if item["endpoint"] == endpoint
    ]
    assert len(matches) == 1
    return matches[0]


def test_prevalence_counts():
    rows = behavior_prevalence_rows([
        row(search=1),
        row(search=1),
        row(search=0),
        row(search=1),
    ])

    result = endpoint_row(
        rows,
        "broad_repo_search_any",
    )

    assert result["n"] == 4
    assert result["positive_count"] == 3
    assert result["negative_count"] == 1
    assert result["prevalence"] == pytest.approx(0.75)
    assert result["prevalence_pct"] == pytest.approx(75.0)


def test_conditions_and_channels_are_separate_groups():
    rows = behavior_prevalence_rows([
        row(
            condition="clean",
            channel="none",
            validation=1,
        ),
        row(
            condition="eval_only",
            channel="source",
            validation=0,
        ),
    ])

    validation_rows = [
        item
        for item in rows
        if item["endpoint"] == "validation_any"
    ]

    assert len(validation_rows) == 2

    keys = {
        (
            item["condition"],
            item["channel"],
            item["positive_count"],
        )
        for item in validation_rows
    }

    assert keys == {
        ("clean", "none", 1),
        ("eval_only", "source", 0),
    }


def test_infrastructure_censored_rows_are_excluded():
    rows = behavior_prevalence_rows([
        row(search=1),
        row(
            status="api_error",
            usable=0,
            search=0,
        ),
    ])

    result = endpoint_row(
        rows,
        "broad_repo_search_any",
    )

    assert result["n"] == 1
    assert result["positive_count"] == 1


def test_refusal_denominator():
    rows = behavior_prevalence_rows([
        row(search=1),
        row(
            status="safety_refusal",
            usable=1,
            search=0,
        ),
    ])

    result = endpoint_row(
        rows,
        "broad_repo_search_any",
    )

    assert result["n"] == 2
    assert result["completed_n"] == 1
    assert result["safety_refusal_n"] == 1
    assert result["positive_count"] == 1
    assert result["prevalence_pct"] == pytest.approx(50.0)


def test_invalid_binary_endpoint_fails_closed():
    bad = row()
    bad["validation_any"] = ""

    with pytest.raises(
        ValueError,
        match="validation_any must be binary",
    ):
        behavior_prevalence_rows([bad])


def test_empty_input_returns_empty_table():
    assert behavior_prevalence_rows([]) == []
