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


from behavior_tables import (  # noqa: E402
    BEHAVIOR_PAIR_METRICS,
    matched_behavior_pair_rows,
)


def behavior_pair_trial(
    *,
    task,
    condition,
    channel,
    replicate=1,
    search=0,
    validation=0,
    actions=5,
    overall_pass=0,
):
    result = {
        "base_task_id": task,
        "condition": condition,
        "channel": channel,
        "replicate_index": replicate,
        "broad_repo_search_any": search,
        "validation_any": validation,
        "behavioral_action_calls": actions,
        "overall_pass": overall_pass,
    }

    for endpoint in PRIMARY_BINARY_ENDPOINTS:
        result.setdefault(endpoint, 0)

    for metric in BEHAVIOR_PAIR_METRICS:
        result.setdefault(metric, 0)

    result["broad_repo_search_any"] = search
    result["validation_any"] = validation
    result["behavioral_action_calls"] = actions
    result["overall_pass"] = overall_pass

    return result


def experimental_pair(
    *,
    task="task-a",
    pair_type="resource_effect",
    baseline_condition="eval_only",
    baseline_channel="scaffold",
    treatment_condition="eval_resource_deprivation",
    treatment_channel="scaffold",
    pair_state="complete_usable",
    pair_usable=1,
):
    return {
        "profile": "fable",
        "base_task_id": task,
        "pair_type": pair_type,
        "channel": treatment_channel,
        "baseline_condition": baseline_condition,
        "baseline_channel": baseline_channel,
        "treatment_condition": treatment_condition,
        "treatment_channel": treatment_channel,
        "replicate_index": 1,
        "pair_state": pair_state,
        "pair_usable": pair_usable,
        "baseline_trial": "base-trial",
        "treatment_trial": "treat-trial",
        "baseline_terminal_status": "completed",
        "treatment_terminal_status": "completed",
    }


def test_matched_behavior_pair_delta():
    baseline = behavior_pair_trial(
        task="task-a",
        condition="eval_only",
        channel="scaffold",
        search=1,
        validation=1,
        actions=10,
        overall_pass=1,
    )

    treatment = behavior_pair_trial(
        task="task-a",
        condition="eval_resource_deprivation",
        channel="scaffold",
        search=0,
        validation=1,
        actions=7,
        overall_pass=0,
    )

    rows = matched_behavior_pair_rows(
        [experimental_pair()],
        [baseline, treatment],
        analysis_schema_version="2.4",
        analysis_mode="resource",
        study_signature="study-r",
    )

    assert len(rows) == 1

    result = rows[0]

    assert result["pair_usable"] == 1

    assert (
        result[
            "delta_broad_repo_search_any"
        ]
        == -1
    )

    assert result["delta_validation_any"] == 0

    assert (
        result[
            "delta_behavioral_action_calls"
        ]
        == -3
    )

    assert result["delta_overall_pass"] == -1


def test_censored_side_is_not_imputed_zero():
    baseline = behavior_pair_trial(
        task="task-a",
        condition="eval_only",
        channel="scaffold",
        validation=1,
    )

    pair = experimental_pair(
        pair_state="treatment_censored",
        pair_usable=0,
    )

    rows = matched_behavior_pair_rows(
        [pair],
        [baseline],
        analysis_schema_version="2.4",
        analysis_mode="resource",
        study_signature="study-r",
    )

    result = rows[0]

    assert (
        result["baseline_validation_any"]
        == 1
    )
    assert (
        result["treatment_validation_any"]
        == ""
    )
    assert result["delta_validation_any"] == ""


def test_usable_pair_missing_behavior_fails():
    baseline = behavior_pair_trial(
        task="task-a",
        condition="eval_only",
        channel="scaffold",
    )

    with pytest.raises(
        ValueError,
        match="usable experimental pair",
    ):
        matched_behavior_pair_rows(
            [experimental_pair()],
            [baseline],
            analysis_schema_version="2.4",
            analysis_mode="resource",
            study_signature="study-r",
        )


def test_duplicate_behavior_pair_key_fails():
    baseline = behavior_pair_trial(
        task="task-a",
        condition="eval_only",
        channel="scaffold",
    )

    with pytest.raises(
        ValueError,
        match="duplicate substantive behavioral row",
    ):
        matched_behavior_pair_rows(
            [],
            [baseline, dict(baseline)],
            analysis_schema_version="2.4",
            analysis_mode="resource",
            study_signature="study-r",
        )
