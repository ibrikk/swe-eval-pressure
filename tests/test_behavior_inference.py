import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from behavior_metrics import PRIMARY_BINARY_ENDPOINTS  # noqa: E402
from behavior_inference import (  # noqa: E402
    primary_binary_effect_rows,
)


def pair(
    *,
    pair_type,
    channel="scaffold",
    usable=1,
    baseline=0,
    treatment=0,
):
    row = {
        "profile": "fable",
        "base_task_id": "task",
        "pair_type": pair_type,
        "channel": channel,
        "baseline_condition": "eval_only",
        "baseline_channel": channel,
        "treatment_condition": (
            "eval_resource_deprivation"
            if pair_type == "resource_effect"
            else "eval_financial"
        ),
        "treatment_channel": channel,
        "pair_usable": usable,
    }

    for endpoint in PRIMARY_BINARY_ENDPOINTS:
        row[
            f"baseline_{endpoint}"
        ] = baseline if usable else ""

        row[
            f"treatment_{endpoint}"
        ] = treatment if usable else ""

    return row


def test_resource_family_has_seven_hypotheses():
    rows = [
        pair(
            pair_type="resource_effect",
            baseline=0,
            treatment=1,
        )
        for _ in range(8)
    ]

    effects, multiplicity = (
        primary_binary_effect_rows(
            rows,
            analysis_schema_version="2.5",
            analysis_mode="resource",
            study_signature="resource-study",
            bootstrap_replicates=1000,
            bootstrap_seed=123,
        )
    )

    assert len(effects) == 7
    assert len(multiplicity) == 1

    assert {
        row["family_name"]
        for row in effects
    } == {
        "resource_primary_binary_holm"
    }

    assert {
        row["family_size"]
        for row in effects
    } == {7}

    assert multiplicity[0]["family_size"] == 7
    assert multiplicity[0]["tested_family_size"] == 7


def test_known_mcnemar_and_holm_resource_case():
    rows = []

    for _ in range(8):
        item = pair(
            pair_type="resource_effect",
            baseline=0,
            treatment=0,
        )

        item[
            "baseline_validation_any"
        ] = 0
        item[
            "treatment_validation_any"
        ] = 1

        rows.append(item)

    effects, _ = primary_binary_effect_rows(
        rows,
        analysis_schema_version="2.5",
        analysis_mode="resource",
        study_signature="resource-study",
        bootstrap_replicates=1000,
        bootstrap_seed=123,
    )

    validation = next(
        row
        for row in effects
        if row["endpoint"] == "validation_any"
    )

    assert validation["n_pairs"] == 8
    assert validation["risk_difference_pp"] == pytest.approx(
        100.0
    )

    assert validation["treatment_only_positive"] == 8
    assert validation["baseline_only_positive"] == 0

    assert validation["mcnemar_exact_p"] == pytest.approx(
        0.0078125
    )

    # Seven-hypothesis Holm family.
    assert validation["holm_adjusted_p"] == pytest.approx(
        0.0546875
    )

    assert validation["unadjusted_reject"] == 1
    assert validation["adjusted_reject"] == 0


def test_resource_eval_control_not_added_to_primary_family():
    rows = [
        pair(
            pair_type="eval_effect",
            baseline=0,
            treatment=1,
        )
        for _ in range(8)
    ]

    effects, multiplicity = (
        primary_binary_effect_rows(
            rows,
            analysis_schema_version="2.5",
            analysis_mode="resource",
            study_signature="resource-study",
            bootstrap_replicates=100,
        )
    )

    assert effects == []
    assert multiplicity == []


def test_historical_family_spans_planned_contrasts():
    rows = []

    for pair_type in (
        "eval_effect",
        "financial_effect",
    ):
        for _ in range(4):
            rows.append(
                pair(
                    pair_type=pair_type,
                    channel="root",
                    baseline=0,
                    treatment=0,
                )
            )

    effects, multiplicity = (
        primary_binary_effect_rows(
            rows,
            analysis_schema_version="2.5",
            analysis_mode="full",
            study_signature="full-study",
            bootstrap_replicates=100,
        )
    )

    assert len(effects) == 14
    assert len(multiplicity) == 1

    assert multiplicity[0]["family_size"] == 14
    assert multiplicity[0]["tested_family_size"] == 14

    assert {
        row["family_name"]
        for row in effects
    } == {
        "historical_primary_binary_holm"
    }


def test_censored_planned_group_preserves_family_size():
    rows = [
        pair(
            pair_type="resource_effect",
            usable=0,
        )
    ]

    effects, multiplicity = (
        primary_binary_effect_rows(
            rows,
            analysis_schema_version="2.5",
            analysis_mode="resource",
            study_signature="resource-study",
            bootstrap_replicates=100,
        )
    )

    assert len(effects) == 7

    assert all(
        row["n_pairs"] == 0
        for row in effects
    )

    assert all(
        row["mcnemar_exact_p"] == ""
        for row in effects
    )

    assert all(
        row["holm_adjusted_p"] == ""
        for row in effects
    )

    assert multiplicity[0]["family_size"] == 7
    assert multiplicity[0]["tested_family_size"] == 0
    assert multiplicity[0]["untested_family_size"] == 7
