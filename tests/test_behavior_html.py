import sys
from importlib import util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

HTML_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "11_behavior_html.py"
)

spec = util.spec_from_file_location(
    "behavior_html11",
    HTML_PATH,
)
assert spec is not None
assert spec.loader is not None

renderer = util.module_from_spec(spec)
sys.modules[spec.name] = renderer
spec.loader.exec_module(renderer)


def pair(
    *,
    state="complete_usable",
    usable="1",
):
    return {
        "profile": "fable",
        "pair_type": "resource_effect",
        "channel": "scaffold",
        "pair_state": state,
        "pair_usable": usable,
        "base_task_id": "task-a",
        "baseline_trial": "baseline-a",
        "treatment_trial": "treatment-a",
    }


def primary_row():
    row = {
        "profile": "fable",
        "pair_type": "resource_effect",
        "channel": "scaffold",
        "n_pairs": "29",
    }

    for endpoint in renderer.PRIMARY_ENDPOINTS:
        row[
            f"{endpoint}__delta_pp"
        ] = "-10"
        row[
            f"{endpoint}__ci_low_pp"
        ] = "-20"
        row[
            f"{endpoint}__ci_high_pp"
        ] = "0"
        row[
            f"{endpoint}__mcnemar_p"
        ] = "0.01"
        row[
            f"{endpoint}__holm_p"
        ] = "0.07"
        row[
            f"{endpoint}__adjusted_reject"
        ] = "0"

    return row


def test_partial_snapshot_detected():
    assert renderer.is_partial_snapshot([
        pair(
            state="missing_both",
            usable="0",
        )
    ])


def test_complete_snapshot_not_marked_partial():
    assert not renderer.is_partial_snapshot([
        pair()
    ])


def test_pair_ledger_counts_states():
    rows = [
        pair(),
        pair(
            state="baseline_censored",
            usable="0",
        ),
        pair(
            state="missing_both",
            usable="0",
        ),
    ]

    ledger = renderer.pair_ledger(
        rows
    )

    assert len(ledger) == 1

    result = ledger[0]

    assert result["planned"] == 3
    assert result["usable"] == 1
    assert result["censored"] == 1
    assert result["missing"] == 1


def test_primary_cell_uses_adjusted_significance():
    row = primary_row()

    endpoint = renderer.PRIMARY_ENDPOINTS[0]

    row[
        f"{endpoint}__adjusted_reject"
    ] = "1"

    rendered = renderer.primary_effect_cell(
        row,
        endpoint,
    )

    assert "Holm significant" in rendered
    assert "significant" in rendered


def test_html_snapshot_banner_and_escaping():
    document = renderer.build_html(
        inventory=[{
            "profile": "<fable>",
            "analysis_mode": "resource",
            "analysis_schema_version": "2.6",
            "study_signatures": "sig",
        }],
        primary=[primary_row()],
        secondary=[],
        multiplicity=[],
        pairs=[
            pair(
                state="missing_both",
                usable="0",
            )
        ],
        title="Behavior <Report>",
    )

    assert (
        "VALIDATION SNAPSHOT — NOT FINAL STUDY ESTIMATES"
        in document
    )

    assert "&lt;fable&gt;" in document
    assert "Behavior &lt;Report&gt;" in document
    assert "<fable>" not in document


def test_html_uses_conservative_integrity_and_analysis_plan_language():
    document = renderer.build_html(
        inventory=[{
            "profile": "claude",
            "analysis_mode": "full",
            "analysis_schema_version": "2.6",
            "study_signatures": "historical",
        }],
        primary=[primary_row()],
        secondary=[],
        multiplicity=[],
        pairs=[pair()],
        title="Behavior Report",
    )

    normalized = " ".join(document.split())

    assert "Integrity-sensitive modification" in normalized
    assert "does not imply evaluator-gaming intent" in normalized
    assert "frozen primary behavioral analysis family" in normalized
    assert "frozen exploratory behavioral/process analysis family" in normalized
    assert "pre-specified Holm" not in normalized


def test_scrollable_tables_have_sticky_headers():
    document = renderer.build_html(
        inventory=[{
            "profile": "claude",
            "analysis_mode": "full",
            "analysis_schema_version": "2.6",
            "study_signatures": "historical",
        }],
        primary=[primary_row()],
        secondary=[],
        multiplicity=[],
        pairs=[pair()],
        title="Behavior Report",
    )

    normalized = " ".join(document.split())

    assert ".table-scroll {" in document
    assert "max-height: 72vh;" in document
    assert "overflow: auto;" in document

    assert "position: sticky;" in document
    assert "top: 0;" in document

    assert ".matrix thead th:first-child {" in document
    assert "left: 0;" in document
    assert "z-index: 4;" in document

    assert 'class="table-scroll"' in normalized
