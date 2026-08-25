import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCHEMA_PATH = (
    ROOT
    / "config"
    / "semantic_judge_schema.json"
)

PLAN_PATH = (
    ROOT
    / "docs"
    / "SEMANTIC_ANALYSIS_PLAN.md"
)


def load_schema():
    return json.loads(
        SCHEMA_PATH.read_text(
            encoding="utf-8"
        )
    )


def test_core_panel_is_two_independent_families():
    schema = load_schema()

    judges = schema["primary_judges"]

    assert len(judges) == 2

    assert {
        judge["family"]
        for judge in judges
    } == {
        "deepseek",
        "gemini",
    }

    assert {
        judge["model"]
        for judge in judges
    } == {
        "azure_ai/DeepSeek-V4-Pro",
        "gemini/gemini-3.6-flash",
    }


def test_evaluated_openai_agent_is_excluded_from_core_panel():
    schema = load_schema()

    assert all(
        "gpt-5.6" not in judge["model"]
        for judge in schema["primary_judges"]
    )


def test_json_mode_and_zero_temperature_are_frozen():
    schema = load_schema()

    assert schema["request"]["temperature"] == 0
    assert (
        schema["request"]["response_format"]
        == {"type": "json_object"}
    )


def test_semantic_fields_are_explicit_state_only():
    schema = load_schema()

    assert set(schema["fields"]) == {
        "evaluation_recognition",
        "pressure_recognition",
        "cue_provenance_stance",
        "explicit_resistance",
        "claimed_behavioral_response",
    }


def test_missing_judgment_is_not_negative():
    schema = load_schema()

    assert (
        schema["consensus"][
            "missing_is_not_negative"
        ]
        is True
    )


def test_individual_outputs_must_be_preserved():
    schema = load_schema()

    assert (
        schema["consensus"][
            "preserve_individual_outputs"
        ]
        is True
    )


def test_semantic_states_are_post_treatment():
    schema = load_schema()

    assert (
        schema["causal_interpretation"][
            "semantic_states_are_post_treatment"
        ]
        is True
    )

    assert (
        schema["causal_interpretation"][
            "conditioning_on_semantic_states_is_descriptive"
        ]
        is True
    )


def test_plan_forbids_semantic_behavior_reconstruction():
    plan = PLAN_PATH.read_text(
        encoding="utf-8"
    )

    assert (
        "must not infer observable execution behavior"
        in plan
    )

    assert (
        "It must not be presented as the causal effect of resistance."
        in plan
    )

    assert (
        "It is not a preregistration."
        in plan
    )



def test_gemini_is_promoted_to_core_preproduction():
    schema = load_schema()

    assert "gemini" in {
        judge["family"]
        for judge in schema["primary_judges"]
    }

    assert "qwen" not in {
        judge["family"]
        for judge in schema["primary_judges"]
    }

    assert "qwen" in {
        judge["family"]
        for judge in schema[
            "retired_preproduction_candidates"
        ]
    }


def test_panel_amendment_is_preproduction_and_not_outcome_selected():
    schema = load_schema()

    amendment = schema["panel_amendment"]

    assert amendment["stage"] == "pre_production"
    assert amendment["outcome_based_selection"] is False
    assert amendment["bulk_semantic_judging_started"] is False

    assert (
        amendment["replacement"]["removed_core_judge"]
        == "fireworks_ai/qwen3p8-max"
    )

    assert (
        amendment["replacement"]["added_core_judge"]
        == "gemini/gemini-3.6-flash"
    )


def test_agreement_plan_matches_two_llm_three_human_design():
    schema = load_schema()

    agreement = schema["agreement"]

    assert agreement["llm_llm_primary"] == "cohen_kappa"
    assert agreement["human_llm"] == "cohen_kappa"
    assert agreement["human_human"] == "fleiss_kappa"
    assert agreement["robustness_metric"] == "gwet_ac1"
    assert agreement["human_validation_raters"] == 3
    assert agreement["target_human_validation_sample"] == 200
