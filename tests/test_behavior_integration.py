import importlib.util
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"

sys.path.insert(0, str(SCRIPTS))

from behavior_metrics import (  # noqa: E402
    PRIMARY_BINARY_ENDPOINTS,
    SECONDARY_ACTION_METRICS,
    extract_action_metrics,
)

spec = importlib.util.spec_from_file_location(
    "analyzer07_behavior_integration",
    SCRIPTS / "07_analyze.py",
)
assert spec is not None
assert spec.loader is not None

analyzer = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = analyzer
spec.loader.exec_module(analyzer)


def test_taxonomy_primary_ids_match_code():
    taxonomy = json.loads(
        (
            PROJECT_ROOT
            / "config"
            / "behavior_taxonomy.json"
        ).read_text()
    )

    ids = tuple(
        endpoint["id"]
        for endpoint in taxonomy["primary_binary_endpoints"]
    )

    assert ids == PRIMARY_BINARY_ENDPOINTS


def test_taxonomy_secondary_metrics_are_emitted():
    taxonomy = json.loads(
        (
            PROJECT_ROOT
            / "config"
            / "behavior_taxonomy.json"
        ).read_text()
    )

    configured = tuple(
        taxonomy["secondary_action_metrics"]
    )

    assert configured == SECONDARY_ACTION_METRICS

    metrics = extract_action_metrics(
        [],
        changed_files=3,
    )

    for name in SECONDARY_ACTION_METRICS:
        assert name in metrics

    assert metrics["unique_files_modified"] == 3
    assert (
        metrics["unique_files_modified_scope"]
        == "final_patch_paths"
    )


def test_behavior_trials_exclude_infrastructure_censoring():
    substantive = {
        "analysis_schema_version": "2.5",
        "analysis_mode": "full",
        "study_signature": "study-a",
        "profile": "claude",
        "trial_name": "trial-good",
        "base_task_id": "task-a",
        "condition": "eval_only",
        "channel": "source",
        "terminal_status": "completed",
        "substantive_usable": 1,
        "overall_pass": 1,
    }

    for field in PRIMARY_BINARY_ENDPOINTS:
        substantive[field] = 0

    for field in SECONDARY_ACTION_METRICS:
        substantive[field] = 0

    censored = dict(substantive)
    censored.update({
        "trial_name": "trial-censored",
        "terminal_status": "api_error",
        "substantive_usable": 0,
    })

    rows = analyzer.behavior_trial_rows(
        [substantive, censored]
    )

    assert len(rows) == 1
    assert rows[0]["trial_name"] == "trial-good"


def test_behavior_trials_keep_model_refusal():
    row = {
        "trial_name": "refusal",
        "terminal_status": "safety_refusal",
        "substantive_usable": 1,
    }

    rows = analyzer.behavior_trial_rows([row])

    assert len(rows) == 1
    assert rows[0]["terminal_status"] == "safety_refusal"


def test_behavior_projection_excludes_semantic_labels():
    row = {
        "trial_name": "trial-a",
        "terminal_status": "completed",
        "substantive_usable": 1,
        "pressure_cue_stance": "rejected_as_prompt_injection",
        "semantic_confidence": 0.99,
    }

    projected = analyzer.behavior_trial_rows([row])[0]

    assert "pressure_cue_stance" not in projected
    assert "semantic_confidence" not in projected


def test_delegation_prompt_does_not_become_external_lookup():
    action = {
        "category": "delegate",
        "command": "",
        "path": "",
        "arguments_text": (
            "Use curl https://github.com/example/repo "
            "to investigate this."
        ),
    }

    metrics = extract_action_metrics([action])

    assert metrics["subagent_delegation_calls"] == 1
    assert metrics["external_lookup_any"] == 0
    assert metrics["external_lookup_calls"] == 0
