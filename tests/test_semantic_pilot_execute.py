import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "scripts"),
)

import semantic_bulk as bulk
import semantic_panel as panel
import semantic_pilot_execute as exe
from semantic_view import (
    SEMANTIC_VIEW_SCHEMA_VERSION,
)


def test_atomic(tmp_path):
    path = tmp_path / "x.json"

    exe.atomic_write_json(
        path,
        {"hello": "world"},
    )

    assert json.loads(
        path.read_text()
    ) == {
        "hello": "world"
    }

    assert not list(
        tmp_path.glob(
            ".x.json.*.tmp"
        )
    )


def test_terminal(tmp_path):
    path = tmp_path / "job.json"

    exe.atomic_write_json(
        path,
        {
            "cache_key": "abc",
            "status": "missing",
            "attempts": [{
                "attempt": 1,
            }],
            "final_cache_entry": None,
        },
    )

    assert exe.artifact_is_terminal(
        path,
        expected_cache_key="abc",
    )


def test_partial(tmp_path):
    cache_key = "abc"

    directory = (
        exe.attempt_directory(
            tmp_path,
            cache_key,
        )
    )

    directory.mkdir(
        parents=True
    )

    (
        directory
        / "attempt-01.json"
    ).write_text("{}")

    paths = exe.partial_attempts(
        output_dir=tmp_path,
        cache_key=cache_key,
    )

    assert len(paths) == 1


def test_mock(tmp_path):
    schema = panel.load_schema()
    judge = schema[
        "primary_judges"
    ][0]

    trajectory_path = (
        tmp_path / "trajectory.json"
    )

    trajectory_path.write_text(
        json.dumps({
            "steps": [{
                "source": "agent",
                "message": (
                    "I will inspect the code."
                ),
            }],
        })
    )

    trajectory_hash = (
        bulk.sha256_file(
            trajectory_path
        )
    )

    trial_name = "mock-trial"

    cache_key = (
        panel.judge_cache_key(
            trial_name=trial_name,
            trajectory_hash=(
                trajectory_hash
            ),
            model=judge["model"],
            schema=schema,
            semantic_view_version=(
                SEMANTIC_VIEW_SCHEMA_VERSION
            ),
        )
    )

    job = {
        "profile": "claude",
        "trial_name": trial_name,
        "condition": "clean",
        "placement": "none",
        "pressure_type": "none",
        "trajectory_path": str(
            trajectory_path
        ),
        "trajectory_hash": (
            trajectory_hash
        ),
        "judge_family": (
            judge["family"]
        ),
        "judge_model": (
            judge["model"]
        ),
        "cache_key": cache_key,
    }

    row = {
        "trial_name": trial_name,
        "substantive_usable": True,
        "condition": "clean",
        "channel": "none",
        "pressure_type": "none",
        "eval_cue_id": "",
        "eval_cue_text": "",
    }

    judgment = {
        "evaluation_recognition": {
            "label": "not_observed",
            "evidence": [],
        },
        "pressure_recognition": {
            "label": "not_observed",
            "evidence": [],
        },
        "cue_provenance_stance": {
            "label": "not_observed",
            "evidence": [],
        },
        "explicit_resistance": {
            "label": "not_observed",
            "evidence": [],
        },
        "claimed_behavioral_response": {
            "label": "none_observed",
            "evidence": [],
        },
    }

    fake_key = (
        "definitely-not-a-real-key"
    )

    def fake_invoke(
        *,
        base_url,
        api_key,
        body,
        timeout,
    ):
        assert (
            api_key == fake_key
        )

        assert (
            body["model"]
            == judge["model"]
        )

        return (
            {
                "choices": [{
                    "finish_reason": (
                        "stop"
                    ),
                    "message": {
                        "content": (
                            json.dumps(
                                judgment
                            )
                        ),
                    },
                }],
            },
            "stop",
        )

    artifact = exe.execute_job(
        job=job,
        row=row,
        schema=schema,
        output_dir=tmp_path,
        base_url=(
            "https://example.test"
        ),
        keys=[fake_key],
        invoke_raw=fake_invoke,
        timeout=1,
        retry_delays=(0, 0, 0),
    )

    assert artifact[
        "status"
    ] == "ok"

    assert artifact[
        "attempt_count"
    ] == 1

    artifact_path = (
        tmp_path
        / "jobs"
        / f"{cache_key}.json"
    )

    assert artifact_path.is_file()

    text = (
        artifact_path
        .read_text()
    )

    assert fake_key not in text

    journals = list(
        (
            tmp_path
            / "attempts"
            / cache_key
        ).glob(
            "attempt-*.json"
        )
    )

    assert len(journals) == 1

    journal_text = (
        journals[0]
        .read_text()
    )

    assert fake_key not in (
        journal_text
    )

    assert (
        "raw_response"
        in json.loads(
            journal_text
        )
    )
