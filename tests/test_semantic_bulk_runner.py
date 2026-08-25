import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "scripts"),
)

import semantic_bulk_runner as runner


SPEC_PATH = (
    ROOT
    / "config"
    / "semantic_pilot_v1.json"
)


def load_spec():
    return json.loads(
        SPEC_PATH.read_text()
    )


def test_spec_shape():
    spec = load_spec()
    trials = spec["trials"]

    assert (
        spec["selection_status"]
        == "frozen_before_semantic_outcomes"
    )
    assert len(trials) == 10
    assert spec["core_judge_jobs"] == 20

    assert len({
        (
            t["profile"],
            t["trial_name"],
        )
        for t in trials
    }) == 10

    assert Counter(
        t["profile"]
        for t in trials
    ) == {
        "claude": 3,
        "fable": 3,
        "codex": 2,
        "llama": 2,
    }

    cells = {
        (
            t["condition"],
            t["placement"],
        )
        for t in trials
    }

    assert cells == set(
        runner.PILOT_CELLS
    )


def test_spec_hashes():
    spec = load_spec()

    for trial in spec["trials"]:
        assert (
            trial["selection_hash"]
            == runner.selection_hash(
                profile=trial[
                    "profile"
                ],
                trial_name=trial[
                    "trial_name"
                ],
            )
        )

        assert len(
            trial["trajectory_hash"]
        ) == 64


def test_cache_ok(tmp_path):
    path = tmp_path / "job.json"

    path.write_text(
        json.dumps({
            "cache_key": "abc",
            "status": "ok",
            "final_cache_entry": {
                "status": "ok",
            },
        })
    )

    assert runner.artifact_is_complete(
        path,
        expected_cache_key="abc",
    )


def test_cache_bad(tmp_path):
    path = tmp_path / "job.json"

    path.write_text(
        json.dumps({
            "cache_key": "abc",
            "status": "missing",
            "final_cache_entry": None,
        })
    )

    assert not runner.artifact_is_complete(
        path,
        expected_cache_key="abc",
    )

    path.write_text(
        "{not-json"
    )

    assert not runner.artifact_is_complete(
        path,
        expected_cache_key="abc",
    )
