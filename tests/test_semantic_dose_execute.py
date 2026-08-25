import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "scripts"),
)

import semantic_dose_execute as exe


def test_identity():
    job = {
        "cache_key": "abc",
        "profile": "claude",
        "condition": "clean",
        "placement": "none",
        "pressure_type": "none",
        "judge_family": "deepseek",
        "judge_model": "model",
        "trial_name": "trial",
        "trajectory_hash": "hash",
    }

    artifact = dict(job)

    assert exe.artifact_identity_ok(
        artifact=artifact,
        job=job,
    )

    artifact["judge_model"] = (
        "different"
    )

    assert not (
        exe.artifact_identity_ok(
            artifact=artifact,
            job=job,
        )
    )


def test_jobs_unique():
    dose = {
        "trials": [{
            "from_frozen_pilot": True,
            "jobs": [
                {
                    "cache_key": "a",
                    "profile": "claude",
                    "condition": "clean",
                    "placement": "none",
                    "trial_name": "x",
                    "judge_family": "a",
                    "judge_model": "a",
                },
                {
                    "cache_key": "b",
                    "profile": "claude",
                    "condition": "clean",
                    "placement": "none",
                    "trial_name": "x",
                    "judge_family": "b",
                    "judge_model": "b",
                },
            ],
        }]
    }

    # dose_jobs intentionally enforces
    # the production size of 200.
    try:
        exe.dose_jobs(dose)
    except ValueError as exc:
        assert (
            "expected 200"
            in str(exc)
        )
    else:
        raise AssertionError(
            "expected size validation"
        )


def test_frozen_signature():
    trial = {
        "profile": "claude",
        "condition": "clean",
        "placement": "none",
        "pressure_type": "none",
        "trial_name": "x",
        "trajectory_hash": "h",
        "dose_hash": "d",
        "from_frozen_pilot": True,
    }

    signature = (
        exe.frozen_signature(
            trial
        )
    )

    assert signature == (
        "claude",
        "clean",
        "none",
        "none",
        "x",
        "h",
        "d",
        True,
    )


def test_source_paths(tmp_path):
    job = {
        "cache_key": "abc",
    }

    writable = (
        exe.writable_path(
            output_dir=tmp_path
            / "dose",
            job=job,
        )
    )

    inherited = (
        exe.inherited_path(
            inherited_root=(
                tmp_path / "pilot"
            ),
            job=job,
        )
    )

    assert writable == (
        tmp_path
        / "dose"
        / "jobs"
        / "abc.json"
    )

    assert inherited == (
        tmp_path
        / "pilot"
        / "jobs"
        / "abc.json"
    )

    assert writable != inherited
