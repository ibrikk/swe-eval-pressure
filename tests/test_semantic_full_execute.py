import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "scripts"),
)

import semantic_full_execute as full


def test_counts():
    assert (
        full.EXPECTED_PLANNED
        == 2800
    )

    assert (
        full.EXPECTED_USABLE
        == 2776
    )

    assert (
        full.EXPECTED_CENSORED
        == 24
    )

    assert (
        full.EXPECTED_JOBS
        == 5552
    )


def test_policy():
    spec = json.loads(
        (
            ROOT
            / "config"
            / "semantic_full_v1.json"
        ).read_text()
    )

    assert (
        spec[
            "request_policy"
        ]["max_tokens"]
        == full.base.MAX_TOKENS
    )

    assert (
        spec[
            "request_policy"
        ]["timeout_seconds"]
        == full.base.TIMEOUT_SECONDS
    )

    assert (
        spec[
            "request_policy"
        ][
            "max_attempts_per_job"
        ]
        == full.orch.DEFAULT_MAX_ATTEMPTS
    )


def test_pending(
    monkeypatch,
    tmp_path,
):
    job = {
        "cache_key": "abc",
    }

    monkeypatch.setattr(
        full.dose,
        "terminal_artifact",
        lambda **kwargs: (
            "absent",
            None,
        ),
    )

    monkeypatch.setattr(
        full.base,
        "partial_attempts",
        lambda **kwargs: [],
    )

    result = full.locate_job(
        output_dir=(
            tmp_path / "full"
        ),
        inherited_roots=[
            tmp_path / "pilot",
            tmp_path / "dose",
        ],
        data_root=tmp_path,
        schema={},
        job=job,
    )

    assert (
        result["state"]
        == "pending"
    )


def test_dupe_cache(
    monkeypatch,
    tmp_path,
):
    job = {
        "cache_key": "abc",
    }

    output = (
        tmp_path / "full"
    )

    roots = [
        tmp_path / "pilot",
        tmp_path / "dose",
    ]

    def fake_terminal(
        *,
        path,
        **kwargs,
    ):
        if output in (
            path.parents
        ):
            return (
                "absent",
                None,
            )

        return (
            "ok",
            {
                "status": "ok",
            },
        )

    monkeypatch.setattr(
        full.dose,
        "terminal_artifact",
        fake_terminal,
    )

    monkeypatch.setattr(
        full.base,
        "partial_attempts",
        lambda **kwargs: [],
    )

    with pytest.raises(
        ValueError,
        match=(
            "multiple inherited"
        ),
    ):
        full.locate_job(
            output_dir=output,
            inherited_roots=roots,
            data_root=tmp_path,
            schema={},
            job=job,
        )
