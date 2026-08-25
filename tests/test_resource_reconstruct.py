import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "scripts"),
)

import resource_reconstruct as rr


POLICY = json.loads(
    (
        ROOT
        / "config"
        / "resource_reconstruction_policy_v1.json"
    ).read_text()
)


def test_no_exception():
    assert (
        rr.classification(
            exception_type=None,
            policy=POLICY,
        )
        == "substantive_model_outcome"
    )


def test_safety_refusal():
    assert (
        rr.classification(
            exception_type=(
                "AgentSafetyRefusalError"
            ),
            policy=POLICY,
        )
        == "substantive_model_outcome"
    )


def test_infrastructure():
    for error in (
        "UnknownApiError",
        "AgentSetupTimeoutError",
        "VerifierTimeoutError",
        "NetworkConnectionError",
        "AgentTimeoutError",
    ):
        assert (
            rr.classification(
                exception_type=error,
                policy=POLICY,
            )
            == "infrastructure_censored"
        )


def test_nonzero_requires_adjudication():
    assert (
        rr.classification(
            exception_type=(
                "NonZeroAgentExitCodeError"
            ),
            policy=POLICY,
        )
        == "needs_adjudication"
    )


def test_unknown_requires_adjudication():
    assert (
        rr.classification(
            exception_type=(
                "SomethingNewError"
            ),
            policy=POLICY,
        )
        == "needs_adjudication"
    )
