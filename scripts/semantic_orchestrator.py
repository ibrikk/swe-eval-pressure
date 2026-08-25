#!/usr/bin/env python3
"""Production orchestration primitives for semantic judging.

This layer is separate from semantic measurement itself.

Responsibilities:
- bounded retry policy;
- preservation of every judge attempt;
- explicit failure/missingness states;
- valid-result cache semantics;
- two-core-judge consensus.

It does not alter semantic labels, infer observable behavior,
or adjudicate disagreements with a third model.
"""

from __future__ import annotations

import json
import ssl
import time
import urllib.error
from typing import Any, Callable

import semantic_panel as panel


ORCHESTRATOR_VERSION = "1.0"

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_RETRY_DELAYS = (0.0, 2.0, 5.0)

RETRYABLE_HTTP_CODES = {
    408,
    409,
    425,
    429,
    500,
    502,
    503,
    504,
}


def exception_status(
    exc: BaseException,
) -> str:
    if isinstance(
        exc,
        urllib.error.HTTPError,
    ):
        return "http_error"

    if isinstance(
        exc,
        (
            TimeoutError,
            ssl.SSLError,
            urllib.error.URLError,
            ConnectionError,
        ),
    ):
        return "transport_error"

    if isinstance(
        exc,
        (
            json.JSONDecodeError,
            ValueError,
            KeyError,
            TypeError,
        ),
    ):
        return "parse_error"

    return "error"


def exception_is_retryable(
    exc: BaseException,
) -> bool:
    if isinstance(
        exc,
        urllib.error.HTTPError,
    ):
        return (
            exc.code
            in RETRYABLE_HTTP_CODES
        )

    return isinstance(
        exc,
        (
            TimeoutError,
            ssl.SSLError,
            urllib.error.URLError,
            ConnectionError,
            json.JSONDecodeError,
            ValueError,
            KeyError,
            TypeError,
        ),
    )


def cache_entry_is_reusable(
    entry: Any,
) -> bool:
    return (
        isinstance(entry, dict)
        and entry.get("status") == "ok"
        and isinstance(
            entry.get("judgment"),
            dict,
        )
    )


def run_judge_with_retries(
    *,
    trial_name: str,
    trajectory_hash: str,
    judge: dict[str, str],
    schema: dict[str, Any],
    semantic_view_version: str,
    agent_blocks: list[dict[str, Any]],
    invoke_once: Callable[
        [],
        tuple[dict[str, Any], str],
    ],
    max_attempts: int = (
        DEFAULT_MAX_ATTEMPTS
    ),
    retry_delays: tuple[
        float,
        ...,
    ] = DEFAULT_RETRY_DELAYS,
    sleep: Callable[[float], None] = (
        time.sleep
    ),
) -> dict[str, Any]:
    """Run one judge while preserving all attempts."""

    if max_attempts < 1:
        raise ValueError(
            "max_attempts must be >= 1"
        )

    attempts: list[
        dict[str, Any]
    ] = []

    final_entry: (
        dict[str, Any] | None
    ) = None

    for attempt_number in range(
        1,
        max_attempts + 1,
    ):
        record: dict[str, Any] = {
            "attempt": attempt_number,
            "status": "",
            "finish_reason": "",
            "validation_error": "",
            "exception_type": "",
            "exception_message": "",
            "raw_response": None,
            "judgment": None,
        }

        try:
            outcome = invoke_once()

            raw_response = None

            # Production path: preserve the provider response
            # before parsing. The legacy two-tuple form remains
            # supported for deterministic tests and callers.
            if (
                isinstance(outcome, dict)
                and "raw_response" in outcome
            ):
                raw_response = outcome[
                    "raw_response"
                ]
                finish_reason = str(
                    outcome.get(
                        "finish_reason",
                        "",
                    )
                )

                record[
                    "raw_response"
                ] = raw_response
                record[
                    "finish_reason"
                ] = finish_reason

                judgment = (
                    panel.parse_chat_completion(
                        raw_response
                    )
                )
            else:
                (
                    judgment,
                    finish_reason,
                ) = outcome

                record[
                    "finish_reason"
                ] = finish_reason

            record[
                "judgment"
            ] = judgment

            validation_error = (
                panel
                .judgment_validation_error(
                    judgment,
                    schema=schema,
                    agent_blocks=(
                        agent_blocks
                    ),
                )
            )

            entry = (
                panel.make_cache_entry(
                    trial_name=trial_name,
                    trajectory_hash=(
                        trajectory_hash
                    ),
                    model=judge["model"],
                    family=judge[
                        "family"
                    ],
                    schema=schema,
                    semantic_view_version=(
                        semantic_view_version
                    ),
                    judgment=judgment,
                    finish_reason=(
                        finish_reason
                    ),
                    validation_error=(
                        validation_error
                    ),
                )
            )

            record["status"] = (
                entry["status"]
            )

            record[
                "validation_error"
            ] = (
                validation_error or ""
            )

            attempts.append(record)

            if entry["status"] == "ok":
                final_entry = entry
                break

            retryable = True

        except Exception as exc:
            status = (
                exception_status(exc)
            )

            record["status"] = status
            record[
                "exception_type"
            ] = type(exc).__name__
            record[
                "exception_message"
            ] = str(exc)

            attempts.append(record)

            retryable = (
                exception_is_retryable(
                    exc
                )
            )

        if (
            attempt_number
            >= max_attempts
            or not retryable
        ):
            break

        delay_index = min(
            attempt_number - 1,
            len(retry_delays) - 1,
        )

        if retry_delays:
            delay = retry_delays[
                delay_index
            ]
        else:
            delay = 0.0

        if delay > 0:
            sleep(delay)

    return {
        "orchestrator_version": (
            ORCHESTRATOR_VERSION
        ),
        "trial_name": trial_name,
        "trajectory_hash": (
            trajectory_hash
        ),
        "judge_family": judge[
            "family"
        ],
        "judge_model": judge[
            "model"
        ],
        "semantic_schema_version": (
            schema["schema_version"]
        ),
        "rubric_version": (
            schema["rubric_version"]
        ),
        "semantic_view_version": (
            semantic_view_version
        ),
        "attempt_count": len(
            attempts
        ),
        "attempts": attempts,
        "status": (
            "ok"
            if final_entry is not None
            else "missing"
        ),
        "final_cache_entry": (
            final_entry
        ),
    }


def judgment_label(
    entry: dict[str, Any],
    field: str,
) -> str | None:
    if not cache_entry_is_reusable(
        entry
    ):
        return None

    judgment = entry[
        "judgment"
    ]

    result = judgment.get(field)

    if not isinstance(
        result,
        dict,
    ):
        return None

    label = result.get("label")

    if not isinstance(
        label,
        str,
    ):
        return None

    return label


def consensus_for_field(
    *,
    field: str,
    judge_entries: list[
        dict[str, Any] | None
    ],
) -> dict[str, Any]:
    valid = [
        entry
        for entry in judge_entries
        if cache_entry_is_reusable(
            entry
        )
    ]

    labels = [
        judgment_label(
            entry,
            field,
        )
        for entry in valid
    ]

    labels = [
        label
        for label in labels
        if label is not None
    ]

    if len(labels) < 2:
        return {
            "field": field,
            "valid_judges": len(
                labels
            ),
            "status": "missing",
            "consensus_exists": (
                False
            ),
            "label": None,
        }

    if len(set(labels)) == 1:
        return {
            "field": field,
            "valid_judges": len(
                labels
            ),
            "status": "agreement",
            "consensus_exists": True,
            "label": labels[0],
        }

    return {
        "field": field,
        "valid_judges": len(
            labels
        ),
        "status": "disagreement",
        "consensus_exists": False,
        "label": None,
    }


def core_panel_consensus(
    *,
    schema: dict[str, Any],
    judge_entries: list[
        dict[str, Any] | None
    ],
) -> dict[str, Any]:
    expected_panel = int(
        schema["consensus"][
            "primary_panel_size"
        ]
    )

    if len(judge_entries) != (
        expected_panel
    ):
        raise ValueError(
            "judge entry count does not "
            "match primary panel size"
        )

    fields = {}

    for field in (
        panel.expected_field_names(
            schema
        )
    ):
        fields[field] = (
            consensus_for_field(
                field=field,
                judge_entries=(
                    judge_entries
                ),
            )
        )

    return {
        "semantic_schema_version": (
            schema["schema_version"]
        ),
        "primary_panel_size": (
            expected_panel
        ),
        "fields": fields,
    }
