#!/usr/bin/env python3
"""Resource-deprivation benchmark mode.

This mode is intentionally separate from the 10-condition primary placement matrix.
It reuses each base task's permanent evaluation cue and renders matched scaffold-native
controls: clean, eval-only, and eval+resource-deprivation.
"""
from __future__ import annotations

import os
from typing import Any

MODE = "resource"
CONDITION = "eval_resource_deprivation"
PRESSURE_TYPE = "resource_deprivation"
DEFAULT_MESSAGE_INDEX = 10
DEFAULT_TASK_COUNT = 70


def task_count(total: int) -> int:
    return min(int(os.getenv("RESOURCE_TASK_COUNT", str(DEFAULT_TASK_COUNT))), total)


def variants_per_task(channels: list[str]) -> int:
    # One clean control plus eval-only and eval+resource for each delivery channel.
    return 1 + 2 * len(channels)


def combinations(channels: list[str]) -> set[tuple[str, str]]:
    result = {("clean", "none")}
    for channel in channels:
        result.add(("eval_only", channel))
        result.add((CONDITION, channel))
    return result


def build_variants(
    channels: list[str],
    resource_pressure: dict[str, Any],
    eval_text: str,
    compose,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = [
        {
            "condition": "clean",
            "channel": "none",
            "pressure_type": "none",
            "pressure_index": None,
            "pressure_level": 0,
            "content": "",
        }
    ]
    for channel in channels:
        result.extend(
            [
                {
                    "condition": "eval_only",
                    "channel": channel,
                    "pressure_type": "none",
                    "pressure_index": None,
                    "pressure_level": 0,
                    "content": compose(eval_text, None),
                },
                {
                    "condition": CONDITION,
                    "channel": channel,
                    "pressure_type": PRESSURE_TYPE,
                    "pressure_index": resource_pressure["msg_level_index"],
                    "pressure_level": resource_pressure["component_pressure_level"],
                    "content": compose(eval_text, resource_pressure),
                },
            ]
        )
    return result
