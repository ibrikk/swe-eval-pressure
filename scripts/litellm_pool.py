#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import threading
import time
from datetime import datetime, timezone


def parse_litellm_keys(
    raw_keys: str | None = None,
    fallback_key: str | None = None,
) -> list[str]:
    """Parse an arbitrary LiteLLM key pool, preserving order and removing duplicates.

    LITE_LLM_KEYS takes precedence when non-empty. It may be comma- or whitespace-
    separated. Legacy LITE_LLM_KEY remains a one-key fallback.
    """
    if raw_keys is None:
        raw_keys = os.getenv("LITE_LLM_KEYS", "")
    if fallback_key is None:
        fallback_key = os.getenv("LITE_LLM_KEY", "")
    raw = raw_keys.strip() or fallback_key.strip()
    if not raw:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in re.split(r"[,\s]+", raw):
        key = item.strip()
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def parse_reset_epoch(detail: str) -> float | None:
    """Recover the Scale gateway reset timestamp from an HTTP 429 body when present."""
    match = re.search(
        r"Limit resets at:\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) UTC",
        detail,
    )
    if not match:
        return None
    try:
        reset_at = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return None
    return reset_at.timestamp()


class RateLimitKeyPool:
    """Thread-safe round-robin key pool with per-key 429 cooldowns."""

    def __init__(self, keys: list[str]):
        if not keys:
            raise ValueError("LiteLLM key pool cannot be empty")
        self.keys = tuple(keys)
        self._lock = threading.Lock()
        self._next_index = 0
        self._cooldown_until = [0.0 for _ in self.keys]

    @property
    def size(self) -> int:
        return len(self.keys)

    def acquire(self) -> str:
        """Return the next available key; wait only if every key is cooling down."""
        while True:
            wait_seconds = 0.0
            with self._lock:
                now = time.time()
                for offset in range(len(self.keys)):
                    idx = (self._next_index + offset) % len(self.keys)
                    if self._cooldown_until[idx] <= now:
                        self._next_index = (idx + 1) % len(self.keys)
                        return self.keys[idx]
                wait_seconds = max(0.05, min(self._cooldown_until) - now)
            time.sleep(wait_seconds)

    def mark_rate_limited(
        self,
        key: str,
        *,
        reset_epoch: float | None = None,
        fallback_seconds: float = 2.0,
    ) -> None:
        """Cool down only the key that received the 429."""
        try:
            idx = self.keys.index(key)
        except ValueError:
            return
        now = time.time()
        until = reset_epoch if reset_epoch is not None else now + max(1.0, fallback_seconds)
        until = max(now + 0.05, until)
        with self._lock:
            self._cooldown_until[idx] = max(self._cooldown_until[idx], until)
