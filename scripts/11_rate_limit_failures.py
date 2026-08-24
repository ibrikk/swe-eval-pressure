#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def safe_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def task_name(result: dict, path: Path) -> str:
    name = result.get("task_name")
    if name:
        return str(name)
    try:
        return str(result["config"]["task"]["path"]).rstrip("/").split("/")[-1]
    except Exception:
        return path.parent.name


def is_rate_limit(result: dict) -> bool:
    exc = result.get("exception_info")
    if not exc:
        return False
    if isinstance(exc, dict):
        text = " ".join(
            str(exc.get(k) or "") for k in ("exception_type", "exception_message", "traceback")
        ).lower()
    else:
        text = str(exc).lower()
    markers = (
        "ratelimit",
        "rate limit",
        "rate_limit",
        "http 429",
        "status 429",
        "too many requests",
        "token limit",
        "token ceiling",
    )
    return any(marker in text for marker in markers)


def main() -> None:
    ap = argparse.ArgumentParser(description="Count latest Harbor results that are rate-limit failures.")
    ap.add_argument("job_dir", type=Path)
    args = ap.parse_args()
    job = args.job_dir
    latest: dict[str, tuple[float, dict]] = {}
    for path in job.rglob("result.json"):
        result = safe_json(path)
        if not result:
            continue
        name = task_name(result, path)
        try:
            stamp = path.stat().st_mtime
        except OSError:
            stamp = 0.0
        old = latest.get(name)
        if old is None or stamp >= old[0]:
            latest[name] = (stamp, result)
    print(sum(1 for _, result in latest.values() if is_rate_limit(result)))


if __name__ == "__main__":
    main()
