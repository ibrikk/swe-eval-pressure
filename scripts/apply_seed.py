#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


def insert_source(path: Path, prefix: str, text: str) -> None:
    if not path.is_file():
        raise SystemExit(f"source target missing: {path}")
    original = path.read_text(encoding="utf-8", errors="surrogateescape")
    lines = original.splitlines(keepends=True)
    index = 0
    if lines and lines[0].startswith("#!"):
        index = 1
    if index < len(lines) and "coding" in lines[index] and lines[index].lstrip().startswith("#"):
        index += 1
    block = "".join(
        f"{prefix} {line}\n" if line else f"{prefix}\n" for line in text.splitlines()
    ) + "\n"
    path.write_text(
        "".join(lines[:index]) + block + "".join(lines[index:]),
        encoding="utf-8",
        errors="surrogateescape",
    )


def append_context(path: Path, text: str) -> None:
    """Append context while preserving any pre-existing repository instructions."""
    original = ""
    if path.exists():
        if not path.is_file():
            raise SystemExit(f"context target is not a file: {path}")
        original = path.read_text(encoding="utf-8", errors="surrogateescape")
    prefix = original
    if prefix and not prefix.endswith("\n"):
        prefix += "\n"
    if prefix:
        prefix += "\n"
    path.write_text(
        prefix + text.strip() + "\n",
        encoding="utf-8",
        errors="surrogateescape",
    )


def main() -> None:
    seed_path = Path(sys.argv[1])
    seed = json.loads(seed_path.read_text(encoding="utf-8"))
    workspace = Path(seed.get("workspace_root", "/code"))
    channel = seed["channel"]
    content = seed.get("content", "").strip()
    if not workspace.is_dir():
        raise SystemExit(f"workspace root missing: {workspace}")
    if seed["condition"] == "clean":
        return
    if not content:
        raise SystemExit("seeded condition has empty content")
    if channel == "source":
        insert_source(
            workspace / seed["source_target"],
            seed["source_comment_prefix"],
            content,
        )
    elif channel == "root":
        append_context(workspace / "BENCHMARK_CONTEXT.md", content)
    elif channel == "scaffold":
        instruction_file = seed.get("instruction_file")
        if instruction_file not in {"CLAUDE.md", "AGENTS.md"}:
            raise SystemExit(f"unexpected scaffold instruction file: {instruction_file}")
        append_context(workspace / instruction_file, content)
    else:
        raise SystemExit(f"unknown channel: {channel}")


if __name__ == "__main__":
    main()
