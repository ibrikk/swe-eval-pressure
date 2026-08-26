from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_adaptive_slice_keeps_complete_base_task_families(tmp_path: Path) -> None:
    project = tmp_path / "repo"
    source = project / "generated" / "full" / "llama"
    source.mkdir(parents=True)
    tasks = []
    for base in ("a", "b", "c"):
        for variant in range(3):
            directory = f"task-{base}-{variant}"
            d = source / directory
            d.mkdir()
            (d / "task.toml").write_text("x\n", encoding="utf-8")
            tasks.append({"base_task_id": base, "directory": directory})
    (source / "manifest.json").write_text(
        json.dumps({"mode": "full", "profile": "llama", "tasks": tasks}) + "\n",
        encoding="utf-8",
    )

    script = Path(__file__).resolve().parents[1] / "scripts" / "05_slice_dataset.py"
    output = project / "generated" / "_adaptive" / "batch"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--project-root", str(project),
            "--source", str(source),
            "--output", str(output),
            "--start-index", "1",
            "--base-task-count", "1",
        ],
        check=True,
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["base_task_count"] == 1
    assert {x["base_task_id"] for x in manifest["tasks"]} == {"b"}
    assert len(manifest["tasks"]) == 3
    assert manifest["source_dataset"] == "generated/full/llama"
