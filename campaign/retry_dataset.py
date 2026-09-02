#!/usr/bin/env python3
"""Build a campaign-local dataset containing EXACTLY ONE task/arm cell.

Why this module exists
----------------------
On 2026-09-02 the FULL shard-1 controller crashed with

    SystemExit: --base-task-count must be >= 1

because `_handle_failure` queued a retry as a normal batch with
`base_count=0`, as a sentinel meaning "one trial, not one base task". The
slicer -- correctly -- refused.

The naive fix is `base_count=1`. That is WRONG: a base task carries all ten
delivery-channel arms, so retrying one failed arm would re-run nine healthy
trials, duplicate nine trajectories, and cost ~9x the budget it should. Under
the campaign's own rules those nine siblings already have accepted results; a
second run of them is at best waste and at worst a silent replacement.

So a retry gets its own dataset shape, built here rather than by
`scripts/05_slice_dataset.py`, which only knows how to slice by base task:

  * exactly one task directory, byte-identical to the original;
  * exactly one manifest entry, with `content_hash` carried over verbatim;
  * mode / profile / shard / campaign provenance copied from the source;
  * a `retry_provenance` block naming the original trial, the retry id, the
    retry number, the failure class and the original run directory.

`assert_single_trial_dataset` re-checks all of that from disk. The controller
calls it immediately before launching Harbor, so a malformed retry dataset can
never reach a runner.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from campaign import lib

RETRY_SCHEMA = "campaign-retry-dataset/1"
REPAIR_SCHEMA = "campaign-repair-dataset/1"


def task_dir_for_trial(trial_dir_name: str) -> str:
    """'ea-10d4b434-eval-src__GAFUntP' -> 'ea-10d4b434-eval-src'.

    Harbor appends '__<nonce>' to make each trial directory unique. The task
    directory in the dataset is the name without that nonce.
    """
    return trial_dir_name.split("__", 1)[0]


def _link_or_copy(src: str, dst: str) -> str:
    try:
        os.link(src, dst)
        return dst
    except OSError:
        return shutil.copy2(src, dst)


def tree_digest(root: Path) -> str:
    """Order-independent digest over (relative path, content) of a tree."""
    h = hashlib.sha256()
    for p in sorted(Path(root).rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode())
            h.update(b"\0")
            h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def find_task_entry(manifest: dict, task_dir: str) -> dict:
    hits = [t for t in manifest.get("tasks", []) if str(t.get("directory")) == task_dir]
    if not hits:
        raise KeyError(f"task directory {task_dir!r} is not in the source manifest")
    if len(hits) > 1:
        raise KeyError(f"task directory {task_dir!r} is ambiguous in the source manifest")
    return hits[0]


def build_repair_dataset(*, source_dataset: Path, output: Path,
                         task_dirs, label: str = "",
                         provenance: dict | None = None) -> Path:
    """Materialise a dataset holding EXACTLY the listed task/arm cells.

    This is the general form; `build_retry_dataset` is the single-cell case.
    Bulk repair of an interrupted shard uses it to re-run 100 scattered cells
    in one Harbor invocation instead of 100, without ever widening the set:
    a sibling arm appears only if that arm is itself in `task_dirs`. Nothing
    here can grow the work, because the output is asserted against the exact
    requested set afterwards.
    """
    source = Path(source_dataset).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    lib.assert_campaign_path(output, "repair dataset output")

    wanted = list(dict.fromkeys(str(t) for t in task_dirs))
    if not wanted:
        raise SystemExit("repair dataset needs at least one task directory")

    manifest_path = source / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing source manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    entries = [find_task_entry(manifest, t) for t in wanted]
    for t in wanted:
        if not (source / t).is_dir():
            raise SystemExit(f"missing source task directory: {source / t}")

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    digests = {}
    for t in wanted:
        shutil.copytree(source / t, output / t, copy_function=_link_or_copy)
        before, after = tree_digest(source / t), tree_digest(output / t)
        if before != after:
            raise SystemExit(f"repair task content changed during copy: {t}")
        digests[t] = after

    base_ids = list(dict.fromkeys(str(e.get("base_task_id", "")) for e in entries))
    out = dict(manifest)
    out["tasks"] = [dict(e) for e in entries]
    out["base_task_count"] = len(base_ids)
    out.pop("adaptive_batch", None)
    out["repair_provenance"] = {
        "schema": REPAIR_SCHEMA,
        "label": label,
        "cell_count": len(wanted),
        "task_directories": wanted,
        "base_task_ids": base_ids,
        "task_tree_sha256": digests,
        "campaign_id": manifest.get("campaign_id", lib.CAMPAIGN_ID),
        "mode": manifest.get("mode", ""),
        "profile": manifest.get("profile", ""),
        "shard": manifest.get("shard"),
        "created_at": lib.now_iso(),
    }
    if provenance:
        out["repair_provenance"].update(provenance)
    if not out.get("source_dataset"):
        out["source_dataset"] = str(source)

    (output / "manifest.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    assert_exact_cells(output, wanted)
    return output


def assert_exact_cells(path: Path, wanted) -> dict:
    """Prove a dataset on disk holds exactly `wanted`, no more and no fewer."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    want = sorted(dict.fromkeys(str(t) for t in wanted))
    on_disk = sorted(d.name for d in path.iterdir() if d.is_dir())
    in_manifest = sorted(str(t.get("directory")) for t in manifest.get("tasks", []))
    if on_disk != want:
        extra = [d for d in on_disk if d not in want]
        missing = [d for d in want if d not in on_disk]
        raise SystemExit(f"{path}: dataset directories do not match the requested "
                         f"cells; unexpected={extra} missing={missing}")
    if in_manifest != want:
        raise SystemExit(f"{path}: manifest tasks {in_manifest} do not match the "
                         f"requested cells {want}")
    return manifest


def build_retry_dataset(*, source_dataset: Path, output: Path,
                        original_trial_id: str, retry_trial_id: str,
                        retry_number: int, failure_class: str,
                        failure_reason: str = "", original_run_dir: str = "",
                        cell: str = "") -> Path:
    """Materialise a one-task/one-arm dataset. Returns `output`.

    `source_dataset` is any dataset whose manifest contains the failed cell --
    normally the shard dataset the original batch was sliced from.
    """
    source = Path(source_dataset).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    lib.assert_campaign_path(output, "retry dataset output")

    manifest_path = source / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing source manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    task_dir = task_dir_for_trial(original_trial_id)
    entry = find_task_entry(manifest, task_dir)
    src_task = source / task_dir
    if not src_task.is_dir():
        raise SystemExit(f"missing source task directory: {src_task}")

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    # copy_function=link_or_copy: hardlinks where possible, so the retry reads
    # byte-identical task content and never mutates the original.
    shutil.copytree(src_task, output / task_dir, copy_function=_link_or_copy)

    before, after = tree_digest(src_task), tree_digest(output / task_dir)
    if before != after:
        raise SystemExit(f"retry task content changed during copy: {before} != {after}")

    out = dict(manifest)
    out["tasks"] = [dict(entry)]          # content_hash carried over verbatim
    out["base_task_count"] = 1
    out["variants_per_task"] = 1
    out.pop("adaptive_batch", None)       # this is not a base-task slice
    out["retry_provenance"] = {
        "schema": RETRY_SCHEMA,
        "original_trial_id": original_trial_id,
        "retry_trial_id": retry_trial_id,
        "retry_number": int(retry_number),
        "failure_class": failure_class,
        "failure_reason": failure_reason,
        "original_run_dir": str(original_run_dir),
        "original_task_directory": task_dir,
        "original_base_task_id": str(entry.get("base_task_id", "")),
        "original_content_hash": str(entry.get("content_hash", "")),
        "task_tree_sha256": after,
        "cell": cell,
        "mode": manifest.get("mode", ""),
        "profile": manifest.get("profile", ""),
        "shard": manifest.get("shard"),
        "campaign_id": manifest.get("campaign_id", lib.CAMPAIGN_ID),
        "created_at": lib.now_iso(),
    }
    if not out.get("source_dataset"):
        out["source_dataset"] = str(source)

    (output / "manifest.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    assert_single_trial_dataset(output)
    return output


def assert_single_trial_dataset(path: Path) -> dict:
    """Re-read from disk and prove the dataset is exactly one task/arm cell.

    Raises SystemExit on any violation. The sibling-arm check is the important
    one: every other arm of the same base task already has a result, and
    re-running one would be an unrequested duplicate.
    """
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))

    rp = manifest.get("retry_provenance")
    if not isinstance(rp, dict) or rp.get("schema") != RETRY_SCHEMA:
        raise SystemExit(f"{path}: not a retry dataset (missing retry_provenance)")
    for key in ("original_trial_id", "retry_trial_id", "retry_number",
                "failure_class", "original_task_directory"):
        if rp.get(key) in (None, ""):
            raise SystemExit(f"{path}: retry_provenance.{key} is empty")

    tasks = manifest.get("tasks", [])
    if len(tasks) != 1:
        raise SystemExit(f"{path}: retry dataset must hold exactly 1 task entry, "
                         f"found {len(tasks)}")
    if manifest.get("base_task_count") != 1:
        raise SystemExit(f"{path}: base_task_count must be 1, "
                         f"found {manifest.get('base_task_count')!r}")
    if "adaptive_batch" in manifest:
        raise SystemExit(f"{path}: retry dataset must not carry adaptive_batch "
                         "(it is not a base-task slice)")

    want = str(rp["original_task_directory"])
    if str(tasks[0].get("directory")) != want:
        raise SystemExit(f"{path}: manifest task {tasks[0].get('directory')!r} does "
                         f"not match retried cell {want!r}")

    on_disk = sorted(d.name for d in path.iterdir() if d.is_dir())
    if on_disk != [want]:
        extra = [d for d in on_disk if d != want]
        raise SystemExit(f"{path}: retry dataset must contain exactly the retried "
                         f"cell {want!r}; sibling arms present: {extra}")

    if rp.get("original_content_hash") != str(tasks[0].get("content_hash", "")):
        raise SystemExit(f"{path}: content_hash was not preserved from the original")

    digest = tree_digest(path / want)
    if rp.get("task_tree_sha256") and rp["task_tree_sha256"] != digest:
        raise SystemExit(f"{path}: task content changed after the dataset was built")
    return manifest


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify", help="assert a retry dataset is a single cell")
    v.add_argument("path", type=Path)
    a = ap.parse_args()
    if a.cmd == "verify":
        m = assert_single_trial_dataset(a.path)
        print(f"OK single-cell retry dataset: "
              f"{m['retry_provenance']['original_task_directory']} "
              f"(retry {m['retry_provenance']['retry_number']})")
