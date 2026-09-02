"""Source-channel seed targets: pick a file that exists BEFORE the gold patch.

WHY THIS MODULE EXISTS
----------------------
The `source` delivery channel seeds its evaluation cue as a comment inside a
real repository file. The file is chosen from the task's gold patch, on the
reasoning that a file the reference solution touches is a file the agent will
read. `scripts/01_inventory.py` picked it off the `b/` side of
`diff --git a/<old> b/<new>`.

For a MODIFIED file the two sides are equal, so that worked for 68 of the 70
base tasks. For a RENAMED file the `b/` side is where the file lands AFTER the
patch, and nothing is there beforehand. `apply_seed.py` then raises

    SystemExit: source target missing: /workspace/<b-side path>

inside the Dockerfile's `RUN`, which Modal surfaces as `ImageBuildError`. On
2026-09-02 that took out 12 FULL shard-1 cells: base task 10d4b434
(Automattic/wp-calypso), whose gold patch opens with

    diff --git a/client/dashboard/sites/overview-card/card.stories.tsx \
             b/client/dashboard/components/overview-card/card.stories.tsx
    similarity index 100%
    rename from client/dashboard/sites/overview-card/card.stories.tsx
    rename to   client/dashboard/components/overview-card/card.stories.tsx

Confirmed against the pinned base image: the `sites/` path exists, the
`components/` path does not. Because the image is content-addressed the same
build failed identically for all four profiles -- one image per source arm,
four cells each.

The same latent defect sits in FULL shard 3: base task 8265afba
(grafana/grafana) selects `createNodeGraphFrames.ts`, which the patch renames
from `utils.ts`. Verified the same way against its own base image. Left
unfixed it would cost another 12 cells.

WHAT "CORRECT" MEANS HERE
-------------------------
The seed must land in a file that exists in the pre-patch tree, because that is
the tree the image is built from. So the target is the file's PRE-patch path:
`rename from` when the section is a rename, the `a/` side otherwise, and
nothing at all when the patch creates the file. Renaming does not change which
file carries the cue -- `sites/overview-card/card.stories.tsx` and
`components/overview-card/card.stories.tsx` are the same file at two moments --
so the treatment is unchanged. Only the path is corrected.

RESOURCE mode is unaffected: its arms are `clean-n`, `eval-scaf` and
`eval-resource-scaf`, none of which uses the source channel.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from campaign import lib

# Extension -> line-comment prefix. Must stay identical to the table in
# scripts/01_inventory.py, which imports this module.
SUPPORTED = {'.py': '#', '.yml': '#', '.yaml': '#', '.toml': '#', '.am': '#',
             '.sh': '#', '.rb': '#',
             '.go': '//', '.ts': '//', '.tsx': '//', '.js': '//', '.jsx': '//',
             '.c': '//', '.cc': '//', '.cpp': '//', '.h': '//', '.hpp': '//',
             '.cue': '//', '.rs': '//', '.java': '//'}
TEST_RE = re.compile(r'(^|/)(test|tests|spec|specs)(/|$)', re.I)

_RENAME_FROM = re.compile(r'(?m)^rename from (.+)$')
_NEW_FILE = re.compile(r'(?m)^new file mode ')


def patch_sections(text: str) -> list[tuple[str, str, str]]:
    """Split a unified diff into (a_path, b_path, section) triples."""
    out = []
    for chunk in re.split(r'(?=^diff --git )', text, flags=re.M):
        m = re.match(r'diff --git a/(.+?) b/(.+?)\n', chunk)
        if m:
            out.append((m.group(1), m.group(2), chunk))
    return out


def pre_patch_path(a_path: str, b_path: str, section: str) -> str | None:
    """Where this section's file lives BEFORE the patch, or None if created.

    Scans the whole section rather than a leading byte window: the old
    `'\\nnew file mode ' in section[:500]` test silently depended on how long
    the preceding paths happened to be.
    """
    m = _RENAME_FROM.search(section)
    if m:
        return m.group(1).strip()
    if _NEW_FILE.search(section) or '\n--- /dev/null\n' in section:
        return None
    return a_path


def candidates(gold_patch_text: str) -> list[str]:
    """Pre-patch paths eligible to carry a seeded comment, in patch order."""
    out = []
    for a_path, b_path, section in patch_sections(gold_patch_text):
        pre = pre_patch_path(a_path, b_path, section)
        if pre is None:
            continue
        ext = Path(pre).suffix.lower()
        if ext in SUPPORTED and not TEST_RE.search(pre):
            out.append(pre)
    return out


def select(gold_patch_text: str) -> tuple[str, str]:
    """(source_target, comment_prefix) for a gold patch. Raises if none fits."""
    cands = candidates(gold_patch_text)
    if not cands:
        raise ValueError('no safe existing non-test source path in gold patch')
    path = cands[0]
    return path, SUPPORTED[Path(path).suffix.lower()]


def all_patch_paths(gold_patch_text: str) -> list[str]:
    """Every path the patch touches, post-patch side, in order, deduplicated."""
    out = []
    for _a, b_path, _s in patch_sections(gold_patch_text):
        if b_path not in out:
            out.append(b_path)
    return out


# --------------------------------------------------------------------------- #
# auditing prepared task directories
# --------------------------------------------------------------------------- #
def audit_task_dir(task_dir: Path) -> dict | None:
    """Return a finding dict if this task directory's seed target is unusable.

    Purely static: reads the task's own gold patch and seed. Builds no image,
    pulls nothing, invokes no model. A source-channel seed whose target is not
    a pre-patch path of its own gold patch WILL fail the image build, so this
    is a hard defect and not a warning.
    """
    task_dir = Path(task_dir)
    seed_path = task_dir / "environment/benchmark_seed/seed.json"
    patch_path = task_dir / "solution/gold.patch"
    if not seed_path.is_file():
        return None
    seed = json.loads(seed_path.read_text(encoding="utf-8"))
    if seed.get("channel") != "source":
        return None
    target = seed.get("source_target") or ""
    if not target:
        return {"task_dir": task_dir.name, "path": str(task_dir),
                "problem": "source channel with no source_target",
                "current": target, "corrected": None}
    if not patch_path.is_file():
        return {"task_dir": task_dir.name, "path": str(task_dir),
                "problem": "no gold patch to validate source_target against",
                "current": target, "corrected": None}
    text = patch_path.read_text(encoding="utf-8", errors="replace")
    pre = set(candidates(text))
    if target in pre:
        return None
    try:
        corrected, prefix = select(text)
    except ValueError:
        corrected, prefix = None, None
    renamed = any(
        b == target and pre_patch_path(a, b, s) not in (None, target)
        for a, b, s in patch_sections(text))
    problem = ("source_target is the post-rename destination and does not "
               "exist in the pre-patch tree") if renamed else (
               "source_target is not a pre-patch path of the gold patch")
    return {"task_dir": task_dir.name, "path": str(task_dir),
            "problem": problem, "current": target, "corrected": corrected,
            "comment_prefix": prefix,
            "workspace_root": seed.get("workspace_root", "")}


def audit_shard(mode: str, shard: int, *, profiles=None, paths=None) -> dict:
    """Static source-target audit of every prepared task dir in one shard."""
    paths = paths or lib.campaign_paths()
    profiles = list(profiles or lib.PROFILES)
    findings, checked = [], 0
    for profile in profiles:
        cell = lib.Cell(mode, profile, shard)
        shard_dir = paths["datasets"] / "_shards" / mode / profile / cell.shard_label
        if not shard_dir.is_dir():
            continue
        for task_dir in sorted(shard_dir.glob("ea-*")):
            checked += 1
            f = audit_task_dir(task_dir)
            if f:
                f["mode"], f["profile"], f["shard"] = mode, profile, shard
                findings.append(f)
    return {"mode": mode, "shard": shard, "checked": checked,
            "ok": not findings, "findings": findings}


def audit_all(*, paths=None) -> dict:
    """Every mode/shard the campaign will ever launch."""
    out = {"ok": True, "checked": 0, "by_shard": {}, "findings": []}
    for mode in lib.MODES:
        for shard in lib.SHARD_INDICES:
            r = audit_shard(mode, shard, paths=paths)
            out["checked"] += r["checked"]
            out["by_shard"][f"{mode}/s{shard}"] = len(r["findings"])
            out["findings"].extend(r["findings"])
    out["ok"] = not out["findings"]
    return out


# --------------------------------------------------------------------------- #
# repairing prepared task directories
# --------------------------------------------------------------------------- #
def repair_shard(mode: str, shard: int, *, frozen: set[str] | None = None,
                 profiles=None, paths=None, apply: bool = False) -> dict:
    """Correct the seed target of broken source-channel task dirs.

    FAILS CLOSED on a cell that already holds a valid completed trajectory.
    Rewriting a task definition under a trajectory that ran against the old one
    would silently break that trajectory's provenance, so pass the audit's
    frozen cell keys and this refuses instead.

    Writes two places, which `scripts/04_validate.py` cross-checks against each
    other: the task's own `seed.json`, and its entry in the shard manifest.
    Nothing else in the task tree mentions the target.
    """
    paths = paths or lib.campaign_paths()
    profiles = list(profiles or lib.PROFILES)
    frozen = frozen or set()
    audit = audit_shard(mode, shard, profiles=profiles, paths=paths)
    changed, refused = [], []

    by_profile: dict[str, list[dict]] = {}
    for f in audit["findings"]:
        by_profile.setdefault(f["profile"], []).append(f)

    for profile, items in sorted(by_profile.items()):
        cell = lib.Cell(mode, profile, shard)
        shard_dir = paths["datasets"] / "_shards" / mode / profile / cell.shard_label
        lib.assert_campaign_path(shard_dir, "shard dataset directory")
        manifest_path = shard_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = {t["directory"]: t for t in manifest["tasks"]}
        dirty = False
        for f in items:
            if f["corrected"] is None:
                refused.append({**f, "refusal": "no usable pre-patch path in "
                                                "the gold patch"})
                continue
            entry = entries.get(f["task_dir"])
            if entry is None:
                refused.append({**f, "refusal": "task dir absent from manifest"})
                continue
            key = (f"{lib.CAMPAIGN_ID}/{mode}/{profile}/s{shard}/"
                   f"{entry['base_task_id']}/{f['task_dir'].split('-', 2)[2]}")
            if key in frozen:
                refused.append({**f, "cell_key": key,
                                "refusal": "cell holds a valid completed "
                                           "trajectory -- not rewriting the "
                                           "task it ran against"})
                continue
            rec = {**f, "cell_key": key, "applied": bool(apply)}
            changed.append(rec)
            if not apply:
                continue
            seed_path = Path(f["path"]) / "environment/benchmark_seed/seed.json"
            seed = json.loads(seed_path.read_text(encoding="utf-8"))
            seed["source_target"] = f["corrected"]
            if f.get("comment_prefix"):
                seed["source_comment_prefix"] = f["comment_prefix"]
            seed_path.write_text(json.dumps(seed, indent=2, ensure_ascii=False)
                                 + "\n", encoding="utf-8")
            entry["source_target"] = f["corrected"]
            dirty = True
        if apply and dirty:
            manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8")

    return {"mode": mode, "shard": shard, "applied": bool(apply),
            "changed": changed, "refused": refused,
            "ok": not refused}


def sync_manifests(*, paths=None, apply: bool = False) -> dict:
    """Bring every derived manifest in line with the corrected seeds.

    `repair_shard` fixes the seed and the shard manifest the repair will run
    from. But the same task directory is also described by the staging manifest
    in `generated/<mode>/<profile>/`, its hardlinked twin under
    `datasets/<mode>/<profile>/`, and `generated/_shards/`. `scripts/04_validate.py`
    asserts manifest and seed agree field-for-field, so a manifest left behind
    turns a fixed task into a validation failure.

    Two trees are deliberately excluded:
      * `datasets/_batches/` -- the record of the 2026-09-02 attempt, which must
        keep describing what that attempt ran (see `restore_run_inputs`).
      * `generated/_adaptive/` -- August 2026 artifacts from a different
        campaign, out of scope and preserved as-is.
    """
    paths = paths or lib.campaign_paths()
    log = paths["provenance"] / "source_target_repairs.jsonl"
    if not log.is_file():
        return {"updated": [], "applied": bool(apply), "entries": 0, "ok": True}
    corrected: dict[str, str] = {}
    for line in log.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            corrected[rec["task_dir"]] = rec["corrected"]

    roots = [(lib.PROJECT_ROOT / "generated", ("_adaptive",)),
             (paths["datasets"], ("_batches",))]
    updated, n_entries = [], 0
    for root, skip in roots:
        if not root.is_dir():
            continue
        for manifest_path in sorted(root.rglob("manifest.json")):
            rel = manifest_path.relative_to(root).parts
            if rel and rel[0] in skip:
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            tasks = manifest.get("tasks")
            if not isinstance(tasks, list):
                continue
            hits = 0
            for entry in tasks:
                want = corrected.get(entry.get("directory", ""))
                if want and entry.get("source_target") != want:
                    hits += 1
                    if apply:
                        entry["source_target"] = want
            if not hits:
                continue
            n_entries += hits
            updated.append({"manifest": str(manifest_path), "entries": hits})
            if apply:
                manifest_path.write_text(
                    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return {"updated": updated, "entries": n_entries, "applied": bool(apply),
            "ok": True}


def restore_run_inputs(*, paths=None, apply: bool = False) -> dict:
    """Un-share the corrected seeds from datasets that a real attempt consumed.

    `campaign/prepare.py` hardlinks task files rather than copying them, so
    `generated/`, `datasets/<mode>/`, `datasets/_shards/` and `datasets/_batches/`
    can all be the SAME inode. Correcting a seed therefore rewrites every copy at
    once -- desirable for the definitions a future run will read, wrong for
    `_batches/`, which is the frozen record of what the 2026-09-02 attempt
    actually ran. Those directories must keep the values that produced the
    observed failure, or the provenance stops describing the run it documents.

    This breaks the link for `_batches/` copies and restores the pre-repair
    value. Nothing else under the campaign root is touched.
    """
    paths = paths or lib.campaign_paths()
    log = paths["provenance"] / "source_target_repairs.jsonl"
    if not log.is_file():
        return {"restored": [], "applied": bool(apply), "ok": True}
    inodes: dict[int, dict] = {}
    for line in log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        seed = Path(rec["path"]) / "environment/benchmark_seed/seed.json"
        if seed.is_file():
            inodes[seed.stat().st_ino] = rec

    batches = paths["datasets"] / "_batches"
    lib.assert_campaign_path(batches, "batch dataset root")
    restored = []
    for seed_path in sorted(batches.rglob("seed.json")):
        rec = inodes.get(seed_path.stat().st_ino)
        if rec is None:
            continue
        restored.append(str(seed_path))
        if not apply:
            continue
        seed = json.loads(seed_path.read_text(encoding="utf-8"))
        seed["source_target"] = rec["current"]
        tmp = seed_path.with_name("seed.json.restore")
        tmp.write_text(json.dumps(seed, indent=2, ensure_ascii=False) + "\n",
                       encoding="utf-8")
        tmp.replace(seed_path)  # replace, not truncate: this breaks the link
    return {"restored": restored, "applied": bool(apply), "ok": True}


def write_repair_record(result: dict, *, paths=None) -> Path:
    """Append what was corrected to the campaign's provenance, append-only."""
    paths = paths or lib.campaign_paths()
    out = paths["provenance"] / "source_target_repairs.jsonl"
    lib.assert_campaign_path(out.parent, "source target repair log")
    from datetime import datetime, timezone
    at = datetime.now(timezone.utc).isoformat()
    with out.open("a", encoding="utf-8") as fh:
        for rec in result["changed"]:
            fh.write(json.dumps({"at": at, "campaign_id": lib.CAMPAIGN_ID,
                                 **rec}, ensure_ascii=False) + "\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("audit", "repair", "restore-run-inputs", "sync-manifests"))
    ap.add_argument("--mode")
    ap.add_argument("--shard", type=int)
    ap.add_argument("--apply", action="store_true",
                    help="write the corrections (default: report only)")
    args = ap.parse_args()
    paths = lib.campaign_paths()

    if args.command == "audit":
        r = (audit_all(paths=paths) if args.mode is None
             else audit_shard(args.mode, args.shard, paths=paths))
        print(json.dumps(r, indent=2))
        sys.exit(0 if r["ok"] else 1)

    if args.command == "sync-manifests":
        r = sync_manifests(paths=paths, apply=args.apply)
        print(json.dumps(r, indent=2))
        sys.exit(0)

    if args.command == "restore-run-inputs":
        r = restore_run_inputs(paths=paths, apply=args.apply)
        print(json.dumps(r, indent=2))
        sys.exit(0)

    if args.mode is None or args.shard is None:
        ap.error("repair needs --mode and --shard")
    from campaign import cells
    frozen = set(cells.frozen_cells(cells.audit(args.mode, args.shard,
                                                paths=paths)))
    r = repair_shard(args.mode, args.shard, frozen=frozen, paths=paths,
                     apply=args.apply)
    if args.apply and r["changed"]:
        r["record"] = str(write_repair_record(r, paths=paths))
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["ok"] else 1)


if __name__ == "__main__":
    main()
