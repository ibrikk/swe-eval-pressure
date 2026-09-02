#!/usr/bin/env python3
"""Read-only forensic diagnostic for ONE campaign shard attempt.

This module NEVER writes inside a run directory, never deletes, never renames
and never touches the accepted dataset. It writes exactly one artifact tree:

    provenance/diagnostics/<mode>-shard<N>-<stamp>/
        summary.json      machine-readable forensic record
        summary.md        operator-readable version
        inventory.json    tamper-evidence inventory of every run dir

`inventory.json` is the proof obligation for "nothing was overwritten": it
records, per trial directory, the file count, total byte size and the sha256 of
the two small evidence files (result.json, exception.txt). Re-running with
`verify --against <inventory>` re-derives it and reports any drift.

Why not hash whole trial trees: the corpus is ~4.7 GB and the interesting
tamper modes (a result replaced, an exception blanked, a trial deleted or
back-filled) are all caught by count + size + evidence-file hash.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from campaign import failures, lib


# --------------------------------------------------------------------------- #
# inventory (tamper evidence)
# --------------------------------------------------------------------------- #
def _sha_or_none(p: Path) -> str | None:
    return lib.sha256_file(p) if p.is_file() else None


def inventory_run_dir(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    trials = {}
    for td in sorted(p for p in run_dir.rglob("ea-*") if p.is_dir()):
        files = [f for f in td.rglob("*") if f.is_file()]
        trials[td.name] = {
            "rel": td.relative_to(run_dir).as_posix(),
            "file_count": len(files),
            "total_bytes": sum(f.stat().st_size for f in files),
            "result_json_sha256": _sha_or_none(td / "result.json"),
            "exception_txt_sha256": _sha_or_none(td / "exception.txt"),
        }
    return {"run_dir": run_dir.name, "trial_count": len(trials), "trials": trials}


def build_inventory(mode_root: Path) -> dict:
    dirs = sorted(d for d in Path(mode_root).iterdir() if d.is_dir())
    return {
        "generated_at": lib.now_iso(),
        "root": str(Path(mode_root).resolve().relative_to(lib.PROJECT_ROOT)),
        "run_dirs": [inventory_run_dir(d) for d in dirs],
    }


def diff_inventory(old: dict, new: dict) -> list[str]:
    """Return human-readable drift lines. Empty list == byte-stable."""
    drift: list[str] = []
    o = {r["run_dir"]: r for r in old.get("run_dirs", [])}
    n = {r["run_dir"]: r for r in new.get("run_dirs", [])}
    for name in sorted(set(o) - set(n)):
        drift.append(f"RUN DIR DISAPPEARED: {name}")
    for name in sorted(set(n) - set(o)):
        drift.append(f"RUN DIR APPEARED: {name}")
    for name in sorted(set(o) & set(n)):
        ot, nt = o[name]["trials"], n[name]["trials"]
        for t in sorted(set(ot) - set(nt)):
            drift.append(f"TRIAL DELETED: {name}/{t}")
        for t in sorted(set(nt) - set(ot)):
            drift.append(f"TRIAL ADDED: {name}/{t}")
        for t in sorted(set(ot) & set(nt)):
            for field in ("file_count", "total_bytes",
                          "result_json_sha256", "exception_txt_sha256"):
                if ot[t][field] != nt[t][field]:
                    drift.append(
                        f"TRIAL MODIFIED: {name}/{t} {field}: "
                        f"{ot[t][field]!r} -> {nt[t][field]!r}")
    return drift


# --------------------------------------------------------------------------- #
# forensic summary
# --------------------------------------------------------------------------- #
def _controller_log(paths, mode: str, shard: int) -> list[dict]:
    p = paths["logs"] / f"controller-{mode}-shard{shard}.jsonl"
    if not p.is_file():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _runner_events(paths) -> list[dict]:
    p = paths["logs"] / "runner.jsonl"
    if not p.is_file():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def diagnose(mode: str, shard: int) -> dict:
    paths = lib.campaign_paths()
    mode_root = paths[mode]
    run_dirs = sorted(d for d in mode_root.iterdir() if d.is_dir()) if mode_root.is_dir() else []

    per_profile = defaultdict(lambda: {"complete": 0, "failed": 0, "by_status": defaultdict(int)})
    failed_trials = []
    for rd in run_dirs:
        profile = next((p for p in lib.PROFILES if f"-{p}-" in rd.name), "?")
        for t in lib.scan_run_dir(rd):
            bucket = per_profile[profile]
            bucket["by_status"][t.status] += 1
            if t.status == lib.STATUS_COMPLETE:
                bucket["complete"] += 1
                continue
            bucket["failed"] += 1
            cand = list(rd.rglob(t.trial_dir))
            cls = (failures.classify_trial_dir(cand[0]) if cand else
                   failures.Classification(failures.PERMANENT, "trial directory not found",
                                           False, False))
            failed_trials.append({
                "profile": profile,
                "run_dir": rd.name,
                "trial_dir": t.trial_dir,
                "base_task_id": t.base_task_id,
                "arm": t.arm,
                "scan_status": t.status,
                "model_name": t.model_name,
                "steps": t.steps,
                "completion_tokens": t.completion_tokens,
                "classification": cls.as_dict(),
            })

    ledger = failures.RetryLedger(paths["provenance"] / "retries.jsonl")
    retries = [r.as_dict() for r in ledger.records()]

    status_rows = _controller_log(paths, mode, shard)
    last = status_rows[-1] if status_rows else {}
    throttles = [
        {"ts": r["ts"], "observed_tpm": (r.get("decision") or {}).get("observed_tpm"),
         "allocation": (r.get("decision") or {}).get("allocation"),
         "reason": (r.get("decision") or {}).get("reason")}
        for r in status_rows if (r.get("decision") or {}).get("throttled")
    ]
    tpm_series = [{"ts": r["ts"], "tpm": r["aggregate_metered_tpm"]}
                  for r in status_rows if r.get("aggregate_metered_tpm")]

    accepted_doc = lib.jload(paths["provenance"] / "accepted_runs.json") or {}
    accepted = accepted_doc.get("accepted", [])
    accepted_here = [a for a in accepted
                     if a.get("mode") == mode and a.get("shard_index") == shard]

    return {
        "campaign_id": lib.CAMPAIGN_ID,
        "mode": mode,
        "shard": shard,
        "generated_at": lib.now_iso(),
        "verdict": "FAILED",
        "attempt_ids": sorted({a.get("attempt_id") for a in accepted_here if a.get("attempt_id")}),
        "attempts_ledger_exists": (paths["provenance"] / "attempts.jsonl").is_file(),
        "run_dirs": [str(d.resolve().relative_to(lib.PROJECT_ROOT)) for d in run_dirs],
        "run_dir_count": len(run_dirs),
        "trials_on_disk": sum(v["complete"] + v["failed"] for v in per_profile.values()),
        "by_profile": {
            p: {"complete": v["complete"], "failed": v["failed"],
                "by_status": dict(v["by_status"])}
            for p, v in sorted(per_profile.items())},
        "controller_last_status": {
            k: last.get(k) for k in
            ("ts", "elapsed_sec", "completed_trials", "failed_trials", "queued_trials",
             "retries", "spend_usd", "remaining_budget_usd", "real_429_total")},
        "failed_trials": failed_trials,
        "retry_ledger": retries,
        "retry_ledger_open": [r for r in retries if r["accepted_status"] == "pending"],
        "retry_trial_dirs_on_disk": [
            r["retry_trial_id"] for r in retries
            if any(list(rd.rglob(r["retry_trial_id"])) for rd in run_dirs)],
        "tpm_series_nonzero": tpm_series,
        "tpm_samples_total": len(status_rows),
        "throttle_events": throttles,
        "accepted_cells_for_this_shard": accepted_here,
        "runner_events": _runner_events(paths),
    }


def render_md(d: dict, drift: list[str] | None = None) -> str:
    L = [f"# Forensic diagnostic - {d['campaign_id']} {d['mode']} shard {d['shard']}",
         "", f"Generated {d['generated_at']}  ", f"**Verdict: {d['verdict']}**", "",
         "This is a READ-ONLY diagnostic. No run directory, trajectory, manifest or",
         "dataset was modified, moved or deleted to produce it.", "",
         "## Accepted cells from this attempt", ""]
    if d["accepted_cells_for_this_shard"]:
        L += [f"- **{len(d['accepted_cells_for_this_shard'])} ACCEPTED CELL(S) PRESENT**"]
    else:
        L += ["- **NONE.** `provenance/accepted_runs.json` "
              f"{'exists but lists no cell for this shard' if d['attempt_ids'] else 'does not exist'}.",
              "- No trajectory from this attempt is reachable as accepted Campaign V2 data."]
    L += ["", "## Run directories produced", ""]
    for r in d["run_dirs"]:
        L.append(f"- `{r}`")
    L += ["", "## Trials by profile (on disk)", "",
          "| profile | complete | failed | statuses |", "|---|---|---|---|"]
    for p, v in d["by_profile"].items():
        L.append(f"| {p} | {v['complete']} | {v['failed']} | "
                 f"{', '.join(f'{k}={n}' for k, n in sorted(v['by_status'].items()))} |")
    L += ["", "## Failed trials", "",
          "| profile | trial | base task | arm | class | retryable | evidence |",
          "|---|---|---|---|---|---|---|"]
    for f in d["failed_trials"]:
        c = f["classification"]
        L.append(f"| {f['profile']} | `{f['trial_dir']}` | {f['base_task_id']} | {f['arm']} | "
                 f"{c['failure_class']} | {'yes' if c['retryable'] else 'NO'} | "
                 f"`{c['evidence'] or '-'}` |")
    L += ["", "## Retry ledger", "",
          f"- {len(d['retry_ledger'])} record(s); "
          f"{len(d['retry_ledger_open'])} still `pending` (never closed)",
          f"- retry trial dirs actually on disk: "
          f"{d['retry_trial_dirs_on_disk'] or 'NONE - no retry trajectory ever began'}", ""]
    L += ["## Live TPM series (non-zero samples only)", "",
          f"- {len(d['tpm_series_nonzero'])} non-zero of {d['tpm_samples_total']} status samples", ""]
    for s in d["tpm_series_nonzero"]:
        L.append(f"  - {s['ts']}  {s['tpm']:,.0f}")
    L += ["", "## Throttle events driven by that series", ""]
    for t in d["throttle_events"]:
        L.append(f"- {t['ts']} observed {t['observed_tpm']:,.0f} -> {t['allocation']}")
    if drift is not None:
        L += ["", "## Tamper check", ""]
        L += ["- inventory re-verified: **NO DRIFT**"] if not drift else \
             [f"- **DRIFT**: {x}" for x in drift]
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #
def cmd_run(args) -> int:
    paths = lib.campaign_paths()
    out = paths["provenance"] / "diagnostics" / f"{args.mode}-shard{args.shard}-{args.stamp}"
    lib.assert_campaign_path(out, "diagnostic output")
    out.mkdir(parents=True, exist_ok=True)

    d = diagnose(args.mode, args.shard)
    inv = build_inventory(paths[args.mode])
    (out / "summary.json").write_text(json.dumps(d, indent=2) + "\n")
    (out / "inventory.json").write_text(json.dumps(inv, indent=2) + "\n")
    (out / "summary.md").write_text(render_md(d))
    print(f"wrote {out.relative_to(lib.PROJECT_ROOT)}/  "
          f"({d['run_dir_count']} run dirs, {d['trials_on_disk']} trials, "
          f"{len(d['failed_trials'])} failures)")
    return 0


def cmd_verify(args) -> int:
    paths = lib.campaign_paths()
    old = json.loads(Path(args.against).read_text())
    new = build_inventory(paths[args.mode])
    drift = diff_inventory(old, new)
    if drift:
        print(f"DRIFT DETECTED ({len(drift)} item(s)):")
        for x in drift[:50]:
            print(f"  {x}")
        return 1
    n = sum(r["trial_count"] for r in new["run_dirs"])
    print(f"NO DRIFT: {len(new['run_dirs'])} run dirs / {n} trial dirs byte-stable")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="write the read-only diagnostic bundle")
    r.add_argument("--mode", required=True, choices=lib.MODES)
    r.add_argument("--shard", required=True, type=int, choices=lib.SHARD_INDICES)
    r.add_argument("--stamp", default="")
    r.set_defaults(fn=cmd_run)
    v = sub.add_parser("verify", help="re-derive the inventory and report drift")
    v.add_argument("--mode", required=True, choices=lib.MODES)
    v.add_argument("--against", required=True)
    v.set_defaults(fn=cmd_verify)
    args = ap.parse_args()
    if getattr(args, "stamp", None) == "":
        args.stamp = lib.now_iso().replace(":", "").replace("-", "")[:15]
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
