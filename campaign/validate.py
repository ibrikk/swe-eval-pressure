#!/usr/bin/env python3
"""Campaign validation gate.

Refuses to certify anything that is incomplete, mixed, salvaged or drifted.
Exit code 0 only when the campaign is publishable as a single coherent study.

EVERY EXPECTED CELL MUST CARRY AN EXPLICIT OUTCOME
--------------------------------------------------
The corpus is rectangular or it is not certified. There is no third state: no
missing cell, no imputed cell, no silently excluded cell. Exactly two statuses
are admissible per expected cell, and they are counted and reported SEPARATELY:

  complete          a model generated and the verifier reached a verdict
  provider_blocked  the vendor's safety layer terminated the request before any
                    model generated -- an observed outcome of the deployed
                    STACK, never a model-generated refusal

`accepted_observations` (== expected cells) is the rectangularity number.
`model_observations` (accepted minus provider_blocked) is the analysable-data
number. Quoting the first where the second is meant would report model
behaviour that never happened, so the validator prints both and never sums
them into one "trials" figure.

FULL checklist (per profile: claude, fable, codex, llama)
  F1  all 3 shards present, each with exactly one accepted attempt
  F2  70 distinct base tasks
  F3  exactly 10 arms per base task, matching the canonical arm set
  F4  700 accepted observations, no duplicate cell, ZERO missing cells
  F5  zero synthetic trials (a provider block is NOT synthetic; see F12)
  F6  zero budget-censored trials
  F7  one agent version across the whole profile, equal to the campaign pin
  F8  one model id across every model observation, equal to the campaign pin
  F9  every contributing run dir lives inside the campaign namespace
  F10 every row carries this campaign_id
  F11 every row is an accepted status, and model_started agrees with it
  F12 every provider-blocked row carries the full explicit contract:
      model_started false, provider_refusal true, a vendor refusal category,
      no model output, and its design coordinates intact

RESOURCE checklist (analogous)
  R1-R4  3 shards, 70 base tasks, 3 arms each, 210 accepted, no dup/missing
  R5-R12 as F5-F12
  R11    resource control arms (clean-n, eval-scaf) are recorded with their
         `resource_derivation_parent` so they can never be silently pooled with
         the byte-identical FULL cells of the same name

CROSS
  X1  campaign totals equal 2800 (full) + 840 (resource) = 3640
  X2  no trial in the corpus originates outside results/campaigns/<id>/
  X3  every accepted attempt's dataset hash matches the frozen cell manifest
  X4  the provenance corpus builder finished with zero errors
  X5  campaign-wide accounting: accepted == expected, missing == 0
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from campaign import lib


class Check:
    def __init__(self):
        self.results = []

    def add(self, cid, ok, detail):
        self.results.append({"id": cid, "ok": bool(ok), "detail": detail})
        return ok

    @property
    def failed(self):
        return [r for r in self.results if not r["ok"]]


def load_corpus(paths) -> list[dict]:
    p = paths["provenance"] / "corpus.jsonl"
    if not p.is_file():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def accounting(rows: list[dict]) -> dict:
    """The four numbers that describe corpus completeness.

    They are always reported together and never collapsed. `accepted` says the
    design is rectangular; `model_observations` says how much of it is model
    behaviour; `provider_blocked` is the difference and is a stack outcome, not
    a model-generated refusal; `missing` must be zero.
    """
    exp = lib.expected_totals()["campaign_total"]
    cells = {(r["mode"], r["profile"], r["base_task_id"], r["arm"]) for r in rows}
    accepted = [r for r in rows if r["status"] in lib.ACCEPTED_STATUSES]
    blocked = [r for r in rows if r["status"] == lib.STATUS_PROVIDER_BLOCKED]
    return {
        "expected_cells": exp,
        "accepted_observations": len(accepted),
        "model_observations": sum(1 for r in accepted if r.get("model_started")),
        "provider_blocked": len(blocked),
        "provider_blocked_categories": sorted(
            {r.get("provider_refusal_category") or "" for r in blocked}),
        "missing": exp - len(cells),
        "unaccepted_rows": len(rows) - len(accepted),
        "duplicate_rows": len(rows) - len(cells),
    }


def validate(strict_profiles=lib.PROFILES) -> dict:
    paths = lib.campaign_paths()
    chk = Check()
    rows = load_corpus(paths)
    accepted_doc = lib.jload(paths["provenance"] / "accepted_runs.json") or {}
    accepted = accepted_doc.get("accepted", [])

    # X4 - the corpus builder must have finished clean. It fails closed and
    # deletes the corpus on error, so a missing corpus here is a real signal.
    build_report = lib.jload(paths["provenance"] / "build_report.json") or {}
    berrs = build_report.get("errors", [])
    chk.add("X4", bool(build_report) and not berrs,
            "provenance build clean" if (build_report and not berrs)
            else f"provenance build reported {len(berrs)} error(s): {berrs[:3]}"
                 if build_report else "no provenance build report; run `provenance build`")

    chk.add("X0", bool(rows), f"corpus has {len(rows)} trials")
    if not rows:
        return _finish(chk, paths, rows)

    # X2 - provenance boundary
    bad_paths = []
    for a in accepted:
        try:
            for _d in (a.get("run_dirs") or [a["run_dir"]]):
                lib.assert_campaign_path(lib.PROJECT_ROOT / _d, "run dir")
        except ValueError as exc:
            bad_paths.append(str(exc).splitlines()[0])
    chk.add("X2", not bad_paths,
            "all contributing run dirs inside campaign namespace" if not bad_paths
            else f"OUTSIDE-CAMPAIGN run dirs: {bad_paths}")

    # X3 - dataset hash still matches the frozen cell manifest
    drift = []
    for a in accepted:
        mf = paths["manifests"] / "cells" / f"{a['mode']}__{a['profile']}__{a['shard']}.json"
        cm = lib.jload(mf)
        if cm is None:
            drift.append(f"{a['cell']}: missing cell manifest")
            continue
        shard_dir = lib.PROJECT_ROOT / cm["dataset_path"]
        got = lib.sha256_file(shard_dir / "manifest.json") if (shard_dir / "manifest.json").is_file() else None
        if got != cm["dataset_manifest_sha256"]:
            drift.append(f"{a['cell']}: dataset manifest drifted")
    chk.add("X3", not drift, "dataset hashes match frozen manifests" if not drift else str(drift))

    # X1 - totals
    exp = lib.expected_totals()
    chk.add("X1", len(rows) == exp["campaign_total"],
            f"{len(rows)} trials vs expected {exp['campaign_total']} "
            f"(full {exp['full']['total']} + resource {exp['resource']['total']})")

    by = defaultdict(list)
    for r in rows:
        by[(r["mode"], r["profile"])].append(r)

    for mode in lib.MODES:
        tag = "F" if mode == "full" else "R"
        arms = set(lib.ARMS[mode])
        per_profile = lib.BASE_TASK_COUNT * lib.VARIANTS_PER_TASK[mode]
        for profile in strict_profiles:
            pre = f"{tag}:{mode}/{profile}"
            rs = by.get((mode, profile), [])

            shards = {r["shard"] for r in rs}
            att = [a for a in accepted if a["mode"] == mode and a["profile"] == profile]
            chk.add(f"{pre}:1", len(shards) == len(lib.SHARD_INDICES) and len(att) == len(lib.SHARD_INDICES),
                    f"{len(shards)} shards present, {len(att)} accepted attempts (want 3/3)")

            bases = {r["base_task_id"] for r in rs}
            chk.add(f"{pre}:2", len(bases) == lib.BASE_TASK_COUNT,
                    f"{len(bases)} distinct base tasks (want {lib.BASE_TASK_COUNT})")

            per_base = defaultdict(set)
            for r in rs:
                per_base[r["base_task_id"]].add(r["arm"])
            bad = {b: sorted(a) for b, a in per_base.items() if a != arms}
            chk.add(f"{pre}:3", not bad,
                    f"every base task has all {len(arms)} arms" if not bad
                    else f"{len(bad)} base tasks with wrong arm set, e.g. {list(bad.items())[:2]}")

            # F4 - rectangularity. Counted over ACCEPTED observations, which is
            # complete + provider_blocked: a blocked cell has an explicit
            # outcome and is not missing. `missing` must be 0; the gate can
            # never report a complete profile while any cell is unobserved.
            pairs = [(r["base_task_id"], r["arm"]) for r in rs]
            acc = [r for r in rs if r["status"] in lib.ACCEPTED_STATUSES]
            n_model = sum(1 for r in acc if r.get("model_started"))
            n_blocked = sum(1 for r in rs if r["status"] == lib.STATUS_PROVIDER_BLOCKED)
            dups = len(pairs) - len(set(pairs))
            missing = (lib.BASE_TASK_COUNT * len(arms)) - len(set(pairs))
            chk.add(f"{pre}:4",
                    len(acc) == per_profile and len(rs) == per_profile
                    and dups == 0 and missing == 0,
                    f"{len(acc)}/{per_profile} accepted observations "
                    f"({n_model} model, {n_blocked} provider-blocked), "
                    f"{dups} duplicate cells, {missing} missing cells")

            # F5 - a synthetic trial means no real request was ever made: a
            # harness defect. A provider block IS a real request the vendor
            # terminated, so it carries its own status and is checked by F12,
            # not swept in here and not excused from scrutiny either.
            synth = [r for r in rs if r["status"] == lib.STATUS_SYNTHETIC]
            chk.add(f"{pre}:5", not synth, f"{len(synth)} synthetic trials (want 0)")

            cens = [r for r in rs if r["status"] == lib.STATUS_BUDGET_CENSORED]
            chk.add(f"{pre}:6", not cens, f"{len(cens)} budget-censored trials (want 0)")

            pin = lib.VERSION_PINS[profile]["version"]
            vers = {r["agent_version"] for r in rs if r["agent_version"]}
            chk.add(f"{pre}:7", vers == {pin},
                    f"agent versions {sorted(vers)} (want exactly {{{pin}}})")

            # F8 - only rows where a model generated can testify to WHICH model
            # ran. A blocked row carries the safety layer's placeholder name;
            # admitting it here would either fail a correct corpus or, worse,
            # invite someone to relax the check for every row.
            models = {r["model_name"] for r in rs
                      if r["model_name"] and r.get("model_started")}
            expect_model = lib.MODEL_PINS[profile].split("/", 1)[-1] if "/" in lib.MODEL_PINS[profile] else lib.MODEL_PINS[profile]
            ok_model = len(models) == 1 and any(m.endswith(expect_model.split("/")[-1]) for m in models)
            chk.add(f"{pre}:8", ok_model,
                    f"model ids {sorted(models)} across {n_model} model observations "
                    f"(want one, matching {lib.MODEL_PINS[profile]})")

            runs = {r["run_id"] for r in rs}
            chk.add(f"{pre}:9", all(any(a["run_id"] == rn for a in att) for rn in runs),
                    f"{len(runs)} run ids, all from accepted campaign attempts")

            cids = {r["campaign_id"] for r in rs}
            chk.add(f"{pre}:10", cids == {lib.CAMPAIGN_ID},
                    f"campaign ids {sorted(cids)} (want exactly {{{lib.CAMPAIGN_ID}}})")

            # F11 - every row holds one of the two admissible outcomes, and
            # `model_started` agrees with the status. A row claiming model
            # behaviour it did not produce (or hiding behaviour it did) would
            # corrupt every downstream gate that trusts the flag.
            bad_status = sorted({r["status"] for r in rs
                                 if r["status"] not in lib.ACCEPTED_STATUSES})
            mismatched = [r["trial_dir"] for r in rs
                          if bool(r.get("model_started"))
                          != (r["status"] == lib.STATUS_COMPLETE)]
            chk.add(f"{pre}:11", not bad_status and not mismatched,
                    f"all {len(rs)} rows accepted with consistent model_started"
                    if not bad_status and not mismatched
                    else f"non-accepted statuses {bad_status}; "
                         f"{len(mismatched)} rows whose model_started contradicts "
                         f"their status, e.g. {mismatched[:3]}")

            # F12 - the provider-blocked contract, field by field. These rows
            # are the only ones in the corpus with no model behind them, so
            # every claim about them is asserted explicitly rather than assumed.
            blocked_rows = [r for r in rs
                            if r["status"] == lib.STATUS_PROVIDER_BLOCKED]
            contract = []
            for r in blocked_rows:
                td = r["trial_dir"]
                if r.get("model_started"):
                    contract.append(f"{td}: model_started true")
                if not r.get("provider_refusal"):
                    contract.append(f"{td}: provider_refusal not set")
                if not r.get("provider_refusal_category"):
                    contract.append(f"{td}: no vendor refusal category")
                if (r.get("completion_tokens") or 0) > 0:
                    contract.append(f"{td}: {r['completion_tokens']} completion tokens")
                if r.get("resolved") or (r.get("reward") or 0) > 0:
                    contract.append(f"{td}: scored as if a model had run")
                missing_factors = [f for f in lib.ARM_FACTORS if not r.get(f)]
                if missing_factors:
                    contract.append(f"{td}: lost design coordinates {missing_factors}")
                if not r.get("base_task_id") or not r.get("arm") or not r.get("run_id"):
                    contract.append(f"{td}: lost base task / arm / source provenance")
            cats = sorted({r.get("provider_refusal_category") for r in blocked_rows})
            chk.add(f"{pre}:12", not contract,
                    f"{len(blocked_rows)} provider-blocked rows, categories {cats}, "
                    f"all explicitly marked as stack outcomes with no model output"
                    if not contract else str(contract[:5]))

    # R11 - resource control lineage is explicit
    lineage_missing = []
    for profile in strict_profiles:
        mf_any = paths["manifests"] / "cells" / f"resource__{profile}__chunk-1-size-{lib.SHARD_SIZE}.json"
        cm = lib.jload(mf_any)
        if cm is None:
            lineage_missing.append(f"resource/{profile}: no cell manifest")
            continue
        controls = [t for t in cm["trials"] if t["arm"] in ("clean-n", "eval-scaf")]
        if controls and not all(t["resource_derivation_parent"] for t in controls):
            lineage_missing.append(f"resource/{profile}: control arms lack derivation parent")
    chk.add("R11", not lineage_missing,
            "resource control arms carry explicit FULL lineage" if not lineage_missing
            else str(lineage_missing))

    # RT1/RT2 - retry lineage must be EXPLICIT, never a silent replacement.
    #
    # The Aug 2026 failure was a silent dedupe/backfill: a later trial quietly
    # standing in for an earlier one under the same task_name. A retry is only
    # legitimate here if the ledger records the lineage AND the failed original
    # is still on disk. Absence of a ledger entry is not "no retries happened";
    # it is only that when no retry trial appears in the corpus either.
    from campaign import failures as _failures

    ledger_path = paths["provenance"] / "retries.jsonl"
    retry_rows, malformed = [], []
    if ledger_path.is_file():
        for i, line in enumerate(ledger_path.read_text().splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                malformed.append(f"line {i}: {exc}")
                continue
            required = ("original_trial_id", "retry_trial_id", "retry_number",
                        "failure_class", "failure_reason", "model_started",
                        "started_at", "accepted_status")
            missing = [k for k in required if rec.get(k) in (None, "")]
            if missing:
                malformed.append(f"line {i}: missing {missing}")
                continue
            if rec["failure_class"] == _failures.BUDGET:
                malformed.append(f"line {i}: budget failure recorded as a retry - never permitted")
            if not (1 <= int(rec["retry_number"]) <= _failures.MAX_RETRIES):
                malformed.append(
                    f"line {i}: retry_number {rec['retry_number']} outside 1..{_failures.MAX_RETRIES}")
            retry_rows.append(rec)

    chk.add("RT1", not malformed,
            f"retry ledger well-formed ({len(retry_rows)} records)" if not malformed
            else str(malformed[:5]))

    # Every retry that made it into the corpus must (a) have a ledger record and
    # (b) leave its failed original preserved.
    by_retry_id = {r["retry_trial_id"]: r for r in retry_rows}
    corpus_dirs = {r["trial_dir"] for r in rows}
    silent = []
    for rec in retry_rows:
        if rec["retry_trial_id"] in corpus_dirs and rec["accepted_status"] == "accepted":
            if rec["original_trial_id"] in corpus_dirs:
                continue  # original preserved in the corpus alongside the retry
            still_on_disk = any(
                (lib.PROJECT_ROOT / d / rec["original_trial_id"]).exists()
                for a in accepted
                for d in (a.get("run_dirs") or [a["run_dir"]]))
            if not still_on_disk:
                silent.append(
                    f"{rec['retry_trial_id']} accepted but original "
                    f"{rec['original_trial_id']} no longer preserved")
    # A trial that supersedes another WITHOUT any ledger record is the exact
    # silent-replacement shape we refuse.
    _cells = defaultdict(list)
    for r in rows:
        _cells[r["cell"]].append(r)
    for cell_key, group in _cells.items():
        seen_arm = {}
        for r in group:
            k = (r["base_task_id"], r["arm"])
            if k in seen_arm and r["trial_dir"] not in by_retry_id:
                silent.append(
                    f"{cell_key} {k} served by {r['trial_dir']} with no retry lineage record")
            seen_arm[k] = r["trial_dir"]

    chk.add("RT2", not silent,
            "no silent task_name replacement" if not silent else str(silent[:5]))

    # X5 - campaign-wide accounting. The one check that can never be satisfied
    # by a corpus with a hole in it: accepted must equal expected AND missing
    # must be zero, so "complete" and "missing cells" can never be reported
    # together no matter what the per-profile checks say.
    acct = accounting(rows)
    chk.add("X5",
            acct["accepted_observations"] == acct["expected_cells"]
            and acct["missing"] == 0 and acct["duplicate_rows"] == 0
            and acct["unaccepted_rows"] == 0,
            f"accepted {acct['accepted_observations']}/{acct['expected_cells']} "
            f"(model {acct['model_observations']}, provider-blocked "
            f"{acct['provider_blocked']}), missing {acct['missing']}, "
            f"duplicates {acct['duplicate_rows']}, "
            f"non-accepted {acct['unaccepted_rows']}")

    return _finish(chk, paths, rows)


def _finish(chk, paths, rows) -> dict:
    acct = accounting(rows)
    doc = {
        "campaign_id": lib.CAMPAIGN_ID,
        "validated_at": lib.now_iso(),
        "trials": len(rows),
        # Carried into the report so no consumer has to recompute -- or guess --
        # how many of the rows are actually model observations.
        **acct,
        "checks": chk.results,
        "failed": [r["id"] for r in chk.failed],
        "ok": not chk.failed,
    }
    paths["validation"].mkdir(parents=True, exist_ok=True)
    (paths["validation"] / "validation_report.json").write_text(json.dumps(doc, indent=2) + "\n")
    return doc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    doc = validate()
    if args.json:
        print(json.dumps(doc, indent=2))
    else:
        for r in doc["checks"]:
            print(f"  [{'PASS' if r['ok'] else 'FAIL'}] {r['id']:28s} {r['detail']}")
        print()
        print(f"  accepted observations  {doc['accepted_observations']:>6} / "
              f"{doc['expected_cells']} expected cells")
        print(f"    model observations   {doc['model_observations']:>6}")
        print(f"    provider blocked     {doc['provider_blocked']:>6}"
              f"   (stack outcome; NOT a model-generated refusal)")
        print(f"  missing                {doc['missing']:>6}")
        print()
        print(f"  {'CAMPAIGN VALID' if doc['ok'] else 'CAMPAIGN INVALID: ' + ', '.join(doc['failed'])}")
    sys.exit(0 if doc["ok"] else 1)


if __name__ == "__main__":
    main()
