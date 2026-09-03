#!/usr/bin/env python3
"""Fail-closed preflight. Runs before the campaign and again before EVERY shard.

Checks, in order, and exits non-zero on the first failure:

  1. namespace     - campaign dirs + manifests exist and are for THIS campaign id
  2. amendments    - any cell manifest that deviates from what `prepare` froze is
                     exactly the archived original plus its approved, append-only
                     task-definition amendments
  3. integrity     - every task dir hash in the cell manifest still matches disk
  4. versions      - all four stacks explicitly pinned and present in the env
  5. models        - the pinned model id is exposed by the gateway
  6. budget        - remaining budget covers the work the CALLING COMMAND is
                     about to launch, through to the end of that work

Checks 2 and 3 together are the integrity rule, and it is not weakened by the
existence of amendments. A task definition on disk must equal EITHER the
originally frozen definition OR the exact definition named by an approved
amendment record. Check 3 pins disk to the manifest; check 2 pins the manifest
to `prepare`'s output plus explicit provenance. Arbitrary drift fails check 3,
and a manifest quietly re-frozen to bless that drift fails check 2.

Rule 6 is the one that matters. The Aug 2026 study died because shard 2 was
launched with enough budget to start and not enough to FINISH, producing 251
synthetic and hundreds of budget-censored trials that then had to be
quarantined. The invariant that prevents a repeat is: never launch a unit of
work the remaining budget cannot see through to the end.

THE UNIT IS WHATEVER THE COMMAND ACTUALLY LAUNCHES
--------------------------------------------------
A standalone preflight, and `run-full` / `run-resource` -- which launch all
three shards of a mode in one uninterruptible sweep -- are gated on the whole
remaining campaign, unchanged.

`run-shard` and `repair-shard` launch exactly one shard, or exactly one repair
plan, and then STOP. Campaign V2 is cell-level resumable, so a shard whose own
outstanding work is fully funded cannot produce a budget-censored trial however
much the rest of the campaign costs: the money is either there for those cells
or the gate refuses before Harbor is invoked. Charging those commands for work
they will not launch does not buy that safety -- it only refuses work the
budget genuinely covers, which is the same class of error as pricing a resumed
shard at its full trial count. So they gate on their own scope, passed with
--budget-scope-mode / --budget-scope-shard, and the whole-campaign projection
is still computed and reported as a WARNING so the operator sees the cliff
coming.

The safety factors are identical in both cases and are not negotiable per
scope: 20% planning contingency inside the planning cost, then a further 10%
margin on top of it.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from campaign import cells, cost_model, lib

# Contingency folded into the planning cost, on top of the expected per-trial
# mean. Same factor `cost_model.planning_total_usd` uses for the whole campaign.
PLANNING_CONTINGENCY = 0.20
# Fraction of headroom demanded on top of the remaining planning cost.
SAFETY_MARGIN = 0.10


@dataclass(frozen=True)
class BudgetScope:
    """The unit of work a launch gate is about to start, and nothing else.

    `None` for both fields is the whole remaining campaign. A scope narrows
    WHICH cells are priced; it never changes HOW they are priced, and it never
    touches the structural checks -- a scoped preflight still hashes every cell
    manifest in the campaign, because integrity drift anywhere is a reason to
    stop regardless of what is about to launch.
    """
    label: str
    mode: str | None = None
    shard: int | None = None

    def selects(self, cell: lib.Cell) -> bool:
        return ((self.mode is None or cell.mode == self.mode) and
                (self.shard is None or cell.shard_index == self.shard))


class Fail(Exception):
    pass


def check_namespace(paths) -> list[str]:
    notes = []
    if not paths["root"].is_dir():
        raise Fail(f"campaign namespace missing: {paths['root']} (run `./campaign.sh prepare`)")
    idx = paths["manifests"] / "cells_index.json"
    if not idx.is_file():
        raise Fail(f"cells index missing: {idx} (run `./campaign.sh prepare`)")
    data = json.loads(idx.read_text())
    if data.get("campaign_id") != lib.CAMPAIGN_ID:
        raise Fail(f"cells index is for campaign {data.get('campaign_id')!r}, expected {lib.CAMPAIGN_ID!r}")
    if len(data["cells"]) != len(lib.all_cells()):
        raise Fail(f"cells index has {len(data['cells'])} cells, expected {len(lib.all_cells())}")
    notes.append(f"namespace OK: {len(data['cells'])} cells, "
                 f"{sum(c['expected_trials'] for c in data['cells'])} trials")
    return notes


def check_amendments(paths, cells: list[lib.Cell]) -> list[str]:
    """Every deviation from the frozen manifests must be explicitly approved.

    `check_integrity` compares disk against the cell manifest, so the manifest
    itself has to be trustworthy. This proves it is: for each selected cell the
    current manifest must be reconstructible from the archived original plus the
    append-only amendment ledger. A manifest with amendments the ledger does not
    record, a ledger record the manifest does not declare, a missing archive, an
    amendment whose "original" hashes do not match what `prepare` actually froze,
    or an amendment against a cell holding a COMPLETE_VALID trajectory all fail
    here -- which is what stops "amend the manifest" from becoming a way to
    launder arbitrary drift past check 3.
    """
    from campaign import amendments as amd
    problems = amd.verify(cells, paths=paths)
    if problems:
        raise Fail("unapproved task-definition amendment(s):\n  " +
                   "\n  ".join(problems) +
                   "\n\n  A cell manifest may differ from what `prepare` froze ONLY by an\n"
                   "  approved record in provenance/" + amd.LEDGER_NAME + ".")
    ledger = amd.load_ledger(paths)
    by_cell = amd.ledger_by_cell(paths, ledger)
    n = sum(len(by_cell.get(c.key, [])) for c in cells)
    if not n:
        return ["amendments OK: no cell manifest deviates from prepare"]
    return [f"amendments OK: {n} approved task-definition amendment(s) across "
            f"{sum(1 for c in cells if by_cell.get(c.key))} cell(s), each matching "
            f"its archived original and ledger record"]


def check_integrity(paths, cells: list[lib.Cell], quick: bool) -> list[str]:
    notes = []
    for cell in cells:
        mf = paths["manifests"] / "cells" / f"{cell.mode}__{cell.profile}__{cell.shard_label}.json"
        if not mf.is_file():
            raise Fail(f"missing cell manifest: {mf}")
        cm = json.loads(mf.read_text())
        if cm["campaign_id"] != lib.CAMPAIGN_ID:
            raise Fail(f"{mf.name}: wrong campaign_id {cm['campaign_id']!r}")
        shard_dir = lib.PROJECT_ROOT / cm["dataset_path"]
        if not shard_dir.is_dir():
            raise Fail(f"{cell.key}: dataset snapshot missing at {shard_dir}")
        got = lib.sha256_file(shard_dir / "manifest.json")
        if got != cm["dataset_manifest_sha256"]:
            raise Fail(f"{cell.key}: dataset manifest changed since prepare\n"
                       f"  expected {cm['dataset_manifest_sha256']}\n  found    {got}")
        rows = cm["trials"]
        if quick:
            # Sample, but never let the sample skip an amended definition: those
            # are the rows whose hashes were changed by hand-approved provenance
            # and so the rows most worth re-proving against disk.
            amended = {r["task_dir"] for r in cm.get("task_definition_amendments") or []}
            sample = rows[:: max(1, len(rows) // 10)]
            seen = {id(r) for r in sample}
            subset = sample + [r for r in rows
                               if r["task_dir"] in amended and id(r) not in seen]
        else:
            subset = rows
        for r in subset:
            td = lib.PROJECT_ROOT / r["snapshot_task_path"]
            if not td.is_dir():
                raise Fail(f"{cell.key}: task dir missing: {r['task_dir']}")
            if lib.sha256_tree(td) != r["task_content_sha256"]:
                raise Fail(f"{cell.key}: task content changed: {r['task_dir']}")
        notes.append(f"integrity OK: {cell.key} ({len(subset)}/{len(rows)} task dirs hashed)")
    return notes


def check_versions(profiles) -> list[str]:
    notes = []
    for p in profiles:
        pin = lib.VERSION_PINS[p]
        env_val = os.environ.get(pin["pin_env"], "")
        if not env_val:
            raise Fail(
                f"{p}: {pin['pin_env']} is not set. Every stack must be explicitly "
                f"pinned before the campaign may run. Expected {pin['version']!r}."
            )
        if env_val != pin["version"]:
            raise Fail(f"{p}: {pin['pin_env']}={env_val!r} but campaign pins {pin['version']!r}")
        notes.append(f"version OK: {p} -> {pin['agent']} {pin['version']} (via {pin['pin_env']})")
    return notes


def check_models(profiles, key: str) -> list[str]:
    notes = []
    base = lib.LITELLM_ANTHROPIC_BASE
    try:
        proc = subprocess.run(
            ["curl", "-sS", "-m", "30", "-H", f"Authorization: Bearer {key}", f"{base}/v1/models"],
            capture_output=True, text=True, timeout=45)
        ids = {m.get("id", "") for m in json.loads(proc.stdout).get("data", [])}
    except Exception as exc:
        raise Fail(f"could not list gateway models: {exc}")
    for p in profiles:
        model = lib.MODEL_PINS[p]
        # litellm strips a leading `openai/` provider prefix before dispatch, so
        # accept either the full id or the stripped wire name.
        wire = model.split("/", 1)[1] if model.startswith("openai/") else model
        if model not in ids and wire not in ids:
            raise Fail(f"{p}: neither {model!r} nor {wire!r} is exposed by the gateway")
        notes.append(f"model OK: {p} -> {model}")
    return notes


def remaining_campaign_cost(paths, est: dict, scope: BudgetScope | None = None
                            ) -> tuple[float, list[str], int]:
    """Planning cost of the EXPERIMENTAL CELLS that still require inference.

    This used to charge every not-yet-accepted profile/shard at its full trial
    count. That is wrong for an interrupted shard, and wrong in an expensive
    direction: after the 2026-09-02 crash, FULL shard 1 held 976 valid
    trajectories and needed 224 more, but this function still priced it as
    1,200 fresh trials -- overstating remaining cost by roughly $2.2k and
    letting the budget gate refuse work the budget actually covered.

    `campaign.cells.remaining_trials` resolves each profile/shard to the number
    of task/arm cells genuinely outstanding, from the written repair plan when
    one exists -- which is also what makes a `repair-shard` scope price exactly
    the cells in that shard's repair plan and nothing else.

    `scope` restricts WHICH profile/shard cells are counted. It does not change
    the pricing: every counted trial still carries the same per-trial mean and
    the same 20% contingency.

    Returns (planning cost incl. contingency, pending cell descriptions, trials).
    """
    remaining = cells.remaining_trials(paths)
    total, pending, trials = 0.0, [], 0
    for cell in lib.all_cells():
        if scope is not None and not scope.selects(cell):
            continue
        info = remaining.get(cell.key) or {"remaining_trials": cell.expected_trials,
                                           "basis": "not_started"}
        n = int(info["remaining_trials"])
        if n <= 0:
            continue
        c = est["cells"][f"{cell.mode}/{cell.profile}"]
        per_trial = c["mean_usd_per_trial"] or 0.0
        total += per_trial * n * (1 + PLANNING_CONTINGENCY)   # same contingency as planning
        trials += n
        pending.append(f"{cell.key} ({n} trials, {info['basis']})")
    return total, pending, trials


def check_budget(paths, est: dict, key: str | None,
                 scope: BudgetScope | None = None) -> list[str]:
    """Refuse to launch unless `scope`'s outstanding work is fully funded.

    The whole-campaign projection is computed unconditionally. When the caller
    scoped the gate, a whole-campaign shortfall is reported as a WARNING rather
    than a refusal -- the operator is told the campaign does not fit end to end,
    while the shard in front of them, which does fit and which the controller
    can resume cell by cell, is allowed to proceed.
    """
    status = lib.probe_budget(key)
    if not status.ok:
        raise Fail(f"budget probe failed: {status.error} (key {status.key_fingerprint})")

    whole_need, whole_pending, whole_trials = remaining_campaign_cost(paths, est)
    whole_required = whole_need * (1 + SAFETY_MARGIN)

    if scope is None:
        need, pending, trials = whole_need, whole_pending, whole_trials
        label = "whole remaining campaign"
    else:
        need, pending, trials = remaining_campaign_cost(paths, est, scope=scope)
        label = scope.label

    required = need * (1 + SAFETY_MARGIN)
    notes = [
        f"key             : {status.key_fingerprint}",
        f"max budget      : ${status.max_budget:,.2f}",
        f"spend           : ${status.spend:,.2f}",
        f"remaining       : ${status.remaining:,.2f}",
        f"tpm/rpm         : {status.tpm_limit}/{status.rpm_limit}",
        f"gating on       : {label}",
        f"cells pending   : {len(pending)}  ({trials:,} trials still requiring inference)",
        f"planning cost   : ${need:,.2f}  (incl. {int(PLANNING_CONTINGENCY*100)}% contingency)",
        f"required        : ${required:,.2f}  (+{int(SAFETY_MARGIN*100)}% safety margin)",
    ]
    if status.remaining < required:
        raise Fail(
            "INSUFFICIENT BUDGET - refusing to launch.\n  " + "\n  ".join(notes) +
            f"\n  short by ${required - status.remaining:,.2f}.\n"
            "  Raise the key budget or reduce campaign scope; do NOT start a partial run."
        )

    if scope is not None:
        # Always shown, fit or not: a scoped gate must never let the operator
        # lose sight of the number it did not gate on.
        notes.append(
            f"whole campaign  : ${whole_need:,.2f} planning / ${whole_required:,.2f} required "
            f"({whole_trials:,} trials across {len(whole_pending)} cells)")
        if status.remaining < whole_required:
            notes.append(
                f"WARNING: the WHOLE remaining campaign does NOT fit. It requires "
                f"${whole_required:,.2f} against ${status.remaining:,.2f} remaining - short by "
                f"${whole_required - status.remaining:,.2f}. This gate cleared {label} ONLY. "
                "Later shards will be refused unless the key budget is raised or the "
                "campaign scope is reduced. Re-run preflight between shards."
            )

    p90 = est["p90_total_usd"]
    if status.remaining < p90:
        notes.append(
            f"WARNING: the p90 (pessimistic) whole-campaign cost is ${p90:,.2f}, which "
            f"exceeds remaining budget ${status.remaining:,.2f}. The expected-cost plan "
            "fits, but a heavy-tail run will not. Re-run preflight between shards."
        )
    return notes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=lib.MODES)
    ap.add_argument("--profile", choices=lib.PROFILES)
    ap.add_argument("--shard", type=int, choices=lib.SHARD_INDICES)
    ap.add_argument("--quick", action="store_true",
                    help="hash a 10%% sample of task dirs instead of all (per-shard use)")
    ap.add_argument("--skip-budget", action="store_true", help="offline structural checks only")
    # Scope the BUDGET GATE ONLY -- never the structural checks above it. Set by
    # run-shard / repair-shard, which launch one shard and stop; omitted by a
    # standalone preflight and by run-full / run-resource, which stay gated on
    # the whole remaining campaign.
    ap.add_argument("--budget-scope-mode", choices=lib.MODES,
                    help="gate the budget on this mode's outstanding work only")
    ap.add_argument("--budget-scope-shard", type=int, choices=lib.SHARD_INDICES,
                    help="gate the budget on this shard's outstanding work only")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    paths = lib.campaign_paths()
    cells = lib.all_cells()
    if args.mode:
        cells = [c for c in cells if c.mode == args.mode]
    if args.profile:
        cells = [c for c in cells if c.profile == args.profile]
    if args.shard:
        cells = [c for c in cells if c.shard_index == args.shard]
    if not cells:
        print("no cells selected", file=sys.stderr)
        sys.exit(2)
    profiles = sorted({c.profile for c in cells})
    key = os.environ.get("LITE_LLM_KEY") or os.environ.get("ANTHROPIC_API_KEY") or ""

    scope = None
    if args.budget_scope_mode or args.budget_scope_shard:
        parts = []
        if args.budget_scope_mode:
            parts.append(args.budget_scope_mode.upper())
        if args.budget_scope_shard:
            parts.append(f"shard {args.budget_scope_shard}")
        scope = BudgetScope(
            label=" ".join(parts) + " outstanding inference work",
            mode=args.budget_scope_mode,
            shard=args.budget_scope_shard,
        )

    notes, failures = [], []
    try:
        notes += check_namespace(paths)
        notes += check_amendments(paths, cells)
        notes += check_integrity(paths, cells, args.quick)
        notes += check_versions(profiles)
        if not args.skip_budget:
            notes += check_models(profiles, key)
            est = cost_model.estimate()
            notes += check_budget(paths, est, key, scope=scope)
    except Fail as exc:
        failures.append(str(exc))

    result = {"campaign_id": lib.CAMPAIGN_ID, "checked_at": lib.now_iso(),
              "cells": [c.key for c in cells],
              # What the budget gate was actually scoped to, on the record.
              "budget_scope": scope.label if scope else "whole remaining campaign",
              "notes": notes,
              "failures": failures, "ok": not failures}
    log = paths["logs"] / "preflight.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as fh:
        fh.write(json.dumps(result) + "\n")

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for n in notes:
            print(f"  {n}")
        for f in failures:
            print(f"\nPREFLIGHT FAILED:\n  {f}", file=sys.stderr)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
