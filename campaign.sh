#!/usr/bin/env bash
# =============================================================================
# SWE-EvalPressure campaign runner - replication-20260902-v1
#
#   ./campaign.sh prepare      replication-20260902-v1
#   ./campaign.sh preflight    replication-20260902-v1
#   ./campaign.sh run-full     replication-20260902-v1
#   ./campaign.sh run-resource replication-20260902-v1
#   ./campaign.sh run-shard    replication-20260902-v1 <full|resource> <1|2|3>
#   ./campaign.sh validate     replication-20260902-v1
#   ./campaign.sh analyze      replication-20260902-v1
#
# DESIGN
#   FULL      70 base tasks x 10 arms x 4 profiles = 2800 fresh trajectories
#   RESOURCE  70 base tasks x  3 arms x 4 profiles =  840 fresh trajectories
#   TOTAL                                            3640 fresh trajectories
#
#   RESOURCE is deliberately SELF-CONTAINED. Its clean-n and eval-scaf arms are
#   executed independently even though those task definitions are byte-identical
#   to the FULL cells of the same name. No FULL trajectory is ever substituted
#   into RESOURCE analysis.
#
# EXECUTION CONTRACT
#   run-full / run-resource each do, per shard i in 1,2,3:
#       preflight  ->  all four profiles  ->  validate-shard  ->  next shard
#   run-shard does exactly ONE iteration of that loop and then STOPS. It is the
#   same contract, not a relaxed one: same preflight, same four profiles, same
#   attempt ledger, same validation gate. It exists so a tight budget can be
#   spent one shard at a time with a decision point in between.
#   ANY failed preflight or validation STOPS the campaign immediately. There is
#   no resume-past-failure, no partial-shard acceptance, and no backfilling. A
#   failed attempt is preserved in the ledger as FAILED; a replacement run gets
#   a NEW attempt_id.
#
# ISOLATION
#   Everything is written beneath results/campaigns/<campaign-id>/ from the
#   moment it is generated. The Python tooling refuses any path outside that
#   namespace, so no historical Aug 2026 directory can enter this campaign.
# =============================================================================
set -euo pipefail

CAMPAIGN_ID_EXPECTED="replication-20260902-v1"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

CMD="${1:-}"; CAMPAIGN_ID="${2:-}"
usage() {
  sed -n '2,35p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 2
}
[[ -n "$CMD" ]] || usage
case "$CMD" in prepare|preflight|run-full|run-resource|run-shard|validate|analyze) ;; *) usage ;; esac
[[ -n "$CAMPAIGN_ID" ]] || { echo "campaign id required (expected $CAMPAIGN_ID_EXPECTED)" >&2; exit 2; }
if [[ "$CAMPAIGN_ID" != "$CAMPAIGN_ID_EXPECTED" ]]; then
  echo "FATAL: campaign id '$CAMPAIGN_ID' does not match the id this tree is pinned to" >&2
  echo "       ('$CAMPAIGN_ID_EXPECTED', set in campaign/lib.py)." >&2
  echo "       Refusing rather than writing into the wrong namespace." >&2
  exit 2
fi

# --------------------------------------------------------------------------- #
# credentials - reuse the repo's own .env loader so the campaign runner and the
# benchmark runner see exactly the same key. set_litellm_aliases maps the single
# LITE_LLM_KEY onto ANTHROPIC_API_KEY / OPENAI_API_KEY / MSWEA_API_KEY, which is
# why all four profiles draw on ONE budget pool.
# --------------------------------------------------------------------------- #
# shellcheck disable=SC1091
source "$PROJECT_ROOT/scripts/00_common.sh"
load_env

# --------------------------------------------------------------------------- #
# pins - every stack this campaign executes is version-locked here
# --------------------------------------------------------------------------- #
export RESULTS_ROOT="$PROJECT_ROOT/results/campaigns/$CAMPAIGN_ID"
export CLAUDE_CODE_VERSION="2.1.247"   # claude AND fable profiles (claude-code)
export CODEX_VERSION="0.147.0"
export MINI_SWE_VERSION="2.4.5"
export MINI_SWE_LITELLM_VERSION="1.83.0"
export HARBOR_REPEATS="${HARBOR_REPEATS:-1}"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PROFILES=(claude fable codex llama)
SHARDS=(1 2 3)
RUN_CONTEXT="run-mode"      # overwritten by run_mode / run_shard, used in notes
SHARD_SIZE=30
PY="$PROJECT_ROOT/.venv/bin/python"
[[ -x "$PY" ]] || PY="python3"
LOG_DIR="$RESULTS_ROOT/logs"

say()  { printf '\n\033[1m=== %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
die()  { printf '\n\033[1;31mFATAL: %s\033[0m\n' "$*" >&2; exit 1; }

audit() {  # append one structured line to the campaign runner log
  mkdir -p "$LOG_DIR"
  "$PY" - "$LOG_DIR/runner.jsonl" "$@" <<'PY'
import json, sys
from datetime import datetime, timezone
out, event, *rest = sys.argv[1:]
rec = {"at": datetime.now(timezone.utc).isoformat(), "event": event}
for kv in rest:
    k, _, v = kv.partition("="); rec[k] = v
with open(out, "a") as fh: fh.write(json.dumps(rec) + "\n")
PY
}

# --------------------------------------------------------------------------- #
do_prepare() {
  say "prepare $CAMPAIGN_ID"
  "$PY" -m campaign.prepare ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} || die "prepare failed"
  info "campaign root: ${RESULTS_ROOT#$PROJECT_ROOT/}"
  audit prepare_ok
}

do_preflight() {  # $1 optional context label
  local ctx="${1:-standalone}"
  say "preflight ($ctx)"
  if ! "$PY" -m campaign.preflight ${PREFLIGHT_ARGS[@]+"${PREFLIGHT_ARGS[@]}"}; then
    audit "preflight_failed" "context=$ctx"
    die "preflight failed ($ctx) - campaign STOPPED. Nothing was launched."
  fi
  audit "preflight_ok" "context=$ctx"
}

run_one() {  # mode profile shard_index
  local mode="$1" profile="$2" shard="$3"
  local label="chunk-${shard}-size-${SHARD_SIZE}"
  local dataset="$RESULTS_ROOT/datasets/_shards/$mode/$profile/$label"
  [[ -d "$dataset" ]] || die "missing prepared shard: $dataset (run: ./campaign.sh prepare $CAMPAIGN_ID)"

  say "run $mode / $profile / shard $shard"
  local outfile; outfile="$(mktemp)"
  local started; started="$($PY -c 'import datetime;print(datetime.datetime.now(datetime.timezone.utc).isoformat())')"
  audit "run_start" "mode=$mode" "profile=$profile" "shard=$shard" "dataset=$dataset"

  local rc=0
  SWE_RUN_OUTPUT_FILE="$outfile" \
    "$PROJECT_ROOT/scripts/05_run_profile.sh" "$mode" "$profile" \
      --dataset-override "$dataset" \
      --shard-label "-$label" || rc=$?

  local run_dir; run_dir="$(cat "$outfile" 2>/dev/null || true)"; rm -f "$outfile"
  [[ -n "$run_dir" ]] || die "$mode/$profile/shard $shard: runner produced no output directory"

  # Record the attempt EITHER WAY. A failed attempt stays in the ledger as
  # evidence; it never silently disappears and is never backfilled.
  local status="auto"; [[ "$rc" -ne 0 ]] && status="failed"
  "$PY" -m campaign.provenance record \
    --mode "$mode" --profile "$profile" --shard "$shard" \
    --run-dir "$run_dir" --status "$status" --started-at "$started" \
    --note "campaign.sh $RUN_CONTEXT $mode shard $shard" || {
      audit "attempt_failed" "mode=$mode" "profile=$profile" "shard=$shard" "run_dir=$run_dir"
      die "$mode/$profile/shard $shard did NOT complete cleanly.
       The attempt is preserved as FAILED in provenance/attempts.jsonl.
       Campaign STOPPED. Do not backfill: rerun this cell to create a NEW
       attempt_id, or investigate first."
    }
  audit "run_ok" "mode=$mode" "profile=$profile" "shard=$shard" "run_dir=$run_dir"
}

validate_progress() {  # validate what is finished so far; stop on any real defect
  local ctx="$1"
  say "validate ($ctx)"
  "$PY" -m campaign.provenance build || {
    audit "corpus_build_failed" "context=$ctx"
    die "corpus build reported errors ($ctx) - campaign STOPPED."
  }
  "$PY" - "$ctx" <<'PY' || { audit "shard_validation_failed" "context=$1"; die "shard validation failed ($1) - campaign STOPPED."; }
import json, sys
from collections import defaultdict
from campaign import lib

ctx = sys.argv[1]
paths = lib.campaign_paths()
acc = lib.jload(paths["provenance"] / "accepted_runs.json") or {}
accepted = acc.get("accepted", [])
corpus = paths["provenance"] / "corpus.jsonl"
rows = [json.loads(l) for l in corpus.read_text().splitlines() if l.strip()] if corpus.is_file() else []

if acc.get("conflicting_cells"):
    print(f"FAIL: multiple accepted attempts for {acc['conflicting_cells']}"); sys.exit(1)

by = defaultdict(list)
for r in rows:
    by[(r["mode"], r["profile"], r["shard"])].append(r)

bad = []
for a in accepted:
    cell = lib.Cell(a["mode"], a["profile"], a["shard_index"])
    got = by.get((a["mode"], a["profile"], a["shard"]), [])
    pairs = {(r["base_task_id"], r["arm"]) for r in got}
    synth = sum(1 for r in got if r["status"] == lib.STATUS_SYNTHETIC)
    cens = sum(1 for r in got if r["status"] == lib.STATUS_BUDGET_CENSORED)
    vers = {r["agent_version"] for r in got if r["agent_version"]}
    pin = lib.VERSION_PINS[a["profile"]]["version"]
    if len(got) != cell.expected_trials:
        bad.append(f"{cell.key}: {len(got)} trials, expected {cell.expected_trials}")
    if len(pairs) != cell.expected_trials:
        bad.append(f"{cell.key}: {cell.expected_trials - len(pairs)} duplicate/missing cells")
    if synth:
        bad.append(f"{cell.key}: {synth} synthetic trials")
    if cens:
        bad.append(f"{cell.key}: {cens} budget-censored trials")
    if vers and vers != {pin}:
        bad.append(f"{cell.key}: agent versions {sorted(vers)}, pin is {pin}")
    if set(r["campaign_id"] for r in got) - {lib.CAMPAIGN_ID}:
        bad.append(f"{cell.key}: foreign campaign_id in corpus")

for b in bad:
    print(f"FAIL: {b}")
exp = lib.expected_totals()
print(f"  [{ctx}] {len(accepted)}/{exp['cells']} cells accepted, {len(rows)}/{exp['campaign_total']} trials")
sys.exit(1 if bad else 0)
PY
  audit "shard_validation_ok" "context=$ctx"
}

run_shard() {  # mode shard_index  -- ONE slice, then stop
  local mode="$1" shard="$2"
  RUN_CONTEXT="run-shard"

  # Fail-closed gate: unknown mode, out-of-range shard, unprepared dataset, or
  # an already-accepted shard are all refused here, BEFORE preflight and before
  # anything is launched.
  local plan_args=(--mode "$mode" --shard "$shard")
  [[ "$SHARD_NEW_ATTEMPT" -eq 1 ]] && plan_args+=(--new-attempt)
  say "run-shard $mode shard $shard"
  "$PY" -m campaign.shard plan "${plan_args[@]}" || {
    audit "shard_plan_refused" "mode=$mode" "shard=$shard"
    die "single-shard plan refused - nothing was launched."
  }
  info "pins     : claude-code=$CLAUDE_CODE_VERSION codex=$CODEX_VERSION mini-swe=$MINI_SWE_VERSION"
  info "root     : ${RESULTS_ROOT#$PROJECT_ROOT/}"
  if [[ "$SHARD_NEW_ATTEMPT" -eq 1 ]]; then
    info "mode     : NEW ATTEMPT - superseded attempts are preserved, not deleted"
  fi
  audit "shard_start" "mode=$mode" "shard=$shard" "new_attempt=$SHARD_NEW_ATTEMPT"

  # Identical contract to one iteration of run_mode's loop.
  do_preflight "$mode shard $shard (single)"
  run_profiles "$mode" "$shard"
  validate_progress "$mode shard $shard (single)"

  audit "shard_complete" "mode=$mode" "shard=$shard"
  say "run-shard $mode shard $shard complete - STOPPING as instructed"
  info "This command runs exactly one shard. Nothing further was launched."
  info "Next shard:  ./campaign.sh run-shard $CAMPAIGN_ID $mode $((shard + 1))"
  info "Final gate:  ./campaign.sh validate $CAMPAIGN_ID"
}

# ---------------------------------------------------------------------------
# All four profiles for ONE shard, under the bounded work-conserving scheduler.
#
# Replaces the old "claude, then fable, then codex, then llama" serialization:
# every profile that still has queued work progresses concurrently, freed
# capacity is redistributed immediately, and llama no longer waits behind the
# fixed profiles. Concurrency is bounded by per-profile caps, a global worker
# ceiling, and a throttle-down-only TPM safety ceiling.
#
# CAMPAIGN_SCHEDULER=legacy restores the old sequential path unchanged, as an
# operator escape hatch. The per-cell contract is identical either way: every
# attempt is recorded in provenance, failures are preserved, never backfilled.
run_profiles() {  # mode shard_index
  local mode="$1" shard="$2"

  if [[ "${CAMPAIGN_SCHEDULER:-adaptive}" == "legacy" ]]; then
    info "CAMPAIGN_SCHEDULER=legacy - running profiles sequentially (old behaviour)"
    for p in "${PROFILES[@]}"; do
      run_one "$mode" "$p" "$shard"
    done
    return
  fi

  say "run $mode / shard $shard / all ${#PROFILES[@]} profiles (bounded adaptive scheduler)"
  local resfile; resfile="$(mktemp)"
  local started; started="$($PY -c 'import datetime;print(datetime.datetime.now(datetime.timezone.utc).isoformat())')"
  audit "run_start" "mode=$mode" "profile=all" "shard=$shard" "scheduler=adaptive"

  local rc=0
  "$PY" -m campaign.controller --mode "$mode" --shard "$shard" \
      --result-file "$resfile" || rc=$?

  # Record every profile's attempt EITHER WAY, with every run directory the
  # scheduler produced for that cell (a cell may span several Harbor batches,
  # plus any permitted retries). A failed attempt stays in the ledger.
  local status="auto"; [[ "$rc" -ne 0 ]] && status="failed"
  local prov_rc=0
  for p in "${PROFILES[@]}"; do
    local dir_args=()
    while IFS= read -r d; do
      [[ -n "$d" ]] && dir_args+=(--run-dir "$d")
    done < <("$PY" - "$resfile" "$p" <<'PYEOF'
import json, sys
try:
    res = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
for d in (res.get("run_dirs") or {}).get(sys.argv[2], []) or []:
    print(d)
PYEOF
)
    if [[ ${#dir_args[@]} -eq 0 ]]; then
      audit "attempt_no_output" "mode=$mode" "profile=$p" "shard=$shard"
      prov_rc=1
      continue
    fi
    "$PY" -m campaign.provenance record \
      --mode "$mode" --profile "$p" --shard "$shard" \
      ${dir_args[@]+"${dir_args[@]}"} --status "$status" --started-at "$started" \
      --note "campaign.sh $RUN_CONTEXT $mode shard $shard (adaptive scheduler)" || prov_rc=1
  done
  rm -f "$resfile"

  if [[ "$rc" -ne 0 || "$prov_rc" -ne 0 ]]; then
    audit "attempt_failed" "mode=$mode" "shard=$shard" "controller_rc=$rc"
    die "$mode/shard $shard did NOT complete cleanly.
       Attempts are preserved in provenance/attempts.jsonl; the controller log is
       under logs/controller-$mode-shard$shard.jsonl.
       Campaign STOPPED. Do not backfill: rerun this shard with --new-attempt to
       create NEW attempt_ids, or investigate first."
  fi
  audit "run_ok" "mode=$mode" "profile=all" "shard=$shard"
}

run_mode() {  # full | resource
  local mode="$1"
  RUN_CONTEXT="run-$mode"
  local per_profile per_shard
  case "$mode" in
    full)     per_profile=700; per_shard=(300 300 100) ;;
    resource) per_profile=210; per_shard=(90 90 30) ;;
    *) die "unknown mode $mode" ;;
  esac
  say "run-$mode  |  ${#PROFILES[@]} profiles x ${#SHARDS[@]} shards = $((per_profile * ${#PROFILES[@]})) fresh trajectories"
  info "profiles : ${PROFILES[*]}"
  info "pins     : claude-code=$CLAUDE_CODE_VERSION codex=$CODEX_VERSION mini-swe=$MINI_SWE_VERSION"
  info "root     : ${RESULTS_ROOT#$PROJECT_ROOT/}"
  if [[ "$mode" == "resource" ]]; then
    info "note     : RESOURCE runs its OWN clean-n and eval-scaf controls. No FULL"
    info "           trajectory is substituted, by design."
  fi
  audit "mode_start" "mode=$mode"

  for i in "${SHARDS[@]}"; do
    do_preflight "$mode shard $i"
    run_profiles "$mode" "$i"
    validate_progress "$mode after shard $i"
  done

  audit "mode_complete" "mode=$mode"
  say "run-$mode complete"
  info "Now run:  ./campaign.sh validate $CAMPAIGN_ID"
}

# --------------------------------------------------------------------------- #
shift 2 || true

# run-shard takes two extra positionals: <mode> <shard>. Every other command's
# argument handling is untouched.
SHARD_MODE=""; SHARD_INDEX=""; SHARD_NEW_ATTEMPT=0
if [[ "$CMD" == "run-shard" ]]; then
  SHARD_MODE="${1:-}"; SHARD_INDEX="${2:-}"
  if [[ -z "$SHARD_MODE" || -z "$SHARD_INDEX" ]]; then
    echo "usage: ./campaign.sh run-shard $CAMPAIGN_ID_EXPECTED <full|resource> <1|2|3> [--new-attempt]" >&2
    exit 2
  fi
  shift 2 || true
  _rest=()
  for a in "$@"; do
    case "$a" in
      --new-attempt) SHARD_NEW_ATTEMPT=1 ;;
      *) _rest+=("$a") ;;
    esac
  done
  set -- ${_rest[@]+"${_rest[@]}"}
fi

EXTRA_ARGS=("$@")
PREFLIGHT_ARGS=("$@")

case "$CMD" in
  prepare)      do_prepare ;;
  preflight)    do_preflight standalone ;;
  run-full)     run_mode full ;;
  run-resource) run_mode resource ;;
  run-shard)    run_shard "$SHARD_MODE" "$SHARD_INDEX" ;;
  validate)
      say "validate $CAMPAIGN_ID (full campaign gate)"
      "$PY" -m campaign.provenance build || die "corpus build reported errors"
      "$PY" -m campaign.validate "$@" ;;
  analyze)
      say "analyze $CAMPAIGN_ID"
      "$PY" -m campaign.analyze "$@" ;;
esac
