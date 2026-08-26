#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-full}"
shift || true
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/00_common.sh"
load_env

DRY_RUN=0
SHARD_SPEC=""
SHARD_SIZE=""
SHARD_INDEX=""
LLAMA_START_BASE=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --shard) SHARD_SPEC="${2:-}"; shift 2 ;;
    --shard-size) SHARD_SIZE="${2:-}"; shift 2 ;;
    --shard-index) SHARD_INDEX="${2:-}"; shift 2 ;;
    --llama-start-base) LLAMA_START_BASE="${2:-}"; shift 2 ;;
    -h|--help)
      cat <<'EOF'
Usage:
  ./lab.sh matrix <mode> --concurrency-preset scale-5m-adaptive [--shard 1/3]
  ./lab.sh matrix <mode> --concurrency-preset scale-5m-adaptive --shard-size 30 --shard-index 1
  PROFILES=llama ./lab.sh matrix <mode> --concurrency-preset scale-5m-adaptive --shard-size 30 --shard-index 1 --llama-start-base 9

Fixed profiles use the 5M starting allocation per key:
  Claude 20, Fable 16, Codex 16.

Llama is globally elastic rather than per-key static. It starts conservatively,
runs base-task-complete batches on one rotating LiteLLM key at a time, grows
multiplicatively after clean batches, and halves after a rate-limit event. It
also observes completion of the fixed Claude/Fable/Codex jobs: as those jobs
finish, Llama receives progressively higher concurrency floors; once all fixed
profiles are done it immediately jumps to ADAPTIVE_LLAMA_MAX (default 40) and,
by default, drains the entire remaining Llama shard in one continuous Harbor job.
This removes the 20-trajectory micro-batch barrier once Llama has the gateway to itself.

--llama-start-base is 1-based and is intended for provenance-safe catch-up after
a stopped adaptive controller. Example: --llama-start-base 9 reruns base task
families 9 onward in the selected outer shard.

Feedback is based on actual HTTP/rate-limit behavior because the gateway does
not expose an authoritative live remaining-TPM counter to this runner.
EOF
      exit 0 ;;
    *) die "unknown adaptive matrix option: $1" ;;
  esac
done

if [[ -n "$SHARD_SPEC" && ( -n "$SHARD_SIZE" || -n "$SHARD_INDEX" ) ]]; then
  die "use either --shard or --shard-size/--shard-index, not both"
fi
if [[ -n "$SHARD_SIZE" && -z "$SHARD_INDEX" ]] || [[ -z "$SHARD_SIZE" && -n "$SHARD_INDEX" ]]; then
  die "--shard-size and --shard-index must be supplied together"
fi

parse_litellm_key_pool
key_count="${#LITELLM_KEYS_PARSED[@]}"
[[ "$key_count" -gt 0 ]] || die "set LITE_LLM_KEY or LITE_LLM_KEYS"

CLAUDE_FIXED="${ADAPTIVE_CLAUDE_CONCURRENCY:-20}"
FABLE_FIXED="${ADAPTIVE_FABLE_CONCURRENCY:-16}"
CODEX_FIXED="${ADAPTIVE_CODEX_CONCURRENCY:-16}"
LLAMA_MIN="${ADAPTIVE_LLAMA_MIN:-4}"
LLAMA_MAX="${ADAPTIVE_LLAMA_MAX:-40}"
LLAMA_STEP="${ADAPTIVE_LLAMA_STEP:-1}"
LLAMA_GROWTH="${ADAPTIVE_LLAMA_GROWTH:-2}"
LLAMA_FLOOR_TWO_FIXED="${ADAPTIVE_LLAMA_FLOOR_TWO_FIXED:-12}"
LLAMA_FLOOR_ONE_FIXED="${ADAPTIVE_LLAMA_FLOOR_ONE_FIXED:-24}"
BATCH_BASE_TASKS="${ADAPTIVE_LLAMA_BATCH_BASE_TASKS:-2}"
DRAIN_WHEN_FIXED_DONE="${ADAPTIVE_LLAMA_DRAIN_WHEN_FIXED_DONE:-1}"
COOLDOWN_SECONDS="${ADAPTIVE_LLAMA_COOLDOWN_SECONDS:-30}"
RESUME_ATTEMPTS="${ADAPTIVE_LLAMA_RESUME_ATTEMPTS:-6}"

for pair in \
  "claude:$CLAUDE_FIXED" \
  "fable:$FABLE_FIXED" \
  "codex:$CODEX_FIXED" \
  "llama-min:$LLAMA_MIN" \
  "llama-max:$LLAMA_MAX" \
  "llama-step:$LLAMA_STEP" \
  "llama-growth:$LLAMA_GROWTH" \
  "llama-floor-two-fixed:$LLAMA_FLOOR_TWO_FIXED" \
  "llama-floor-one-fixed:$LLAMA_FLOOR_ONE_FIXED" \
  "llama-batch:$BATCH_BASE_TASKS" \
  "llama-resume:$RESUME_ATTEMPTS"; do
  name="${pair%%:*}"; value="${pair#*:}"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "$name must be a positive integer, got '$value'"
done
[[ "$COOLDOWN_SECONDS" =~ ^[0-9]+$ ]] || die "adaptive cooldown must be a non-negative integer"
[[ "$DRAIN_WHEN_FIXED_DONE" == 0 || "$DRAIN_WHEN_FIXED_DONE" == 1 ]] || die "ADAPTIVE_LLAMA_DRAIN_WHEN_FIXED_DONE must be 0 or 1"
[[ "$LLAMA_MAX" -ge "$LLAMA_MIN" ]] || die "ADAPTIVE_LLAMA_MAX must be >= ADAPTIVE_LLAMA_MIN"
[[ "$LLAMA_START_BASE" =~ ^[1-9][0-9]*$ ]] || die "--llama-start-base must be a positive 1-based integer"

if [[ "$LITE_LLM_TPM_LIMIT" =~ ^[0-9]+$ ]] && [[ "$LITE_LLM_TPM_LIMIT" -lt 5000000 ]]; then
  echo "WARNING: LITE_LLM_TPM_LIMIT=$LITE_LLM_TPM_LIMIT is below 5000000." >&2
fi

contains_profile() {
  case " $PROFILES " in
    *" $1 "*) return 0 ;;
    *) return 1 ;;
  esac
}

outer_args=()
if [[ -n "$SHARD_SPEC" ]]; then
  outer_args+=(--shard "$SHARD_SPEC")
elif [[ -n "$SHARD_SIZE" ]]; then
  outer_args+=(--shard-size "$SHARD_SIZE" --shard-index "$SHARD_INDEX")
fi

llama_outer="$GENERATED_ROOT/$MODE/llama"
if contains_profile llama; then
  require_dir "$llama_outer" "llama dataset; run ./lab.sh prepare $MODE"
  if [[ -n "$SHARD_SPEC" ]]; then
    llama_outer="$(python3 "$SCRIPT_DIR/05_shard_dataset.py" --project-root "$PROJECT_ROOT" --mode "$MODE" --profile llama --shard "$SHARD_SPEC")"
  elif [[ -n "$SHARD_SIZE" ]]; then
    llama_outer="$(python3 "$SCRIPT_DIR/05_shard_dataset.py" --project-root "$PROJECT_ROOT" --mode "$MODE" --profile llama --shard-size "$SHARD_SIZE" --shard-index "$SHARD_INDEX")"
  fi
fi

base_task_count=0
if contains_profile llama; then
  base_task_count="$(python3 - "$llama_outer/manifest.json" <<'PY'
import json, sys
from pathlib import Path
m=json.loads(Path(sys.argv[1]).read_text())
seen=[]
for item in m.get("tasks", []):
    b=str(item.get("base_task_id", ""))
    if b and b not in seen: seen.append(b)
print(len(seen))
PY
)"
  [[ "$base_task_count" =~ ^[1-9][0-9]*$ ]] || die "adaptive llama source has no base tasks"
fi

printf 'Adaptive 5M matrix mode=%s profiles=%s keys=%s TPM/key=%s\n' "$MODE" "$PROFILES" "$key_count" "$LITE_LLM_TPM_LIMIT"
contains_profile claude && printf '  claude fixed per-key concurrency=%s\n' "$CLAUDE_FIXED"
contains_profile fable  && printf '  fable  fixed per-key concurrency=%s\n' "$FABLE_FIXED"
contains_profile codex  && printf '  codex  fixed per-key concurrency=%s\n' "$CODEX_FIXED"
if contains_profile llama; then
  printf '  llama global adaptive concurrency=%s..%s growth=x%s step=%s batch_base_tasks=%s start_base=%s drain_when_fixed_done=%s\n' "$LLAMA_MIN" "$LLAMA_MAX" "$LLAMA_GROWTH" "$LLAMA_STEP" "$BATCH_BASE_TASKS" "$LLAMA_START_BASE" "$DRAIN_WHEN_FIXED_DONE"
  printf '  llama selected outer base tasks=%s; initial active Llama trials globally=%s (not multiplied by key count)\n' "$base_task_count" "$LLAMA_MIN"
fi
printf '  results root=%s\n' "$RESULTS_ROOT"
printf '  feedback=phase-aware floors + multiplicative clean growth; rate-limit multiplicative decrease\n'

if [[ "$DRY_RUN" == 1 ]]; then
  echo 'Dry run: no Harbor jobs started.'
  exit 0
fi

stamp="$(date +%Y%m%d-%H%M%S)"
controller_dir="$RESULTS_ROOT/$MODE/adaptive-controller-$stamp"
mkdir -p "$controller_dir"
printf 'batch\tbase_start\tbase_count\tkey_index\tfixed_profiles_alive\tconcurrency_before\trate_event\tremaining_after_resume\tconcurrency_after\touter_output\n' > "$controller_dir/llama_decisions.tsv"

pids=(); names=()
descendants() {
  local parent="$1" child
  while IFS= read -r child; do
    [[ -n "$child" ]] || continue
    descendants "$child"
    printf '%s\n' "$child"
  done < <(pgrep -P "$parent" 2>/dev/null || true)
}
signal_children() {
  local sig="$1" p child
  for p in "${pids[@]:-}"; do
    [[ -n "$p" ]] || continue
    while IFS= read -r child; do
      [[ -n "$child" ]] || continue
      kill "-$sig" "$child" 2>/dev/null || true
    done < <(descendants "$p")
    kill "-$sig" "$p" 2>/dev/null || true
  done
}
on_interrupt() {
  trap - INT TERM
  echo 'Interrupted: stopping adaptive matrix-owned process trees...' >&2
  signal_children TERM
  sleep 2
  signal_children KILL
  for p in "${pids[@]:-}"; do wait "$p" 2>/dev/null || true; done
  exit 130
}
trap 'on_interrupt' INT TERM

launch_fixed() {
  local profile="$1" concurrency="$2"
  (
    export HARBOR_CONCURRENCY="$concurrency"
    if [[ "${#outer_args[@]}" -gt 0 ]]; then
      exec bash "$SCRIPT_DIR/05_run_profile_pool.sh" "$MODE" "$profile" "${outer_args[@]}"
    else
      exec bash "$SCRIPT_DIR/05_run_profile_pool.sh" "$MODE" "$profile"
    fi
  ) &
  pids+=("$!"); names+=("$profile")
}

contains_profile claude && launch_fixed claude "$CLAUDE_FIXED"
contains_profile fable  && launch_fixed fable "$FABLE_FIXED"
contains_profile codex  && launch_fixed codex "$CODEX_FIXED"

fixed_alive_count() {
  local count=0 i
  for i in "${!pids[@]}"; do
    case "${names[$i]}" in
      claude|fable|codex)
        if kill -0 "${pids[$i]}" 2>/dev/null; then count=$((count + 1)); fi
        ;;
    esac
  done
  printf '%s\n' "$count"
}

phase_floor() {
  local alive="$1"
  case "$alive" in
    0) printf '%s\n' "$LLAMA_MAX" ;;
    1) printf '%s\n' "$LLAMA_FLOOR_ONE_FIXED" ;;
    2) printf '%s\n' "$LLAMA_FLOOR_TWO_FIXED" ;;
    *) printf '%s\n' "$LLAMA_MIN" ;;
  esac
}

adaptive_status=0
if contains_profile llama; then
  current="$LLAMA_MIN"
  start=$((LLAMA_START_BASE - 1))
  [[ "$start" -le "$base_task_count" ]] || die "--llama-start-base $LLAMA_START_BASE exceeds selected shard base-task count $base_task_count"
  batch=0
  backoff_active=0
  adaptive_root="$GENERATED_ROOT/_adaptive/$MODE/llama/$stamp"
  mkdir -p "$adaptive_root"

  while [[ "$start" -lt "$base_task_count" ]]; do
    fixed_alive="$(fixed_alive_count)"
    floor="$(phase_floor "$fixed_alive")"
    [[ "$floor" -le "$LLAMA_MAX" ]] || floor="$LLAMA_MAX"
    if [[ "$backoff_active" == 0 && "$current" -lt "$floor" ]]; then
      printf "  capacity phase change: fixed profiles alive=%s -> raise Llama floor %s -> %s\n" "$fixed_alive" "$current" "$floor"
      current="$floor"
    fi
    batch=$((batch + 1))
    remaining_tasks=$((base_task_count - start))
    count="$BATCH_BASE_TASKS"
    drain_mode=0
    if [[ "$fixed_alive" -eq 0 && "$DRAIN_WHEN_FIXED_DONE" == 1 ]]; then
      count="$remaining_tasks"
      current="$LLAMA_MAX"
      drain_mode=1
      printf '  fixed profiles complete -> drain remaining %s base task(s) continuously at Llama concurrency=%s\n' "$remaining_tasks" "$current"
    fi
    [[ "$count" -le "$remaining_tasks" ]] || count="$remaining_tasks"
    batch_dir="$adaptive_root/batch-$(printf '%03d' "$batch")"
    python3 "$SCRIPT_DIR/05_slice_dataset.py" \
      --project-root "$PROJECT_ROOT" \
      --source "$llama_outer" \
      --output "$batch_dir" \
      --start-index "$start" \
      --base-task-count "$count" \
      --label "adaptive-batch-$batch" >/dev/null

    key_index=$(( (batch - 1) % key_count + 1 ))
    key="${LITELLM_KEYS_PARSED[$((key_index - 1))]}"
    pointer="$controller_dir/llama-batch-$(printf '%03d' "$batch").output-path"
    log="$controller_dir/llama-batch-$(printf '%03d' "$batch").log"
    before="$current"

    printf '\nAdaptive Llama batch %s: base tasks %s..%s, key %s/%s, fixed profiles alive=%s, global concurrency=%s, drain=%s\n' \
      "$batch" "$((start + 1))" "$((start + count))" "$key_index" "$key_count" "$fixed_alive" "$current" "$drain_mode"

    set +e
    (
      export LITE_LLM_KEY="$key"
      export LITE_LLM_KEYS=""
      export SWE_LITELLM_KEY_INDEX="$key_index"
      export SWE_LITELLM_KEY_COUNT="$key_count"
      export SWE_RUN_OUTPUT_FILE="$pointer"
      export HARBOR_CONCURRENCY="$current"
      exec bash "$SCRIPT_DIR/05_run_profile.sh" "$MODE" llama --dataset-override "$batch_dir"
    ) 2>&1 | tee "$log"
    batch_status="${PIPESTATUS[0]}"
    set -e

    outer_output=""
    [[ -s "$pointer" ]] && outer_output="$(cat "$pointer")"
    harbor_job=""
    if [[ -n "$outer_output" && -d "$outer_output" ]]; then
      config_path="$(find "$outer_output" -type f -name config.json -print 2>/dev/null | head -n1)"
      [[ -n "$config_path" ]] && harbor_job="${config_path%/config.json}"
    fi

    rate_event=0
    if grep -Eqi 'rate.?limit|too many requests|(^|[^0-9])429([^0-9]|$)' "$log"; then
      rate_event=1
    fi
    remaining=0
    if [[ -n "$harbor_job" ]]; then
      remaining="$(python3 "$SCRIPT_DIR/11_rate_limit_failures.py" "$harbor_job" 2>/dev/null || echo 0)"
      [[ "$remaining" =~ ^[0-9]+$ ]] || remaining=0
      [[ "$remaining" -gt 0 ]] && rate_event=1
    fi

    if [[ "$rate_event" == 1 ]]; then
      current=$((current / 2))
      [[ "$current" -ge "$LLAMA_MIN" ]] || current="$LLAMA_MIN"
      backoff_active=1
      printf '  rate-limit feedback -> backoff global Llama concurrency to %s\n' "$current" >&2
    fi

    if [[ "$remaining" -gt 0 ]]; then
      [[ -n "$harbor_job" ]] || die "rate-limited batch has no recoverable Harbor job path"
      attempt=0
      while [[ "$remaining" -gt 0 && "$attempt" -lt "$RESUME_ATTEMPTS" ]]; do
        attempt=$((attempt + 1))
        [[ "$COOLDOWN_SECONDS" -eq 0 ]] || sleep "$COOLDOWN_SECONDS"
        printf '  resume %s/%s: %s rate-limited trial(s), concurrency=%s\n' "$attempt" "$RESUME_ATTEMPTS" "$remaining" "$current" >&2
        set +e
        LITE_LLM_KEY="$key" LITE_LLM_KEYS="" \
        SWE_LITELLM_KEY_INDEX="$key_index" SWE_LITELLM_KEY_COUNT="$key_count" \
          bash "$SCRIPT_DIR/11_resume_job.sh" llama "$harbor_job" --concurrency "$current" --rate-limit-only
        resume_status=$?
        set -e
        remaining="$(python3 "$SCRIPT_DIR/11_rate_limit_failures.py" "$harbor_job" 2>/dev/null || echo 0)"
        [[ "$remaining" =~ ^[0-9]+$ ]] || remaining=0
        if [[ "$resume_status" -ne 0 && "$remaining" -eq 0 ]]; then
          echo "WARNING: resume command exited $resume_status but no latest rate-limit failures remain." >&2
        fi
      done
      if [[ "$remaining" -gt 0 ]]; then
        echo "ERROR: adaptive Llama batch $batch still has $remaining rate-limit failure(s) after $RESUME_ATTEMPTS resume attempts." >&2
        adaptive_status=1
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$batch" "$start" "$count" "$key_index" "$fixed_alive" "$before" "$rate_event" "$remaining" "$current" "$outer_output" >> "$controller_dir/llama_decisions.tsv"
        break
      fi
    fi

    if [[ "$batch_status" -ne 0 && "$rate_event" -eq 0 ]]; then
      echo "ERROR: adaptive Llama batch $batch failed for a non-rate-limit reason; inspect $log" >&2
      adaptive_status=1
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$batch" "$start" "$count" "$key_index" "$fixed_alive" "$before" "$rate_event" "$remaining" "$current" "$outer_output" >> "$controller_dir/llama_decisions.tsv"
      break
    fi

    if [[ "$rate_event" -eq 0 ]]; then
      additive=$((current + LLAMA_STEP))
      multiplicative=$((current * LLAMA_GROWTH))
      [[ "$multiplicative" -ge "$additive" ]] || multiplicative="$additive"
      current="$multiplicative"
      [[ "$current" -le "$LLAMA_MAX" ]] || current="$LLAMA_MAX"
      backoff_active=0
      printf '  clean batch -> raise next global Llama concurrency to %s\n' "$current"
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$batch" "$start" "$count" "$key_index" "$fixed_alive" "$before" "$rate_event" "$remaining" "$current" "$outer_output" >> "$controller_dir/llama_decisions.tsv"
    start=$((start + count))
  done
fi

status="$adaptive_status"
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "FAILED: ${names[$i]}" >&2
    status=1
  fi
done
trap - INT TERM

printf '\nAdaptive controller log: %s\n' "$controller_dir/llama_decisions.tsv"
exit "$status"
