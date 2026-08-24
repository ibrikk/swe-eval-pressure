#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-pilot}"; PROFILE="${2:-}"; shift 2 || true
[[ -n "$PROFILE" ]] || { echo "Usage: $0 <pilot|sample|full|resource> <claude|fable|codex|llama> [run options]" >&2; exit 2; }

ORIGINAL_ARGS=("$@")
SHARD_SPEC=""
SHARD_SIZE=""
SHARD_INDEX=""
INSTALL_ONLY=0
DATASET_OVERRIDE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --shard) SHARD_SPEC="${2:-}"; shift 2 ;;
    --shard-size) SHARD_SIZE="${2:-}"; shift 2 ;;
    --shard-index) SHARD_INDEX="${2:-}"; shift 2 ;;
    --install-only) INSTALL_ONLY=1; shift ;;
    --dataset-override) DATASET_OVERRIDE="${2:-}"; shift 2 ;;
    -h|--help) exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/05_run_profile.sh" "$MODE" "$PROFILE" --help ;;
    *) echo "Unknown run option: $1" >&2; exit 2 ;;
  esac
done
if [[ -n "$SHARD_SPEC" && ( -n "$SHARD_SIZE" || -n "$SHARD_INDEX" ) ]]; then
  echo "Use either --shard or --shard-size/--shard-index, not both." >&2; exit 2
fi
if [[ -n "$SHARD_SIZE" && -z "$SHARD_INDEX" ]] || [[ -z "$SHARD_SIZE" && -n "$SHARD_INDEX" ]]; then
  echo "--shard-size and --shard-index must be used together." >&2; exit 2
fi
if [[ -n "$DATASET_OVERRIDE" && ( -n "$SHARD_SPEC" || -n "$SHARD_SIZE" || -n "$SHARD_INDEX" ) ]]; then
  echo "--dataset-override cannot be combined with shard options." >&2; exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/00_common.sh"
load_env
cd "$PROJECT_ROOT"
parse_litellm_key_pool
key_count="${#LITELLM_KEYS_PARSED[@]}"
[[ "$key_count" -gt 0 ]] || die "set LITE_LLM_KEY or LITE_LLM_KEYS in .env"

# Install-only performs no model generation. A pool adds no value there.
if [[ "$key_count" -eq 1 || "$INSTALL_ONLY" == 1 ]]; then
  exec bash "$SCRIPT_DIR/05_run_profile.sh" "$MODE" "$PROFILE" "${ORIGINAL_ARGS[@]}"
fi

source_dataset="${DATASET_OVERRIDE:-$GENERATED_ROOT/$MODE/$PROFILE}"
require_dir "$source_dataset" "dataset; run ./lab.sh prepare $MODE"
if [[ -z "$DATASET_OVERRIDE" && -n "$SHARD_SPEC" ]]; then
  source_dataset="$(python3 "$SCRIPT_DIR/05_shard_dataset.py" --project-root "$PROJECT_ROOT" --mode "$MODE" --profile "$PROFILE" --shard "$SHARD_SPEC")"
elif [[ -z "$DATASET_OVERRIDE" && -n "$SHARD_SIZE" ]]; then
  source_dataset="$(python3 "$SCRIPT_DIR/05_shard_dataset.py" --project-root "$PROJECT_ROOT" --mode "$MODE" --profile "$PROFILE" --shard-size "$SHARD_SIZE" --shard-index "$SHARD_INDEX")"
fi

base_task_count="$(python3 - "$source_dataset/manifest.json" <<'PY'
import json, sys
from pathlib import Path
m=json.loads(Path(sys.argv[1]).read_text())
seen=[]
for item in m.get("tasks", []):
    bid=str(item.get("base_task_id", ""))
    if bid and bid not in seen:
        seen.append(bid)
print(len(seen))
PY
)"
[[ "$base_task_count" =~ ^[1-9][0-9]*$ ]] || die "selected dataset has no base tasks"
active_key_count="$key_count"
[[ "$active_key_count" -le "$base_task_count" ]] || active_key_count="$base_task_count"

printf 'Multi-key runner: profile=%s keys=%s active-key-shards=%s per-key-concurrency=%s TPM/key=%s\n' \
  "$PROFILE" "$key_count" "$active_key_count" "$HARBOR_CONCURRENCY" "$LITE_LLM_TPM_LIMIT"

work="$(mktemp -d "${TMPDIR:-/tmp}/swe-eval-pressure-keypool.XXXXXX")"
pids=(); output_files=(); key_indices=()
cleanup() {
  local p
  for p in "${pids[@]:-}"; do
    [[ -n "$p" ]] && kill "$p" 2>/dev/null || true
  done
  rm -rf "$work"
}
trap cleanup INT TERM EXIT

for ((i=0; i<active_key_count; i++)); do
  key_index=$((i+1))
  key="${LITELLM_KEYS_PARSED[$i]}"
  key_dataset="$(python3 "$SCRIPT_DIR/05_key_shard_dataset.py" \
    --project-root "$PROJECT_ROOT" \
    --source "$source_dataset" \
    --mode "$MODE" \
    --profile "$PROFILE" \
    --index "$key_index" \
    --total "$active_key_count")"
  output_file="$work/output-$key_index.txt"
  (
    export LITE_LLM_KEY="$key"
    export LITE_LLM_KEYS=""
    export SWE_LITELLM_KEY_INDEX="$key_index"
    export SWE_LITELLM_KEY_COUNT="$key_count"
    export SWE_RUN_OUTPUT_FILE="$output_file"
    exec bash "$SCRIPT_DIR/05_run_profile.sh" "$MODE" "$PROFILE" --dataset-override "$key_dataset"
  ) &
  pids+=("$!")
  output_files+=("$output_file")
  key_indices+=("$key_index")
done

status=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    status=1
  fi
done

# Retry latest rate-limit failures on a different key. Harbor's job config keeps
# credential placeholders, so the resume wrapper resolves them from the newly
# activated key without rewriting agent internals.
for i in "${!output_files[@]}"; do
  output_file="${output_files[$i]}"
  [[ -s "$output_file" ]] || continue
  outer_output="$(cat "$output_file")"
  [[ -d "$outer_output" ]] || continue
  config_path="$(find "$outer_output" -type f -name config.json -print 2>/dev/null | head -n1)"
  [[ -n "$config_path" ]] || continue
  harbor_job="${config_path%/config.json}"
  [[ -d "$harbor_job" ]] || continue
  remaining="$(python3 "$SCRIPT_DIR/11_rate_limit_failures.py" "$harbor_job")"
  [[ "$remaining" =~ ^[0-9]+$ ]] || remaining=0
  [[ "$remaining" -gt 0 ]] || continue

  original_index="${key_indices[$i]}"
  printf 'Rate-limit failover: %s has %s latest rate-limit failure(s); original key %s/%s.\n' \
    "$PROFILE" "$remaining" "$original_index" "$key_count" >&2
  for ((offset=1; offset<key_count && remaining>0; offset++)); do
    next_index=$(( (original_index - 1 + offset) % key_count + 1 ))
    next_key="${LITELLM_KEYS_PARSED[$((next_index-1))]}"
    printf '  retrying rate-limited work with key %s/%s\n' "$next_index" "$key_count" >&2
    LITE_LLM_KEY="$next_key" \
    LITE_LLM_KEYS="" \
    SWE_LITELLM_KEY_INDEX="$next_index" \
    SWE_LITELLM_KEY_COUNT="$key_count" \
      bash "$SCRIPT_DIR/11_resume_job.sh" "$PROFILE" "$harbor_job" --rate-limit-only || true
    remaining="$(python3 "$SCRIPT_DIR/11_rate_limit_failures.py" "$harbor_job")"
    [[ "$remaining" =~ ^[0-9]+$ ]] || remaining=0
  done
  if [[ "$remaining" -gt 0 ]]; then
    echo "ERROR: $remaining rate-limit failure(s) remain after trying alternate LiteLLM keys for $PROFILE." >&2
    status=1
  fi
done

trap - INT TERM EXIT
rm -rf "$work"
exit "$status"
