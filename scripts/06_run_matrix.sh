#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-pilot}"; shift || true
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/00_common.sh"
load_env

PRESET="$MATRIX_CONCURRENCY_PRESET"
DRY_RUN=0
PASSTHRU=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --concurrency-preset)
      PRESET="${2:-}"; shift 2 ;;
    --dry-run)
      DRY_RUN=1; shift ;;
    -h|--help)
      cat <<'EOF'
Usage:
  ./lab.sh matrix <mode> [run/shard options]
  ./lab.sh matrix <mode> --concurrency-preset serial [run/shard options]
  ./lab.sh matrix <mode> --concurrency-preset scale-200k [run/shard options]
  ./lab.sh matrix <mode> --concurrency-preset scale-5m [run/shard options]
  ./lab.sh matrix <mode> --concurrency-preset scale-5m-adaptive [run/shard options]
  ./lab.sh matrix <mode> --concurrency-preset custom [run/shard options]
  ./lab.sh matrix <mode> --dry-run [options]

Concurrency presets:
  serial      1/1/1/1 for claude/fable/codex/llama.
  scale-200k  5/4/4/1, the existing empirically tested starting point for a 200k TPM key.
  scale-5m    20/16/16/4, a conservative high-throughput starting point for a confirmed 5M TPM key.
  scale-5m-adaptive  fixed Claude/Fable/Codex plus globally elastic Llama (starts at 1; AIMD feedback).
  custom      CLAUDE_CONCURRENCY/FABLE_CONCURRENCY/CODEX_CONCURRENCY/LLAMA_CONCURRENCY.

Always confirm the gateway key's TPM limit before selecting a quota-specific preset.
With multiple LITE_LLM_KEYS, these concurrency values apply per key; the runner shards work across keys automatically.
EOF
      exit 0 ;;
    *)
      PASSTHRU+=("$1"); shift ;;
  esac
done


if [[ "$PRESET" == "scale-5m" || "$PRESET" == "scale-5m-adaptive" ]]; then
  export LITE_LLM_TPM_LIMIT=5000000
fi

if [[ "$PRESET" == "scale-5m-adaptive" ]]; then
  adaptive_args=("$MODE")
  [[ "$DRY_RUN" == 1 ]] && adaptive_args+=(--dry-run)
  if [[ -n "${PASSTHRU[*]-}" ]]; then
    adaptive_args+=("${PASSTHRU[@]}")
  fi
  exec bash "$SCRIPT_DIR/06_run_matrix_adaptive.sh" "${adaptive_args[@]}"
fi

case "$PRESET" in
  serial)
    CLAUDE_CONCURRENCY=1; FABLE_CONCURRENCY=1; CODEX_CONCURRENCY=1; LLAMA_CONCURRENCY=1 ;;
  scale-200k)
    CLAUDE_CONCURRENCY=5; FABLE_CONCURRENCY=4; CODEX_CONCURRENCY=4; LLAMA_CONCURRENCY=1 ;;
  scale-5m)
    CLAUDE_CONCURRENCY=20; FABLE_CONCURRENCY=16; CODEX_CONCURRENCY=16; LLAMA_CONCURRENCY=4 ;;
  custom)
    : ;;
  *)
    die "unknown concurrency preset '$PRESET'; use serial, scale-200k, scale-5m, or custom" ;;
esac

profile_concurrency() {
  case "$1" in
    claude) printf '%s\n' "$CLAUDE_CONCURRENCY" ;;
    fable)  printf '%s\n' "$FABLE_CONCURRENCY" ;;
    codex)  printf '%s\n' "$CODEX_CONCURRENCY" ;;
    llama)  printf '%s\n' "$LLAMA_CONCURRENCY" ;;
    *) die "unknown profile '$1'" ;;
  esac
}

validate_concurrency() {
  local profile="$1" n="$2"
  [[ "$n" =~ ^[1-9][0-9]*$ ]] || die "$profile concurrency must be a positive integer, got '$n'"
}

parse_litellm_key_pool
key_count="${#LITELLM_KEYS_PARSED[@]}"
[[ "$key_count" -gt 0 ]] || key_count=1
total=0
printf 'Matrix mode=%s preset=%s profiles=%s keys=%s TPM/key=%s\n' "$MODE" "$PRESET" "$PROFILES" "$key_count" "$LITE_LLM_TPM_LIMIT"
for profile in $PROFILES; do
  n="$(profile_concurrency "$profile")"
  validate_concurrency "$profile" "$n"
  total=$((total+n))
  printf '  %-7s concurrency=%s\n' "$profile" "$n"
done
printf '  per-key active Harbor trials=%s\n' "$total"
printf '  aggregate active Harbor trials across key pool=%s\n' "$((total * key_count))"
if [[ "$PRESET" == "scale-200k" ]]; then
  echo '  note: scale-200k preserves the empirically tested ~200000 TPM allocation per key.'
elif [[ "$PRESET" == "scale-5m" ]]; then
  echo '  note: scale-5m assumes a confirmed ~5000000 TPM limit per key; it is a conservative starting point, not an empirically optimized ceiling.'
  if [[ "$LITE_LLM_TPM_LIMIT" =~ ^[0-9]+$ ]] && [[ "$LITE_LLM_TPM_LIMIT" -lt 5000000 ]]; then
    echo "  WARNING: configured LITE_LLM_TPM_LIMIT=$LITE_LLM_TPM_LIMIT is below 5000000; use scale-200k/custom unless the gateway quota was upgraded." >&2
  fi
fi

if [[ "$DRY_RUN" == 1 ]]; then
  echo 'Dry run: no Harbor jobs started.'
  exit 0
fi

pids=(); names=()

# Print descendants of a PID in post-order (deepest children first).
# This lets an interrupted matrix terminate only the process trees that it
# started, instead of using broad pkill/killall patterns that could affect
# unrelated Harbor jobs.
descendants() {
  local parent="$1" child
  while IFS= read -r child; do
    [[ -n "$child" ]] || continue
    descendants "$child"
    printf '%s\n' "$child"
  done < <(pgrep -P "$parent" 2>/dev/null || true)
}

signal_matrix_children() {
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
  local sig="$1" code=143
  [[ "$sig" == "INT" ]] && code=130

  # Avoid recursively re-entering this handler while tearing down children.
  trap - INT TERM
  echo "Interrupted: stopping matrix-owned profile/Harbor process trees..." >&2

  signal_matrix_children TERM
  sleep 2
  signal_matrix_children KILL

  # Reap anything that has already exited; never fail cleanup because of wait.
  for p in "${pids[@]:-}"; do
    wait "$p" 2>/dev/null || true
  done
  exit "$code"
}

trap 'on_interrupt INT' INT
trap 'on_interrupt TERM' TERM
for profile in $PROFILES; do
  n="$(profile_concurrency "$profile")"
  (
    export HARBOR_CONCURRENCY="$n"
    # macOS ships Bash 3.2. With `set -u`, expanding an empty array via
    # "${PASSTHRU[@]}" raises "unbound variable". Avoid expanding it when empty.
    if [[ -n "${PASSTHRU[*]-}" ]]; then
      exec bash "$SCRIPT_DIR/05_run_profile_pool.sh" "$MODE" "$profile" "${PASSTHRU[@]}"
    else
      exec bash "$SCRIPT_DIR/05_run_profile_pool.sh" "$MODE" "$profile"
    fi
  ) &
  pids+=("$!"); names+=("$profile")
done
status=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then echo "FAILED: ${names[$i]}" >&2; status=1; fi
done
trap - INT TERM
exit "$status"
