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
  ./lab.sh matrix <mode> --concurrency-preset custom [run/shard options]
  ./lab.sh matrix <mode> --dry-run [options]

Concurrency presets:
  serial      1/1/1/1 for claude/fable/codex/llama.
  scale-200k  5/4/4/1, an empirically tested starting point for a 200k TPM key.
  custom      CLAUDE_CONCURRENCY/FABLE_CONCURRENCY/CODEX_CONCURRENCY/LLAMA_CONCURRENCY.

Always confirm the gateway key's TPM limit before using scale-200k on a new account.
EOF
      exit 0 ;;
    *)
      PASSTHRU+=("$1"); shift ;;
  esac
done

case "$PRESET" in
  serial)
    CLAUDE_CONCURRENCY=1; FABLE_CONCURRENCY=1; CODEX_CONCURRENCY=1; LLAMA_CONCURRENCY=1 ;;
  scale-200k)
    CLAUDE_CONCURRENCY=5; FABLE_CONCURRENCY=4; CODEX_CONCURRENCY=4; LLAMA_CONCURRENCY=1 ;;
  custom)
    : ;;
  *)
    die "unknown concurrency preset '$PRESET'; use serial, scale-200k, or custom" ;;
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

total=0
printf 'Matrix mode=%s preset=%s profiles=%s\n' "$MODE" "$PRESET" "$PROFILES"
for profile in $PROFILES; do
  n="$(profile_concurrency "$profile")"
  validate_concurrency "$profile" "$n"
  total=$((total+n))
  printf '  %-7s concurrency=%s\n' "$profile" "$n"
done
printf '  total active Harbor trials=%s\n' "$total"
if [[ "$PRESET" == "scale-200k" ]]; then
  echo '  note: scale-200k assumes a confirmed ~200000 TPM key; monitor 429s/TPM during the initial window.'
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
    exec bash "$SCRIPT_DIR/05_run_profile.sh" "$MODE" "$profile" "${PASSTHRU[@]}"
  ) &
  pids+=("$!"); names+=("$profile")
done
status=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then echo "FAILED: ${names[$i]}" >&2; status=1; fi
done
trap - INT TERM
exit "$status"
