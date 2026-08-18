#!/usr/bin/env bash
set -euo pipefail
PROFILE="${1:-}"
JOB_ARG="${2:-}"
shift 2 || true

[[ -n "$PROFILE" && -n "$JOB_ARG" ]] || {
  cat >&2 <<'HELP'
Usage:
  ./lab.sh resume <claude|fable|codex|llama> <harbor-job-dir> [--concurrency N] [--retry-nonzero] [--dry-run]

Default retry filters include transient/provider infrastructure failures only.
NonZeroAgentExitCodeError is NOT retried by default because it can represent a deterministic
agent-install/setup incompatibility. AgentSafetyRefusalError is never an infrastructure default.
HELP
  exit 2
}

NEW_CONCURRENCY=""
RETRY_NONZERO=0
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --concurrency) NEW_CONCURRENCY="${2:-}"; shift 2 ;;
    --retry-nonzero) RETRY_NONZERO=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) exec "$0" ;;
    *) echo "Unknown resume option: $1" >&2; exit 2 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/00_common.sh"
load_env
profile_values "$PROFILE"
cd "$PROJECT_ROOT"
require_command harbor
require_command jq

[[ -d "$JOB_ARG" ]] || die "job directory not found: $JOB_ARG"
JOB="$(cd "$JOB_ARG" && pwd)"
require_file "$JOB/config.json" "Harbor job config"
require_file "$JOB/lock.json" "Harbor job lock"

cfg_n="$(jq -r '.n_concurrent_trials' "$JOB/config.json")"
lock_n="$(jq -r '.n_concurrent_trials' "$JOB/lock.json")"
[[ "$cfg_n" == "$lock_n" ]] || die "config/lock concurrency mismatch before resume: config=$cfg_n lock=$lock_n"

if [[ -n "$NEW_CONCURRENCY" ]]; then
  [[ "$NEW_CONCURRENCY" =~ ^[1-9][0-9]*$ ]] || die "--concurrency must be a positive integer"
fi

filters=()
case "$PROFILE" in
  claude) filters=(CancelledError UnknownApiError RemoteError) ;;
  codex)  filters=(CancelledError ApiRateLimitError RemoteError) ;;
  fable)  filters=(CancelledError UnknownApiError ApiRateLimitError RemoteError) ;;
  llama)  filters=(CancelledError RemoteError AgentSetupTimeoutError NetworkConnectionError) ;;
  *) die "unknown profile '$PROFILE'" ;;
esac
if [[ "$RETRY_NONZERO" == 1 ]]; then
  filters+=(NonZeroAgentExitCodeError)
  echo 'WARNING: explicitly retrying NonZeroAgentExitCodeError; classify the setup failure first.' >&2
fi

printf 'Resume profile: %s\n' "$PROFILE"
printf 'Job: %s\n' "$JOB"
printf 'Current concurrency: %s\n' "$cfg_n"
printf 'Requested concurrency: %s\n' "${NEW_CONCURRENCY:-$cfg_n}"
printf 'Retry filters:'; printf ' %s' "${filters[@]}"; printf '\n'
printf 'AgentSafetyRefusalError retry: NO\n'
printf 'NonZeroAgentExitCodeError retry: %s\n' "$([[ "$RETRY_NONZERO" == 1 ]] && echo YES || echo NO)"

if [[ "$DRY_RUN" == 1 ]]; then
  echo 'Dry run: no files changed and Harbor not started.'
  exit 0
fi

stamp="$(date +%Y%m%d-%H%M%S)"
prov="$RESULTS_ROOT/_resume_provenance/$(basename "$JOB")/$stamp"
mkdir -p "$prov"
cp "$JOB/config.json" "$prov/config.before.json"
cp "$JOB/lock.json" "$prov/lock.before.json"

if [[ -n "$NEW_CONCURRENCY" && "$NEW_CONCURRENCY" != "$cfg_n" ]]; then
  jq --argjson n "$NEW_CONCURRENCY" '.n_concurrent_trials=$n' "$JOB/config.json" > "$JOB/config.tmp"
  mv "$JOB/config.tmp" "$JOB/config.json"
  jq --argjson n "$NEW_CONCURRENCY" '.n_concurrent_trials=$n' "$JOB/lock.json" > "$JOB/lock.tmp"
  mv "$JOB/lock.tmp" "$JOB/lock.json"
fi

cfg_after="$(jq -r '.n_concurrent_trials' "$JOB/config.json")"
lock_after="$(jq -r '.n_concurrent_trials' "$JOB/lock.json")"
[[ "$cfg_after" == "$lock_after" ]] || die "config/lock concurrency mismatch after edit"
cp "$JOB/config.json" "$prov/config.after.json"
cp "$JOB/lock.json" "$prov/lock.after.json"

metadata="$JOB/run_metadata.json"
[[ -f "$metadata" ]] || metadata="$(dirname "$JOB")/run_metadata.json"
if [[ -f "$metadata" ]]; then
  python3 - "$metadata" "$PROFILE" "$cfg_n" "$cfg_after" "$RETRY_NONZERO" "${filters[*]}" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
path, profile, before, after, retry_nonzero, filters = sys.argv[1:]
p = Path(path)
data = json.loads(p.read_text())
data.setdefault("resume_history", []).append({
    "at": datetime.now(timezone.utc).isoformat(),
    "profile": profile,
    "concurrency_before": int(before),
    "concurrency_after": int(after),
    "retry_filters": filters.split(),
    "retry_nonzero_agent_exit": retry_nonzero == "1",
    "orchestrator": "lab.sh resume",
})
p.write_text(json.dumps(data, indent=2) + "\n")
PY
fi

args=(harbor job resume -p "$JOB")
for f in "${filters[@]}"; do args+=(-f "$f"); done
printf 'Command:'; printf ' %q' "${args[@]}"; printf '\n'

if command -v caffeinate >/dev/null 2>&1; then
  exec caffeinate -dimsu "${args[@]}"
else
  exec "${args[@]}"
fi
