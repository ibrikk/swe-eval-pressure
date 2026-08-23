#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-pilot}"
TARGET="${2:-all}"
shift 2 || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/00_common.sh"
load_env

RESULTS_DIR=""
MANIFEST=""
RUN_DIRS=()
STRICT_RECONSTRUCTION=0
CENSORED_TASK_ALLOWLIST=""
LIVE=0
SEMANTIC=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --results-dir) RESULTS_DIR="${2:-}"; shift 2 ;;
    --manifest) MANIFEST="${2:-}"; shift 2 ;;
    --run-dir) RUN_DIRS+=("${2:-}"); shift 2 ;;
    --strict-reconstruction) STRICT_RECONSTRUCTION=1; shift ;;
    --censored-task-allowlist) CENSORED_TASK_ALLOWLIST="${2:-}"; shift 2 ;;
    --live) LIVE=1; shift ;;
    --semantic) SEMANTIC=1; shift ;;
    --no-semantic) SEMANTIC=0; shift ;;
    -h|--help)
      cat <<'EOF'
Usage:
  ./lab.sh analyze <mode> <profile|all>
  ./lab.sh analyze <mode> <profile|all> --live
  ./lab.sh analyze <mode> <profile|all> --no-semantic

The LLM semantic judge is enabled by default (ANALYSIS_USE_LLM=1).
Use --no-semantic for deterministic-only analysis; --semantic forces it on.

Advanced:
  ./lab.sh analyze <mode> <profile> --results-dir /path/to/results
  ./lab.sh analyze <mode> <profile> --results-dir /path/to/run --manifest /path/to/manifest.json
  ./lab.sh analyze <mode> <profile> --run-dir /path/to/run --run-dir /path/to/repair --manifest /path/to/manifest.json
  ./lab.sh analyze <mode> <profile> --strict-reconstruction --censored-task-allowlist /path/to/allowlist.txt

Analysis is trajectory-first and automatically merges compatible shards. It only
forms matched effects when both trajectories in a within-task pair are available
and substantively usable.
EOF
      exit 0 ;;
    *) die "unknown analyze option: $1" ;;
  esac
done

if [[ "$TARGET" == all && -n "$RESULTS_DIR" ]]; then
  die "--results-dir with target=all is ambiguous; analyze one profile at a time"
fi
if [[ "$TARGET" == all && -n "$MANIFEST" ]]; then
  die "--manifest with target=all is ambiguous; analyze one profile at a time"
fi
if [[ "$TARGET" == all && "${#RUN_DIRS[@]}" -gt 0 ]]; then
  die "--run-dir with target=all is ambiguous; analyze one profile at a time"
fi
if [[ "$TARGET" == all && "$STRICT_RECONSTRUCTION" == 1 ]]; then
  die "--strict-reconstruction with target=all requires per-profile provenance; analyze one profile at a time"
fi
if [[ "$TARGET" == all && -n "$CENSORED_TASK_ALLOWLIST" ]]; then
  die "--censored-task-allowlist with target=all requires per-profile provenance; analyze one profile at a time"
fi

analyze_one() {
  local profile="$1"
  local output="$PROJECT_ROOT/analysis/$MODE/$profile"
  local args=(
    python3 "$SCRIPT_DIR/07_analyze.py"
    --project-root "$PROJECT_ROOT"
    --mode "$MODE"
    --profile "$profile"
    --output-dir "$output"
  )
  [[ -n "$RESULTS_DIR" ]] && args+=(--results-dir "$RESULTS_DIR")
  [[ -n "$MANIFEST" ]] && args+=(--manifest "$MANIFEST")
  for run_dir in "${RUN_DIRS[@]}"; do
    args+=(--run-dir "$run_dir")
  done
  [[ "$STRICT_RECONSTRUCTION" == 1 ]] && args+=(--strict-reconstruction)
  [[ -n "$CENSORED_TASK_ALLOWLIST" ]] && args+=(--censored-task-allowlist "$CENSORED_TASK_ALLOWLIST")
  [[ "$LIVE" == 1 ]] && args+=(--live)
  [[ "$SEMANTIC" == 1 ]] && args+=(--semantic)
  [[ "$SEMANTIC" == 0 ]] && args+=(--no-semantic)
  "${args[@]}"
}

if [[ "$TARGET" != all ]]; then
  analyze_one "$TARGET"
  exit
fi

mkdir -p "$PROJECT_ROOT/analysis/$MODE"
combined="$PROJECT_ROOT/analysis/$MODE/report-all.md"
: > "$combined"
ok=0
failures=0
for profile in $PROFILES; do
  echo "# $profile" >> "$combined"
  echo >> "$combined"
  if report="$(analyze_one "$profile" | tail -n1)" && [[ -f "$report" ]]; then
    cat "$report" >> "$combined"
    ok=1
  else
    echo "No analyzable run." >> "$combined"
    failures=$((failures + 1))
  fi
  printf '\n---\n\n' >> "$combined"
done
[[ "$ok" == 1 ]] || die "no analyzable profile results found"
[[ "$failures" == 0 ]] || die "$failures profile analysis run(s) failed; inspect stderr and semantic cache status"

echo "Combined report: $combined"
