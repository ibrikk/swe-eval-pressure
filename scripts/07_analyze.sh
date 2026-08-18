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
LIVE=0
SEMANTIC=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --results-dir) RESULTS_DIR="${2:-}"; shift 2 ;;
    --manifest) MANIFEST="${2:-}"; shift 2 ;;
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
for profile in $PROFILES; do
  echo "# $profile" >> "$combined"
  echo >> "$combined"
  if report="$(analyze_one "$profile" | tail -n1)" && [[ -f "$report" ]]; then
    cat "$report" >> "$combined"
    ok=1
  else
    echo "No analyzable run." >> "$combined"
  fi
  printf '\n---\n\n' >> "$combined"
done
[[ "$ok" == 1 ]] || die "no analyzable profile results found"
echo "Combined report: $combined"
