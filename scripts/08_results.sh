#!/usr/bin/env bash

set -euo pipefail

MODE="${1:-full}"
TARGET="${2:-all}"
shift 2 || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

INPUT_ROOT=""
OUTPUT_DIR=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-root) INPUT_ROOT="${2:-}"; shift 2 ;;
    --output-dir) OUTPUT_DIR="${2:-}"; shift 2 ;;
    -h|--help)
      echo "Usage: ./lab.sh results <mode> <profile|all> [--input-root PATH] [--output-dir PATH] [--allow-partial] [08_results.py options]"
      exit 0 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

case "$MODE" in
  pilot|sample|full|resource) ;;
  *) echo "ERROR: unknown mode: $MODE" >&2; exit 2 ;;
esac

if [[ -z "$INPUT_ROOT" ]]; then
  if [[ "$TARGET" == "all" ]]; then
    INPUT_ROOT="$PROJECT_ROOT/analysis/$MODE"
  else
    INPUT_ROOT="$PROJECT_ROOT/analysis/$MODE/$TARGET"
  fi
fi

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$PROJECT_ROOT/analysis/results/$MODE/$TARGET"
fi

[[ -e "$INPUT_ROOT" ]] || { echo "ERROR: analysis input not found: $INPUT_ROOT" >&2; exit 1; }

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  exec python3 "$SCRIPT_DIR/08_results.py" \
    --input-root "$INPUT_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    "${EXTRA_ARGS[@]}"
else
  exec python3 "$SCRIPT_DIR/08_results.py" \
    --input-root "$INPUT_ROOT" \
    --output-dir "$OUTPUT_DIR"
fi
