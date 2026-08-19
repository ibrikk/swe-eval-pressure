#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

mode="${1:-full}"
profile="${2:-all}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ $# -gt 0 ]]; then shift; fi

cd "$PROJECT_ROOT"
exec python3 "$SCRIPT_DIR/08_build_report.py" \
  --mode "$mode" \
  --profile "$profile" \
  "$@"
