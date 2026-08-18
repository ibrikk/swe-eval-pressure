#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-pilot}"
TARGET="${2:-all}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/00_common.sh"
load_env
if [[ "$TARGET" == all ]]; then list="$PROFILES"; else list="$TARGET"; fi
for profile in $list; do
  profile_values "$profile"
  python3 "$SCRIPT_DIR/04_validate.py" --project-root "$PROJECT_ROOT" --mode "$MODE" --profile "$profile"
done
