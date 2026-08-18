#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-pilot}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/00_common.sh"
load_env
[[ -f "$MANIFEST_ROOT/task_registry.json" ]] || python3 "$SCRIPT_DIR/01_inventory.py" --project-root "$PROJECT_ROOT"
python3 "$SCRIPT_DIR/02_assign_cues.py" --project-root "$PROJECT_ROOT" --seed "$CUE_ASSIGNMENT_SEED" --output "$CUE_ASSIGNMENT_FILE"
python3 "$SCRIPT_DIR/02_plan.py" --project-root "$PROJECT_ROOT" --mode "$MODE" --profiles "$PROFILES" --repeats "$HARBOR_REPEATS" --assignment-file "$CUE_ASSIGNMENT_FILE"
