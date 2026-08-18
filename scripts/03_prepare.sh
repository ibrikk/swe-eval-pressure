#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-pilot}"
PROFILE_LIST="${2:-${PROFILES:-}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/00_common.sh"
load_env
[[ -n "$PROFILE_LIST" ]] || PROFILE_LIST="$PROFILES"
[[ -f "$MANIFEST_ROOT/task_registry.json" ]] || python3 "$SCRIPT_DIR/01_inventory.py" --project-root "$PROJECT_ROOT"
python3 "$SCRIPT_DIR/02_assign_cues.py" \
  --project-root "$PROJECT_ROOT" \
  --seed "$CUE_ASSIGNMENT_SEED" \
  --output "$CUE_ASSIGNMENT_FILE"
for profile in $PROFILE_LIST; do
  profile_values "$profile"
  python3 "$SCRIPT_DIR/03_prepare.py" \
    --project-root "$PROJECT_ROOT" --mode "$MODE" --profile "$profile" \
    --instruction-file "$PROFILE_INSTRUCTION_FILE" \
    --financial-index "$FINANCIAL_MESSAGE_INDEX" \
    --self-index "$SELF_PRESERVATION_MESSAGE_INDEX" \
    --resource-index "$RESOURCE_DEPRIVATION_MESSAGE_INDEX" \
    --assignment-file "$CUE_ASSIGNMENT_FILE" \
    --assignment-seed "$CUE_ASSIGNMENT_SEED" \
    --allow-internet "$ALLOW_INTERNET"
done
