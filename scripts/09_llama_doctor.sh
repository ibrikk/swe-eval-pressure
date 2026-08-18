#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/00_common.sh"
load_env

HARBOR_PY="$(harbor_python)"
"$HARBOR_PY" - <<'PY'
from custom_agents.llama_textbased_mini_swe import LlamaTextBasedMiniSweAgent
print("Harbor custom-agent import: PASS")
PY

exec "$HARBOR_PY" "$SCRIPT_DIR/09_llama_doctor.py"
