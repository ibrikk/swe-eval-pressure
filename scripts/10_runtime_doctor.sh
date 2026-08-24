#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/00_common.sh"
load_env
cd "$PROJECT_ROOT"

fail=0
check_cmd() {
  if command -v "$1" >/dev/null 2>&1; then
    printf '[PASS] command %-10s %s\n' "$1" "$(command -v "$1")"
  else
    printf '[FAIL] command %-10s missing\n' "$1"
    fail=1
  fi
}

for cmd in python3 harbor modal jq curl; do check_cmd "$cmd"; done

parse_litellm_key_pool
if [[ "${#LITELLM_KEYS_PARSED[@]}" -gt 0 ]]; then
  printf '[PASS] LiteLLM key pool configured: %d key(s)\n' "${#LITELLM_KEYS_PARSED[@]}"
  printf '[PASS] LiteLLM TPM/key config: %s\n' "$LITE_LLM_TPM_LIMIT"
else
  printf '[FAIL] LITE_LLM_KEY/LITE_LLM_KEYS empty\n'
  fail=1
fi

for var in OPENAI_API_KEY OPENAI_API_BASE OPENAI_BASE_URL ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN ANTHROPIC_BASE_URL MSWEA_API_KEY PYTHONPATH; do
  if [[ -n "${!var:-}" ]]; then
    printf '[PASS] env %-24s set\n' "$var"
  else
    printf '[FAIL] env %-24s empty\n' "$var"
    fail=1
  fi
done

if [[ -n "${LITE_LLM_KEY:-}" && "${OPENAI_API_KEY:-}" == "${LITE_LLM_KEY:-}" && "${MSWEA_API_KEY:-}" == "${LITE_LLM_KEY:-}" ]]; then
  printf '[PASS] active LiteLLM/OpenAI/Mini-SWE key aliases are consistent\n'
else
  printf '[FAIL] active OpenAI/Mini-SWE key aliases differ from selected LiteLLM key\n'
  fail=1
fi
printf '[PASS] semantic workers configured: %s\n' "$ANALYSIS_SEMANTIC_WORKERS"
if [[ "${ANALYSIS_MAX_CHARS:-}" == "0" ]]; then
  printf '[PASS] semantic final truncation: disabled (ANALYSIS_MAX_CHARS=0)\n'
else
  printf '[WARN] semantic final truncation enabled: ANALYSIS_MAX_CHARS=%s; paper-grade primary analysis uses 0\n' "$ANALYSIS_MAX_CHARS"
fi
if [[ "${ANALYSIS_MAX_RETRIES:-0}" =~ ^[0-9]+$ ]] && [[ "$ANALYSIS_MAX_RETRIES" -ge 6 ]]; then
  printf '[PASS] semantic retries configured: %s\n' "$ANALYSIS_MAX_RETRIES"
else
  printf '[WARN] semantic retries configured: %s; paper-grade primary analysis uses at least 6\n' "${ANALYSIS_MAX_RETRIES:-unset}"
fi

if command -v harbor >/dev/null 2>&1; then
  HARBOR_PY="$(harbor_python)"
  printf '[PASS] Harbor Python %s\n' "$HARBOR_PY"
  if "$HARBOR_PY" - <<'PY'
from custom_agents.bootstrap_compatible_codex import BootstrapCompatibleCodex
from custom_agents.bootstrap_compatible_llama import BootstrapCompatibleLlamaTextBasedMiniSweAgent
print('[PASS] bootstrap-compatible Codex/Llama imports under Harbor Python')
PY
  then :; else
    printf '[FAIL] bootstrap-compatible Codex/Llama import under Harbor Python\n'
    fail=1
  fi
fi

for f in "$CODEX_CONFIG_FILE" "$LLAMA_CONFIG_FILE"; do
  if [[ -f "$f" ]]; then printf '[PASS] config %s\n' "$f"; else printf '[FAIL] missing config %s\n' "$f"; fail=1; fi
done

if [[ -n "${CODEX_VERSION:-}" ]]; then
  printf '[PASS] CODEX_VERSION pinned: %s\n' "$CODEX_VERSION"
else
  printf '[WARN] CODEX_VERSION is not pinned; record/pin the version from the smoke run before a production benchmark.\n'
fi
printf '[PASS] MINI_SWE_VERSION pinned: %s\n' "$MINI_SWE_VERSION"

syntax_fail=0
for f in lab.sh scripts/*.sh; do
  bash -n "$f" || syntax_fail=1
done
if [[ "$syntax_fail" == 0 ]]; then printf '[PASS] shell syntax\n'; else printf '[FAIL] shell syntax\n'; fail=1; fi

if python3 -m compileall -q scripts custom_agents; then
  printf '[PASS] Python compilation\n'
else
  printf '[FAIL] Python compilation\n'
  fail=1
fi

if [[ -n "${MINI_SWE_LITELLM_VERSION:-}" ]]; then
  printf '[PASS] MINI_SWE_LITELLM_VERSION pinned: %s\n' "$MINI_SWE_LITELLM_VERSION"
else
  printf '[FAIL] MINI_SWE_LITELLM_VERSION is not pinned\n'
  fail=1
fi

if [[ "$fail" != 0 ]]; then
  echo 'Runtime doctor: FAIL' >&2
  exit 1
fi

echo 'Runtime doctor: PASS'
