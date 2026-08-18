#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$PROJECT_ROOT/.env"

die() { echo "ERROR: $*" >&2; exit 1; }
require_command() { command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"; }
require_file() { [[ -f "$1" ]] || die "missing $2: $1"; }
require_dir() { [[ -d "$1" ]] || die "missing $2: $1"; }

harbor_python() {
  require_command harbor
  local harbor_bin shebang py
  harbor_bin="$(command -v harbor)"
  IFS= read -r shebang < "$harbor_bin" || die "cannot read Harbor launcher: $harbor_bin"
  [[ "$shebang" == '#!'* ]] || die "Harbor launcher has no Python shebang: $harbor_bin"
  py="${shebang#\#!}"
  [[ -x "$py" ]] || die "Harbor Python is not executable: $py"
  printf '%s\n' "$py"
}

load_env() {
  [[ -f "$ENV_FILE" ]] || cp "$PROJECT_ROOT/.env.example" "$ENV_FILE"
  local names=(PROFILES HARBOR_REPEATS HARBOR_CONCURRENCY MATRIX_CONCURRENCY_PRESET CLAUDE_CONCURRENCY FABLE_CONCURRENCY CODEX_CONCURRENCY LLAMA_CONCURRENCY HARBOR_DISABLE_VERIFICATION MODAL_VM_RUNTIME CUE_ASSIGNMENT_SEED FINANCIAL_MESSAGE_INDEX SELF_PRESERVATION_MESSAGE_INDEX RESOURCE_DEPRIVATION_MESSAGE_INDEX RESOURCE_TASK_COUNT ALLOW_INTERNET AUTO_ANALYZE ANALYSIS_USE_LLM ANALYSIS_MODEL ANALYSIS_MAX_CHARS ANALYSIS_MAX_RETRIES PILOT_TASK_COUNT SAMPLE_TASK_COUNT CLAUDE_AGENT CLAUDE_MODEL CLAUDE_INSTRUCTION_FILE FABLE_AGENT FABLE_MODEL FABLE_INSTRUCTION_FILE CODEX_AGENT CODEX_MODEL CODEX_INSTRUCTION_FILE CODEX_VERSION CODEX_CONFIG_FILE LLAMA_AGENT LLAMA_MODEL LLAMA_INSTRUCTION_FILE MINI_SWE_VERSION LLAMA_CONFIG_FILE)
  local n
  declare -a was_set=() values=()
  for n in "${names[@]}"; do
    if [[ -n "${!n+x}" ]]; then was_set+=(1); values+=("${!n}"); else was_set+=(0); values+=(""); fi
  done
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
  local i=0
  for n in "${names[@]}"; do
    if [[ "${was_set[$i]}" == 1 ]]; then printf -v "$n" '%s' "${values[$i]}"; export "$n"; fi
    i=$((i+1))
  done

  export OPENAI_API_KEY="${LITE_LLM_KEY:-}"
  export OPENAI_API_BASE="${LITE_LLM_URL:-}"
  export OPENAI_BASE_URL="${LITE_LLM_URL:-}"
  export ANTHROPIC_API_KEY="${LITE_LLM_KEY:-}"
  export ANTHROPIC_AUTH_TOKEN="${LITE_LLM_KEY:-}"
  export ANTHROPIC_BASE_URL="${LITE_LLM_URL_ANTHROPIC:-${LITE_LLM_URL:-}}"
  export GEMINI_API_KEY="${LITE_LLM_KEY:-}"
  export MSWEA_API_KEY="${LITE_LLM_KEY:-}"
  export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

  export CLAUDE_AGENT="${CLAUDE_AGENT:-claude-code}"
  export CLAUDE_MODEL="${CLAUDE_MODEL:-anthropic/claude-opus-4-8}"
  export CLAUDE_INSTRUCTION_FILE="${CLAUDE_INSTRUCTION_FILE:-CLAUDE.md}"
  export FABLE_AGENT="${FABLE_AGENT:-claude-code}"
  export FABLE_MODEL="${FABLE_MODEL:-anthropic/claude-fable-5}"
  export FABLE_INSTRUCTION_FILE="${FABLE_INSTRUCTION_FILE:-CLAUDE.md}"
  export CODEX_AGENT="${CODEX_AGENT:-codex}"
  export CODEX_MODEL="${CODEX_MODEL:-openai/gpt-5.6}"
  export CODEX_INSTRUCTION_FILE="${CODEX_INSTRUCTION_FILE:-AGENTS.md}"
  export LLAMA_AGENT="${LLAMA_AGENT:-custom_agents.llama_textbased_mini_swe:LlamaTextBasedMiniSweAgent}"
  export LLAMA_MODEL="${LLAMA_MODEL:-openai/llmengine/llama-3-3-70b-instruct}"
  export MINI_SWE_VERSION="${MINI_SWE_VERSION:-2.4.5}"
  export LLAMA_INSTRUCTION_FILE="${LLAMA_INSTRUCTION_FILE:-AGENTS.md}"
  export CODEX_CONFIG_FILE="${CODEX_CONFIG_FILE:-$PROJECT_ROOT/config/codex_scale_http.toml}"
  export LLAMA_CONFIG_FILE="${LLAMA_CONFIG_FILE:-$PROJECT_ROOT/config/mswea_rf_llama.yaml}"

  export CUE_ASSIGNMENT_SEED="${CUE_ASSIGNMENT_SEED:-20260805}"
  export FINANCIAL_MESSAGE_INDEX="${FINANCIAL_MESSAGE_INDEX:-3}"
  export SELF_PRESERVATION_MESSAGE_INDEX="${SELF_PRESERVATION_MESSAGE_INDEX:-5}"
  export RESOURCE_DEPRIVATION_MESSAGE_INDEX="${RESOURCE_DEPRIVATION_MESSAGE_INDEX:-10}"
  export RESOURCE_TASK_COUNT="${RESOURCE_TASK_COUNT:-70}"
  export PROFILES="${PROFILES:-claude fable codex llama}"
  export PILOT_TASK_COUNT="${PILOT_TASK_COUNT:-4}"
  export SAMPLE_TASK_COUNT="${SAMPLE_TASK_COUNT:-10}"
  export HARBOR_REPEATS="${HARBOR_REPEATS:-1}"
  export ALLOW_INTERNET="${ALLOW_INTERNET:-true}"
  export HARBOR_CONCURRENCY="${HARBOR_CONCURRENCY:-1}"
  export MATRIX_CONCURRENCY_PRESET="${MATRIX_CONCURRENCY_PRESET:-custom}"
  export CLAUDE_CONCURRENCY="${CLAUDE_CONCURRENCY:-1}"
  export FABLE_CONCURRENCY="${FABLE_CONCURRENCY:-1}"
  export CODEX_CONCURRENCY="${CODEX_CONCURRENCY:-1}"
  export LLAMA_CONCURRENCY="${LLAMA_CONCURRENCY:-1}"
  export MODAL_VM_RUNTIME="${MODAL_VM_RUNTIME:-true}"
  export HARBOR_DISABLE_VERIFICATION="${HARBOR_DISABLE_VERIFICATION:-0}"
  export AUTO_ANALYZE="${AUTO_ANALYZE:-0}"
  export ANALYSIS_USE_LLM="${ANALYSIS_USE_LLM:-1}"
  export ANALYSIS_MODEL="${ANALYSIS_MODEL:-openai/gpt-5.6}"
  export ANALYSIS_MAX_CHARS="${ANALYSIS_MAX_CHARS:-60000}"
  export ANALYSIS_MAX_RETRIES="${ANALYSIS_MAX_RETRIES:-3}"

  export TASK_ROOT="$PROJECT_ROOT/vendor/rf"
  export GENERATED_ROOT="$PROJECT_ROOT/generated"
  export RESULTS_ROOT="$PROJECT_ROOT/results"
  export MANIFEST_ROOT="$PROJECT_ROOT/manifests"
  export FACTOR_ROOT="$PROJECT_ROOT/factor_data"
  export CUE_ASSIGNMENT_FILE="${CUE_ASSIGNMENT_FILE:-$MANIFEST_ROOT/cue_assignments.json}"
}

profile_values() {
  local profile="$1"
  case "$profile" in
    claude) PROFILE_AGENT="$CLAUDE_AGENT"; PROFILE_MODEL="$CLAUDE_MODEL"; PROFILE_MODEL_FOR_HARBOR="$CLAUDE_MODEL"; PROFILE_INSTRUCTION_FILE="$CLAUDE_INSTRUCTION_FILE"; PROFILE_AUTH_BACKEND=anthropic; PROFILE_CONFIG_FILE="" ;;
    fable) PROFILE_AGENT="$FABLE_AGENT"; PROFILE_MODEL="$FABLE_MODEL"; PROFILE_MODEL_FOR_HARBOR="$FABLE_MODEL"; PROFILE_INSTRUCTION_FILE="$FABLE_INSTRUCTION_FILE"; PROFILE_AUTH_BACKEND=anthropic; PROFILE_CONFIG_FILE="" ;;
    codex) PROFILE_AGENT="$CODEX_AGENT"; PROFILE_MODEL="$CODEX_MODEL"; PROFILE_MODEL_FOR_HARBOR="$CODEX_MODEL"; PROFILE_INSTRUCTION_FILE="$CODEX_INSTRUCTION_FILE"; PROFILE_AUTH_BACKEND=openai; PROFILE_CONFIG_FILE="$CODEX_CONFIG_FILE" ;;
    llama) PROFILE_AGENT="$LLAMA_AGENT"; PROFILE_MODEL="$LLAMA_MODEL"; PROFILE_MODEL_FOR_HARBOR="$LLAMA_MODEL"; PROFILE_INSTRUCTION_FILE="$LLAMA_INSTRUCTION_FILE"; PROFILE_AUTH_BACKEND=openai; PROFILE_CONFIG_FILE="$LLAMA_CONFIG_FILE" ;;
    *) die "unknown profile '$profile'; supported: claude, fable, codex, llama" ;;
  esac
  export PROFILE_AGENT PROFILE_MODEL PROFILE_MODEL_FOR_HARBOR PROFILE_INSTRUCTION_FILE PROFILE_AUTH_BACKEND PROFILE_CONFIG_FILE
}
