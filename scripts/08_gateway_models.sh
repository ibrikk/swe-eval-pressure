#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/00_common.sh"
load_env
parse_litellm_key_pool

[[ "${#LITELLM_KEYS_PARSED[@]}" -gt 0 ]] || die "LITE_LLM_KEY/LITE_LLM_KEYS is empty; set credentials in .env"
[[ -n "${LITE_LLM_URL:-}" ]] || die "LITE_LLM_URL is empty; set it in .env"

base="${LITE_LLM_URL%/}"
if [[ "$base" == */v1 ]]; then
  primary="$base/models"
  fallback="${base%/v1}/models"
else
  primary="$base/v1/models"
  fallback="$base/models"
fi

printf 'LiteLLM URL: %s\n' "$LITE_LLM_URL"
printf 'Key pool: %s key(s); TPM/key config=%s\n\n' "${#LITELLM_KEYS_PARSED[@]}" "$LITE_LLM_TPM_LIMIT"

probe() {
  local url="$1" key="$2" label="$3"
  local tmp code
  tmp="$(mktemp)"
  code="$(curl -sS -o "$tmp" -w '%{http_code}' "$url" -H "Authorization: Bearer $key" || true)"
  printf '=== GET %s with key %s ===\nHTTP %s\n' "$url" "$label" "$code"
  if jq -e . "$tmp" >/dev/null 2>&1; then
    jq . "$tmp"
  else
    cat "$tmp"; printf '\n'
  fi
  if jq -e '.data | type == "array"' "$tmp" >/dev/null 2>&1; then
    printf '\nAccessible model IDs (%s):\n' "$(jq '.data | length' "$tmp")"
    jq -r '.data[].id' "$tmp" | sort
    rm -f "$tmp"
    return 0
  fi
  rm -f "$tmp"
  return 1
}

for i in "${!LITELLM_KEYS_PARSED[@]}"; do
  label="$((i+1))/${#LITELLM_KEYS_PARSED[@]}"
  if probe "$primary" "${LITELLM_KEYS_PARSED[$i]}" "$label"; then
    exit 0
  fi
done

printf '\nThe standard /v1/models shape was not returned by any configured key. Trying the non-v1 fallback for diagnostics only.\n\n'
for i in "${!LITELLM_KEYS_PARSED[@]}"; do
  label="$((i+1))/${#LITELLM_KEYS_PARSED[@]}"
  probe "$fallback" "${LITELLM_KEYS_PARSED[$i]}" "$label" && exit 0 || true
done

cat <<'EOF'

No OpenAI-style {"data": [...]} model list was returned.
The HTTP status/body above is the useful part: it usually indicates a key permission,
custom proxy route, or endpoint-policy issue rather than a jq problem.

You can still probe known model routes directly with:
  ./lab.sh llama-doctor
EOF
