#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/00_common.sh"
load_env

[[ -n "${LITE_LLM_KEY:-}" ]] || die "LITE_LLM_KEY is empty; set it in .env"
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
printf 'Key: set (%s chars)\n\n' "${#LITE_LLM_KEY}"

probe() {
  local url="$1"
  local tmp code
  tmp="$(mktemp)"
  code="$(curl -sS -o "$tmp" -w '%{http_code}' "$url" -H "Authorization: Bearer $LITE_LLM_KEY" || true)"
  printf '=== GET %s ===\nHTTP %s\n' "$url" "$code"
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

if probe "$primary"; then
  exit 0
fi

printf '\nThe standard /v1/models shape was not returned. Trying the non-v1 fallback for diagnostics only.\n\n'
probe "$fallback" || true

cat <<'EOF'

No OpenAI-style {"data": [...]} model list was returned.
That is why `.data[].id` produced `Cannot iterate over null`.
The HTTP status/body above is the useful part: it usually indicates a key permission,
custom proxy route, or endpoint-policy issue rather than a jq problem.

You can still probe known model routes directly with:
  ./lab.sh llama-doctor
EOF
