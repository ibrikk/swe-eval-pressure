#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-pilot}"; PROFILE="${2:-}"; shift 2 || true
[[ -n "$PROFILE" ]] || { echo "Usage: $0 <pilot|sample|full|resource> <claude|fable|codex|llama> [--shard 1/3 | --shard-size 30 --shard-index 1]" >&2; exit 2; }

SHARD_SPEC=""
SHARD_SIZE=""
SHARD_INDEX=""
INSTALL_ONLY=0
DATASET_OVERRIDE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --shard) SHARD_SPEC="${2:-}"; shift 2 ;;
    --shard-size) SHARD_SIZE="${2:-}"; shift 2 ;;
    --shard-index) SHARD_INDEX="${2:-}"; shift 2 ;;
    --install-only) INSTALL_ONLY=1; shift ;;
    --dataset-override) DATASET_OVERRIDE="${2:-}"; shift 2 ;;
    -h|--help)
      cat <<'EOF'
Usage:
  ./lab.sh run <mode> <profile>
  ./lab.sh run <mode> <profile> --shard 1/3
  ./lab.sh run <mode> <profile> --shard-size 30 --shard-index 1
  ./lab.sh run <mode> <profile> --install-only

Advanced/internal:
  ./lab.sh run <mode> <profile> --dataset-override /path/to/generated/shard

`--install-only` builds the task environment and installs the agent without running the model.
It automatically selects one representative (prefer clean) per base task, so cue variants are not redundantly installed.
Use it as a task-image compatibility smoke before expensive runs.

Sharding is base-task aware: all variants of a SWE-Atlas base task stay in the same shard.
For full mode, --shard-size 30 creates chunks of 300, 300, and 100 trajectories.
For resource mode, the same chunks contain 90, 90, and 30 trajectories.
EOF
      exit 0 ;;
    *) echo "Unknown run option: $1" >&2; exit 2 ;;
  esac
done
if [[ -n "$SHARD_SPEC" && ( -n "$SHARD_SIZE" || -n "$SHARD_INDEX" ) ]]; then
  echo "Use either --shard or --shard-size/--shard-index, not both." >&2; exit 2
fi
if [[ -n "$SHARD_SIZE" && -z "$SHARD_INDEX" ]] || [[ -z "$SHARD_SIZE" && -n "$SHARD_INDEX" ]]; then
  echo "--shard-size and --shard-index must be used together." >&2; exit 2
fi
if [[ -n "$DATASET_OVERRIDE" && ( -n "$SHARD_SPEC" || -n "$SHARD_SIZE" || -n "$SHARD_INDEX" ) ]]; then
  echo "--dataset-override cannot be combined with shard options." >&2; exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/00_common.sh"
load_env; profile_values "$PROFILE"; cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
require_command harbor; require_command modal
[[ -n "${LITE_LLM_KEY:-}" ]] || die "set LITE_LLM_KEY or LITE_LLM_KEYS in .env"

DATASET="$GENERATED_ROOT/$MODE/$PROFILE"
shard_label=""
if [[ -n "$DATASET_OVERRIDE" ]]; then
  DATASET="$DATASET_OVERRIDE"
  shard_label="-key-shard"
fi
require_dir "$DATASET" "dataset; run ./lab.sh prepare $MODE"
if [[ -z "$DATASET_OVERRIDE" && -n "$SHARD_SPEC" ]]; then
  DATASET="$(python3 "$SCRIPT_DIR/05_shard_dataset.py" --project-root "$PROJECT_ROOT" --mode "$MODE" --profile "$PROFILE" --shard "$SHARD_SPEC")"
  shard_label="-shard-${SHARD_SPEC//\//-of-}"
elif [[ -z "$DATASET_OVERRIDE" && -n "$SHARD_SIZE" ]]; then
  DATASET="$(python3 "$SCRIPT_DIR/05_shard_dataset.py" --project-root "$PROJECT_ROOT" --mode "$MODE" --profile "$PROFILE" --shard-size "$SHARD_SIZE" --shard-index "$SHARD_INDEX")"
  shard_label="-chunk-${SHARD_INDEX}-size-${SHARD_SIZE}"
fi

# Install-only is a task-image/agent compatibility smoke. Cue variants of the
# same base task share that compatibility surface, so execute one representative
# (prefer clean) per base task to avoid redundant Modal builds and setup work.
INSTALL_ONLY_DATASET=""
if [[ "$INSTALL_ONLY" == 1 ]]; then
  mkdir -p "$GENERATED_ROOT/.install-only"
  INSTALL_ONLY_DATASET="$(mktemp -d "$GENERATED_ROOT/.install-only/${MODE}-${PROFILE}.XXXXXX")"
  trap '[[ -n "${INSTALL_ONLY_DATASET:-}" ]] && rm -rf "$INSTALL_ONLY_DATASET"' EXIT
  DATASET="$(python3 "$SCRIPT_DIR/12_install_only_dataset.py" \
    --project-root "$PROJECT_ROOT" \
    --source "$DATASET" \
    --output "$INSTALL_ONLY_DATASET")"
fi

timestamp="$(date +%Y%m%d-%H%M%S)"
install_label=""
[[ "$INSTALL_ONLY" == 1 ]] && install_label="-install-only"
key_label=""
if [[ "${SWE_LITELLM_KEY_COUNT:-1}" -gt 1 ]]; then
  key_label="-key-${SWE_LITELLM_KEY_INDEX:-1}-of-${SWE_LITELLM_KEY_COUNT}"
fi
job_name="swe-eval-pressure-${MODE}-${PROFILE}${shard_label}${key_label}${install_label}-${timestamp}"
output="$RESULTS_ROOT/$MODE/$job_name"; mkdir -p "$output"
if [[ -n "${SWE_RUN_OUTPUT_FILE:-}" ]]; then printf '%s\n' "$output" > "$SWE_RUN_OUTPUT_FILE"; fi
cp "$DATASET/manifest.json" "$output/dataset_manifest.json"
# Keep two manifests for reproducibility. dataset_manifest.json is the exact
# dataset executed by this Harbor job (possibly one shard). study_manifest.json
# is the complete prepared experiment, so live analysis has the same denominator
# whether execution is sharded or unsharded.
study_manifest="$DATASET/manifest.json"
source_dataset="$(python3 - "$DATASET/manifest.json" <<'PYDATA'
import json, sys
from pathlib import Path
data=json.loads(Path(sys.argv[1]).read_text())
print(data.get("source_dataset", ""))
PYDATA
)"
if [[ -n "$source_dataset" && -f "$PROJECT_ROOT/$source_dataset/manifest.json" ]]; then
  study_manifest="$PROJECT_ROOT/$source_dataset/manifest.json"
fi
cp "$study_manifest" "$output/study_manifest.json"
profile_version_requested=""
case "$PROFILE" in
  codex) profile_version_requested="${CODEX_VERSION:-}" ;;
  llama) profile_version_requested="${MINI_SWE_VERSION:-}" ;;
esac
python3 - "$output/run_metadata.json" "$MODE" "$PROFILE" "$PROFILE_AGENT" "$PROFILE_MODEL_FOR_HARBOR" "$DATASET" "$ALLOW_INTERNET" "$HARBOR_REPEATS" "$HARBOR_CONCURRENCY" "$SHARD_SPEC" "$SHARD_SIZE" "$SHARD_INDEX" "$profile_version_requested" "$PROFILE_CONFIG_FILE" "$HARBOR_DISABLE_VERIFICATION" "$study_manifest" "$INSTALL_ONLY" "${SWE_LITELLM_KEY_INDEX:-1}" "${SWE_LITELLM_KEY_COUNT:-1}" "$LITE_LLM_TPM_LIMIT" <<'PYMETA'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
(out, mode, profile, agent, model, dataset, internet, repeats, concurrency,
 shard_spec, shard_size, shard_index, agent_version_requested, config_file, disable_verification,
 study_manifest_path, install_only, litellm_key_index, litellm_key_count, litellm_tpm_limit) = sys.argv[1:]
dataset_path = Path(dataset)
manifest_path = dataset_path / "manifest.json"
manifest_bytes = manifest_path.read_bytes()
manifest = json.loads(manifest_bytes)
study_manifest_bytes = Path(study_manifest_path).read_bytes()
study_manifest = json.loads(study_manifest_bytes)
study_identity = {
    "mode": study_manifest.get("mode", manifest.get("mode", mode)),
    "profile": study_manifest.get("profile", manifest.get("profile", profile)),
    "agent": agent,
    "model": model,
    "allow_internet": study_manifest.get("allow_internet", manifest.get("allow_internet", internet.lower() == "true")),
    "cue_assignment_seed": study_manifest.get("cue_assignment_seed", manifest.get("cue_assignment_seed")),
    "cue_assignment_registry_fingerprint": study_manifest.get("cue_assignment_registry_fingerprint", manifest.get("cue_assignment_registry_fingerprint")),
    "cue_library_fingerprint": study_manifest.get("cue_library_fingerprint", manifest.get("cue_library_fingerprint")),
    "financial_message_index": study_manifest.get("financial_message_index", manifest.get("financial_message_index")),
    "self_preservation_message_index": study_manifest.get("self_preservation_message_index", manifest.get("self_preservation_message_index")),
    "resource_deprivation_message_index": study_manifest.get("resource_deprivation_message_index", manifest.get("resource_deprivation_message_index")),
    "delivery_channels": study_manifest.get("delivery_channels", manifest.get("delivery_channels")),
    "variants_per_task": study_manifest.get("variants_per_task", manifest.get("variants_per_task")),
    "scaffold_instruction_file": study_manifest.get("scaffold_instruction_file", manifest.get("scaffold_instruction_file")),
    "harbor_repeats": int(repeats),
    "agent_version_requested": agent_version_requested or None,
    "agent_config_sha256": (
        hashlib.sha256(Path(config_file).read_bytes()).hexdigest() if config_file and Path(config_file).is_file() else None
    ),
    "verification_enabled": disable_verification != "1",
}
study_signature = hashlib.sha256(
    json.dumps(study_identity, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()[:16]
metadata = {
    "schema_version": "2.0",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "mode": mode,
    "profile": profile,
    "agent": agent,
    "model": model,
    "dataset": str(dataset_path),
    "dataset_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    "study_manifest_sha256": hashlib.sha256(study_manifest_bytes).hexdigest(),
    "study_base_task_count": study_manifest.get("base_task_count"),
    "study_trajectory_count": len(study_manifest.get("tasks", [])),
    "study_signature": study_signature,
    "allow_internet": internet.lower() == "true",
    "harbor_repeats": int(repeats),
    "harbor_concurrency": int(concurrency),
    "litellm_key_index": int(litellm_key_index),
    "litellm_key_count": int(litellm_key_count),
    "litellm_tpm_limit": int(litellm_tpm_limit) if litellm_tpm_limit.isdigit() else None,
    "install_only": install_only == "1",
    "agent_version_requested": agent_version_requested or None,
    "agent_config_file": config_file or None,
    "agent_config_sha256": (
        hashlib.sha256(Path(config_file).read_bytes()).hexdigest() if config_file and Path(config_file).is_file() else None
    ),
    "verification_enabled": disable_verification != "1",
    "shard": {
        "balanced_spec": shard_spec or None,
        "base_task_chunk_size": int(shard_size) if shard_size else None,
        "chunk_index": int(shard_index) if shard_index else None,
    },
}
Path(out).write_text(json.dumps(metadata, indent=2) + "\n")
PYMETA
args=(harbor run -p "$DATASET" -a "$PROFILE_AGENT" -m "$PROFILE_MODEL_FOR_HARBOR" -e modal --ek "modal_vm_runtime=$MODAL_VM_RUNTIME" --force-build -k "$HARBOR_REPEATS" -n "$HARBOR_CONCURRENCY" -o "$output" --job-name "$job_name" -y)
[[ "$INSTALL_ONLY" == 1 ]] && args+=(--install-only)
case "$PROFILE_AUTH_BACKEND" in
 anthropic) args+=(--ae 'ANTHROPIC_AUTH_TOKEN=${ANTHROPIC_AUTH_TOKEN}' --ae 'ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}' --ae "ANTHROPIC_BASE_URL=$ANTHROPIC_BASE_URL" --ae "ANTHROPIC_MODEL=$PROFILE_MODEL_FOR_HARBOR" --ae "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1") ;;
 openai) args+=(--ae 'OPENAI_API_KEY=${OPENAI_API_KEY}' --ae "OPENAI_API_BASE=$OPENAI_API_BASE" --ae "OPENAI_BASE_URL=$OPENAI_BASE_URL" --ae 'MSWEA_API_KEY=${MSWEA_API_KEY}') ;;
 *) die "unsupported auth backend" ;;
esac
if [[ -n "$PROFILE_CONFIG_FILE" ]]; then require_file "$PROFILE_CONFIG_FILE" "$PROFILE config"; args+=(--ak "config_file=$PROFILE_CONFIG_FILE"); fi
if [[ "$PROFILE" == "codex" && -n "${CODEX_VERSION:-}" ]]; then args+=(--ak "version=$CODEX_VERSION"); fi
if [[ "$PROFILE" == "llama" && -n "${MINI_SWE_VERSION:-}" ]]; then args+=(--ak "version=$MINI_SWE_VERSION"); fi
if [[ "$HARBOR_DISABLE_VERIFICATION" == 1 ]]; then args+=(--disable-verification); fi
count=0; for d in "$DATASET"/*; do [[ -d "$d" && -f "$d/task.toml" ]] && count=$((count+1)); done
[[ "$count" -gt 0 ]] || die "no tasks in $DATASET"
echo "Mode=$MODE Profile=$PROFILE Model=$PROFILE_MODEL_FOR_HARBOR Tasks=$count Repeats=$HARBOR_REPEATS Total=$((count*HARBOR_REPEATS)) Internet=$ALLOW_INTERNET InstallOnly=$INSTALL_ONLY Key=${SWE_LITELLM_KEY_INDEX:-1}/${SWE_LITELLM_KEY_COUNT:-1} TPMKey=$LITE_LLM_TPM_LIMIT"
printf 'Command:'
for arg in "${args[@]}"; do
  shown="$arg"
  case "$shown" in
    OPENAI_API_KEY=*|ANTHROPIC_API_KEY=*|ANTHROPIC_AUTH_TOKEN=*|MSWEA_API_KEY=*|GEMINI_API_KEY=*)
      shown="${shown%%=*}=<redacted>" ;;
  esac
  printf ' %q' "$shown"
done
printf '\n'
"${args[@]}"
if [[ "$AUTO_ANALYZE" == 1 && "$INSTALL_ONLY" == 0 ]]; then
  bash "$SCRIPT_DIR/07_analyze.sh" "$MODE" "$PROFILE" || echo "WARNING: analysis failed"
fi
