#!/usr/bin/env bash
set -euo pipefail

# SWE-EvalPressure report command
if [[ "${1:-}" == "report" ]]; then
  shift
  exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/08_report.sh" "$@"
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
usage() {
cat <<'HELP'
SWE-EvalPressure

Usage:
  ./lab.sh inventory
  ./lab.sh assign-cues
  ./lab.sh plan [pilot|sample|full|resource]
  ./lab.sh prepare [pilot|sample|full|resource] ["profile list"]
  ./lab.sh validate [pilot|sample|full|resource] [profile|all]
  ./lab.sh run [pilot|sample|full|resource] [claude|fable|codex|llama] [shard options] [--install-only]
  ./lab.sh matrix [pilot|sample|full|resource] [--concurrency-preset serial|scale-200k|custom] [--dry-run] [shard options]
  ./lab.sh analyze [pilot|sample|full|resource] [profile|all] [--live] [--no-semantic]
  ./lab.sh results [pilot|sample|full|resource] [profile|all]
  ./lab.sh models
  ./lab.sh llama-doctor
  ./lab.sh doctor
  ./lab.sh resume <profile> <harbor-job-dir> [--concurrency N] [--retry-nonzero] [--dry-run]

Sharding examples:
  ./lab.sh run full claude --shard 1/3
  ./lab.sh run full claude --shard-size 30 --shard-index 1

The second form groups 30 base tasks per shard. In primary full mode that is
300 trajectories per shard; in resource mode it is 90.
HELP
}
cmd="${1:-}"; shift || true
case "$cmd" in
  inventory) exec python3 "$ROOT/scripts/01_inventory.py" --project-root "$ROOT" "$@" ;;
  assign-cues) source "$ROOT/scripts/00_common.sh"; load_env; exec python3 "$ROOT/scripts/02_assign_cues.py" --project-root "$ROOT" --seed "$CUE_ASSIGNMENT_SEED" --output "$CUE_ASSIGNMENT_FILE" "$@" ;;
  plan) exec bash "$ROOT/scripts/02_plan.sh" "$@" ;;
  prepare) exec bash "$ROOT/scripts/03_prepare.sh" "$@" ;;
  validate) exec bash "$ROOT/scripts/04_validate.sh" "$@" ;;
  run) exec bash "$ROOT/scripts/05_run_profile.sh" "$@" ;;
  matrix) exec bash "$ROOT/scripts/06_run_matrix.sh" "$@" ;;
  analyze) exec bash "$ROOT/scripts/07_analyze.sh" "$@" ;;
  results) exec bash "$ROOT/scripts/08_results.sh" "$@" ;;
  models) exec bash "$ROOT/scripts/08_gateway_models.sh" "$@" ;;
  llama-doctor) exec bash "$ROOT/scripts/09_llama_doctor.sh" "$@" ;;
  doctor) exec bash "$ROOT/scripts/10_runtime_doctor.sh" "$@" ;;
  resume) exec bash "$ROOT/scripts/11_resume_job.sh" "$@" ;;
  help|-h|--help|"") usage ;;
  *) echo "Unknown command: $cmd" >&2; usage >&2; exit 2 ;;
esac
