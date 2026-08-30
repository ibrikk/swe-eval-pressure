#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT_ONLY=0
SKIP_REEXTRACT=0
SKIP_PREFLIGHT=0

usage() {
  cat <<'EOF'
Usage:
  ./lab.sh analyze current
  ./lab.sh analyze current --report-only
  ./lab.sh analyze current --skip-reextract
  ./lab.sh current
  ./lab.sh current --report-only

Build the canonical cross-study current analysis and the ICLR team pre-read.

Default full pipeline:
  1. current preflight
  2. verifier forensics
  3. deterministic behavior re-extraction
  4. fast matched statistical analysis
  5. findings digest
  6. integrity decomposition
  7. validated integrated current report
  8. scholarly ICLR team pre-read
  9. colorful visual enhancement suite
  10. benchmark provenance / answer-recovery audit
  11. presentation-first team briefing HTML

--report-only
  Rebuild presentation artifacts from existing analysis/current outputs.
  No statistical analysis is rerun.

--skip-reextract
  Skip deterministic behavior re-extraction. Use only when the canonical
  source lock already contains the current re-extracted behavior fields.

--skip-preflight
  Skip scripts/26_current_preflight.py. Intended for debugging only.

Data note:
  analysis/ and results/ are gitignored. A fresh clone still needs the
  complete trajectory/result corpus (and semantic judge artifacts, when
  semantic sections are required) restored locally before this command
  can reproduce the full paper-grade report.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --report-only) REPORT_ONLY=1; shift ;;
    --skip-reextract) SKIP_REEXTRACT=1; shift ;;
    --skip-preflight) SKIP_PREFLIGHT=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown current-analysis option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

run_py() {
  local script="$1"
  echo
  echo "================================================================================"
  echo "RUNNING: $script"
  echo "================================================================================"
  uv run --group inference python "$ROOT/scripts/$script"
}

require_file() {
  local path="$1"
  [[ -f "$path" ]] || {
    echo "Missing required file: $path" >&2
    exit 2
  }
}

for script in \
  26_current_preflight.py \
  28_verifier_forensics.py \
  30_reextract_primary_behavior.py \
  32_current_analysis_fast.py \
  33_findings_digest.py \
  34_integrity_decomposition.py \
  35_current_html.py \
  36_iclr_team_preread_v2.py \
  37_visual_report.py \
  39_benchmark_provenance_audit.py \
  40_benchmark_provenance_report.py \
  38_team_briefing_html.py
do
  require_file "$ROOT/scripts/$script"
done

if [[ "$REPORT_ONLY" == 0 ]]; then
  if [[ "$SKIP_PREFLIGHT" == 0 ]]; then
    run_py 26_current_preflight.py
  fi

  run_py 28_verifier_forensics.py

  if [[ "$SKIP_REEXTRACT" == 0 ]]; then
    run_py 30_reextract_primary_behavior.py
  fi

  run_py 32_current_analysis_fast.py
  run_py 33_findings_digest.py
  run_py 34_integrity_decomposition.py
fi

# The report stack is deliberately presentation-only downstream of the
# canonical current statistical outputs.
run_py 35_current_html.py
run_py 36_iclr_team_preread_v2.py
run_py 37_visual_report.py
run_py 39_benchmark_provenance_audit.py
run_py 40_benchmark_provenance_report.py
run_py 38_team_briefing_html.py

REPORT="$ROOT/reports/iclr-current/index.html"
BRIEFING="$ROOT/reports/iclr-current/team-briefing.html"
PROVENANCE_AUDIT="$ROOT/analysis/current/findings/benchmark_provenance_summary.csv"
MANIFEST="$ROOT/reports/iclr-current/visual_manifest.json"

require_file "$REPORT"
require_file "$BRIEFING"
require_file "$PROVENANCE_AUDIT"
require_file "$MANIFEST"

echo
echo "================================================================================"
echo "CURRENT ANALYSIS + ICLR TEAM PRE-READ: PASS"
echo "================================================================================"
echo "Detailed report:  $REPORT"
echo "Team briefing:    $BRIEFING"
echo "Provenance audit: $PROVENANCE_AUDIT"
echo "Manifest:         $MANIFEST"
echo
echo "macOS:"
echo "  open \"$BRIEFING\""
echo
echo "Linux:"
echo "  xdg-open \"$BRIEFING\""
