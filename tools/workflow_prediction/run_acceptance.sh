#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 3 ]]; then
  echo 'usage: run_acceptance.sh PIBT_LIBRARY LACAM_WORKER NEW_OUTPUT_DIRECTORY' >&2
  exit 2
fi
WM_PIBT_LIBRARY="$(realpath "$1")"
WM_LACAM_WORKER="$(realpath "$2")"
WM_OUT="$3"
if [[ -e "$WM_OUT" ]] && [[ -n "$(ls -A "$WM_OUT")" ]]; then
  echo "refusing to overwrite $WM_OUT" >&2
  exit 2
fi
mkdir -p "$WM_OUT"
for profile in development confirmation; do
  for suffix in '' '-repeat'; do
    timeout 900s python -m experiments.workflow_prediction.evaluate --profile "$profile" \
      --pibt-library "$WM_PIBT_LIBRARY" --lacam-worker "$WM_LACAM_WORKER" --output "$WM_OUT/$profile$suffix"
  done
  python tools/workflow_prediction/compare.py "$WM_OUT/$profile/raw.json" "$WM_OUT/$profile-repeat/raw.json"
  python -m experiments.workflow_prediction.verify "$WM_OUT/$profile" "$profile"
  python -m experiments.workflow_prediction.coverage_report "$WM_OUT/$profile/raw.json" "$WM_OUT/$profile/coverage.json"
  python -m experiments.workflow_prediction.decision_diagnostics "$WM_OUT/$profile/raw.json" "$WM_OUT/$profile/decision-diagnostics.json"
done
python -m experiments.workflow_prediction.edge_diagnostics "$WM_OUT/edge-diagnostics.json"
