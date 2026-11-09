#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 ]]; then
  echo 'usage: run_acceptance.sh NEW_EMPTY_OUTPUT_DIRECTORY' >&2
  exit 2
fi
WM_PAPER_OUT="$1"
if [[ -e "$WM_PAPER_OUT" ]] && [[ -n "$(ls -A "$WM_PAPER_OUT")" ]]; then
  echo "refusing to overwrite $WM_PAPER_OUT" >&2
  exit 2
fi
mkdir -p "$WM_PAPER_OUT"
WM_PAPER_EVIDENCE=examples/evidence/paper-reuse-integration-20261109
python -m experiments.paper_reuse.config --output "$WM_PAPER_OUT/scenarios.json"
python -m lab_transport.paper_reuse.pets_service --dataset "$WM_PAPER_EVIDENCE/existing-service-labels.json.gz" \
  --checkpoint "$WM_PAPER_OUT/retrained-cpu.pt" > "$WM_PAPER_OUT/training.log"
for profile in development confirmation; do
  python -m experiments.paper_reuse.acceptance --profile "$profile" --output "$WM_PAPER_OUT/$profile"
  python -m experiments.paper_reuse.analyze --input "$WM_PAPER_OUT/$profile/raw.json" --output "$WM_PAPER_OUT/$profile/analysis.json"
  python -m experiments.paper_reuse.acceptance --profile "$profile" --output "$WM_PAPER_OUT/repeat-$profile"
  python tools/paper_reuse/compare_replay.py "$WM_PAPER_OUT/$profile/raw.json" "$WM_PAPER_OUT/repeat-$profile/raw.json"
done
python -m experiments.paper_reuse.verify "$WM_PAPER_OUT"
