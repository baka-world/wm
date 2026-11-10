#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 4 ]]; then
  echo 'usage: run_acceptance.sh LACAM_WORKER PIBT_LIBRARY PINNED_SOURCE_DIRECTORY NEW_OUTPUT_DIRECTORY' >&2
  exit 2
fi
WM_LACAM_WORKER="$(realpath "$1")"
WM_PIBT_LIBRARY="$(realpath "$2")"
WM_SOURCES="$(realpath "$3")"
WM_OUT="$4"
if [[ -e "$WM_OUT" ]] && [[ -n "$(ls -A "$WM_OUT")" ]]; then
  echo "refusing to overwrite $WM_OUT" >&2
  exit 2
fi
mkdir -p "$WM_OUT"
for suffix in '' '-repeat'; do
  python -m experiments.lacam3_recovery.clinic_gate --worker "$WM_LACAM_WORKER" --output "$WM_OUT/clinic-gate$suffix.json"
  python -m experiments.lacam3_recovery.benchmark --source-dir "$WM_SOURCES" --pibt-library "$WM_PIBT_LIBRARY" --lacam-worker "$WM_LACAM_WORKER" --output "$WM_OUT/paired$suffix.json" --ablation-output "$WM_OUT/ablation$suffix.json"
  python -m experiments.lacam3_recovery.model_alignment --source-dir "$WM_SOURCES" --pibt-library "$WM_PIBT_LIBRARY" --lacam-worker "$WM_LACAM_WORKER" --output "$WM_OUT/model-alignment$suffix.json"
done
for stem in clinic-gate paired ablation model-alignment; do
  python tools/lacam3_recovery/compare.py "$WM_OUT/$stem.json" "$WM_OUT/$stem-repeat.json"
done
python -m experiments.lacam3_recovery.verify "$WM_OUT"
python -m experiments.lacam3_recovery.summarize "$WM_OUT/paired.json" "$WM_OUT/summary.json"
python -m experiments.lacam3_recovery.resource_profile --source-dir "$WM_SOURCES" --pibt-library "$WM_PIBT_LIBRARY" --lacam-worker "$WM_LACAM_WORKER" --paired "$WM_OUT/paired.json" --output "$WM_OUT/resource-profile.json"
