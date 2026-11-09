#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 3 ]]; then
  echo 'usage: run_acceptance.sh PIBT_LIBRARY PINNED_SOURCE_DIRECTORY NEW_OUTPUT_DIRECTORY' >&2
  exit 2
fi
WM_PIBT_LIBRARY="$(realpath "$1")"
WM_PIBT_SOURCES="$(realpath "$2")"
WM_PIBT_OUT="$3"
if [[ -e "$WM_PIBT_OUT" ]] && [[ -n "$(ls -A "$WM_PIBT_OUT")" ]]; then
  echo "refusing to overwrite $WM_PIBT_OUT" >&2
  exit 2
fi
mkdir -p "$WM_PIBT_OUT"
for suffix in '' '-repeat'; do
  python -m experiments.pibt_motion.hospital_reproduction --source-dir "$WM_PIBT_SOURCES" --backend legacy --output "$WM_PIBT_OUT/hospital-legacy$suffix.json"
  python -m experiments.pibt_motion.hospital_reproduction --source-dir "$WM_PIBT_SOURCES" --backend pibt --library "$WM_PIBT_LIBRARY" --output "$WM_PIBT_OUT/hospital-pibt$suffix.json"
  python -m experiments.pibt_motion.benchmark --library "$WM_PIBT_LIBRARY" --output "$WM_PIBT_OUT/grid-benchmark$suffix.json"
  python -m experiments.pibt_motion.model_alignment --library "$WM_PIBT_LIBRARY" --output "$WM_PIBT_OUT/model-alignment$suffix.json"
done
for stem in hospital-legacy hospital-pibt grid-benchmark model-alignment; do
  python tools/pibt_motion/compare.py "$WM_PIBT_OUT/$stem.json" "$WM_PIBT_OUT/$stem-repeat.json"
done
python -m experiments.pibt_motion.joint_diagnostic --input "$WM_PIBT_OUT/hospital-legacy.json" --output "$WM_PIBT_OUT/joint-diagnostic.json"
python -m experiments.pibt_motion.topology_audit --hospital "$WM_PIBT_OUT/hospital-legacy.json" --output "$WM_PIBT_OUT/topology-audit.json"
python -m experiments.pibt_motion.verify "$WM_PIBT_OUT"
