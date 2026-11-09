#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 ]]; then
  echo 'usage: run_cpu.sh NEW_EMPTY_OUTPUT_DIRECTORY' >&2
  exit 2
fi
WM_BUSINESS_OUT="$1"
if [[ -e "$WM_BUSINESS_OUT" ]] && [[ -n "$(ls -A "$WM_BUSINESS_OUT")" ]]; then
  echo "refusing to overwrite $WM_BUSINESS_OUT" >&2
  exit 2
fi
mkdir -p "$WM_BUSINESS_OUT"
cp business_transport/scenarios/benchmark.json "$WM_BUSINESS_OUT/benchmark.json"
for profile in development confirmation; do
  python -m experiments.business_constraints.run --profile "$profile" --output "$WM_BUSINESS_OUT/$profile"
  python -m experiments.business_constraints.run --profile "$profile" --output "$WM_BUSINESS_OUT/repeat-$profile"
  python tools/business_constraints/compare.py "$WM_BUSINESS_OUT/$profile/raw.json" "$WM_BUSINESS_OUT/repeat-$profile/raw.json"
done
python -m experiments.business_constraints.full_state_audit --reference "$WM_BUSINESS_OUT/confirmation/raw.json" --output "$WM_BUSINESS_OUT/full-state-confirmation.json"
python -m experiments.business_constraints.full_state_audit --reference "$WM_BUSINESS_OUT/confirmation/raw.json" --output "$WM_BUSINESS_OUT/repeat-full-state-confirmation.json"
cmp "$WM_BUSINESS_OUT/full-state-confirmation.json" "$WM_BUSINESS_OUT/repeat-full-state-confirmation.json"
echo 'full-state exploratory diagnostic reproduced exactly'
python -m experiments.business_constraints.verify "$WM_BUSINESS_OUT"
