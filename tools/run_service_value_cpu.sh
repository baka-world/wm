#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo 'Usage: run_service_value_cpu.sh EMPTY_OUTPUT_DIRECTORY' >&2
  exit 2
fi
WM_VALUE_OUTPUT="$1"
if [[ -e "$WM_VALUE_OUTPUT" ]] && [[ -n "$(ls -A "$WM_VALUE_OUTPUT")" ]]; then
  echo "Refusing to overwrite $WM_VALUE_OUTPUT" >&2
  exit 2
fi
mkdir -p "$WM_VALUE_OUTPUT"
python -c "import torch; print('CPU only; PyTorch', torch.__version__)"
python -m experiments.service_value.config --output "$WM_VALUE_OUTPUT/scenario-config.json"
for profile in development confirmation; do
  python -m experiments.service_value.counterfactual --profile "$profile" --output "$WM_VALUE_OUTPUT/counterfactual-$profile"
  python -m experiments.service_value.oracle_audit --input "$WM_VALUE_OUTPUT/counterfactual-$profile/raw.json" --output "$WM_VALUE_OUTPUT/counterfactual-$profile/oracle-audit.json"
  python -m experiments.service_value.report --input "$WM_VALUE_OUTPUT/counterfactual-$profile/raw.json" --output "$WM_VALUE_OUTPUT/counterfactual-$profile/analysis.json"
  python -m experiments.service_value.aligned --input "$WM_VALUE_OUTPUT/counterfactual-$profile/raw.json" --output "$WM_VALUE_OUTPUT/aligned-$profile"
  python -m experiments.service_value.report --input "$WM_VALUE_OUTPUT/aligned-$profile/raw.json" --output "$WM_VALUE_OUTPUT/aligned-$profile/analysis.json"
  python -m experiments.service_value.online --profile "$profile" --output "$WM_VALUE_OUTPUT/online-$profile"
  python -m experiments.service_value.online_report --input "$WM_VALUE_OUTPUT/online-$profile/raw.json" --output "$WM_VALUE_OUTPUT/online-$profile/paired.json"
done
python -m experiments.service_value.check "$WM_VALUE_OUTPUT"
