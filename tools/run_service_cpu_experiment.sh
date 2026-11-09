#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo 'Usage: run_service_cpu_experiment.sh EMPTY_OUTPUT_DIRECTORY' >&2
  exit 2
fi
WM_SERVICE_OUTPUT="$1"
if [[ -e "$WM_SERVICE_OUTPUT" ]] && [[ -n "$(ls -A "$WM_SERVICE_OUTPUT")" ]]; then
  echo "Refusing to overwrite $WM_SERVICE_OUTPUT" >&2
  exit 2
fi
mkdir -p "$WM_SERVICE_OUTPUT"
python -c "import torch; print('CPU experiment, PyTorch', torch.__version__, 'CUDA build', torch.version.cuda)"
python -m lab_transport.service_fault_data --output "$WM_SERVICE_OUTPUT/dataset"
python -m lab_transport.service_learning --dataset "$WM_SERVICE_OUTPUT/dataset" --output "$WM_SERVICE_OUTPUT/training" --device cpu
python -m lab_transport.service_evaluation --dataset "$WM_SERVICE_OUTPUT/dataset" --training "$WM_SERVICE_OUTPUT/training" --output "$WM_SERVICE_OUTPUT/evaluation" --device cpu
python tests/check_service_experiment.py "$WM_SERVICE_OUTPUT"
