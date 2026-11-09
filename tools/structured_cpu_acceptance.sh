#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo 'Usage: structured_cpu_acceptance.sh EMPTY_OUTPUT_DIRECTORY' >&2
  exit 2
fi
WM_ACCEPTANCE_OUTPUT="$1"
export WM_ACCEPTANCE_OUTPUT
if [[ -e "$WM_ACCEPTANCE_OUTPUT" ]] && [[ -n "$(ls -A "$WM_ACCEPTANCE_OUTPUT")" ]]; then
  echo "Refusing to overwrite $WM_ACCEPTANCE_OUTPUT" >&2
  exit 2
fi
mkdir -p "$WM_ACCEPTANCE_OUTPUT"

python -c "import torch; assert torch.version.cuda is None; print('CPU PyTorch', torch.__version__)"
python -m lab_transport.training_data export --download-sources --profile smoke --output "$WM_ACCEPTANCE_OUTPUT/dataset"
python -m lab_transport.torch_training --dataset "$WM_ACCEPTANCE_OUTPUT/dataset" \
  --output "$WM_ACCEPTANCE_OUTPUT/train" --adapter lab_transport.torch_structured_adapter \
  --device cpu --epochs 1 --batch-size 32 --train-limit 400 --validation-limit 200
python -m lab_transport.torch_training --dataset "$WM_ACCEPTANCE_OUTPUT/dataset" \
  --output "$WM_ACCEPTANCE_OUTPUT/train" --adapter lab_transport.torch_structured_adapter \
  --device cpu --epochs 2 --batch-size 32 --train-limit 400 --validation-limit 200 \
  --resume "$WM_ACCEPTANCE_OUTPUT/train/epoch-0001.pt"
python -m lab_transport.torch_evaluate --dataset "$WM_ACCEPTANCE_OUTPUT/dataset" \
  --checkpoint "$WM_ACCEPTANCE_OUTPUT/train/epoch-0002.pt" --output "$WM_ACCEPTANCE_OUTPUT/evaluate" \
  --device cpu --limit-per-split 200
python -m lab_transport.torch_rollout --checkpoint "$WM_ACCEPTANCE_OUTPUT/train/epoch-0002.pt" \
  --output "$WM_ACCEPTANCE_OUTPUT/rollout-grid" --device cpu --horizon 3 --budget 4

python - <<'PY'
import json
import os
from pathlib import Path
import torch
from lab_transport.clinic_map import make_clinic_scenario
from lab_transport.hospital_case import make_scenario
from lab_transport.multisite_experiment import make_case, prepare_tartu
from lab_transport.torch_rollout import run
from lab_transport.torch_structured_adapter import build_model
from lab_transport.torch_training import load_checkpoint
from lab_transport.training_data import download_sources

root = Path(os.environ['WM_ACCEPTANCE_OUTPUT'])
checkpoint = root / 'train/epoch-0002.pt'
torch.manual_seed(123)
initial = build_model().state_dict()
trained = load_checkpoint(checkpoint)['model_state']
assert any(not torch.equal(initial[k], trained[k]) for k in initial), 'model parameters did not update'
sources = download_sources()
bases = {'tartu': prepare_tartu(make_scenario(sources['hospital_map'], seconds_per_tick=10,
                                             tiago_config=sources['tiago_config'])),
         'clinic': make_clinic_scenario(sources['clinic_map'], sources['clinic_config'])}
for layout, seed in [('tartu', 11000), ('clinic', 12000)]:
    scenario = make_case(bases[layout], layout, seed, 'short_outage')
    evidence = run(checkpoint, root / ('rollout-' + layout), scenario, seed=seed, horizon=3, budget=4)
    assert evidence['model_steps'] > 0 and evidence['fallbacks'] == 0
    assert evidence['discrete_safety_violations'] == 0
    print(layout, json.dumps(evidence, ensure_ascii=False))
grid = json.loads((root / 'rollout-grid/evidence.json').read_text())
assert grid['model_steps'] > 0 and grid['fallbacks'] == grid['discrete_safety_violations'] == 0
print('CPU acceptance: parameters updated, checkpoint resumed, held-out inference and three closed loops ran')
PY
