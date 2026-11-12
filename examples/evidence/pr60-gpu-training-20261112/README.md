# PR60 A100 GPU teacher-forced world-model training

This directory is the compact evidence for a real A100 GPU run of the PR59/PR60
`PublicMapTemporalRelationalWorld` entrypoint, based on PR60 head
`e32c402ecbf67d19788c0a5c99e5282d6b5017e5`.

## Scope and constraints

- Real CUDA training only; no CPU-only substitute.
- Train on 60 true TRAIN scenes / 9,046 public t→t+1 transitions; validate on
  12 independent VALIDATION scenes / 1,517 transitions.
- **No BLIND/TEST files read or generated.**
- No H20 autoregressive model, no online scheduling or decision-value claim.
- Same parameter count for `map_on` and `map_off` (52,953); hidden 48; batch 16;
  20 epochs; seeds 1810/1811/1812; validation-only model selection with node
  MSE + 0.25 device Brier.
- Raw data, full checkpoints/Adam/RNG, and source artifacts stay in the local
  experiment directory; only compact CSV/JSON/checksums are committed.

## Data/SHA verification

All downloaded artifacts passed `sha256sum -c SHA256SUMS`:

- PR59 TRAIN 24: 3,627 transitions
- PR60 new TRAIN congestion/multi_fault/staggered: 5,419 transitions
- TRAIN total: **9,046**
- VALIDATION: 12 scenes / **1,517**
- PR59 source head `48acd54b19805c4582bd616bb93fc81c7c92b10d`
- PR60 new source head `a1b004f1f615165b8d6ce550b64c4f83f076ec88`
- quality verdict PASS; BLIND_TEST source scenes generated=0

## GPU G0

- A100-SXM4-80GB, GPU 0–1/6–7 assigned; GPU 2–5 untouched (other users' vLLM).
- smoke: 2 epochs × map_off/map_on × seed 1810; 127 s wall; 52,953 params;
  allocator peak 39.48 MB; nvidia-smi 601 MiB; GPU util mean 9.14%/max 14%.
- finite loss; map-branch gradient L1=0.01178; entity relation gradient
  L1=0.63693; no NaN.

## Full training

Six single-arm/seed runs were used to avoid a single long process and to allow
parallel use of idle GPUs:

- GPU 0: `map_off` 1810/1811/1812 sequential
- GPU 1: `map_on` 1810
- GPU 6: `map_on` 1811
- GPU 7: `map_on` 1812

### Per-seed

| arm | seed | wall (s) | best epoch | node MSE | device Brier | new-task MAE | peak alloc (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| map_off | 1810 | 506.6 | 19 | 0.005236 | 0.004397 | 0.232941 | 37.65 |
| map_off | 1811 | 481.9 | 17 | 0.006151 | 0.004325 | 0.188078 | 37.79 |
| map_off | 1812 | 508.5 | 19 | 0.006214 | 0.004451 | 0.158438 | 37.86 |
| map_on | 1810 | 535.9 | 19 | 0.004196 | 0.004457 | 0.163585 | 37.77 |
| map_on | 1811 | 543.7 | 17 | 0.005582 | 0.004355 | 0.155641 | 37.91 |
| map_on | 1812 | 331.3 | 19 | 0.005623 | 0.004318 | 0.212555 | 37.98 |

### Aggregate (3 seeds)

| arm | params | node MSE mean ± sd | device Brier mean ± sd | new-task MAE mean |
|---|---:|---:|---:|---:|
| map_off | 52,953 | 0.005867 ± 0.000447 | 0.0043910 ± 0.0000515 | 0.193152 |
| map_on | 52,953 | **0.005134 ± 0.000663** | **0.0043768 ± 0.0000588** | **0.177260** |
| relative | — | **−12.50%** | −0.32% | −8.23% |

`map_on` wins node MSE in 3/3 seeds and 3/3 regimes; device Brier is better in
1/3 seeds (mean numerically better; staggered subgroup +0.84% worse); new-task
MAE is better in 2/3 seeds. Per-regime:

| regime | node MSE map_off/on | device Brier map_off/on |
|---|---:|---:|
| congestion | 0.005800 / **0.005272** | 0.002826 / **0.002755** |
| multi_fault | 0.006773 / **0.006112** | 0.006679 / **0.006677** |
| staggered_arrivals | 0.005029 / **0.004017** | **0.003668** / 0.003699 |

Results are in `arm_metrics.csv`, `arm_summary.csv`, `per_scenario.csv`,
`per_epoch.csv`, `summary.json`; `aggregate.py` reproduces them.

## Resources and checkpoints

- Summed GPU/card time: **2,907.85 s = 0.8077 GPU-hour**; actual parallel wall
  ~27 min on 4 GPUs. Allocator peak 37.65–37.98 MB; nvidia-smi peak 601 MiB;
  GPU util means 10.36%/3.98%/3.99%/3.95% (GPU 0/1/6/7), maxima 26/18/22/27%.
- Local archive `pr60-gpu-checkpoints-20261112.tar.gz` (6,661,520 B) holds all
  six best/last checkpoints (model + Adam + CPU/CUDA RNG), train reports and
  selection files. SHA256:
  `919a0b625659e492811a74bc5cdf76a60ccbf639acb32c33a420b7db30a74e14`.
  `checkpoint_manifest.csv` lists every best/last SHA and size;
  `resource.json` records GPU usage/card time.

## Verdict

**Conditional GO** for map information in one-step teacher-forced state
prediction: node MSE improves in every seed and regime, with no NaN and no BLIND
leakage. **NO-GO for checkpoint promotion, H5/H20 recursion, or online/decision
claims**: device calibration is mixed (1/3 seed Brier wins, staggered subgroup
slightly worse) and this is teacher-forced one-step only.

## Limitations

Teacher-forced one-step validation only. Before any H5/H20 or decision-aware
stage, run device on/off/recovery calibration and threshold analysis; freeze
selection before any BLIND evaluation. No BLIND result exists.

## Reproduce

```bash
python aggregate.py --runs-root /data/wm-a100-pr60-gpu-20261112/runs \
  --output ./aggregated \
  --gpu-monitor /data/wm-a100-pr60-gpu-20261112/logs/gpu-full-singles.csv
```
