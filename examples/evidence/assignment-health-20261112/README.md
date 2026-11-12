# Assignment-specific features and health observability (2026-11-12)

This directory contains compact evidence for the research note
[`docs/assignment-health-observability-20261112.md`](../../../../docs/assignment-health-observability-20261112.md).

Scope: offline/secondary analysis on PR #16/#17 artifacts and a new diverse fault dataset
generated with the PR #16 `BusinessSimulator`. No model/training code changes, no checkpoints,
no full data committed, and no deployable scheduling claim.

## Headline results

- 8 base cases × 31 variants × 4 plans = 248 decisions / 992 records; all completed, 0 safety violations.
- Structured-5 fault regret: 376.333 (mean 1.568, 71/240 hits).
- Per-station ridge with immediate/accurate station health: **187.333** (mean 0.781, 140/240), normal mean flow 41.562 (Structured 41.667).
- Best fixed plan oracle: 233.000; station lookup: 191.333; best station oracle: 112.833; full oracle: 0.
- Health detection sensitivity: 75% → 235.9; 50% → 283.2; no signal → 376.3; 25% false positives add ~0.076 normal mean flow.
- Health latency: 0 tick → 187.3; 2 ticks → 1291.5; 5 ticks → 2220.2; 10 ticks → 3338.3.
- Partial commitment with late health: 452.7 (2 ticks) / 835.2 (5 ticks), still worse than Structured 376.3.
- Historical prior only: deployable prior lookup 2.373 weighted mean regret, worse than Structured 2.106; exact health 0.820.

## Files

| file | content |
|---|---|
| `hierarchy.csv` | policy hierarchy: Structured, best fixed, lookup, per-station ridge, oracles |
| `health-sensor.csv` | detection/false-positive sensitivity |
| `health-latency.csv` | online health latency |
| `partial-commitment.csv` | late-health partial commitment |
| `prior-policy.csv` | historical prior vs exact health |
| `summary.json` | machine-readable key numbers |
| `SHA256SUMS` | checksums |

## Limitations

- The health flag is synthetic (exact station outage); real deployment requires device health,
  fault/recovery/alarm/error logs, service start/end and event-time telemetry.
- The reward/labels are counterfactual simulator flows for all candidates; production observes
  only the selected candidate.
- The dataset is a small synthetic benchmark; it is not a clinical or physical safety claim.
- The current V4-H20 stack is separate; this PR does not modify it or claim integration.

## Reproduction

The analysis used PR #16 raw oracle confirmation, PR #17 observability audit, PR #21/27 artifacts,
and the PR #16 `BusinessSimulator` with MovingAI maps. Raw artifacts and analysis scripts were
kept local (see the research note); only compact results are committed.
