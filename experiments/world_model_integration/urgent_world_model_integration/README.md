# PR18: Frozen V4 real CPU integration in urgent/retest transport

This code runs actual frozen V4 neural network inference. It is not a
surrogate and does not train or select new weights.

P4 uses ProgressAwareRule + WorkflowDynamics. P5 uses the unchanged
SessionRuleDynamics(feature_version='v4', experimental=True) in the same
WorkflowDynamics. Both use the same legal BusinessCPSAT candidates, current
public priority observation, fixed H=5 particles=3, and identical job-level
urgent objective and subsequent nearest dispatch. The leading lexicographic
urgency-coverage term is frozen and can hide later prediction differences.

The real ConditionalRetestSimulator generates retest events only after
detection. Neither forecast gets the private keyed retest truth. The
prediction follows the currently public default route until a real retest
event is observed. Forecasts are deliberately approximate, not true
conditional joint distributions over retest and future resource congestion.

The checkpoint is downloaded from GitHub Actions training artifact ID
13765135588, v4-data.pt, verified with SHA256:
60c8a6c3d832da3e6519072688ea099f64ae120ba19d3bd2ec705f7f1a11ac0c.

Previously seen smoke and development groups are used for integration
diagnostics ONLY; they do not establish new independent generalization.
No changed production CP-SAT, V4 weights, real simulator or motion planner.

Commands with CPU-only pinned dependencies:
  python -m unittest discover -s experiments/world_model_integration/urgent_world_model_integration/tests -v
  python -m experiments.world_model_integration.urgent_world_model_integration.run --checkpoint /path/v4-data.pt --group smoke --out /tmp/urgent-v4-smoke
  python -m experiments.world_model_integration.urgent_world_model_integration.run --checkpoint /path/v4-data.pt --group development --out /tmp/urgent-v4-dev

The Actions workflow uploads case results, per-candidate predictions and
paired common-observation diagnostics. P0-P3 original outcomes are unchanged,
see urgent_workflow_feasibility/evidence/summary.json.

## Candidate-invariant V4 service randomness diagnostic

The original experiment used the same random generator seed per candidate.
But SessionRuleDynamics consumes random values in *query-list order*, so
when candidate actions change which sessions are queried, the same active
session can receive a different uniform variate. This creates a potential
Monte-Carlo coupling confound, particularly with only three particles.

A second protocol uses
`--random-coupling keyed`. Each eligible service session is assigned a
deterministic uniform based on (scenario seed, particle, forecast tick,
robot, sample, phase, station); it is **not** based on the private true
service duration. This couples equivalent public sessions across candidate
plans while keeping the V4 model and objective fixed.

Both methods remain stochastic finite-particle approximations. This is a
**post-hoc diagnostic ablation on already examined seeds**, not a replacement
for the original outcome or an independent performance claim. The original
`--random-coupling stream` is unchanged and remains the default.

    python -m experiments.world_model_integration.urgent_world_model_integration.run --checkpoint /path/v4-data.pt --group smoke --random-coupling keyed --out /tmp/keyed-smoke
    python -m experiments.world_model_integration.urgent_world_model_integration.run --checkpoint /path/v4-data.pt --group development --random-coupling keyed --out /tmp/keyed-development
