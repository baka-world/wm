# Frozen budgeted public queue continuation protocol

The 20-tick budgeted terminal extension is fixed **before any run on
the 12 new synthetic confirmation seeds** in budgeted_confirmation_protocol.json.
There are no model updates, hyperparameter searches or business target changes.

After identical H5 rollouts for Rule (P4) and frozen V4 (P5), each candidate
continues for at most 20 steps using the SAME existing legal public Rule
workflow transition and nearest dispatch; the existing optimistic terminal
cost then values unfinished tasks. The goal is to capture useful near-term
robot/workstation queue formation at a fraction of the CPU cost of the
original unbounded until-tick-165 continuation.

Each new confirmation scene also runs actual P0-P3 business baselines
(including urgent-first P1). Model variants P4/P5 use identical initial
candidate space, objective, random coupling and next-step simulator.
The original V4 weights are SHA-verified and remain frozen.

The 3 known previously examined failure cases are run separately as
diagnostics; they are NOT pooled into the 12 new synthetic confirmation cases.

Limitations: same synthetic generator, no actual laboratory data, no
private future retest or outages, non-calibrated future completion proxy.
No positive result is automatically attributable to learned V4 relative
to the strong Rule and P1 baselines.

The GitHub CPU Actions workflow runs actual inference, 5 tests, known
diagnostics, 12 precommitted synthetic scenes × 6 arms, and uploads raw
predictions/decisions, task-level results, CPU timers and SHA-256.
