# Targeted post-hoc terminal queueing continuation protocol

Three previously examined cases: 902147 (Smoke), 904096 and 904388
(Development). These seeds were selected because results were already seen,
so this is NOT an independent comparison.

Original five-step _projected_job_cost sums independently optimistic
residual work. At 904096:76 the Rule forecast scored three different
candidate assignments identically for all business components, although
the real one-robot system finishes their work in different orders.

The experimental --terminal-mode rule_continuation changes only terminal
evaluation: both P4 and P5 use their original H=5 predictor (including
the same frozen V4 weights), then a shared public deterministic
ProgressAwareRule+WorkflowDynamics with nearest legal follow-on actions,
until all currently observed jobs finish or the synthetic cutoff. This
explicitly predicts legal resource queues and stage sequencing.

Neither arm is given hidden future retest truth, future faults, unreleased
jobs, private simulator countdown or extra training. Missing future
arrivals and conditional retest mean this is NOT a real-world calibrated
forecast. Do not compare this post-hoc selected set as a generalization
result. The workflow also checks historical P4/P5 snapshot hashes to
ensure optimistic-mode behavior has not changed.
