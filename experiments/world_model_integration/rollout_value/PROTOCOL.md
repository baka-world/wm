# PR18 frozen offline candidate cost-to-go protocol (DEVELOPMENT ONLY)

## Source, controls and literature audit

- Source head: `3663d93b1348fec068996bb86a10c6b80c2c1c08`; PR #18 remains Draft.
- Models frozen: V4 SHA256 `60c8a6c3d832da3e6519072688ea099f64ae120ba19d3bd2ec705f7f1a11ac0c`. Historical 5/10/20-step model rollouts are read from the archived CPU evidence, NOT rerun or fitted here.
- Three viewed independent-test seeds are reused solely for **development mechanism analysis**; results CANNOT be reported as fresh independent confirmation.
- Li & Womer, *European Journal of Operational Research* 246 (2015) 20–33, doi:10.1016/j.ejor.2015.04.015, sections 3.1–3.3: for each feasible action, simulate future sample paths under a fixed base heuristic; compare cost-to-go; optionally use CP as stronger base policy. The paper's sequential improvement result depends on terminating, sequentially consistent heuristics and accurate expectation; none is assumed for this repository.
  Primary PDF: https://profiles.umsl.edu/ws/portalfiles/portal/39828815/PublishedOnline-April2015.pdf
- Satic, Jacko & Kirkbride, *EJOR* 315 (2024) 454–469, doi:10.1016/j.ejor.2023.10.046, §§ 3.2–3.3 and Algorithms 1–2: simulation produces state features / realized discounted returns, fitted offline by least squares into linear action values for online use. Its objective is discounted profit, **not** our restricted mean flow. The present phase deliberately does **not fit** ADP coefficients.
  Open-access paper: https://gcris.agu.edu.tr/bitstreams/c9288a25-7b0d-49d1-a051-903a4a1bd675/download
  Official implementation: https://github.com/ugursatic/EJOR-DSRCMPSP (Julia, reference only).

## Predeclared actual evaluation (fixed before opening any future-rollout results)

1. Single frozen tick0 observation and CP-SAT candidate set per seed **43011, 43015, 43019**. Candidate counts = 2,3,4 respectively; no case searching or new scenarios.
2. Candidate `a` is implemented as the **one actual first-tick batch**, subsequent decisions follow one explicitly named policy:
   - **P1**: `nearest_batch(observation,candidates,geometry)`, exactly the legacy `replay_branch` continuation.
   - **P2**: new `BusinessCPSAT(... candidate_limit=4, time_limit_seconds=.5, seed=1810)` per FULL branch; at each next observed tick, call `proposals` on `BusinessObserver`'s actual history and choose its FIRST feasible plan. This is *receding CP-SAT*; no recursive model rollout.
3. Each branch restarts the ORIGINAL `BusinessSimulator` from the complete fixed scenario, not an injected observation. Tick0 observed state MUST equal the original archive state. Original base-seed P1 outcomes must match `replay_branch` and `v4-dispatch.json` EXACTLY. Same-seed duplicated branch must have identical full history and events hashes. Abort if any gate fails.
4. Exogenous service jitter is keyed by `seed + sample_id + phase + station_id`; apply the SAME seed to all candidate and policy branches (common random conditions). Fixed outage windows, workflow releases and geometry are retained. All selected states are tick0 before service randomness; **this does not justify conditional reseeding of later tick states**.
5. Frozen offsets `[0,101,203,307,409,503,607,709]` (at most eight candidate-paired futures); no adaptive seed selection. If every station has jitter=0 (43011), only offset 0 is evaluated: repeated seeds would not generate independent futures. Station outages remain fixed; independent multi-outage sampling is NOT available. These are design-based finite synthetic jitter futures, NOT established real-world conditional expectation estimates.
6. Primary objective = **full original business restricted mean flow**, from public job release to last workflow stage completion. For unfinished jobs, use simulated cutoff tick minus release (floored at zero); all configured jobs including later releases are counted as offline truth for consistency with prior business acceptance. Neither scoring model nor continuation policies read the future fault plan, future release table or realized sampled service duration as online inputs.
7. Report per-candidate paired seed sample outcomes and dispersion, offline P1/P2 ranks, E0=Rule5, E2=V4 5/10/20, E3=Rule 5/10/20. Model nominal score `unfinished_job_ticks + terminal_work` is **not** identical to restricted mean flow: compare ranks, chosen-policy regret and **candidate-centered gap error**; do NOT interpret raw absolute score difference as a calibrated flow-time MAE.
8. Existing B2 is an offline `EmpiricalHazards.predict` over factual feature rows, not a `predict_next` adapter with repeatable sampling, state transitions and budget matching. Not adding a new adapter solely for a table; its Brier remains historical prediction evidence, not a comparable cost-to-go result.
9. Preflight: frozen archive parity; original P1 oracle reproduction; first-seed P1/P2 full branch; same-seed repeated full simulation; strict legality and capacity audit. Main run only if all preflight gates pass. Max wall time 1500s, 0 GPU/0 V4 training, fixed configurations and CPU-only PyTorch. CI outputs written write-once, always archived.
10. Methodological stop: reject future-value claims if any initial state/candidate parity or prefix gate fails, any same-seed re-run differs, if replay uses outcome data online, or if full cost reference is inconsistent with old `replay_branch`. Any reliable E1 reference remains an *offline simulator benchmark*, not a deployable predictor or Oracle estimator of unknown stochastic futures.

## Determinism and incomplete conditional futures

`replay_branch()` restarts the entire simulator with original seed, replays each factual prefix assignment, compares observations tick by tick, and compares full history snapshots at the prefix. Internal services, workflow state, outages, dynamic service estimates and robot recovery are recreated by deterministic simulation; **the original post-tick0 hidden state is not reconstructed merely by copying observation**. The sampled random service durations depend on scenario-defined jitter; 43011 uses zero jitter and is structurally deterministic under this mechanism. Multiple seeds elsewhere are paired, not i.i.d. evidence from real laboratories.

## Separate scope of work

No changes to the online policy, `workflow_components`, CP-SAT, PIBT/LaCAM3, service model weights, B2 fit, Rule logic, data generator, or PR #19. This is an offline hypothesis-screening audit to decide whether a new, independently validated ADP study is justified.
