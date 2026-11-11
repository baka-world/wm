# PR18 urgent + conditional-retest feasibility (synthetic only)

This isolated experiment reuses BusinessSimulator, BusinessObserver, BusinessCPSAT, and the existing motion safety authority. Production CP-SAT and workflow scoring are never modified. No V4, GPU, training or old holdout.

## Run

The dedicated workflow installs requirements-paper-reuse.lock, runs the 10 CPU unit tests, then executes:

    python -m experiments.world_model_integration.urgent_workflow_feasibility.run --out /tmp/urgent-result

Protocol is frozen in protocol.json BEFORE any outcomes. It runs four smoke groups, then twenty development groups, with four deterministic policies P0/P1/P2/P3 sharing one bounded CP-SAT candidate set. Then it seals at most twelve P0 observational opportunities BEFORE factual alternative replay. Only the same original seed is used for every candidate.

P0 uses BusinessCPSAT's first proposed matching, P1 urgent-first deterministic, P2 current public-workflow ETA, P3 approximate analytic deadline risk under a uniformly distributed declared synthetic jitter. P3 explicitly does not know a future outage or retest result and is not a real-workflow probabilistic Oracle.

The conditional retest branch is materialized inside the experiment simulator only when the detection stage completes. Before then, its truth and potential extra stage are absent from observations. A seeded SHA-256 keyed event decides it independently of strategy execution order. All job metrics count the final realized stage once. Time limit censoring remains in denominators. Reviewer should not mistake this for clinical end-to-end report TAT.

All outcomes are development-only. The evidence directory contains full_raw.json.gz with raw events, histories, task metrics, and decisions for all groups and policies; oracle.json with per-candidate observed-cost bounds; selection_lock.json; paired.json; summary.json; SHA-256 manifest; original CPU logs and test results.

Reference: [annual report PDF page 13](https://disc.static.szse.cn/disc/disk03/finalpage/2026-04-29/be7411b3-bfff-4727-b8f3-16bed8ac801f.PDF); [LABAS MAX urgent/routine intake](https://www.maccura.com/syszt/38.html); [LABAS urgent automation](https://www.maccura.com/new/343.html). All rates, deadlines, service durations, robot counts and retest rates here are invented synthetic research controls, never claimed as Maccura production parameters.
