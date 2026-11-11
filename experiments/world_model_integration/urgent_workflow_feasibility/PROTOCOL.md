# Precommitted urgent / dynamic retest development protocol

Immutable machine-readable parameter matrix: protocol.json. Prior thesis initial/dynamic Test groups are not reused. Source PR18 HEAD 4a3ca9edc10350d4fbe42538c00c3b19ad28fffb.

* Four smoke seeds: 902001, 902074, 902147, 902220.
* Twenty development seeds: 903001 + 73 * i (0 <= i < 20).
* Systematically varied factors: 4-6 jobs; 1-3 urgent; 1-2 idle-capable robots; bottleneck pool capacity 1 or 2; synthetic outage enabled for indices mod 4 in 2,3; retest probability 0%,35%,65%; publicly declared synthetic jitter 1 or 2; static urgency synthetic due offset 65/70/75 ticks; regular due 145 ticks. Total horizon 165.
* Workflows: A->B receive, B->C prep, C->D detect, D->A archive. Branch discovered only after stage3 detection completion: optional D->C retest, then C->A archive. No new clinical job is created. Exact task semantics are station-to-station transport/service abstraction, not actual laboratory reactions.
* Valid candidates: unchanged BusinessCPSAT (<=4 distinct matchings, 0.12 s cap and 0.02 deterministic). P0 first plan, P1 urgent-first, P2 nominal public full-workflow ETA, P3 analytic jitter-based approximate lateness risk. All use identical known matched plans, no model training or V4.
* Strict event policy: retest outcome SHA256(seed, job id) independent of tick, policy, evaluation ordering; no outcome until stage3 completed and visible only from next tick. Unknown future station outages never appear as a scheduling feature.
* Main operational job-level indicators urgent on-time / lateness lower bound / incompletion, routine restricted flow, all restricted flow, capacity wait, blocked motion, zero safety violations. Every unfinished job remains in every denominator.
* Factual P0 observable state and legal matching, before any future costs, select at most 12 candidate states, max one per scenario, earliest tick where urgent and routine waiting together and >=2 legal distinct candidate matchings. Prefix replay with same original seed only, P0 after selected action. No multiple conditional futures. Oracle is limited to these CP-SAT candidate matchings; never global.
* Hard stop on illegal outcome, hidden-information leak, replay prefix mismatch, or 1740 CPU seconds. All files hashed, jobs grouped by original scene, development-only. This is NOT clinical validation or independent test.
