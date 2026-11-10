# PR18 read-only predictive information audit

This directory contains ONLY an offline parser for already committed evidence. It does not import, initialize, or call BusinessSimulator, V4, CP-SAT, or any planner. It never creates new trajectories.

Original PR18 baseline: `9aa554bc541e5a68cb74089feeecbbe62ada1aa0` (Draft). Inputs are the *immutable* prior `experiments/world_model_integration/urgent_workflow_feasibility/evidence` files, verified by full SHA-256 in `audit.py`.

From the repo root, with Python >=3.10 and no third-party packages:

```bash
python -m experiments.world_model_integration.predictive_information_audit.audit \
  --evidence experiments/world_model_integration/urgent_workflow_feasibility/evidence \
  --out /tmp/pr18-predictive-information-audit
python -m unittest discover \
  -s experiments/world_model_integration/predictive_information_audit/tests -v
```

Output directory must be empty: it writes SHA-indexed `score_masking.json.gz`, `job_transitions.json.gz`, `retest_information.json`, `three_fixed_oracles.json.gz`, `service_information.json`, `summary.json`, and `sha256.json`. The 3 gz files contain JSON, gzip header mtime=0; decompress via `gzip -dc <file>` or `gzip.open`. CPU runtime in summary is expected to vary slightly across machines; scientific counts and source hashes must match. `evidence/` contains one measured set of outputs, local offline test log and provenance.

**Comparisons:** P2/P3 are paired only after checking original same scenario, exact event/history contents, tick, candidate IDs/order, selected actions. Five mutually exclusive C0–C5 diagnoses are defined in `audit.class_one`, with `candidate_index` excluded from numerical business-score equality but included for actual choice ranking.

**Restrictions:** no new policy training, scenarios, V4 inference, old held-out reuse, reselecting the original Oracle states, simulator readjustment, changed randomness, or write access to original evidence. Single-seed Oracle candidates are **one-realization** evidence only; the alternative event timelines were not originally archived, so mechanistic event attribution is **BLOCKED**.

Human-readable findings and relevant literature: [docs/world-model-predictive-information-value.md](../../../docs/world-model-predictive-information-value.md).
