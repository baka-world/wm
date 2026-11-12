"""PR51 immutable study boundaries: source architecture, true-world labels, public features."""
from __future__ import annotations
from pathlib import Path
import json

from experiments.world_model_integration.stress_pr48.pr48_scenarios import PROTOCOL as PR48_PROTOCOL
from experiments.world_model_integration.unconstrained_pr49.pr49_online import PROTOCOL as PR49_PROTOCOL

PROTOCOL=Path(__file__).with_name("pr51_protocol.json")
REGIMES=("congestion","multi_fault","staggered_arrivals")
SPLITS=("train","validation","test")

def validate_protocol(p):
    if p["protocol"]!="PR51_DECISION_AWARE_RELATIONAL_SUPERVISION_20261112_V1":
        raise ValueError("PR51 protocol ID changed")
    if p["base_commit"]!="c85c0b02c7ef34d44a9347ffd113ae9fee6e3952":
        raise ValueError("source PR49 HEAD changed")
    if p["counterfactual_max_candidates"]!=4 or p["prediction_horizons"]!=[5,20]:
        raise ValueError("frozen true label comparison design changed")
    if p["resources"]["initial_training"] or not p["resources"]["initial_labels_are_cpu_true_world_simulation"]:
        raise ValueError("phase zero is actual factual label generation, not premature training")
    if p["action_branch_policy"]!=(
        "replay original CP-SAT[0] through tick anchor; at anchor replace only first legal plan with candidate j, then CP-SAT[0] for ALL subsequent ticks; compare equal prefix and candidate list SHA on each branch"):
        raise ValueError("causal branch action intervention changed")
    cases=p["cohort"]
    if len(cases)!=72 or len({x["seed"] for x in cases})!=72:
        raise ValueError("expected 72 fresh standalone scenarios")
    old=json.loads(PR49_PROTOCOL.read_text())
    pr48=json.loads(PR48_PROTOCOL.read_text())
    earlier={x["seed"] for x in old["cohort"]+pr48["cohort"]}
    if earlier & {x["seed"] for x in cases}:
        raise ValueError("reused older test cases")
    if any(x["seed"]<25000000 for x in cases):
        raise ValueError("unexpected previous seed namespace")
    for r in REGIMES:
        for split,count in (("train",12),("validation",4),("test",8)):
            cohort=[x for x in cases if x["regime"]==r and x["split"]==split]
            if len(cohort)!=count:
                raise ValueError("unbalanced independent splitting")
            if any(x["job_count"]!=(14 if r=="congestion" else 12) or
                   x["secondary_station_id"]!=("C" if r=="multi_fault" else None)
                   for x in cohort):
                raise ValueError("real workload parameters changed")
            if sum(x["mechanism"]=="within_support" for x in cohort)!=count//2:
                raise ValueError("mechanism balance changed")
    return True
