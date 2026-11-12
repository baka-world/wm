"""PR53 new, disjoint true-world counterfactual entity dataset.

Reuses EXACT existing PR52 public observation, CP-SAT candidate and same
factual-prefix branch code. Nothing here reads future task releases or
private fault duration into world-model graph inputs. Hidden future is
used only as the factual supervisor inside BusinessSimulator.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import hashlib
import json
import time

from experiments.world_model_integration.stress_pr48.pr48_scenarios import PROTOCOL as STRESS_PROTOCOL
from experiments.world_model_integration.entity_pr52.pr52_labels import collect_case

PROTOCOL=Path(__file__).with_name("pr53_protocol.json")
REGIMES=("congestion","multi_fault","staggered_arrivals")
SPLITS=("train","validation","test")


def validate(p):
    if p.get("protocol")!="PR53_ACTION_CONTRAST_CONSISTENT_WORLD_DYNAMICS_20261112_V1":
        raise ValueError("unexpected PR53 protocol")
    if p["parent_pr51_head"]!="2bf9d4266ccd9c7e9fa169da9d31b108068f7f01":
        raise ValueError("parent research baseline changed")
    if p["counterfactual_max_candidates"]!=4 or p["prediction_horizons"]!=[5,20]:
        raise ValueError("candidate/horizon changed")
    episodes=p["cohort"]
    if len(episodes)!=228 or len({v["seed"] for v in episodes})!=228:
        raise ValueError("expected exactly 228 fresh seeds")
    if any(v["seed"]<43000000 for v in episodes):
        raise ValueError("prior study seed namespace reused")
    for r in REGIMES:
        for split,n in (("train",48),("validation",12),("test",16)):
            found=[v for v in episodes if v["regime"]==r and v["split"]==split]
            if len(found)!=n or sum(x["mechanism"]=="within_support" for x in found)!=n//2:
                raise ValueError("unbalanced independent split")
            if any(x["job_count"]!=(14 if r=="congestion" else 12)
                   or x["secondary_station_id"]!=("C" if r=="multi_fault" else None)
                   for x in found):
                raise ValueError("original BusinessSimulator pressure mechanics changed")
    arms=p["train_arms"]
    if len(arms)!=5 or len({a["id"] for a in arms})!=5:
        raise ValueError("frozen control/contrast/decision ablation missing")
    if p["train"]["seeds"]!=[1810,1811,1812]:
        raise ValueError("neural training seed precommitment changed")
    return True


def run(regime,out):
    cfg=json.loads(PROTOCOL.read_text())
    validate(cfg)
    if out.exists():
        raise FileExistsError("true-world labels are write-once")
    stress=json.loads(STRESS_PROTOCOL.read_text())
    started=time.process_time()
    groups=[]
    for s in cfg["cohort"]:
        if s["regime"]!=regime:
            continue
        result=collect_case(s,cfg,stress)
        groups.append(result)
        if len(groups)%12==0:
            print("PR53_TRUE_LABEL_PROGRESS",regime,len(groups),
                  "decision_differences",sum(x.get("nontrivial_rank_signal",False) for x in groups),
                  flush=True)
    if len(groups)!=76:
        raise AssertionError("source stage not exhaustive")
    summary={
        "regime":regime,
        "new_scenarios":76,
        "counterfactual_true_branches":sum(x.get("candidate_count",0) for x in groups),
        "labelable":sum(x["labelable"] for x in groups),
        "full_H5_H20_public_entity_labels":sum(x.get("full_entity_horizon_labels_available",False)
                                               for x in groups),
        "nontrivial_actions":sum(x.get("nontrivial_rank_signal",False) for x in groups),
        "censored":sum(x["labelable"] and not x.get("all_branches_completed",False) for x in groups),
        "by_split":{split:sum(x["split"]==split for x in groups) for split in SPLITS},
        "new_neural_training":0,"gpu":0,"cpu_process_seconds":time.process_time()-started
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({
        "schema":"pr53-new-blind-entity-action-paired-truth-v1",
        "regime":regime,
        "protocol_sha256":hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "original_generator":"pr52_labels.collect_case and original PR48 BusinessSimulator",
        "cases":groups,"summary":summary
    },indent=2,ensure_ascii=False)+"\n")
    print("PR53_TRUE_LABEL_RESULT",json.dumps(summary,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--regime",required=True,choices=REGIMES)
    p.add_argument("--out",required=True,type=Path)
    x=p.parse_args()
    run(x.regime,x.out)


if __name__=="__main__":
    main()
