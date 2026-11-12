"""PR54 independent SCENARIO bootstrap, strictly read-only original TEST.

Two trained network seeds are averaged WITHIN each of 12 independently
simulated scenarios, not treated as 24 independent datasets. Compare
validation-selected historical GRU/RGCN vs same-param snapshot-only and
persistence. This analysis does not train, retune, add simulated examples,
or change any model-selection threshold.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import json
import random


def mean(x):
    return sum(x)/len(x)


def interval(samples):
    samples=sorted(samples)
    n=len(samples)
    def at(q):
        pos=(n-1)*q
        i=int(pos)
        return samples[i]+(samples[min(i+1,n-1)]-samples[i])*(pos-i)
    return [at(.025),at(.975)]


def scenario_bootstrap(rows,field,replicas=5000,seed=1810):
    strata={reg:[x for x in rows if x["regime"]==reg]
            for reg in ("congestion","multi_fault","staggered_arrivals")}
    if any(len(v)!=4 for v in strata.values()):
        raise ValueError("expected 4 independent test scenarios per stress family")
    observed=mean([r[field] for r in rows])
    rng=random.Random(seed)
    values=[]
    for _ in range(replicas):
        draw=[r[rng.randrange(4)] for r in strata.values() for _ in range(4)]
        values.append(mean([v[field] for v in draw]))
    return {
        "difference_history_minus_snapshot":observed,
        "scenario_stratified_95pct_CI":interval(values),
        "wins":sum(r[field]<0 for r in rows),
        "ties":sum(r[field]==0 for r in rows),
        "losses":sum(r[field]>0 for r in rows),
        "independent_scenario_count":12,
        "replicas":replicas,
        "interval_conditions":"2 fixed learned seeds averaged within scene, same synthetic map, not inter-training-seed or cross-lab generalization"
    }


def run(input_file,out):
    if out.exists():
        raise FileExistsError("source read-only analysis must not overwrite")
    r=json.loads(input_file.read_text())
    if (r["schema"]!="pr54-new-one-step-public-history-independent-test-v1"
        or r["selection"]["schema"]!="pr54-locked-validation-selected-temporal-world-model-v1"
        or not r["selection"]["test_was_not_read_before_selection"]
        or r["selected_before_test"]!="public_history_gru"):
        raise AssertionError("frozen train/validation selected experiment not matched")
    test=r["test"]
    arms=("snapshot_only","public_history_gru")
    seeds=("1810","1811")
    by={}
    for arm in arms:
        for seed in seeds:
            x=test[arm][seed]
            if x["independent_scenarios"]!=12 or x["real_public_transitions"]!=672:
                raise AssertionError("not exact original 12 independent public-world episodes")
            for row in x["per_scenario"]:
                k=row["id"]
                by.setdefault(k,{"regime":row["regime"],"id":k})[(arm,seed)]=row
    if len(by)!=12:
        raise AssertionError("missing original independently tested scenes")
    rows=[]
    for episode in by.values():
        if any((arm,seed) not in episode for arm in arms for seed in seeds):
            raise AssertionError("missing frozen arm/seed score")
        node={arm:mean([episode[(arm,s)]["node_masked_mse"] for s in seeds])
              for arm in arms}
        brier={arm:mean([episode[(arm,s)]["device_brier"] for s in seeds])
               for arm in arms}
        pers=mean([episode[("snapshot_only",s)]["persistence_node_mse"]
                   for s in seeds])
        rows.append({
            "id":episode["id"],"regime":episode["regime"],
            "snapshot_node_mse":node["snapshot_only"],
            "history_node_mse":node["public_history_gru"],
            "snapshot_device_brier":brier["snapshot_only"],
            "history_device_brier":brier["public_history_gru"],
            "persistence_node_mse":pers,
            "delta_node_mse":node["public_history_gru"]-node["snapshot_only"],
            "delta_device_brier":brier["public_history_gru"]-brier["snapshot_only"]
        })
    summary={
        "schema":"pr54-original-independent-public-world-one-step-paired-scenes-v1",
        "scenarios":12,"neural_training":0,"new_factual_episodes":0,"gpu":0,
        "original_validation_model_choice_not_changed":r["selected_before_test"],
        "history_node_mse":mean([x["history_node_mse"] for x in rows]),
        "snapshot_node_mse":mean([x["snapshot_node_mse"] for x in rows]),
        "persistence_node_mse":mean([x["persistence_node_mse"] for x in rows]),
        "history_device_brier":mean([x["history_device_brier"] for x in rows]),
        "snapshot_device_brier":mean([x["snapshot_device_brier"] for x in rows]),
        "history_minus_snapshot_entity_mse":scenario_bootstrap(rows,"delta_node_mse"),
        "history_minus_snapshot_device_brier":scenario_bootstrap(rows,"delta_device_brier"),
        "per_true_independent_scene":rows,
        "limits":"teacher-forced one-step public states, small n=12 independent worlds; no H20 recursive or online decision result"
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
    print("PR54_INDEPENDENT_SCENARIO_PAIRED_AUDIT",json.dumps({
        k:v for k,v in summary.items() if k!="per_true_independent_scene"
    },ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--report",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    x=p.parse_args()
    run(x.report,x.out)


if __name__=="__main__":
    main()
