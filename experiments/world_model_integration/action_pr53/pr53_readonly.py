"""PR53 one-shot independent test inference audit on FROZEN trained models.

Does not change checkpoint parameters, choose a different model, fit any new
score or run the simulator. The three neural seeds are NOT treated as
independent business scenarios. Bootstrap operates on 48 truly independent
scenario seeds, preserving 16 each in each registered stress family.

The primary validated model remains the one selected on VALIDATION before
TEST. No reinterpretation from independent TEST comparison is permitted.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import argparse
import json
import random
import math

import torch

from experiments.world_model_integration.entity_pr52.pr52_train import prepare,decision_cost
from experiments.world_model_integration.action_pr53.pr53_train import (
    ARMS,SEEDS,REGIMES,PROTOCOL,ConsistentEntityWorld,
    load_physical_split
)

def mean(items):
    return sum(items)/len(items)

def quantile(items,p):
    seq=sorted(items)
    z=(len(seq)-1)*p
    l=int(z)
    h=min(l+1,len(seq)-1)
    return seq[l]+(seq[h]-seq[l])*(z-l)

def bootstrap(rows,new,base,reps=4000):
    strata={r:[x for x in rows if x["regime"]==r] for r in REGIMES}
    if any(len(v)!=16 for v in strata.values()):
        raise ValueError("requires 16 independent scenarios in each stress regime")
    def d(x):
        return x["mean_three_seed_regret"][new]-x["mean_three_seed_regret"][base]
    rng=random.Random(1810)
    delta=[d(x) for x in rows]
    samples=[]
    for _ in range(reps):
        selected=[]
        for r in REGIMES:
            group=strata[r]
            selected.extend(group[rng.randrange(16)] for _ in range(16))
        samples.append(mean([d(x) for x in selected]))
    return {
        "comparison":new+"_minus_"+base,
        "true_regret_delta_ticks":mean(delta),
        "stratified_scenario_bootstrap_95pct": [
            quantile(samples,.025),quantile(samples,.975)],
        "wins":sum(x<0 for x in delta),
        "ties":sum(x==0 for x in delta),
        "losses":sum(x>0 for x in delta),
        "resamples":reps,
        "inference_unit":"48 independent true-world scenarios; each model's 3 neural initializations averaged within scenario",
        "limits":"conditional on these 3 trained networks and synthetic map/mechanisms; not online policy or training-seed generalization"
    }

def run(data_root,model_root,out):
    if out.exists():
        raise FileExistsError("PR53 blinded audit cannot overwrite")
    cfg=json.loads(PROTOCOL.read_text())
    result_files=list(model_root.rglob("result.json"))
    selection_files=list(model_root.rglob("selection.json"))
    if len(result_files)!=1 or len(selection_files)!=1:
        raise FileNotFoundError("exactly one archived ORIGINAL result and selection required")
    if result_files[0].parent!=selection_files[0].parent:
        raise ValueError("mismatched archived model selection and test report")
    reference=json.loads(result_files[0].read_text())
    freeze=json.loads(selection_files[0].read_text())
    if (reference["schema"]!="pr53-consistent-relational-action-contrast-independent-test-v1"
        or freeze["schema"]!="pr53-immutable-validation-only-calibrated-model-selection-v1"
        or not freeze["independent_test_not_loaded"]):
        raise ValueError("not original frozen model selection and test report")
    selected=freeze["selected"]
    if selected not in ARMS or selected!=reference["selected_by_validation_before_test"]:
        raise AssertionError("independent test selected a different arm")
    data=load_physical_split(data_root,"test")
    torch.set_num_threads(1)
    records={x["id"]:{"id":x["id"],"regime":x["regime"],
                       "split":x["split"],"mechanism":x["mechanism"],
                       "seed":x["seed"],"arm_seed":{}}
             for x in data}
    replay={}
    for arm in cfg["train_arms"]:
        name=arm["id"]
        prepared,discarded=prepare(data,True,"test")
        if discarded:
            raise AssertionError("cannot silently drop registered test scenarios")
        replay[name]={}
        for seed in SEEDS:
            key=name+"_seed"+str(seed)
            candidates=list(model_root.rglob(key+".pt"))
            if len(candidates)!=1:
                raise FileNotFoundError("exact original frozen checkpoint absent "+key)
            state=torch.load(candidates[0],map_location="cpu",weights_only=False)
            if state["seed"]!=seed or state["arm"]!=arm:
                raise ValueError("mismatched seed or declared objective")
            model=ConsistentEntityWorld(cfg["train"]["hidden_size"],arm["consistent"])
            model.load_state_dict(state["model"])
            model.eval()
            regrets=[]
            margins=[]
            with torch.no_grad():
                for group in prepared:
                    n,e=model(group["batch"])
                    pred=decision_cost(n,e,group["batch"]).tolist()
                    costs=group["true_cost"].tolist()
                    picked=min(range(len(pred)),key=lambda i:(pred[i],i))
                    regret=costs[picked]-min(costs)
                    score_gap=sorted(pred)[1]-sorted(pred)[0]
                    regrets.append(regret)
                    margins.append(score_gap)
                    case=records[group["id"]]
                    case["arm_seed"].setdefault(name,{})[str(seed)]={
                        "picked":picked,"true_regret":regret,
                        "true_costs":costs,
                        "predicted_score":pred,
                        "top2_score_gap":score_gap,
                        "near_tie_1e5":score_gap<1e-5,
                        "near_tie_1e3":score_gap<1e-3,
                    }
                    if "cp_sat_first_true_regret" not in case:
                        case["cp_sat_first_true_regret"]=costs[0]-min(costs)
                    elif case["cp_sat_first_true_regret"]!=costs[0]-min(costs):
                        raise AssertionError("underlying factual world changed")
            original=reference["all_arms_new_blind_test"][name][str(seed)]
            replay[name][str(seed)]={
                "archived_regret":original["true_regret_ticks"],
                "readonly_regret":mean(regrets),
                "difference":mean(regrets)-original["true_regret_ticks"],
                "archived_near_tie_1e5":original["near_top2_1e_5"],
                "readonly_near_tie_1e5":sum(x<1e-5 for x in margins)
            }
    rows=list(records.values())
    for case in rows:
        case["mean_three_seed_regret"]={
            arm:mean([case["arm_seed"][arm][str(seed)]["true_regret"]
                      for seed in SEEDS])
            for arm in ARMS}
        case["mean_three_seed_regret"]["cp_sat_first"]=case["cp_sat_first_true_regret"]
    comparisons={}
    for arm in ARMS:
        if arm!="consistent_pred_only":
            comparisons[arm+"_vs_consistent_prediction"]=bootstrap(
                rows,arm,"consistent_pred_only")
        comparisons[arm+"_vs_cp_sat"]=bootstrap(rows,arm,"cp_sat_first")
    fully_reproduced=all(
        abs(v["difference"])<1e-8 and
        v["archived_near_tie_1e5"]==v["readonly_near_tie_1e5"]
        for arm in replay.values() for v in arm.values())
    result={
        "schema":"pr53-stratified-readonly-fresh-test-intervals-v1",
        "selected_validation_arm_unchanged":selected,
        "full_training_report_replayed_exactly":fully_reproduced,
        "original_replay_by_arm_and_seed":replay,
        "mean_three_seed_regret_by_arm":{
            arm:mean([x["mean_three_seed_regret"][arm] for x in rows])
            for arm in (*ARMS,"cp_sat_first")},
        "scenarios":48,
        "paired_stratified_scene_bootstrap":comparisons,
        "new_training":0,"new_true_simulations":0,
        "unselected_models_not_promoted_from_test":True,
        "public_graph_future_state_only":True,
        "cases":rows
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print("PR53_FROZEN_BLIND_TEST_SENSITIVITY",json.dumps({
        "selected":selected,"all_replayed":fully_reproduced,
        "regret":result["mean_three_seed_regret_by_arm"],
        "bootstraps":result["paired_stratified_scene_bootstrap"],
        "original_replay":replay
    },ensure_ascii=False),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True,type=Path)
    p.add_argument("--models",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    x=p.parse_args()
    run(x.input,x.models,x.out)

if __name__=="__main__":
    main()
