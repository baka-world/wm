"""PR52 READ-ONLY independent-test model/regret attribution.

No training, no model selection, no real simulator, and no new true
trajectories. Restores SHA-verified already-fitted eight checkpoints
and computes paired candidate choice consequences in the original
48 independent test scenarios.

Bootstrap resamples SCENARIOS (16 within each of three stress regimes),
not 4 candidates or 2 model seeds, to avoid pseudoreplication. These
intervals condition on the already-trained two model seeds and do not
capture all run-to-run/real-company distribution uncertainty.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import math
import random

import torch
from experiments.world_model_integration.entity_pr52.pr52_train import (
    ARMS,EntityConditionalRelationalWorld,decision_cost,prepare,load_split,
    PROTOCOL
)

REGIMES=("congestion","multi_fault","staggered_arrivals")


def _mean(values):
    return sum(values)/len(values) if values else None


def interval(values,lower=.025,upper=.975):
    values=sorted(values)
    n=len(values)
    if not n:raise ValueError("no bootstrap values")
    def at(p):
        x=(n-1)*p
        i=math.floor(x);j=math.ceil(x)
        return values[i]*(1-(x-i))+values[j]*(x-i)
    return [at(lower),at(upper)]


def paired_bootstrap(cases,active,reference,reps=4000,seed=1810):
    if not cases: raise ValueError("no scenarios for confidence interval")
    rng=random.Random(seed)
    strata={r:[c for c in cases if c["regime"]==r] for r in REGIMES}
    if any(len(v)!=len(cases)//3 for v in strata.values()):
        raise ValueError("broken 3-equal-regime independent cohort")
    def delta(row):
        return row["mean_two_training_seeds_regret"][active]-row[
            "mean_two_training_seeds_regret"][reference]
    original=_mean([delta(c) for c in cases])
    draw=[]
    for _ in range(reps):
        selected=[]
        for r in REGIMES:
            group=strata[r]
            selected += [group[rng.randrange(len(group))] for _ in range(len(group))]
        draw.append(_mean([delta(c) for c in selected]))
    ci=interval(draw)
    return {
        "comparison":active+"_minus_"+reference,
        "mean_delta_true_regret_ticks":original,
        "bootstrap_95pct_CI_ticks":ci,
        "wins":sum(delta(c)<-1e-8 for c in cases),
        "ties":sum(abs(delta(c))<=1e-8 for c in cases),
        "losses":sum(delta(c)>1e-8 for c in cases),
        "unit":"48 public-state scenario seeds, stratified bootstrap within each 16-case regime",
        "repetitions":reps,
        "ci_conditioning":"fixed learned two-seed models, not neural-retraining or external deployment uncertainty",
    }


def restore_ckpt(path,arm,seed,hidden):
    raw=torch.load(path,map_location="cpu",weights_only=False)
    if raw["arm"]["id"]!=arm or raw["seed"]!=seed:
        raise ValueError("model id/seed archive mismatch")
    net=EntityConditionalRelationalWorld(hidden,raw["arm"]["relations"])
    net.load_state_dict(raw["model"])
    net.eval()
    return net


def evaluate(data,models_dir,selection_path,train_report_path=None):
    cfg=json.loads(PROTOCOL.read_text())
    chosen=json.loads(selection_path.read_text())
    if (chosen["schema"]!="pr52-validation-only-multi-seed-world-model-selection-v1"
        or not chosen["independent_test_not_read_during_training_or_selection"]):
        raise ValueError("no prior frozen validation model selection")
    if chosen["selected_arm"] not in ARMS:
        raise ValueError("selected model not in frozen roster")
    if len(data)!=48 or len({s["seed"] for s in data})!=48:
        raise ValueError("not original full 48 test scenarios")
    trained={}
    for arm in cfg["train_arms"]:
        a=arm["id"]
        for seed in cfg["training"]["seeds"]:
            candidates=list(models_dir.rglob(a+"_seed"+str(seed)+".pt"))
            if len(candidates)!=1:
                raise FileNotFoundError("exact archived trained checkpoint missing "+a+"/"+str(seed))
            trained[(a,seed)]=restore_ckpt(
                candidates[0],a,seed,cfg["training"]["hidden_size"])
    results={c["id"]:{"id":c["id"],"seed":c["seed"],
                       "regime":c["regime"],"mechanism":c["mechanism"],
                       "regret_by_arm_seed":{}}
             for c in data}
    for arm in cfg["train_arms"]:
        a=arm["id"]
        prepared,discarded=prepare(data,arm["relations"],"test")
        if discarded:
            raise AssertionError("read-only analysis cannot silently exclude test scenarios")
        for s in cfg["training"]["seeds"]:
            model=trained[(a,s)]
            with torch.no_grad():
                for g in prepared:
                    b=g["batch"]
                    node,event=model(b)
                    predicted=decision_cost(node,event,b).tolist()
                    costs=g["true_cost"].tolist()
                    pick=min(range(len(predicted)),key=lambda i:(predicted[i],i))
                    regret=costs[pick]-min(costs)
                    original_regret=costs[0]-min(costs)
                    case=results[g["id"]]
                    if "cp_sat_first_candidate_regret" not in case:
                        case["cp_sat_first_candidate_regret"]=original_regret
                        case["real_oracle_best_makespan_ticks"]=min(costs)
                        case["real_all_candidate_makespan_ticks"]=costs
                    elif case["cp_sat_first_candidate_regret"]!=original_regret:
                        raise AssertionError("true business outcomes changed during readonly audit")
                    ordered=sorted(predicted)
                    minimum_rank_gap=(ordered[1]-ordered[0] if len(ordered)>=2 else None)
                    case["regret_by_arm_seed"].setdefault(a,{})[str(s)]={
                        "chosen_index":pick,
                        "candidate_score_gap_top_two":minimum_rank_gap,
                        "candidate_scoring_float_values":predicted,
                        "true_regret_ticks":regret,
                        "estimated_completed_h20":[-v for v in predicted],
                        "actual_selected_makespan_ticks":costs[pick]
                    }
    rows=list(results.values())
    for case in rows:
        case["mean_two_training_seeds_regret"]={
            a:_mean([case["regret_by_arm_seed"][a][str(s)]["true_regret_ticks"]
                     for s in cfg["training"]["seeds"]])
            for a in ARMS}
        case["mean_two_training_seeds_regret"]["cp_sat_first"] = (
            case["cp_sat_first_candidate_regret"])
    replay={}
    if train_report_path is not None:
        archived=json.loads(train_report_path.read_text())
        if archived.get("schema")!="pr52-trained-entity-conditional-world-model-independent-test-v1":
            raise ValueError("not original independently evaluated training result")
        for arm in ARMS:
            replay[arm]={}
            for seed in cfg["training"]["seeds"]:
                key=str(seed)
                original=archived["independent_test"][arm][key]["metrics"][
                    "mean_true_candidate_regret_ticks"]
                reproduced=_mean([
                    r["regret_by_arm_seed"][arm][key]["true_regret_ticks"]
                    for r in rows])
                tiny=sum(
                    r["regret_by_arm_seed"][arm][key]["candidate_score_gap_top_two"]
                    is not None and
                    r["regret_by_arm_seed"][arm][key]["candidate_score_gap_top_two"]<1e-5
                    for r in rows)
                replay[arm][key]={
                    "original_training_report_regret":original,
                    "readonly_recomputed_regret":reproduced,
                    "difference_ticks":reproduced-original,
                    "cases_top2_forecast_gap_below_1e_minus_5":tiny,
                    "note":"Numerical instability diagnostic only. Rank selection is UNCHANGED."
                }
    paired={}
    for arm in ARMS:
        if arm!="rel_entity_pred_only":
            paired[arm]=paired_bootstrap(rows,arm,"rel_entity_pred_only")
        paired[arm+"_vs_cp_sat"]=paired_bootstrap(rows,arm,"cp_sat_first")
    return {
        "schema":"pr52-readonly-48-paired-scene-regret-intervals-v1",
        "frozen_validation_selected_arm":chosen["selected_arm"],
        "source":"original SHA-verified PR52 228-case true-world data and 8 frozen PyG optimizer checkpoints",
        "new_training":0,"new_simulator_episodes":0,"gpu":0,
        "scenarios":len(rows),
        "point_estimates":{a:{
            "regret_two_seed_mean":_mean([r["mean_two_training_seeds_regret"][a] for r in rows]),
            "by_regime":{reg:_mean([r["mean_two_training_seeds_regret"][a]
                                    for r in rows if r["regime"]==reg])
                         for reg in REGIMES}}
            for a in (*ARMS,"cp_sat_first")},
        "pairwise_stratified_95pct_bootstrap":paired,
        "replay_consistency_against_original_training_report":replay,
        "cases":rows,
        "interpretation_limits":"Fixed candidate counterfactual regret at first action; NOT full-online policy performance or neural-training seed uncertainty; test-based loss changes forbidden",
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--test",type=Path,required=True)
    p.add_argument("--models",type=Path,required=True)
    p.add_argument("--selection",type=Path,required=True)
    p.add_argument("--original-train-result",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():
        raise FileExistsError("original test audit write-once")
    data=load_split(a.test,"test")
    torch.set_num_threads(1)
    result=evaluate(data,a.models,a.selection,a.original_train_result)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print("PR52_READONLY_PAIRED_TEST_BOOTSTRAP",json.dumps({
        "selected":result["frozen_validation_selected_arm"],
        "estimates":result["point_estimates"],
        "bootstrap":result["pairwise_stratified_95pct_bootstrap"],
        "frozen_training_replay_audit":result["replay_consistency_against_original_training_report"]
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
