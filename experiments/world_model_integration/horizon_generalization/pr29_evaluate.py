"""PR29: prospective horizon and scenario-group generalization of frozen RHWM V4.

The only prediction estimators are the existing public Rule and frozen V4.
The first two distinct CP-SAT legal actions are executed in paired *factual*
episodes, all subsequent steps use the same native CP-SAT algorithm. The
scenario-level oracle is only used for OFFLINE limited-candidate regret, never
the online predictor. No predictor selection or scene retuning on test results.

Published reference ideas: Reijnen et al. dynamic FJSP workload comparison,
PETS 2018 bounded particle planning, Wilder et al. decision-focused evaluation.
No new neural models, loss, policy optimizer, or random-event generator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from business_transport.dispatch import BusinessCPSAT
from business_transport.world_model import HybridSystemWorldModel
from experiments.world_model_integration.dynamic_identifiability.pr28_experiment import (
    _plans,run_factual,
)
from experiments.world_model_integration.system_world_model.evaluate_horizons import (
    METRICS,score,file_sha,
)
from lab_transport.decision_value import candidates_for

PROTOCOL=Path(__file__).resolve().parent/"pr29_protocol.json"
ARMS=("rule","v4")
HORIZONS=(5,10,20)


def first_action_config(proto):
    s=proto["first_action_pair"]["solver"]
    return {"max_ticks":proto["max_ticks"],
            "bounded_counterfactual":{"candidates":{
                "limit":s["limit"],"wall_seconds":s["wall_seconds"],
                "deterministic_seconds":s["deterministic_seconds"]}}}


def _scenario_seed_set(proto):
    previous=Path(__file__).resolve().parents[1]
    prior=json.loads((previous/"dynamic_identifiability"/"pr28_protocol.json").read_text())
    traffic=json.loads((previous/"transport_events"/"transport_protocol.json").read_text())
    cohorts=json.loads((previous/"method_transfer"/"pycox_protocol.json").read_text())
    known={x["seed"] for x in prior["scenarios"]}
    known.update(x["seed"] for x in (traffic["groups"]+
        traffic["future_horizon_groups"]+traffic["future_decision_groups"]))
    for split in ("train_groups","validation_groups","iid_test_groups","shift_test_groups"):
        known.update(x["seed"] for x in cohorts[split])
    new=[g["seed"] for g in proto["scene_groups"]]
    if len(new)!=len(set(new)) or set(new)&known:
        raise AssertionError("new PR29 group seeds overlap pre-existing development/test groups")


def paired_truth(spec,proto):
    """Evaluate both factual first actions under the same fixed future policy."""
    config=first_action_config(proto)
    a=run_factual(spec,config,0)
    bs=[a]
    if len(a["first_plans"])>=2:
        b=run_factual(spec,config,1)
        if (a["first_state"]!=b["first_state"] or
            a["first_plans"]!=b["first_plans"]):
            raise AssertionError("paired intervention changed initial observable facts or legal candidates")
        bs.append(b)
    if len(a["first_plans"])==0:
        raise AssertionError("no CP-SAT legal proposal")
    for branch in bs:
        if branch["outcomes"]["safety_violations"]:
            raise AssertionError("unsafe factual outcome")
    return bs


def forecast_pair(first,proto,checkpoint,*,arm,horizon):
    geo=first["geometry"]
    business=first["business"]
    if arm=="rule":
        model=HybridSystemWorldModel(geo,business.rules,kind="rule")
    elif arm=="v4":
        model=HybridSystemWorldModel(
            geo,business.rules,kind="v4",checkpoint=checkpoint,experimental=True)
    else:raise ValueError("only frozen rule/V4 methods")
    s=proto["first_action_pair"]["solver"]
    # Each arm / H / candidate restarts the existing deterministic solver.
    # It must never use either actual branch's future trajectory.
    forecasts=[]
    costs=[]
    for idx,plan in enumerate(first["first_plans"][:2]):
        future=BusinessCPSAT(
            geo,business.rules,candidate_limit=s["limit"],
            time_limit_seconds=s["wall_seconds"],
            deterministic_time_limit=s["deterministic_seconds"],
            seed=s["seed"])
        continuation=lambda state, f=future:_plans(
            f,state,candidates_for(state))[0]
        started=time.process_time()
        batch=model.rollout(
            first["first_state"],plan,
            continuation_policy=continuation,
            continuation_policy_name="public_CP_SAT_first_proposal",
            horizon_ticks=horizon,
            particles=proto["prediction"]["particles"],
            sampling_seed=1810+first["first_state"].tick*101+
                          first["result"]["summary"]["sample_count"]*271)
        costs.append(time.process_time()-started)
        forecasts.append({
            "arm":arm,"horizon":horizon,"candidate_index":idx,
            "forecast":batch,
            "expected_completed":float(batch.distributions[-1].expected_completed_samples),
            "expected_waiting":float(batch.distributions[-1].expected_waiting_samples),
            "cpu_seconds":costs[-1],
            "safety_ticks":len(batch.particles)*horizon,
        })
    return forecasts


def _group_predictive_results(arm,h,forecasts,branches,proto):
    out=[]
    for i,f in enumerate(forecasts):
        public=branches[i]["public_observations"]
        truth=public.get(h)
        if truth is not None:
            metrics=score(f["forecast"],truth,h)
        else:
            # An actual run might have completed before h. Never artificially
            # pad with a terminal state or score an invented future.
            metrics=None
        out.append({
            "candidate_index":i,
            "expected_completed":f["expected_completed"],
            "expected_waiting":f["expected_waiting"],
            "model_cpu_seconds":f["cpu_seconds"],
            "full_state_error":metrics,
            "truth_at_h_available":truth is not None,
            "truth_completed_stages_at_h":(
                sum(x.status=="completed" for x in truth.samples)
                if truth is not None else None),
        })
    return out


def _select(forecast_rows):
    if len(forecast_rows)==1:return 0
    values=[x["expected_completed"] for x in forecast_rows]
    if abs(values[0]-values[1])<=1e-9:return 0
    return int(values[1]>values[0])


def evaluate_group(spec,proto,checkpoint):
    bs=paired_truth(spec,proto)
    first=bs[0]
    candidate_count=len(bs)
    make=[b["outcomes"]["end_tick"] for b in bs]
    late=[b["outcomes"]["late_jobs"] for b in bs]
    censored=[b["outcomes"]["censored"] for b in bs]
    complete=not any(censored)
    result={
        "group":spec["id"],"regime":spec["regime"],"seed":spec["seed"],
        "candidate_count_total":len(first["first_plans"]),
        "candidate_count_evaluated":candidate_count,
        "first_observable_tick":first["first_state"].tick,
        "factual":[{
            "candidate":i,
            "first_action":[{"robot":a.robot_id,"sample":a.sample_id}
                            for a in b["first_selected"]],
            "makespan_tick":make[i],
            "late_jobs":late[i],
            "resource_wait_ticks":b["outcomes"]["resource_wait_ticks"],
            "completed_jobs":b["outcomes"]["completed_jobs"],
            "censored":censored[i],
            "event_sha256":hashlib.sha256(json.dumps(
                b["result"]["events"],sort_keys=True).encode()).hexdigest(),
        } for i,b in enumerate(bs)],
        "true_makespan_sensitive":candidate_count==2 and complete and make[0]!=make[1],
        "true_late_sensitive":candidate_count==2 and late[0]!=late[1],
        "oracle_first_action_makespan":min(make) if complete else None,
        "cp_sat_first_regret":make[0]-min(make) if complete else None,
        "cp_sat_first_late_regret":late[0]-min(late) if complete else None,
        "predictions":{},
    }
    for arm in ARMS:
        result["predictions"][arm]={}
        for h in HORIZONS:
            forecasts=forecast_pair(first,proto,checkpoint,arm=arm,horizon=h)
            rows=_group_predictive_results(arm,h,forecasts,bs,proto)
            selected=_select(rows)
            pair_cpu=sum(row["model_cpu_seconds"] for row in rows)
            result["predictions"][arm][str(h)]={
                "candidate_rows":rows,
                "selected_candidate":selected,
                "changed_from_CP_SAT_first":selected!=0,
                "predicted_progress_delta_second_minus_first":(
                    rows[1]["expected_completed"]-rows[0]["expected_completed"]
                    if candidate_count==2 else None),
                "makespan_regret":make[selected]-min(make) if complete else None,
                "late_regret":late[selected]-min(late) if complete else None,
                "selected_makespan_tick":make[selected] if complete else None,
                "selected_late_jobs":late[selected],
                "cpu_seconds_pair":pair_cpu,
                "over_fixed_cpu_budget":pair_cpu>proto["prediction"]["time_budget_seconds_per_pair"],
            }
    return result


def _mean(v):
    return sum(v)/len(v) if v else None


def _by_regime(groups):
    out={}
    for regime in sorted({g["regime"] for g in groups}):
        own=[g for g in groups if g["regime"]==regime]
        pair=[g for g in own if g["candidate_count_evaluated"]==2]
        completed=[g for g in own if g["cp_sat_first_regret"] is not None]
        sensitivity=[g for g in pair if g["true_makespan_sensitive"]]
        arms={}
        for arm in ARMS:
            horizons={}
            for h in HORIZONS:
                values=[g["predictions"][arm][str(h)] for g in own]
                measured=[(g,v) for g,v in zip(own,values) if v["makespan_regret"] is not None]
                metric_groups=[]
                coverage=0
                for v in values:
                    valid=[r["full_state_error"] for r in v["candidate_rows"]
                           if r["full_state_error"] is not None]
                    coverage+=len(valid)
                    if valid:
                        metric_groups.append({
                            key:_mean([x[key] for x in valid])
                            for key in METRICS})
                correct=0
                eligible=0
                predicted_nonzero=0
                for g in sensitivity:
                    p=g["predictions"][arm][str(h)]["predicted_progress_delta_second_minus_first"]
                    if p is None:continue
                    if abs(p)>1e-9:
                        predicted_nonzero+=1
                        eligible+=1
                        # Progress bigger for second should imply smaller true
                        # final makespan; opposite signs signal correct rank.
                        truth=g["factual"][1]["makespan_tick"]-g["factual"][0]["makespan_tick"]
                        correct+=int((p>0)==(truth<0))
                horizons[str(h)]={
                    "groups":len(own),
                    "two_choice_groups":len(pair),
                    "oracle_comparable_complete_groups":len(measured),
                    "mean_limited_candidate_regret_ticks":_mean([
                        v["makespan_regret"] for _,v in measured]),
                    "mean_late_job_regret":_mean([v["late_regret"] for _,v in measured]),
                    "mean_selected_makespan":_mean([v["selected_makespan_tick"] for _,v in measured]),
                    "candidate_flip_groups":sum(v["changed_from_CP_SAT_first"] for v in values),
                    "nonzero_progress_forecast_in_makespan_sensitive_groups":predicted_nonzero,
                    "direction_eligible":eligible,
                    "direction_correct":correct,
                    "mean_cpu_seconds_per_pair":_mean([v["cpu_seconds_pair"] for v in values]),
                    "max_cpu_seconds_per_pair":max(v["cpu_seconds_pair"] for v in values),
                    "budget_overrun_groups":sum(v["over_fixed_cpu_budget"] for v in values),
                    "truth_state_candidate_coverage":coverage,
                    "truth_state_group_coverage":len(metric_groups),
                    "full_state_metrics_scenario_mean":{
                        k:_mean([s[k] for s in metric_groups]) for k in METRICS},
                }
            arms[arm]=horizons
        out[regime]={
            "groups":len(own),
            "first_action_distinct_pair_groups":len(pair),
            "complete_factual_groups":len(completed),
            "true_makespan_sensitive_groups":len(sensitivity),
            "true_late_sensitive_groups":sum(g["true_late_sensitive"] for g in pair),
            "cp_sat_first_mean_limited_regret":_mean([
                g["cp_sat_first_regret"] for g in completed]),
            "cp_sat_first_mean_late_regret":_mean([
                g["cp_sat_first_late_regret"] for g in completed]),
            "arms":arms,
        }
    return out


def evaluate(checkpoint,*,protocol=PROTOCOL,cpu_budget=480):
    config=json.loads(Path(protocol).read_text())
    if config["schema"]!="RHWM_PR29_FROZEN_HORIZON_SCENE_GENERALIZATION_20261112_V1":
        raise ValueError("wrong frozen PR29 protocol")
    if config["prediction"]["horizons"]!=list(HORIZONS) or config["baselines"]!=list(ARMS):
        raise AssertionError("horizons or model arm set changed")
    if len(config["scene_groups"])!=32 or sorted(set(
            x["regime"] for x in config["scene_groups"]))!=sorted(
            ("light","medium","heavy_new_seed","heavy_service_shift")):
        raise AssertionError("precommitted 4-regime groups modified")
    if any(sum(g["regime"]==name for g in config["scene_groups"])!=8
           for name in ("light","medium","heavy_new_seed","heavy_service_shift")):
        raise AssertionError("expected eight groups per regime")
    _scenario_seed_set(config)
    if file_sha(checkpoint)!=config["source_model"]["checkpoint_sha256"]:
        raise ValueError("frozen V4 checkpoint SHA256 mismatch")
    from torch import set_num_threads
    set_num_threads(1)
    begun=time.process_time()
    groups=[]
    for spec in config["scene_groups"]:
        if time.process_time()-begun>cpu_budget:
            raise TimeoutError("CPU budget exhausted before next scene")
        groups.append(evaluate_group(spec,config,checkpoint))
    elapsed=time.process_time()-begun
    if elapsed>cpu_budget:raise TimeoutError("precommitted CPU cap exceeded")
    return {
        "schema":config["schema"],
        "protocol_sha256":file_sha(protocol),
        "checkpoint_sha256":file_sha(checkpoint),
        "groups":groups,
        "summary":{
            "total_independent_groups":len(groups),
            "distinct_pair_groups":sum(g["candidate_count_evaluated"]==2 for g in groups),
            "total_factual_runs":sum(g["candidate_count_evaluated"] for g in groups),
            "by_regime":_by_regime(groups),
            "cpu_process_seconds":elapsed,
            "gpu":0,"new_training":0,
            "scope":"One initial action decision followed by identical online CP-SAT; offline paired counterfactual regret, NOT a whole-episode online world-model policy",
            "limitations":"synthetic workloads; only first two CP-SAT proposals; V4 not fine-tuned; no true robot telemetry; known scheduled release cases excluded",
        },
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--cpu-budget",type=int,default=480)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError("pr29 results are write-once")
    result=evaluate(args.checkpoint,cpu_budget=args.cpu_budget)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    # Compact CI summary; detailed scenario-level evidence always in result.json
    small={}
    for regime,v in result["summary"]["by_regime"].items():
        small[regime]={
            "groups":v["groups"],
            "distinct_pair_groups":v["first_action_distinct_pair_groups"],
            "makespan_sensitive":v["true_makespan_sensitive_groups"],
            "cp_regret":v["cp_sat_first_mean_limited_regret"],
            "horizons":{
                name:{h:{
                    "regret":v["arms"][name][h]["mean_limited_candidate_regret_ticks"],
                    "flips":v["arms"][name][h]["candidate_flip_groups"],
                    "cpu":v["arms"][name][h]["mean_cpu_seconds_per_pair"],
                    "budget_overrun":v["arms"][name][h]["budget_overrun_groups"],
                    "state_error":v["arms"][name][h]["full_state_metrics_scenario_mean"]["sample_status_error"],
                    "direction_correct":v["arms"][name][h]["direction_correct"],
                    "direction_eligible":v["arms"][name][h]["direction_eligible"],
                } for h in ("5","10","20")}
                for name in ARMS},
        }
    print("PR29_HORIZON_SCENARIO_GENERALIZATION",json.dumps({
        "groups":result["summary"]["total_independent_groups"],
        "paired":result["summary"]["distinct_pair_groups"],
        "cpu":result["summary"]["cpu_process_seconds"],
        "regimes":small},ensure_ascii=False))


if __name__=="__main__":
    main()
