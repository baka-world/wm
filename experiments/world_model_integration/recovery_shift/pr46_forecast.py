"""PR46 phase 1: independent recovery-mechanism-shift H5/H20 state evaluation.

This module does NOT read any online cohort outcomes and does not adjust
frozen V4 / KM models. All factual future states are offline labels.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import hashlib
import json
import time

import torch

from business_transport.observation import BusinessObserver
from business_transport.policy import BusinessPolicy
from business_transport.simulation import BusinessSimulator, resource_audit
from business_transport.world_model import HybridSystemWorldModel
from experiments.world_model_integration.recovery_integration.pr45_recovery_world import (
    PublicDownClock, ObservedRecoveryWorld, predict_metrics,
)
from experiments.world_model_integration.recovery_shift.pr46_shared import (
    PROTOCOL, FORECAST_ARMS, MECHANISMS, build_business, frozen_setup,
)
from experiments.world_model_integration.system_world_model.evaluate_horizons import file_sha
from lab_transport.decision_value import candidates_for, nearest_batch
from lab_transport.multisite_experiment import discrete_safety_violations


METRICS = (
    "target_station_device_brier", "sample_status_error", "robot_task_error",
    "station_busy_brier", "resource_occupancy_mae", "completed_count_mae",
    "waiting_count_mae",
)


def factual_episode(spec, proto):
    business,geo=build_business(spec,proto)
    observer=BusinessObserver()
    clock=PublicDownClock()
    policy=BusinessPolicy("nearest",geo,business.rules,observer)
    states={}
    ages={}
    transitions=[]

    def observe(raw):
        state=observer.observe(raw)
        states[state.tick]=state
        ages[state.tick]=clock.observe(state)

    factual=BusinessSimulator(business,"world",spec["seed"],collect_metrics=True).run(
        dispatch_policy=policy,on_observation=observe,
        on_transition=transitions.append)
    if transitions and transitions[-1].next_boundary=="end_of_run":
        last=transitions[-1].next_observation
        observed=observer.observe(last)
        states[observed.tick]=observed
        ages[observed.tick]=clock.observe(observed)
    audit=resource_audit(factual,business.rules)
    if audit["violations"] or discrete_safety_violations(factual):
        raise AssertionError("factual simulator safety failure")
    actions={t.observation.tick:t.assignments for t in transitions}
    if len(actions)!=len(transitions):
        raise AssertionError("duplicate true tick")
    return business,geo,states,ages,actions,factual


def one(spec,proto,checkpoint,km):
    business,geo,states,ages,actions,factual=factual_episode(spec,proto)
    orig=HybridSystemWorldModel(
        geo,business.rules,kind="v4",checkpoint=checkpoint,experimental=True)
    adapted=ObservedRecoveryWorld(geo,business.rules,checkpoint,km)
    results=[]
    for tick in proto["forecast"]["anchors"]:
        if tick not in actions:continue
        initial=states[tick]
        station=next(s for s in initial.stations if s.id==spec["station_id"])
        age=ages[tick].get(spec["station_id"])
        if station.device_available or age is None:
            continue
        horizon_list=[h for h in proto["forecast"]["horizons"] if tick+h in states]
        if not horizon_list:continue
        horizon=max(horizon_list)
        first_action=actions[tick]
        if any(a not in candidates_for(initial) for a in first_action):
            raise AssertionError("factual first action not publicly legal")
        continuation=lambda state:nearest_batch(state,candidates_for(state),geo)
        master=1810+spec["seed"]*101+tick
        begun=time.process_time()
        old=orig.rollout(
            initial,first_action,continuation_policy=continuation,
            continuation_policy_name="nearest_predicted_public",
            horizon_ticks=horizon,particles=proto["forecast"]["particles"],
            sampling_seed=master)
        old_cpu=time.process_time()-begun
        begun=time.process_time()
        new,draws=adapted.rollout(
            initial,first_action,down_ages=ages[tick],
            continuation_policy=continuation,
            horizon_ticks=horizon,particles=proto["forecast"]["particles"],
            sampling_seed=master)
        new_cpu=time.process_time()-begun
        for h in horizon_list:
            truth=states[tick+h]
            results.append({
                "anchor_tick":tick,"horizon":h,"public_down_age":age,
                "facts_available_only_at_scoring_time":True,
                "methods":{
                    FORECAST_ARMS[0]:predict_metrics(old,truth,spec["station_id"],h),
                    FORECAST_ARMS[1]:predict_metrics(new,truth,spec["station_id"],h),
                },
                "recovery_draws_until_h":sum(d["tick"]<=tick+h for d in draws),
                "sampled_recoveries_until_h":sum(d["tick"]<=tick+h and d["sampled_recovery"]
                                                for d in draws),
                "cpu_seconds_full_prediction":{
                    FORECAST_ARMS[0]:old_cpu,FORECAST_ARMS[1]:new_cpu}
            })
    if not results:
        raise AssertionError("no publicly observed-down forecast anchor")
    return {
        "id":spec["id"],"seed":spec["seed"],"station_id":spec["station_id"],
        "mechanism":spec["mechanism"],
        "factual_status":factual["summary"]["status"],
        "factual_ticks":factual["summary"]["ticks"],
        "factual_sha256":hashlib.sha256(json.dumps(
            factual["events"],sort_keys=True).encode()).hexdigest(),
        "rows":results,
    }


def aggregate(groups,horizons):
    out={}
    for mechanism in MECHANISMS:
        own=[x for x in groups if x["mechanism"]==mechanism]
        by_h={}
        for h in horizons:
            grouped=[]
            for group in own:
                relevant=[row for row in group["rows"] if row["horizon"]==h]
                if not relevant:continue
                grouped.append({
                    arm:{metric:sum(x["methods"][arm][metric] for x in relevant)/len(relevant)
                         for metric in METRICS}
                    for arm in FORECAST_ARMS
                })
            averages={
                arm:{m:(sum(g[arm][m] for g in grouped)/len(grouped) if grouped else None)
                     for m in METRICS}
                for arm in FORECAST_ARMS
            }
            by_h[str(h)]={
                "scenario_groups":len(grouped),"metrics":averages,
                "recovery_minus_original":{
                    m:(averages[FORECAST_ARMS[1]][m]-averages[FORECAST_ARMS[0]][m]
                       if grouped else None) for m in METRICS}
            }
        out[mechanism]={"groups":len(own),"by_horizon":by_h}
    return out


def run(checkpoint,km_file,output):
    p,km=frozen_setup(checkpoint,km_file)
    if output.exists():raise FileExistsError("PR46 forecast evidence is write-once")
    torch.set_num_threads(1)
    begun=time.process_time()
    groups=[]
    for spec in p["cohort"]["forecast"]:
        if time.process_time()-begun>=p["resources"]["process_cpu_budget_seconds"]:
            raise TimeoutError("forecast CPU process budget exceeded")
        groups.append(one(spec,p,checkpoint,km))
    elapsed=time.process_time()-begun
    if elapsed>=p["resources"]["process_cpu_budget_seconds"]:
        raise TimeoutError("forecast CPU budget exceeded after final group")
    result={
        "schema":"pr46-recovery-shift-state-forecast-v1",
        "cohort":"forecast_only_not_online","protocol_sha256":file_sha(PROTOCOL),
        "source_pr44_train_artifact_sha256":file_sha(km_file),
        "frozen_v4_checkpoint_sha256":file_sha(checkpoint),
        "gpu":0,"new_training":0,"world_model_default_unchanged":True,
        "groups":groups,"summary":{"groups":len(groups),
                "by_mechanism":aggregate(groups,p["forecast"]["horizons"]),
                "cpu_process_seconds":elapsed,
                "scope":"known-down recovery, synthetic heldout seeds; extended outage outside training duration support"},
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print("PR46_FORECAST_RESULT",json.dumps(result["summary"],ensure_ascii=False))
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True,type=Path)
    p.add_argument("--pr44-result",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args()
    run(args.checkpoint,args.pr44_result,args.out)


if __name__=="__main__":
    main()
