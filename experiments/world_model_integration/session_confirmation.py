"""Frozen six-arm synthetic decision evaluation using actual simulator execution.

No results used for model training/threshold choice. F is research-only because
its offline v2 checkpoint has not passed online deployment verification.
Common CP-SAT proposal generation, particles, movement and external schedules.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

from business_transport.observation import BusinessObserver
from business_transport.policy import WorkflowPolicy
from business_transport.session_dynamics import SessionRuleDynamics
from business_transport.simulation import BusinessSimulator
from business_transport.dispatch import BusinessCPSAT
from experiments.business_constraints.run import (make_model,summarize_result,
                                                   replay_branch,rich_rows)
from experiments.service_value.counterfactual import load_pr9_model
from experiments.world_model_integration.build_dataset import case_for,rows_from_factual
from lab_transport.decision_value import candidates_for,nearest_batch
from lab_transport.dynamics_model import Geometry
from business_transport.workflow_transition import check_business_transition
from experiments.world_model_integration.prediction_metrics import (_joint,_events,_device_event,_external_release)

CONFIG=Path(__file__).with_name("session_confirmation_config.json")
ARMS=("A_rule","B_pets_original","C_pets_active","D_pets_guarded_future",
      "E_structured","F_session_experimental")

def _policy(arm,business,geometry,network,info,checkpoint,cfg):
    modeltype=("rule" if arm=="A_rule" else "pets" if arm.startswith(("B_","C_","D_"))
                else "structured" if arm=="E_structured" else "session")
    if modeltype=="session":
        model=SessionRuleDynamics(geometry,business.rules,checkpoint,experimental=True,
                                   seed=cfg["planner_seed"])
    else:
        model=make_model(modeltype,geometry,network,info["threshold"],cfg["planner_seed"])
    return WorkflowPolicy("cpsat_"+modeltype,geometry,business.rules,
        BusinessObserver(),model,seed=cfg["planner_seed"],horizon=cfg["horizon"],
        particles=cfg["particles"],budget=cfg["budget"],
        solver_seconds=cfg["solver_seconds"],max_prediction_seconds=cfg["max_prediction_seconds"],
        risk_weight=cfg["risk_weight"],model_aware=arm not in ("A_rule","B_pets_original"),
        future_pets_starts=(arm=="D_pets_guarded_future"))


def _forecast_quality(policy, observed_rows, business, geometry, seed, stride):
    """Conditional accuracy on factual t+h labels, including external-event windows."""
    report={}
    for horizon in (1,3,5,10,20):
        stats=Counter()
        picks=list(range(0,max(0,len(observed_rows)-horizon+1),stride))[:4]
        stats["attempts"]=len(picks)
        for index in picks:
            slice_=observed_rows[index:index+horizon]
            initial=slice_[0].observation
            actual=slice_[-1].next_observation
            ext=any(_device_event(row.observation,row.next_observation) or
                    _external_release(row.observation,row.next_observation) for row in slice_)
            stats["external_windows" if ext else "normal_windows"]+=1
            state=initial
            try:
                policy.model.reset_seed(seed+index*1009+horizon)
                for step in range(horizon):
                    actions=(slice_[0].assignments if step==0 else
                             nearest_batch(state,candidates_for(state),geometry))
                    predicted=policy.model.predict_next(state,actions)
                    check_business_transition(state,predicted,geometry,business.rules,actions)
                    state=predicted
                values=_joint(state,actual)
                stats["valid"]+=1
                stats["joint_exact"]+=values["joint_correct"]
                stats["robot_position_errors"]+=values["robot_position_errors"]
                stats["resource_errors"]+=values["resource_errors"]
                initial_ids={s.id for s in initial.samples}
                pred_released={s.id for s in state.samples}-initial_ids
                actual_released={s.id for s in actual.samples}-initial_ids
                stats["successor_release_errors"]+=len(pred_released ^ actual_released)
                if horizon==1:
                    truth=_events(initial,actual)
                    predicted_events=_events(initial,state)
                    stats["event_tp"]+=len(truth & predicted_events)
                    stats["event_fp"]+=len(predicted_events-truth)
                    stats["event_fn"]+=len(truth-predicted_events)
            except (ValueError,RuntimeError,TypeError,KeyError,OverflowError) as exc:
                stats["invalid"]+=1
        report[str(horizon)]={**dict(stats),
            "legal_coverage":stats["valid"]/stats["attempts"] if stats["attempts"] else None,
            "joint_exact_accuracy":stats["joint_exact"]/stats["attempts"] if stats["attempts"] else None,
            "note":"legal coverage and factual joint accuracy are different metrics"}
    return report


def evaluate(checkpoint,output):
    cfgbytes=CONFIG.read_bytes();cfg=json.loads(cfgbytes)
    if (cfg.get("status")!="frozen_independent_seed_and_task_holdout" or
        cfg.get("no_tuning_after_results") is not True or cfg.get("risk_weight")!=0):
        raise ValueError("confirmation configuration not frozen")
    torch.set_num_threads(1)
    network,info=load_pr9_model()
    results=[]
    for regime in cfg["regimes"]:
        for seed in cfg["seeds"]:
            spec={"seed":seed,"layout":regime["layout"],"outage":regime["outage"]}
            business,geo=case_for(spec,cfg)
            arms={};fact=None;fact_rows=None
            for arm in ARMS:
                policy=_policy(arm,business,geo,network,info,checkpoint,cfg)
                rows=[]
                started=time.perf_counter()
                result=BusinessSimulator(business,"world",seed,collect_metrics=True).run(
                    dispatch_policy=policy,on_observation=policy.observer.observe,
                    on_transition=rows.append if arm=="A_rule" else None)
                measured=summarize_result(result,business,policy)
                arms[arm]={"jobs_completed":measured["jobs_completed"],
                    "status":measured["status"],"restricted_mean_flow":measured["restricted_mean_flow"],
                    "mean_stage_wait_per_job":measured["mean_stage_wait_per_job"],
                    "resource_wait_ticks":measured["resource"]["capacity_wait_ticks"],
                    "decision_count":measured["decision_count"],
                    "decision_wall_mean_ms":measured["decision_wall_mean_ms"],
                    "decision_wall_p95_ms":measured["decision_wall_p95_ms"],
                    "model_fallbacks":measured["model_fallbacks"],
                    "capacity_violations":len(measured["resource"]["violations"]),
                    "discrete_safety_violations":measured["discrete_safety_violations"],
                    "lifecycle_violations":len(measured["lifecycle_violations"]),
                    "prediction_calls":policy.prediction_calls,
                    "actual_wall_seconds":time.perf_counter()-started}
                if arm=="A_rule":
                    fact,fact_rows=result,rows
            states=rich_rows(fact_rows)
            prediction_quality={}
            for arm in ARMS:
                diagnostic_policy=_policy(arm,business,geo,network,info,checkpoint,cfg)
                prediction_quality[arm]=_forecast_quality(diagnostic_policy,states,business,geo,seed,
                                                          cfg["candidate_sample_stride"])
            simultaneous=sum(sum(robot.service_elapsed is not None for robot in row.observation.robots)>=2
                             for row in states)
            observed_outage=sum(any(not getattr(s,"device_available",s.available)
                                    for s in row.observation.stations) for row in states)
            competition=sum(any(getattr(s,"capacity_blocked",False) for s in row.observation.stations)
                            for row in states)
            eligible=[row for row in states
                      if len(candidates_for(row.observation))>=2]
            indices=list(range(0,len(eligible),cfg["candidate_sample_stride"]))
            indices=indices[:cfg["per_case_max_fixed_states"]]
            fixed=[]
            for idx in indices:
                obs=eligible[idx].observation
                plans=BusinessCPSAT(geo,business.rules,candidate_limit=cfg["budget"],
                    time_limit_seconds=cfg["solver_seconds"],seed=cfg["planner_seed"]).proposals(
                        obs,candidates_for(obs))
                if len(plans)<2:continue
                actual=[replay_branch(business,seed,fact,fact_rows,obs.tick,plan) for plan in plans]
                choices={}
                for arm in ARMS:
                    policy=_policy(arm,business,geo,network,info,checkpoint,cfg)
                    rec=policy.evaluate(obs,plans)
                    chosen=min(range(len(rec)),key=lambda i:(rec[i]["score"],i))
                    results_flow=[branch["flow"] for branch in actual]
                    best=min((x for x in results_flow if x is not None),default=None)
                    selected_flow=results_flow[chosen]
                    choices[arm]={"choice":chosen,"scores":[r["score"] for r in rec],
                                  "rank":sorted(range(len(rec)),key=lambda i:(rec[i]["score"],i)),
                                  "decision_loss":(selected_flow-best) if
                                     selected_flow is not None and best is not None else None,
                                  "prediction_calls":policy.prediction_calls}
                fixed.append({"tick":obs.tick,
                    "plans":[[asdict(a) for a in plan] for plan in plans],
                    "counterfactual_factual_simulator":actual,"choices":choices})
            # Current facts contain outages only once observed; scenario's
            # schedule is never passed to a predictor.
            results.append({"regime":regime["name"],"seed":seed,
                            "arms":arms,"fixed_state_decisions":fixed,
                            "prediction_quality":prediction_quality,
                            "factual_scenario_audit":{
                                "simultaneous_service_steps":simultaneous,
                                "observed_device_outage_steps":observed_outage,
                                "capacity_blocked_observations":competition},
                            "eligible_fact_states":len(eligible)})
    summary={}
    for arm in ARMS:
        rows=[case["arms"][arm] for case in results]
        summary[arm]={
            "jobs_completed":sum(r["jobs_completed"] for r in rows),
            "restricted_mean_flow_mean":statistics.mean(r["restricted_mean_flow"] for r in rows),
            "decision_wall_mean_ms":statistics.mean(r["decision_wall_mean_ms"] for r in rows),
            "capacity_violations":sum(r["capacity_violations"] for r in rows),
            "motion_violations":sum(r["discrete_safety_violations"] for r in rows),
            "lifecycle_violations":sum(r["lifecycle_violations"] for r in rows),
            "fallbacks":sum(r["model_fallbacks"] for r in rows)}
    fixed=[d for c in results for d in c["fixed_state_decisions"]]
    compare={"fixed_states":len(fixed)}
    for arm in ARMS[1:]:
        compare[arm]={"changed_vs_A":sum(d["choices"][arm]["choice"]!=d["choices"]["A_rule"]["choice"] for d in fixed),
                      "different_ranking_vs_A":sum(d["choices"][arm]["rank"]!=d["choices"]["A_rule"]["rank"] for d in fixed)}
    out={"status":"fixed_synthetic_confirmation_no_hindsight",
         "config_sha256":hashlib.sha256(cfgbytes).hexdigest(),
         "F_online_status":"research_only_unverified",
         "arms":ARMS,"cases":results,"summary":summary,"same_state_comparison":compare,
         "limitations":["small synthetic seed set","no real-world device health forecast",
                        "F is NOT approved for production deployment"]}
    path=Path(output)
    if path.exists():raise FileExistsError("confirmation result is write-once")
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(out,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({"summary":summary,"same_state_comparison":compare},
                     ensure_ascii=False))
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    evaluate(a.checkpoint,a.output)


if __name__=="__main__":
    main()
