"""Prediction-gated V4 closed-loop decision validation (experimental only).

If independent V4 Brier fails frozen B0/multi-trajectory gate: exactly one
legal forward step; do not tune, choose different trajectories, or rerun.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import time
import torch

from business_transport.observation import BusinessObserver
from business_transport.policy import WorkflowPolicy
from business_transport.session_dynamics import SessionRuleDynamics
from business_transport.workflow_transition import check_business_transition
from business_transport.simulation import BusinessSimulator
from business_transport.dispatch import BusinessCPSAT
from experiments.business_constraints.run import summarize_result,rich_rows,replay_branch
from experiments.world_model_integration.build_dataset import case_for
from experiments.world_model_integration.build_dataset_v4 import CONFIG
from experiments.world_model_integration.train_v4 import sha,load_v4
from lab_transport.decision_value import candidates_for,nearest_batch
from lab_transport.torch_structured_adapter import ProgressAwareRule

# This budget is fixed before the independent event test, not selected afterward.
BUDGET={"seed":1810,"horizon":5,"particles":3,"candidate_budget":4,
        "solver_seconds":0.5,"risk_weight":0.,
        "case_count_if_pass":3,"states_per_case":2,"candidate_stride":13}


def policy_for(arm,business,geo,v3,v4):
    if arm=="Rule":model=ProgressAwareRule(geo)
    elif arm=="V3-C":model=SessionRuleDynamics(geo,business.rules,v3,
           experimental=True,seed=BUDGET["seed"],feature_version="v3")
    elif arm=="V4":model=SessionRuleDynamics(geo,business.rules,v4,
           experimental=True,seed=BUDGET["seed"],feature_version="v4")
    else:raise ValueError("unexpected frozen comparison arm")
    mode="cpsat_rule" if arm=="Rule" else "cpsat_session"
    return WorkflowPolicy(mode,geo,business.rules,BusinessObserver(),model,
        seed=BUDGET["seed"],horizon=BUDGET["horizon"],
        particles=BUDGET["particles"],budget=BUDGET["candidate_budget"],
        solver_seconds=BUDGET["solver_seconds"],model_aware=True,risk_weight=0)


def run(independent,v3_path,v4_path,output):
    if Path(output).exists():raise FileExistsError("decision report must be immutable")
    hold=json.loads(Path(independent).read_text())
    cfg=json.loads(CONFIG.read_text())
    if hold["evaluation_scope"]!="independent_in" or hold["one_shot"] is not True or (
        hold["source_config_sha256"]!=sha(CONFIG) or
        hold["v4_checkpoint_sha256"]!=sha(v4_path) or
        hold["v3_frozen_checkpoint_sha256"]!=sha(v3_path)):
        raise ValueError("decision gate requires actual one-shot matched models")
    valid=bool(hold["gate"]["eligible_for_limited_dispatch"])
    result={"protocol":cfg["protocol"],"gate_pass":valid,
            "config":BUDGET,"models":["Rule","V3-C","V4"],
            "model_online_verified":False,
            "cases":[],"only_frozen_precommitted_test":True}
    if not valid:
        from experiments.business_constraints.tests.test_business import fixture
        from lab_transport.dynamics_model import Geometry
        base,cfg0,business=fixture(1)
        geo=Geometry(base.grid)
        rows=[]
        BusinessSimulator(business,"world",1).run(
            on_transition=rows.append,
            dispatch_policy=lambda o,c:nearest_batch(o,c,geo))
        obs=rich_rows(rows)[0].observation
        model=policy_for("V4",business,geo,v3_path,v4_path).model
        actions=nearest_batch(obs,candidates_for(obs),geo)
        predicted=model.predict_next(obs,actions)
        check_business_transition(obs,predicted,geo,business.rules,actions)
        result["smoke"]={"valid":True,"forecast_steps":1}
        result["stop_reason"]="V4 in-range Brier or multi-trajectory criterion did not beat frozen B0"
    else:
        torch.set_num_threads(1)
        specs=cfg["groups"]["independent_in"][:BUDGET["case_count_if_pass"]]
        for spec in specs:
            business,geo=case_for(spec,cfg)
            case={"seed":spec["seed"],"layout":spec["layout"],"arms":{},"fixed_states":[]}
            reference=None;raw=None
            for arm in result["models"]:
                policy=policy_for(arm,business,geo,v3_path,v4_path)
                events=[];start=time.perf_counter()
                factual=BusinessSimulator(business,"world",spec["seed"],collect_metrics=True).run(
                    dispatch_policy=policy,on_observation=policy.observer.observe,
                    on_transition=events.append if arm=="Rule" else None)
                outcome=summarize_result(factual,business,policy)
                case["arms"][arm]={
                    "completed":outcome["jobs_completed"],
                    "restricted_mean_flow":outcome["restricted_mean_flow"],
                    "mean_wait":outcome["mean_stage_wait_per_job"],
                    "resource_wait_ticks":outcome["resource"]["capacity_wait_ticks"],
                    "decision_mean_ms":outcome["decision_wall_mean_ms"],
                    "decision_p95_ms":outcome["decision_wall_p95_ms"],
                    "model_calls":policy.prediction_calls,
                    "fallbacks":policy.fallbacks,
                    "resource_violations":len(outcome["resource"]["violations"]),
                    "motion_violations":outcome["discrete_safety_violations"],
                    "lifecycle_violations":len(outcome["lifecycle_violations"]),
                    "elapsed_wall_seconds":time.perf_counter()-start}
                if arm=="Rule":reference,raw=factual,events
            states=[r.observation for r in rich_rows(raw) if len(candidates_for(r.observation))>=2]
            selected=states[::BUDGET["candidate_stride"]][:BUDGET["states_per_case"]]
            for obs in selected:
                plans=BusinessCPSAT(geo,business.rules,
                    candidate_limit=BUDGET["candidate_budget"],
                    time_limit_seconds=BUDGET["solver_seconds"],seed=BUDGET["seed"]).proposals(
                    obs,candidates_for(obs))
                if len(plans)<2:continue
                truth=[replay_branch(business,spec["seed"],reference,raw,obs.tick,p) for p in plans]
                actual=[entry["flow"] for entry in truth]
                best=min((f for f in actual if f is not None),default=None)
                scores={}
                for arm in result["models"]:
                    pol=policy_for(arm,business,geo,v3_path,v4_path)
                    predicted=pol.evaluate(obs,plans)
                    ranked=sorted(range(len(plans)),key=lambda i:(predicted[i]["score"],i))
                    chosen=ranked[0]
                    loss=(actual[chosen]-best if actual[chosen] is not None and best is not None else None)
                    scores[arm]={"rank":ranked,"chosen":chosen,
                        "score":[r["score"] for r in predicted],
                        "true_regret":loss}
                case["fixed_states"].append({"tick":obs.tick,
                    "candidates":[[asdict(a) for a in p] for p in plans],
                    "factual_counterfactual_flow":actual,"predictions":scores})
            result["cases"].append(case)
        fixed=[item for case in result["cases"] for item in case["fixed_states"]]
        result["comparison"]={
          "fixed_candidate_states":len(fixed),
          "v4_action_changed_vs_rule":sum(item["predictions"]["V4"]["chosen"]!=
              item["predictions"]["Rule"]["chosen"] for item in fixed),
          "v4_rank_changed_vs_rule":sum(item["predictions"]["V4"]["rank"]!=
              item["predictions"]["Rule"]["rank"] for item in fixed),
          "v4_cost_changed_same_choice":sum(
              item["predictions"]["V4"]["score"]!=item["predictions"]["Rule"]["score"]
              and item["predictions"]["V4"]["chosen"]==item["predictions"]["Rule"]["chosen"]
              for item in fixed)}
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"gate":valid,"stop":result.get("stop_reason"),
          "cases":len(result["cases"]),
          "comparison":result.get("comparison")},ensure_ascii=False))
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--inrange",type=Path,required=True)
    p.add_argument("--v3-frozen",type=Path,required=True)
    p.add_argument("--v4",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    run(a.inrange,a.v3_frozen,a.v4,a.output)


if __name__=="__main__":
    main()
