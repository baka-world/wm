"""Predeclared capacity stress mechanism regression (NOT independent confirmation).

The frozen six-arm holdout was already evaluated. This additional stress uses
the existing legacy resource-capacity fixture; it is a mechanism test only.
No estimator weight/threshold tuning is allowed from these results.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
from business_transport.observation import BusinessObserver
from business_transport.simulation import BusinessSimulator
from experiments.business_constraints.tests.test_business import fixture
from experiments.business_constraints.run import summarize_result
from experiments.service_value.counterfactual import load_pr9_model
from experiments.world_model_integration.session_confirmation import ARMS,_policy
from experiments.world_model_integration.session_confirmation import CONFIG
from lab_transport.decision_value import nearest_batch
from lab_transport.dynamics_model import Geometry


def run(checkpoint,output):
    if Path(output).exists():raise FileExistsError("refuse stress overwrite")
    _,_,business=fixture(1)
    geometry=Geometry(business.scenario.grid)
    cfg=json.loads(CONFIG.read_text(encoding="utf-8"))
    network,info=load_pr9_model()
    baseline=BusinessSimulator(business,"world",1,collect_metrics=True).run(
        dispatch_policy=lambda obs,c:nearest_batch(obs,c,geometry))
    nearest_wait=baseline["summary"].get("resource_wait_ticks",0)
    if nearest_wait<=0:
        raise ValueError("capacity stress did not produce actual observed wait")
    arms={}
    for arm in ARMS:
        policy=_policy(arm,business,geometry,network,info,checkpoint,cfg)
        actual=BusinessSimulator(business,"world",1,collect_metrics=True).run(
            dispatch_policy=policy,on_observation=policy.observer.observe)
        metrics=summarize_result(actual,business,policy)
        sources=Counter()
        reasons=Counter()
        for event in policy.log:
            for plan in event["plans"]:
                for component in plan.get("components",()):
                    for work in component.get("service_estimates",()):
                        sources[work["source"]]+=1
                        if work.get("fallback_reason"):
                            reasons[work["fallback_reason"]]+=1
        arms[arm]={
          "jobs_completed":metrics["jobs_completed"],
          "restricted_mean_flow":metrics["restricted_mean_flow"],
          "resource_wait_ticks":metrics["resource"]["capacity_wait_ticks"],
          "resource_violations":len(metrics["resource"]["violations"]),
          "motion_violations":metrics["discrete_safety_violations"],
          "lifecycle_violations":len(metrics["lifecycle_violations"]),
          "mean_decision_ms":metrics["decision_wall_mean_ms"],
          "p95_decision_ms":metrics["decision_wall_p95_ms"],
          "model_fallbacks":policy.fallbacks,
          "scoring_sources":dict(sources),
          "scoring_fallback_reasons":dict(reasons),
        }
    out={"scope":"POST_HOC_MECHANISM_ONLY_NOT_CONFIRMATION",
         "scenario":"frozen_existing_business_capacity_fixture",
         "seed":1,"nearest_fact_capacity_wait_ticks":nearest_wait,
         "arms":arms,"new_model_online_verified":False,
         "note":"comparison with historical setup, no threshold or weight selection"}
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(out,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(out,ensure_ascii=False))
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    run(a.checkpoint,a.output)


if __name__=="__main__":
    main()
