"""PR28: literature-grounded dynamic workload action-identifiability experiment.

No new predictor, scheduler, synthetic stochastic mechanism, or simulator
rewrite. Reuses the original BusinessCPSAT, the original BusinessSimulator,
and the frozen V4 via HybridSystemWorldModel.

Research precedent:
- Reijnen et al. (2026), online FJSP arrivals and cross-policy benchmarking,
  DOI 10.1007/s10472-026-10007-3, upstream MIT repo.
- Flores-Gómez & Dauzère-Pérès (2026), deadline service levels under
  processing-time uncertainty, DOI 10.1007/s10479-026-07197-8.

At tick=0 execute *either* first or second legal CP-SAT proposal; thereafter
both factual branches replan with identical deterministic CP-SAT policy.
Common sample/phase/station keyed random draws make interventions paired.
Counterfactual actual outcomes are an OFFLINE evaluation only. No result is
used by a live policy or by the V4 forecast. This is a first-action sensitivity
diagnostic, NOT a causal inference study with an independently controlled
enterprise environment, nor an unknown-job-arrival experiment.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import argparse
import hashlib
import json
import time

from business_transport.dispatch import BusinessCPSAT
from business_transport.observation import BusinessObserver
from business_transport.scenario import compile_scenario
from business_transport.simulation import BusinessSimulator, resource_audit
from business_transport.world_model import HybridSystemWorldModel
from experiments.business_constraints.run import lifecycle_audit
from experiments.business_constraints.tests.test_business import fixture
from experiments.world_model_integration.system_world_model.evaluate_horizons import file_sha
from lab_transport.decision_value import candidates_for
from lab_transport.dynamics_model import Geometry
from lab_transport.multisite_experiment import discrete_safety_violations

ROOT=Path(__file__).resolve().parent
PROTOCOL=ROOT/"pr28_protocol.json"
FORECAST_H=(5,10,20)


def build_business(spec, *, max_ticks=240):
    """Predeclared small synthetic workload, no new environment dynamics.

    Future job releases are scheduled and may be visible in public metadata.
    This does NOT simulate surprise arrivals concealed from the dispatcher.
    """
    base,cfg,_=fixture(capacity=spec["capacity"])
    if len(spec["release_ticks"])!=spec["job_count"]:
        raise ValueError("invalid frozen releases")
    stations={sid:replace(st,
                          service_ticks=st.service_ticks+spec["service_add"],
                          prior_ticks=st.prior_ticks+spec["service_add"],
                          jitter_ticks=spec["jitter"])
              for sid,st in base.stations.items()}
    base=replace(base,name="pr28-"+spec["id"],
                 stations=stations,robots=base.robots[:spec["robot_count"]],
                 duration_jitter=spec["jitter"],max_ticks=max_ticks)
    jobs=[
        {"id":f"J{i+1}","workflow":"w0" if i%2==0 else "w1",
         "release_tick":tick,
         "deadline_tick":tick+spec["deadline_slack"]}
        for i,tick in enumerate(spec["release_ticks"])
    ]
    cfg={**cfg,"jobs":jobs,"max_ticks":max_ticks,"name":"pr28-"+spec["id"]}
    return compile_scenario(base,cfg),Geometry(base.grid)


def _plans(dispatcher,public_state,legal):
    raw=dispatcher.proposals(public_state,legal)
    plans=[]
    for batch in raw:
        batch=tuple(batch)
        if batch not in plans:plans.append(batch)
    if not plans:plans=[()]
    allowed=set(legal)
    if any(any(a not in allowed for a in p) for p in plans):
        raise AssertionError("CP-SAT proposed a nonlegal first action")
    return tuple(plans)


class FixedFirstThenCP:
    """Observes only current public state, then uses native CP-SAT."""

    def __init__(self,geometry,rules,config,choice):
        self.geometry=geometry
        self.observer=BusinessObserver()
        self.solver=BusinessCPSAT(
            geometry,rules,
            candidate_limit=config["bounded_counterfactual"]["candidates"]["limit"],
            time_limit_seconds=config["bounded_counterfactual"]["candidates"]["wall_seconds"],
            deterministic_time_limit=config["bounded_counterfactual"]["candidates"]["deterministic_seconds"],
            seed=1810)
        self.choice=choice
        self.first_plans=None
        self.first_public=None
        self.observations={}
        self.calls=0
        self.multiple=0

    def on_observation(self,raw):
        public=self.observer.observe(raw)
        self.observations[public.tick]=public

    def __call__(self,raw,legal):
        public=self.observations[raw.tick]
        plans=_plans(self.solver,public,legal)
        self.calls+=1
        self.multiple+=len(plans)>1
        if public.tick==0:
            if self.first_plans is not None:
                raise AssertionError("first tick repeated")
            self.first_plans=plans
            self.first_public=public
            if self.choice>=len(plans):
                raise ValueError("requested second proposal does not exist")
            return plans[self.choice]
        return plans[0]


def _job_outcomes(result,business,max_ticks):
    byid={x["id"]:x for x in result["samples"]}
    completions=[]
    late=0
    for job in business.jobs:
        ticks=[byid[sid]["completed_tick"] for sid in job["stages"]]
        finished=ticks[-1]
        if finished is None:
            late+=1
            completions.append(max_ticks)
        else:
            completions.append(finished)
            late+=int(finished>job["deadline_tick"])
    return {
        "completed_jobs":sum(byid[j["stages"][-1]]["completed_tick"] is not None
                             for j in business.jobs),
        "late_jobs":late,
        "mean_job_completion_tick_or_censor":sum(completions)/len(completions),
    }


def run_factual(spec,config,choice):
    business,geometry=build_business(spec,max_ticks=config["max_ticks"])
    policy=FixedFirstThenCP(geometry,business.rules,config,choice)
    result=BusinessSimulator(
        business,"world",spec["seed"],collect_metrics=True).run(
        on_observation=policy.on_observation,dispatch_policy=policy)
    if policy.first_public is None or policy.first_public.tick!=0:
        raise AssertionError("no first public observation")
    audit=resource_audit(result,business.rules)
    motion=discrete_safety_violations(result)
    lifecycle=lifecycle_audit(result["history"])
    if audit["violations"] or motion or lifecycle:
        raise AssertionError("actual motion, capacity or lifecycle violation")
    outcomes=_job_outcomes(result,business,config["max_ticks"])
    return {
        "business":business,"geometry":geometry,"first_state":policy.first_public,
        "first_plans":policy.first_plans,
        "first_selected":policy.first_plans[choice],
        "public_observations":policy.observations,  # history-derived facts; not model input
        "result":result,
        "outcomes":{
            "status":result["summary"]["status"],
            "end_tick":result["summary"]["ticks"],
            "censored":result["summary"]["status"]!="completed",
            "resource_wait_ticks":audit["capacity_wait_ticks"],
            "blocked_move_attempts":result["summary"]["blocked_move_attempts"],
            "policy_calls":policy.calls,
            "multi_candidate_calls":policy.multiple,
            "safety_violations":0,
            **outcomes,
        },
    }


def _factual_completed(result,h):
    snapshot=min(result["history"],key=lambda s:abs(s["tick"]-h)
                 if s["tick"]<=h else 100000+s["tick"])
    return sum(s["status"]=="completed" for s in snapshot["samples"])


def _forecast_for(first,*,checkpoint,config):
    geo=first["geometry"]
    rules=first["business"].rules
    public=first["first_state"]
    model=HybridSystemWorldModel(geo,rules,kind="v4",
                                 checkpoint=checkpoint,experimental=True)
    # Exactly the same CP-SAT *algorithm* is used for future predicted states,
    # applied to each own predicted state. The true future is never consulted.
    future_cp=BusinessCPSAT(
        geo,rules,
        candidate_limit=config["bounded_counterfactual"]["candidates"]["limit"],
        time_limit_seconds=config["bounded_counterfactual"]["candidates"]["wall_seconds"],
        deterministic_time_limit=config["bounded_counterfactual"]["candidates"]["deterministic_seconds"],
        seed=1810)
    continuation=lambda state:_plans(future_cp,state,candidates_for(state))[0]
    options={}
    for i,plan in enumerate(first["first_plans"][:2]):
        batch=model.rollout(public,plan,
                 continuation_policy=continuation,
                 continuation_policy_name="public_CP_SAT_first_proposal",
                 horizon_ticks=max(FORECAST_H),
                 particles=config["prediction_audit"]["particles"],
                 sampling_seed=1810+public.tick*101+first["result"]["summary"]["sample_count"]*271)
        options[str(i)]={
            str(h):{
                "expected_completed_stages":batch.distributions[h-1].expected_completed_samples,
                "expected_waiting_stages":batch.distributions[h-1].expected_waiting_samples,
                "expected_in_transit_stages":batch.distributions[h-1].expected_in_transit_samples,
            }
            for h in FORECAST_H
        }
    return options


def _serialize_plan(p):
    return [{"robot_id":a.robot_id,"sample_id":a.sample_id} for a in p]


def evaluate_one(spec,config,checkpoint):
    first=run_factual(spec,config,0)
    plans=first["first_plans"]
    info={
        "group":spec["id"],"regime":spec["regime"],"seed":spec["seed"],
        "scenario_jobs":spec["job_count"],
        "proposal_count_tick0":len(plans),
        "first_plan":_serialize_plan(plans[0]),
        "first_outcome":first["outcomes"],
        "first_trace_sha256":hashlib.sha256(json.dumps(
            first["result"]["events"],sort_keys=True).encode()).hexdigest(),
        "first_realized_completed_stages":{
            str(h):_factual_completed(first["result"],h) for h in FORECAST_H},
        "frozen_model_predictions":None,
        "second_outcome":None,
        "second_plan":None,
        "second_realized_completed_stages":None,
        "first_action_effect":None,
    }
    if len(plans)<2:
        return info
    other=run_factual(spec,config,1)
    if (other["first_state"]!=first["first_state"]
            or other["first_plans"]!=first["first_plans"]):
        raise AssertionError("initial state or legal candidate set changed between counterfactual runs")
    a,b=first["outcomes"],other["outcomes"]
    sensitivity={
        "delta_end_tick_second_minus_first":b["end_tick"]-a["end_tick"],
        "delta_resource_wait_second_minus_first":b["resource_wait_ticks"]-a["resource_wait_ticks"],
        "delta_late_jobs_second_minus_first":b["late_jobs"]-a["late_jobs"],
        "delta_mean_job_completion_tick_second_minus_first":
            b["mean_job_completion_tick_or_censor"]-a["mean_job_completion_tick_or_censor"],
        "both_completed":not(a["censored"] or b["censored"]),
        "completion_sensitive":a["completed_jobs"]!=b["completed_jobs"],
        "makespan_sensitive":not(a["censored"] or b["censored"]) and
                              a["end_tick"]!=b["end_tick"],
        "waiting_sensitive":a["resource_wait_ticks"]!=b["resource_wait_ticks"],
    }
    preds=_forecast_for(first,checkpoint=checkpoint,config=config)
    info.update({
        "second_plan":_serialize_plan(plans[1]),
        "second_outcome":b,
        "second_trace_sha256":hashlib.sha256(json.dumps(
            other["result"]["events"],sort_keys=True).encode()).hexdigest(),
        "second_realized_completed_stages":{
            str(h):_factual_completed(other["result"],h) for h in FORECAST_H},
        "frozen_model_predictions":preds,
        "first_action_effect":sensitivity,
    })
    return info


def _aggregate(groups):
    summary={}
    for regime in sorted({g["regime"] for g in groups}):
        cases=[g for g in groups if g["regime"]==regime]
        paired=[g for g in cases if g["second_outcome"] is not None]
        sufficient=[g for g in paired if g["first_action_effect"]["both_completed"]]
        def avg(vals):
            return sum(vals)/len(vals) if vals else None
        accuracy={}
        for h in FORECAST_H:
            h=str(h)
            expected=[]
            factual=[]
            direction_valid=[]
            agree=0
            for g in paired:
                pred=g["frozen_model_predictions"]
                diff=pred["1"][h]["expected_completed_stages"]-pred["0"][h]["expected_completed_stages"]
                truth=g["second_realized_completed_stages"][h]-g["first_realized_completed_stages"][h]
                expected.append(diff)
                factual.append(truth)
                if abs(diff)>1e-9 and truth!=0:
                    direction_valid.append(g["group"])
                    agree+=int((diff>0)==(truth>0))
            accuracy[h]={
                "paired_groups":len(paired),
                "forecast_difference_mean":avg(expected),
                "realized_difference_mean":avg(factual),
                "non_tied_prediction_and_factual_groups":len(direction_valid),
                "sign_agreement_on_non_ties":agree,
                "note":"Prediction horizon stage-completion ranking; not ultimate makespan utility",
            }
        summary[regime]={
            "groups":len(cases),
            "with_distinct_tick0_cp_sat_proposals":len(paired),
            "both_outcomes_complete":len(sufficient),
            "makespan_sensitive_groups":sum(g["first_action_effect"]["makespan_sensitive"]
                                            for g in paired),
            "resource_wait_sensitive_groups":sum(g["first_action_effect"]["waiting_sensitive"]
                                                 for g in paired),
            "late_job_sensitive_groups":sum(g["first_action_effect"]["delta_late_jobs_second_minus_first"]!=0
                                           for g in paired),
            "mean_end_tick_first":avg([g["first_outcome"]["end_tick"] for g in cases]),
            "mean_late_jobs_first":avg([g["first_outcome"]["late_jobs"] for g in cases]),
            "mean_resource_wait_first":avg([g["first_outcome"]["resource_wait_ticks"] for g in cases]),
            "mean_end_tick_pair_difference":avg([
                g["first_action_effect"]["delta_end_tick_second_minus_first"]
                for g in sufficient]),
            "prediction_vs_factual_horizons":accuracy,
        }
    return summary


def evaluate(checkpoint,*,protocol=PROTOCOL,cpu_budget=360):
    config=json.loads(Path(protocol).read_text())
    if config["schema"]!="RHWM_PR28_DYNAMIC_ACTION_IDENTIFIABILITY_20261112_V1":
        raise ValueError("wrong precommitted research protocol")
    if ([g["regime"] for g in config["scenarios"]].count("simultaneous_load")!=6 or
        [g["regime"] for g in config["scenarios"]].count("scheduled_release")!=6 or
        config["prediction_audit"]["forecast_horizons"]!=[5,10,20] or
        config["prediction_audit"]["particles"]!=3):
        raise AssertionError("changed fixed scene/horizon protocol")
    if file_sha(checkpoint)!=config["frozen_v4_sha256"]:
        raise ValueError("frozen V4 checkpoint SHA-256 mismatch")
    from torch import set_num_threads
    set_num_threads(1)
    begun=time.process_time()
    reports=[]
    for spec in config["scenarios"]:
        if time.process_time()-begun>cpu_budget:
            raise TimeoutError("precommitted experiment exceeded CPU cap")
        reports.append(evaluate_one(spec,config,checkpoint))
    cpu=time.process_time()-begun
    if cpu>cpu_budget:
        raise TimeoutError("precommitted experiment exceeded CPU cap")
    return {
        "schema":config["schema"],
        "protocol_sha256":file_sha(protocol),
        "checkpoint_sha256":file_sha(checkpoint),
        "unit":"independent synthetic scenario seed; two first-action branches per seed",
        "groups":reports,
        "summary":{
            "by_regime":_aggregate(reports),
            "groups":len(reports),
            "factual_runs":sum(1+int(g["second_outcome"] is not None) for g in reports),
            "both_plans_tested":sum(g["second_outcome"] is not None for g in reports),
            "total_model_candidate_forecasts":sum(2*(g["second_outcome"] is not None)
                                                    for g in reports),
            "gpu":0,"training":0,"cpu_process_seconds":cpu,
            "limitations":"controlled synthetic service jitter; scheduled releases are not unknown surprise arrivals; one-step action sensitivity not guaranteed meaningful for full policy optimization",
        },
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--cpu-budget",type=int,default=360)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError("research evidence is write-once")
    result=evaluate(args.checkpoint,cpu_budget=args.cpu_budget)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print("PR28_DYNAMIC_IDENTIFIABILITY_RESULT",json.dumps(result["summary"],ensure_ascii=False))


if __name__=="__main__":
    main()
