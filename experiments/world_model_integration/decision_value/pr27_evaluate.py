"""PR27: genuinely execute three *online* policies in paired synthetic scenes.

Refs: BusinessCPSAT current repository, RHWM-v1 lawful V4 state rollouts,
PETS (Chua et al 2018) particle expected costs, Tang & Wang (2025) AGV
Bayesian delay risk. No invented model training, ground-truth future actions,
or simulator-private world-model information.

This is a frozen evaluation of a simple additive expected-delay ranking
adapter. It is NOT a new scheduling solver or a reproduction of PETS CEM.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import argparse
import hashlib
import json
import math
import time

from business_transport.observation import BusinessObserver
from business_transport.dispatch import BusinessCPSAT
from business_transport.simulation import BusinessSimulator, resource_audit
from business_transport.world_model import HybridSystemWorldModel
from business_transport.world_model.joint_transport import JointServiceTransportWorldModel
from business_transport.world_model.transport_bayes import TransportBayesModel
from experiments.business_constraints.run import lifecycle_audit
from experiments.world_model_integration.system_world_model.evaluate_horizons import (
    file_sha, group_scenario,
)
from experiments.world_model_integration.transport_events.benchmark import PROTOCOL as SCENES
from lab_transport.decision_value import candidates_for,nearest_batch,terminal_distance
from lab_transport.multisite_experiment import discrete_safety_violations


ROOT=Path(__file__).resolve().parent
PROTOCOL=ROOT/"pr27_protocol.json"
ARMS=("cpsat","v4","v4_plus_risk")


def _scenario_with_deadlines(spec,config):
    business,geo=group_scenario(spec,max_ticks=config["scenario_max_ticks"])
    ordered=sorted(j["id"] for j in business.jobs)
    by_job={jid:45+5*i for i,jid in enumerate(ordered)}
    # Explicit predeclared artificial deadlines, independent of model arm,
    # never learned from observed future outcomes.
    case=replace(business.scenario,samples=tuple(
        replace(sample,deadline_tick=by_job[sample.id.split("::stage")[0]])
        for sample in business.scenario.samples))
    jobs=tuple({**j,"deadline_tick":by_job[j["id"]]} for j in business.jobs)
    return replace(business,scenario=case,jobs=jobs),geo


def _trace_cost(initial,forecast,geometry):
    """Reuses decision_value.trace_score's unfinished/late and distance terms."""
    known={s.id for s in initial.samples}
    totals=[]
    for particle in forecast.particles:
        running=0
        for step in particle.steps:
            state=step.next_state
            samples={s.id:s for s in state.samples}
            if not known.issubset(samples):
                raise ValueError("predicted future lost a known task")
            for sid in known:
                sample=samples[sid]
                if sample.status!="completed":
                    running+=1+4*int(sample.deadline_tick is not None
                                      and state.tick>sample.deadline_tick)
        remaining=terminal_distance(particle.final_state,known,geometry)
        totals.append((running,remaining))
    n=len(totals)
    if n<1:raise AssertionError("missing prediction particle")
    return sum(x for x,_ in totals)/n,sum(x for _,x in totals)/n


class OnlineBusinessPolicy:
    """Fixed CP-SAT candidates, optional finite-budget learned reranker."""

    def __init__(self,name,geometry,rules,config,*,v4_checkpoint,bayes,scene_seed):
        if name not in ARMS:raise ValueError("unknown policy")
        self.name,self.geometry,self.rules=name,geometry,rules
        self.config,self.scene_seed=config,scene_seed
        self.observer=BusinessObserver()
        self.solver=BusinessCPSAT(
            geometry,rules,candidate_limit=config["candidate_limit"],
            time_limit_seconds=config["solver_time_limit_seconds"],
            deterministic_time_limit=config["solver_deterministic_time_limit"],
            seed=config["sampling_seed"])
        self.model=None
        if name=="v4":
            self.model=HybridSystemWorldModel(
                geometry,rules,kind="v4",
                checkpoint=v4_checkpoint,experimental=True)
        elif name=="v4_plus_risk":
            self.model=JointServiceTransportWorldModel(
                geometry,rules,v4_checkpoint=v4_checkpoint,transport_model=bayes)
        self.calls=self.multi=0
        self.changed=0
        self.failures=0
        self.model_steps=0
        self.risk_queries=0
        self.cpu_seconds=0.
        self.truncated=0
        self.records=[]

    def observe(self,raw):
        self.observer.observe(raw)

    def __call__(self,raw,legal_candidates):
        started=time.process_time()
        obs=self.observer.observe(raw)
        proposals=self.solver.proposals(obs,legal_candidates)
        plans=[]
        for plan in proposals:
            plan=tuple(plan)
            if plan not in plans:
                plans.append(plan)
        if not plans:plans=[()]
        legal=set(legal_candidates)
        if any(any(a not in legal for a in plan) for plan in plans):
            raise AssertionError("CP-SAT returned an illegal current assignment")
        base=plans[0]
        selected=base
        self.calls+=1
        if len(plans)>1:self.multi+=1
        scored=[]
        timed_out=False
        if self.model is not None and len(plans)>1:
            for index,plan in enumerate(plans):
                if time.process_time()-started>self.config["per_decision_cpu_limit_seconds"]:
                    timed_out=True
                    self.truncated+=1
                    break
                seed=(self.config["sampling_seed"]+self.scene_seed*17+
                      obs.tick*101)
                opts=dict(
                    continuation_policy=lambda st:nearest_batch(
                        st,candidates_for(st),self.geometry),
                    continuation_policy_name=self.config["continuation_policy"],
                    horizon_ticks=self.config["horizon_ticks"],
                    particles=self.config["particles"],
                    sampling_seed=seed)
                try:
                    if self.name=="v4":
                        forecast=self.model.rollout(obs,plan,**opts)
                        risk=0.
                    else:
                        joint=self.model.rollout(obs,plan,mode="shadow",**opts)
                        forecast=joint.forecast
                        risk=sum(x.p_delay for x in joint.transport_risks)
                        risk*=self.model.transport.positive_delay_ticks / self.config["particles"]
                        self.risk_queries+=len(joint.transport_risks)
                    primary,secondary=_trace_cost(obs,forecast,self.geometry)
                    self.model_steps+=self.config["horizon_ticks"]*self.config["particles"]
                    # Fixed one-time-unit conversion: expected delayed ticks
                    # affects the first existing completion-time cost, not the
                    # CP-SAT feasibility constraints or simulation physics.
                    score=(primary+risk,secondary,index)
                    scored.append((score,plan))
                except (ValueError,AssertionError,RuntimeError) as exc:
                    self.failures+=1
                    self.records.append({"tick":obs.tick,"model_error":str(exc)[:250]})
                    scored=[]
                    break
            if len(scored)==len(plans) and not timed_out:
                selected=min(scored,key=lambda x:x[0])[1]
        if selected!=base:self.changed+=1
        elapsed=time.process_time()-started
        self.cpu_seconds+=elapsed
        self.records.append({
            "tick":obs.tick,"candidate_plans":len(plans),
            "base":[[a.robot_id,a.sample_id] for a in base],
            "selected":[[a.robot_id,a.sample_id] for a in selected],
            "ranked":len(scored) if self.model is not None else 0,
            "model_skipped":timed_out or len(scored)<len(plans),
            "cpu_seconds":elapsed})
        return selected


def _job_metrics(result,business):
    by_id={s["id"]:s for s in result["samples"]}
    out=[]
    for job in business.jobs:
        final=by_id[job["stages"][-1]]["completed_tick"]
        due=job["deadline_tick"]
        out.append({
            "job":job["id"],"finished_tick":final,"deadline":due,
            "complete":final is not None,
            "late":final is None or final>due,
        })
    return {
        "jobs":len(out),"completed_jobs":sum(x["complete"] for x in out),
        "late_jobs":sum(x["late"] for x in out),
        "job_finish_mean":sum(x["finished_tick"] or business.scenario.max_ticks
                               for x in out)/len(out),
        "all_jobs_completed":all(x["complete"] for x in out),
    }


def evaluate_one(spec,settings,arm,*,checkpoint,bayes):
    business,geo=_scenario_with_deadlines(spec,settings)
    policy=OnlineBusinessPolicy(
        arm,geo,business.rules,settings,
        v4_checkpoint=checkpoint,bayes=bayes,scene_seed=spec["seed"])
    sim=BusinessSimulator(business,"world",spec["seed"],collect_metrics=True)
    start=time.process_time()
    result=sim.run(dispatch_policy=policy,on_observation=policy.observe)
    total_cpu=time.process_time()-start
    resource=resource_audit(result,business.rules)
    motion=discrete_safety_violations(result)
    lifecycle=lifecycle_audit(result["history"])
    if resource["violations"] or motion or lifecycle:
        raise AssertionError("the actual simulator violated motion/resource/task rules")
    jobs=_job_metrics(result,business)
    return {
        "arm":arm,"seed":spec["seed"],"group":spec["id"],
        "summary":result["summary"],
        "jobs":jobs,
        "time_until_end":result["summary"]["ticks"],
        "resource_wait_ticks":resource["capacity_wait_ticks"],
        "robot_path_blocked_moves":result["summary"]["blocked_move_attempts"],
        "policy_calls":policy.calls,"multiple_options":policy.multi,
        "changed_from_cpsat_first_plan":policy.changed,
        "model_steps":policy.model_steps,
        "transport_risk_queries":policy.risk_queries,
        "model_failures":policy.failures,
        "truncated_decisions":policy.truncated,
        "solver_fallbacks":policy.solver.fallbacks,
        "policy_cpu_seconds":policy.cpu_seconds,
        "total_episode_cpu_seconds":total_cpu,
        "resource_violations":len(resource["violations"]),
        "motion_violations":motion,"lifecycle_violations":len(lifecycle),
        "event_hash":hashlib.sha256(
            json.dumps(result["events"],sort_keys=True).encode()).hexdigest(),
        "decision_records":policy.records,
    }


def _summary(results):
    out={}
    for arm in ARMS:
        rows=[g["arms"][arm] for g in results]
        out[arm]={
            "groups":len(rows),
            "mean_ticks":sum(x["time_until_end"] for x in rows)/len(rows),
            "mean_late_jobs":sum(x["jobs"]["late_jobs"] for x in rows)/len(rows),
            "total_late_jobs":sum(x["jobs"]["late_jobs"] for x in rows),
            "mean_resource_wait_ticks":sum(x["resource_wait_ticks"] for x in rows)/len(rows),
            "mean_blocked_moves":sum(x["robot_path_blocked_moves"] for x in rows)/len(rows),
            "mean_job_finish":sum(x["jobs"]["job_finish_mean"] for x in rows)/len(rows),
            "policy_cpu_total":sum(x["policy_cpu_seconds"] for x in rows),
            "model_steps":sum(x["model_steps"] for x in rows),
            "risk_queries":sum(x["transport_risk_queries"] for x in rows),
            "solver_fallbacks":sum(x["solver_fallbacks"] for x in rows),
            "changed_from_first_plan":sum(x["changed_from_cpsat_first_plan"] for x in rows),
            "multi_candidate_decisions":sum(x["multiple_options"] for x in rows),
            "model_failures":sum(x["model_failures"] for x in rows),
            "truncated_decisions":sum(x["truncated_decisions"] for x in rows),
            "all_completed":sum(x["jobs"]["all_jobs_completed"] for x in rows),
            "safety_violations":sum(
                x["resource_violations"]+x["motion_violations"]+
                x["lifecycle_violations"] for x in rows),
        }
    paired={}
    for arm in ARMS[1:]:
        paired[arm]={
            "ticks_delta":sum(g["arms"][arm]["time_until_end"]-
                             g["arms"]["cpsat"]["time_until_end"]
                             for g in results)/len(results),
            "late_jobs_delta":sum(g["arms"][arm]["jobs"]["late_jobs"]-
                                 g["arms"]["cpsat"]["jobs"]["late_jobs"]
                                 for g in results)/len(results),
            "wins_makespan":sum(g["arms"][arm]["time_until_end"] <
                                g["arms"]["cpsat"]["time_until_end"]
                                for g in results),
            "ties_makespan":sum(g["arms"][arm]["time_until_end"] ==
                                g["arms"]["cpsat"]["time_until_end"]
                                for g in results),
            "losses_makespan":sum(g["arms"][arm]["time_until_end"] >
                                  g["arms"]["cpsat"]["time_until_end"]
                                  for g in results),
        }
    return out,paired


def evaluate(*,checkpoint,transport_json,cpu_budget=360):
    proto=json.loads(PROTOCOL.read_text())
    if (proto["protocol"]!="RHWM_PR27_PAIRED_POLICY_EVALUATION_V1"
            or proto["arms"]!=list(ARMS)
            or proto["horizon_ticks"]!=5
            or proto["particles"]!=3):
        raise ValueError("fixed decision protocol altered")
    if file_sha(checkpoint)!=proto["checkpoints"]["v4_sha256"]:
        raise ValueError("wrong V4 frozen checkpoint")
    scene=json.loads(SCENES.read_text())
    groups=scene["future_decision_groups"]
    if len(groups)!=12:
        raise AssertionError("PR24 sealed decision scenes changed")
    bayes=TransportBayesModel.from_payload(json.loads(Path(transport_json).read_text()))
    if bayes.total_legs!=240:
        raise ValueError("wrong frozen reference transport training set")
    from torch import set_num_threads
    set_num_threads(1)
    start=time.process_time()
    results=[]
    for spec in groups:
        if time.process_time()-start>cpu_budget:
            raise TimeoutError("CPU experiment cap reached")
        arms={}
        for arm in ARMS:
            arms[arm]=evaluate_one(spec,proto,arm,checkpoint=checkpoint,bayes=bayes)
        results.append({"group":spec["id"],"seed":spec["seed"],"arms":arms})
    byarm,paired=_summary(results)
    elapsed=time.process_time()-start
    if elapsed>cpu_budget:raise TimeoutError("CPU evaluation exceeded frozen cap")
    return {
        "protocol":proto["protocol"],
        "decision_protocol_sha256":file_sha(PROTOCOL),
        "scenario_protocol_sha256":file_sha(SCENES),
        "transport_model_sha256":file_sha(transport_json),
        "v4_sha256":file_sha(checkpoint),
        "groups":results,
        "summary":{
            "arms":byarm,"paired_vs_cpsat":paired,
            "independent_scene_groups":len(groups),"cpu_seconds":elapsed,
            "gpu":0,"new_training":0,
            "key_limitations":"12 small synthetic scene groups; no real robot telemetry; same CP-SAT generation algorithm on possibly different later observed states",
            "model_source":"frozen V4 + upstream pgmpy Bayesian transport risk",
        }
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--transport",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError("paired decision evidence is immutable")
    r=evaluate(checkpoint=args.checkpoint,transport_json=args.transport)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(r,indent=2,ensure_ascii=False)+"\n")
    print("PR27_PAIRED_WORLD_MODEL_DECISION_RESULT",json.dumps(r["summary"],ensure_ascii=False))


if __name__=="__main__":
    main()
