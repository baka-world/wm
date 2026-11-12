"""PR30 independent fully online horizon comparison on existing legal scheduling.

All options share BusinessCPSAT first-proposal generation, actual
BusinessSimulator steps, public BusinessObserver observation, existing RHWM
Rule or frozen V4. PETS-style probability particles, no new learner/loss.
Online forecast lookahead never receives factual future.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

from business_transport.dispatch import BusinessCPSAT
from business_transport.observation import BusinessObserver
from business_transport.simulation import BusinessSimulator, resource_audit
from business_transport.world_model import HybridSystemWorldModel
from experiments.business_constraints.run import lifecycle_audit
from experiments.world_model_integration.dynamic_identifiability.pr28_experiment import build_business
from experiments.world_model_integration.horizon_generalization.pr29_evaluate import _scenario_seed_set
from experiments.world_model_integration.system_world_model.evaluate_horizons import file_sha
from lab_transport.decision_value import candidates_for
from lab_transport.multisite_experiment import discrete_safety_violations

PROTOCOL=Path(__file__).with_name("pr30_protocol.json")
ARMS=("cp_sat","rule_h20","v4_h5","v4_h20")


def _unique_plans(dispatcher,state,candidates):
    plans=[]
    for plan in dispatcher.proposals(state,candidates):
        plan=tuple(plan)
        if plan not in plans: plans.append(plan)
    if not plans:plans=[()]
    legal=set(candidates)
    if any(a not in legal for plan in plans for a in plan):
        raise AssertionError("CP-SAT proposed an illegal current assignment")
    return plans


class OnlineHorizonPolicy:
    """Score existing lawful CP-SAT proposals using a bounded RHWM forecast."""

    def __init__(self,arm,geometry,rules,protocol,*,checkpoint,seed):
        self.arm=arm
        self.geometry=geometry
        self.rules=rules
        self.protocol=protocol
        self.seed=seed
        self.observer=BusinessObserver()
        self.solver=BusinessCPSAT(
            geometry,rules,
            candidate_limit=protocol["candidate_limit"],
            time_limit_seconds=0.08,
            deterministic_time_limit=0.04,seed=1810)
        self.forecast_solver=BusinessCPSAT(
            geometry,rules,
            candidate_limit=protocol["candidate_limit"],
            time_limit_seconds=0.08,
            deterministic_time_limit=0.04,seed=1810)
        self.model=None
        if arm in ("v4_h5","v4_h20"):
            self.model=HybridSystemWorldModel(
                geometry,rules,kind="v4",checkpoint=checkpoint,experimental=True)
        elif arm=="rule_h20":
            self.model=HybridSystemWorldModel(geometry,rules,kind="rule")
        self.horizon=5 if arm=="v4_h5" else 20
        self.n_calls=0
        self.n_multi=0
        self.n_flip=0
        self.n_model_queries=0
        self.n_model_steps=0
        self.n_budget_exceeded=0
        self.n_fallback=0
        self.n_errors=0
        self.decisions=[]
        self.cpu_decision_seconds=0.
        self.cpu_model_init_seconds=0.

    def observe(self,raw):
        self.observer.observe(raw)

    def __call__(self,raw,legal_candidates):
        start=time.process_time()
        obs=self.observer.observe(raw)
        plans=_unique_plans(self.solver,obs,legal_candidates)
        selected=plans[0]
        self.n_calls+=1
        if len(plans)>1:self.n_multi+=1
        values=[]
        if self.model is not None and len(plans)>1:
            seed=1810+self.seed*17+obs.tick*101
            # The CP-SAT continuation algorithm is applied to the
            # current *predicted* public particle state only.
            continuation=lambda state:_unique_plans(
                self.forecast_solver,state,candidates_for(state))[0]
            for plan in plans:
                if time.process_time()-start>=self.protocol["decision_cpu_budget_seconds"]:
                    self.n_budget_exceeded+=1
                    values=[]
                    break
                try:
                    batch=self.model.rollout(
                        obs,plan,continuation_policy=continuation,
                        continuation_policy_name="CP_SAT_predicted_state",
                        horizon_ticks=self.horizon,
                        particles=self.protocol["particles"],
                        sampling_seed=seed)
                    value=float(batch.distributions[-1].expected_completed_samples)
                    values.append(value)
                    self.n_model_queries+=1
                    self.n_model_steps+=self.horizon*self.protocol["particles"]
                except (RuntimeError,ValueError,AssertionError) as exc:
                    self.n_errors+=1
                    values=[]
                    self.decisions.append({
                        "tick":obs.tick,"error":str(exc)[:200]})
                    break
            # Fall back to the original legal proposal if the budget is
            # exhausted or any candidate is invalid. Do not partially rank.
            if len(values)==len(plans):
                best=max(values)
                best_index=next(i for i,v in enumerate(values) if best-v<=1e-9)
                selected=plans[best_index]
            else:self.n_fallback+=1
        self.n_flip+=int(selected!=plans[0])
        elapsed=time.process_time()-start
        self.cpu_decision_seconds+=elapsed
        self.decisions.append({
            "tick":obs.tick,"legal_plans":len(plans),
            "selected_index":plans.index(selected),
            "first_proposal":str(plans[0]),"selected":str(selected),
            "predicted_completed":values,
            "process_cpu_seconds":elapsed,
        })
        return selected


def _facts(result,business):
    sample={x["id"]:x for x in result["samples"]}
    completed=0
    late=0
    finish=[]
    for job in business.jobs:
        done=sample[job["stages"][-1]]["completed_tick"]
        completed+=int(done is not None)
        late+=int(done is None or done>job["deadline_tick"])
        if done is not None:finish.append(done)
    return {
        "completed_jobs":completed,
        "job_count":len(business.jobs),
        "late_jobs":late,
        "mean_completed_job_finish":sum(finish)/len(finish) if finish else None,
        "censored":completed!=len(business.jobs)
    }


def one(spec,protocol,arm,checkpoint):
    business,geo=build_business(spec,max_ticks=protocol["max_ticks"])
    t0=time.process_time()
    policy=OnlineHorizonPolicy(arm,geo,business.rules,protocol,checkpoint=checkpoint,seed=spec["seed"])
    init_elapsed=time.process_time()-t0
    begin=time.process_time()
    factual=BusinessSimulator(
        business,"world",spec["seed"],collect_metrics=True).run(
            dispatch_policy=policy,on_observation=policy.observe)
    execution_cpu=time.process_time()-begin
    audit=resource_audit(factual,business.rules)
    motion=discrete_safety_violations(factual)
    lifecycle=lifecycle_audit(factual["history"])
    if audit["violations"] or motion or lifecycle:
        raise AssertionError("invalid real transition or resource state")
    return {
        "arm":arm,
        "seed":spec["seed"],
        "regime":spec["regime"],
        "status":factual["summary"]["status"],
        "makespan_ticks":factual["summary"]["ticks"],
        "job_metrics":_facts(factual,business),
        "resource_wait_ticks":audit["capacity_wait_ticks"],
        "blocked_moves":factual["summary"]["blocked_move_attempts"],
        "policy_calls":policy.n_calls,
        "multiple_candidates":policy.n_multi,
        "changed_plan_calls":policy.n_flip,
        "model_prediction_calls":policy.n_model_queries,
        "model_prediction_steps":policy.n_model_steps,
        "decision_budget_exceeded":policy.n_budget_exceeded,
        "fallbacks":policy.n_fallback,
        "prediction_errors":policy.n_errors,
        "model_init_cpu_seconds":init_elapsed,
        "full_episode_cpu_seconds":execution_cpu+init_elapsed,
        "policy_decision_cpu_seconds":policy.cpu_decision_seconds,
        "solver_fallbacks":policy.solver.fallbacks,
        "safety_violations":0,
        "factual_event_sha256":hashlib.sha256(json.dumps(
            factual["events"],sort_keys=True).encode()).hexdigest(),
        "decision_records":policy.decisions,
    }


def summarize(groups):
    all_regimes=sorted(set(g["regime"] for g in groups))
    out={}
    for regime in all_regimes:
        cases=[g for g in groups if g["regime"]==regime]
        scores={}
        for arm in ARMS:
            values=[g["arms"][arm] for g in cases]
            def avg(key):
                return sum(v[key] for v in values)/len(values)
            scores[arm]={
                "groups":len(values),
                "mean_makespan":avg("makespan_ticks"),
                "mean_late_jobs":sum(v["job_metrics"]["late_jobs"] for v in values)/len(values),
                "mean_wait_ticks":avg("resource_wait_ticks"),
                "mean_blocked_moves":avg("blocked_moves"),
                "completed_groups":sum(not v["job_metrics"]["censored"] for v in values),
                "mean_total_cpu_seconds":avg("full_episode_cpu_seconds"),
                "mean_policy_cpu_seconds":avg("policy_decision_cpu_seconds"),
                "mean_model_init_cpu_seconds":avg("model_init_cpu_seconds"),
                "candidate_decisions":sum(v["multiple_candidates"] for v in values),
                "changed_proposals":sum(v["changed_plan_calls"] for v in values),
                "fallbacks":sum(v["fallbacks"] for v in values),
                "cpu_budget_exceeded":sum(v["decision_budget_exceeded"] for v in values),
                "prediction_errors":sum(v["prediction_errors"] for v in values),
                "safety_violations":sum(v["safety_violations"] for v in values),
            }
        paired={}
        for arm in ARMS[1:]:
            differences=[g["arms"][arm]["makespan_ticks"]-
                         g["arms"]["cp_sat"]["makespan_ticks"] for g in cases]
            paired[arm]={
                "mean_delta_makespan_ticks":sum(differences)/len(differences),
                "wins":sum(x<0 for x in differences),
                "ties":sum(x==0 for x in differences),
                "losses":sum(x>0 for x in differences),
                "mean_delta_late_jobs":sum(
                    g["arms"][arm]["job_metrics"]["late_jobs"]-
                    g["arms"]["cp_sat"]["job_metrics"]["late_jobs"]
                    for g in cases)/len(cases),
            }
        out[regime]={"groups":len(cases),"arms":scores,"paired_vs_cp_sat":paired}
    return out


def run(checkpoint,*,protocol_path=PROTOCOL,cpu_budget=480):
    proto=json.loads(Path(protocol_path).read_text())
    if (proto["protocol"]!="RHWM_PR30_INDEPENDENT_H20_ONLINE_20261112_V1" or
        proto["arms"]!=list(ARMS) or len(proto["groups"])!=24 or
        proto["horizon_v4_h20"]!=20 or proto["horizon_v4_h5"]!=5):
        raise ValueError("frozen PR30 experiment changed")
    if file_sha(checkpoint)!=proto["frozen_v4_sha256"]:
        raise ValueError("frozen V4 checkpoint mismatch")
    # This is a one-shot independent new cohort. All old development/test
    # seed groups are excluded before running.
    _scenario_seed_set({"scene_groups":proto["groups"]})
    from torch import set_num_threads
    set_num_threads(1)
    started=time.process_time()
    groups=[]
    for spec in proto["groups"]:
        if time.process_time()-started>cpu_budget:
            raise TimeoutError("PR30 CPU budget exceeded before next scene")
        groups.append({
            "id":spec["id"],"regime":spec["regime"],"seed":spec["seed"],
            "arms":{arm:one(spec,proto,arm,checkpoint) for arm in ARMS},
        })
    elapsed=time.process_time()-started
    if elapsed>cpu_budget:raise TimeoutError("PR30 CPU budget exceeded")
    return {
        "protocol":proto["protocol"],
        "protocol_sha256":file_sha(protocol_path),
        "v4_checkpoint_sha256":file_sha(checkpoint),
        "groups":groups,
        "summary":{
            "groups":len(groups),
            "by_regime":summarize(groups),
            "total_cpu_process_seconds":elapsed,
            "gpu":0,"training":0,
            "scope":"Whole-episode online decision-making by each arm, not offline oracle regret",
            "limitations":"predeclared synthetic groups, no real telemetry; model-prediction latency includes CP-SAT per-particle solver",
        },
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError("evidence is write once")
    output=run(args.checkpoint)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(output,indent=2,ensure_ascii=False)+"\n")
    print("PR30_ONLINE_H20_COMPLETED",json.dumps({
        "groups":output["summary"]["groups"],
        "total_cpu_process_seconds":output["summary"]["total_cpu_process_seconds"],
        "by_regime":output["summary"]["by_regime"]},ensure_ascii=False))


if __name__=="__main__":
    main()
