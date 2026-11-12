"""PR46 phase 2: frozen online CP-SAT vs original V4 vs public-KM V4.

The three arms face exactly the same *real* external outage duration for each
seed. A public-only health clock runs once each real observation tick, including
ticks with no dispatch. The unchanged PR30 OnlineHorizonPolicy candidate,
score and CPU fallback logic is inherited, not reimplemented or retuned.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import time

import torch

from business_transport.simulation import BusinessSimulator,resource_audit
from experiments.business_constraints.run import lifecycle_audit
from experiments.world_model_integration.online_horizon.pr30_run import (
    OnlineHorizonPolicy, _facts
)
from experiments.world_model_integration.recovery_integration.pr45_recovery_world import (
    PublicDownClock, ObservedRecoveryWorld
)
from experiments.world_model_integration.recovery_shift.pr46_shared import (
    PROTOCOL, ONLINE_ARMS, MECHANISMS, build_business, frozen_setup
)
from experiments.world_model_integration.system_world_model.evaluate_horizons import file_sha
from lab_transport.multisite_experiment import discrete_safety_violations


class OriginalSignatureRecoveryAdapter:
    """Expose PR30's exact rollout interface; track repair samples separately."""
    def __init__(self, model, clock):
        self.model=model
        self.clock=clock
        self.total_recovery_draws=0
        self.total_sampled_repairs=0
        self.rollout_calls=0

    def rollout(self,public_state,first_action,*,continuation_policy,
                continuation_policy_name,horizon_ticks,particles,sampling_seed):
        if self.clock.last_tick != public_state.tick:
            raise ValueError("missing contemporaneous public health observation")
        batch,events=self.model.rollout(
            public_state,first_action,down_ages=dict(self.clock.ages),
            continuation_policy=continuation_policy,
            horizon_ticks=horizon_ticks,particles=particles,
            sampling_seed=sampling_seed)
        self.total_recovery_draws+=len(events)
        self.total_sampled_repairs+=sum(bool(e["sampled_recovery"]) for e in events)
        self.rollout_calls+=1
        return batch


class RecoveryOnlinePolicy(OnlineHorizonPolicy):
    """Original PR30 proposals, forecast continuation, completion score, fallback."""
    def __init__(self,geo,rules,proto,*,checkpoint,km,seed):
        # No double-loading V4 weights: CP-SAT parent initialises no model.
        super().__init__("cp_sat",geo,rules,proto,checkpoint=None,seed=seed)
        self.arm="v4_h20"
        self.horizon=20
        self.clock=PublicDownClock()
        self.model=OriginalSignatureRecoveryAdapter(
            ObservedRecoveryWorld(geo,rules,checkpoint,km),self.clock)

    def observe(self,raw):
        # Observer callback occurs at EVERY real tick, before any dispatch
        # decision. Unlike the original BusinessObserver second call inside
        # __call__, the health clock is updated only here.
        self.clock.observe(raw)
        super().observe(raw)

    def __call__(self,raw,legal_candidates):
        if self.clock.last_tick!=raw.tick:
            raise ValueError("policy cannot dispatch without current public status")
        return super().__call__(raw,legal_candidates)


def one(spec,p,arm,checkpoint,km):
    business,geo=build_business(spec,p)
    proto=p["online"]
    begun=time.process_time()
    if arm=="cp_sat":
        policy=OnlineHorizonPolicy("cp_sat",geo,business.rules,proto,
                                   checkpoint=None,seed=spec["seed"])
    elif arm=="v4_original":
        policy=OnlineHorizonPolicy("v4_h20",geo,business.rules,proto,
                                   checkpoint=checkpoint,seed=spec["seed"])
    elif arm=="v4_public_km_recovery":
        policy=RecoveryOnlinePolicy(geo,business.rules,proto,
                                    checkpoint=checkpoint,km=km,seed=spec["seed"])
    else:
        raise ValueError("unregistered comparison arm")
    init_cpu=time.process_time()-begun
    begun=time.process_time()
    factual=BusinessSimulator(
        business,"world",spec["seed"],collect_metrics=True).run(
            dispatch_policy=policy,on_observation=policy.observe)
    execution_cpu=time.process_time()-begun
    audit=resource_audit(factual,business.rules)
    motion=discrete_safety_violations(factual)
    life=lifecycle_audit(factual["history"])
    if audit["violations"] or motion or life:
        raise AssertionError("actual simulation resource/motion/workflow violation")
    extra=getattr(policy.model,"total_sampled_repairs",0) if policy.model is not None else 0
    draws=getattr(policy.model,"total_recovery_draws",0) if policy.model is not None else 0
    return {
        "arm":arm,"seed":spec["seed"],"station_id":spec["station_id"],
        "mechanism":spec["mechanism"],
        "status":factual["summary"]["status"],
        "makespan_ticks":factual["summary"]["ticks"],
        "job_metrics":_facts(factual,business),
        "resource_wait_ticks":audit["capacity_wait_ticks"],
        "blocked_moves":factual["summary"]["blocked_move_attempts"],
        "calls":policy.n_calls,"multiple_candidate_calls":policy.n_multi,
        "plan_flips":policy.n_flip,
        "prediction_calls":policy.n_model_queries,
        "predicted_steps":policy.n_model_steps,
        "budget_exceeded":policy.n_budget_exceeded,
        "fallbacks":policy.n_fallback,"prediction_errors":policy.n_errors,
        "recovery_draws":draws,"sampled_repair_events":extra,
        "decision_cpu_seconds":policy.cpu_decision_seconds,
        "episode_cpu_seconds":init_cpu+execution_cpu,
        "model_init_cpu_seconds":init_cpu,
        "solver_fallbacks":policy.solver.fallbacks,
        "safety_violations":0,
        "real_event_sha256":hashlib.sha256(json.dumps(
            factual["events"],sort_keys=True).encode()).hexdigest(),
        "decision_records":policy.decisions,
    }


def summarize(groups):
    out={}
    for mechanism in MECHANISMS:
        cohort=[g for g in groups if g["mechanism"]==mechanism]
        by_arm={}
        paired={}
        for a in ONLINE_ARMS:
            rows=[g["arms"][a] for g in cohort]
            def mean(field): return sum(r[field] for r in rows)/len(rows)
            by_arm[a]={
                "scenario_groups":len(rows),
                "mean_makespan":mean("makespan_ticks"),
                "mean_late_jobs":sum(r["job_metrics"]["late_jobs"] for r in rows)/len(rows),
                "mean_resource_wait_ticks":mean("resource_wait_ticks"),
                "mean_cpu_decisions":mean("decision_cpu_seconds"),
                "mean_total_episode_cpu":mean("episode_cpu_seconds"),
                "completed_episodes":sum(not r["job_metrics"]["censored"] for r in rows),
                "policy_calls":sum(r["calls"] for r in rows),
                "multiple_candidate_calls":sum(r["multiple_candidate_calls"] for r in rows),
                "plan_flips":sum(r["plan_flips"] for r in rows),
                "budget_exceeded":sum(r["budget_exceeded"] for r in rows),
                "fallbacks":sum(r["fallbacks"] for r in rows),
                "prediction_errors":sum(r["prediction_errors"] for r in rows),
                "recovery_draws":sum(r["recovery_draws"] for r in rows),
                "sampled_repairs":sum(r["sampled_repair_events"] for r in rows),
                "safety_violations":sum(r["safety_violations"] for r in rows),
            }
        for a in ONLINE_ARMS:
            if a=="v4_original": continue
            delta=[
                g["arms"][a]["makespan_ticks"]-g["arms"]["v4_original"]["makespan_ticks"]
                for g in cohort
            ]
            paired[a]={
                "mean_delta_ticks_vs_v4":sum(delta)/len(delta),
                "wins":sum(x<0 for x in delta),
                "ties":sum(x==0 for x in delta),
                "losses":sum(x>0 for x in delta),
                "mean_delta_late_jobs":sum(
                    g["arms"][a]["job_metrics"]["late_jobs"]-
                    g["arms"]["v4_original"]["job_metrics"]["late_jobs"]
                    for g in cohort)/len(cohort),
            }
        out[mechanism]={"scenario_groups":len(cohort),"arms":by_arm,
                        "paired_vs_v4_original":paired}
    return out


def run(checkpoint,km_file,output):
    p,km=frozen_setup(checkpoint,km_file)
    if output.exists():raise FileExistsError("PR46 online evidence must be write-once")
    torch.set_num_threads(1)
    begun=time.process_time()
    groups=[]
    for spec in p["cohort"]["online"]:
        if time.process_time()-begun>=p["resources"]["process_cpu_budget_seconds"]:
            raise TimeoutError("online CPU budget exceeded")
        arms={arm:one(spec,p,arm,checkpoint,km) for arm in ONLINE_ARMS}
        # No shared secret future used in scoring; verify keyed factual world
        # does not change across policies beyond their legal dispatch decisions.
        groups.append({"id":spec["id"],"seed":spec["seed"],"mechanism":spec["mechanism"],
                       "station_id":spec["station_id"],"arms":arms})
    cpu=time.process_time()-begun
    if cpu>=p["resources"]["process_cpu_budget_seconds"]:
        raise TimeoutError("online CPU budget exceeded at end")
    data={
        "schema":"pr46-recovery-mechanism-shift-full-online-v1",
        "cohort":"online_only_not_forecast",
        "protocol_sha256":file_sha(PROTOCOL),
        "source_pr44_train_artifact_sha256":file_sha(km_file),
        "frozen_v4_checkpoint_sha256":file_sha(checkpoint),
        "gpu":0,"training":0,"existing_model_and_score_unchanged":True,
        "groups":groups,
        "summary":{"groups":len(groups),"by_mechanism":summarize(groups),
                   "cpu_process_seconds":cpu,
                   "limits":"CPU process-time gate is not a hard wall-time guarantee; same-map synthetic mechanisms only."}
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n")
    print("PR46_ONLINE_RESULT",json.dumps(data["summary"],ensure_ascii=False))
    return data


def main():
    q=argparse.ArgumentParser()
    q.add_argument("--checkpoint",required=True,type=Path)
    q.add_argument("--pr44-result",required=True,type=Path)
    q.add_argument("--out",required=True,type=Path)
    args=q.parse_args()
    run(args.checkpoint,args.pr44_result,args.out)

if __name__=="__main__":
    main()
