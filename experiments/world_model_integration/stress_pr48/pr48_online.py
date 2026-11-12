"""PR48 real fully-online external stress generalization, frozen H20/P3 and 1.5s.

All three arms use original BusinessCPSAT and original simulator; no new
world-model architecture, network training, scoring weights, horizon adaptation
or private future events. Wall-clock decision latency is explicitly recorded
in addition to PR30 process-CPU time for the candidate-ranking gate.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import math
import time

import torch

from business_transport.simulation import BusinessSimulator,resource_audit
from experiments.business_constraints.run import lifecycle_audit
from experiments.world_model_integration.online_horizon.pr30_run import (
    OnlineHorizonPolicy,_facts
)
from experiments.world_model_integration.recovery_shift.pr46_online import (
    RecoveryOnlinePolicy
)
from experiments.world_model_integration.recovery_integration.pr45_recovery_world import load_archived_km
from experiments.world_model_integration.stress_pr48.pr48_scenarios import (
    PROTOCOL,REGIMES,MECHANISMS,ARMS,build_business,validate
)
from experiments.world_model_integration.system_world_model.evaluate_horizons import file_sha
from lab_transport.multisite_experiment import discrete_safety_violations


def percentile(values,frac):
    if not values:
        return None
    if not 0<=frac<=1:
        raise ValueError("invalid percentile")
    v=sorted(values)
    idx=(len(v)-1)*frac
    lo=math.floor(idx)
    hi=math.ceil(idx)
    return v[lo]+(idx-lo)*(v[hi]-v[lo])


def real_episode(spec,proto,arm,checkpoint,km):
    case,geometry=build_business(spec,proto)
    settings={
        "candidate_limit":proto["candidate_limit"],
        "particles":proto["particles"],
        "decision_cpu_budget_seconds":proto["fixed_per_decision_cpu_process_budget_seconds"]
    }
    start=time.process_time()
    if arm=="cp_sat":
        policy=OnlineHorizonPolicy(
            "cp_sat",geometry,case.rules,settings,
            checkpoint=None,seed=spec["seed"])
    elif arm=="v4_original_15":
        policy=OnlineHorizonPolicy(
            "v4_h20",geometry,case.rules,settings,
            checkpoint=checkpoint,seed=spec["seed"])
    elif arm=="v4_recovery_15":
        policy=RecoveryOnlinePolicy(
            geometry,case.rules,settings,
            checkpoint=checkpoint,km=km,seed=spec["seed"])
    else:
        raise ValueError("arm not registered")
    init_cpu=time.process_time()-start
    wall_samples=[]
    cpu_samples=[]
    def wall_instrumented_dispatch(raw,candidates):
        before_wall=time.perf_counter()
        before_cpu=time.process_time()
        chosen=policy(raw,candidates)
        wall_samples.append(time.perf_counter()-before_wall)
        cpu_samples.append(time.process_time()-before_cpu)
        return chosen
    begin=time.process_time()
    begin_wall=time.perf_counter()
    factual=BusinessSimulator(
        case,"world",spec["seed"],collect_metrics=True).run(
        dispatch_policy=wall_instrumented_dispatch,
        on_observation=policy.observe)
    execution_cpu=time.process_time()-begin
    execution_wall=time.perf_counter()-begin_wall
    audit=resource_audit(factual,case.rules)
    motion=discrete_safety_violations(factual)
    lifecycle=lifecycle_audit(factual["history"])
    if audit["violations"] or motion or lifecycle:
        raise AssertionError("real-world collision/resource/workflow violation")
    if len(wall_samples)!=policy.n_calls or len(cpu_samples)!=policy.n_calls:
        raise AssertionError("lost wall clock decision timing")
    if not all(x>=0 for x in wall_samples+cpu_samples):
        raise AssertionError("negative decision timing")
    task=_facts(factual,case)
    if task["job_count"]!=spec["job_count"]:
        raise AssertionError("number of actual jobs changed")
    if policy.n_errors:
        raise AssertionError("frozen V4 or recovery model threw a prediction error")
    draws=getattr(policy.model,"total_recovery_draws",0) if policy.model else 0
    repairs=getattr(policy.model,"total_sampled_repairs",0) if policy.model else 0
    return {
        "arm":arm,
        "seed":spec["seed"],"regime":spec["regime"],
        "mechanism":spec["mechanism"],
        "job_count":spec["job_count"],
        "status":factual["summary"]["status"],
        "makespan_ticks":factual["summary"]["ticks"],
        "job_metrics":task,
        "capacity_wait_ticks":audit["capacity_wait_ticks"],
        "blocked_moves":factual["summary"]["blocked_move_attempts"],
        "decision_calls":policy.n_calls,
        "multi_candidate_calls":policy.n_multi,
        "changed_proposals":policy.n_flip,
        "model_queries":policy.n_model_queries,
        "world_model_steps":policy.n_model_steps,
        "budget_overruns":policy.n_budget_exceeded,
        "fallbacks":policy.n_fallback,
        "model_errors":policy.n_errors,
        "solver_fallbacks":policy.solver.fallbacks,
        "recovery_draws":draws,"sampled_repair_events":repairs,
        "model_init_cpu_seconds":init_cpu,
        "full_episode_cpu_seconds":init_cpu+execution_cpu,
        "full_episode_wall_seconds":execution_wall,
        "total_policy_process_seconds":policy.cpu_decision_seconds,
        "wall_clock_decision_p50_seconds":percentile(wall_samples,.5),
        "wall_clock_decision_p95_seconds":percentile(wall_samples,.95),
        "wall_clock_decision_max_seconds":max(wall_samples) if wall_samples else None,
        "raw_wall_clock_decisions_seconds":wall_samples,
        "raw_cpu_process_decisions_seconds":cpu_samples,
        "factual_event_sha256":hashlib.sha256(json.dumps(
            factual["events"],sort_keys=True).encode()).hexdigest(),
        "safety_violations":0,
        "decision_records":policy.decisions,
    }


def _mean(v):
    return sum(v)/len(v) if v else None


def summarize(groups):
    """Independent unit is one group/seed, not decision or particle."""
    result={}
    for kind in REGIMES:
        selected=[g for g in groups if g["regime"]==kind]
        if not selected:
            continue
        arms={}
        paired={}
        for arm in ARMS:
            rows=[g["arms"][arm] for g in selected]
            arms[arm]={
                "groups":len(rows),
                "completed_episodes":sum(not x["job_metrics"]["censored"] for x in rows),
                "mean_makespan":_mean([x["makespan_ticks"] for x in rows]),
                "mean_completed_jobs":_mean([x["job_metrics"]["completed_jobs"] for x in rows]),
                "mean_late_jobs":_mean([x["job_metrics"]["late_jobs"] for x in rows]),
                "mean_capacity_wait_ticks":_mean([x["capacity_wait_ticks"] for x in rows]),
                "mean_full_episode_cpu_seconds":_mean([x["full_episode_cpu_seconds"] for x in rows]),
                "mean_wall_p95_decision_seconds":_mean([x["wall_clock_decision_p95_seconds"] for x in rows]),
                "overall_wall_decision_p95_seconds":percentile([
                    t for x in rows for t in x["raw_wall_clock_decisions_seconds"]],.95),
                "overall_wall_decision_max_seconds":max(
                    t for x in rows for t in x["raw_wall_clock_decisions_seconds"]),
                "decision_calls":sum(x["decision_calls"] for x in rows),
                "multi_candidate_calls":sum(x["multi_candidate_calls"] for x in rows),
                "model_queries":sum(x["model_queries"] for x in rows),
                "budget_overruns":sum(x["budget_overruns"] for x in rows),
                "fallbacks":sum(x["fallbacks"] for x in rows),
                "model_errors":sum(x["model_errors"] for x in rows),
                "changed_proposals":sum(x["changed_proposals"] for x in rows),
                "safety_violations":sum(x["safety_violations"] for x in rows),
            }
        for active in ("v4_original_15","v4_recovery_15"):
            baseline="cp_sat" if active=="v4_original_15" else "v4_original_15"
            deltas=[g["arms"][active]["makespan_ticks"]-g["arms"][baseline]["makespan_ticks"]
                    for g in selected]
            lateness=[g["arms"][active]["job_metrics"]["late_jobs"]
                      -g["arms"][baseline]["job_metrics"]["late_jobs"]
                      for g in selected]
            paired[f"{active}_minus_{baseline}"]={
                "mean_delta_ticks":_mean(deltas),
                "wins":sum(x<0 for x in deltas),"ties":sum(x==0 for x in deltas),
                "losses":sum(x>0 for x in deltas),
                "mean_extra_late_jobs":_mean(lateness),
                "paired_deltas_ticks":deltas,
            }
        for mech in MECHANISMS:
            slice_=[g for g in selected if g["mechanism"]==mech]
            deltas=[g["arms"]["v4_recovery_15"]["makespan_ticks"]-
                    g["arms"]["v4_original_15"]["makespan_ticks"] for g in slice_]
            paired[f"recovery_minus_original_{mech}"]={
                "groups":len(slice_),"mean_delta_ticks":_mean(deltas),
                "wins":sum(x<0 for x in deltas),"ties":sum(x==0 for x in deltas),
                "losses":sum(x>0 for x in deltas),
                "paired_deltas_ticks":deltas,
            }
        result[kind]={"groups":len(selected),"arms":arms,"paired":paired}
    return result


def evaluate(checkpoint,km_file,regime,out):
    cfg=json.loads(PROTOCOL.read_text())
    validate(cfg)
    if regime not in REGIMES:
        raise ValueError("unregistered stress regime")
    if file_sha(checkpoint)!=cfg["v4_sha256"]:
        raise ValueError("frozen V4 checkpoint SHA mismatch")
    km=load_archived_km(km_file)
    if out.exists():
        raise FileExistsError("no overwriting PR48 stress experiment evidence")
    torch.set_num_threads(1)
    start=time.process_time()
    subset=[x for x in cfg["cohort"] if x["regime"]==regime]
    groups=[]
    for i,spec in enumerate(subset):
        if time.process_time()-start>=cfg["resources"]["experiment_process_cpu_limit"]:
            raise TimeoutError("CPU experiment limit exceeded")
        order=["cp_sat","v4_original_15","v4_recovery_15"]
        if spec["seed"]%2:
            order=["cp_sat","v4_recovery_15","v4_original_15"]
        arm_results={name:real_episode(spec,cfg,name,checkpoint,km) for name in order}
        groups.append({"id":spec["id"],"seed":spec["seed"],
                       "regime":regime,"mechanism":spec["mechanism"],
                       "station_id":spec["station_id"],"arms":arm_results})
        print("PR48_SCENE_COMPLETED",json.dumps({
            "i":i+1,"regime":regime,"id":spec["id"],
            "makespan":{k:v["makespan_ticks"] for k,v in arm_results.items()},
            "cpu_so_far":time.process_time()-start}),flush=True)
    elapsed=time.process_time()-start
    if elapsed>=cfg["resources"]["experiment_process_cpu_limit"]:
        raise TimeoutError("PR48 CPU budget exceeded after last scene")
    result={
        "schema":"pr48-full-online-stress-independent-v1",
        "regime":regime,
        "protocol_sha256":file_sha(PROTOCOL),
        "v4_checkpoint_sha256":file_sha(checkpoint),
        "pr44_train_only_model_sha256":file_sha(km_file),
        "new_gpu":0,"new_training":0,
        "groups":groups,
        "summary":{"groups":len(groups),"actual_episodes":len(groups)*len(ARMS),
                   "by_regime":summarize(groups),
                   "process_cpu_seconds":elapsed,
                   "scope":"same map, new load/event mechanisms, frozen 1.5s equal-process CPU budget, no real plant data"}
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print("PR48_REGIME_RESULT",json.dumps(result["summary"],ensure_ascii=False),flush=True)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True,type=Path)
    p.add_argument("--pr44-result",required=True,type=Path)
    p.add_argument("--regime",required=True,choices=REGIMES)
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args()
    evaluate(args.checkpoint,args.pr44_result,args.regime,args.out)


if __name__=="__main__":
    main()
