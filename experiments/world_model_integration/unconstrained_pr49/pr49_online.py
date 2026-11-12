"""PR49 48-new-scenario offline world-model quality with generous, symmetric compute.

CPU on hosted GitHub Actions is only a reproducible execution proxy. Neither
the 30-second emergency process-time gate nor CPU wall-latency is used as a
scientific optimization objective or deployment guarantee. Same V4 checkpoint,
PR44 station-KM, PR30 H20/P3 scoring and identical CP-SAT candidate path.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import json
import time

import torch

from experiments.world_model_integration.stress_pr48.pr48_online import (
    real_episode,percentile
)
from experiments.world_model_integration.stress_pr48.pr48_scenarios import (
    PROTOCOL as PR48_PROTOCOL
)
from experiments.world_model_integration.recovery_integration.pr45_recovery_world import load_archived_km
from experiments.world_model_integration.recovery_budget.pr47_online import PROTOCOL as PR47_PROTOCOL
from experiments.world_model_integration.recovery_shift.pr46_shared import PROTOCOL as PR46_PROTOCOL
from experiments.world_model_integration.health_recovery.pr44_observed_recovery import PROTOCOL as PR44_PROTOCOL
from experiments.world_model_integration.system_world_model.evaluate_horizons import file_sha

PROTOCOL=Path(__file__).with_name("pr49_protocol.json")
REGIMES=("congestion","multi_fault","staggered_arrivals")
MECHANISMS=("within_support","extended_outage")
ARMS=("cp_sat","v4_original_h20_full","v4_recovery_h20_full")
SOURCE_ARMS={"cp_sat":"cp_sat","v4_original_h20_full":"v4_original_15",
             "v4_recovery_h20_full":"v4_recovery_15"}


def validate(cfg):
    old=json.loads(PR48_PROTOCOL.read_text())
    if cfg["protocol"]!="PR49_WORLD_MODEL_QUALITY_HIGH_COMPUTE_20261112_V1":
        raise ValueError("PR49 protocol identity changed")
    if cfg["source_pr48_commit"]!="cc5d01bed776a2b32a0a72db1ff3eaf4f53caacc":
        raise ValueError("wrong source baseline")
    if cfg["v4_sha256"]!=old["v4_sha256"]:
        raise ValueError("V4 digest changed")
    if cfg["predefined_scenario_mechanisms"]!=old["predefined_scenario_mechanisms"]:
        raise ValueError("stress scenario mechanics changed")
    if cfg["arms"]!=list(ARMS) or cfg["process_cpu_safety_guard_seconds"]!=30:
        raise ValueError("wrong frozen arms or catastrophe guard")
    if cfg["candidate_limit"]!=4 or cfg["particles"]!=3 or cfg["horizon_ticks"]!=20:
        raise ValueError("proposal scorer not frozen")
    if cfg["resources"]["neural_training"] or cfg["resources"]["gpu_training"]:
        raise ValueError("training not authorized within test cohort")
    if len(cfg["cohort"])!=48 or len({c["seed"] for c in cfg["cohort"]})!=48:
        raise ValueError("not 48 new unique cases")
    prior={c["seed"] for c in old["cohort"]}
    for path in (PR47_PROTOCOL,PR46_PROTOCOL,PR44_PROTOCOL):
        p=json.loads(path.read_text())
        cases=p["cohort"]
        if isinstance(cases,dict):
            for group in cases.values():
                prior.update(c["seed"] for c in group)
        else:
            prior.update(c["seed"] for c in cases)
    prior.update(7100009+s*10000+i*271 for s in range(3) for i in range(4))
    if prior & {c["seed"] for c in cfg["cohort"]}:
        raise ValueError("previously exposed seed reused")
    for regime in REGIMES:
        selected=[c for c in cfg["cohort"] if c["regime"]==regime]
        if len(selected)!=16 or any(c["job_count"]!=(14 if regime=="congestion" else 12) for c in selected):
            raise ValueError("stratum workload changed")
        if sorted(c["mechanism"] for c in selected)!=sorted(
            ["within_support"]*8+["extended_outage"]*8):
            raise ValueError("stratum duration mechanism not balanced")
    return True


def evaluate_case(spec,cfg,checkpoint,km):
    # Source PR48 implementation actually constructs the original simulator
    # and checks business constraints. Only the identical per-arm CPU allowance
    # changes, from 1.5 to a common 30 seconds.
    order=list(ARMS)
    if spec["seed"]%2:
        order=["cp_sat","v4_recovery_h20_full","v4_original_h20_full"]
    outcomes={}
    for public in order:
        actual=real_episode(spec,cfg,SOURCE_ARMS[public],checkpoint,km)
        actual["public_arm"]=public
        outcomes[public]=actual
    for arm in ARMS[1:]:
        assert outcomes[arm]["model_errors"]==0
    return {"id":spec["id"],"seed":spec["seed"],
            "regime":spec["regime"],"mechanism":spec["mechanism"],
            "station_id":spec["station_id"],"evaluation_order":order,
            "arms":outcomes}


def summarize(rows):
    if not rows:
        raise ValueError("no scenarios")
    old=[r["arms"][ARMS[1]] for r in rows]
    new=[r["arms"][ARMS[2]] for r in rows]
    base=[r["arms"][ARMS[0]] for r in rows]
    deltas=[n["makespan_ticks"]-o["makespan_ticks"] for n,o in zip(new,old)]
    complete=[
        sum(not r["job_metrics"]["censored"] for r in arm)
        for arm in (base,old,new)]
    summary={
        "scenario_groups":len(rows),
        "avg_actual_makespan_ticks":{
            arm:sum(g["arms"][arm]["makespan_ticks"] for g in rows)/len(rows)
            for arm in ARMS},
        "completed_episodes":dict(zip(ARMS,complete)),
        "mean_recovery_minus_original_ticks":sum(deltas)/len(deltas),
        "wins":sum(x<0 for x in deltas),
        "ties":sum(x==0 for x in deltas),
        "losses":sum(x>0 for x in deltas),
        "paired_deltas":deltas,
        "budget_overruns":{
            arm:sum(g["arms"][arm]["budget_overruns"] for g in rows)
            for arm in ARMS},
        "legal_fallbacks":{
            arm:sum(g["arms"][arm]["fallbacks"] for g in rows)
            for arm in ARMS},
        "late_jobs":{
            arm:sum(g["arms"][arm]["job_metrics"]["late_jobs"] for g in rows)
            for arm in ARMS},
        "safety_violations":sum(
            g["arms"][arm]["safety_violations"]
            for g in rows for arm in ARMS),
        "cpu_episode_seconds_mean":{
            arm:sum(g["arms"][arm]["full_episode_cpu_seconds"] for g in rows)/len(rows)
            for arm in ARMS},
    }
    summary["all_candidates_fully_scored_in_both_models"]=all(
        summary["budget_overruns"][arm]==0 and summary["legal_fallbacks"][arm]==0
        for arm in ARMS[1:])
    return summary


def run(checkpoint,kmfile,regime,out):
    cfg=json.loads(PROTOCOL.read_text())
    validate(cfg)
    if regime not in REGIMES:
        raise ValueError("invalid stress regime")
    if file_sha(checkpoint)!=cfg["v4_sha256"]:
        raise ValueError("V4 checksum mismatch")
    if out.exists():
        raise FileExistsError("cannot overwrite independent PR49 results")
    km=load_archived_km(kmfile)
    torch.set_num_threads(1)
    start=time.process_time()
    # Source simulator has no per-decision wall-clock hard SLA when the
    # generous CPU process guard is used. This is a model-quality experiment.
    subset=[c for c in cfg["cohort"] if c["regime"]==regime]
    groups=[]
    for item in subset:
        if time.process_time()-start>=cfg["resources"]["per_regime_process_seconds_max"]:
            raise TimeoutError("PR49 safe overall runner time exceeded")
        groups.append(evaluate_case(item,cfg,checkpoint,km))
        print("PR49_CASE",item["id"],{
            a:groups[-1]["arms"][a]["makespan_ticks"] for a in ARMS},
            flush=True)
    elapsed=time.process_time()-start
    report={
        "schema":"pr49-new-independent-high-compute-full-online-v1",
        "regime":regime,"protocol_sha256":file_sha(PROTOCOL),
        "source_v4_sha256":file_sha(checkpoint),
        "source_pr44_train_only_sha256":file_sha(kmfile),
        "gpu_training":0,"neural_training":0,
        "same_compute_safety_guard_for_both_v4":True,
        "groups":groups,
        "summary":{"regime":regime,"cpu_process_seconds":elapsed,
                   **summarize(groups),
                   "limits":"CPU hosted runner for reproducibility only; no GPU inference speed or real hardware benchmark"}
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print("PR49_REGIME_RESULT",json.dumps(report["summary"],ensure_ascii=False),flush=True)
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--pr44-result",type=Path,required=True)
    p.add_argument("--regime",choices=REGIMES,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    run(a.checkpoint,a.pr44_result,a.regime,a.out)


if __name__=="__main__":
    main()
