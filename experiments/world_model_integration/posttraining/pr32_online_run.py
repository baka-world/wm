"""PR32 strict full-online comparison of frozen V4 vs trained V4.

Everything executed through existing PR30 OnlineHorizonPolicy and
BusinessSimulator. Candidate rule, common exogenous noise, and CPU budgets are
identical; the only changed dimension between V4 arms is pretrained/fine-tuned
weights. PR31 Validation-selected scratch model is the primary candidate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from experiments.world_model_integration.online_horizon.pr30_run import one
from experiments.world_model_integration.posttraining.pr31_dataset import PROTOCOL as TRAIN_PROTOCOL
from experiments.world_model_integration.horizon_generalization.pr29_evaluate import _scenario_seed_set
from experiments.world_model_integration.posttraining.repro_environment import sha
from experiments.world_model_integration.train_v4 import load_v4

PROTOCOL=Path(__file__).with_name("pr32_online_protocol.json")
ARMS=("cp_sat","rule_h20","original_v4_h20","warmstart_v4_h20","scratch_v4_h20")


def _no_seen_scenarios(proto):
    former=json.loads(TRAIN_PROTOCOL.read_text())
    new=[g["seed"] for g in proto["groups"]]
    if len(new)!=24 or len(set(new))!=24:raise AssertionError("wrong frozen online scene grouping")
    previous={x["seed"] for section in former["groups"].values() for x in section}
    previous.update(x["seed"] for x in former["decision_test_groups"])
    if set(new)&previous:raise AssertionError("PR31 development/independent test seed reused")
    from experiments.world_model_integration.online_horizon.pr30_run import PROTOCOL as PR30_PROTOCOL
    old=json.loads(PR30_PROTOCOL.read_text())
    if set(new)&{x["seed"] for x in old["groups"]}:
        raise AssertionError("previous online scene repeated")
    _scenario_seed_set({"scene_groups":proto["groups"]})


def _config(proto):
    return {
        "max_ticks":proto["scenario_max_ticks"],
        "candidate_limit":proto["candidate_limit"],
        "particles":proto["particles"],
        "decision_cpu_budget_seconds":proto["decision_cpu_budget_seconds"],
    }


def eval_one(spec,proto,checkpoint_by_arm):
    output={}
    for arm in ARMS:
        if arm=="cp_sat":source="cp_sat"
        elif arm=="rule_h20":source="rule_h20"
        else:source="v4_h20"
        checkpoint=checkpoint_by_arm.get(arm,checkpoint_by_arm["original_v4_h20"])
        factual=one(spec,_config(proto),source,checkpoint)
        factual["arm"]=arm
        if factual["safety_violations"] or factual["prediction_errors"]:
            raise AssertionError("real online state or inference invariant violated")
        output[arm]=factual
    return {"group":spec["id"],"regime":spec["regime"],"seed":spec["seed"],
            "arms":output}


def _mean(vals):
    return sum(vals)/len(vals) if vals else None


def grouped_summary(groups):
    out={}
    for regime in sorted({x["regime"] for x in groups}):
        sub=[x for x in groups if x["regime"]==regime]
        arms={}
        for arm in ARMS:
            vals=[s["arms"][arm] for s in sub]
            arms[arm]={
                "groups":len(vals),
                "mean_makespan":_mean([v["makespan_ticks"] for v in vals]),
                "mean_late_jobs":_mean([v["job_metrics"]["late_jobs"] for v in vals]),
                "mean_resource_wait_ticks":_mean([v["resource_wait_ticks"] for v in vals]),
                "completed":sum(not v["job_metrics"]["censored"] for v in vals),
                "model_decisions_changed":sum(v["changed_plan_calls"] for v in vals),
                "fallbacks":sum(v["fallbacks"] for v in vals),
                "decision_budget_overruns":sum(v["decision_budget_exceeded"] for v in vals),
                "mean_total_cpu_seconds":_mean([v["full_episode_cpu_seconds"] for v in vals]),
                "mean_model_init_cpu_seconds":_mean([v["model_init_cpu_seconds"] for v in vals]),
            }
        comparisons={}
        for variant in ARMS[1:]:
            diffs=[s["arms"][variant]["makespan_ticks"]-
                   s["arms"]["original_v4_h20"]["makespan_ticks"] for s in sub]
            baseline=[s["arms"][variant]["makespan_ticks"]-
                   s["arms"]["cp_sat"]["makespan_ticks"] for s in sub]
            comparisons[variant]={
                "mean_delta_vs_original_v4":_mean(diffs),
                "wins_vs_original_v4":sum(x<0 for x in diffs),
                "ties_vs_original_v4":sum(x==0 for x in diffs),
                "losses_vs_original_v4":sum(x>0 for x in diffs),
                "mean_delta_vs_cp_sat":_mean(baseline),
                "wins_vs_cp_sat":sum(x<0 for x in baseline),
                "ties_vs_cp_sat":sum(x==0 for x in baseline),
                "losses_vs_cp_sat":sum(x>0 for x in baseline),
            }
        out[regime]={"groups":len(sub),"arms":arms,"paired":comparisons}
    return out


def run(original,warm,scratch,selection_path,*,protocol=PROTOCOL):
    proto=json.loads(Path(protocol).read_text())
    if proto["protocol"]!="RHWM_PR32_FROZEN_POSTTRAINED_H20_ONLINE_20261112_V1":
        raise ValueError("unexpected PR32 protocol")
    if proto["arms"]!=list(ARMS):raise ValueError("model arms changed")
    _no_seen_scenarios(proto)
    selection=json.loads(Path(selection_path).read_text())
    if (selection["best_by_validation_brier"]!="scratch_v4" or
        sha(original)!=proto["source"]["original_v4_sha256"]):
        raise ValueError("trained checkpoint provenance or preregistered primary arm mismatch")
    expected=selection["arms"]
    if (sha(warm)!=expected["warmstart_v4"]["files"]["best"]["sha256"] or
        sha(scratch)!=expected["scratch_v4"]["files"]["best"]["sha256"]):
        raise ValueError("new source weights differ from archived PR31 validation-selected assets")
    for name,path in (("warmstart_v4",warm),("scratch_v4",scratch)):
        _,metadata=load_v4(path,experimental=True)
        if metadata["training_arm"]!=name or metadata["training_run_protocol"]!=selection["protocol"]:
            raise ValueError("trained model interface/provenance mismatch")
    checkpoint_by_arm={
        "original_v4_h20":original,
        "warmstart_v4_h20":warm,
        "scratch_v4_h20":scratch,
    }
    from torch import set_num_threads
    set_num_threads(1)
    start=time.process_time()
    groups=[]
    for spec in proto["groups"]:
        if time.process_time()-start>proto["budget"]["total_cpu_seconds"]:
            raise TimeoutError("scenario budget exceeded before start")
        groups.append(eval_one(spec,proto,checkpoint_by_arm))
    elapsed=time.process_time()-start
    if elapsed>proto["budget"]["total_cpu_seconds"]:
        raise TimeoutError("total CPU budget exceeded")
    return {
        "protocol":proto["protocol"],
        "protocol_sha256":sha(protocol),
        "pr31_validation_selection_sha256":sha(selection_path),
        "checkpoint_sha256s":{k:sha(v) for k,v in checkpoint_by_arm.items()},
        "groups":groups,
        "summary":{
            "total_independent_groups":len(groups),
            "by_regime":grouped_summary(groups),
            "total_process_cpu_seconds":elapsed,"gpu":0,"new_training":0,
            "primary_model":"scratch_v4_h20 (selected on PR31 validation)",
            "method":"reuse existing PR30 online CP-SAT/Rule/RHWM without altering model architecture, decision scoring or business mechanics",
        },
    }


def main():
    parser=argparse.ArgumentParser()
    for name in ("original","warm","scratch","selection","out"):
        parser.add_argument("--"+name,required=True,type=Path)
    a=parser.parse_args()
    if a.out.exists():raise FileExistsError("test evidence is write-once")
    r=run(a.original,a.warm,a.scratch,a.selection)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(r,indent=2,ensure_ascii=False)+"\n")
    print("PR32_POSTTRAINING_FULL_ONLINE",json.dumps(r["summary"],ensure_ascii=False))


if __name__=="__main__":
    main()
