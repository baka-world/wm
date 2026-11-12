"""Real independent online evaluation after PR33 post-candidate CPU gate fix.

Reuses original BusinessSimulator/BusinessCPSAT and the five PR32 policies.
Only change to execution: the child-branch PR30 planner checks consumed CPU
AFTER *each* candidate forecast and falls back to CP-SAT on overrun.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from experiments.world_model_integration.posttraining.pr32_online_run import (
    ARMS,eval_one,grouped_summary,_no_seen_scenarios,
)
from experiments.world_model_integration.posttraining.repro_environment import sha
from experiments.world_model_integration.posttraining.pr31_dataset import PROTOCOL as PR31_PROTOCOL
from experiments.world_model_integration.posttraining.pr32_online_run import PROTOCOL as PR32_PROTOCOL
from experiments.world_model_integration.train_v4 import load_v4

PROTOCOL=Path(__file__).with_name("pr34_protocol.json")


def _verify_new_scenes(proto):
    if (proto["protocol"]!="RHWM_PR34_POSTCANDIDATE_BUDGET_ENFORCEMENT_20261112_V1"
            or proto["arms"]!=list(ARMS)
            or len(proto["groups"])!=24
            or proto["decision_cpu_budget_seconds"]!=0.5):
        raise ValueError("PR34 experimental protocol changed")
    prior=json.loads(PR32_PROTOCOL.read_text())
    old={x["seed"] for x in prior["groups"]}
    if {x["seed"] for x in proto["groups"]}&old:
        raise AssertionError("PR32 test scene leaked into PR34")
    _no_seen_scenarios(proto)
    for regime in ("light","medium","heavy","shift"):
        if sum(x["regime"]==regime for x in proto["groups"])!=6:
            raise AssertionError("wrong frozen count: "+regime)


def run(original,warm,scratch,selection_path,*,protocol=PROTOCOL):
    cfg=json.loads(Path(protocol).read_text())
    _verify_new_scenes(cfg)
    selection=json.loads(Path(selection_path).read_text())
    if (sha(original)!=cfg["original_checkpoint_sha256"] or
        selection["best_by_validation_brier"]!="scratch_v4"):
        raise ValueError("frozen V4/checkpoint selection mismatch")
    for name,path in (("warmstart_v4",warm),("scratch_v4",scratch)):
        expected=selection["arms"][name]["files"]["best"]["sha256"]
        if sha(path)!=expected:
            raise ValueError("new model not the original PR31 selected checkpoint")
        _,metadata=load_v4(path,experimental=True)
        if metadata["training_arm"]!=name:
            raise ValueError("wrong checkpoint training arm")
    checkpoints={
        "original_v4_h20":original,
        "warmstart_v4_h20":warm,
        "scratch_v4_h20":scratch,
    }
    from torch import set_num_threads
    set_num_threads(1)
    started=time.process_time()
    outputs=[]
    observed_late={}
    violations=[]
    for spec in cfg["groups"]:
        if time.process_time()-started>cfg["sample_and_budget"]["cpu_process_total_cap_seconds"]:
            raise TimeoutError("PR34 precommitted CPU budget exceeded before next case")
        result=eval_one(spec,cfg,checkpoints)
        for arm in ARMS:
            record=result["arms"][arm]
            timed=[
                d for d in record["decision_records"]
                if d.get("process_cpu_seconds",0)>cfg["decision_cpu_budget_seconds"]
            ]
            wrong=[
                d for d in timed if d.get("selected_index",0)!=0]
            if wrong:
                violations.append({"group":spec["id"],"arm":arm,"decisions":wrong})
            observed_late[arm]=observed_late.get(arm,0)+len(timed)
            # Enforcing no over-budget non-first action is a hard gate for
            # result validity. The nonpreemptive call may *return* late; we
            # explicitly distinguish that from accepting a late re-rank.
            if record["safety_violations"] or record["prediction_errors"]:
                raise AssertionError("invalid real execution")
        outputs.append(result)
    if violations:raise AssertionError(
        "accepted over-budget model reranking: "+str(violations[:2]))
    elapsed=time.process_time()-started
    if elapsed>cfg["sample_and_budget"]["cpu_process_total_cap_seconds"]:
        raise TimeoutError("PR34 precommitted CPU budget exceeded")
    return {
        "schema":cfg["protocol"],
        "protocol_sha256":sha(protocol),
        "pr31_validation_selection_sha256":sha(selection_path),
        "checkpoints_sha256":{k:sha(v) for k,v in checkpoints.items()},
        "groups":outputs,
        "summary":{
            "total_independent_groups":len(outputs),
            "by_regime":grouped_summary(outputs),
            "measured_late_cpu_decisions_by_arm":observed_late,
            "applied_after_budget_nonfirst_actions":len(violations),
            "cpu_process_seconds":elapsed,
            "gpu":0,"new_training":0,
            "interpretation":"Timely decision gating verified; actual CPU duration may still exceed 0.5s because individual forecasts cannot be interrupted",
        },
    }


def main():
    p=argparse.ArgumentParser()
    for name in ("original","warm","scratch","selection","out"):
        p.add_argument("--"+name,required=True,type=Path)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError("independent result write-once")
    result=run(a.original,a.warm,a.scratch,a.selection)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    summary={
        "groups":result["summary"]["total_independent_groups"],
        "cpu":result["summary"]["cpu_process_seconds"],
        "measured_overruns":result["summary"]["measured_late_cpu_decisions_by_arm"],
        "applied_violations":result["summary"]["applied_after_budget_nonfirst_actions"],
        "by_regime":{
            reg:{
                "arms":{arm:{
                    "makespan":a["mean_makespan"],
                    "late_jobs":a["mean_late_jobs"],
                    "changed":a["model_decisions_changed"],
                    "fallbacks":a["fallbacks"],
                    "reported_overrun":a["decision_budget_overruns"],
                    "cpu":a["mean_total_cpu_seconds"]}
                    for arm,a in cohort["arms"].items()},
                "paired":{arm:{
                    "delta_vs_original":v["mean_delta_vs_original_v4"],
                    "wins":v["wins_vs_original_v4"],
                    "ties":v["ties_vs_original_v4"],
                    "losses":v["losses_vs_original_v4"]}
                    for arm,v in cohort["paired"].items()}
            }for reg,cohort in result["summary"]["by_regime"].items()},
    }
    print("PR34_BUDGET_CORRECTED_ONLINE_RESULT",json.dumps(summary,ensure_ascii=False))


if __name__=="__main__":
    main()
