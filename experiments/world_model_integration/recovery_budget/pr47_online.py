"""PR47: equal-budget CPU ablation of original vs recovery-enhanced V4-H20.

Real original BusinessSimulator, original PR30 candidate/ranking/fallback;
reuses PR46's tested online policy. New heldout seeds. No new weights.
The per-decision CPU-process budget is increased for BOTH V4 models.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import argparse
import copy
import json
import time

import torch

from experiments.world_model_integration.recovery_shift.pr46_online import one
from experiments.world_model_integration.recovery_shift.pr46_shared import PROTOCOL as PR46_PROTOCOL
from experiments.world_model_integration.health_recovery.pr44_observed_recovery import PROTOCOL as PR44_PROTOCOL
from experiments.world_model_integration.recovery_integration.pr45_recovery_world import PROTOCOL as PR45_PROTOCOL
from experiments.world_model_integration.system_world_model.evaluate_horizons import file_sha
from experiments.world_model_integration.recovery_shift.pr46_shared import frozen_setup as original_frozen_setup

PROTOCOL=Path(__file__).with_name("pr47_protocol.json")
ARMS=("cp_sat","v4_original_05","v4_recovery_05","v4_original_15","v4_recovery_15")
MECHANISMS=("within_support","extended_outage")
MAPPED={
    "cp_sat":("cp_sat",.5),
    "v4_original_05":("v4_original",.5),
    "v4_recovery_05":("v4_public_km_recovery",.5),
    "v4_original_15":("v4_original",1.5),
    "v4_recovery_15":("v4_public_km_recovery",1.5),
}


def validate(config):
    if config.get("protocol")!="PR47_EQUAL_CPU_BUDGET_RECOVERY_H20_20261112_V1":
        raise ValueError("PR47 identity changed")
    if config["arms"]!=list(ARMS) or config["cpu_budgets_seconds"]!=[.5,1.5]:
        raise ValueError("arm or CPU budget changed")
    if (config["online"]["particles"]!=3 or config["online"]["horizon"]!=20
        or config["online"]["candidate_limit"]!=4
        or config["online"]["decision_cpu_budget_seconds"]!=.5):
        raise ValueError("forecast or candidate contract changed")
    if not config["no_weight_or_threshold_tuning"]:
        raise ValueError("cannot tune scores on test outcomes")
    if config["resource_limits"]["gpu"] or config["resource_limits"]["new_training"]:
        raise ValueError("GPU or training forbidden")
    if len(config["cohort"])!=24 or len(set(g["seed"] for g in config["cohort"]))!=24:
        raise ValueError("not 24 new unique scenarios")
    old=json.loads(PR46_PROTOCOL.read_text())
    earlier={s["seed"] for track in old["cohort"].values() for s in track}
    before=json.loads(PR44_PROTOCOL.read_text())
    earlier.update(s["seed"] for split in before["cohort"].values() for s in split)
    earlier.update(7100009+s*10000+i*271 for s in range(3) for i in range(4))
    if {g["seed"] for g in config["cohort"]} & earlier:
        raise ValueError("PR47 cohort overlaps a prior test cohort")
    if (config["hidden_environment_only"]!=old["hidden_environment_only"] or
        config["online"]["arms"]!=old["online"]["arms"]):
        raise ValueError("real-world generator or policy methods differ from PR46")
    for mech in MECHANISMS:
        for station in ("B0","B1","C"):
            xs=[s for s in config["cohort"] if s["mechanism"]==mech and s["station_id"]==station]
            if len(xs)!=4:raise ValueError("precommitted mechanism/station cohort changed")
            if any(s["track"]!="online" for s in xs):
                raise ValueError("PR47 only evaluates the full online cohort")
    return True


def run_arm(spec, cfg, label, checkpoint, km):
    method,budget=MAPPED[label]
    public=copy.copy(cfg)
    public["online"]={**cfg["online"],"decision_cpu_budget_seconds":budget}
    result=one(spec,public,method,checkpoint,km)
    result["budget_seconds"]=budget
    result["policy_id"]=label
    if result["safety_violations"]:
        raise AssertionError("safety violation during actual online episode")
    return result


def scenario(spec,cfg,checkpoint,km):
    # Alternate the presentation/test order of the V4 variants, so warming
    # and same-process CPU load do not always privilege one model.
    alternate=[
        "v4_original_05","v4_recovery_05",
        "v4_original_15","v4_recovery_15"
    ]
    offset=spec["seed"]%4
    order=["cp_sat"]+alternate[offset:]+alternate[:offset]
    results={arm:run_arm(spec,cfg,arm,checkpoint,km) for arm in order}
    # Every arm must have seen the exact same keyed factual world if it
    # chooses the same actions. If not, environment RNG would be leaking.
    originals={r["factual_event_sha256"] for r in results.values()}
    if len(originals)==1 and len({r["makespan_ticks"] for r in results.values()})!=1:
        raise AssertionError("identical factual events yielded different completion times")
    return {
        "id":spec["id"],"mechanism":spec["mechanism"],"station_id":spec["station_id"],
        "seed":spec["seed"],"evaluation_order":order,
        "arms":{arm:results[arm] for arm in ARMS}
    }


def avg(values):
    return sum(values)/len(values) if values else None


def summarized(groups):
    out={}
    for mech in MECHANISMS:
        g=[case for case in groups if case["mechanism"]==mech]
        data={}
        for arm in ARMS:
            episodes=[case["arms"][arm] for case in g]
            data[arm]={
                "groups":len(episodes),
                "mean_makespan":avg([x["makespan_ticks"] for x in episodes]),
                "mean_late_jobs":avg([x["job_metrics"]["late_jobs"] for x in episodes]),
                "mean_capacity_wait_ticks":avg([x["resource_wait_ticks"] for x in episodes]),
                "mean_total_episode_cpu_seconds":avg([x["episode_cpu_seconds"] for x in episodes]),
                "mean_decision_cpu_seconds":avg([x["decision_cpu_seconds"] for x in episodes]),
                "completed_episodes":sum(not x["job_metrics"]["censored"] for x in episodes),
                "candidate_decisions":sum(x["multiple_candidate_calls"] for x in episodes),
                "changed_plan_calls":sum(x["plan_flips"] for x in episodes),
                "forecast_calls":sum(x["prediction_calls"] for x in episodes),
                "budget_exceeded":sum(x["budget_exceeded"] for x in episodes),
                "fallbacks":sum(x["fallbacks"] for x in episodes),
                "prediction_errors":sum(x["prediction_errors"] for x in episodes),
                "sampled_repair_events":sum(x["sampled_repair_events"] for x in episodes),
                "safety_violations":sum(x["safety_violations"] for x in episodes),
            }
        contrasts={}
        comparisons=[
            ("v4_recovery_05","v4_original_05"),
            ("v4_recovery_15","v4_original_15"),
            ("v4_original_15","v4_original_05"),
            ("v4_recovery_15","v4_recovery_05"),
            ("v4_original_15","cp_sat"),
            ("v4_recovery_15","cp_sat"),
        ]
        for active,reference in comparisons:
            delta=[case["arms"][active]["makespan_ticks"]-case["arms"][reference]["makespan_ticks"]
                   for case in g]
            late=[case["arms"][active]["job_metrics"]["late_jobs"]
                  -case["arms"][reference]["job_metrics"]["late_jobs"] for case in g]
            contrasts[active+"_minus_"+reference]={
                "mean_makespan_delta":avg(delta),
                "wins":sum(x<0 for x in delta),
                "ties":sum(x==0 for x in delta),
                "losses":sum(x>0 for x in delta),
                "mean_late_job_delta":avg(late),
            }
        out[mech]={"groups":len(g),"arms":data,"paired":contrasts}
    return out


def run(checkpoint, pr44_result, output):
    cfg=json.loads(PROTOCOL.read_text())
    validate(cfg)
    # Reuse PR46's source/checkpoint validation without editing its protocol.
    orig=json.loads(PR46_PROTOCOL.read_text())
    if file_sha(checkpoint)!=cfg["frozen_v4_sha256"] or file_sha(checkpoint)!=orig["frozen_original_v4_sha256"]:
        raise ValueError("changed original V4")
    from experiments.world_model_integration.recovery_integration.pr45_recovery_world import load_archived_km
    km=load_archived_km(pr44_result)
    if output.exists():raise FileExistsError("primary score evidence must be write-once")
    torch.set_num_threads(1)
    start=time.process_time()
    groups=[]
    for spec in cfg["cohort"]:
        if time.process_time()-start>=cfg["resource_limits"]["allow_cpu_process_seconds"]:
            raise TimeoutError("precommitted PR47 CPU process limit exceeded")
        groups.append(scenario(spec,cfg,checkpoint,km))
    elapsed=time.process_time()-start
    if elapsed>=cfg["resource_limits"]["allow_cpu_process_seconds"]:
        raise TimeoutError("CPU process limit exceeded on final case")
    result={
        "schema":"pr47-equal-budget-closed-loop-new-cohort-v1",
        "protocol_sha256":file_sha(PROTOCOL),
        "pr44_model_source_sha256":file_sha(pr44_result),
        "v4_checkpoint_sha256":file_sha(checkpoint),
        "no_new_training":True,"gpu":0,
        "method":"PR30 exact completion-score/CP-SAT score and 0.5 vs 1.5s both-arm budgets",
        "scenes":groups,
        "summary":{"scenes":len(groups),"total_episodes":len(groups)*len(ARMS),
                   "by_mechanism":summarized(groups),
                   "total_cpu_process_seconds":elapsed,
                   "scope":"new independent seeds, same synthetic generation mechanics, true full-online episodes"}
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print("PR47_EQUAL_BUDGET_RESULT",json.dumps(result["summary"],ensure_ascii=False))
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",required=True,type=Path)
    ap.add_argument("--pr44-result",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    args=ap.parse_args()
    run(args.checkpoint,args.pr44_result,args.out)


if __name__=="__main__":
    main()
