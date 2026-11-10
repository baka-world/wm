"""Run real CPU Rule/V4 rollouts on previously frozen synthetic urgent cases.

These are already examined smoke/development scenes, never an independent test.
No new training, hyperparameter selection, future labels, or changed baseline.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import time
import torch

from experiments.world_model_integration.train_v4 import load_v4
from experiments.world_model_integration.urgent_workflow_feasibility.core import (
    PROTOCOL, public_spec,
)
from .core import run_case

EXPECTED_V4_SHA256="60c8a6c3d832da3e6519072688ea099f64ae120ba19d3bd2ec705f7f1a11ac0c"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _mean(xs):
    return statistics.mean(xs) if xs else None


def _numeric_pair(p4,p5):
    a=p4["metrics"];b=p5["metrics"]
    return {
        "urgent_ontime_delta_v4_minus_rule":
            b["urgent"]["on_time_rate"]-a["urgent"]["on_time_rate"],
        "urgent_late_delta_v4_minus_rule":
            b["urgent"]["late_mean_lower_bound"]-a["urgent"]["late_mean_lower_bound"],
        "urgent_incomplete_delta_v4_minus_rule":
            b["urgent"]["incomplete_rate"]-a["urgent"]["incomplete_rate"],
        "routine_flow_delta_v4_minus_rule":
            b["routine"]["restricted_mean_flow"]-a["routine"]["restricted_mean_flow"],
        "all_flow_delta_v4_minus_rule":
            b["all"]["restricted_mean_flow"]-a["all"]["restricted_mean_flow"],
    }


def _strict_shared_decisions(rule,model):
    a={(r["tick"],r["observation_sha256"]):r for r in rule["decisions"]}
    b={(r["tick"],r["observation_sha256"]):r for r in model["decisions"]}
    shared=sorted(set(a)&set(b))
    out={"shared_observable_decisions":len(shared),
         "shared_multiple_candidate":0,"candidate_set_mismatch":0,
         "score_changed":0,"rank_changed":0,"action_changed":0,
         "decision_rows":[]}
    for key in shared:
        x,y=a[key],b[key]
        if x["candidate_count"]<2:continue
        out["shared_multiple_candidate"]+=1
        ids=lambda rows:[[ (item["robot_id"],item["sample_id"]) for item in p["matching"]]
                          for p in rows["plans"]]
        if ids(x)!=ids(y):
            out["candidate_set_mismatch"]+=1
            continue
        scores_a=[p["score"] for p in x["plans"]]
        scores_b=[p["score"] for p in y["plans"]]
        changed=scores_a!=scores_b
        ranked=x["rank"]!=y["rank"]
        action=x["selected"]!=y["selected"]
        out["score_changed"]+=int(changed)
        out["rank_changed"]+=int(ranked)
        out["action_changed"]+=int(action)
        if changed or ranked or action:
            out["decision_rows"].append({
                "tick":key[0],"observation_sha256":key[1],
                "candidate_count":x["candidate_count"],
                "rule_scores":scores_a,"v4_scores":scores_b,
                "rule_rank":x["rank"],"v4_rank":y["rank"],
                "rule_selected":x["selected"],"v4_selected":y["selected"]})
    return out


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--group",choices=("smoke","development"),required=True)
    parser.add_argument("--cpu-budget-seconds",type=float,default=900.)
    args=parser.parse_args()
    if args.out.exists():
        raise FileExistsError("integration evidence is write-once: "+str(args.out))
    if sha(args.checkpoint)!=EXPECTED_V4_SHA256:
        raise ValueError("frozen V4 checkpoint SHA mismatch")
    if torch.version.cuda is not None:
        raise AssertionError("CPU-only torch is required")
    torch.set_num_threads(1)
    _,meta=load_v4(args.checkpoint,experimental=True)
    if meta["model_version"]!="business-sets-session-v4-data":
        raise AssertionError("wrong frozen model version")
    p=json.loads(PROTOCOL.read_text())
    seeds=(list(enumerate(p["smoke_seeds"])) if args.group=="smoke" else
           list(enumerate(p["development_seeds"],start=len(p["smoke_seeds"]))))
    start=time.process_time()
    cases=[];pairs=[]
    for index,seed in seeds:
        spec=public_spec(index,seed,p["scenario_max_ticks"])
        pair={}
        for arm in ("P4","P5"):
            pair[arm]=run_case(spec,p,arm,args.checkpoint if arm=="P5" else None)
            print("CASE",args.group,index,seed,arm,
                  "urgent_ontime",pair[arm]["metrics"]["urgent"]["on_time_rate"],
                  "urgent_late",pair[arm]["metrics"]["urgent"]["late_mean_lower_bound"],
                  "calls",pair[arm]["query_calls"],
                  "cpu",round(pair[arm]["cpu_seconds"],3),flush=True)
            if time.process_time()-start>args.cpu_budget_seconds:
                raise TimeoutError("frozen CPU integration budget exceeded")
        differences=_strict_shared_decisions(pair["P4"],pair["P5"])
        if differences["candidate_set_mismatch"]:
            raise AssertionError("different CP-SAT candidate sets for the same observable state")
        pairs.append({"seed":seed,"index":index,
                      "deltas":_numeric_pair(pair["P4"],pair["P5"]),
                      "same_state":differences})
        cases.extend(pair.values())
    if not any(x["query_calls"]>0 for x in cases if x["arm"]=="P5"):
        raise AssertionError("V4 never made a service-event inference")
    result={"source_class":"previously_examined_synthetic_smoke_development",
            "independent_test":False,"new_training":False,"gpu":0,
            "checkpoint_sha256":sha(args.checkpoint),
            "protocol_sha256":sha(PROTOCOL),
            "group":args.group,"scenarios":len(seeds),
            "case_count":len(cases),
            "cpu_process_seconds":time.process_time()-start,
            "metrics":{},"paired":pairs}
    for arm in ("P4","P5"):
        rows=[x for x in cases if x["arm"]==arm]
        result["metrics"][arm]={
            "urgent_ontime_rate_mean":_mean([x["metrics"]["urgent"]["on_time_rate"] for x in rows]),
            "urgent_late_mean_lower_bound":_mean([x["metrics"]["urgent"]["late_mean_lower_bound"] for x in rows]),
            "urgent_unfinished_rate_mean":_mean([x["metrics"]["urgent"]["incomplete_rate"] for x in rows]),
            "routine_restricted_flow_mean":_mean([x["metrics"]["routine"]["restricted_mean_flow"] for x in rows]),
            "all_restricted_flow_mean":_mean([x["metrics"]["all"]["restricted_mean_flow"] for x in rows]),
            "total_model_query_calls":sum(x["query_calls"] for x in rows),
            "total_rollout_steps":sum(x["rollout_steps"] for x in rows),
            "total_decisions":sum(x["decision_count"] for x in rows),
            "total_prediction_cpu_seconds":sum(x["prediction_cpu_seconds"] for x in rows),
            "fallback_reason_counts":{
                k:sum(x["fallback_reasons"].get(k,0) for x in rows)
                for k in sorted({k for x in rows for k in x["fallback_reasons"]})},
            "resource_violations":sum(x["resource_violations"] for x in rows),
            "motion_violations":sum(x["motion_violations"] for x in rows),
            "lifecycle_violations":sum(x["lifecycle_violations"] for x in rows),
        }
    args.out.mkdir(parents=True)
    (args.out/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    with gzip.GzipFile(filename="",mode="wb",fileobj=(args.out/"cases.json.gz").open("wb"),mtime=0) as gz:
        gz.write(json.dumps(cases,ensure_ascii=False,sort_keys=True).encode())
    digests={f.name:sha(f) for f in args.out.iterdir() if f.is_file()}
    (args.out/"sha256.json").write_text(json.dumps(digests,indent=2)+"\n")
    print("SUMMARY",args.group,json.dumps({"metric":result["metrics"],
           "shared_decisions":sum(x["same_state"]["shared_multiple_candidate"] for x in pairs),
           "action_changes":sum(x["same_state"]["action_changed"] for x in pairs),
           "cpu_process_seconds":result["cpu_process_seconds"]},ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
