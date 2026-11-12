"""Two-source read-only replication of strict-budget H20 world-model value.

PR36 and PR37 use different independently preregistered scene seeds but
otherwise matching business fixture: jobs, robots, resource capacity,
keyed service jitter, H20/P3 and CP-SAT first-proposal selection.
The original-V4 progress objective is the same in both: maximize predicted
number of completed sample stages at horizon H20, tie first proposal.

This does not train/re-run models or make new factual trajectories.
It recomputes *paired per-scene* utility differences from checksum-verified
archived PR36/PR37 experiment JSONs and discloses implementation differences.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

SOURCE36="RHWM_PR33_SOURCE_BACKED_MPC_COST_20261112_V1"
SOURCE37="RHWM_PR34_POSTCANDIDATE_BUDGET_ENFORCEMENT_20261112_V1"
COMMON=("cp","rule","v4","scratch")
ARMS36={"cp":"cp_sat","rule":"rule_progress",
        "v4":"original_progress","scratch":"scratch_progress"}
ARMS37={"cp":"cp_sat","rule":"rule_h20",
        "v4":"original_v4_h20","scratch":"scratch_v4_h20"}


def mean(a):
    return sum(a)/len(a) if a else None


def descriptive_ci(samples,seed=1810,n=4000):
    if not samples:return None
    rng=random.Random(seed)
    size=len(samples)
    means=[mean([samples[rng.randrange(size)] for _ in range(size)])
           for _ in range(n)]
    means.sort()
    return [means[int(n*.025)],means[int(n*.975)]]


def normalize(doc,source):
    schema=SOURCE36 if source=="pr36" else SOURCE37
    if doc.get("protocol",doc.get("schema"))!=schema:
        raise ValueError("wrong source protocol "+source)
    if source=="pr36":
        if doc["summary"]["independent_groups"]!=24:
            raise ValueError("source cohort incomplete")
        mapping=ARMS36
    else:
        if doc["summary"]["total_independent_groups"]!=24:
            raise ValueError("source cohort incomplete")
        mapping=ARMS37
    rows=[]
    for group in doc["groups"]:
        arms={}
        for key,name in mapping.items():
            raw=group["arms"][name]
            if raw["job_metrics"]["censored"] or raw["safety_violations"]:
                raise AssertionError("invalid or incomplete actual scenario")
            tr=raw["decision_trace"] if source=="pr36" else raw["decision_records"]
            late=[d for d in tr
                  if (d.get("cpu_seconds",d.get("process_cpu_seconds",0))>0.5
                      and d.get("selected_index",0)!=0)]
            arms[key]={
                "makespan":raw["makespan_ticks"],
                "late_jobs":raw["job_metrics"]["late_jobs"],
                "resource_wait_ticks":raw["resource_wait_ticks"],
                "full_episode_cpu":raw["whole_episode_cpu_seconds"]
                if source=="pr36" else raw["full_episode_cpu_seconds"],
                "budget_applied_late_nonfirst_count":len(late),
            }
        rows.append({
            "source":source,"group":group["group"],"regime":group["regime"],
            "seed":group["seed"],"arms":arms,
        })
    return rows


def summarize(rows):
    per={}
    for regime in ("light","medium","heavy","shift"):
        sub=[g for g in rows if g["regime"]==regime]
        if len(sub)!=12:
            raise AssertionError("expected six scenes from EACH independent source")
        if {g["source"] for g in sub}!={"pr36","pr37"}:
            raise AssertionError("source cohort missing")
        armdata={}
        for arm in COMMON:
            values=[g["arms"][arm] for g in sub]
            armdata[arm]={
                "mean_makespan":mean([v["makespan"] for v in values]),
                "mean_late_jobs":mean([v["late_jobs"] for v in values]),
                "mean_full_cpu_seconds":mean([v["full_episode_cpu"] for v in values]),
            }
        comparisons={}
        for name,ref in (("v4_vs_cp","cp"),("v4_vs_rule","rule"),
                         ("scratch_vs_v4","v4"),("rule_vs_cp","cp")):
            left="scratch" if name=="scratch_vs_v4" else (
                "rule" if name=="rule_vs_cp" else "v4")
            diff=[g["arms"][left]["makespan"]-g["arms"][ref]["makespan"]
                  for g in sub]
            comparisons[name]={
                "per_source":{
                    cohort:mean([g["arms"][left]["makespan"]-
                                 g["arms"][ref]["makespan"]
                                 for g in sub if g["source"]==cohort])
                    for cohort in ("pr36","pr37")
                },
                "mean_paired_delta_tick":mean(diff),
                "wins":sum(x<0 for x in diff),
                "ties":sum(x==0 for x in diff),
                "losses":sum(x>0 for x in diff),
                "descriptive_group_bootstrap_ci95":descriptive_ci(
                    diff,seed=20261112+("v4_vs_cp","v4_vs_rule","scratch_vs_v4","rule_vs_cp").index(name)),
            }
        per[regime]={
            "groups":len(sub),
            "arms":armdata,
            "comparisons":comparisons,
        }
    return per


def run(doc36,doc37):
    a=normalize(doc36,"pr36")
    b=normalize(doc37,"pr37")
    rows=a+b
    seeds=[x["seed"] for x in rows]
    if len(rows)!=48 or len(set(seeds))!=48:
        raise AssertionError("cross-cohort scene seed collision")
    late=[(g["group"],name) for g in rows for name,arm in g["arms"].items()
          if arm["budget_applied_late_nonfirst_count"]]
    return {
        "schema":"rhwm-h20-pr36-pr37-two-cohort-replication-v1",
        "source_runs":[43147122344,43147634979],
        "source_artifacts":[13912451562,13911523921],
        "groups":rows,
        "summary":{
            "groups":len(rows),
            "per_regime":summarize(rows),
            "late_applied_nonfirst":late,
            "source_method_compatibility":"Both original V4 progress scoring H20/P3+CP-SAT same business fixture. PR36 enforces after-candidate CPU gate; PR37 adds final gate and artificial-clock regression. Not identical branch implementation, so treat as two independent source cohorts, not a pooled randomized trial.",
            "gpu":0,"new_simulation":0,"new_training":0,"new_model_inference":0,
            "limitations":"Independent synthetic scene seeds, 12 per regime; bootstrap intervals descriptive, not a proof of real-life performance or strict wall-time SLA",
        }
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source36",required=True,type=Path)
    p.add_argument("--source37",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError("write-once independent synthesis")
    result=run(json.loads(args.source36.read_text()),json.loads(args.source37.read_text()))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    lines={k:{
        "mean_makespan":{name:round(x["mean_makespan"],6)
                         for name,x in block["arms"].items()},
        "paired":{name:{p:v for p,v in comp.items() if p in
                         ("mean_paired_delta_tick","wins","ties","losses",
                          "descriptive_group_bootstrap_ci95","per_source")}
                  for name,comp in block["comparisons"].items()},
    } for k,block in result["summary"]["per_regime"].items()}
    print("H20_TWO_COHORT_REPLICATION",json.dumps({
        "groups":len(result["groups"]),
        "late_applied_nonfirst":result["summary"]["late_applied_nonfirst"],
        "regimes":lines},ensure_ascii=False))


if __name__=="__main__":
    main()
