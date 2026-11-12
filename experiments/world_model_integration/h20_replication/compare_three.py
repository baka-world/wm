"""Three-source read-only extension: PR36 + PR37 + PR38 H20 original V4.

All three source cohorts were independently frozen and executed in CI before
this descriptive synthesis. Keep the original two-source 48-scene artifact
unchanged. This extension consumes only verified source JSON and does not
call the model or modify dispatch/simulator/training data.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .compare import mean,normalize,COMMON

SCHEMA38="RHWM_PR35_STRICT_POST_CANDIDATE_H20_BUDGET_20261112_V1"
NAMES38={
    "cp":"cp_sat","rule":"rule_h20",
    "v4":"original_v4_h20","scratch":"scratch_v4_h20",
}


def parse38(doc):
    if (doc.get("schema")!=SCHEMA38 or
        doc.get("summary",{}).get("independent_groups")!=24):
        raise ValueError("third source checksum-verified schema mismatch")
    rows=[]
    for group in doc["groups"]:
        arms={}
        for key,name in NAMES38.items():
            raw=group["arms"][name]
            if raw["safety_violations"] or raw["job_metrics"]["censored"]:
                raise AssertionError("third cohort has incomplete/invalid episode")
            recorded=raw["decision_records"]
            late=[d for d in recorded
                  if d.get("selected_index",0)!=0 and
                     d.get("process_cpu_seconds",0)>=.5]
            if late or raw["strict_budget_audit"]["model_changes_after_cpu_cap"]:
                raise AssertionError("third cohort accepted late model reranking")
            arms[key]={
                "makespan":raw["makespan_ticks"],
                "late_jobs":raw["job_metrics"]["late_jobs"],
                "resource_wait_ticks":raw["resource_wait_ticks"],
                "full_episode_cpu":raw["full_episode_cpu_seconds"],
                "budget_applied_late_nonfirst_count":len(late),
            }
        rows.append({
            "source":"pr38","group":group["group"],
            "regime":group["regime"],"seed":group["seed"],
            "arms":arms,
        })
    return rows


def stratified_ci(groups,left,right,*,seed=20261112,reps=4000):
    rng=random.Random(seed)
    cohort={source:[g["arms"][left]["makespan"]-g["arms"][right]["makespan"]
                    for g in groups if g["source"]==source]
            for source in ("pr36","pr37","pr38")}
    if any(len(v)!=6 for v in cohort.values()):
        raise AssertionError("three source cohorts must each have six independent scenes")
    samples=[]
    for _ in range(reps):
        draw=[sample[rng.randrange(len(sample))]
              for sample in cohort.values() for _ in range(6)]
        samples.append(mean(draw))
    samples.sort()
    return [samples[int(reps*.025)],samples[int(reps*.975)]]


def run(a,b,c):
    groups=normalize(a,"pr36")+normalize(b,"pr37")+parse38(c)
    if len(groups)!=72 or len({g["seed"] for g in groups})!=72:
        raise AssertionError("duplicate seed across three sources")
    bad=[(g["source"],g["group"],name)
         for g in groups for name,r in g["arms"].items()
         if r["budget_applied_late_nonfirst_count"]]
    if bad:raise AssertionError("source cohort accepted overdeadline reranking")
    blocks={}
    for regime in ("light","medium","heavy","shift"):
        rows=[x for x in groups if x["regime"]==regime]
        if (len(rows)!=18 or
            any(sum(g["source"]==name for g in rows)!=6
                for name in ("pr36","pr37","pr38"))):
            raise AssertionError("unbalanced or incomplete three-way cohort")
        results={}
        for arm in COMMON:
            results[arm]={
                "mean_makespan":mean([g["arms"][arm]["makespan"] for g in rows]),
                "mean_late_jobs":mean([g["arms"][arm]["late_jobs"] for g in rows]),
                "mean_cpu_seconds":mean([g["arms"][arm]["full_episode_cpu"] for g in rows]),
            }
        comparison={}
        for key,left,right in (
            ("v4_vs_cp","v4","cp"),
            ("v4_vs_rule","v4","rule"),
            ("scratch_vs_v4","scratch","v4"),
            ("rule_vs_cp","rule","cp")):
            diff=[x["arms"][left]["makespan"]-x["arms"][right]["makespan"]
                  for x in rows]
            comparison[key]={
                "mean_paired_delta_tick":mean(diff),
                "wins":sum(v<0 for v in diff),
                "ties":sum(v==0 for v in diff),
                "losses":sum(v>0 for v in diff),
                "by_source":{
                    name:mean([g["arms"][left]["makespan"]-
                               g["arms"][right]["makespan"]
                               for g in rows if g["source"]==name])
                    for name in ("pr36","pr37","pr38")},
                "stratified_scenario_bootstrap95":stratified_ci(
                    rows,left,right,seed=20261112+
                    ("v4_vs_cp","v4_vs_rule","scratch_vs_v4","rule_vs_cp").index(key)),
            }
        blocks[regime]={
            "groups":18,
            "means":results,
            "paired":comparison,
        }
    return {
        "schema":"rhwm-h20-three-independent-cohorts-v1",
        "input_runs":[43147122344,43147634979,43148139899],
        "input_artifacts":[13912451562,13911523921,13912587157],
        "groups":groups,
        "summary":{
            "groups":len(groups),
            "by_regime":blocks,
            "late_model_reranking_accepted":0,
            "gpu":0,"new_training":0,"new_trajectories":0,"new_inference":0,
            "scope":"Three independently preregistered source cohorts, 6 scenes per regime per source. Shared H20/P3 V4-progress score and CP-SAT; different code branches for the postcandidate CPU gate. This is a retrospective descriptive synthesis, not one pre-registered 72-scene randomized trial.",
        }
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source36",required=True,type=Path)
    p.add_argument("--source37",required=True,type=Path)
    p.add_argument("--source38",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError("preserve read-only evidence as write-once")
    r=run(*[json.loads(x.read_text()) for x in (
        args.source36,args.source37,args.source38)])
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(r,ensure_ascii=False,indent=2)+"\n")
    compact={reg:{
        "makespan":{arm:round(d["mean_makespan"],6)
                    for arm,d in row["means"].items()},
        "paired":{name:{
            "delta":round(v["mean_paired_delta_tick"],6),
            "w":v["wins"],"t":v["ties"],"l":v["losses"],
            "CI":v["stratified_scenario_bootstrap95"],
            "sources":v["by_source"]}
            for name,v in row["paired"].items()},
    }for reg,row in r["summary"]["by_regime"].items()}
    print("H20_THREE_COHORT_REPLICATION",json.dumps({
        "groups":len(r["groups"]),
        "late":r["summary"]["late_model_reranking_accepted"],
        "regimes":compact},ensure_ascii=False))


if __name__=="__main__":
    main()
