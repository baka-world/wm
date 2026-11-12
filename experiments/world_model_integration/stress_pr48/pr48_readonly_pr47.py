"""PR48 read-only decision mechanism audit on ORIGINAL PR47 online evidence.

No new simulator, model inference, reranking, or outcome collection. Each
first divergence is compared only while both prior actual selections and
candidate pools are identical; later diverged trajectories are not mistaken
for paired same-state controls.
"""
from __future__ import annotations
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json

from experiments.world_model_integration.pr46_evidence_audit.pr46_first_divergence import (
    split_point
)

ARMS=("v4_original_15","v4_recovery_15")
MECHANISMS=("within_support","extended_outage")


def analyze(source):
    if (source.get("schema")!="pr47-equal-budget-closed-loop-new-cohort-v1"
        or source.get("summary",{}).get("scenes")!=24
        or source.get("summary",{}).get("total_episodes")!=120
        or source.get("gpu")!=0 or not source.get("no_new_training")
        or len(source.get("scenes",[]))!=24):
        raise ValueError("only original complete PR47 online data are admissible")
    result=[]
    for g in source["scenes"]:
        old=g["arms"][ARMS[0]]
        new=g["arms"][ARMS[1]]
        if old["budget_exceeded"] or new["budget_exceeded"] or old["fallbacks"] or new["fallbacks"]:
            raise AssertionError("original PR47 1.5s budget did not cover full scores")
        first=split_point(
            {"calls":old["calls"] if "calls" in old else old["calls"],
             "decision_records":old["decision_records"]},
            {"calls":new["calls"] if "calls" in new else new["calls"],
             "decision_records":new["decision_records"]})
        result.append({
            "id":g["id"],"mechanism":g["mechanism"],"station_id":g["station_id"],
            "seed":g["seed"],
            "first_difference":first,
            "actual_makespan_delta_ticks":new["makespan_ticks"]-old["makespan_ticks"],
            "actual_late_jobs_delta":new["job_metrics"]["late_jobs"]-old["job_metrics"]["late_jobs"],
            "old_selected_nonfirst":old["plan_flips"],
            "recovery_selected_nonfirst":new["plan_flips"],
            "recovery_sampled_repair_events":new["sampled_repair_events"]
        })
    groups={}
    for mechanism in MECHANISMS:
        members=[x for x in result if x["mechanism"]==mechanism]
        if len(members)!=12:
            raise ValueError("changed PR47 cohort")
        types=Counter(x["first_difference"]["kind"]
                      if x["first_difference"] else "identical_real_decisions"
                      for x in members)
        classes=Counter(
            x["first_difference"].get("score_or_budget_class")
            for x in members if x["first_difference"]
            and x["first_difference"]["kind"]=="same_prefix_different_selected_action")
        groups[mechanism]={
            "groups":len(members),
            "first_difference_types":dict(types),
            "different_full_scoring_class":dict(classes),
            "first_differences_before_outage":sum(
                x["first_difference"] is not None and
                (x["first_difference"].get("tick",999)<7) for x in members),
            "identical_choices":types.get("identical_real_decisions",0),
            "improvements":sum(x["actual_makespan_delta_ticks"]<0 for x in members),
            "ties":sum(x["actual_makespan_delta_ticks"]==0 for x in members),
            "regressions":sum(x["actual_makespan_delta_ticks"]>0 for x in members),
            "mean_delta_ticks":sum(x["actual_makespan_delta_ticks"] for x in members)/len(members)
        }
    return {"schema":"pr48-readonly-pr47-first-complete-score-divergence-v1",
            "source":"original_sha256_verified_pr47_archive",
            "zero_new_simulation":True,"zero_new_training":True,
            "zero_new_inference":True,"gpu":0,
            "summary":groups,"cases":result,
            "caution":"Common-prefix first-action attribution only, not causal decomposition of entire later online trajectory."}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    p=parser.parse_args()
    if p.out.exists():
        raise FileExistsError("read-only audit output is write-once")
    data=json.loads(p.source.read_text())
    result=analyze(data)
    p.out.parent.mkdir(parents=True,exist_ok=True)
    p.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print("PR48_READONLY_PR47_DECISION_PATHS",json.dumps(result["summary"],ensure_ascii=False))


if __name__=="__main__":
    main()
