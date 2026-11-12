"""PR46 read-only, first-common-prefix causal-accounting of online decisions.

Original online artifact only. No BusinessSimulator, V4 model, new rollouts or
new outcomes. The first differing action is compared only after the earlier
action sequence and candidate-proposal inputs are verified identical.
After that point the real trajectories may differ and are never compared
as though they share the same public state.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import argparse
import json

ARMS=("cp_sat","v4_original","v4_public_km_recovery")
MECHANISMS=("within_support","extended_outage")


def check_episode(row):
    log=row["decision_records"]
    if not isinstance(log,list) or len(log)!=row["calls"]:
        raise ValueError("online decision trace not complete")
    if len({d["tick"] for d in log})!=len(log):
        raise ValueError("repeated public decision tick")
    if any(d["tick"]<0 or d["legal_plans"]<1 for d in log):
        raise ValueError("invalid public decision trace")
    return log


def split_point(a,b):
    """Find first same-prefix action discrepancy; later ticks are NOT comparable."""
    x=check_episode(a)
    y=check_episode(b)
    for orig,recovery in zip(x,y):
        if orig["tick"]!=recovery["tick"]:
            return {"kind":"clock_or_candidate_path_diverged",
                    "original_tick":orig["tick"],"recovery_tick":recovery["tick"]}
        if (orig["first_proposal"]!=recovery["first_proposal"]
            or orig["legal_plans"]!=recovery["legal_plans"]):
            return {"kind":"candidate_pool_diverged_before_selected_action",
                    "tick":orig["tick"]}
        if orig["selected"]!=recovery["selected"]:
            original_full=(len(orig["predicted_completed"])==orig["legal_plans"])
            recovery_full=(len(recovery["predicted_completed"])==recovery["legal_plans"])
            return {
                "kind":"same_prefix_different_selected_action",
                "tick":orig["tick"],
                "original_selected_index":orig["selected_index"],
                "recovery_selected_index":recovery["selected_index"],
                "same_first_proposal_and_count":True,
                "original_candidates_fully_scored":original_full,
                "recovery_candidates_fully_scored":recovery_full,
                "score_or_budget_class":(
                    "both_completed_scoring" if original_full and recovery_full
                    else "original_complete_recovery_incomplete"
                    if original_full and not recovery_full
                    else "original_incomplete_recovery_complete"
                    if recovery_full and not original_full
                    else "both_incomplete_scoring"
                ),
            }
    if len(x)!=len(y):
        return {"kind":"decision_log_length_differs_without_prior_action_disagreement",
                "orig_count":len(x),"recovery_count":len(y)}
    return None


def case_analysis(group):
    original=group["arms"]["v4_original"]
    recover=group["arms"]["v4_public_km_recovery"]
    first=split_point(original,recover)
    makespan_delta=recover["makespan_ticks"]-original["makespan_ticks"]
    if first is None and makespan_delta:
        raise AssertionError("identical deterministic online choices yielded changed result")
    return {
        "id":group["id"],"seed":group["seed"],
        "mechanism":group["mechanism"],"station_id":group["station_id"],
        "makespan_delta":makespan_delta,
        "late_job_delta":recover["job_metrics"]["late_jobs"]-original["job_metrics"]["late_jobs"],
        "fallbacks_delta":recover["fallbacks"]-original["fallbacks"],
        "budget_exceeded_delta":recover["budget_exceeded"]-original["budget_exceeded"],
        "changed_action_delta":recover["plan_flips"]-original["plan_flips"],
        "first_difference":first,
    }


def audit(source):
    if (source.get("schema")!="pr46-recovery-mechanism-shift-full-online-v1"
        or source.get("cohort")!="online_only_not_forecast"
        or source.get("gpu")!=0 or source.get("training")!=0
        or len(source.get("groups",[]))!=24):
        raise ValueError("not the original PR46 frozen online evidence")
    cases=[case_analysis(g) for g in source["groups"]]
    result={}
    for mechanism in MECHANISMS:
        part=[c for c in cases if c["mechanism"]==mechanism]
        if len(part)!=12:raise AssertionError("missing preregistered mechanism cohort")
        kinds=Counter(c["first_difference"]["kind"] if c["first_difference"]
                      else "identical_actions" for c in part)
        ranked=Counter(
            c["first_difference"].get("score_or_budget_class")
            for c in part if c["first_difference"] and
            c["first_difference"]["kind"]=="same_prefix_different_selected_action")
        result[mechanism]={
            "groups":12,
            "first_difference_types":dict(kinds),
            "first_different_action_explanations":dict(ranked),
            "first_difference_at_tick0":sum(
                c["first_difference"] is not None
                and c["first_difference"].get("tick")==0 for c in part),
            "first_different_action_at_publicly_down_window":sum(
                c["first_difference"] is not None
                and (c["first_difference"].get("tick") or -1)>=7 for c in part),
            "wins":sum(c["makespan_delta"]<0 for c in part),
            "ties":sum(c["makespan_delta"]==0 for c in part),
            "losses":sum(c["makespan_delta"]>0 for c in part),
            "mean_makespan_delta":sum(c["makespan_delta"] for c in part)/len(part),
            "additional_budget_overruns":sum(c["budget_exceeded_delta"] for c in part),
            "additional_fallbacks":sum(c["fallbacks_delta"] for c in part),
            "disagreements_by_station":dict(Counter(
                c["station_id"] for c in part if c["first_difference"])),
            "mean_late_job_delta":sum(c["late_job_delta"] for c in part)/len(part),
        }
    return {
        "schema":"pr46-readonly-original-online-first-divergence-v1",
        "source":"verified_original_pr46_online_artifact_only",
        "protocol_sha256":source["protocol_sha256"],
        "v4_checkpoint_sha256":source["frozen_v4_checkpoint_sha256"],
        "km_artifact_sha256":source["source_pr44_train_artifact_sha256"],
        "new_simulations":0,"new_model_calls":0,"new_training":0,"gpu":0,
        "summary":result,"cases":cases,
        "limits":"Budget contributions cannot be causally separated after online trajectories diverge; same-prefix attribution applies to first difference only.",
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError("read-only evidence output cannot overwrite")
    data=audit(json.loads(a.source.read_text()))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(data,indent=2,ensure_ascii=False)+"\n")
    print("PR46_READONLY_FIRST_DIVERGENCE",json.dumps(data["summary"],ensure_ascii=False))


if __name__=="__main__":
    main()
