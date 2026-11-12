"""Read-only attribution of PR48 archived fully-online three-stress experiments.

All sources are frozen original GitHub Actions artifacts, independently SHA-
verified by CI. This module contains NO simulator, NO V4, NO future oracle, NO
new trajectories. Compare first choice ONLY while actual chosen-action
histories, ticks and candidate-pool summaries are still identical.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import argparse
import json

from experiments.world_model_integration.pr46_evidence_audit.pr46_first_divergence import split_point

REGIMES=("congestion","multi_fault","staggered_arrivals")
ARMS=("v4_original_15","v4_recovery_15")


def per_case(g):
    old=g["arms"][ARMS[0]]
    newer=g["arms"][ARMS[1]]
    first=split_point(
        {"calls":old["decision_calls"],"decision_records":old["decision_records"]},
        {"calls":newer["decision_calls"],"decision_records":newer["decision_records"]}
    )
    return {
        "id":g["id"],"seed":g["seed"],"regime":g["regime"],
        "mechanism":g["mechanism"],"station_id":g["station_id"],
        "first_difference":first,
        "true_completion_delta_ticks":newer["makespan_ticks"]-old["makespan_ticks"],
        "late_job_delta":newer["job_metrics"]["late_jobs"]-old["job_metrics"]["late_jobs"],
        "original_budget_overruns":old["budget_overruns"],
        "recovery_budget_overruns":newer["budget_overruns"],
        "original_fallbacks":old["fallbacks"],
        "recovery_fallbacks":newer["fallbacks"],
        "original_policy_changes":old["changed_proposals"],
        "recovery_policy_changes":newer["changed_proposals"]
    }


def describe(cases):
    diff=[x["true_completion_delta_ticks"] for x in cases]
    kinds=Counter((x["first_difference"] or {}).get("kind","identical_real_actions")
                  for x in cases)
    select=[x["first_difference"] for x in cases if x["first_difference"] and
            x["first_difference"]["kind"]=="same_prefix_different_selected_action"]
    classes=Counter(x.get("score_or_budget_class") for x in select)
    return {
        "groups":len(cases),
        "mean_delta_ticks":sum(diff)/len(diff) if diff else None,
        "wins":sum(x<0 for x in diff),
        "ties":sum(x==0 for x in diff),
        "losses":sum(x>0 for x in diff),
        "first_difference_types":dict(kinds),
        "first_changed_choice_same_candidate_set":len(select),
        "first_changed_choice_scoring_class":dict(classes),
        "first_differences_before_fault_tick7":sum(
            x["first_difference"] is not None and
            x["first_difference"].get("tick",999)<7 for x in cases),
        "first_differences_after_secondary_fault_tick13":sum(
            x["first_difference"] is not None and
            x["first_difference"].get("tick",-1)>=13 for x in cases),
        "original_budget_overruns":sum(x["original_budget_overruns"] for x in cases),
        "recovery_budget_overruns":sum(x["recovery_budget_overruns"] for x in cases),
        "original_fallbacks":sum(x["original_fallbacks"] for x in cases),
        "recovery_fallbacks":sum(x["recovery_fallbacks"] for x in cases),
        "losses_with_first_common_prefix_budget_mismatch":sum(
            x["true_completion_delta_ticks"]>0 and x["first_difference"] is not None and
            (x["first_difference"].get("score_or_budget_class") in (
                "original_complete_recovery_incomplete",
                "original_incomplete_recovery_complete"
            )) for x in cases),
        "losses_with_both_fully_scored_at_first_difference":sum(
            x["true_completion_delta_ticks"]>0 and x["first_difference"] is not None and
            x["first_difference"].get("score_or_budget_class")=="both_completed_scoring"
            for x in cases),
    }


def analyze(documents):
    regimes=set(REGIMES)
    if set(documents)!=regimes:
        raise ValueError("require all 3 original PR48 stress artifacts")
    seen=set()
    cases=[]
    fingerprint=set()
    for regime,data in documents.items():
        if (data.get("schema")!="pr48-full-online-stress-independent-v1"
            or data.get("regime")!=regime
            or data.get("new_gpu")!=0 or data.get("new_training")!=0
            or len(data.get("groups",[]))!=16
            or data.get("summary",{}).get("actual_episodes")!=48):
            raise ValueError("unrecognized original PR48 factual evidence")
        fingerprint.add((
            data["protocol_sha256"],data["v4_checkpoint_sha256"],
            data["pr44_train_only_model_sha256"]
        ))
        for x in data["groups"]:
            if x["seed"] in seen:
                raise ValueError("reused seed across PR48 stress groups")
            if set(x["arms"])!={"cp_sat",*ARMS}:
                raise ValueError("unregistered real simulator policy arm")
            seen.add(x["seed"])
            cases.append(per_case(x))
    if len(fingerprint)!=1 or len(cases)!=48:
        raise ValueError("inconsistent research source fingerprints")
    summary={
        "overall":describe(cases),
        "by_regime":{r:describe([x for x in cases if x["regime"]==r])
                     for r in REGIMES},
        "by_mechanism":{m:describe([x for x in cases if x["mechanism"]==m])
                        for m in ("within_support","extended_outage")}
    }
    return {
        "schema":"pr48-readonly-first-common-prefix-decision-diagnostics-v1",
        "source_fingerprints":list(next(iter(fingerprint))),
        "scope":"48 actual online scenarios from frozen original PR48 artifacts only",
        "zero_new_simulation":True,"zero_new_model_calls":True,
        "gpu":0,"training":0,"summary":summary,"cases":cases,
        "caution":"Divergence classifications identify first common-prefix reasons, not full-episode causal attribution after states diverge."
    }


def main():
    parser=argparse.ArgumentParser()
    for name in REGIMES:
        parser.add_argument("--"+name.replace("_","-"),required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    paths={r:getattr(args,r) for r in REGIMES}
    if args.out.exists():
        raise FileExistsError("diagnostics output cannot overwrite")
    result=analyze({r:json.loads(path.read_text()) for r,path in paths.items()})
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print("PR48_FIRST_DECISION_DIAGNOSTICS",json.dumps(result["summary"],ensure_ascii=False))


if __name__=="__main__":
    main()
