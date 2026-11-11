"""Verify strict losslessness of urgent-first dominance pruning against actual episodes.

Usage:
  python -m experiments.world_model_integration.urgent_world_model_integration.verify_pruning \
    --original /tmp/pr18-original/confirmation/cases.json.gz \
    --optimized /tmp/current/confirmation-dominance/cases.json.gz \
    --new-full /tmp/current/new-full/cases.json.gz \
    --new-optimized /tmp/current/new-dominance/cases.json.gz \
    --output /tmp/current/regression.json
"""
from __future__ import annotations
import argparse
import gzip
import json
from pathlib import Path
import statistics


def cases_from(path):
    with gzip.open(path,"rt",encoding="utf-8") as handle:
        result=json.load(handle)
    assert len({(int(c["seed"]),c["arm"]) for c in result})==len(result)
    return {(int(c["seed"]),c["arm"]):c for c in result}


def verify_pair(full_cases,optimized_cases):
    if full_cases.keys()!=optimized_cases.keys():
        raise AssertionError("FULL_AND_OPTIMIZED_SCENE_SETS_DIFFER")
    total=0;pruned=0;query_full=query_fast=0;cpu_full=cpu_fast=0.
    changes=[]
    per_case=[]
    for key in sorted(full_cases):
        a,b=full_cases[key],optimized_cases[key]
        if a.get("evaluation_mode") not in (None,"full") or b["evaluation_mode"]!="dominance":
            raise AssertionError("BAD_EVALUATION_MODE")
        for k in ("history_sha256","events_sha256","metrics","retest_results",
                  "decision_count","workflow_status","completed_stages","resource_violations",
                  "motion_violations","lifecycle_violations"):
            if a[k]!=b[k]:
                raise AssertionError(f"LOSSLESS_REGRESSION_FAILED_{key}_{k}")
        if len(a["decisions"])!=len(b["decisions"]):
            raise AssertionError("DECISION_TIMELINE_DRIFT")
        local_pruned=0;local_total=0
        for aa,bb in zip(a["decisions"],b["decisions"]):
            for k in ("tick","observation_sha256","candidate_count","selected",
                      "selected_candidate_index","urgent_waiting"):
                if aa[k]!=bb[k]:
                    raise AssertionError(f"SELECTED_DECISION_CHANGED_{key}_{k}")
            if len(aa["plans"])!=len(bb["plans"]):
                raise AssertionError("CANDIDATE_COUNT_DRIFT")
            missed=[p["score"][0] for p in aa["plans"]]
            smallest=min(missed)
            for i,(original,filtered) in enumerate(zip(aa["plans"],bb["plans"])):
                local_total+=1
                if original["matching"]!=filtered["matching"]:
                    raise AssertionError(f"CANDIDATE_MATCHING_DRIFT_{key}_{i}")
                if filtered["uncovered_urgent"]!=missed[i]:
                    raise AssertionError("WRONG_UNCOVERED_URGENT_COUNT")
                if missed[i]>smallest:
                    local_pruned+=1
                    if filtered["evaluated"] or filtered["score"] is not None or filtered["particle_costs"]:
                        raise AssertionError("STRICTLY_DOMINATED_CANDIDATE_WAS_EVALUATED")
                else:
                    if not filtered["evaluated"]:
                        raise AssertionError("POSSIBLE_WINNER_WAS_PRUNED")
                    for fld in ("score","particle_costs"):
                        if original[fld]!=filtered[fld]:
                            raise AssertionError(f"NONDOMINATED_FORECAST_CHANGED_{key}_{i}_{fld}")
            if aa["selected_candidate_index"] not in [
                i for i,m in enumerate(missed) if m==smallest]:
                raise AssertionError("BASELINE_CHOSE_STRICTLY_DOMINATED_CANDIDATE")
        total+=local_total
        pruned+=local_pruned
        if b["candidates_pruned"]!=local_pruned:
            raise AssertionError("INCORRECT_PRUNED_COUNTER")
        if b["candidates_evaluated"]!=local_total-local_pruned:
            raise AssertionError("INCORRECT_EVALUATED_COUNTER")
        if a["arm"]=="P5":
            query_full+=a["query_calls"]
            query_fast+=b["query_calls"]
        cpu_full+=a["cpu_seconds"]
        cpu_fast+=b["cpu_seconds"]
        per_case.append({"seed":key[0],"arm":key[1],
                         "pruned":local_pruned,"total":local_total,
                         "full_cpu_seconds":a["cpu_seconds"],
                         "pruned_cpu_seconds":b["cpu_seconds"],
                         "full_model_queries":a["query_calls"],
                         "pruned_model_queries":b["query_calls"]})
    if pruned==0:
        raise AssertionError("NO_DOMINATED_CANDIDATES_SKIPPED")
    if query_fast>=query_full:
        raise AssertionError("V4_SERVICE_EVENT_CALLS_NOT_REDUCED")
    return {
        "lossless":True,"scenes":len(full_cases)//2,"arms":len(full_cases),
        "original_candidates":total,"dominated_candidates_eliminated":pruned,
        "fraction_candidate_eliminated":pruned/total,
        "V4_actual_service_queries_original":query_full,
        "V4_actual_service_queries_optimized":query_fast,
        "V4_query_reduction":query_full-query_fast,
        "total_cpu_seconds_original":cpu_full,
        "total_cpu_seconds_optimized":cpu_fast,
        "observed_cpu_reduction_seconds":cpu_full-cpu_fast,
        "observed_cpu_reduction_fraction":(cpu_full-cpu_fast)/cpu_full,
        "per_case":per_case
    }


def main():
    pa=argparse.ArgumentParser()
    for flag in ("original","optimized","new-full","new-optimized","output"):
        pa.add_argument("--"+flag,type=Path,required=True)
    args=vars(pa.parse_args())
    old=verify_pair(cases_from(args["original"]),cases_from(args["optimized"]))
    fresh=verify_pair(cases_from(args["new_full"]),cases_from(args["new_optimized"]))
    result={"historical_development_regression":old,
            "frozen_unseen_synthetic_regression":fresh,
            "business_metrics_changed":False,
            "source_class":"synthetic_cpu_lossless_optimization_only"}
    path=args["output"]
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    for scope,d in (("historical",old),("fresh",fresh)):
        print("VERIFIED",scope,"scenarios",d["scenes"],
              "pruned",d["dominated_candidates_eliminated"],
              "candidate_total",d["original_candidates"],
              "queries_saved",d["V4_query_reduction"],
              "CPU_saved",round(d["observed_cpu_reduction_seconds"],3),flush=True)


if __name__=="__main__":
    main()
