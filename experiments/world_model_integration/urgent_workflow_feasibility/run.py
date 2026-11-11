"""Fixed CPU development feasibility. No V4, neural training or old holdouts."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from dataclasses import asdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import time

from lab_transport.transition_schema import Assignment
from .core import (ROOT,PROTOCOL,public_spec,run_one,judge_score,
                   matchkey,classify_opportunities,digest,synthetic_retest)


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def immutable_json(path,payload):
    path=Path(path)
    if path.exists():raise FileExistsError("immutable evidence already exists: "+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,sort_keys=True,ensure_ascii=False,
                               indent=2)+"\n",encoding="utf-8")


def immutable_gzip(path,payload):
    if Path(path).exists():raise FileExistsError("immutable gzip already exists")
    raw=json.dumps(payload,sort_keys=True,ensure_ascii=False,
                   separators=(",",":")).encode("utf8")
    with gzip.GzipFile(filename="",mode="wb",fileobj=open(path,"wb"),mtime=0) as gz:
        gz.write(raw)


def strip_big(r):
    return {k:v for k,v in r.items()
            if k not in ("event_rows","history_rows","decisions","observed_rows")}


def mean_or_none(xs):
    return statistics.mean(xs) if xs else None


def group_statistics(cases,arm):
    selected=[x for x in cases if x["arm"]==arm]
    return {
      "groups":len(selected),
      "urgent_ontime_rate_mean":mean_or_none([c["metrics"]["urgent"]["on_time_rate"] for c in selected]),
      "urgent_late_mean_lower_bound":mean_or_none([c["metrics"]["urgent"]["late_mean_lower_bound"] for c in selected]),
      "urgent_unfinished_rate_mean":mean_or_none([c["metrics"]["urgent"]["incomplete_rate"] for c in selected]),
      "routine_restricted_flow_mean":mean_or_none([c["metrics"]["routine"]["restricted_mean_flow"] for c in selected]),
      "all_restricted_flow_mean":mean_or_none([c["metrics"]["all"]["restricted_mean_flow"] for c in selected]),
      "resource_wait_mean":mean_or_none([c["resource_wait_ticks"] for c in selected]),
      "blocked_moves_mean":mean_or_none([c["motion_blocked_attempts"] for c in selected]),
      "cpu_per_scenario_mean":mean_or_none([c["cpu_seconds"] for c in selected]),
      "policy_cpu_mean":mean_or_none([c["strategy_cpu_seconds"] for c in selected]),
      "retest_triggered_total":sum(c["metrics"]["retest_triggered"] for c in selected),
      "completed_workflows_total":sum(c["metrics"]["job_count"]-c["metrics"]["uncompleted_count"] for c in selected),
      "resource_violations_total":sum(c["resource_violations"] for c in selected),
      "motion_violations_total":sum(c["motion_violations"] for c in selected),
      "lifecycle_violations_total":sum(c["lifecycle_violations"] for c in selected),
    }


def freeze_opportunities(p0,protocol):
    """Select by current observation, seed and tick, BEFORE any counterfactual."""
    rows=[]
    max_states=protocol["oracle_selected_max_states"]
    for factual in p0:
        for decision in factual["decisions"]:
            if decision["candidate_count"]<2:continue
            if decision["urgent_waiting"]<1 or decision["routine_waiting"]<1:continue
            chosen=[item["matching"] for item in decision["plans"]]
            if len(chosen)>protocol["oracle_candidate_max"]:raise AssertionError("candidate budget")
            rows.append({
                "id":str(factual["spec"]["seed"])+":"+str(decision["tick"]),
                "seed":factual["spec"]["seed"],"tick":decision["tick"],
                "spec":factual["spec"],
                "candidate_matchings":chosen,
                "urgent_waiting":decision["urgent_waiting"],
                "routine_waiting":decision["routine_waiting"],
                "visible_retest_waiting":decision["retest_visible"],
                "visible_active_service":decision["active_service"],
                "visible_outage":decision["outage_visible"]})
    rows.sort(key=lambda x:(x["seed"],x["tick"]))
    chosen=[];per_seed=Counter()
    for row in rows:
        if per_seed[row["seed"]]>=protocol["oracle_max_per_scenario"]:continue
        per_seed[row["seed"]]+=1;chosen.append(row)
        if len(chosen)>=max_states:break
    return {"state_selection_stage":"SEALED_BEFORE_CANDIDATE_FUTURE_COST",
            "selection_rule":protocol["oracle_selection"],
            "selected":chosen,"eligible":len(rows),
            "unselected":len(rows)-len(chosen),"selected_groups":len(per_seed),
            "no_future_label_used":True,
            "only_development_seeds":True,
            "protocol_sha256":sha_file(PROTOCOL)}


def evaluate_oracle(lock,p0,protocol,start_cpu):
    by_seed={r["spec"]["seed"]:r for r in p0}
    detail=[];failures=[]
    for state in lock["selected"]:
        spec=state["spec"];factual=by_seed[spec["seed"]]
        tick=state["tick"]
        prefix={d["tick"]:tuple(Assignment(**a) for a in d["selected"])
                for d in factual["decisions"] if d["tick"]<tick}
        rows=[]
        for index,matching in enumerate(state["candidate_matchings"]):
            action=tuple(Assignment(**a) for a in matching)
            if len(action)!=len(set(action)) or len({a.robot_id for a in action})!=len(action) or len({a.sample_id for a in action})!=len(action):
                raise AssertionError("invalid frozen matching")
            try:
                alternate=run_one(spec,protocol,"P0",forced=action,
                                  prefix=prefix,target=tick,collect_obs=True)
                if alternate["history_rows"][:tick+1]!=factual["history_rows"][:tick+1]:
                    raise AssertionError("PRIVATE_HISTORY_PREFIX_MISMATCH")
                ev=lambda a:[x for x in a["event_rows"] if x["tick"]<tick]
                if ev(alternate)!=ev(factual):
                    raise AssertionError("EVENT_PREFIX_MISMATCH")
                ob=lambda a:[x for x in a["observed_rows"] if x["tick"]<=tick]
                if ob(alternate)!=ob(factual):
                    raise AssertionError("OBSERVED_PREFIX_MISMATCH")
                chosen_at=next((d for d in alternate["decisions"] if d["tick"]==tick),None)
                if chosen_at is None or matchkey(Assignment(**x) for x in chosen_at["selected"])!=matchkey(action):
                    raise AssertionError("REPLACEMENT_NOT_EXECUTED")
                for jid,truth in alternate["retest_results"].items():
                    if truth!=synthetic_retest(spec["seed"],jid,spec["retest_rate_percent"]):
                        raise AssertionError("RETEST_EXOGENOUS_CHANGED")
                rows.append({"candidate_index":index,"matching":matching,
                             "objective":alternate["objective"],
                             "metrics":alternate["metrics"],
                             "resource_wait_ticks":alternate["resource_wait_ticks"],
                             "cpu_seconds":alternate["cpu_seconds"],
                             "events_sha256":alternate["events_sha256"],
                             "history_sha256":alternate["history_sha256"],
                             "stage_result_count":len(alternate["retest_results"]),
                             "prefix_verified":True})
            except (AssertionError,RuntimeError,ValueError) as e:
                failures.append({"id":state["id"],"candidate":index,"reason":str(e)})
                rows=[];break
            if time.process_time()-start_cpu>protocol["cpu_timeout_seconds"]:
                raise TimeoutError("CPU total budget exceeded during oracle")
        if not rows:continue
        best=min(range(len(rows)),key=lambda i:(rows[i]["objective"],i))
        p0match=next((d["selected"] for d in factual["decisions"] if d["tick"]==tick),None)
        p0choice=next((i for i,row in enumerate(rows) if row["matching"]==p0match),None)
        if p0choice is None:
            # Comparison is not allowed if solver candidate set didn't include factual P0 matching.
            failures.append({"id":state["id"],"reason":"FACTUAL_P0_MATCHING_NOT_IN_FROZEN_CANDIDATES"})
            continue
        p0_obj=rows[p0choice]["objective"]
        best_obj=rows[best]["objective"]
        detail.append({"id":state["id"],"seed":spec["seed"],"tick":tick,
                       "category":"urgent_routine_competition",
                       "candidate_count":len(rows),"candidates":rows,
                       "factual_p0_index":p0choice,"best_within_candidates":best,
                       "p0_objective":p0_obj,"best_objective":best_obj,
                       "p0_urgent_lateness_minus_best":
                           rows[p0choice]["metrics"]["urgent"]["late_mean_lower_bound"]-
                           rows[best]["metrics"]["urgent"]["late_mean_lower_bound"],
                       "p0_routine_flow_minus_best":
                           rows[p0choice]["metrics"]["routine"]["restricted_mean_flow"]-
                           rows[best]["metrics"]["routine"]["restricted_mean_flow"],
                       "single_realization_only":True,
                       "finite_candidates_only":True})
    return detail,failures


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    p=json.loads(PROTOCOL.read_text())
    if p["source_head"]!="4a3ca9edc10350d4fbe42538c00c3b19ad28fffb":
        raise AssertionError("wrong source head")
    if not p["no_old_holdout_reuse"]:raise AssertionError("old test reuse prohibited")
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("immutable evidence dir already populated")
    args.out.mkdir(parents=True,exist_ok=True)
    start_cpu=time.process_time();start_wall=time.monotonic()
    all_cases=[];p0_cases=[];counts=Counter()
    seeds=[("smoke",i,s) for i,s in enumerate(p["smoke_seeds"])]+[
        ("development",i+len(p["smoke_seeds"]),s)
        for i,s in enumerate(p["development_seeds"])]
    if len(seeds)!=24 or len(set(seed for _,_,seed in seeds))!=len(seeds):
        raise AssertionError("frozen group count or seeds changed")
    for group,index,seed in seeds:
        spec=public_spec(index,seed,p["scenario_max_ticks"])
        arm_cases=[]
        for arm in p["baselines"]:
            case=run_one(spec,p,arm,collect_obs=(arm=="P0"))
            case["group"]=group
            arm_cases.append(case);all_cases.append(case)
            if arm=="P0":
                p0_cases.append(case)
                counts.update(classify_opportunities(case))
            print("CASE",group,index,seed,arm,
                  "urgent_ontime",case["metrics"]["urgent"]["on_time_rate"],
                  "urgent_late",case["metrics"]["urgent"]["late_mean_lower_bound"],
                  "regular_restricted_flow",case["metrics"]["routine"]["restricted_mean_flow"],
                  "retets",case["metrics"]["retest_triggered"],flush=True)
            if time.process_time()-start_cpu>p["cpu_timeout_seconds"]:
                raise TimeoutError("total CPU budget exhausted")
        # Cross-policy exogenous results are keyed by job, independent of
        # whether a particular policy had time to reach detection before cutoff.
        for jobid in {job["job_id"] for c in arm_cases for job in c["metrics"]["jobs"]}:
            seen={c["retest_results"][jobid] for c in arm_cases
                  if jobid in c["retest_results"]}
            if len(seen)>1:raise AssertionError("policy-dependent retest outcome")
        if group=="smoke" and index==3:
            if not all(c["resource_violations"]==0 and
                       c["motion_violations"]==0 and
                       c["lifecycle_violations"]==0 for c in all_cases):
                raise AssertionError("smoke safety failed")
            print("SMOKE_GATE_PASS",flush=True)
    # Source outcomes were already generated for baseline comparison; oracle
    # SELECTION uses only original P0 current observation and frozen plan IDs.
    selection=freeze_opportunities(p0_cases,p)
    immutable_json(args.out/"selection_lock.json",selection)
    print("LOCK_SHA",sha_file(args.out/"selection_lock.json"),flush=True)
    oracle,failures=evaluate_oracle(selection,p0_cases,p,start_cpu)
    raw={"protocol_sha256":sha_file(PROTOCOL),"groups":all_cases,
         "interpretation":"synthetic development only; job-level final stage; no clinical TAT"}
    immutable_gzip(args.out/"full_raw.json.gz",raw)
    immutable_json(args.out/"oracle.json",oracle)
    immutable_json(args.out/"oracle_failures.json",failures)
    tab={arm:group_statistics(all_cases,arm) for arm in p["baselines"]}
    cohort={
       "smoke":{arm:group_statistics([c for c in all_cases if c["group"]=="smoke"],arm)
                for arm in p["baselines"]},
       "development":{arm:group_statistics([c for c in all_cases if c["group"]=="development"],arm)
                       for arm in p["baselines"]}}
    pairs={}
    by={(c["spec"]["seed"],c["arm"]):c for c in all_cases}
    for target in ("P1","P2","P3"):
        items=[]
        for _,_,seed in seeds:
            a,b=by[(seed,"P0")],by[(seed,target)]
            items.append({"seed":seed,"baseline":"P0","arm":target,
                "delta_urgent_on_time":b["metrics"]["urgent"]["on_time_rate"]-a["metrics"]["urgent"]["on_time_rate"],
                "delta_urgent_late_lower_bound":b["metrics"]["urgent"]["late_mean_lower_bound"]-a["metrics"]["urgent"]["late_mean_lower_bound"],
                "delta_routine_restricted_flow":b["metrics"]["routine"]["restricted_mean_flow"]-a["metrics"]["routine"]["restricted_mean_flow"],
                "same_exogenous_retest_for_shared_completed_detection":all(
                  b["retest_results"][jid]==truth for jid,truth in a["retest_results"].items()
                  if jid in b["retest_results"])})
        pairs[target]=items
    immutable_json(args.out/"paired.json",pairs)
    observed_change=sum(x["best_within_candidates"]!=x["factual_p0_index"] for x in oracle)
    urgent_gains=[x["p0_urgent_lateness_minus_best"] for x in oracle]
    regular_gains=[x["p0_routine_flow_minus_best"] for x in oracle]
    summary={
      "classification":"synthetic_development_feasibility_not_independent_generalization",
      "source_head":p["source_head"],"protocol_sha256":sha_file(PROTOCOL),
      "groups":len(seeds),"smoke_groups":len(p["smoke_seeds"]),
      "development_groups":len(p["development_seeds"]),
      "no_old_independent_test":True,"no_v4":True,
      "aggregated_all":tab,"by_cohort":cohort,"coverage":dict(counts),
      "oracle_selected":len(selection["selected"]),"oracle_valid":len(oracle),
      "oracle_blocked":len(failures),"oracle_changed_optimal_candidate":observed_change,
      "oracle_positive_urgent_lateness_differences":sum(x>0 for x in urgent_gains),
      "oracle_urgent_lateness_value_mean":mean_or_none(urgent_gains),
      "oracle_routine_cost_tradeoff_mean":mean_or_none(regular_gains),
      "conditional_future_expectation_estimated":False,
      "retest_branch_created":sum(x["metrics"]["retest_triggered"] for x in p0_cases),
      "retest_no_decision":counts.get("retest_decision_no",0),
      "cpu_process_seconds":time.process_time()-start_cpu,
      "wall_seconds":time.monotonic()-start_wall,
      "runtime":platform.python_version(),
      "result_guidance":"P1 vs P0 is priority-goal value not world-model value; P3 analytic public assumptions are only proxy"}
    if failures:status="BLOCKED_ORACLE_REPLAY"
    elif selection["selected"] and oracle and (observed_change>0 or
               sum(x["delta_urgent_late_lower_bound"]<0 for x in pairs["P1"])>0):
        status="GO_FOR_NEW_PREDICTION_EXPERIMENT_DESIGN_ONLY"
    elif not selection["selected"]:status="BLOCKED_NO_FAIR_URGENT_ROUTINE_CHOICE"
    else:status="NO_GO_DECISION_VALUE_NOT_OBSERVED"
    summary["verdict"]=status
    immutable_json(args.out/"summary.json",summary)
    immutable_json(args.out/"provenance.json",{
        "protocol_sha256":sha_file(PROTOCOL),"core_sha256":sha_file(ROOT/"core.py"),
        "runner_sha256":sha_file(ROOT/"run.py"),
        "source_commit":p["source_head"],"python":platform.python_version(),
        "cpu_only":True,"training_calls":0,"v4_calls":0,
        "conditional_retest_keyed_by_seed_job":True,
        "one_single_realization_per_candidate":True,
        "oracle_selection_preceded_any_candidate_replay":True})
    hashes={x.name:sha_file(x) for x in sorted(args.out.iterdir()) if x.is_file()}
    immutable_json(args.out/"sha256.json",hashes)
    print("SUMMARY",json.dumps({k:summary[k] for k in
          ("verdict","groups","coverage","oracle_selected","oracle_valid",
           "oracle_changed_optimal_candidate","cpu_process_seconds")}),flush=True)

if __name__=="__main__":
    main()
