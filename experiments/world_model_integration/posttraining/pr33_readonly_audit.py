"""Read-only PR32 online action-ranking and CPU budget audit.

Consumes only the immutable PR32 factual + public model decision records.
No model calls, reruns, training, new scene seeds or reoptimized weights.
Detects a concrete scheduling time-budget accounting issue: the original
PR30 policy tests deadline before each candidate rollout, not after the final
candidate. Here we distinguish recorded overruns from applied over-budget
decisions rather than treating a false '0 fallback' as budget compliance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROTOCOL="RHWM_PR32_FROZEN_POSTTRAINED_H20_ONLINE_20261112_V1"
ARMS=("cp_sat","rule_h20","original_v4_h20","warmstart_v4_h20","scratch_v4_h20")


def _mean(xs):
    return sum(xs)/len(xs) if xs else None


def _decision_rows(arm):
    return [d for d in arm["decision_records"]
            if "tick" in d and "selected_index" in d and "process_cpu_seconds" in d]


def audit(evidence,*,cpu_budget=0.5):
    if evidence.get("protocol")!=PROTOCOL or len(evidence.get("groups",[]))!=24:
        raise ValueError("not frozen PR32 original evidence")
    per_group=[]
    for item in evidence["groups"]:
        arm_data=item["arms"]
        if set(arm_data)!=set(ARMS):
            raise AssertionError("unexpected factual policy set")
        base=arm_data["cp_sat"]
        by_arm={}
        for name in ARMS:
            raw=arm_data[name]
            rows=_decision_rows(raw)
            timed=[d for d in rows if d["process_cpu_seconds"]>cpu_budget]
            # Any non-first plan chosen after budget elapsed is a real
            # violation of hard CPU deadline, even when no timeout flag exists.
            late_chosen=[d for d in timed if d["selected_index"]!=0]
            nonbase=[d for d in rows if d["selected_index"]!=0]
            by_arm[name]={
                "makespan":raw["makespan_ticks"],
                "late_jobs":raw["job_metrics"]["late_jobs"],
                "mean_decision_cpu_seconds":_mean(
                    [d["process_cpu_seconds"] for d in rows]),
                "max_decision_cpu_seconds":max(
                    [d["process_cpu_seconds"] for d in rows],default=0),
                "decision_records":len(rows),
                "multi_candidate_decisions":sum(d["legal_plans"]>1 for d in rows),
                "selected_nonfirst_count":len(nonbase),
                "actual_late_decisions":len(timed),
                "late_selected_nonfirst_count":len(late_chosen),
                "reported_overrun_flag":raw["decision_budget_exceeded"],
                "reported_fallbacks":raw["fallbacks"],
                "late_tick_examples":[{"tick":d["tick"],
                    "candidate_count":d["legal_plans"],
                    "selected_index":d["selected_index"],
                    "cpu":d["process_cpu_seconds"]}
                    for d in timed[:8]],
                "selected_nonfirst_ticks":[d["tick"] for d in nonbase],
                "mismatch_between_measured_and_recorded_budget":
                    len(timed)-raw["decision_budget_exceeded"],
                "delta_makespan_vs_cp_sat":raw["makespan_ticks"]-base["makespan_ticks"],
            }
        per_group.append({
            "group":item["group"],"seed":item["seed"],
            "regime":item["regime"],
            "by_arm":by_arm,
        })
    by_regime={}
    for regime in sorted({g["regime"] for g in per_group}):
        rows=[g for g in per_group if g["regime"]==regime]
        arms={}
        for arm in ARMS:
            samples=[r["by_arm"][arm] for r in rows]
            arms[arm]={
                "groups":len(rows),
                "reported_overruns":sum(r["reported_overrun_flag"] for r in samples),
                "measured_cpu_overruns":sum(r["actual_late_decisions"] for r in samples),
                "late_nonfirst_selections":sum(
                    r["late_selected_nonfirst_count"] for r in samples),
                "groups_with_measured_overruns":sum(
                    r["actual_late_decisions"]>0 for r in samples),
                "mean_max_decision_cpu_seconds":_mean(
                    [r["max_decision_cpu_seconds"] for r in samples]),
                "max_single_decision_cpu_seconds":max(
                    r["max_decision_cpu_seconds"] for r in samples),
                "total_candidate_flips":sum(
                    r["selected_nonfirst_count"] for r in samples),
                "mean_makespan_delta_vs_cp_sat":_mean(
                    [r["delta_makespan_vs_cp_sat"] for r in samples]),
                "first_divergence_after_which_futures_may_differ":[
                    {"group":g["group"],
                     "tick":next(iter(g["by_arm"][arm]["selected_nonfirst_ticks"]),None)}
                    for g in rows],
            }
        by_regime[regime]={"groups":len(rows),"arms":arms}
    all_late=[(g["group"],arm) for g in per_group for arm in ARMS
              if g["by_arm"][arm]["late_selected_nonfirst_count"]]
    return {
        "schema":"rhwm-pr33-read-only-online-budget-audit-v1",
        "source_pr32_protocol":PROTOCOL,
        "cpu_budget_seconds":cpu_budget,
        "groups":per_group,
        "summary":{
            "groups":len(per_group),
            "by_regime":by_regime,
            "late_nonfirst_group_arms":all_late,
            "any_late_nonfirst_selections":bool(all_late),
            "original_policy_budget_check":"pre-candidate only; last rollout can end after budget with no final deadline check",
            "limits":"CPU process time not wall clock; outcome comparison is descriptive across diverging online policies, not direct causal attribution to one tick",
            "gpu":0,"new_simulation":0,"new_model_inference":0,"new_training":0,
        },
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    args=ap.parse_args()
    if args.out.exists():raise FileExistsError("read-only audit output write-once")
    x=json.loads(args.source.read_text())
    result=audit(x)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    sample={reg:{arm:{
        "reported_budget":record["reported_overruns"],
        "actual_over_budget":record["measured_cpu_overruns"],
        "late_nonfirst":record["late_nonfirst_selections"],
        "max_cpu":record["max_single_decision_cpu_seconds"],
    } for arm,record in v["arms"].items()}
        for reg,v in result["summary"]["by_regime"].items()}
    print("PR33_ONLINE_BUDGET_AUDIT",json.dumps(sample,ensure_ascii=False))


if __name__=="__main__":
    main()
