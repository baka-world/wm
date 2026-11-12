"""Read-only PR43 original evidence audit: no simulation, inference or training.

Diagnoses when each alternative first changes an ONLINE action compared with
frozen V4 completed-stage objective, and whether a changed action is accompanied
by budget/fallback differences. Archived records are immutable input.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ARMS = ("cp_sat", "v4_completed", "v4_tie_trajectory", "v4_trajectory")
REGIMES = ("light", "medium", "heavy", "shift")


def selected_records(episode):
    rows = [r for r in episode["decision_records"] if "selected" in r]
    if len(rows) != episode["policy_calls"]:
        raise AssertionError("one selected decision required per factual tick")
    if len({r["tick"] for r in rows}) != len(rows):
        raise AssertionError("duplicate factual tick")
    return rows


def first_divergence(reference, alternative):
    """Only same-start, identical-action prefixes are comparable in isolation.

    Once actions diverge, later public states also diverge. It is wrong to
    interpret later per-tick score differences as counterfactual same-state
    action evaluation.
    """
    base = selected_records(reference)
    other = selected_records(alternative)
    for a,b in zip(base,other):
        if a["tick"] != b["tick"]:
            return {"type":"tick_sequence_diverged",
                    "base_tick":a["tick"],"alternative_tick":b["tick"]}
        same_pool = (a["first_proposal"] == b["first_proposal"]
                     and a["legal_plans"] == b["legal_plans"])
        if not same_pool:
            return {"type":"candidate_pool_changed_before_action",
                    "tick":a["tick"],
                    "base_candidates":a["legal_plans"],
                    "alternative_candidates":b["legal_plans"]}
        if a["selected"] != b["selected"]:
            return {"type":"different_action_on_same_prefix",
                    "tick":a["tick"],
                    "base_index":a["selected_index"],
                    "alternative_index":b["selected_index"],
                    "completion_tie_at_split": b.get("completed_tie"),
                    "full_state_distinct_at_split": b.get("future_state_different"),
                    "alternative_fallback_reason":b.get("fallback_reason"),
                    "same_initial_legal_candidate_pool":True}
    if len(base)!=len(other):
        return {"type":"episode_length_diverged_without_action_change",
                "base_decisions":len(base),"alternative_decisions":len(other)}
    return None


def group_report(group):
    source = group["arms"]["v4_completed"]
    findings = {}
    for arm in ("v4_tie_trajectory","v4_trajectory"):
        target = group["arms"][arm]
        first = first_divergence(source,target)
        delta = target["makespan_ticks"]-source["makespan_ticks"]
        if first is None and delta:
            raise AssertionError("identical online action history cannot produce different keyed-world outcome")
        findings[arm] = {
            "first_divergence":first,
            "makespan_delta_vs_frozen_v4":delta,
            "late_jobs_delta_vs_frozen_v4":(
                target["job_metrics"]["late_jobs"]-source["job_metrics"]["late_jobs"]),
            "budget_overruns_delta":(
                target["decision_budget_exceeded"]-source["decision_budget_exceeded"]),
            "fallback_delta":target["fallbacks"]-source["fallbacks"],
            "online_action_flips_vs_cp_sat_delta":(
                target["changed_plan_calls"]-source["changed_plan_calls"]),
            "different_first_online_action":(
                first is not None and first.get("tick")==0 and
                first.get("type")=="different_action_on_same_prefix"),
        }
    return {"id":group["id"],"regime":group["regime"],
            "seed":group["seed"],"methods":findings}


def report(source):
    if (source.get("schema")!="rhwm-pr43-independent-online-forecast-utility-v1"
        or source.get("evidence_scope")!="one-shot_new_synthetic_independent_full_online"
        or len(source.get("groups",[]))!=24
        or source.get("gpu")!=0 or source.get("training")!=0):
        raise ValueError("wrong PR43 authentic archived source")
    rows=[group_report(g) for g in source["groups"]]
    summary={}
    for arm in ("v4_tie_trajectory","v4_trajectory"):
        summary[arm]={}
        for regime in REGIMES:
            z=[r["methods"][arm] for r in rows if r["regime"]==regime]
            if len(z)!=6:raise AssertionError("PR43 regime missing")
            summary[arm][regime]={
                "groups":len(z),
                "first_divergence_count":sum(r["first_divergence"] is not None for r in z),
                "first_diff_same_candidate_pool":sum(
                    r["first_divergence"] is not None and
                    r["first_divergence"]["type"]=="different_action_on_same_prefix"
                    for r in z),
                "first_action_changed_at_tick0":sum(r["different_first_online_action"] for r in z),
                "different_actions_on_completion_ties":sum(
                    r["first_divergence"] is not None and
                    r["first_divergence"].get("completion_tie_at_split") is True
                    for r in z),
                "different_actions_with_full_state_variation":sum(
                    r["first_divergence"] is not None and
                    r["first_divergence"].get("full_state_distinct_at_split") is True
                    for r in z),
                "wins":sum(r["makespan_delta_vs_frozen_v4"]<0 for r in z),
                "ties":sum(r["makespan_delta_vs_frozen_v4"]==0 for r in z),
                "losses":sum(r["makespan_delta_vs_frozen_v4"]>0 for r in z),
                "mean_delta_makespan":sum(r["makespan_delta_vs_frozen_v4"] for r in z)/len(z),
                "mean_delta_budget_overruns":sum(r["budget_overruns_delta"] for r in z)/len(z),
            }
    return {
        "schema":"pr43-readonly-first-divergence-audit-v1",
        "evidence_scope":"existing_pr43_online_artifact_only",
        "original_protocol_sha256":source["protocol_sha256"],
        "original_v4_checkpoint_sha256":source["frozen_v4_sha256"],
        "new_training":0,"new_simulation":0,"new_model_inference":0,"gpu":0,
        "summary":summary,"scenario_first_divergences":rows,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():
        raise FileExistsError("write-once source evidence")
    analyzed=report(json.loads(args.source.read_text()))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(analyzed,ensure_ascii=False,indent=2)+"\n")
    print("PR43_FIRST_DIVERGENCE_AUDIT",json.dumps(analyzed["summary"],ensure_ascii=False))


if __name__=="__main__":
    main()
