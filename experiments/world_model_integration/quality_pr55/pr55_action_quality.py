"""PR55 second QUALITY gate: actual same-public-prefix action information.

Use original audited PR52 EntityTruthBranch to compare legal CP-SAT
candidate actions on SAME current public state and source business scene.
Only FIRST dispatch at registered anchor changes, original CP-SAT first
candidate resumes afterward. Do not invent oracle training labels or
change simulator rules. Device fault durations and future releases are
facts of the environment, never graph input.

IMPORTANT: A shared simulator seed + schedule does not guarantee identical
post-intervention random draw sequences if execution changes their order.
This is a paired-simulator audit, not an identification theorem.
"""
from __future__ import annotations
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import time

from business_transport.simulation import BusinessSimulator, resource_audit
from experiments.business_constraints.run import lifecycle_audit
from experiments.world_model_integration.quality_pr55.pr55_generate import (
    PROTOCOL as QUALITY_PROTOCOL,REGIMES,build_quality_business,validate
)
from experiments.world_model_integration.stress_pr48.pr48_scenarios import (
    PROTOCOL as STRESS_PROTOCOL
)
from experiments.world_model_integration.entity_pr52.pr52_labels import EntityTruthBranch
from experiments.world_model_integration.online_horizon.pr30_run import _facts
from experiments.world_model_integration.decision_learning_pr51.pr51_labels import (
    _fingerprint
)
from lab_transport.multisite_experiment import discrete_safety_violations

PROTOCOL=Path(__file__).with_name("action_protocol.json")


def validate_action(p,q):
    validate(q)
    if (p.get("protocol")!="PR55_ACTION_CONTRAST_INFORMATION_QUALITY_DEV_ONLY_V1"
        or p.get("parent_quality_protocol")!=q["protocol"]):
        raise ValueError("wrong pre-registered action information contract")
    cases={x["id"] for x in q["pilot_cases"]}
    if len(p["ids"])!=18 or len(set(p["ids"]))!=18 or set(p["ids"])!=cases:
        raise ValueError("paired action gate must cover all original 18 developer scenes")
    if p["candidate_limit"]!=4 or p["public_horizons"]!=[5,20]:
        raise ValueError("frozen action/horizon protocol changed")
    if p["resource"]["GPU"] or p["resource"]["neural_training"]:
        raise ValueError("CPU only; no fitted network in this gate")
    return True


def paired_episode(spec,scenario,*,expected=None,index=0):
    business,geometry=build_quality_business(spec,scenario)
    p=EntityTruthBranch(geometry,business.rules,spec["seed"],
                        {"counterfactual_max_candidates":4},
                        expected=expected,branch_index=index)
    actual=BusinessSimulator(business,"world",spec["seed"],collect_metrics=True).run(
        dispatch_policy=p,on_observation=p.observe)
    resource=resource_audit(actual,business.rules)
    if resource["violations"] or discrete_safety_violations(actual) or lifecycle_audit(actual["history"]):
        raise AssertionError("actual business action branch violated safety")
    return p,{
        "branch_index":index,
        "complete":actual["summary"]["status"]=="completed",
        "true_makespan_ticks":actual["summary"]["ticks"],
        "true_late_jobs":_facts(actual,business)["late_jobs"],
        "public_future_entity_labels":p.future_nodes,
        "factual_events_sha256":_fingerprint(actual["events"]),
    }


def state_difference(source,other,h):
    if h not in source or h not in other:
        return None
    a=source[h];b=other[h]
    if len(a["node_target"])!=len(b["node_target"]):
        raise AssertionError("different current-public entity identity alignment")
    if len(a["node_target_mask"])!=len(b["node_target_mask"]):
        raise AssertionError("changed supervision node mask dimensions")
    count=0
    changed=0
    l1=0.
    for x,y,xm,ym in zip(a["node_target"],b["node_target"],
                         a["node_target_mask"],b["node_target_mask"]):
        if not len(x)==len(y)==len(xm)==len(ym)==6:
            raise AssertionError("nonstandard public future node target")
        for u,v,m,n in zip(x,y,xm,ym):
            if m and n:
                count+=1
                l1+=abs(float(u)-float(v))
                changed+=int(abs(float(u)-float(v))>1e-6)
    if not count:
        raise AssertionError("no common public future dynamic states")
    new_events=["global_newly_visible_stage_count","global_new_stage_completed_count",
                "global_total_completed_stage_count"]
    event_delta=sum(abs(float(a[z])-float(b[z])) for z in new_events)
    return {
        "horizon":int(h),"common_fields":count,
        "changed_public_fields":changed,
        "common_node_l1":l1,
        "mean_public_state_l1":l1/count,
        "anonymous_future_event_l1":event_delta,
        "nonzero_true_public_state_effect":changed>0 or event_delta>0
    }


def paired_case(spec,scenario):
    reference,ref=paired_episode(spec,scenario)
    if reference.anchor is None:
        return {
            "id":spec["id"],"profile":spec["profile"],
            "regime":spec["regime"],"labelable":False,
            "reason":"no_two_or_more_different_legal_CPSAT_proposals_at_registered_tick"
        }
    anchor={
        **reference.anchor,
        "observation_sha256":reference.anchor_fingerprint,
        "plans_sha256":reference.anchor_plan_fingerprint
    }
    start=anchor["tick"]
    original=[x for x in reference.trace if x["tick"]<=start]
    outcomes=[ref]
    comparisons=[]
    for index in range(1,anchor["candidate_count"]):
        p,outcome=paired_episode(spec,scenario,expected=anchor,index=index)
        new=[x for x in p.trace if x["tick"]<=start]
        if len(new)!=len(original):
            raise AssertionError("counterfactual ran on different dispatch history length")
        for first,other in zip(original,new):
            if (first["tick"]!=other["tick"] or
                first["observation_sha256"]!=other["observation_sha256"] or
                first["plans_sha256"]!=other["plans_sha256"] or
                (first["tick"]!=start and first["selected_plan"]!=other["selected_plan"])):
                raise AssertionError("paired branch NOT from exact same public state/candidates")
        if p.node_anchor["node_keys"]!=reference.node_anchor["node_keys"]:
            raise AssertionError("factual simulator public nodes have changed before intervention")
        outcomes.append(outcome)
        comparisons.append({
            "alternative_index":index,
            "relative_makespan_ticks":outcome["true_makespan_ticks"]-ref["true_makespan_ticks"],
            "H5_public_entity_difference":state_difference(
                ref["public_future_entity_labels"],outcome["public_future_entity_labels"],"5"),
            "H20_public_entity_difference":state_difference(
                ref["public_future_entity_labels"],outcome["public_future_entity_labels"],"20")
        })
    all_complete=all(v["complete"] for v in outcomes)
    complete_horizons=all(set(v["public_future_entity_labels"])=={"5","20"} for v in outcomes)
    true_h5=any(x["H5_public_entity_difference"] is not None and x[
        "H5_public_entity_difference"]["nonzero_true_public_state_effect"]
        for x in comparisons)
    true_h20=any(x["H20_public_entity_difference"] is not None and x[
        "H20_public_entity_difference"]["nonzero_true_public_state_effect"]
        for x in comparisons)
    costs=[v["true_makespan_ticks"] for v in outcomes]
    return {
        "id":spec["id"],"profile":spec["profile"],
        "regime":spec["regime"],"seed":spec["seed"],
        "labelable":True,
        "actual_action_branch_count":len(outcomes),
        "all_branches_completed":all_complete,
        "full_H5H20_labels":complete_horizons,
        "has_true_H5_action_state_effect":true_h5,
        "has_true_H20_action_state_effect":true_h20,
        "has_true_final_makespan_effect":len(set(costs))>1,
        "baseline_true_makespan_ticks":costs[0],
        "true_makespan_ticks_by_candidate":costs,
        "anchored_same_public_prefix":anchor,
        "factual_current_public_action_graph":reference.graph,
        "real_counterfactual_futures":outcomes,
        "paired_effects_vs_first_proposal":comparisons,
        "replay_verified":"Exact public state and CP-SAT proposal fingerprint at anchor and all prior actions match; only one legal intervention",
    }


def summarize(cases,regime):
    return {
        "regime":regime,"scenarios":len(cases),
        "labelable_groups":sum(x["labelable"] for x in cases),
        "total_legal_branches":sum(x.get("actual_action_branch_count",0) for x in cases),
        "all_completed":sum(x.get("all_branches_completed",False) for x in cases),
        "full_H5H20":sum(x.get("full_H5H20_labels",False) for x in cases),
        "H5_state_action_effect":sum(x.get("has_true_H5_action_state_effect",False) for x in cases),
        "H20_state_action_effect":sum(x.get("has_true_H20_action_state_effect",False) for x in cases),
        "final_true_makespan_effect":sum(x.get("has_true_final_makespan_effect",False) for x in cases),
        "GPU":0,"training":0
    }


def run(regime,out):
    a=json.loads(PROTOCOL.read_text())
    q=json.loads(QUALITY_PROTOCOL.read_text())
    validate_action(a,q)
    if regime not in REGIMES:
        raise ValueError("unregistered regime")
    if out.exists():
        raise FileExistsError("original action fact result is immutable")
    scenario=json.loads(STRESS_PROTOCOL.read_text())
    cases=[]
    started=time.process_time()
    for spec in q["pilot_cases"]:
        if spec["regime"]!=regime:
            continue
        case=paired_case(spec,scenario)
        cases.append(case)
        print("PR55_TRUE_ACTION_INFORMATION_SCENE",spec["id"],
              json.dumps({
                  "labelable":case["labelable"],
                  "H5_state_effect":case.get("has_true_H5_action_state_effect"),
                  "H20_state_effect":case.get("has_true_H20_action_state_effect"),
                  "makespan_effect":case.get("has_true_final_makespan_effect"),
                  "branches":case.get("actual_action_branch_count")
              }),flush=True)
    if len(cases)!=6:
        raise AssertionError("did not run all six development pilot profiles")
    result={
        "schema":"pr55-actual-public-cp_sat-paired-future-action-quality-v1",
        "study_protocol_sha256":hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "quality_protocol_sha256":hashlib.sha256(QUALITY_PROTOCOL.read_bytes()).hexdigest(),
        "regime":regime,"cases":cases,
        "summary":summarize(cases,regime),
        "cpu_process_seconds":time.process_time()-started,
        "limitations":"Source hidden schedule/seed fixed but sequential future stochastic draws may differ after action intervention; no causal effect claimed beyond paired factual simulator outcomes."
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print("PR55_REAL_ACTION_INFORMATION_SUMMARY",
          json.dumps(result["summary"],ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--regime",choices=REGIMES,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    run(a.regime,a.out)


if __name__=="__main__":
    main()
