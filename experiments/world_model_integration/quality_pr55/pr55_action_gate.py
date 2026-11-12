"""PR55 immutable decision-relevant data INFORMATION gate, not model training."""
from __future__ import annotations
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json

from experiments.world_model_integration.quality_pr55.pr55_generate import (
    PROTOCOL as QUALITY_PROTOCOL,REGIMES,validate
)
from experiments.world_model_integration.quality_pr55.pr55_action_quality import (
    PROTOCOL,validate_action,state_difference
)


def aggregate(paths,qa_path,out):
    if out.exists():
        raise FileExistsError("do not replace independent factual evidence")
    q=json.loads(QUALITY_PROTOCOL.read_text())
    p=json.loads(PROTOCOL.read_text())
    validate_action(p,q)
    qa=json.loads(qa_path.read_text())
    if (qa["schema"]!="pr55-raw-world-dynamics-data-quality-admission-v1"
        or qa["decision"]!="PASS_NEXT_PHASE_DESIGN"
        or qa["protocol_sha256"]!=hashlib.sha256(QUALITY_PROTOCOL.read_bytes()).hexdigest()):
        raise AssertionError("initial genuine data quality gate must pass first")
    if len(paths)!=3:
        raise ValueError("all three actual stress action studies required")
    cases=[]
    digest={}
    for path in paths:
        d=json.loads(path.read_text())
        if (d["schema"]!="pr55-actual-public-cp_sat-paired-future-action-quality-v1"
            or d["study_protocol_sha256"]!=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
            or d["quality_protocol_sha256"]!=qa["protocol_sha256"]
            or d["regime"] in digest):
            raise AssertionError("source action branch dataset provenance changed")
        if len(d["cases"])!=6 or d["summary"]["GPU"]!=0 or d["summary"]["training"]!=0:
            raise AssertionError("invalid preregistered factual action evidence")
        digest[d["regime"]]=hashlib.sha256(path.read_bytes()).hexdigest()
        cases.extend(d["cases"])
    if (len(cases)!=18 or set(digest)!=set(REGIMES) or
        {x["id"] for x in cases}!=set(p["ids"])):
        raise AssertionError("pilot case grouping is incomplete/duplicate")
    groups=Counter(c["regime"] for c in cases)
    labelable=sum(x["labelable"] for x in cases)
    completed=sum(x.get("all_branches_completed",False) for x in cases)
    h5h20=sum(x.get("full_H5H20_labels",False) for x in cases)
    h5=sum(x.get("has_true_H5_action_state_effect",False) for x in cases)
    h20=sum(x.get("has_true_H20_action_state_effect",False) for x in cases)
    makespan=sum(x.get("has_true_final_makespan_effect",False) for x in cases)
    branches=sum(x.get("actual_action_branch_count",0) for x in cases)
    effects=[]
    for x in cases:
        if not x["labelable"]:
            continue
        graph=x["factual_current_public_action_graph"]
        if (len(graph["candidate_actions"])!=x["actual_action_branch_count"]
            or len(x["real_counterfactual_futures"])!=x["actual_action_branch_count"]):
            raise AssertionError("CP-SAT proposed branch actions not verified")
        if x["anchored_same_public_prefix"]["candidate_count"]!=x["actual_action_branch_count"]:
            raise AssertionError("candidate count changed across true replay")
        if x["anchored_same_public_prefix"]["tick"]<p["anchor_tick_min"] or x[
            "anchored_same_public_prefix"]["tick"]>p["anchor_tick_max"]:
            raise AssertionError("lookahead action choice used a different anchor")
        if "future" in str(graph.get("private_fault_schedule","")):
            raise AssertionError("private information in public graph")
        for intervention in x["paired_effects_vs_first_proposal"]:
            for horizon in ("5","20"):
                reported=intervention[f"H{horizon}_public_entity_difference"]
                true=state_difference(
                    x["real_counterfactual_futures"][0]["public_future_entity_labels"],
                    x["real_counterfactual_futures"][intervention["alternative_index"]]["public_future_entity_labels"],
                    horizon)
                if reported!=true:
                    raise AssertionError("future state effect was not computed from true branch labels")
            effects.append(intervention)
    checks={
        "source_18_dev_scenarios_and_all_stress_profiles":len(cases)==18 and all(
            groups[r]==6 for r in REGIMES),
        "exact_same_public_state_and_candidates_prefix":True,
        "all_18_have_multiple_legal_choices":labelable>=p["minimum_labelable_scenes"],
        "all_18_branches_completed":completed>=p["minimum_labelable_scenes"],
        "all_18_have_genuine_H5H20_public_targets":h5h20>=p["minimum_complete_h5h20_scenes"],
        "H20_true_entity_change_is_frequent":h20>=p["minimum_scenes_with_true_h20_entity_effect"],
        "final_business_makespan_actions_nontrivial":makespan>=p["minimum_scenes_with_true_final_makespan_effect"],
        "earlier_event_quality_passed":qa["decision"]=="PASS_NEXT_PHASE_DESIGN"
    }
    rejects=[k for k,v in checks.items() if not v]
    report={
        "schema":"pr55-same-public-state-legal-action-information-gate-v1",
        "decision":"PASS_ACTION_DATA_SIGNAL_DEV_ONLY" if not rejects
                   else "NO_GO_DECISION_AWARE_DATA_EXPANSION",
        "checks":checks,"failed":rejects,
        "source_action_sha256":digest,
        "source_original_full_public_quality_decision":qa["decision"],
        "development_scenarios":18,
        "legal_action_branch_groups":labelable,
        "actual_counterfactual_factual_branches":branches,
        "actual_legitimate_action_comparisons":len(effects),
        "full_real_completion_groups":completed,
        "complete_h5h20_truth_groups":h5h20,
        "true_H5_entity_action_effect_groups":h5,
        "true_H20_entity_action_effect_groups":h20,
        "different_true_makespan_groups":makespan,
        "independent_blind_test_executed":False,
        "GPU":0,"new_model_training":0,
        "topology_OOD_gate":"NOT PASSED: all profiles still same 7x5 geometry",
        "training_corpus_status":"DEVELOPMENT_ONLY_NOT_RELEASED_FOR_BLIND_TRAINING",
        "limits":"Identical public prefix and candidate pool, one legal dispatch action intervention, common scenario seed/schedule; changing trajectory order may change future stochastic draws.",
        "individual_scenarios":[{
            "id":x["id"],"profile":x["profile"],"regime":x["regime"],
            "labelable":x["labelable"],
            "H5_effect":x.get("has_true_H5_action_state_effect",False),
            "H20_effect":x.get("has_true_H20_action_state_effect",False),
            "makespan_effect":x.get("has_true_final_makespan_effect",False),
            "branch_count":x.get("actual_action_branch_count",0),
            "all_complete":x.get("all_branches_completed",False)
        } for x in cases]
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print("PR55_ACTION_INFORMATION_GATE",json.dumps({
        k:v for k,v in report.items() if k!="individual_scenarios"
    },ensure_ascii=False),flush=True)
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--previous-quality",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    aggregate(sorted(args.source.glob("*/data.json")),args.previous_quality,args.out)


if __name__=="__main__":
    main()
