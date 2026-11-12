"""Read-only decision-value interpretation of frozen PR28 evidence.

The PR28 evaluator measured H20 predicted stage progress and whole-episode
factual makespan separately. This audit tests their directional agreement
without rerunning any simulator or changing seeds, deadlines, actions or
predictor. No use of these factual results is allowed for online action
selection. Do not construe 5/5 stage-progress agreement as makespan evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def sign(x):
    return (x>0)-(x<0)


def audit(data):
    if data.get("schema")!="RHWM_PR28_DYNAMIC_ACTION_IDENTIFIABILITY_20261112_V1":
        raise ValueError("wrong frozen report")
    output=[]
    for g in data["groups"]:
        if g["second_outcome"] is None:
            output.append({"group":g["group"],"regime":g["regime"],
                           "reason":"no distinct legal candidate"})
            continue
        true_delta=g["first_action_effect"]["delta_end_tick_second_minus_first"]
        factual_stage=(g["second_realized_completed_stages"]["20"]-
                       g["first_realized_completed_stages"]["20"])
        forecast=g["frozen_model_predictions"]
        pred_stage=(forecast["1"]["20"]["expected_completed_stages"]-
                    forecast["0"]["20"]["expected_completed_stages"])
        output.append({
            "group":g["group"],"regime":g["regime"],
            "makespan_delta_second_minus_first":true_delta,
            "factual_h20_stage_completion_delta":factual_stage,
            "predicted_h20_stage_completion_delta":pred_stage,
            "actual_makespan_sensitive":bool(g["first_action_effect"]["makespan_sensitive"]),
            "predicted_stage_non_tie":abs(pred_stage)>1e-9,
            "factual_stage_non_tie":factual_stage!=0,
            "stage_direction_correct":(
                sign(pred_stage)==sign(factual_stage)
                if pred_stage!=0 and factual_stage!=0 else None),
            "predicted_stage_direction_matches_makespan":(
                sign(pred_stage)==-sign(true_delta)
                if abs(pred_stage)>1e-9 and true_delta!=0 else None),
            "realized_stage_direction_matches_makespan":(
                sign(factual_stage)==-sign(true_delta)
                if factual_stage!=0 and true_delta!=0 else None),
        })
    by_regime={}
    for regime in sorted({g["regime"] for g in output}):
        current=[g for g in output if g["regime"]==regime]
        def counted(field):
            return {"eligible":sum(g.get(field) is not None for g in current),
                    "matches":sum(g.get(field) is True for g in current)}
        by_regime[regime]={
            "groups":len(current),
            "actual_makespan_sensitive":sum(g.get("actual_makespan_sensitive",False)
                                            for g in current),
            "predicted_non_tie_groups":sum(g.get("predicted_stage_non_tie",False)
                                            for g in current),
            "stage_vs_stage":counted("stage_direction_correct"),
            "predicted_stage_vs_final_makespan":counted(
                "predicted_stage_direction_matches_makespan"),
            "factual_stage_vs_final_makespan":counted(
                "realized_stage_direction_matches_makespan"),
        }
    return {
        "schema":"rhwm-pr28-h20-utility-audit-v1",
        "source":data["schema"],
        "scenario_groups":output,
        "by_regime":by_regime,
        "gpu":0,"new_factual_simulations":0,"new_model_inference":0,"new_training":0,
        "interpretation":"Descriptive, on already examined 12 synthetic groups. Not blind validation, no prospective policy benefit."
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--infile",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError("read-only audit output is write-once")
    result=audit(json.loads(args.infile.read_text()))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print("PR28_H20_UTILITY_AUDIT",json.dumps(result["by_regime"],ensure_ascii=False))


if __name__=="__main__":
    main()
