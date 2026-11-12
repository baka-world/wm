"""Independent PR31 evaluation after checkpoint choice frozen on Validation.

No training or model selection on unseen Test. Compare original V4, warm-start
and scratch event Brier/logloss and first-action H20 limited-candidate regret
with identical paired factual worlds. Saved checkpoint compatibility uses
existing load_v4 and HybridSystemWorldModel, not a custom predictor.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import torch

from business_transport.session_features_v3 import INPUT_SCHEMA_V3
from experiments.world_model_integration.train_v4 import load_v4
from experiments.world_model_integration.posttraining.pr31_dataset import PROTOCOL
from experiments.world_model_integration.posttraining.repro_environment import sha
from experiments.world_model_integration.horizon_generalization.pr29_evaluate import (
    paired_truth,forecast_pair,_select,
)
from lab_transport.business_session_v3 import probs,stats

ARMS=("frozen_v4","warmstart_v4","scratch_v4")


def _decision_config(cfg):
    return {
        "max_ticks":240,
        "bounded_counterfactual":{
            "candidates":{"limit":4,"wall_seconds":0.08,"deterministic_seconds":0.04}},
        "first_action_pair":{
            "solver":{"limit":4,"wall_seconds":0.08,
                      "deterministic_seconds":0.04,"seed":1810}},
        "prediction":{"particles":3},
    }


def _test_event_metrics(rows,meta,checkpoints):
    if not rows or "independent_test" not in meta:
        raise AssertionError("newly generated independent test missing")
    if len(meta["independent_test"])!=16:
        raise AssertionError("unexpected independent test cohort")
    groupmeta={g["seed"]:g for g in meta["independent_test"]}
    out={}
    for arm,checkpoint in checkpoints.items():
        model,info=load_v4(checkpoint,experimental=True)
        predictions=probs(model,rows,INPUT_SCHEMA_V3)
        full=stats(rows,predictions)
        by_regime={}
        for regime in ("medium3","heavy5","shift5","low2"):
            values=[]
            for item in [g for g in meta["independent_test"]
                         if g["id"].split("_")[2]==regime]:
                idx=[k for k,r in enumerate(rows) if r["seed"]==item["seed"]]
                if not idx:
                    raise AssertionError("missing independent group labels")
                values.append(stats([rows[k] for k in idx],[predictions[k] for k in idx]))
            by_regime[regime]={
                "groups":len(values),
                "mean_group_brier":sum(v["brier"] for v in values)/len(values),
                "mean_group_log_loss":sum(v["log_loss"] for v in values)/len(values),
            }
        out[arm]={
            "test_overall":{k:full[k] for k in
                            ("count","positive","brier","log_loss","ece5")},
            "by_regime":by_regime,
            "checkpoint_sha256":sha(checkpoint),
            "training_arm":info.get("training_arm","original_v4"),
        }
    return out


def _decision_group(spec,cfg,checkpoints):
    factual=paired_truth(spec,_decision_config(cfg))
    first=factual[0]
    results={
        "group":spec["id"],"regime":spec["regime"],"seed":spec["seed"],
        "number_of_legally_distinct_candidates":len(factual),
        "factual":[{
            "candidate_index":i,
            "makespan_ticks":branch["outcomes"]["end_tick"],
            "late_jobs":branch["outcomes"]["late_jobs"],
            "censored":branch["outcomes"]["censored"],
            "resource_wait_ticks":branch["outcomes"]["resource_wait_ticks"],
        } for i,branch in enumerate(factual)],
        "models":{},
    }
    complete=all(not b["outcomes"]["censored"] for b in factual)
    if not complete:
        return results
    actual=[b["outcomes"]["end_tick"] for b in factual]
    late=[b["outcomes"]["late_jobs"] for b in factual]
    for arm,checkpoint in checkpoints.items():
        start=time.process_time()
        forecasts=forecast_pair(
            first,_decision_config(cfg),checkpoint,arm="v4",horizon=20)
        selected=_select(forecasts)
        elapsed=time.process_time()-start
        actual_true_delta=actual[selected]-min(actual)
        if actual_true_delta<0:raise AssertionError("negative finite-candidate regret")
        results["models"][arm]={
            "selected":selected,
            "forecast_expected_completed":[f["expected_completed"] for f in forecasts],
            "limited_candidate_regret_ticks":actual_true_delta,
            "selected_factual_makespan_ticks":actual[selected],
            "selected_factual_late_jobs":late[selected],
            "prediction_cpu_seconds_including_model_load":elapsed,
        }
    return results


def evaluate(test_path,selection_path,old_v4,training_dir,output):
    cfg=json.loads(PROTOCOL.read_text())
    if cfg["protocol"]!="RHWM_PR31_V4_WARMSTART_SCRATCH_REPRO_TRAIN_20261112_V1":
        raise ValueError("precommitted protocol altered")
    selection=json.loads(Path(selection_path).read_text())
    if selection["protocol"]!=cfg["protocol"]:
        raise ValueError("validation choice belongs to different protocol")
    if selection["data_sha256"]==sha(test_path):
        raise AssertionError("validation data unexpectedly equal to test data")
    data=json.loads(Path(test_path).read_text())
    if data["schema"]!="rhwm-pr31-factual-service-query-v3" or data["part"]!="independent_test":
        raise ValueError("test material not independently generated")
    if data["protocol_sha256"]!=sha(PROTOCOL):
        raise ValueError("changed independent data protocol")
    if sha(old_v4)!=cfg["old_checkpoint_sha256"]:
        raise ValueError("historical V4 source checkpoint changed")
    ckpts={
        "frozen_v4":Path(old_v4),
        "warmstart_v4":Path(training_dir)/"warmstart_v4-best.pt",
        "scratch_v4":Path(training_dir)/"scratch_v4-best.pt",
    }
    for arm in ARMS:
        if not ckpts[arm].is_file():raise FileNotFoundError(ckpts[arm])
    torch.set_num_threads(1)
    began=time.process_time()
    event=_test_event_metrics(data["independent_test"],
                              data["source_groups"],ckpts)
    decision=[]
    for spec in cfg["decision_test_groups"]:
        if time.process_time()-began>360:raise TimeoutError("decision test CPU limit exceeded")
        decision.append(_decision_group(spec,cfg,ckpts))
    summary={}
    for arm in ARMS:
        own=[g for g in decision if arm in g["models"]]
        by_regime={}
        for regime in ("light","medium","heavy","shift"):
            cases=[g for g in own if g["regime"]==regime]
            if not cases:raise AssertionError("empty independent business regime")
            vals=[g["models"][arm]["limited_candidate_regret_ticks"] for g in cases]
            by_regime[regime]={
                "groups":len(cases),"mean_candidate_regret_ticks":sum(vals)/len(vals),
                "mean_makespan_if_first_action_selected":sum(
                    g["models"][arm]["selected_factual_makespan_ticks"] for g in cases)/len(cases),
                "mean_prediction_cpu_seconds_including_model_load":sum(
                    g["models"][arm]["prediction_cpu_seconds_including_model_load"]
                    for g in cases)/len(cases),
            }
        summary[arm]={
            "event_brier":event[arm]["test_overall"]["brier"],
            "event_logloss":event[arm]["test_overall"]["log_loss"],
            "decision_by_regime":by_regime,
        }
    result={
        "schema":"pr31-independent-event-and-h20-decision-v1",
        "protocol_sha256":sha(PROTOCOL),
        "independent_dataset_sha256":sha(test_path),
        "validation_selection_sha256":sha(selection_path),
        "selected_from_validation_only":selection["best_by_validation_brier"],
        "event_results":event,
        "decision_group_results":decision,
        "summary":{
            "training_comparison":summary,
            "independent_event_groups":16,
            "independent_decision_groups":len(decision),
            "gpu":0,"new_training_during_test":0,
            "cpu_process_seconds":time.process_time()-began,
            "model_promotion":"No automatic promotion; independent metrics and workload tradeoffs must be judged together",
        },
    }
    target=Path(output)
    if target.exists():raise FileExistsError("independent metrics write-once")
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print("PR31_INDEPENDENT_EVALUATION",json.dumps({
        "chosen_validation":result["selected_from_validation_only"],
        "cpu":result["summary"]["cpu_process_seconds"],
        "summary":summary},ensure_ascii=False))
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--independent",required=True,type=Path)
    ap.add_argument("--selection",required=True,type=Path)
    ap.add_argument("--original",required=True,type=Path)
    ap.add_argument("--training-dir",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    a=ap.parse_args()
    evaluate(a.independent,a.selection,a.original,a.training_dir,a.out)


if __name__=="__main__":
    main()
