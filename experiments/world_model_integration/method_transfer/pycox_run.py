"""One-shot source-method transfer using official pycox LogisticHazard.

Everything learned here comes from upstream pycox/torchtuples. This module
only assembles public service-start covariates and actually executed session
durations, then runs independent scenario-group validation against Rule, KM,
frozen V4 and pycox. The test data never selects epochs or duration cuts.

Reference: Kvamme & Borgan (2019), arXiv:1910.06724.
Source: https://github.com/havakv/pycox/blob/master/pycox/models/logistic_hazard.py
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import time

from business_transport.world_model import HybridSystemWorldModel
from business_transport.world_model.kaplan_meier import KaplanMeierServiceModel
from business_transport.world_model.pycox_survival import (
    PycoxLogisticHazardServiceModel, public_service_features,
)
from experiments.world_model_integration.method_transfer.compare_km_v4 import (
    extract_sessions, event_brier_on_factual, state_rollout_on_factual,
)
from experiments.world_model_integration.system_world_model.evaluate_horizons import (
    METRICS, factual_episode, file_sha, group_scenario,
)

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "pycox_protocol.json"
ARMS = ("rule","kaplan_meier","v4","pycox_logistic_hazard")


def _key(event):
    return (event["robot_id"], event["sample_id"], event["phase"], event["station_id"])


def public_service_rows(states, events, *, final_tick):
    """Public start features and factual stop/censor labels, no private countdown."""
    active = {}
    rows = []
    for event in events:
        kind = event["type"]
        if kind == "service_started":
            key = _key(event)
            if key in active:
                raise AssertionError("duplicate started service session")
            if event["tick"] not in states:
                raise AssertionError("service start lacks public observation")
            features = public_service_features(states[event["tick"]], event).tolist()
            active[key] = (event["tick"], features)
        elif kind == "service_finished":
            key = _key(event)
            if key not in active:
                raise AssertionError("finished service absent from started sessions")
            tick, features = active.pop(key)
            duration = event["tick"] - tick
            if duration < 1 or duration != event["observed_ticks"]:
                raise AssertionError("inconsistent factual finished duration")
            rows.append({
                "station_id":event["station_id"],"phase":event["phase"],
                "duration":duration,"event_observed":True,
                "features":features,
            })
    for key, (tick, features) in sorted(active.items()):
        age = final_tick - tick
        if age > 0:
            rows.append({
                "station_id":key[3],"phase":key[2],
                "duration":age,"event_observed":False,
                "features":features,
            })
    return rows


def _load_group(spec, proto):
    business, geo = group_scenario(spec, max_ticks=proto["scene_max_ticks"])
    states, actions, factual = factual_episode(business, geo, spec["seed"])
    rows = public_service_rows(states, factual["events"],
                               final_tick=factual["summary"]["ticks"])
    km_rows = extract_sessions(factual["events"],
                               end_tick=factual["summary"]["ticks"])
    if len(rows) != len(km_rows):
        raise AssertionError("KM and pycox observe different factual service sessions")
    if not rows:
        raise AssertionError("no factual service sessions")
    return business,geo,states,actions,factual,rows


def _group_summaries(results, horizons):
    result = {}
    for split in sorted({r["split"] for r in results}):
        groups = [r for r in results if r["split"]==split]
        by_h = {}
        for horizon in horizons:
            arms = {}
            for arm in ARMS:
                per_group = []
                for group in groups:
                    records = [r["metrics"] for r in group["arms"][arm]["state_rows"]
                               if r["horizon"] == horizon]
                    if not records:
                        continue
                    per_group.append({key:sum(row[key] for row in records)/len(records)
                                      for key in METRICS})
                if not per_group:
                    raise AssertionError("no coverage for predeclared horizon")
                arms[arm]={"groups":len(per_group),
                           "metrics":{k:sum(g[k] for g in per_group)/len(per_group)
                                      for k in METRICS}}
            by_h[str(horizon)] = arms
        by_brier = {}
        for arm in ARMS[1:]:
            vals = [g["arms"][arm]["service_event"]
                    for g in groups]
            scored = [g["brier"] for g in vals if g["brier"] is not None]
            by_brier[arm] = {
                "groups":len(scored),
                "eligible_factual_queries":sum(g["eligible_queries"] for g in vals),
                "scenario_mean":sum(scored)/len(scored) if scored else None,
            }
        result[split]={"groups":len(groups),"by_horizon":by_h,
                       "service_event_brier":by_brier}
    return result


def run(*,checkpoint,proto_path=PROTOCOL,checkpoint_output=None,cpu_budget=480):
    proto=json.loads(Path(proto_path).read_text())
    if (proto["protocol"]!="RHWM_PYCOX_LOGISTIC_HAZARD_TRANSFER_20261112_V1"
            or proto["horizons"]!=[1,5,10]
            or proto["particles"]!=3
            or [len(proto[k]) for k in ("train_groups","validation_groups","iid_test_groups","shift_test_groups")] != [48,12,24,12]):
        raise ValueError("published model transfer protocol altered")
    if file_sha(checkpoint)!=proto["checkpoint_v4_sha256"]:
        raise ValueError("frozen V4 checkpoint mismatch")
    all_groups=sum((proto[k] for k in
                    ("train_groups","validation_groups","iid_test_groups","shift_test_groups")),[])
    seed_set={g["seed"] for g in all_groups}
    if len(seed_set)!=len(all_groups):
        raise AssertionError("scenario seeds leaked between groups")
    prior=json.loads((ROOT.parent/"system_world_model"/"generalization_protocol.json").read_text())
    previous=json.loads((ROOT/"km_reference_protocol.json").read_text())
    used_prior={g["seed"] for g in prior["groups"]}
    used_prior.update(g["seed"] for k in ("train_groups","test_groups") for g in previous[k])
    if not seed_set.isdisjoint(used_prior):
        raise AssertionError("reuse of previous developed/tested scenario seeds")
    import pycox
    import torchtuples
    from torch import set_num_threads, save
    assert pycox.__version__=="0.3.0"
    set_num_threads(1)
    started=time.process_time()
    train,valid=[],[]
    groups_covered=[]
    for split in ("train_groups","validation_groups"):
        for spec in proto[split]:
            if time.process_time()-started>cpu_budget:
                raise TimeoutError("CPU budget exceeded preparing public data")
            _,_,states,_,factual,rows=_load_group(spec,proto)
            if split=="train_groups":
                train.extend(rows)
            else:
                valid.extend(rows)
            groups_covered.append({
                "id":spec["id"],"split":split,"seed":spec["seed"],
                "factual_service_sessions":len(rows),
                "censored":sum(not r["event_observed"] for r in rows),
            })
    if min(len(train),len(valid))<10:
        raise AssertionError("insufficient observed services for baseline training")
    # Official upstream model/loss + upstream best-validation-weights callback.
    trained=PycoxLogisticHazardServiceModel.train(train,valid,settings=proto)
    km=KaplanMeierServiceModel(train)
    if checkpoint_output is not None:
        target=Path(checkpoint_output)
        if target.exists():
            raise FileExistsError("do not replace archived trained baseline")
        target.parent.mkdir(parents=True,exist_ok=True)
        save({"state_dict":trained.model.net.state_dict(),
              "time_bins":trained.cuts.tolist(),
              "model_metadata":trained.training_summary,
              "protocol_sha256":file_sha(proto_path),
              "original_reference":"pycox.models.LogisticHazard 0.3.0, torchtuples 0.2.2"},
             target)
    holdout=[]
    for cohort in ("iid_test_groups","shift_test_groups"):
        for spec in proto[cohort]:
            if time.process_time()-started>cpu_budget:
                raise TimeoutError("CPU budget exceeded before next independent test group")
            business,geo,states,actions,factual,rows=_load_group(spec,proto)
            result={
                "group_id":spec["id"],"seed":spec["seed"],
                "split":"independent_iid" if cohort=="iid_test_groups" else "synthetic_shift",
                "events_sha256":hashlib.sha256(json.dumps(
                    factual["events"],sort_keys=True).encode()).hexdigest(),
                "factual_sessions":len(rows),
                "arms":{},
            }
            for arm in ARMS:
                model=HybridSystemWorldModel(
                    geo,business.rules,kind=arm,
                    checkpoint=checkpoint if arm=="v4" else None,
                    experimental=(arm=="v4"),
                    service_baseline=km if arm=="kaplan_meier" else
                    trained if arm=="pycox_logistic_hazard" else None)
                event=(
                    event_brier_on_factual(model,states,actions)
                    if arm!="rule" else {"eligible_queries":0,"brier":None})
                forecasts=state_rollout_on_factual(model,states,actions,geo,{
                    "fixed_anchors":proto["anchors"],
                    "fixed_horizons":proto["horizons"],
                    "particles":proto["particles"],
                    "seed_for_model":510001+spec["seed"]*7,
                })
                result["arms"][arm]={
                    "service_event":event,
                    "state_rows":forecasts,
                }
            holdout.append(result)
    total=time.process_time()-started
    return {
        "protocol":proto["protocol"],"protocol_sha256":file_sha(proto_path),
        "original_frozen_v4_sha256":file_sha(checkpoint),
        "method_reference":"Kvamme and Borgan 2019; pycox LogisticHazard 0.3.0",
        "upstream_implementation":"pycox.models.LogisticHazard / NLLLogistiHazardLoss",
        "training":trained.training_summary,
        "data":{"train_sessions":len(train),"validation_sessions":len(valid),
                "groups_covered":groups_covered},
        "group_results":holdout,
        "summary":{
            "train_groups":len(proto["train_groups"]),
            "validation_groups":len(proto["validation_groups"]),
            "iid_test_groups":len(proto["iid_test_groups"]),
            "shift_test_groups":len(proto["shift_test_groups"]),
            "grouped_metrics":_group_summaries(holdout,proto["horizons"]),
            "cpu_seconds":total,"gpu":0,
            "only_upstream_training_algorithm":True,
            "factual_input":"BusinessObserver current public state + actually completed service session logs",
            "model_selection":"only validation NLL; no test optimization",
            "caveat":"simulation duration-jitter, no real AGV service records; no real-world efficacy claim",
        },
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--trained-output",type=Path,required=True)
    ap.add_argument("--cpu-budget",type=int,default=480)
    args=ap.parse_args()
    if args.out.exists() or args.trained_output.exists():
        raise FileExistsError("research evidence is immutable per run")
    result=run(checkpoint=args.checkpoint,checkpoint_output=args.trained_output,
               cpu_budget=args.cpu_budget)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    s=result["summary"]
    digest={
        "train_sessions":result["data"]["train_sessions"],
        "validation_sessions":result["data"]["validation_sessions"],
        "train_epochs":result["training"]["epochs_executed"],
        "groups":s["iid_test_groups"]+s["shift_test_groups"],
        "cpu_seconds":s["cpu_seconds"],
        "grouped_metrics":s["grouped_metrics"],
    }
    print("OFFICIAL_PYCOX_TRANSFER_RESULT",json.dumps(digest,ensure_ascii=False))


if __name__=="__main__":
    main()
