"""One-shot source-library AGV delay inference, frozen train/val/test.

The upstream method is pgmpy BayesianEstimator+VariableElimination (MIT).
Scenario-level averages avoid robot/leg pseudoreplication. No simulator
mechanism is changed and no claims of measured enterprise obstacles.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from business_transport.world_model.transport_bayes import TransportBayesModel
from experiments.world_model_integration.transport_events.benchmark import (
    PROTOCOL, observed_episode, scenario_metrics, transport_rows,
)


def _split_rows(cfg, groups):
    result=[]
    for spec in groups:
        sim,states,geo=observed_episode(spec,max_ticks=cfg["max_ticks"])
        rows=transport_rows(sim["events"],states,geo,episode_end=sim["summary"]["ticks"])
        result.append({
            "id":spec["id"],"split":spec["split"],"seed":spec["seed"],"rows":rows,
            "sim_events_sha256":hashlib.sha256(
                json.dumps(sim["events"],sort_keys=True).encode()).hexdigest(),
        })
    return result


def score_group(group, model):
    valid=[r for r in group["rows"] if not r["right_censored"]]
    if not valid:raise AssertionError("unscorable holdout group")
    p_constant=model.positive_legs/model.total_legs
    sq={"distance_no_delay":0,"train_prevalence":0,"pgmpy":0}
    mae={"distance_no_delay":0,"train_prevalence":0,"pgmpy":0}
    for row in valid:
        observed=row["duration_ticks"]
        y=row["delay_label"]
        p=model.predict_delay_probability(row)
        for key,pred_p in (
            ("distance_no_delay",0.),
            ("train_prevalence",p_constant),
            ("pgmpy",p)):
            sq[key]+=(pred_p-y)**2
            forecast=row["path_only_baseline_ticks"]+pred_p*model.positive_delay_ticks
            mae[key]+=abs(forecast-observed)
    return {
        "id":group["id"],"seed":group["seed"],"split":group["split"],
        "legs":len(valid),"censored":len(group["rows"])-len(valid),
        "delay_rate":sum(r["delay_label"] for r in valid)/len(valid),
        "brier":{k:v/len(valid) for k,v in sq.items()},
        "duration_mae":{k:v/len(valid) for k,v in mae.items()},
        "sim_events_sha256":group["sim_events_sha256"],
    }


def evaluate(*,proto=PROTOCOL,save_model=None,cpu_budget=240):
    cfg=json.loads(Path(proto).read_text())
    if (cfg["protocol"]!="TRANSPORT_PUBLIC_EVENTS_V1"
            or [sum(x["split"]==split for x in cfg["groups"])
                for split in ("train","validation","holdout")] != [24,8,12]):
        raise ValueError("frozen transport split changed")
    started=time.process_time()
    groups=_split_rows(cfg,cfg["groups"])
    train=[row for group in groups if group["split"]=="train" for row in group["rows"]]
    model=TransportBayesModel.fit(train)
    if save_model is not None:
        path=Path(save_model)
        if path.exists():
            raise FileExistsError("cannot overwrite fitted model evidence")
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(model.to_payload(),indent=2)+"\n")
    reports=[score_group(group,model) for group in groups]
    summary={}
    for split in ("train","validation","holdout"):
        selected=[r for r in reports if r["split"]==split]
        summary[split]={
            "groups":len(selected),"legs":sum(x["legs"] for x in selected),
            "delay_rate_group_mean":sum(x["delay_rate"] for x in selected)/len(selected),
            "brier":{key:sum(x["brier"][key] for x in selected)/len(selected)
                     for key in ("distance_no_delay","train_prevalence","pgmpy")},
            "duration_mae":{key:sum(x["duration_mae"][key] for x in selected)/len(selected)
                            for key in ("distance_no_delay","train_prevalence","pgmpy")},
        }
    elapsed=time.process_time()-started
    if elapsed>cpu_budget:
        raise TimeoutError("CPU cap exceeded, benchmark non-compliant")
    return {
        "protocol":cfg["protocol"],
        "method":"Tang and Wang 2025 AGV Bayesian risk adaptation; pgmpy 1.0.0 (MIT)",
        "fitted_reference":model.to_payload(),
        "group_reports":reports,
        "summary":{
            "splits":summary,
            "train_legs":model.total_legs,
            "train_positive":model.positive_legs,
            "train_delay_rate":model.positive_legs/model.total_legs,
            "cpu_seconds":elapsed,"gpu":0,
            "limits":"Synthetic collision/delay, not genuine AGV obstacles or hospital data",
        },
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--trained",type=Path,required=True)
    args=p.parse_args()
    if args.out.exists() or args.trained.exists():
        raise FileExistsError("evidence outputs are write-once")
    result=evaluate(save_model=args.trained)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+"\n")
    print("PR25_PGMPY_TRANSPORT_BAYES_EVAL",json.dumps(result["summary"],ensure_ascii=False))


if __name__=="__main__":
    main()
