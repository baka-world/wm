"""Frozen factual AGV arrival-time benchmark, separate from service admission.

First-leg start: 'assigned'; second-leg start: public 'picked_up'.
End: first physical arrival 'transport_arrived' for that transport phase.
The business simulator cannot leak its hidden service countdown into features.

Distinguishes transport duration (start -> arrival) from resource waiting
(arrival -> service_started). All metrics grouped by independent scene seed.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import hashlib
import json
from math import sqrt

from business_transport.observation import BusinessObserver
from business_transport.transport_observability import ObservableTransportSimulator
from experiments.world_model_integration.system_world_model.evaluate_horizons import group_scenario
from lab_transport.decision_value import nearest_batch


ROOT=Path(__file__).resolve().parent
PROTOCOL=ROOT/"transport_protocol.json"


def observed_episode(spec, *,max_ticks):
    business,geo=group_scenario(spec,max_ticks=max_ticks)
    observer=BusinessObserver()
    states={}
    def observe(raw):
        states[raw.tick]=observer.observe(raw)
    def policy(raw,candidates):
        return nearest_batch(raw,candidates,geo)
    result=ObservableTransportSimulator(business,"world",spec["seed"],collect_metrics=True).run(
        dispatch_policy=policy,on_observation=observe)
    if result["summary"]["status"]!="completed":
        raise AssertionError("synthetic episode incomplete")
    return result,states,geo


def transport_rows(events,observations,geometry,*,episode_end):
    """Observable start snapshot and factual end label, pairing by entity/phase."""
    start={}
    arrived={}
    admissions={}
    for event in events:
        typ=event["type"]
        if typ in ("assigned","picked_up"):
            phase="to_pickup" if typ=="assigned" else "to_dropoff"
            key=(event["robot_id"],event["sample_id"],phase)
            if key in start:
                raise AssertionError("duplicated transport start")
            tick=event["tick"]
            if tick not in observations:
                raise AssertionError("start event without same-tick public observation")
            obs=observations[tick]
            robot=next(r for r in obs.robots if r.id==event["robot_id"])
            sample=next(s for s in obs.samples if s.id==event["sample_id"])
            station_id=sample.pickup if phase=="to_pickup" else sample.dropoff
            station=next(st for st in obs.stations if st.id==station_id)
            distance=geometry.distance(robot.position,station.position)
            if distance==geometry.unreachable:
                raise AssertionError("unreachable known target")
            other_active=sum(r.task_id is not None for r in obs.robots if r.id!=robot.id)
            other_near=sum(geometry.distance(r.position,station.position)<=2
                           for r in obs.robots if r.id!=robot.id)
            start[key]={
                "robot_id":robot.id,"sample_id":sample.id,"phase":phase,
                "target_station":station_id,"start_tick":tick,
                "distance":distance,"other_active":other_active,
                "other_near_target":other_near,
                "target_capacity":station.resource_capacity,
                "target_occupied":station.resource_occupied,
                "robot_count":len(obs.robots),
                # A tick is spent at pickup when phase becomes to_dropoff.
                "path_only_baseline_ticks":distance+int(phase=="to_dropoff" and distance>0),
            }
        elif typ=="transport_arrived":
            key=(event["robot_id"],event["sample_id"],event["phase"])
            if key in arrived:
                raise AssertionError("multiple arrivals for same transport leg")
            arrived[key]=event["tick"]
        elif typ=="service_started":
            key=(event["robot_id"],event["sample_id"],event["phase"])
            if key in admissions:
                raise AssertionError("multiple service admissions for same leg")
            admissions[key]=event["tick"]
    rows=[]
    if set(arrived)-set(start):
        raise AssertionError("arrival without factual transport start")
    for key,features in sorted(start.items()):
        arrival=arrived.get(key)
        admission=admissions.get(key)
        if arrival is not None and arrival<features["start_tick"]:
            raise AssertionError("arrival precedes dispatched transport")
        if arrival is not None and admission is not None and admission<arrival:
            raise AssertionError("service admission preceded actual arrival")
        duration=arrival-features["start_tick"] if arrival is not None else episode_end-features["start_tick"]
        baseline=features["path_only_baseline_ticks"]
        if duration<0:raise AssertionError("invalid elapsed transport duration")
        row={
            **features,
            "arrived_tick":arrival,
            "service_admitted_tick":admission,
            "duration_ticks":duration,
            "right_censored":arrival is None,
            "delay_ticks":max(0,duration-baseline) if arrival is not None else None,
            "delay_label":int(duration>baseline) if arrival is not None else None,
            "admission_wait_ticks":admission-arrival if admission is not None and arrival is not None else None,
        }
        rows.append(row)
    return rows


def scenario_metrics(rows):
    complete=[r for r in rows if not r["right_censored"]]
    if not complete:
        raise AssertionError("no observed completed transport legs")
    n=len(complete)
    diff=[r["duration_ticks"]-r["path_only_baseline_ticks"] for r in complete]
    return {
        "legs":n,
        "censored":len(rows)-n,
        "baseline_mae_ticks":sum(abs(x) for x in diff)/n,
        "baseline_rmse_ticks":sqrt(sum(x*x for x in diff)/n),
        "delay_rate":sum(r["delay_label"] for r in complete)/n,
        "resource_wait_legs":sum((r["admission_wait_ticks"] or 0)>0 for r in complete),
        "mean_resource_wait":sum(r["admission_wait_ticks"] or 0 for r in complete)/n,
    }


def evaluate(protocol=PROTOCOL):
    settings=json.loads(Path(protocol).read_text())
    if settings["protocol"]!="TRANSPORT_PUBLIC_EVENTS_V1":
        raise ValueError("frozen transport protocol changed")
    out=[]
    for group in settings["groups"]:
        result,states,geo=observed_episode(group,max_ticks=settings["max_ticks"])
        rows=transport_rows(result["events"],states,geo,episode_end=result["summary"]["ticks"])
        out.append({
            "group":group["id"],"split":group["split"],"seed":group["seed"],
            "metrics":scenario_metrics(rows),
            "rows":rows,
            "events_digest":hashlib.sha256(json.dumps(
                result["events"],sort_keys=True).encode()).hexdigest(),
        })
    splits={}
    for split in sorted({x["split"] for x in out}):
        v=[x["metrics"] for x in out if x["split"]==split]
        splits[split]={"groups":len(v),"legs":sum(x["legs"] for x in v),
            "censored":sum(x["censored"] for x in v),
            "baseline_mae_group_mean":sum(x["baseline_mae_ticks"] for x in v)/len(v),
            "delay_rate_group_mean":sum(x["delay_rate"] for x in v)/len(v),
            "resource_wait_legs":sum(x["resource_wait_legs"] for x in v),
        }
    return {"protocol":settings["protocol"],"groups":out,"summary":splits}


def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError("write-once factual evidence")
    value=evaluate()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n")
    print("TRANSPORT_FACTUAL_BASELINE",json.dumps(value["summary"],ensure_ascii=False))


if __name__=="__main__":
    main()
