"""Factual business-service session dataset builder, fixed grouped split.

Only BusinessSimulator factual (t, actual_action, t+1) transitions enter
labels. Invalid/paused/not-yet-started sessions are counted, not mislabeled
as failures. No PR15/16 old confirmation episodes are imported.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from business_transport.scenario import compile_scenario
from business_transport.session_events import eligible_sessions, QUERY_LABEL_VERSION
from business_transport.state_features import SCHEMA_VERSION, build_business_features
from business_transport.simulation import BusinessSimulator
from experiments.business_constraints.run import rich_rows
from experiments.business_constraints.tests.test_business import fixture
from lab_transport.decision_value import nearest_batch
from lab_transport.domain import Grid
from lab_transport.dynamics_model import Geometry

CONFIG=Path(__file__).with_name("dataset_config.json")


def case_for(spec, config):
    effective={**config,**spec}
    base,cfg,_=fixture(effective.get('resource_capacity_A',1))
    layout=spec["layout"]
    if layout not in range(9):raise ValueError("unsupported frozen map layout")
    # Distinct map/group geometry with public station and robot positions.
    dx=layout
    obstacles=({1:{(4,2)},2:{(4,2),(4,3)},
                3:{(5,2),(6,2),(5,5)},4:{(6,2),(7,4),(8,3)},
                5:{(9,2),(11,5)},6:{(10,2),(12,5)},
                7:{(11,2),(13,5)},8:{(12,2),(14,5)}}.get(layout,set()))
    new_grid=Grid(7+layout,5+layout,frozenset(obstacles))
    stations={}
    public_per_station=effective.get("station_service_ticks",{})
    for sid,s in base.stations.items():
        public_ticks=public_per_station.get(sid,effective.get("station_service_base_ticks",s.service_ticks))
        if type(public_ticks) is not int or public_ticks<1:
            raise ValueError("invalid public station service duration")
        station=replace(s,
            position=(s.position[0]+dx,s.position[1]+(1 if layout else 0)),
            service_ticks=public_ticks,
            prior_ticks=public_ticks,
            jitter_ticks=effective["jitter_ticks"],
            outages=(tuple(effective["outage_window"]),) if spec["outage"] and sid==effective["outage_station"] else ())
        if not new_grid.contains(station.position):
            raise ValueError("configured station is blocked or outside map")
        stations[sid]=station
    robots=tuple(replace(r,
        position=(r.position[0]+dx,r.position[1]+(1 if layout else 0)),
        home=(r.home[0]+dx,r.home[1]+(1 if layout else 0)))
        for r in base.robots)
    for r in robots:
        if not new_grid.contains(r.position) or not new_grid.contains(r.home):
            raise ValueError("configured robot outside map")
    if "robot_count" in effective:
        count=effective["robot_count"]
        if type(count) is not int or not 1<=count<=len(robots):
            raise ValueError("invalid declared robot count")
        robots=robots[:count]
    if effective.get("workflow_variant")=="three":
        # Only public route topology changes, never the simulator's hidden state.
        cfg["workflows"]={**cfg["workflows"],
            "w0":cfg["workflows"]["w0"]+[{"pickup":"C","dropoff":"B1"}],
            "w1":cfg["workflows"]["w1"]+[{"pickup":"C","dropoff":"B0"}]}
    elif effective.get("workflow_variant","two")!="two":
        raise ValueError("unknown workflow variant")
    if "job_release_ticks" in effective:
        releases=effective["job_release_ticks"]
    else:
        patterns=((0,0,2,6),(0,2,4,10),(0,0,5,10),(0,1,7,12))
        releases=patterns[effective["release_pattern"]]
    if len(effective["job_workflows"])>len(releases):
        raise ValueError("missing public release timestamps for jobs")
    jobs=[{"id":f"F{spec['seed']}-{i}","workflow":flow,"release_tick":tick}
          for i,(flow,tick) in enumerate(zip(effective["job_workflows"],releases))]
    cfg={**cfg,"jobs":jobs,"max_ticks":effective["max_ticks"]}
    business=compile_scenario(replace(base,grid=new_grid,stations=stations,robots=robots),cfg)
    return business,Geometry(new_grid)


def rows_from_factual(business,geometry,seed):
    raw=[]
    result=BusinessSimulator(business,"world",seed,collect_metrics=True).run(
        on_transition=raw.append,
        dispatch_policy=lambda o,c:nearest_batch(o,c,geometry))
    return rich_rows(raw),result


def label_for(before, after, query):
    if after.tick!=before.tick+1:
        return None,"discontinuous"
    sid=query["sample_id"]
    old={s.id:s for s in before.samples}
    new={s.id:s for s in after.samples}
    if sid not in old or sid not in new:
        return None,"censored_missing_sample"
    phase=query["phase"]
    expected="in_transit" if phase=="to_pickup" else "completed"
    ongoing="assigned" if phase=="to_pickup" else "in_transit"
    if old[sid].status!=ongoing:
        return None,"invalid_predecessor_status"
    if new[sid].status==expected:
        return 1,"finished"
    if new[sid].status!=ongoing:
        return None,"broken_session"
    robot={r.id:r for r in after.robots}.get(query["robot_id"])
    if robot is None or robot.task_id!=sid or robot.phase!=phase:
        return None,"broken_session"
    return 0,"continues"


def build_dataset(config_path=CONFIG):
    config_path=Path(config_path)
    config_bytes=config_path.read_bytes()
    config=json.loads(config_bytes)
    if config.get("contract")!=QUERY_LABEL_VERSION:
        raise ValueError("dataset contract version mismatch")
    sets={};counts={};groups_used=set();source=[]
    for split in ("train","validation","test"):
        sets[split]=[];counter=Counter()
        for spec in config["groups"][split]:
            business,geometry=case_for(spec,config)
            fingerprint=hashlib.sha256(json.dumps({
                "seed":spec["seed"],"layout":spec["layout"],"jobs":business.jobs,
                "outage":spec["outage"]},sort_keys=True).encode()).hexdigest()[:20]
            group_id=f"synthetic:{fingerprint}"
            if group_id in groups_used:raise ValueError("train/validation/test group leak")
            groups_used.add(group_id)
            rows,result=rows_from_factual(business,geometry,spec["seed"])
            source.append({"group":group_id,"split":split,"seed":spec["seed"],
                           "layout":spec["layout"],"outage":spec["outage"],
                           "factual_steps":len(rows),"simulation_status":result["summary"]["status"],
                           "jobs":[j["id"] for j in business.jobs]})
            for idx,row in enumerate(rows):
                before,after=row.observation,row.next_observation
                queries=eligible_sessions(before,row.assignments)
                for query,reason in queries:
                    counter["considered"]+=1
                    if reason is not None:
                        counter["excluded_"+reason]+=1;continue
                    label,event=label_for(before,after,query)
                    if label is None:
                        counter["excluded_"+event]+=1;continue
                    try:
                        features=build_business_features(before,row.assignments,geometry,business.rules)
                        # Validate query-target alignment before retaining sample.
                        from business_transport.session_events import encode_query
                        encode_query(features,query)
                    except (KeyError,ValueError,TypeError) as exc:
                        counter["excluded_missing_or_invalid_observation"]+=1
                        continue
                    rowid=f"{group_id}:{before.tick}:{query['robot_id']}:{query['sample_id']}:{query['phase']}"
                    sets[split].append({"id":rowid,"trajectory_group":group_id,
                        "source":"BusinessSimulator_factual_transition",
                        "seed":spec["seed"],"map_layout":spec["layout"],
                        "tick":before.tick,"query":query,"features":features,
                        "label":label,"event":event})
                    counter["positive" if label else "negative"]+=1
            counter["groups"]+=1
        counts[split]=dict(counter)
    dataset={"schema_version":SCHEMA_VERSION,"label_version":QUERY_LABEL_VERSION,
             "source_type":"factual_observed_transition",
             "source_config_sha256":hashlib.sha256(config_bytes).hexdigest(),
             "source_groups":source,"counts":counts,**sets}
    from lab_transport.business_session_learning import verify_dataset
    dataset["verified_splits"]=verify_dataset(dataset)
    return dataset


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,default=CONFIG)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError("dataset is immutable and must not be overwritten")
    result=build_dataset(args.config)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,sort_keys=True,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({"counts":result["counts"],
                      "verified_splits":result["verified_splits"],
                      "source_config_sha256":result["source_config_sha256"]},ensure_ascii=False))


if __name__=="__main__":
    main()
