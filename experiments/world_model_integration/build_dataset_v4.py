"""V4: strictly factual, grouped synthetic service-event dataset with coverage proof.

Never uses private service countdowns, future faults, counterfactual outcome,
or not-yet-released tasks as input. Train/validation are generated in a
separate invocation from one-time in-range/out-of-range confirmation.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path

from business_transport.session_events import eligible_sessions, QUERY_LABEL_VERSION
from business_transport.session_features_v3 import encode_query_v3
from business_transport.state_features import build_business_features,SCHEMA_VERSION
from experiments.world_model_integration.build_dataset import case_for,rows_from_factual,label_for

CONFIG=Path(__file__).with_name("v4_frozen_config.json")
PRIOR_SETS=("train","validation","independent_in","independent_ood")
DEVELOPMENT=("train","validation")


def _field(row,name):
    f=row[name]
    if not isinstance(f,dict) or f.get("known") is not True:
        raise ValueError("required observable field missing: "+name)
    return f["value"]


def record_attributes(record):
    facts=record["features"]["observed_facts"]
    q=record["query"]
    r=next(r for r in facts["robots"] if r["id"]==q["robot_id"])
    s=next(s for s in facts["stations"] if s["id"]==q["station_id"])
    e=_field(r,"service_active_elapsed")
    d=_field(s,"expected_service_ticks_public")
    cap=_field(s,"resource_capacity")
    occupied=_field(s,"resource_occupied")
    if not isinstance(e,int) or not isinstance(d,(int,float)) or d<=0:
        raise ValueError("invalid observed effective progress or public prior")
    return {"prior":int(d),"nominal":int(record.get("nominal_public_prior",d)),
            "elapsed":e,"station":q["station_id"],
            "phase":q["phase"],"capacity":int(cap),"occupied":int(occupied),
            "competing":bool(_field(s,"capacity_blocked")),
            "robot_count":len(facts["robots"])}


def coverage_of(rows,counts,source):
    axes=("prior","nominal","elapsed","station","phase","capacity","occupied","competing","robot_count")
    hist={key:defaultdict(lambda:{"total":0,"positive":0,"negative":0}) for key in axes+("prior_elapsed","nominal_elapsed")}
    for row in rows:
        a=record_attributes(row)
        a["prior_elapsed"]=str(a["prior"])+"/"+str(a["elapsed"])
        a["nominal_elapsed"]=str(a["nominal"])+"/"+str(a["elapsed"])
        y=int(row["label"])
        for key in axes+("prior_elapsed","nominal_elapsed"):
            b=hist[key][str(a[key])]
            b["total"]+=1
            b["positive"]+=y
            b["negative"]+=1-y
    return {
      "valid_events":len(rows),"positive":sum(r["label"] for r in rows),
      "negative":len(rows)-sum(r["label"] for r in rows),
      "axes":{key:dict(sorted(values.items())) for key,values in hist.items()},
      "considered_and_censor_counts":dict(sorted(counts.items())),
      "source_groups":source,
      "observed_paused_session_ticks":counts.get("excluded_observed_device_pause",0),
      "observed_resume_events":counts.get("resumed_service",0),
      "observed_capacity_blocked_station_ticks":counts.get("observed_capacity_blocked_station_ticks",0),
      "configured_outage_groups":sum(int(row["outage"]) for row in source),
      "groups_with_observed_pause":sum(int(row["observed_paused_session_ticks"]>0) for row in source)
    }


def build_one(spec,config,split):
    business,geo=case_for(spec,config)
    raw,result=rows_from_factual(business,geo,spec["seed"])
    source_digest=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:20]
    group=f"synthetic-v4:{split}:{spec['seed']}:{source_digest}"
    kept=[];counter=Counter()
    last_reason={}
    for row in raw:
        before,after=row.observation,row.next_observation
        if after.tick != before.tick+1:
            raise ValueError("not a factual adjacent transition")
        for station in before.stations:
            if getattr(station,"capacity_blocked",False):
                counter["observed_capacity_blocked_station_ticks"]+=1
        for q,reason in eligible_sessions(before,row.assignments):
            counter["considered"]+=1
            key=(q["robot_id"],q["sample_id"],q["phase"],q["station_id"])
            if reason is not None:
                counter["excluded_"+reason]+=1
                last_reason[key]=reason
                if reason=="observed_device_pause":
                    previous=next((r for r in before.robots if r.id==q["robot_id"]),None)
                    later=next((r for r in after.robots if r.id==q["robot_id"]),None)
                    if (later is not None and previous is not None and
                        later.task_id==previous.task_id and later.phase==previous.phase and
                        later.service_active_elapsed is not None and
                        previous.service_active_elapsed is not None and
                        later.service_active_elapsed!=previous.service_active_elapsed):
                        raise ValueError("paused service has advanced effective progress")
                continue
            if last_reason.pop(key,None)=="observed_device_pause":
                counter["resumed_service"]+=1
            y,event=label_for(before,after,q)
            if y is None:
                counter["excluded_"+event]+=1
                continue
            try:
                features=build_business_features(before,row.assignments,geo,business.rules)
                encode_query_v3(features,q)
            except (ValueError,TypeError,KeyError) as e:
                counter["excluded_missing_or_invalid_input"]+=1
                continue
            record={
              "id":group+":"+str(before.tick)+":"+q["robot_id"]+":"+q["sample_id"],
              "trajectory_group":group,
              "source":"BusinessSimulator_factual_transition",
              "seed":spec["seed"],"map_layout":spec["layout"],
              "tick":before.tick,"query":q,"features":features,
              # Config-public nominal is provenance metadata, not an extra
              # model input; v3 56-d query reads the currently observable
              # (possibly adaptive) station expectation instead.
              "nominal_public_prior":int(spec["station_service_ticks"][q["station_id"]]),
              "label":y,"event":event}
            kept.append(record)
            counter["completed" if y else "continued"]+=1
        # First transition into an observed (ongoing) service state.
        for new in after.robots:
            old=next((r for r in before.robots if r.id==new.id),None)
            if old and old.service_elapsed is None and new.service_elapsed is not None:
                counter["service_started"]+=1
    if raw:
        final=raw[-1].next_observation
        counter["censored_end_active_service"]+=sum(r.service_elapsed is not None
                                                    for r in final.robots)
    summary={"trajectory_group":group,"seed":spec["seed"],"layout":spec["layout"],
      "prior_by_station":spec["station_service_ticks"],
      "jitter_ticks":spec["jitter_ticks"],"outage":bool(spec["outage"]),
      "outage_station":spec["outage_station"],"resource_capacity_A":spec["resource_capacity_A"],
      "robot_count":spec["robot_count"],"workflow_variant":spec["workflow_variant"],
      "task_count":len(spec["job_workflows"]),
      "tasks":[j["id"] for j in business.jobs],
      "factual_steps":len(raw),"simulation_status":result["summary"]["status"],
      "valid_events":len(kept),"positive":sum(r["label"] for r in kept),
      "observed_paused_session_ticks":counter["excluded_observed_device_pause"],
      "observed_resume_events":counter["resumed_service"],
      "capacity_blocked_station_ticks":counter["observed_capacity_blocked_station_ticks"],
      "censor_reasons":{k:v for k,v in counter.items() if k.startswith("excluded_")}}
    return kept,counter,summary


def build_part(part,config_path=CONFIG):
    config_bytes=Path(config_path).read_bytes()
    c=json.loads(config_bytes)
    if c.get("protocol")!="pr18-v4-frozen-synthetic-mechanism-20261110":
        raise ValueError("not frozen V4 config")
    if (len(c["groups"]["train"]),len(c["groups"]["validation"]),
        len(c["groups"]["independent_in"]),len(c["groups"]["independent_ood"]))!=(32,8,8,8):
        raise ValueError("trajectory group counts changed")
    groups=DEVELOPMENT if part=="development" else (part,)
    if any(x not in PRIOR_SETS for x in groups):
        raise ValueError("unsupported V4 part")
    all_seeds=[s["seed"] for split in PRIOR_SETS for s in c["groups"][split]]
    if len(set(all_seeds))!=len(all_seeds) or set(all_seeds)&set(c["old_seeds_forbidden"]):
        raise ValueError("group seeds reused or overlap historical confirmation")
    dataset={"schema_version":SCHEMA_VERSION,"label_version":QUERY_LABEL_VERSION,
      "source_type":"factual_observed_transition","protocol":c["protocol"],
      "source_config_sha256":hashlib.sha256(config_bytes).hexdigest(),
      "part":part,"groups":{},"coverage":{}}
    seen=set()
    for split in groups:
        records=[];count=Counter();sources=[]
        for spec in c["groups"][split]:
            data,sub,source=build_one(spec,c,split)
            if source["trajectory_group"] in seen:raise ValueError("trajectory group overlap")
            seen.add(source["trajectory_group"])
            records.extend(data);count.update(sub);sources.append(source)
        if not records or not 0<sum(r["label"] for r in records)<len(records):
            raise ValueError("no valid positive/negative events in split "+split)
        dataset[split]=records
        dataset["groups"][split]=[r["trajectory_group"] for r in sources]
        dataset["coverage"][split]=coverage_of(records,count,sources)
    if part=="development":
        train=dataset["coverage"]["train"]
        for prior in range(3,9):
            group=train["axes"]["nominal"].get(str(prior),{})
            if (group.get("positive",0)<c["data_gate"]["each_prior_train_min_complete"] or
                group.get("negative",0)<c["data_gate"]["each_prior_train_min_continue"]):
                raise ValueError("undercovered train prior "+str(prior))
        if (c["data_gate"]["at_least_one_observed_pause_in_train"] and
            train["observed_paused_session_ticks"]<1):
            raise ValueError("configured outage never produced actual pause in train")
    return dataset


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--part",choices=("development","independent_in","independent_ood"),required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--coverage",type=Path,required=True)
    a=p.parse_args()
    for dest in (a.output,a.coverage):
        if dest.exists():raise FileExistsError("V4 evidence is write-once")
    d=build_part(a.part)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(d,sort_keys=True,separators=(",",":"))+"\n")
    a.coverage.write_text(json.dumps({"config_sha256":d["source_config_sha256"],
       "protocol":d["protocol"],"part":d["part"],"coverage":d["coverage"]},
       ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"part":a.part,"config_sha256":d["source_config_sha256"],
       "coverage":{k:{"rows":v["valid_events"],"positive":v["positive"],
                      "prior":v["axes"]["prior"],
                      "paused":v["observed_paused_session_ticks"],
                      "resumed":v["observed_resume_events"],
                      "blocked":v["observed_capacity_blocked_station_ticks"],
                      "groups":len(v["source_groups"])}
                   for k,v in d["coverage"].items()}},ensure_ascii=False))


if __name__=="__main__":
    main()
