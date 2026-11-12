"""PR45 research-only explicit exogenous recovery extension of frozen V4-H20.

The original RHWM deterministic/service step is validated before applying any
exogenous event. A second strictly bounded contract permits only recovery of
stations already publicly observed DOWN. No new outages are predicted, no
future schedules reach the forecaster, and all business relationship fields are
recomputed after recovery. This does NOT modify RHWM default behavior.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import argparse
import hashlib
import json
import time

import torch

from business_transport.observation import BusinessObserver, capacity_view
from business_transport.policy import BusinessPolicy
from business_transport.scenario import compile_scenario
from business_transport.simulation import BusinessSimulator, resource_audit
from business_transport.world_model import HybridSystemWorldModel
from business_transport.world_model.contracts import ForecastBatch, ForecastParticle, ForecastStep
from business_transport.world_model.hybrid import _distribution, _seed
from experiments.business_constraints.tests.test_business import fixture
from experiments.world_model_integration.health_recovery.pr44_observed_recovery import (
    PROTOCOL as PR44_PROTOCOL, private_duration, recovery_probability
)
from experiments.world_model_integration.system_world_model.evaluate_horizons import (
    file_sha, score
)
from lab_transport.decision_value import candidates_for, nearest_batch
from lab_transport.dynamics_model import Geometry
from lab_transport.multisite_experiment import discrete_safety_violations
from lab_transport.state_v2 import ObservationV2, rebuild

PROTOCOL = Path(__file__).with_name("pr45_protocol.json")
KINDS = ("original_v4_h20_no_future_faults", "v4_h20_observed_down_km_recovery")
RELATIONS = (
    "occupant_robot_id", "service_task_id", "service_session_active",
    "service_in_progress", "waiting_pickups", "approaching_pickups",
    "approaching_dropoffs",
)
UNKNOWN_AFTER_EXOGENOUS = (
    "future_unreleased_jobs", "future_new_device_outage_onsets",
    "future_conditional_retests",
)


class PublicDownClock:
    """Consecutive PUBLIC device_available observations. No schedule lookups."""
    def __init__(self):
        self.last_tick = None
        self.last_device = {}
        self.ages = {}

    def observe(self, state):
        if self.last_tick is not None and state.tick <= self.last_tick:
            raise ValueError("public health history must be strictly chronological")
        consecutive = self.last_tick is not None and state.tick == self.last_tick + 1
        ages = {}
        devices = {s.id: bool(s.device_available) for s in state.stations}
        for sid, up in devices.items():
            if up:
                ages[sid] = None
            elif consecutive and sid in self.last_device and self.last_device[sid]:
                ages[sid] = 0
            elif consecutive and sid in self.last_device and not self.last_device[sid]:
                prior = self.ages.get(sid)
                ages[sid] = None if prior is None else prior + 1
            else:
                # Joined an outage in progress without its observed onset.
                ages[sid] = None
        self.last_tick, self.last_device, self.ages = state.tick, devices, ages
        return dict(ages)


def _sample_recovery(seed, particle, tick, station_id, probability):
    if not 0. <= probability <= 1.:
        raise ValueError("invalid recovery probability")
    raw = f"pr45-recovery|{seed}|{particle}|{tick}|{station_id}".encode()
    u = int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") / 2**64
    return u < probability


def inject_recovery(baseline, station_ids, rules):
    """Apply only a NEXT-OBSERVATION-boundary external repair.

    Service movement/completion happened during [t,t+1) while the device was
    still down. The recovered capacity is available for the FOLLOWING tick.
    """
    ids = frozenset(station_ids)
    if not ids:
        return baseline
    present = {s.id:s for s in baseline.stations}
    if not ids <= set(present):
        raise ValueError("unknown station in external recovery event")
    if any(present[sid].device_available for sid in ids):
        raise ValueError("recovery event cannot target an already healthy device")
    devices = {s.id:(s.id in ids or bool(s.device_available))
               for s in baseline.stations}
    changed = replace(baseline, stations=tuple(
        replace(s, device_available=devices[s.id]) for s in baseline.stations
    ))
    capacity, _ = capacity_view(changed, (), rules, device_status=devices)
    relations = rebuild(capacity,capacity.robots)
    after = replace(capacity, stations=tuple(
        replace(station, **{
            name:getattr(next(x for x in relations.stations if x.id==station.id),name)
            for name in RELATIONS
        }) for station in capacity.stations))
    check_recovery_boundary(baseline, after, ids, rules)
    return after


def check_recovery_boundary(baseline, after, recovered, rules):
    """Separate exogenous event contract, NEVER weakening ordinary checker."""
    recovered=frozenset(recovered)
    if not isinstance(baseline, ObservationV2) or not isinstance(after, ObservationV2):
        raise ValueError("external transition requires public observation")
    if not recovered:
        if after != baseline:
            raise ValueError("state changed with no recovery event")
        return True
    if baseline.tick != after.tick or baseline.robots != after.robots or baseline.samples != after.samples:
        raise ValueError("external boundary changed time, tasks or robot state")
    old={s.id:s for s in baseline.stations}
    new={s.id:s for s in after.stations}
    if len(old)!=len(baseline.stations) or len(new)!=len(after.stations) or set(old)!=set(new):
        raise ValueError("station identities changed")
    if not recovered <= set(old):
        raise ValueError("unrecognized recovery event station")
    for sid, previous in old.items():
        nxt=new[sid]
        if (previous.position != nxt.position or
            previous.resource_id != nxt.resource_id or
            previous.resource_capacity != nxt.resource_capacity or
            previous.resource_occupied != nxt.resource_occupied or
            previous.expected_service_ticks != nxt.expected_service_ticks):
            raise ValueError("external recovery changed immutable capacity/map state")
        if sid in recovered:
            if previous.device_available or not nxt.device_available:
                raise ValueError("only an observed unavailable station may recover")
        elif previous.device_available != nxt.device_available:
            raise ValueError("unlisted future device event injected")
    devices={sid:bool(s.device_available) for sid,s in new.items()}
    expected,_=capacity_view(after,(),rules,device_status=devices)
    expected_map={s.id:s for s in expected.stations}
    relation=rebuild(after,after.robots)
    relation_map={s.id:s for s in relation.stations}
    for sid,station in new.items():
        for field in ("available","resource_occupied","capacity_blocked"):
            if getattr(station,field)!=getattr(expected_map[sid],field):
                raise ValueError("recovery violated capacity/admission relation: "+field)
        for field in RELATIONS:
            if getattr(station,field)!=getattr(relation_map[sid],field):
                raise ValueError("recovery violated observable station relation: "+field)
    return True


class ObservedRecoveryWorld:
    """Compose original V4 service dynamics and public-history-only KM repair."""

    def __init__(self, geometry, rules, checkpoint, station_models):
        self.base=HybridSystemWorldModel(
            geometry,rules,kind="v4",checkpoint=checkpoint,experimental=True)
        self.rules=rules
        self.station_models=station_models
        self.model_version="rhwm-pr45-v4-known-down-recovery-research"

    def rollout(self, initial, first_action, *, down_ages, continuation_policy,
                horizon_ticks, particles, sampling_seed):
        if not isinstance(initial,ObservationV2):
            raise ValueError("no public observation")
        if not 1 <= horizon_ticks <= 20 or not 1 <= particles <= 8:
            raise ValueError("invalid RHWM bounds")
        first_action=tuple(first_action)
        traces=[]
        recovery_traces=[]
        for particle in range(particles):
            state=initial
            age={sid:down_ages.get(sid) for sid in (s.id for s in initial.stations)}
            trace=[]
            recovery_events=[]
            for step_index in range(horizon_ticks):
                action=(first_action if step_index==0 else tuple(continuation_policy(state)))
                # Unmodified V4 prediction validates the original transition.
                forecast=self.base.predict_next(
                    state, action, sampling_seed=_seed(sampling_seed,particle,state.tick))
                recovered=[]
                for station in sorted(state.stations,key=lambda x:x.id):
                    sid=station.id
                    if station.device_available:
                        age[sid]=None
                        continue
                    known_age=age.get(sid)
                    if known_age is None or sid not in self.station_models:
                        continue
                    p=recovery_probability(self.station_models[sid],known_age,1)
                    if p is None:
                        age[sid]=known_age+1
                        continue
                    if _sample_recovery(sampling_seed,particle,state.tick,sid,p):
                        recovered.append(sid)
                        age[sid]=None
                    else:
                        age[sid]=known_age+1
                    recovery_events.append({
                        "particle":particle,"tick":forecast.next_state.tick,
                        "station_id":sid, "public_down_age":known_age,
                        "predicted_recovery_probability":p,
                        "sampled_recovery":sid in recovered,
                    })
                next_state=inject_recovery(forecast.next_state,recovered,self.rules)
                trace.append(ForecastStep(
                    state,action,next_state,forecast.service_events,
                    self.model_version,UNKNOWN_AFTER_EXOGENOUS))
                state=next_state
            traces.append(ForecastParticle(particle,tuple(trace)))
            recovery_traces.extend(recovery_events)
        traces=tuple(traces)
        result=ForecastBatch(
            self.model_version,initial,first_action,"nearest_on_predicted_public",
            traces,tuple(_distribution(traces,i) for i in range(horizon_ticks)),
            UNKNOWN_AFTER_EXOGENOUS)
        return result,recovery_traces


def load_archived_km(path):
    """Read only PR44 training estimates; reject arbitrary fabricated models."""
    d=json.loads(Path(path).read_text())
    if (d.get("schema")!="pr44-observed-downstation-recovery-cpu-v1"
        or d.get("train_episodes")!=36
        or d.get("heldout_episodes")!=24
        or not d.get("no_fault_oracle_model_features")
        or d.get("protocol_sha256")!=file_sha(PR44_PROTOCOL)
        or d.get("source_commit")!="e6988a322d29f6224ead69ff26590236a9927919"):
        raise ValueError("wrong PR44 archived train-only recovery evidence")
    models=d["train_repair_models"]["station"]
    if set(models)!={"B0","B1","C"} or any(m["records"]!=12 for m in models.values()):
        raise ValueError("unexpected KM support/scope")
    return models


def factual_case(spec, cfg):
    """Truth-only construction: hidden duration never enters the predictor."""
    base, raw_cfg, _ = fixture(capacity=1)
    stations=dict(base.stations)
    station=stations[spec["station_id"]]
    # PRIVATE scenario creation only. No station.outages passed into rollout.
    duration=private_duration(spec,cfg["source_pr44_protocol"])
    start=cfg["source_pr44_protocol"]["synthetic_outage_design"]["start_tick"]
    stations[spec["station_id"]]=replace(
        station,outages=((start,start+duration),))
    max_ticks=cfg["max_ticks"]
    scenario=replace(base,name="pr45-"+spec["id"],stations=stations,max_ticks=max_ticks)
    jobs=[{"id":"J"+str(i+1),"workflow":"w0" if i%2==0 else "w1",
           "release_tick":0,"deadline_tick":max_ticks-15}
          for i in range(cfg["source_pr44_protocol"]["synthetic_outage_design"]["num_jobs"])]
    business=compile_scenario(
        scenario,{**raw_cfg,"name":"pr45-"+spec["id"],"jobs":jobs,"max_ticks":max_ticks})
    geometry=Geometry(scenario.grid)
    observer=BusinessObserver()
    policy=BusinessPolicy("nearest",geometry,business.rules,observer)
    health=PublicDownClock()
    observations={}
    ages={}
    transitions=[]

    def observe(raw):
        state=observer.observe(raw)
        observations[state.tick]=state
        ages[state.tick]=health.observe(state)

    factual=BusinessSimulator(
        business,"world",spec["seed"],collect_metrics=True).run(
            on_observation=observe,on_transition=transitions.append,
            dispatch_policy=policy)
    if transitions and transitions[-1].next_boundary=="end_of_run":
        final=transitions[-1].next_observation
        observations[final.tick]=observer.observe(final)
        ages[final.tick]=health.observe(final)
    if resource_audit(factual,business.rules)["violations"] or discrete_safety_violations(factual):
        raise AssertionError("factual simulator safety violated")
    actions={x.observation.tick:x.assignments for x in transitions}
    if len(actions)!=len(transitions):
        raise AssertionError("duplicate real tick")
    if not set(actions)<=set(observations):
        raise AssertionError("missing public observation")
    return business,geometry,observations,ages,actions,factual


def predict_metrics(batch,truth,station_id,h):
    metric=score(batch,truth,h)
    p=sum(
        float(next(s.device_available for s in x.steps[h-1].next_state.stations
                   if s.id==station_id))
        for x in batch.particles)/len(batch.particles)
    actual=next(s.device_available for s in truth.stations if s.id==station_id)
    metric["target_station_device_brier"]=(p-float(actual))**2
    metric["target_station_predicted_available"]=p
    metric["target_station_true_available"]=int(actual)
    return metric


def audit_case(spec,cfg,checkpoint,km):
    business,geometry,obs,ages,actions,factual=factual_case(spec,cfg)
    baseline=HybridSystemWorldModel(
        geometry,business.rules,kind="v4",checkpoint=checkpoint,experimental=True)
    recovery=ObservedRecoveryWorld(geometry,business.rules,checkpoint,km)
    rows=[]
    for tick in cfg["anchors"]:
        if tick not in actions:
            continue
        initial=obs[tick]
        station=next(s for s in initial.stations if s.id==spec["station_id"])
        down_age=ages[tick].get(spec["station_id"])
        if station.device_available or down_age is None:
            # This must never imply that a non-observed failure was forecast.
            continue
        eligible=[h for h in cfg["horizons"] if tick+h in obs]
        if not eligible:continue
        horizon=max(eligible)
        first=actions[tick]
        if any(a not in candidates_for(initial) for a in first):
            raise AssertionError("real first action is not publicly legal")
        continuation=lambda state:nearest_batch(state,candidates_for(state),geometry)
        seed=cfg["prediction_seed"]+spec["seed"]*101+tick
        start=time.process_time()
        base=baseline.rollout(initial,first,continuation_policy=continuation,
                              continuation_policy_name="nearest_on_predicted_public",
                              horizon_ticks=horizon,particles=cfg["particles"],
                              sampling_seed=seed)
        base_cpu=time.process_time()-start
        start=time.process_time()
        changed,repair_trace=recovery.rollout(
            initial,first,down_ages=ages[tick],continuation_policy=continuation,
            horizon_ticks=horizon,particles=cfg["particles"],sampling_seed=seed)
        repair_cpu=time.process_time()-start
        if any(not st.device_available for st in initial.stations):
            # Original model does not generate an external recovery.
            assert all(next(s.device_available for s in p.final_state.stations
                            if s.id==spec["station_id"]) is False for p in base.particles)
        for h in eligible:
            truth=obs[tick+h]
            rows.append({
                "anchor_tick":tick,"initial_public_down_age":down_age,
                "horizon":h,"truth_tick":truth.tick,
                "methods":{
                    KINDS[0]:predict_metrics(base,truth,spec["station_id"],h),
                    KINDS[1]:predict_metrics(changed,truth,spec["station_id"],h),
                },
                "repair_event_samples_until_h":sum(
                    q["sampled_recovery"] and q["tick"]<=tick+h for q in repair_trace),
                "eligible_public_repair_draws_until_h":sum(
                    q["tick"]<=tick+h for q in repair_trace),
                "source_checkpoint_equal":True,
                "cpu_seconds_full_rollout":{
                    KINDS[0]:base_cpu,KINDS[1]:repair_cpu,
                },
            })
    return {
        "id":spec["id"],"seed":spec["seed"],"station_id":spec["station_id"],
        "factual_status":factual["summary"]["status"],
        "factual_ticks":factual["summary"]["ticks"],
        "factual_events_sha256":hashlib.sha256(json.dumps(
            factual["events"],sort_keys=True).encode()).hexdigest(),
        "eligible_anchored_forecasts":len(rows),"rows":rows,
    }


def aggregate(groups,horizons):
    output={}
    metrics=("target_station_device_brier","sample_status_error","station_busy_brier",
             "resource_occupancy_mae","completed_count_mae","waiting_count_mae")
    for h in horizons:
        own=[]
        for group in groups:
            rows=[r for r in group["rows"] if r["horizon"]==h]
            if rows:
                own.append({
                    method:{k:sum(r["methods"][method][k] for r in rows)/len(rows)
                            for k in metrics} for method in KINDS})
        result={"scenario_groups":len(own),"metrics":{}}
        for model in KINDS:
            result["metrics"][model]={
                k:(sum(g[model][k] for g in own)/len(own) if own else None)
                for k in metrics}
        result["recovery_minus_baseline"]={
            k:(result["metrics"][KINDS[1]][k]-result["metrics"][KINDS[0]][k]
               if own else None) for k in metrics}
        output[str(h)]=result
    return output


def run(checkpoint,pr44_result,output):
    cfg=json.loads(PROTOCOL.read_text())
    if (cfg["protocol"]!="PR45_EXPLICIT_OBSERVED_RECOVERY_WORLD_TRANSITION_20261112_V1"
        or cfg["methods"]!=list(KINDS)
        or cfg["horizons"]!=[5,20] or cfg["anchors"]!=[8,10]
        or cfg["particles"]!=3 or len(cfg["cohort"])!=12
        or len({x["seed"] for x in cfg["cohort"]})!=12):
        raise ValueError("changed frozen PR45 protocol")
    if file_sha(checkpoint)!=cfg["frozen_original_v4_sha256"]:
        raise ValueError("V4 checkpoint SHA mismatch")
    if output.exists():raise FileExistsError("no overwrite of primary research evidence")
    km=load_archived_km(pr44_result)
    pr44_cfg=json.loads(PR44_PROTOCOL.read_text())
    cfg["source_pr44_protocol"]=pr44_cfg
    forbidden={x["seed"] for k in ("train","test") for x in pr44_cfg["cohort"][k]}
    if any(x["seed"] in forbidden for x in cfg["cohort"]):
        raise ValueError("PR45 test seed overlaps earlier recovery fit/evaluation")
    torch.set_num_threads(1)
    start=time.process_time()
    groups=[]
    for item in cfg["cohort"]:
        if time.process_time()-start>=cfg["resources"]["cpu_budget_seconds"]:
            raise TimeoutError("PR45 CPU process budget exceeded")
        groups.append(audit_case(item,cfg,checkpoint,km))
    elapsed=time.process_time()-start
    if elapsed>=cfg["resources"]["cpu_budget_seconds"]:
        raise TimeoutError("PR45 CPU budget exceeded on last group")
    if not all(g["eligible_anchored_forecasts"] for g in groups):
        raise AssertionError("missing an expected public-down prediction cohort")
    result={
        "schema":"pr45-exogenous-conditioned-recovery-world-state-v1",
        "experiment":"frozen_original_V4_vs_V4_plus_km_known_down_recovery",
        "evidence_scope":"new_seed_same_synthetic_mechanism_H20_state_forecast_not_online_policy",
        "protocol_sha256":file_sha(PROTOCOL),
        "pr44_training_evidence_sha256":file_sha(pr44_result),
        "pr44_protocol_sha256":file_sha(PR44_PROTOCOL),
        "v4_checkpoint_sha256":file_sha(checkpoint),
        "gpu":0,"new_training":0,"original_v4_unchanged":True,
        "no_future_fault_onset_prediction":True,
        "groups":groups,
        "summary":{"groups":len(groups),"by_horizon":aggregate(groups,cfg["horizons"]),
                   "cpu_process_seconds":elapsed,
                   "limits":"same synthetic mechanism, 3 particles, 12 new seeds, synthetic truth known only to scenario generation, not an online policy gain"},
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print("PR45_H20_STATE_EVIDENCE",json.dumps(result["summary"],ensure_ascii=False))
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True,type=Path)
    p.add_argument("--pr44-result",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args()
    run(args.checkpoint,args.pr44_result,args.out)


if __name__=="__main__":
    main()
