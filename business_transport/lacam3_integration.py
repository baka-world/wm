"""Hybrid PIBT/LaCAM3 execution with separately copied rollout state."""
from __future__ import annotations

from dataclasses import dataclass,replace
import time

from business_transport.lacam3_coordinator import RecoveryCoordinator,RecoverySnapshot
from business_transport.lacam3_motion import LaCAM3Planner
from business_transport.observation import capacity_view
from business_transport.pibt_integration import CoordinatedBusinessSimulator,MotionObserver
from business_transport.pibt_motion import PIBTMotion
from lab_transport.dynamics_model import Geometry,_intent,_target
from lab_transport.state_v2 import ObservationV2,rebuild
from lab_transport.transition_schema import Observation


@dataclass(frozen=True)
class RecoveryRawObservation(Observation):
    recovery_snapshot: RecoverySnapshot = RecoverySnapshot()


@dataclass(frozen=True)
class RecoveryObservation(ObservationV2):
    recovery_snapshot: RecoverySnapshot = RecoverySnapshot()


def _with_snapshot(observation,snapshot,cls):
    return cls(**{name:getattr(observation,name) for name in Observation.__dataclass_fields__},
               recovery_snapshot=snapshot)


class RecoveryObserver(MotionObserver):
    def observe(self,observation):
        rich=super().observe(observation)
        return _with_snapshot(rich,getattr(observation,'recovery_snapshot',RecoverySnapshot()),RecoveryObservation)


def _station_key(stations):
    return tuple((s.id,bool(s.available),bool(getattr(s,'device_available',s.available)),
                  getattr(s,'resource_occupied',None),getattr(s,'capacity_blocked',None)) for s in stations)


def _current_robots(observation,locked):
    samples={s.id:s for s in observation.samples}
    stations={s.id:s.position for s in observation.stations}
    rows=[]
    for r in observation.robots:
        sample=samples.get(r.task_id)
        rows.append({'id':r.id,'position':r.position,'task_id':r.task_id,'phase':r.phase,
            'goal':_target(r,(r.phase,r.task_id),samples,stations),
            'release_tick':sample.release_tick if sample is not None else None,'locked':r.id in locked})
    return rows


class HybridBusinessSimulator(CoordinatedBusinessSimulator):
    """PIBT normally, official full LaCAM3 paths only after detected livelock."""
    def __init__(self,business,*args,motion_backend='hybrid',pibt_library=None,lacam3_worker=None,
                 planner_seed=20261109,**kwargs):
        if motion_backend not in ('legacy','pibt','hybrid'):raise ValueError('unknown motion backend')
        if motion_backend=='hybrid' and not lacam3_worker:raise ValueError('LaCAM3 worker is required')
        self.hybrid_enabled=motion_backend=='hybrid';self.recovery_snapshot=RecoverySnapshot();self.recovery_log=[]
        super().__init__(business,*args,motion_backend='pibt' if self.hybrid_enabled else motion_backend,
                         pibt_library=pibt_library,planner_seed=planner_seed,**kwargs)
        if self.hybrid_enabled:
            geometry=Geometry(self.scenario.grid)
            self.recovery=RecoveryCoordinator(LaCAM3Planner(geometry,lacam3_worker,seed=planner_seed),self.pibt)

    def _observable_state(self,tick):
        observation=super()._observable_state(tick)
        return (_with_snapshot(observation,self.recovery_snapshot,RecoveryRawObservation)
                if self.hybrid_enabled else observation)

    def _move(self,tick,serviced):
        if not self.hybrid_enabled:return super()._move(tick,serviced)
        started=time.perf_counter()
        paused={rid for rid,intervals in self.motion_pauses.items() if any(a<=tick<b for a,b in intervals)}
        if self.collect_metrics:self.metrics['motion_pause_ticks']+=len(paused-serviced)
        locked=serviced|paused
        # _handle_station has just run. These are current observations, never
        # the sampled hidden service countdown or future fault schedule.
        raw=super()._observable_state(tick)
        capacity_wait={e.get('robot_id') for e in self.events if e['tick']==tick and e['type']=='resource_wait'}
        at_station={s.position:s for s in self.scenario.stations.values()}
        normal_service=any(r.id in serviced and r.id not in capacity_wait and r.position in at_station and
                           at_station[r.position].available(tick) for r in self.robots.values())
        counts={resource.id:0 for resource in self.business.rules.resources}
        for r in self.robots.values():
            if r.service_started is not None and r.position in at_station:
                counts[self.business.rules.station_resources[at_station[r.position].id].id]+=1
        for resource in self.business.rules.resources:
            if counts[resource.id]>resource.capacity:raise RuntimeError('workstation capacity already exceeded')
        robots=_current_robots(raw,locked)
        completed=sum(s.status=='completed' for s in raw.samples)
        self.recovery_snapshot,positions,event=self.recovery.step(self.recovery_snapshot,robots,
            _station_key(raw.stations),completed,normal_service,tick)
        for rid,pos in positions.items():self.robots[rid].position=pos
        event['positions_before']={r['id']:r['position'] for r in robots}
        event['goals']={r['id']:r['goal'] for r in robots};event['locked']=tuple(sorted(locked))
        event['normal_service']=normal_service;event['no_stage_steps']=self.recovery_snapshot.no_stage_steps
        self.recovery_log.append(event)
        self.motion_wall_seconds.append(time.perf_counter()-started)


class RecoveryMovementModel:
    """Clone observed recovery state per prediction branch, including full cache."""
    def __init__(self,model,geometry,rules,pibt_library,lacam3_worker,*,planner_seed=20261109):
        self.model=model;self.geometry=geometry;self.rules=rules
        self.pibt=PIBTMotion(geometry,pibt_library,planner_seed=planner_seed)
        self.coordinator=RecoveryCoordinator(LaCAM3Planner(geometry,lacam3_worker,seed=planner_seed),self.pibt)

    @property
    def distribution(self):return getattr(self.model,'distribution',None)
    @property
    def service_calls(self):return getattr(self.model,'service_calls',0)
    def reset_seed(self,seed):
        if hasattr(self.model,'reset_seed'):self.model.reset_seed(seed)

    def predict_next(self,observation,assignments):
        snapshot=getattr(observation,'recovery_snapshot',None)
        if snapshot is None:raise ValueError('LaCAM3 rollout needs an explicit copied recovery snapshot')
        assignments=tuple(assignments);intents=_intent(observation,assignments)
        samples={s.id:s for s in observation.samples};stations={s.id:s for s in observation.stations}
        positions={r.id:r.position for r in observation.robots}
        old_goals={r.id:_target(r,intents[r.id],samples,{s.id:s.position for s in observation.stations})
                   for r in observation.robots}
        locked={r.id for r in observation.robots if
                (intents[r.id][1] is not None and r.position==old_goals[r.id]) or
                getattr(r,'operational_available',None) is False}
        normal_service=False
        for r in observation.robots:
            phase,sid=intents[r.id]
            if sid is not None and r.position==old_goals[r.id]:
                sample=samples[sid];station=stations[sample.pickup if phase=='to_pickup' else sample.dropoff]
                normal_service|=bool(station.available)
        predicted=self.model.predict_next(observation,assignments)
        prior={r.id:r for r in observation.robots}
        corrected=[]
        for r in predicted.robots:
            old=prior[r.id];phase,sid=intents[r.id]
            # The legacy service predictor temporarily treats an unavailable
            # capacity-blocked station as a started session. Clear that ghost
            # before computing the recovery/cache context, just as the outer
            # CapacityDynamics adapter does before exposing its next state.
            if sid is not None and old.position==old_goals[r.id] and old.service_elapsed is None:
                sample=samples[sid];station=stations[sample.pickup if phase=='to_pickup' else sample.dropoff]
                if getattr(station,'capacity_blocked',False):
                    r=replace(r,service_elapsed=None,service_active_elapsed=None,service_paused_elapsed=None)
            corrected.append(replace(r,position=positions[r.id],operational_available=
                                     getattr(old,'operational_available',None)))
        robots=tuple(corrected)
        # Current device status is observed; no future station event is read.
        current=replace(predicted,robots=robots,tick=observation.tick)
        devices={s.id:getattr(s,'device_available',s.available) for s in observation.stations}
        current=capacity_view(current,(),self.rules,device_status=devices)[0]
        rows=_current_robots(current,locked)
        completed=sum(s.status=='completed' for s in current.samples)
        new_snapshot,moved,_=self.coordinator.step(snapshot,rows,_station_key(current.stations),
                                                  completed,normal_service,observation.tick)
        robots=tuple(replace(r,position=moved[r.id]) for r in robots)
        result=rebuild(predicted,robots)
        return _with_snapshot(result,new_snapshot,RecoveryObservation)
