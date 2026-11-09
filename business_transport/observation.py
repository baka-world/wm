"""Observable workstation capacity and workflow-stage adapters.

No future job release, fault schedule or private service countdown is read.
A resource is held from observed service start until observed completion,
including a device outage. The exposed `available` means service admission;
`device_available` keeps equipment status separate from capacity blocking.
"""
from __future__ import annotations

from dataclasses import dataclass,replace
from lab_transport.dynamics_model import _intent
from lab_transport.state_v2 import ObservableHistory,RobotV2,SampleV2,StationV2,ObservationV2
from lab_transport.transition_schema import ObservedSample,ObservedStation


@dataclass(frozen=True)
class BusinessSample(ObservedSample):
    workflow: str = ''
    stage_index: int = 0
    job_id: str = ''


@dataclass(frozen=True)
class BusinessSampleV2(SampleV2):
    workflow: str = ''
    stage_index: int = 0
    job_id: str = ''


@dataclass(frozen=True)
class BusinessStation(ObservedStation):
    resource_id: str = ''
    resource_capacity: int = 1
    resource_occupied: int = 0
    device_available: bool = True
    capacity_blocked: bool = False


@dataclass(frozen=True)
class BusinessStationV2(StationV2):
    resource_id: str = ''
    resource_capacity: int = 1
    resource_occupied: int = 0
    device_available: bool = True
    capacity_blocked: bool = False


class BusinessObserver:
    def __init__(self):
        self.observer=ObservableHistory()

    def observe(self, observation):
        rich=self.observer.observe(observation)
        raw_samples={s.id:s for s in observation.samples}
        raw_stations={s.id:s for s in observation.stations}
        samples=tuple(BusinessSampleV2(**{k:getattr(s,k) for k in SampleV2.__dataclass_fields__},
                    workflow=getattr(raw_samples[s.id],'workflow',''),
                    stage_index=getattr(raw_samples[s.id],'stage_index',0),
                    job_id=getattr(raw_samples[s.id],'job_id','')) for s in rich.samples)
        stations=tuple(BusinessStationV2(**{k:getattr(s,k) for k in StationV2.__dataclass_fields__},
                     resource_id=getattr(raw_stations[s.id],'resource_id',''),
                     resource_capacity=getattr(raw_stations[s.id],'resource_capacity',1),
                     resource_occupied=getattr(raw_stations[s.id],'resource_occupied',0),
                     device_available=getattr(raw_stations[s.id],'device_available',s.available),
                     capacity_blocked=getattr(raw_stations[s.id],'capacity_blocked',False)) for s in rich.stations)
        return replace(rich,samples=samples,stations=stations)


def capacity_view(observation, assignments, rules, device_status=None):
    """Select actual/predicted admission using visible positions and progress.

    Existing service sessions retain slots. Remaining slots are assigned in
    robot-ID order for this step, independently of unknown service countdowns.
    """
    by_station=rules.station_resources
    stations={s.id:s for s in observation.stations}
    samples={s.id:s for s in observation.samples}
    intent=_intent(observation,tuple(assignments))
    active={r.id:[] for r in rules.resources}
    waiting={r.id:[] for r in rules.resources}
    for robot in sorted(observation.robots,key=lambda r:r.id):
        phase,sid=intent[robot.id]
        if sid is None:continue
        sample=samples[sid]
        station_id=sample.pickup if phase=='to_pickup' else sample.dropoff
        if robot.position!=stations[station_id].position:continue
        group=by_station[station_id]
        item=(robot.id,station_id)
        if robot.service_elapsed is not None and robot.task_id==sid and robot.phase==phase:
            active[group.id].append(item)
        else:
            waiting[group.id].append(item)
    allowed=set();occupied={}
    for group in rules.resources:
        existing=active[group.id]
        if len(existing)>group.capacity:
            raise ValueError('observed workstation resource is already over capacity')
        occupied[group.id]=len(existing)
        allowed.update(rid for rid,_ in existing)
        allowed.update(rid for rid,_ in waiting[group.id][:group.capacity-len(existing)])
    port_robot={sid:rid for group in waiting.values() for rid,sid in group}
    active_ports={sid for group in active.values() for _,sid in group}
    result=[]
    for station in observation.stations:
        group=by_station[station.id]
        device=(device_status[station.id] if device_status is not None else
                getattr(station,'device_available',station.available))
        rid=port_robot.get(station.id)
        blocked=((rid is not None and rid not in allowed) or
                 (rid is None and station.id not in active_ports and occupied[group.id]>=group.capacity))
        values={k:getattr(station,k) for k in ObservedStation.__dataclass_fields__}
        values['available']=bool(device and not blocked)
        cls=BusinessStationV2 if isinstance(station,StationV2) else BusinessStation
        if cls is BusinessStationV2:
            values={k:getattr(station,k) for k in StationV2.__dataclass_fields__}
            values['available']=bool(device and not blocked)
        result.append(cls(**values,resource_id=group.id,resource_capacity=group.capacity,
                          resource_occupied=occupied[group.id],device_available=bool(device),capacity_blocked=blocked))
    return replace(observation,stations=tuple(result)),allowed


def attach_prediction(before, predicted):
    """Restore visible workflow metadata after legacy models rebuild v2 state."""
    known={s.id:s for s in before.samples}
    samples=[]
    for s in predicted.samples:
        old=known[s.id]
        samples.append(BusinessSampleV2(**{k:getattr(s,k) for k in SampleV2.__dataclass_fields__},
            workflow=getattr(old,'workflow',''),stage_index=getattr(old,'stage_index',0),job_id=getattr(old,'job_id','')))
    return replace(predicted,samples=tuple(samples))
