"""Public workflow expansion and strict business-only transition checks.

No simulator, future job table, outage plan, or hidden service duration is
accepted here. A stage appearing at tick t+1 results only from a completion
predicted during [t,t+1) for a job already visible at tick t.
"""
from __future__ import annotations

from dataclasses import replace

from business_transport.observation import BusinessSampleV2, attach_prediction, capacity_view
from business_transport.scenario import stage_id
from lab_transport.dynamics_model import _intent
from lab_transport.paper_reuse.contracts import check_transition
from lab_transport.state_v2 import SampleV2, rebuild

IMMUTABLE = ('pickup', 'dropoff', 'release_tick', 'deadline_tick', 'specimen_type',
             'packaging', 'temperature_condition', 'rule_profile', 'priority', 'workflow', 'stage_index', 'job_id')
INHERITED = ('deadline_tick', 'specimen_type', 'packaging', 'temperature_condition', 'rule_profile', 'priority')


def _route(sample, rules):
    workflow = getattr(sample, 'workflow', '')
    job = getattr(sample, 'job_id', '')
    if not workflow and not job:
        return None
    index = getattr(sample, 'stage_index', None)
    if (not workflow or not job or workflow not in rules.workflows or not isinstance(index, int)
            or isinstance(index, bool) or not 0 <= index < len(rules.workflows[workflow])
            or sample.id != stage_id(job, index)
            or (sample.pickup, sample.dropoff) != rules.workflows[workflow][index]):
        raise ValueError('sample identity or stage differs from its public workflow')
    return rules.workflows[workflow]


def expected_successors(before, after, rules):
    old = {s.id: s for s in before.samples}
    new = {s.id: s for s in after.samples}
    expected = {}
    for sid, sample in old.items():
        route = _route(sample, rules)
        if route is None or sid not in new:
            continue
        index = sample.stage_index
        if sample.status != 'completed' and new[sid].status == 'completed' and index + 1 < len(route):
            successor = stage_id(sample.job_id, index + 1)
            if successor in old or successor in expected:
                raise ValueError('successor already exists or would be released twice')
            expected[successor] = sample
    return expected


def release_successors(before, after, rules):
    """Expand only stages completed by the model, at the next boundary."""
    if after.tick != before.tick + 1:
        raise ValueError('successor release requires the next observation boundary')
    expected = expected_successors(before, after, rules)
    existing = {s.id for s in after.samples}
    if existing & set(expected):
        raise ValueError('model already emitted a workflow successor')
    samples = list(after.samples)
    for sid, previous in sorted(expected.items()):
        index = previous.stage_index + 1
        pickup, dropoff = rules.workflows[previous.workflow][index]
        values = {k: getattr(previous, k) for k in SampleV2.__dataclass_fields__}
        values.update(id=sid, pickup=pickup, dropoff=dropoff, status='waiting', robot_id=None,
                      release_tick=after.tick, unassigned_wait_ticks=0)
        samples.append(BusinessSampleV2(**values, workflow=previous.workflow,
                                       stage_index=index, job_id=previous.job_id))
    extended = replace(after, samples=tuple(sorted(samples, key=lambda s: s.id)))
    # rebuild() recomputes observable station demand but returns a plain V2
    # observation; restore its fields into extended to preserve immutable
    # RecoveryObservation snapshots and other business-specific subclasses.
    rich = attach_prediction(extended, rebuild(extended, extended.robots))
    restored = replace(extended, samples=rich.samples, stations=rich.stations)
    devices = {s.id: getattr(s, 'device_available', s.available) for s in after.stations}
    return capacity_view(restored, (), rules, device_status=devices)[0]


def _check_job_states(observation, rules):
    grouped = {}
    for sample in observation.samples:
        route = _route(sample, rules)
        if route is None:
            continue
        grouped.setdefault(sample.job_id, []).append(sample)
    for job, samples in grouped.items():
        if len({s.workflow for s in samples}) != 1:
            raise ValueError('one job has multiple workflow identities')
        by_index = {s.stage_index: s for s in samples}
        if len(by_index) != len(samples) or set(by_index) != set(range(max(by_index) + 1)):
            raise ValueError('workflow contains duplicate or missing stages')
        if any(by_index[i].status != 'completed' for i in range(max(by_index))):
            raise ValueError('successor appeared before its predecessor completed')
        if sum(s.status != 'completed' for s in samples) > 1:
            raise ValueError('one job has parallel active stages')


def check_business_transition(before, after, geometry, rules, assignments=()):
    """Validate dynamic stages without weakening the generic fixed-ID check."""
    old = {s.id: s for s in before.samples}
    new = {s.id: s for s in after.samples}
    if len(old) != len(before.samples) or len(new) != len(after.samples):
        raise ValueError('duplicate sample ID')
    if not set(old) <= set(new):
        raise ValueError('known sample disappeared')
    expected = expected_successors(before, after, rules)
    if set(new) - set(old) != set(expected):
        raise ValueError('unknown, missing, duplicate or prematurely released successor')
    # Reuse the unchanged generic checker over the old fixed entity set.
    check_transition(before, replace(after, samples=tuple(s for s in after.samples if s.id in old)), geometry)
    for sid, sample in old.items():
        for field in IMMUTABLE:
            if getattr(sample, field, None) != getattr(new[sid], field, None):
                raise ValueError(f'prediction changed immutable {field}')
    for sid, previous in expected.items():
        sample = new[sid]
        route = _route(sample, rules)
        if (route is None or sample.workflow != previous.workflow or sample.job_id != previous.job_id or
                sample.stage_index != previous.stage_index + 1 or sample.release_tick != after.tick or
                sample.status != 'waiting' or sample.robot_id is not None or
                getattr(sample, 'unassigned_wait_ticks', 0) != 0):
            raise ValueError('invalid successor identity, release or initial status')
        for field in INHERITED:
            if getattr(sample, field, None) != getattr(previous, field, None):
                raise ValueError(f'successor changed inherited {field}')
    _check_job_states(before, rules)
    _check_job_states(after, rules)
    original_stations = {s.id:s for s in before.stations}
    next_stations = {s.id:s for s in after.stations}
    if len(next_stations) != len(after.stations) or set(original_stations) != set(next_stations):
        raise ValueError('station IDs changed')
    for sid, station in original_stations.items():
        nxt = next_stations[sid]
        if station.position != nxt.position:
            raise ValueError('station position changed')
        resource = rules.station_resources[sid]
        if (getattr(nxt,'resource_id',resource.id) != resource.id or
                getattr(nxt,'resource_capacity',resource.capacity) != resource.capacity):
            raise ValueError('station resource identity or capacity changed')
        if getattr(station,'device_available',station.available) != getattr(nxt,'device_available',nxt.available):
            raise ValueError('unknown future device change was injected')
    devices = {sid:getattr(s,'device_available',s.available) for sid,s in next_stations.items()}
    recomputed,_ = capacity_view(after, (), rules, device_status=devices)
    for actual, checked in zip(after.stations, recomputed.stations):
        for field in ('available','resource_occupied','capacity_blocked'):
            if getattr(actual,field,None) != getattr(checked,field,None):
                raise ValueError(f'inconsistent workstation capacity field: {field}')
    relationships=rebuild(after,after.robots)
    for actual,checked in zip(after.stations,relationships.stations):
        for field in ('occupant_robot_id','service_task_id','service_session_active','service_in_progress',
                      'waiting_pickups','approaching_pickups','approaching_dropoffs'):
            if getattr(actual,field,None)!=getattr(checked,field,None):
                raise ValueError(f'inconsistent observable workstation relation: {field}')
    intents = _intent(before, tuple(assignments))
    robots = {r.id:r for r in after.robots}
    samples = {s.id:s for s in before.samples}
    for robot in before.robots:
        phase, sid = intents[robot.id]
        service_station = (original_stations[samples[sid].pickup if phase == 'to_pickup' else samples[sid].dropoff]
                           if sid is not None else None)
        locked = getattr(robot,'operational_available',None) is False or (
            service_station is not None and robot.position == service_station.position)
        if locked and robots[robot.id].position != robot.position:
            raise ValueError('service or paused robot moved')
    return True
