"""Frozen 20-step observable workload score; no hidden or future inputs."""
from __future__ import annotations

from lab_transport.decision_value import terminal_distance


def _prior(station):
    observed = getattr(station, 'observed_effective_service_mean', None)
    return max(1.0, observed if observed is not None else station.expected_service_ticks)


def workload_components(initial, trace, geometry):
    if not trace:
        raise ValueError('score requires a nonempty prediction')
    known = {s.id for s in initial.samples}
    elapsed = 0
    for state in trace:
        samples = {s.id:s for s in state.samples}
        if not known.issubset(samples):
            raise ValueError('known sample disappeared')
        elapsed += sum(samples[sid].status != 'completed' for sid in known)
    end = trace[-1]
    stations = {s.id:s for s in end.stations}
    robots = {r.id:r for r in end.robots}
    remaining_distance = terminal_distance(end, known, geometry)
    service = 0.0
    demand = {sid:set() for sid in stations}
    for sample in end.samples:
        if sample.id not in known or sample.status == 'completed':
            continue
        if sample.status not in ('waiting', 'assigned', 'in_transit'):
            raise ValueError('unsupported task status')
        target_id = sample.dropoff if sample.status == 'in_transit' else sample.pickup
        demand[target_id].add(sample.id)
        remaining = _prior(stations[target_id])
        robot = robots.get(sample.robot_id)
        phase = 'to_dropoff' if sample.status == 'in_transit' else 'to_pickup'
        if (robot is not None and robot.phase == phase and robot.position == stations[target_id].position
                and robot.service_elapsed is not None):
            progress = getattr(robot, 'service_active_elapsed', None)
            remaining = max(0.0, remaining - (progress if progress is not None else 0))
        service += remaining
        if sample.status in ('waiting','assigned'):
            service += _prior(stations[sample.dropoff])
    contention = 0
    for sid, station in stations.items():
        wanted = demand[sid]
        if not wanted:
            continue
        others = sum(r.position == station.position and r.task_id not in wanted for r in end.robots)
        contention += max(0, len(wanted) + others - 1) + int(not station.available)
    terminal_work = remaining_distance + service
    return {'known_unfinished_ticks':elapsed, 'remaining_distance':remaining_distance,
            'estimated_remaining_service':service, 'terminal_work':terminal_work,
            'contention_proxy':contention}


def workload_score(initial, trace, geometry):
    components = workload_components(initial, trace, geometry)
    return (round(components['known_unfinished_ticks'] + components['terminal_work'], 8),
            components['contention_proxy'], components['remaining_distance'])
