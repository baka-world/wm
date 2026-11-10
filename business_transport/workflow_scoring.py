"""Score known jobs through their final configured stage, without future jobs.

Costs are expressed in nominal simulation steps: one unit per unfinished job
per step, one per remaining map edge and one per expected service step. Resource
contention is a separate lexicographic tie-breaker, not another waiting cost.
"""
from __future__ import annotations

from experiments.service_value.scoring import _prior
from business_transport.workflow_transition import _route


def _key(sample):
    return ('job',sample.job_id) if getattr(sample,'job_id','') else ('sample',sample.id)


def _groups(state, known):
    result={key:[] for key in known}
    for sample in state.samples:
        key=_key(sample)
        if key in result:result[key].append(sample)
    if any(not value for value in result.values()):
        raise ValueError('a known business job disappeared')
    return result


def workflow_components(initial,trace,geometry,rules):
    if not trace:raise ValueError('score requires a nonempty prediction')
    initial_groups={}
    for sample in initial.samples:initial_groups.setdefault(_key(sample),[]).append(sample)
    known=set(initial_groups)
    routes={}
    for key,samples in initial_groups.items():
        sample=samples[0]
        routes[key]=_route(sample,rules) or ((sample.pickup,sample.dropoff),)
    def done(samples,key):
        last=len(routes[key])-1
        return any(getattr(s,'stage_index',0)==last and s.status=='completed' for s in samples)
    elapsed=sum(sum(not done(samples,key) for key,samples in _groups(state,known).items()) for state in trace)
    end=trace[-1];grouped=_groups(end,known)
    stations={s.id:s for s in end.stations};robots={r.id:r for r in end.robots}
    remaining_distance=0;service=0.0;future_stages=0
    demand={resource.id:[] for resource in rules.resources}
    for key,samples in grouped.items():
        if done(samples,key):continue
        current=max(samples,key=lambda s:getattr(s,'stage_index',0))
        route=routes[key];idx=getattr(current,'stage_index',0)
        active=current.status!='completed'
        first=idx if active else idx+1
        if first>=len(route):raise ValueError('unfinished job has no remaining stage')
        for index in range(first,len(route)):
            pickup,dropoff=route[index]
            if index>first:future_stages+=1
            if index==first and active:
                if current.status=='in_transit':
                    robot=robots.get(current.robot_id)
                    if robot is None:raise ValueError('in-transit job has no robot')
                    remaining_distance+=geometry.distance(robot.position,stations[dropoff].position)
                    target=dropoff
                    due_services=[dropoff]
                elif current.status in ('waiting','assigned'):
                    robot=robots.get(current.robot_id)
                    if current.status=='assigned' and robot is None:raise ValueError('assigned job has no robot')
                    start=(geometry.distance(robot.position,stations[pickup].position) if robot else
                           min(geometry.distance(r.position,stations[pickup].position) for r in robots.values()))
                    remaining_distance+=start+geometry.distance(stations[pickup].position,stations[dropoff].position)
                    target=pickup;due_services=[pickup,dropoff]
                else:raise ValueError('unsupported known job state')
                demand[rules.station_resources[target].id].append(current.id)
                for station_id in due_services:
                    amount=_prior(stations[station_id])
                    if (station_id==target and robot is not None and robot.position==stations[target].position
                            and robot.service_elapsed is not None):
                        progress=getattr(robot,'service_active_elapsed',None)
                        amount=max(0.0,amount-(progress if progress is not None else 0))
                    service+=amount
            else:
                # Future stage has known endpoints, not a known future robot.
                # Estimate one approach for the first not-yet-released stage;
                # subsequent stages use the contiguous workflow endpoints.
                if index==first:
                    future_stages+=1
                    remaining_distance+=min(geometry.distance(r.position,stations[pickup].position)
                                            for r in robots.values())
                remaining_distance+=geometry.distance(stations[pickup].position,stations[dropoff].position)
                service+=_prior(stations[pickup])+_prior(stations[dropoff])
    # A resource-demand proxy, not a real FIFO queue. Jobs already counted in
    # demand are not counted again as occupants; external current services are.
    contention=0
    known_samples={s.id for samples in grouped.values() for s in samples}
    for resource in rules.resources:
        wanted=demand[resource.id]
        occupied_elsewhere=sum(r.service_elapsed is not None and r.task_id not in known_samples and
            any(r.position==stations[sid].position for sid in resource.stations) for r in end.robots)
        contention+=max(0,len(wanted)+occupied_elsewhere-resource.capacity)
    return {'known_jobs':len(known),'unfinished_job_ticks':elapsed,
            'remaining_distance':remaining_distance,'estimated_remaining_service':service,
            'unreleased_future_stages':future_stages,'terminal_work':remaining_distance+service,
            'resource_contention_proxy':contention}


def workflow_score(initial,trace,geometry,rules):
    return score_components(workflow_components(initial,trace,geometry,rules))


def score_components(c):
    return (round(c['unfinished_job_ticks']+c['terminal_work'],8),
            c['resource_contention_proxy'],c['remaining_distance'])
