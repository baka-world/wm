"""Extend Simulator at observation/service hooks; no new route planner."""
from __future__ import annotations

from dataclasses import replace
from business_transport.observation import BusinessSample,capacity_view
from lab_transport.simulation import Simulator
from lab_transport.transition_schema import ObservedSample


class BusinessSimulator(Simulator):
    def __init__(self, business, policy='world',seed=0,**kwargs):
        self.business=business
        self._capacity_tick=None
        self._capacity_allowed=set()
        super().__init__(business.scenario if business is not None else kwargs.pop('scenario'),policy,seed,**kwargs)
        if business is not None and self.collect_metrics:
            self.metrics['resource_wait_ticks']=0

    def _observable_state(self,tick):
        obs=super()._observable_state(tick)
        if self.business is None:return obs
        samples=[]
        for s in obs.samples:
            name,index,jid=self.business.stage_metadata[s.id]
            samples.append(BusinessSample(**{k:getattr(s,k) for k in ObservedSample.__dataclass_fields__},
                                          workflow=name,stage_index=index,job_id=jid))
        return capacity_view(replace(obs,samples=tuple(samples)),(),self.business.rules)[0]

    def _handle_station(self,tick,robot):
        if self.business is None:return super()._handle_station(tick,robot)
        if self._capacity_tick!=tick:
            obs=super()._observable_state(tick)
            self._capacity_allowed=capacity_view(obs,(),self.business.rules)[1]
            self._capacity_tick=tick
        if (robot.task_id is not None and robot.position==self._target(robot) and
                robot.service_started is None and robot.id not in self._capacity_allowed):
            sample=self.samples[robot.task_id]
            sid=sample.spec.pickup if robot.phase=='to_pickup' else sample.spec.dropoff
            if self.collect_metrics:self.metrics['resource_wait_ticks']+=1
            self._event(tick,'resource_wait',robot_id=robot.id,sample_id=sample.spec.id,
                        station_id=sid,resource_id=self.business.rules.station_resources[sid].id)
            return True
        stage=robot.task_id
        before=self.samples[stage].status if stage else None
        result=super()._handle_station(tick,robot)
        if stage is not None and before!='completed' and self.samples[stage].status=='completed':
            successor=self.business.stage_next.get(stage)
            if successor is not None:
                nxt=self.samples[successor]
                if nxt.status!='unreleased':raise ValueError('workflow successor released twice')
                nxt.spec=replace(nxt.spec,release_tick=tick+1)
                self._event(tick+1,'stage_ready',sample_id=successor,predecessor=stage)
        return result


def resource_audit(result,rules):
    """Replay service event intervals; end is processed before start at a tick."""
    group_by_station=rules.station_resources
    ordered=sorted((e for e in result['events'] if e['type'] in ('service_started','service_finished')),
                   key=lambda e:(e['tick'],0 if e['type']=='service_finished' else 1,e['robot_id']))
    active={r.id:set() for r in rules.resources};peaks={r.id:0 for r in rules.resources};violations=[]
    busy_ticks={r.id:0 for r in rules.resources};last_tick=0
    for event in ordered:
        elapsed=event['tick']-last_tick
        for rid,sessions in active.items():busy_ticks[rid]+=len(sessions)*elapsed
        last_tick=event['tick']
        group=group_by_station[event['station_id']]
        key=(event['robot_id'],event['sample_id'],event['phase'])
        if event['type']=='service_started':
            if key in active[group.id]:raise ValueError('duplicate service start')
            active[group.id].add(key)
        else:
            if key not in active[group.id]:raise ValueError('service finished without start')
            active[group.id].remove(key)
        peaks[group.id]=max(peaks[group.id],len(active[group.id]))
        if len(active[group.id])>group.capacity:
            violations.append({'tick':event['tick'],'resource_id':group.id,'active':len(active[group.id]),
                               'capacity':group.capacity})
    elapsed=result['summary']['ticks']-last_tick
    for rid,sessions in active.items():busy_ticks[rid]+=len(sessions)*elapsed
    return {'peaks':peaks,'busy_slot_ticks':busy_ticks,'violations':violations,
            'open_sessions_at_end':sum(len(v) for v in active.values()),
            'capacity_wait_ticks':result['summary'].get('resource_wait_ticks',0)}
