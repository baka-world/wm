"""Compile explicitly configured transport workflows and resource pools.

A capacity resource groups distinct configured station ports. Sequential
transport legs are released only after the preceding leg finishes. All routes,
capacities, release times and service assumptions are configuration, not
claims about any real laboratory.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import json

from lab_transport.domain import Sample, Scenario


@dataclass(frozen=True)
class Resource:
    id: str
    capacity: int
    stations: tuple[str,...]


@dataclass(frozen=True)
class PublicBusinessRules:
    resources: tuple[Resource,...]
    workflows: dict[str,tuple[tuple[str,str],...]]

    @property
    def station_resources(self):
        return {station:resource for resource in self.resources for station in resource.stations}


@dataclass(frozen=True)
class BusinessScenario:
    scenario: Scenario
    rules: PublicBusinessRules
    stage_metadata: dict[str,tuple[str,int,str]]
    stage_next: dict[str,str]
    jobs: tuple[dict,...]


def compile_scenario(base: Scenario, config: dict) -> BusinessScenario:
    if not isinstance(config,dict):
        raise ValueError('business config must be an object')
    resources=[]
    covered=set()
    for item in config.get('resources',[]):
        rid=item.get('id')
        ports=tuple(item.get('stations',()))
        cap=item.get('capacity')
        if (not isinstance(rid,str) or not rid or not ports or len(set(ports))!=len(ports) or
                any(p not in base.stations or p in covered for p in ports) or
                not isinstance(cap,int) or isinstance(cap,bool) or not 1<=cap<=len(ports)):
            raise ValueError('resource must have unique known physical ports and 1 <= capacity <= port count')
        if any(r.id==rid for r in resources):
            raise ValueError('resource IDs must be unique')
        resources.append(Resource(rid,cap,ports));covered.update(ports)
    for sid in base.stations:
        if sid not in covered:
            name=f'station:{sid}'
            if any(r.id==name for r in resources):raise ValueError('resource ID collides with station default')
            resources.append(Resource(name,1,(sid,)))
    workflows={}
    for name,stages in config.get('workflows',{}).items():
        if not isinstance(name,str) or not name or not isinstance(stages,list) or not stages:
            raise ValueError('workflow must have a name and at least one stage')
        route=[]
        for stage in stages:
            if not isinstance(stage,dict):raise ValueError('workflow stage must be an object')
            pickup,dropoff=stage.get('pickup'),stage.get('dropoff')
            if pickup not in base.stations or dropoff not in base.stations or pickup==dropoff:
                raise ValueError('workflow stage must name two distinct known stations')
            if route and route[-1][1]!=pickup:
                raise ValueError('workflow stages must connect without teleporting the sample')
            route.append((pickup,dropoff))
        workflows[name]=tuple(route)
    jobs=config.get('jobs')
    if not isinstance(jobs,list) or not jobs:
        raise ValueError('business jobs must be a nonempty list')
    max_ticks=config.get('max_ticks',base.max_ticks)
    if not isinstance(max_ticks,int) or isinstance(max_ticks,bool) or max_ticks<1:
        raise ValueError('max_ticks must be positive integer')
    samples=[];metadata={};next_stage={};seen=set(); normalized=[]
    for job in jobs:
        jid=job.get('id');name=job.get('workflow');release=job.get('release_tick');due=job.get('deadline_tick')
        if (not isinstance(jid,str) or not jid or '::stage' in jid or jid in seen or name not in workflows or
                not isinstance(release,int) or isinstance(release,bool) or not 0<=release<max_ticks or
                (due is not None and (not isinstance(due,int) or isinstance(due,bool) or due<release))):
            raise ValueError('invalid workflow job or release/deadline')
        seen.add(jid);route=workflows[name];ids=[]
        for index,(pickup,dropoff) in enumerate(route):
            sid=f'{jid}::stage{index+1}'
            ids.append(sid)
            samples.append(Sample(sid,pickup,dropoff,release if index==0 else max_ticks+1,due,
                                  specimen_type=job.get('specimen_type')))
            metadata[sid]=(name,index,jid)
        for left,right in zip(ids,ids[1:]):next_stage[left]=right
        normalized.append({'id':jid,'workflow':name,'release_tick':release,'deadline_tick':due,'stages':ids})
    case=replace(base,name=config.get('name',base.name+' workflow (synthetic)'),
                 samples=tuple(samples),max_ticks=max_ticks)
    return BusinessScenario(case,PublicBusinessRules(tuple(resources),workflows),metadata,next_stage,tuple(normalized))


def load_config(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))
