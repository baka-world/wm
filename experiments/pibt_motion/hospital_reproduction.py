"""Reproduce the archived PR #12 three-robot hospital pilot verbatim.

The original temporary script is archived with this experiment. The only
translation is making its former implicit singleton station resources explicit,
as required by the current business schema; station capacities and all other
scenario semantics remain unchanged.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import statistics
import time

from business_transport.pibt_integration import CoordinatedBusinessSimulator,MotionObserver
from business_transport.policy import BusinessPolicy
from business_transport.scenario import compile_scenario
from business_transport.simulation import resource_audit
from lab_transport.clinic_map import make_clinic_scenario
from lab_transport.domain import RobotSpec
from lab_transport.dynamics_model import Geometry
from lab_transport.decision_value_round2 import quantile
from lab_transport.hospital_case import make_scenario
from lab_transport.multisite_experiment import prepare_tartu
from lab_transport.training_data import SOURCES, download_sources
from experiments.business_constraints.run import summarize_result
from experiments.pibt_motion.benchmark import safety

NAMES={'hospital_map':'hospital_2nd_floor.building.yaml',
       'tiago_config':'tiago_fleet_adapter.launch.xml',
       'clinic_map':'openrmf_clinic.building.yaml',
       'clinic_config':'openrmf_clinic_fleet.yaml'}


def load_bases(source_dir=None):
    data=({name:(Path(source_dir)/filename).read_bytes() for name,filename in NAMES.items()}
          if source_dir else download_sources())
    for name,(_,expected) in SOURCES.items():
        if hashlib.sha256(data[name]).hexdigest()!=expected:
            raise ValueError('pinned source hash mismatch: '+name)
    bases={'tartu':prepare_tartu(make_scenario(data['hospital_map'],seconds_per_tick=10,
                                                 tiago_config=data['tiago_config'])),
           'clinic':make_clinic_scenario(data['clinic_map'],data['clinic_config'])}
    return bases,{k:hashlib.sha256(v).hexdigest() for k,v in data.items()}


def pilot_case(name,base):
    keys=list(base.stations)
    if name=='tartu':
        a,b,c=keys; endpoints=(a,c);stage2=(b,c);home=(166,0)
    elif name=='clinic':
        a,c,b,d=keys; endpoints=(a,c);stage2=(b,d);home=(433,0)
    else:raise ValueError('unknown archived pilot')
    base=replace(base,robots=base.robots+(RobotSpec('R3',home,home),),max_ticks=800)
    resources=[{'id':'collection','stations':list(endpoints),'capacity':1}]
    # Original commit 6136488 added these exact singleton defaults implicitly.
    resources.extend({'id':'station:'+sid,'stations':[sid],'capacity':1}
                     for sid in base.stations if sid not in endpoints)
    cfg={'resources':resources,'workflows':{
        'wa':[{'pickup':endpoints[0],'dropoff':b},{'pickup':b,'dropoff':stage2[1]}],
        'wb':[{'pickup':endpoints[1],'dropoff':b},{'pickup':b,'dropoff':stage2[1]}],
        'wc':[{'pickup':endpoints[0],'dropoff':b}]},
        'jobs':[{'id':'J1','workflow':'wa','release_tick':0},
                {'id':'J2','workflow':'wb','release_tick':0},
                {'id':'J3','workflow':'wc','release_tick':1},
                {'id':'J4','workflow':'wa','release_tick':5},
                {'id':'J5','workflow':'wb','release_tick':9}], 'max_ticks':800}
    return compile_scenario(base,cfg),cfg


class TracingSimulator(CoordinatedBusinessSimulator):
    """Offline diagnostics surrounding the existing, unmodified move hook."""
    def __init__(self,*args,**kwargs):
        self.motion_trace=[]
        super().__init__(*args,**kwargs)

    def _move(self,tick,serviced):
        before={r.id:r.position for r in self.robots.values()}
        occupied={r.position:r.id for r in self.robots.values()}
        robots=[]
        for robot in sorted(self.robots.values(),key=lambda r:r.id):
            target=self._target(robot)
            pause=any(a<=tick<b for a,b in self.motion_pauses.get(robot.id,()))
            station=next((s for s in self.scenario.stations.values() if s.position==robot.position),None)
            capacity_wait=any(e['tick']==tick and e['type']=='resource_wait' and e.get('robot_id')==robot.id
                              for e in self.events)
            device_wait=bool(robot.id in serviced and station is not None and not station.available(tick))
            robots.append({'id':robot.id,'position':list(robot.position),'target':list(target),
                'task_id':robot.task_id,'phase':robot.phase,'serviced_this_tick':robot.id in serviced,
                'motion_paused':pause,'service_started':robot.service_started,
                'capacity_wait':capacity_wait,'device_unavailable_wait':device_wait,
                'station_at_position':station.id if station else None,
                'device_available':station.available(tick) if station else None,
                'target_occupied_by':occupied.get(target) if occupied.get(target)!=robot.id else None,
                'static_reachable':self.scenario.grid.shortest_path(robot.position,target) is not None})
        started=time.perf_counter();super()._move(tick,serviced);duration=time.perf_counter()-started
        for row in robots:row['next_position']=list(self.robots[row['id']].position)
        self.motion_trace.append({'tick':tick,'robots':robots,'wall_seconds':duration,
                                  'any_moved':any(before[rid]!=r.position for rid,r in self.robots.items())})


def topology(case):
    graph=case.scenario.grid;edges=set(graph.edges)
    adjacency=Geometry(graph)
    points=[r.position for r in case.scenario.robots]+[s.position for s in case.scenario.stations.values()]
    return {'nodes':len(graph.coordinates),'directed_edges':len(edges),
            'one_way_edges':sum((b,a) not in edges for a,b in edges),
            'all_start_and_station_pairs_reachable':all(
                adjacency.distance(a,b)<adjacency.unreachable for a in points for b in points),
            'coordinates':[{'position':list(p),'source_coordinates':list(xy)}
                           for p,xy in sorted(graph.coordinates.items())],
            'edges':[[list(a),list(b)] for a,b in sorted(edges)]}


def run_case(name,business,backend='legacy',library=None):
    geo=Geometry(business.scenario.grid);observer=MotionObserver()
    policy=BusinessPolicy('nearest',geo,business.rules,observer)
    sim=TracingSimulator(business,'world',61000,collect_metrics=True,
                         motion_backend=backend,pibt_library=library)
    result=sim.run(dispatch_policy=policy,on_observation=observer.observe)
    metrics=summarize_result(result,business,policy)
    timing=sorted(v*1000 for v in sim.motion_wall_seconds)
    longest=0;span=0
    for row in sim.motion_trace:
        span=0 if row['any_moved'] else span+1;longest=max(longest,span)
    return {'layout':name,'backend':backend,'config':{},'topology':topology(business),'summary':result['summary'],
            'resources':resource_audit(result,business.rules),'longest_no_movement':longest,
            'pibt_calls':sim.pibt.solver_calls if sim.pibt else 0,
            'jobs':metrics['jobs'],'jobs_completed':metrics['jobs_completed'],
            'jobs_unfinished':metrics['jobs']-metrics['jobs_completed'],
            'job_completion_rate':metrics['completion_rate'],
            'mean_flow_completed_only':metrics['mean_flow_completed'],
            'restricted_mean_flow':metrics['restricted_mean_flow'],
            'job_outcomes':metrics['job_outcomes'],'duplicate_assignments':metrics['duplicate_assignments'],
            'lifecycle_violations':metrics['lifecycle_violations'],'safety':safety(result['history'],geo),
            'path_wall_mean_ms':statistics.mean(timing) if timing else None,
            'path_wall_p95_ms':quantile(timing,.95) if timing else None,
            'samples':result['samples'],'events':result['events'],'history':result['history'],
            'motion_trace':sim.motion_trace}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--backend',choices=('legacy','pibt'),default='legacy')
    p.add_argument('--library',type=Path)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('refusing to overwrite pilot evidence')
    bases,source_hashes=load_bases(args.source_dir);cases=[]
    for name,base in bases.items():
        business,cfg=pilot_case(name,base);case=run_case(name,business,args.backend,args.library);case['config']=cfg
        case['initial_robots']=[asdict(r) for r in business.scenario.robots]
        case['station_configuration']={sid:asdict(station) for sid,station in business.scenario.stations.items()}
        case['motion_pauses']={};case['duration_jitter']=business.scenario.duration_jitter
        cases.append(case)
        print(name,case['summary']['completed_count'],'/',case['summary']['sample_count'],
              'longest_no_movement',case['longest_no_movement'],flush=True)
    data={'scope':'exact legacy hospital pilot replay, offline diagnostics',
          'source_sha256':source_hashes,'seed':61000,'cases':cases}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
