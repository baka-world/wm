"""Frozen PR #12 regression and additional five-robot motion comparison."""
from __future__ import annotations

import argparse
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
import statistics

from business_transport.pibt_integration import CoordinatedBusinessSimulator,MotionObserver
from business_transport.policy import BusinessPolicy
from experiments.business_constraints.benchmark import make_case,read_config
from experiments.business_constraints.run import summarize_result
from lab_transport.decision_value_round2 import quantile
from lab_transport.domain import RobotSpec
from lab_transport.dynamics_model import Geometry

CONFIG=Path('experiments/pibt_motion/benchmark_config.json')


def augment(business,robots):
    specs=tuple(RobotSpec(x['id'],tuple(x['position']),tuple(x['home'])) for x in robots)
    all_robots=business.scenario.robots+specs
    if len({r.id for r in all_robots})!=len(all_robots) or len({r.position for r in all_robots})!=len(all_robots):
        raise ValueError('extra robots must have unique identities and starting positions')
    if len({r.home for r in all_robots})!=len(all_robots):raise ValueError('extra homes must be unique')
    if any(not business.scenario.grid.contains(r.position) or not business.scenario.grid.contains(r.home)
           for r in all_robots):raise ValueError('extra robot is outside the topology')
    return replace(business,scenario=replace(business.scenario,robots=all_robots))


def safety(history,geometry):
    vertices=swaps=invalid=0;longest=0;run=0
    for before,after in zip(history,history[1:]):
        a={r['id']:tuple(r['position']) for r in before['robots']}
        b={r['id']:tuple(r['position']) for r in after['robots']}
        vertices+=len(b)-len(set(b.values()))
        ids=sorted(a)
        swaps+=sum(a[x]!=a[y] and a[x]==b[y] and a[y]==b[x]
                   for i,x in enumerate(ids) for y in ids[i+1:])
        invalid+=sum(b[rid]!=pos and b[rid] not in geometry.adjacency[pos] for rid,pos in a.items())
        run=run+1 if a==b else 0;longest=max(longest,run)
    return {'vertex_collisions':vertices,'edge_swaps':swaps,'invalid_edges':invalid,
            'longest_no_robot_move':longest}


def run_one(business,seed,backend,library,planner_seed):
    geometry=Geometry(business.scenario.grid);observer=MotionObserver()
    policy=BusinessPolicy('nearest',geometry,business.rules,observer)
    sim=CoordinatedBusinessSimulator(business,'world',seed,motion_backend=backend,
        pibt_library=library,planner_seed=planner_seed,collect_metrics=True)
    result=sim.run(dispatch_policy=policy,on_observation=observer.observe)
    metrics=summarize_result(result,business,policy)
    timing=sorted(v*1000 for v in sim.motion_wall_seconds)
    return {'jobs':metrics['jobs'],'jobs_completed':metrics['jobs_completed'],
       'jobs_unfinished':metrics['jobs']-metrics['jobs_completed'],
       'job_completion_rate':metrics['completion_rate'],
       'stages':result['summary']['sample_count'],'stages_completed':result['summary']['completed_count'],
       'mean_flow_completed_only':metrics['mean_flow_completed'],'restricted_mean_flow':metrics['restricted_mean_flow'],
       'ticks':result['summary']['ticks'],'status':result['summary']['status'],
       'station_unavailable_wait_ticks':metrics['station_unavailable_wait_ticks'],
       'capacity_wait_ticks':metrics['resource']['capacity_wait_ticks'],
       'blocked_move_attempts_legacy':result['summary']['blocked_move_attempts'] if backend=='legacy' else None,
       'pibt_native_calls':sim.pibt.solver_calls if sim.pibt else 0,
       'pibt_blocked_goals':sim.pibt.locked_goals if sim.pibt else None,
       'pibt_unreachable_goals':sim.pibt.unreachable_goals if sim.pibt else None,
       'path_wall_mean_ms':statistics.mean(timing) if timing else None,
       'path_wall_p95_ms':quantile(timing,.95) if timing else None,'path_wall_ms':timing,
       'duplicate_assignments':metrics['duplicate_assignments'],
       'lifecycle_violations':metrics['lifecycle_violations'],
       'resource_violations':metrics['resource']['violations'],
       'safety':safety(result['history'],geometry),'job_outcomes':metrics['job_outcomes'],
       'history':result['history'],'events':result['events']}


def run(output,library):
    if output.exists():raise FileExistsError('refusing to overwrite motion experiment')
    config=json.loads(CONFIG.read_text());pr12=read_config()
    if config['planner_seed'] in sum((list(v) for v in pr12['seeds']['confirmation'].values()),[]):
        raise ValueError('planner seed must be independent of environment seeds')
    cases=[]
    for layout,seeds in pr12['seeds']['confirmation'].items():
        for seed in seeds:
            for regime in pr12['regimes']:
                business,assumptions=make_case(pr12,layout,seed,regime)
                cases.append((layout,seed,regime,3,business,assumptions))
    for item in config['extra_five_robot_cases']:
        for regime in item['regimes']:
            business,assumptions=make_case(pr12,item['layout'],item['seed'],regime)
            cases.append((item['layout'],item['seed'],regime,5,augment(business,item['robots']),assumptions))
    records=[]
    for layout,seed,regime,count,business,assumptions in cases:
        pair={}
        for backend in ('legacy','pibt'):
            pair[backend]=run_one(business,seed,backend,library,config['planner_seed'])
        records.append({'layout':layout,'seed':seed,'regime':regime,'robots':count,
                        'scenario':assumptions,'methods':pair})
        print(layout,seed,regime,count,
              *(f"{k}:{v['jobs_completed']}/{v['jobs']}" for k,v in pair.items()),flush=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({'config':config,'config_sha256':hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
                                  'cases':records},ensure_ascii=False,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();run(a.output,a.library)


if __name__=='__main__':main()
