"""Paired 22-scenario motion comparison, including unfinished tasks."""
from __future__ import annotations

import argparse
from collections import Counter,deque
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import resource
import statistics
import time

from business_transport.lacam3_integration import HybridBusinessSimulator,RecoveryObserver
from business_transport.pibt_integration import MotionObserver
from business_transport.policy import BusinessPolicy
from experiments.business_constraints.benchmark import make_case,read_config
from experiments.business_constraints.run import summarize_result
from experiments.pibt_motion.benchmark import augment,safety
from experiments.pibt_motion.hospital_reproduction import load_bases,pilot_case
from lab_transport.decision_value_round2 import quantile
from lab_transport.dynamics_model import Geometry

CONFIG=Path('experiments/lacam3_recovery/benchmark_config.json')


class AuditSimulator(HybridBusinessSimulator):
    def __init__(self,*args,**kwargs):
        self.motion_audit=[]
        super().__init__(*args,**kwargs)

    def _move(self,tick,serviced):
        before={rid:r.position for rid,r in self.robots.items()}
        paused={rid for rid,intervals in self.motion_pauses.items() if any(a<=tick<b for a,b in intervals)}
        locked=set(serviced)|paused
        targets={rid:self._target(r) for rid,r in self.robots.items()}
        tasks={rid:{'task_id':r.task_id,'phase':r.phase} for rid,r in self.robots.items()}
        super()._move(tick,serviced)
        after={rid:r.position for rid,r in self.robots.items()}
        blocked={before[rid] for rid in locked}
        locked_moves=sum(before[rid]!=after[rid] for rid in locked)
        locked_crossings=sum(rid not in locked and before[rid]!=after[rid] and after[rid] in blocked for rid in before)
        self.motion_audit.append({'tick':tick,'before':before,'after':after,'targets':targets,
            'tasks':tasks,'locked':tuple(sorted(locked)),'locked_moves':locked_moves,
            'locked_crossings':locked_crossings})


def progress(result):
    end=result['summary']['ticks']
    deliveries=sorted(e['tick'] for e in result['events'] if e['type']=='delivered')
    last=0;longest=0
    for tick in deliveries:longest=max(longest,tick-last);last=tick
    longest=max(longest,end-last)
    recent=deque(maxlen=16);cycles=0
    for snapshot in result['history']:
        state=tuple(tuple(r['position']) for r in snapshot['robots'])
        for i,old in enumerate(recent):
            if old==state and any(v!=state for v in list(recent)[i+1:]):
                cycles+=1;break
        recent.append(state)
    return {'last_stage_completion_tick':deliveries[-1] if deliveries else None,
            'longest_no_stage_progress':longest,'repeated_joint_state_visits':cycles}


def plan_sessions(events):
    """Associate each complete plan with the actual cached steps executed."""
    sessions=[];active=None
    def close():
        nonlocal active
        if active is not None:
            active['executed_over_planned']=active['executed_steps']/active['planned_steps']
            sessions.append(active);active=None
    for event in events:
        if active is not None and (event['cache_invalidated'] or not event['cache_used']):close()
        if event['triggered'] and event['status']=='solved' and event['plan_steps']>0:
            close();active={'tick':event['tick'],'planned_steps':event['plan_steps'],'executed_steps':0}
        if active is not None and event['cache_used']:active['executed_steps']+=1
    close()
    return sessions


def run_one(business,seed,backend,pibt_library,lacam_worker,*,first_step_only=False):
    geometry=Geometry(business.scenario.grid)
    observer=RecoveryObserver() if backend=='hybrid' else MotionObserver()
    policy=BusinessPolicy('nearest',geometry,business.rules,observer)
    class Simulator(AuditSimulator):
        def _move(self,tick,serviced):
            super()._move(tick,serviced)
            if (first_step_only and self.hybrid_enabled and self.recovery_log and
                    self.recovery_log[-1]['triggered'] and self.recovery_log[-1]['status']=='solved'):
                # Post-hoc controlled diagnostic: same trigger and full solve,
                # discard the remaining plan; do not change any business rule.
                self.recovery_snapshot=replace(self.recovery_snapshot,cached_path=(),cached_key=())
    sim=Simulator(business,'world',seed,motion_backend=backend,pibt_library=pibt_library,
                  lacam3_worker=lacam_worker,collect_metrics=True)
    cpu_start=time.process_time();wall_start=time.perf_counter()
    result=sim.run(dispatch_policy=policy,on_observation=observer.observe)
    parent_cpu_ms=(time.process_time()-cpu_start)*1000;simulation_wall_ms=(time.perf_counter()-wall_start)*1000
    metrics=summarize_result(result,business,policy);safe=safety(result['history'],geometry)
    attempts=[e for e in sim.recovery_log if e['triggered']]
    invoked=[e for e in attempts if e.get('worker_invoked')]
    timing=sorted(e['planning_wall_ms'] for e in invoked if e['planning_wall_ms'] is not None)
    step_times=sorted(v*1000 for v in sim.motion_wall_seconds)
    unfinished=[]
    for job,outcome in zip(business.jobs,metrics['job_outcomes']):
        if not outcome['completed']:
            unfinished.append({'job_id':job['id'],'restricted_flow':max(0,result['summary']['ticks']-job['release_tick'])})
    successful=[e for e in attempts if e['status']=='solved'];used=sum(bool(e['cache_used']) for e in sim.recovery_log)
    sessions=plan_sessions(sim.recovery_log)
    return {'backend':backend,'jobs':metrics['jobs'],'jobs_completed':metrics['jobs_completed'],
      'jobs_unfinished':metrics['jobs']-metrics['jobs_completed'],'job_completion_rate':metrics['completion_rate'],
      'stages':result['summary']['sample_count'],'stages_completed':result['summary']['completed_count'],
      'ticks':result['summary']['ticks'],'status':result['summary']['status'],
      'mean_flow_completed_only':metrics['mean_flow_completed'],'restricted_mean_flow':metrics['restricted_mean_flow'],
      'unfinished_restricted_flow':unfinished,'progress':progress(result),'safety':safe,
      'locked_moves':sum(e['locked_moves'] for e in sim.motion_audit),
      'locked_crossings':sum(e['locked_crossings'] for e in sim.motion_audit),
      'resource_violations':metrics['resource']['violations'],'duplicate_assignments':metrics['duplicate_assignments'],
      'lifecycle_violations':metrics['lifecycle_violations'],
      'station_unavailable_wait_ticks':metrics['station_unavailable_wait_ticks'],
      'capacity_wait_ticks':metrics['resource']['capacity_wait_ticks'],
      'recovery_triggers':len(attempts),'recovery_status_counts':dict(Counter(e['status'] for e in attempts)),
      'official_solver_calls':len(invoked),'full_plan_successes':len(successful),
      'official_replanning_calls':max(0,len(invoked)-1),'recovery_retriggers':max(0,len(attempts)-1),
      'full_plan_lengths':[e['plan_steps'] for e in successful],
      'cache_steps_executed':used,'cache_invalidations':sum(e['cache_invalidated'] for e in sim.recovery_log),
      'plan_execution_sessions':sessions,
      'plan_steps_per_success':statistics.mean(e['plan_steps'] for e in successful) if successful else None,
      'executed_steps_per_success':used/len(successful) if successful else None,
      'planning_wall_mean_ms':statistics.mean(timing) if timing else None,
      'planning_wall_p95_ms':quantile(timing,.95) if timing else None,
      'planning_total_wall_ms':sum(timing),'planning_wall_ms':timing,
      'planning_cpu_ms_known':sum(e['planning_cpu_ms'] for e in invoked if e['planning_cpu_ms'] is not None),
      'planning_cpu_unknown_calls':sum(e['planning_cpu_ms'] is None for e in invoked),
      'planner_peak_rss_kb':max((e['planning_peak_rss_kb'] for e in invoked
                                 if e['planning_peak_rss_kb'] is not None),default=None),
      'parent_process_peak_rss_kb_cumulative':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      'parent_process_cpu_ms':parent_cpu_ms,'simulation_wall_ms':simulation_wall_ms,
      'motion_total_wall_ms':sum(step_times),'motion_wall_mean_ms':statistics.mean(step_times) if step_times else None,
      'motion_wall_p95_ms':quantile(step_times,.95) if step_times else None,'motion_wall_ms':step_times,
      'pibt_native_calls':sim.pibt.solver_calls if sim.pibt else 0,
      'recovery_log':sim.recovery_log,'motion_audit':sim.motion_audit,'events':result['events'],
      'history':result['history'],'job_outcomes':metrics['job_outcomes']}


def scenarios(source_dir):
    bases,hashes=load_bases(source_dir);cases=[]
    for name,base in bases.items():
        business,config=pilot_case(name,base);cases.append((name,61000,'hospital_pilot',3,business,config))
    cfg=read_config();fixed=json.loads(Path('experiments/pibt_motion/benchmark_config.json').read_text())
    for layout,seeds in cfg['seeds']['confirmation'].items():
        for seed in seeds:
            for regime in cfg['regimes']:
                business,config=make_case(cfg,layout,seed,regime)
                cases.append((layout,seed,regime,3,business,config))
    for item in fixed['extra_five_robot_cases']:
        for regime in item['regimes']:
            business,config=make_case(cfg,item['layout'],item['seed'],regime)
            cases.append((item['layout'],item['seed'],regime,5,augment(business,item['robots']),config))
    return cases,hashes


def run(source_dir,pibt_library,lacam_worker,output,ablation_output=None):
    if output.exists() or (ablation_output and ablation_output.exists()):raise FileExistsError('refuse to overwrite experiment')
    config=json.loads(CONFIG.read_text());cases,hashes=scenarios(source_dir);records=[]
    for layout,seed,regime,count,business,assumptions in cases:
        adjacency=Geometry(business.scenario.grid).adjacency
        edges={(a,b) for a,values in adjacency.items() for b in values}
        one_way=sum((b,a) not in edges for a,b in edges)
        methods={name:run_one(business,seed,name,pibt_library,lacam_worker) for name in ('legacy','pibt','hybrid')}
        records.append({'layout':layout,'seed':seed,'regime':regime,'robots':count,
                        'scenario_config':assumptions,'lacam3_topology_support':
                        {'supported':one_way==0,'one_way_edges':one_way,
                         'reason':'bidirectional' if one_way==0 else 'unsupported_directed'},'methods':methods})
        print(layout,seed,regime,count,*(f"{k}:{v['jobs_completed']}/{v['jobs']}" for k,v in methods.items()),flush=True)
    data={'config':config,'config_sha256':hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
          'source_sha256':hashes,'cases':records}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    if ablation_output is not None:
        clinic=next(c for c in cases if c[0]=='clinic')
        outcome=run_one(clinic[4],clinic[1],'hybrid',pibt_library,lacam_worker,first_step_only=True)
        ablation_output.write_text(json.dumps({'scope':'post-hoc diagnostic: discard cached path after first step',
            'layout':'clinic','outcome':outcome},ensure_ascii=False,indent=2)+'\n')
        print('clinic first-step-only diagnostic',outcome['jobs_completed'],'/',outcome['jobs'],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-dir',type=Path,required=True)
    p.add_argument('--pibt-library',type=Path,required=True);p.add_argument('--lacam-worker',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--ablation-output',type=Path)
    a=p.parse_args();run(a.source_dir,a.pibt_library,a.lacam_worker,a.output,a.ablation_output)


if __name__=='__main__':main()
