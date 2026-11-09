"""CPU paired workflow/capacity experiment and independent counterfactuals."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics

import torch

from experiments.business_constraints.benchmark import CONFIG,audit,make_case,read_config
from experiments.service_value.counterfactual import load_pr9_model
from business_transport.dispatch import BusinessCPSAT
from business_transport.dynamics import CapacityDynamics
from business_transport.models import CountingStructuredService
from business_transport.observation import BusinessObserver
from business_transport.policy import BusinessPolicy
from business_transport.simulation import BusinessSimulator,resource_audit
from lab_transport.decision_value import batch_candidates,candidates_for,nearest_batch
from lab_transport.decision_value_round2 import bootstrap_paired,quantile
from lab_transport.dynamics_experiment import compare,eligible_windows
from lab_transport.dynamics_model import Geometry
from lab_transport.multisite_experiment import discrete_safety_violations
from lab_transport.paper_reuse.contracts import check_transition
from lab_transport.paper_reuse.pets_service import PETSRuleDynamics,ServiceDistribution
from lab_transport.torch_structured_adapter import ProgressAwareRule
from lab_transport.transition_schema import Transition

METHODS=('nearest','nearest_rollout','cpsat_rule','cpsat_pets','cpsat_structured')
MODELS=('rule','pets','structured')
CHECKPOINT=Path('examples/evidence/paper-reuse-integration-20261109/pets-service-cpu.pt')


def lifecycle_audit(history):
    """Check actual snapshot status transitions and robot/sample relations."""
    allowed={'unreleased':{'unreleased','waiting','assigned','in_transit'},
             'waiting':{'waiting','assigned','in_transit'},'assigned':{'assigned','in_transit'},
             'in_transit':{'in_transit','completed'},'completed':{'completed'}}
    issues=[];previous={}
    for snapshot in history:
        samples={s['id']:s for s in snapshot['samples']}
        robots={r['id']:r for r in snapshot['robots']}
        for sid,sample in samples.items():
            prior=previous.get(sid)
            if prior is not None and sample['status'] not in allowed.get(prior,set()):
                issues.append({'tick':snapshot['tick'],'sample_id':sid,'type':'invalid_status_transition'})
            if sample['status'] in ('assigned','in_transit'):
                robot=robots.get(sample['robot_id'])
                if robot is None or robot['task_id']!=sid:
                    issues.append({'tick':snapshot['tick'],'sample_id':sid,'type':'robot_sample_mismatch'})
        for robot in robots.values():
            if robot['task_id'] is not None:
                sample=samples.get(robot['task_id'])
                expected={'to_pickup':'assigned','to_dropoff':'in_transit'}.get(robot['phase'])
                if sample is None or sample['status']!=expected or sample['robot_id']!=robot['id']:
                    issues.append({'tick':snapshot['tick'],'robot_id':robot['id'],'type':'robot_phase_mismatch'})
        previous={sid:s['status'] for sid,s in samples.items()}
    return issues


def make_model(name,geometry,network,threshold,seed):
    if name=='rule':return ProgressAwareRule(geometry)
    if name=='pets':return PETSRuleDynamics(geometry,ServiceDistribution.load(CHECKPOINT),seed)
    if name=='structured':return CountingStructuredService(geometry,network,torch.device('cpu'),threshold)
    raise ValueError('unknown model')


def summarize_result(result,business,policy):
    by_id={s['id']:s for s in result['samples']}
    rows=[];violations=lifecycle_audit(result['history'])
    for job in business.jobs:
        stages=[by_id[sid] for sid in job['stages']]
        completed=all(s['completed_tick'] is not None for s in stages)
        for first,second in zip(stages,stages[1:]):
            if second['assigned_tick'] is not None and (first['completed_tick'] is None or
                    second['assigned_tick']<first['completed_tick']):
                violations.append({'job':job['id'],'type':'predecessor_not_finished'})
        if completed:
            flow=stages[-1]['completed_tick']-job['release_tick']
        else:flow=None
        waits=[s['assigned_tick']-s['release_tick'] for s in stages if s['assigned_tick'] is not None]
        rows.append({'job_id':job['id'],'completed':completed,'flow':flow,
                     'stage_wait_sum':sum(waits),'assigned_stages':len(waits),'stages':stages})
    assigned=[e['sample_id'] for e in result['events'] if e['type']=='assigned']
    duplicates=sum(v-1 for v in Counter(assigned).values() if v>1)
    audit=resource_audit(result,business.rules)
    durations=[x['wall_seconds']*1000 for x in policy.log]
    solver=policy.solver
    statuses=[x for e in (solver.log if solver else []) for x in e.get('statuses',[])]
    complete=sum(x['completed'] for x in rows)
    return {'jobs':len(rows),'jobs_completed':complete,'completion_rate':complete/len(rows),
      'status':result['summary']['status'],'ticks':result['summary']['ticks'],
      'mean_flow_completed':statistics.mean(x['flow'] for x in rows if x['flow'] is not None) if complete else None,
      'restricted_mean_flow':statistics.mean((x['flow'] if x['flow'] is not None else
          max(0,result['summary']['ticks']-j['release_tick'])) for x,j in zip(rows,business.jobs)),
      'mean_stage_wait_per_job':statistics.mean(x['stage_wait_sum'] for x in rows),
      'defined_deadlines':sum(j['deadline_tick'] is not None for j in business.jobs),
      'late_jobs':sum(x['completed'] and j['deadline_tick'] is not None and
           x['stages'][-1]['completed_tick']>j['deadline_tick'] for x,j in zip(rows,business.jobs))
           if any(j['deadline_tick'] is not None for j in business.jobs) else None,
      'resource':audit,'station_unavailable_wait_ticks':result['summary'].get('station_unavailable_wait_ticks',0),
      'blocked_station_target_attempts':result['summary'].get('blocked_station_target_attempts',0),
      'discrete_safety_violations':discrete_safety_violations(result),
      'duplicate_assignments':duplicates,'lifecycle_violations':violations,
      'decision_count':len(durations),'decision_wall_ms':durations,
      'decision_wall_mean_ms':statistics.mean(durations) if durations else None,
      'decision_wall_p95_ms':quantile(sorted(durations),.95) if durations else None,
      'solver_calls':solver.calls if solver else 0,'solver_fallbacks':solver.fallbacks if solver else 0,
      'solver_truncations':solver.truncated if solver else 0,'solver_status_counts':dict(Counter(statuses)),
      'model_fallbacks':policy.fallbacks,'rollout_steps':policy.rollout_steps,
      'pets_forward_calls':sum(x['model_forward_calls'] for x in policy.log),
      'structured_service_calls':sum(x['structured_service_calls'] for x in policy.log),
      'decisions':policy.log,'solver_log':solver.log if solver else [],'job_outcomes':rows}


def run_policy(business,seed,name,network,threshold,capture=False):
    geo=Geometry(business.scenario.grid);observer=BusinessObserver()
    model_kind={'nearest':None,'nearest_rollout':'rule','cpsat_rule':'rule',
                'cpsat_pets':'pets','cpsat_structured':'structured'}[name]
    model=make_model(model_kind,geo,network,threshold,seed) if model_kind else None
    policy=BusinessPolicy(name,geo,business.rules,observer,model,seed=seed)
    rows=[]
    result=BusinessSimulator(business,'world',seed,collect_metrics=True).run(
        on_transition=rows.append if capture else None,dispatch_policy=policy,on_observation=observer.observe)
    return result,rows,summarize_result(result,business,policy)


def rich_rows(rows):
    observer=BusinessObserver();out=[]
    for row in rows:
        current=observer.observe(row.observation);future=observer.observe(row.next_observation)
        out.append(Transition(current,row.assignments,future,row.next_boundary,row.termination))
    return out


def state_prediction(rows,business,seed,network,threshold):
    geo=Geometry(business.scenario.grid);out={}
    for h in (1,3,5):
        windows,coverage=eligible_windows(rows,h);windows=windows[::5]
        metrics={name:{'correct':0,'total':0} for name in MODELS}
        for i,window in enumerate(windows):
            initial=window[0].observation;actual=window[-1].next_observation
            for name in MODELS:
                model=CapacityDynamics(make_model(name,geo,network,threshold,seed+h*10000+i),business.rules)
                state=initial
                for step,row in enumerate(window):
                    nxt=model.predict_next(state,row.assignments if step==0 else ())
                    check_transition(state,nxt,geo);state=nxt
                counts=compare(initial,actual,state)
                metrics[name]['correct']+=counts['joint_correct'];metrics[name]['total']+=counts['joint_total']
        out[str(h)]={'eligible_coverage':coverage,'selected_every_nth':5,
            'models':{name:{**values,'accuracy':values['correct']/values['total'] if values['total'] else None}
                      for name,values in metrics.items()}}
    return out


def branch_flow(result,business):
    samples={s['id']:s for s in result['samples']}
    if any(samples[j['stages'][-1]]['completed_tick'] is None for j in business.jobs):return None
    return statistics.mean(samples[j['stages'][-1]]['completed_tick']-j['release_tick'] for j in business.jobs)


def replay_branch(business,seed,factual,rows,tick,plan):
    source={r.observation.tick:r for r in rows};geo=Geometry(business.scenario.grid)
    def controller(obs,candidates):
        if obs.tick<=tick:
            if obs.tick not in source or obs!=source[obs.tick].observation:
                raise ValueError('counterfactual did not reproduce observable factual prefix')
            return source[obs.tick].assignments if obs.tick<tick else plan
        return nearest_batch(obs,candidates,geo)
    replay=BusinessSimulator(business,'world',seed,collect_metrics=True).run(dispatch_policy=controller)
    if replay['history'][:tick+1]!=factual['history'][:tick+1]:
        raise ValueError('counterfactual did not reproduce complete simulator prefix')
    audit=resource_audit(replay,business.rules)
    samples={s['id']:s for s in replay['samples']}
    bad_dependencies=any(samples[right]['assigned_tick'] is not None and
       (samples[left]['completed_tick'] is None or samples[right]['assigned_tick']<samples[left]['completed_tick'])
       for job in business.jobs for left,right in zip(job['stages'],job['stages'][1:]))
    if audit['violations'] or discrete_safety_violations(replay) or lifecycle_audit(replay['history']) or bad_dependencies:
        raise ValueError('unsafe counterfactual execution')
    return {'flow':branch_flow(replay,business),'completed_stages':replay['summary']['completed_count'],
            'status':replay['summary']['status'],'capacity_wait_ticks':audit['capacity_wait_ticks']}


def same_state(factual,rows,rich,business,seed,network,threshold):
    geo=Geometry(business.scenario.grid)
    eligible=[i for i,row in enumerate(rows) if row.assignments and len(candidates_for(row.observation))>=2]
    if not eligible:return []
    indices=sorted(set((eligible[0],eligible[len(eligible)//2])))
    results=[]
    for index in indices:
        obs=rich[index].observation;current=candidates_for(obs)
        solver=BusinessCPSAT(geo,business.rules,seed=seed,candidate_limit=4,time_limit_seconds=.5)
        plans=solver.proposals(obs,current)
        nearest,_=batch_candidates(obs,current,geo,5);nearest=tuple(x for x in nearest if x)
        forecasts={}
        for name in MODELS:
            policy=BusinessPolicy('cpsat_'+name,geo,business.rules,BusinessObserver(),
                 make_model(name,geo,network,threshold,seed),seed=seed)
            records=policy.evaluate(obs,plans)
            choice=min(range(len(plans)),key=lambda i:(records[i]['score'],i))
            forecasts[name]={'choice':choice,'scores':[r['score'] for r in records],
                             'rollout_steps':policy.rollout_steps}
        branches=[replay_branch(business,seed,factual,rows,obs.tick,plan) for plan in plans]
        # The same factual branch is never reused as a label for another action.
        candidates=[{'assignments':[asdict(a) for a in plan],**branch} for plan,branch in zip(plans,branches)]
        norm=lambda plan:tuple(sorted((a.robot_id,a.sample_id) for a in plan))
        left={norm(p) for p in nearest};right={norm(p) for p in plans}
        results.append({'tick':obs.tick,'observed_state':asdict(obs),'nearest_proposals':[list(x) for x in sorted(left)],
            'cpsat_proposals':[list(x) for x in sorted(right)],'proposal_intersection':len(left&right),
            'proposal_union':len(left|right),'solver_statuses':solver.log[-1]['statuses'],
            'solver_truncated':solver.truncated>0,'forecasts':forecasts,'branches':candidates})
    return results


def average(values):
    xs=[x for x in values if x is not None]
    return statistics.mean(xs) if xs else None


def summarize(cases):
    results={}
    for name in METHODS:
        entries=[c['methods'][name] for c in cases];times=sorted(t for e in entries for t in e['decision_wall_ms'])
        results[name]={'cases':len(entries),'jobs':sum(e['jobs'] for e in entries),
            'jobs_completed':sum(e['jobs_completed'] for e in entries),
            'mean_flow_completed':average(e['mean_flow_completed'] for e in entries),
            'restricted_mean_flow':average(e['restricted_mean_flow'] for e in entries),
            'mean_stage_wait_per_job':average(e['mean_stage_wait_per_job'] for e in entries),
            'capacity_wait_ticks':sum(e['resource']['capacity_wait_ticks'] for e in entries),
            'resource_busy_slot_ticks':sum(sum(e['resource']['busy_slot_ticks'].values()) for e in entries),
            'resource_capacity_violations':sum(len(e['resource']['violations']) for e in entries),
            'discrete_safety_violations':sum(e['discrete_safety_violations'] for e in entries),
            'duplicate_assignments':sum(e['duplicate_assignments'] for e in entries),
            'lifecycle_violations':sum(len(e['lifecycle_violations']) for e in entries),
            'decision_wall_mean_ms':statistics.mean(times) if times else None,
            'decision_wall_p95_ms':quantile(times,.95) if times else None,
            'solver_calls':sum(e['solver_calls'] for e in entries),
            'solver_success_calls':sum(e['solver_calls']-e['solver_fallbacks'] for e in entries),
            'solver_fallbacks':sum(e['solver_fallbacks'] for e in entries),
            'solver_truncations':sum(e['solver_truncations'] for e in entries),
            'model_fallbacks':sum(e['model_fallbacks'] for e in entries),
            'pets_forward_calls':sum(e['pets_forward_calls'] for e in entries),
            'structured_service_calls':sum(e['structured_service_calls'] for e in entries),
            'rollout_steps':sum(e['rollout_steps'] for e in entries)}
    paired={}
    for name in METHODS[1:]:
        for ref in (('nearest','nearest_rollout','cpsat_rule') if name.startswith('cpsat_') else ('nearest',)):
            values=[(c['layout'],c['seed'],c['methods'][name]['restricted_mean_flow']-
                     c['methods'][ref]['restricted_mean_flow']) for c in cases]
            paired[name+'_minus_'+ref]={'clustered':bootstrap_paired(values),
              'wins':sum(v < -1e-9 for _,_,v in values),'losses':sum(v > 1e-9 for _,_,v in values),
              'ties':sum(abs(v)<=1e-9 for _,_,v in values)}
    state={}
    for h in ('1','3','5'):
        state[h]={}
        for name in MODELS:
            values=[c['state_prediction'][h]['models'][name] for c in cases]
            state[h][name]={'mean_scenario_accuracy':average(x['accuracy'] for x in values),
                            'correct':sum(x['correct'] for x in values),'windows':sum(x['total'] for x in values)}
    decisions=[d for c in cases for d in c['same_state']]
    def differences(first,second):
        counts=Counter()
        for d in decisions:
            a=d['forecasts'][first]['choice'];b=d['forecasts'][second]['choice']
            if a==b:counts['same']+=1;continue
            counts['different']+=1
            fa=d['branches'][a]['flow'];fb=d['branches'][b]['flow']
            if fa is None or fb is None:counts['censored']+=1
            elif fa<fb-1e-9:counts['improved']+=1
            elif fa>fb+1e-9:counts['worse']+=1
            else:counts['equal_outcome']+=1
        return dict(counts)
    return {'scope':'synthetic paired full scenarios; cluster by layout/seed, decision points are not independent',
      'scenarios':len(cases),'clusters':len({(c['layout'],c['seed']) for c in cases}),
      'methods':results,'paired':paired,'joint_state_prediction':state,'same_state_decisions':len(decisions),
      'pets_vs_rule':differences('pets','rule'),'structured_vs_rule':differences('structured','rule'),
      'candidate_overlap_intersection':sum(d['proposal_intersection'] for d in decisions),
      'candidate_overlap_union':sum(d['proposal_union'] for d in decisions)}


def run(profile,output):
    if output.exists() and any(output.iterdir()):raise FileExistsError('refuse to overwrite experiment')
    output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);config=read_config();proof=audit(config);network,info=load_pr9_model()
    cases=[]
    for layout,seeds in config['seeds'][profile].items():
        for seed in seeds:
            for regime in config['regimes']:
                business,assumptions=make_case(config,layout,seed,regime)
                values={};factual=None;rows=None
                for name in METHODS:
                    result,transitions,metric=run_policy(business,seed,name,network,info['threshold'],capture=name=='nearest')
                    values[name]=metric
                    if name=='nearest':factual,rows=result,transitions
                rich=rich_rows(rows)
                state=state_prediction(rich,business,seed,network,info['threshold'])
                diagnostic=same_state(factual,rows,rich,business,seed,network,info['threshold'])
                cases.append({'layout':layout,'seed':seed,'regime':regime,'config':assumptions,
                              'methods':values,'state_prediction':state,'same_state':diagnostic})
                print(profile,layout,seed,regime,'done',flush=True)
    result={'profile':profile,'config_sha256':hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
       'pets_checkpoint_sha256':hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
       'structured_checkpoint':info,'scenario_audit':proof,'cases':cases,'summary':summarize(cases)}
    (output/'raw.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (output/'summary.json').write_text(json.dumps(result['summary'],ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result['summary'],ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',choices=('development','confirmation'),required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.profile,a.output)


if __name__=='__main__':main()
