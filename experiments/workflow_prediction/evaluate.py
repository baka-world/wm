"""Frozen CPU paired workflow forecasts, counterfactuals and closed loops."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
import statistics

import torch

from business_transport.dispatch import BusinessCPSAT
from business_transport.dynamics import CapacityDynamics,WorkflowDynamics
from business_transport.lacam3_integration import HybridBusinessSimulator,RecoveryMovementModel,RecoveryObserver
from business_transport.policy import BusinessPolicy
from business_transport.simulation import resource_audit
from business_transport.workflow_transition import check_business_transition
from experiments.business_constraints.benchmark import make_case,read_config
from experiments.business_constraints.run import make_model,summarize_result,lifecycle_audit
from experiments.pibt_motion.benchmark import safety as motion_safety
from experiments.service_value.counterfactual import load_pr9_model
from experiments.workflow_prediction.config_audit import CONFIG,audit
from lab_transport.decision_value import candidates_for,nearest_batch
from lab_transport.decision_value_round2 import bootstrap_paired,quantile
from lab_transport.dynamics_experiment import compare,eligible_windows
from lab_transport.dynamics_model import Geometry
from lab_transport.paper_reuse.contracts import check_transition
from lab_transport.transition_schema import Transition

METHODS=('nearest','old_rule','workflow_rule','workflow_pets','workflow_structured')
MODELS=('rule','pets','structured')


def make_adapter(name,geometry,rules,network,threshold,pibt,lacam,seed):
    model=make_model(name,geometry,network,threshold,seed)
    return RecoveryMovementModel(model,geometry,rules,pibt,lacam)


def policy_for(method,business,geometry,network,threshold,pibt,lacam,config):
    observer=RecoveryObserver()
    kinds={'nearest':None,'old_rule':'rule','workflow_rule':'rule',
           'workflow_pets':'pets','workflow_structured':'structured'}
    kind=kinds[method]
    model=make_adapter(kind,geometry,business.rules,network,threshold,pibt,lacam,config['planner_seed']) if kind else None
    name='nearest' if not kind else 'cpsat_'+kind
    workflow=method.startswith('workflow_')
    policy=BusinessPolicy(name,geometry,business.rules,observer,model,seed=config['planner_seed'],
        horizon=config['horizon'],particles=config['particles'],budget=config['candidate_budget'],
        solver_seconds=config['cp_sat_time_limit_seconds'],workflow_prediction=workflow,
        scoring='workflow' if workflow else 'legacy')
    return policy,observer,model


def run_method(business,seed,method,network,threshold,pibt,lacam,config):
    geometry=Geometry(business.scenario.grid)
    policy,observer,model=policy_for(method,business,geometry,network,threshold,pibt,lacam,config)
    sim=HybridBusinessSimulator(business,'world',seed,motion_backend=config['motion_backend'],
        pibt_library=pibt,lacam3_worker=lacam,planner_seed=config['planner_seed'],collect_metrics=True)
    rows=[]
    result=sim.run(dispatch_policy=policy,on_observation=observer.observe,
                   on_transition=rows.append if method=='nearest' else None)
    metrics=summarize_result(result,business,policy)
    metrics['prediction_motion_calls']=model.pibt.solver_calls if model else 0
    metrics['model_prediction_calls']=policy.prediction_calls
    metrics['model_prediction_total_wall_ms']=policy.prediction_wall_seconds*1000
    metrics['execution_motion_calls']=sim.pibt.solver_calls
    metrics['motion_safety']=motion_safety(result['history'],geometry)
    metrics['locked_move_violations']=sum(
        tuple(next(r['position'] for r in result['history'][e['tick']+1]['robots'] if r['id']==rid)) != tuple(e['positions_before'][rid])
        for e in sim.recovery_log for rid in e['locked'])
    metrics['recovery_triggers']=sum(e['triggered'] for e in sim.recovery_log)
    metrics['recovery_solver_calls']=sum(bool(e.get('worker_invoked')) for e in sim.recovery_log)
    metrics['stage_release_events']=[{'tick':e['tick'],'sample_id':e['sample_id'],'predecessor':e['predecessor']}
                                     for e in result['events'] if e['type']=='stage_ready']
    return result,rows,metrics


def rich_rows(rows):
    observer=RecoveryObserver();output=[]
    for row in rows:
        first=observer.observe(row.observation);second=observer.observe(row.next_observation)
        output.append(Transition(first,row.assignments,second,row.next_boundary,row.termination))
    return output


def job_key(sample):return getattr(sample,'job_id','') or sample.id


def known_samples(state,initial):
    known={job_key(s) for s in initial.samples}
    return {s.id:s for s in state.samples if job_key(s) in known}


def sample_signature(state,initial):
    samples=known_samples(state,initial)
    fields=('id','job_id','workflow','stage_index','pickup','dropoff','status','robot_id',
            'release_tick','deadline_tick','specimen_type','packaging','temperature_condition','rule_profile')
    return tuple((sid,tuple(getattr(s,f,None) for f in fields)) for sid,s in sorted(samples.items()))


def robot_signature(state):
    return tuple((r.id,r.position,r.task_id,r.phase) for r in state.robots)


def progress_signature(state):
    return tuple((r.id,r.service_elapsed,getattr(r,'service_active_elapsed',None),
                  getattr(r,'service_paused_elapsed',None),getattr(r,'operational_available',None)) for r in state.robots)


def station_signature(state):
    return tuple((s.id,getattr(s,'resource_id',None),getattr(s,'resource_capacity',None),
                  getattr(s,'resource_occupied',None),getattr(s,'capacity_blocked',None),
                  getattr(s,'device_available',s.available),getattr(s,'occupant_robot_id',None),
                  getattr(s,'service_task_id',None),getattr(s,'service_session_active',None),
                  getattr(s,'service_in_progress',None),getattr(s,'waiting_pickups',None),
                  getattr(s,'approaching_pickups',None),getattr(s,'approaching_dropoffs',None)) for s in state.stations)


def service_events(before,after,initial):
    old=known_samples(before,initial);new=known_samples(after,initial);events=set()
    for sid,sample in old.items():
        if sid not in new:continue
        if sample.status in ('waiting','assigned') and new[sid].status=='in_transit':events.add((sid,'pickup'))
        if sample.status=='in_transit' and new[sid].status=='completed':events.add((sid,'dropoff'))
    return events


def successor_events(before,after,initial):
    old=known_samples(before,initial);new=known_samples(after,initial)
    return {sid for sid in new if sid not in old and getattr(new[sid],'stage_index',0)>0}


def final_jobs(state,initial,rules):
    grouped={}
    for s in known_samples(state,initial).values():grouped.setdefault(job_key(s),[]).append(s)
    answer={}
    for job,samples in grouped.items():
        workflow=getattr(samples[0],'workflow','')
        last=len(rules.workflows[workflow])-1 if workflow else 0
        answer[job]=any(getattr(s,'stage_index',0)==last and s.status=='completed' for s in samples)
    return answer


def dynamic_windows(rows,horizon,stride):
    counts=Counter(candidate_windows=max(0,len(rows)-horizon+1))
    selected=[]
    for i in range(max(0,len(rows)-horizon+1)):
        window=rows[i:i+horizon];initial=window[0].observation
        if any(row.assignments for row in window[1:]):counts['future_assignments']+=1;continue
        jobs={job_key(s) for s in initial.samples}
        if any(job_key(s) not in jobs for row in window for s in row.next_observation.samples):
            counts['external_arrival']+=1;continue
        device={s.id:getattr(s,'device_available',s.available) for s in initial.stations}
        if any({s.id:getattr(s,'device_available',s.available) for s in row.next_observation.stations}!=device
               for row in window):counts['device_change']+=1;continue
        counts['eligible']+=1
        event=any(successor_events(row.observation,row.next_observation,initial) for row in window)
        if i%stride==0 or event:
            selected.append((i,window));counts['selected']+=1;counts['selected_with_successor']+=bool(event)
    return selected,dict(counts)


def evaluate_state_windows(rows,business,geometry,network,threshold,pibt,lacam,config):
    result={};stride=config['prediction_window_stride']
    # A one-step old-model audit counts each actual legal successor once.
    misses=[];old_models={name:CapacityDynamics(make_adapter(name,geometry,business.rules,network,threshold,
        pibt,lacam,config['planner_seed']),business.rules) for name in MODELS}
    for index,row in enumerate(rows):
        created=successor_events(row.observation,row.next_observation,row.observation)
        if not created:continue
        details={}
        for name,old_model in old_models.items():
            old_model.reset_seed(config['planner_seed']+index)
            prediction=old_model.predict_next(row.observation,row.assignments)
            check_transition(row.observation,prediction,geometry)
            predicted_ids={s.id for s in prediction.samples}
            prior={s.id:s for s in row.observation.samples}
            actual={s.id:s for s in row.next_observation.samples}
            predicted={s.id:s for s in prediction.samples}
            predecessors=[s.id for s in prior.values() if s.status!='completed' and
                          actual[s.id].status=='completed' and
                          any(actual[sid].job_id==s.job_id and actual[sid].stage_index==s.stage_index+1
                              for sid in created)]
            details[name]={'missing_successor_ids':sorted(created-predicted_ids),
                           'predecessor_predicted_completed':sum(predicted[sid].status=='completed' for sid in predecessors)}
        misses.append({'tick':row.next_observation.tick,'ids':sorted(created),'legacy_models':details})
    for horizon in config['prediction_horizons']:
        selected,coverage=dynamic_windows(rows,horizon,stride)
        accum={name:Counter() for name in MODELS}
        details=[]
        adapters={name:WorkflowDynamics(make_adapter(name,geometry,business.rules,network,threshold,pibt,lacam,
                                                       config['planner_seed']),business.rules) for name in MODELS}
        for index,window in selected:
            initial=window[0].observation;actual=window[-1].next_observation
            entry={'start_tick':initial.tick,'end_tick':actual.tick,'models':{}}
            for name,adapter in adapters.items():
                adapter.reset_seed(config['planner_seed']+horizon*10000+index)
                state=initial;service_true=service_pred=service_tp=0;actual_new=pred_new=release_tp=0
                conditional=logic_errors=0;conditioned_resource=resource_errors=0
                service_mismatch=release_disagreement=0;step_records=[]
                for j,row in enumerate(window):
                    actions=row.assignments if j==0 else ()
                    nxt=adapter.predict_next(state,actions)
                    check_business_transition(state,nxt,geometry,business.rules,actions)
                    truth=service_events(row.observation,row.next_observation,initial)
                    pred=service_events(state,nxt,initial)
                    true_release=successor_events(row.observation,row.next_observation,initial)
                    pred_release=successor_events(state,nxt,initial)
                    service_true+=len(truth);service_pred+=len(pred);service_tp+=len(truth&pred)
                    actual_new+=len(true_release);pred_new+=len(pred_release);release_tp+=len(true_release&pred_release)
                    same_before=(sample_signature(state,initial)==sample_signature(row.observation,initial) and
                                 robot_signature(state)==robot_signature(row.observation) and
                                 progress_signature(state)==progress_signature(row.observation) and
                                 station_signature(state)==station_signature(row.observation))
                    service_mismatch+=truth!=pred
                    release_disagreement+=true_release!=pred_release
                    if same_before and truth==pred:
                        conditional+=1;logic_errors+=true_release!=pred_release
                        if (sample_signature(nxt,initial)==sample_signature(row.next_observation,initial) and
                                robot_signature(nxt)==robot_signature(row.next_observation) and
                                progress_signature(nxt)==progress_signature(row.next_observation)):
                            conditioned_resource+=1
                            resource_errors+=station_signature(nxt)!=station_signature(row.next_observation)
                    step_records.append({'tick':state.tick,'factual_service_events':sorted(truth),
                        'predicted_service_events':sorted(pred),'factual_successor_releases':sorted(true_release),
                        'predicted_successor_releases':sorted(pred_release),'same_observed_input':same_before})
                    state=nxt
                sample_ok=sample_signature(state,initial)==sample_signature(actual,initial)
                motion_ok=robot_signature(state)==robot_signature(actual)
                station_ok=station_signature(state)==station_signature(actual)
                jobs_ok=final_jobs(state,initial,business.rules)==final_jobs(actual,initial,business.rules)
                values={'joint_correct':int(sample_ok and motion_ok and station_ok and jobs_ok),'windows':1,
                    'sample_identity_status_correct':int(sample_ok),'robot_position_task_correct':int(motion_ok),
                    'station_capacity_correct':int(station_ok),'job_final_state_correct':int(jobs_ok),
                    'service_actual':service_true,'service_predicted':service_pred,'service_true_positive':service_tp,
                    'successor_actual':actual_new,'successor_predicted':pred_new,'successor_exact_tick':release_tp,
                    'service_mismatch_steps':service_mismatch,'successor_release_disagreement_steps':release_disagreement,
                    'conditional_service_matching_steps':conditional,'conditional_release_logic_errors':logic_errors,
                    'conditional_resource_checks':conditioned_resource,'conditional_resource_errors':resource_errors}
                accum[name].update(values);entry['models'][name]={**values,'steps':step_records}
            details.append(entry)
        legacy_windows,legacy_coverage=eligible_windows(rows,horizon);legacy_windows=legacy_windows[::stride]
        fixed={}
        for name in MODELS:
            fixed[name]={}
            for upgraded in (False,True):
                adapter_cls=WorkflowDynamics if upgraded else CapacityDynamics
                model=adapter_cls(make_adapter(name,geometry,business.rules,network,threshold,pibt,lacam,
                                               config['planner_seed']),business.rules)
                correct=total=0
                for index,window in enumerate(legacy_windows):
                    initial=window[0].observation;model.reset_seed(config['planner_seed']+horizon*10000+index)
                    state=initial
                    for step,row in enumerate(window):
                        actions=row.assignments if step==0 else ()
                        nxt=model.predict_next(state,actions)
                        if upgraded:check_business_transition(state,nxt,geometry,business.rules,actions)
                        else:check_transition(state,nxt,geometry)
                        state=nxt
                    original_ids={s.id for s in initial.samples}
                    projection=replace(state,samples=tuple(s for s in state.samples if s.id in original_ids))
                    counts=compare(initial,window[-1].next_observation,projection)
                    correct+=counts['joint_correct'];total+=counts['joint_total']
                fixed[name]['workflow' if upgraded else 'legacy']={'correct':correct,'total':total}
        result[str(horizon)]={'dynamic_coverage':coverage,'dynamic':{n:dict(v) for n,v in accum.items()},
                              'dynamic_windows':details,'fixed_legacy_coverage':legacy_coverage,
                              'fixed_selected_every_nth':stride,'fixed_entity':fixed}
    return {'factual_successor_release_events':misses,
            'old_fixed_entity_model_missing_successors':sum(len(x['legacy_models']['rule']['missing_successor_ids']) for x in misses),
            'horizons':result}


def replay_branch(business,seed,rows,factual,tick,plan,pibt,lacam,config,*,trace_limit=0,future_service_seed=None):
    source={r.observation.tick:r for r in rows};geometry=Geometry(business.scenario.grid)
    def controller(obs,candidates):
        if obs.tick<=tick:
            if obs.tick not in source or obs!=source[obs.tick].observation:
                raise ValueError('counterfactual observable prefix differs')
            return source[obs.tick].assignments if obs.tick<tick else plan
        return nearest_batch(obs,candidates,geometry)
    sim=HybridBusinessSimulator(business,'world',seed,motion_backend=config['motion_backend'],
            pibt_library=pibt,lacam3_worker=lacam,planner_seed=config['planner_seed'],collect_metrics=True)
    captured=[]
    # Offline robustness only: preserve the factual prefix's existing keyed
    # service draws, while changing the independent draw for sessions STARTED
    # at or after this decision. Never alter future task releases/outage plans.
    if future_service_seed is not None:
        if not isinstance(future_service_seed,int) or isinstance(future_service_seed,bool):
            raise ValueError('future service seed must be an integer')
        original_handle=sim._handle_station
        original_duration=sim._service_duration
        current_tick={'value':-1}
        def handle_with_clock(current,robot):
            current_tick['value']=current
            return original_handle(current,robot)
        def duration_with_future_noise(sample_id,phase,station_id):
            if current_tick['value']<tick:
                return original_duration(sample_id,phase,station_id)
            station=sim.scenario.stations[station_id]
            limit=(sim.scenario.duration_jitter if station.jitter_ticks is None
                   else station.jitter_ticks)
            key=f"{future_service_seed}\\0{sample_id}\\0{phase}\\0{station_id}".encode('utf-8')
            draw=int.from_bytes(hashlib.sha256(key).digest()[:8],'big')
            return station.service_ticks + draw%(limit+1)
        sim._handle_station=handle_with_clock
        sim._service_duration=duration_with_future_noise
    result=sim.run(dispatch_policy=controller,on_transition=captured.append if trace_limit else None)
    if result['history'][:tick+1]!=factual['history'][:tick+1]:
        raise ValueError('counterfactual simulator prefix differs')
    samples={s['id']:s for s in result['samples']};flows=[];restricted=[]
    for job in business.jobs:
        end=samples[job['stages'][-1]]['completed_tick']
        if end is not None:flows.append(end-job['release_tick'])
        restricted.append((end if end is not None else result['summary']['ticks'])-job['release_tick'])
    safety=motion_safety(result['history'],geometry);resources=resource_audit(result,business.rules)
    lifecycle=lifecycle_audit(result['history'])
    assignments=[e['sample_id'] for e in result['events'] if e['type']=='assigned']
    duplicate=sum(n-1 for n in Counter(assignments).values() if n>1)
    if any(safety[k] for k in ('vertex_collisions','edge_swaps','invalid_edges')) or resources['violations'] or lifecycle or duplicate:
        raise ValueError('counterfactual execution violates safety or lifecycle')
    outcome={'jobs':len(business.jobs),'jobs_completed':len(flows),'status':result['summary']['status'],
        'mean_flow_completed':statistics.mean(flows) if flows else None,
        'restricted_mean_flow':statistics.mean(restricted),
        'capacity_wait_ticks':resources['capacity_wait_ticks'],'safety':safety,
        'events':result['events']}
    if trace_limit:
        future=[row.next_observation for row in rich_rows(captured)
                if row.observation.tick>=tick][:trace_limit]
        evidence={'recovery_triggers':sum(e['triggered'] for e in sim.recovery_log),
                  'recovery_worker_calls':sum(bool(e.get('worker_invoked')) for e in sim.recovery_log),
                  'history_sha256':hashlib.sha256(json.dumps(result['history'],sort_keys=True).encode()).hexdigest(),
                  'complete_factual_prefix':result['history'][:tick+1]==factual['history'][:tick+1]}
        return outcome,future,evidence
    return outcome


def same_state(rows,factual,business,seed,geometry,network,threshold,pibt,lacam,config):
    rich=rich_rows(rows)
    eligible=[i for i,row in enumerate(rich) if row.assignments and len(candidates_for(row.observation))>=2]
    if not eligible:return []
    indices=sorted(set((eligible[0],eligible[len(eligible)//2])))
    records=[]
    definitions=(('old_transition_old_score','rule',False,'legacy'),
                 ('old_transition_new_score','rule',False,'workflow'),
                 ('new_transition_old_score','rule',True,'legacy'),
                 ('new_transition_new_score','rule',True,'workflow'),
                 ('pets_new','pets',True,'workflow'),('structured_new','structured',True,'workflow'))
    for index in indices:
        obs=rich[index].observation;candidates=candidates_for(obs)
        solver=BusinessCPSAT(geometry,business.rules,seed=config['planner_seed'],
             candidate_limit=config['candidate_budget'],time_limit_seconds=config['cp_sat_time_limit_seconds'])
        plans=solver.proposals(obs,candidates);forecasts={}
        for label,model_name,workflow,score in definitions:
            model=make_adapter(model_name,geometry,business.rules,network,threshold,pibt,lacam,config['planner_seed'])
            policy=BusinessPolicy('cpsat_'+model_name,geometry,business.rules,RecoveryObserver(),model,
                seed=config['planner_seed'],horizon=config['horizon'],particles=config['particles'],
                budget=config['candidate_budget'],solver_seconds=config['cp_sat_time_limit_seconds'],
                workflow_prediction=workflow,scoring=score)
            evaluations=policy.evaluate(obs,plans)
            rank=sorted(range(len(plans)),key=lambda i:(evaluations[i]['score'],i))
            forecasts[label]={'choice':rank[0],'ranking':rank,'evaluations':evaluations,
                              'rollout_steps':policy.rollout_steps,'motion_calls':model.pibt.solver_calls}
        branches=[replay_branch(business,seed,rows,factual,obs.tick,plan,pibt,lacam,config) for plan in plans]
        records.append({'tick':obs.tick,'observable_state':asdict(obs),
            'candidate_pairs':[asdict(a) for a in candidates],
            'plans':[[asdict(a) for a in plan] for plan in plans],
            'solver_statuses':solver.log[-1]['statuses'],'solver_fallback':solver.log[-1]['fallback'],
            'forecasts':forecasts,'branches':branches})
    return records


def summarize(cases):
    methods={}
    for name in METHODS:
        values=[case['methods'][name] for case in cases]
        timings=sorted(t for v in values for t in v['decision_wall_ms'])
        completed=[v['mean_flow_completed'] for v in values if v['mean_flow_completed'] is not None]
        stages={}
        for value in values:
            for job in value['job_outcomes']:
                for index,stage in enumerate(job['stages'],1):
                    entry=stages.setdefault(index,{'assigned_waits':[],'released_unassigned':0,'unreleased':0})
                    if stage['assigned_tick'] is not None:
                        entry['assigned_waits'].append(stage['assigned_tick']-stage['release_tick'])
                    elif stage['status']=='unreleased':entry['unreleased']+=1
                    else:entry['released_unassigned']+=1
        stage_waits={str(i):{'assigned_count':len(v['assigned_waits']),
                            'mean_wait_assigned':statistics.mean(v['assigned_waits']) if v['assigned_waits'] else None,
                            'released_unassigned':v['released_unassigned'],'unreleased':v['unreleased']}
                     for i,v in stages.items()}
        methods[name]={'scenarios':len(values),'jobs':sum(v['jobs'] for v in values),
            'jobs_completed':sum(v['jobs_completed'] for v in values),
            'mean_scenario_flow_completed':statistics.mean(completed) if completed else None,
            'mean_scenario_restricted_flow':statistics.mean(v['restricted_mean_flow'] for v in values),
            'mean_stage_wait_per_job':statistics.mean(v['mean_stage_wait_per_job'] for v in values),
            'stage_wait_by_index':stage_waits,
            'capacity_wait_ticks':sum(v['resource']['capacity_wait_ticks'] for v in values),
            'resource_busy_slot_ticks':sum(sum(v['resource']['busy_slot_ticks'].values()) for v in values),
            'station_unavailable_wait_ticks':sum(v['station_unavailable_wait_ticks'] for v in values),
            'resource_violations':sum(len(v['resource']['violations']) for v in values),
            'safety_violations':sum(v['discrete_safety_violations'] for v in values),
            'invalid_edge_violations':sum(v['motion_safety']['invalid_edges'] for v in values),
            'locked_move_violations':sum(v['locked_move_violations'] for v in values),
            'lifecycle_violations':sum(len(v['lifecycle_violations']) for v in values),
            'duplicate_assignments':sum(v['duplicate_assignments'] for v in values),
            'model_fallbacks':sum(v['model_fallbacks'] for v in values),
            'solver_fallbacks':sum(v['solver_fallbacks'] for v in values),
            'rollout_steps':sum(v['rollout_steps'] for v in values),
            'model_prediction_calls':sum(v['model_prediction_calls'] for v in values),
            'model_prediction_total_wall_ms':sum(v['model_prediction_total_wall_ms'] for v in values),
            'model_prediction_mean_wall_ms':(sum(v['model_prediction_total_wall_ms'] for v in values)/
                  sum(v['model_prediction_calls'] for v in values) if sum(v['model_prediction_calls'] for v in values) else None),
            'pets_forward_calls':sum(v['pets_forward_calls'] for v in values),
            'structured_service_calls':sum(v['structured_service_calls'] for v in values),
            'prediction_motion_calls':sum(v['prediction_motion_calls'] for v in values),
            'decision_wall_mean_ms':statistics.mean(timings) if timings else None,
            'decision_wall_p95_ms':quantile(timings,.95) if timings else None}
    paired={}
    for name in METHODS[2:]:
        for ref in ('old_rule','workflow_rule'):
            if name==ref:continue
            diffs=[(case['layout'],case['seed'],case['methods'][name]['restricted_mean_flow']-
                    case['methods'][ref]['restricted_mean_flow']) for case in cases]
            paired[name+'_minus_'+ref]={'clusters':bootstrap_paired(diffs),
                'scenario_wins':sum(v < -1e-9 for _,_,v in diffs),'scenario_ties':sum(abs(v)<=1e-9 for _,_,v in diffs),
                'scenario_losses':sum(v > 1e-9 for _,_,v in diffs)}
    decisions=[d for c in cases for d in c['same_state']]
    contrasts={}
    for a,b in (('old_transition_new_score','old_transition_old_score'),
                ('new_transition_old_score','old_transition_old_score'),
                ('new_transition_new_score','old_transition_old_score'),
                ('new_transition_new_score','new_transition_old_score'),
                ('new_transition_new_score','old_transition_new_score'),
                ('pets_new','new_transition_new_score'),('structured_new','new_transition_new_score')):
        counts=Counter()
        for d in decisions:
            left=d['forecasts'][a];right=d['forecasts'][b]
            counts['ranking_changed']+=left['ranking']!=right['ranking']
            if left['choice']==right['choice']:counts['same_choice']+=1;continue
            counts['different_choice']+=1
            x=d['branches'][left['choice']];y=d['branches'][right['choice']]
            if x['jobs_completed']!=x['jobs'] or y['jobs_completed']!=y['jobs']:
                counts['censored']+=1
            elif x['restricted_mean_flow']<y['restricted_mean_flow']-1e-9:counts['branch_improved']+=1
            elif x['restricted_mean_flow']>y['restricted_mean_flow']+1e-9:counts['branch_worse']+=1
            else:counts['branch_equal']+=1
        contrasts[a+'_vs_'+b]=dict(counts)
    horizon={}
    for h in ('1','3','5'):
        dynamic={}
        for name in MODELS:
            merged=Counter()
            for c in cases:merged.update(c['state_prediction']['horizons'][h]['dynamic'][name])
            dynamic[name]={**dict(merged),
                'joint_accuracy':merged['joint_correct']/merged['windows'] if merged['windows'] else None,
                'service_event_recall':merged['service_true_positive']/merged['service_actual'] if merged['service_actual'] else None,
                'service_event_precision':merged['service_true_positive']/merged['service_predicted'] if merged['service_predicted'] else None,
                'successor_exact_tick_recall':merged['successor_exact_tick']/merged['successor_actual'] if merged['successor_actual'] else None}
        horizon[h]={'dynamic':dynamic,'coverage':dict(sum((Counter(c['state_prediction']['horizons'][h]['dynamic_coverage'])
                                                             for c in cases),Counter())),
                    'fixed_entity':{name:{kind:{'correct':sum(c['state_prediction']['horizons'][h]['fixed_entity'][name][kind]['correct'] for c in cases),
                                                  'total':sum(c['state_prediction']['horizons'][h]['fixed_entity'][name][kind]['total'] for c in cases)}
                                           for kind in ('legacy','workflow')} for name in MODELS}}
    old_release={name:{'missed':sum(len(e['legacy_models'][name]['missing_successor_ids'])
                     for c in cases for e in c['state_prediction']['factual_successor_release_events']),
                    'predecessor_predicted_completed':sum(e['legacy_models'][name]['predecessor_predicted_completed']
                     for c in cases for e in c['state_prediction']['factual_successor_release_events'])} for name in MODELS}
    return {'scenarios':len(cases),'independent_layout_seed_clusters':len({(c['layout'],c['seed']) for c in cases}),
        'methods':methods,'paired':paired,'same_state_decisions':len(decisions),'decision_contrasts':contrasts,
        'actual_successors':sum(len(e['ids']) for c in cases for e in c['state_prediction']['factual_successor_release_events']),
        'old_missing_successors':sum(c['state_prediction']['old_fixed_entity_model_missing_successors'] for c in cases),
        'old_model_successor_audit':old_release,
        'state_prediction':horizon}


def run(profile,output,pibt,lacam,limit=None):
    if output.exists() and any(output.iterdir()):raise FileExistsError('refusing to overwrite previous experiment')
    output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);config=json.loads(CONFIG.read_text());proof=audit()
    network,info=load_pr9_model();base_config=read_config();cases=[]
    settings=[(entry['layout'],entry['seed'],regime) for entry in config[profile] for regime in entry['regimes']]
    if limit is not None:settings=settings[:limit]
    for layout,seed,regime in settings:
        business,assumptions=make_case(base_config,layout,seed,regime)
        geometry=Geometry(business.scenario.grid);values={};factual=rows=None
        for name in METHODS:
            result,transitions,metrics=run_method(business,seed,name,network,info['threshold'],pibt,lacam,config)
            values[name]=metrics
            if name=='nearest':factual,rows=result,transitions
        observed=rich_rows(rows)
        state=evaluate_state_windows(observed,business,geometry,network,info['threshold'],pibt,lacam,config)
        decisions=same_state(rows,factual,business,seed,geometry,network,info['threshold'],pibt,lacam,config)
        cases.append({'layout':layout,'seed':seed,'regime':regime,'assumptions':assumptions,
                      'methods':values,'state_prediction':state,'same_state':decisions,
                      'factual_visible_transitions':[{'observation':asdict(row.observation),
                          'assignments':[asdict(a) for a in row.assignments],
                          'next_observation':asdict(row.next_observation),
                          'next_boundary':row.next_boundary,'termination':row.termination} for row in observed]})
        print(profile,layout,seed,regime,*(f'{name}:{values[name]["jobs_completed"]}/{values[name]["jobs"]}'
                                           for name in METHODS),flush=True)
    data={'profile':profile,'limited_pilot':limit is not None,'config_sha256':hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
          'config_audit':proof,'model_checkpoint':info,'cases':cases,'summary':summarize(cases)}
    (output/'raw.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    (output/'summary.json').write_text(json.dumps(data['summary'],ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(data['summary'],ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=('development','confirmation'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pibt-library',type=Path,required=True)
    parser.add_argument('--lacam-worker',type=Path,required=True)
    parser.add_argument('--limit',type=int)
    a=parser.parse_args();run(a.profile,a.output,a.pibt_library,a.lacam_worker,a.limit)


if __name__=='__main__':main()
