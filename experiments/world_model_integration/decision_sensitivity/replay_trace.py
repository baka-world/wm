"""Frozen PR18 CPU-only *development replay* of the already-seen 3 dispatch states.

This script must be run from the wm repository root with the original V4 binary.
No optimization, checkpoint mutation, simulator oracle input, or new case search.
5/10/20 are diagnostic windows, never rebranded as independent tests.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time

V4_SHA = '60c8a6c3d832da3e6519072688ea099f64ae120ba19d3bd2ec705f7f1a11ac0c'
HEAD = 'f9ba33515406ae077494e0cd9e5faf7c39708d3e'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _snapshot(state):
    return {
        'tick': state.tick,
        'robots': {r.id: {'position': list(r.position), 'task': r.task_id,
                          'phase': r.phase, 'service_elapsed': r.service_elapsed,
                          'service_active_elapsed': r.service_active_elapsed,
                          'service_paused_elapsed': r.service_paused_elapsed}
                   for r in state.robots},
        'samples': {s.id: {'status': s.status, 'robot_id': s.robot_id,
                           'stage_index': getattr(s, 'stage_index', 0),
                           'release_tick': s.release_tick}
                    for s in state.samples},
        'stations': {s.id: {'available': s.available,
                            'device_available': getattr(s, 'device_available', s.available),
                            'resource_occupied': getattr(s, 'resource_occupied', None),
                            'capacity_blocked': getattr(s, 'capacity_blocked', None),
                            'expected_service_ticks': s.expected_service_ticks,
                            'observed_effective_service_mean': getattr(s, 'observed_effective_service_mean', None)}
                     for s in state.stations},
    }


def _transition_events(before, after):
    from business_transport.session_events import eligible_sessions
    from experiments.world_model_integration.build_dataset import label_for
    rows=[]
    for q, reason in eligible_sessions(before, ()):
        label, why = label_for(before, after, q) if reason is None else (None, reason)
        rows.append({'query':q,'eligibility':reason,'finished':label,'reason':why})
    starts=sum(a.service_elapsed is None and (b:=next((b for b in after.robots if b.id==a.id),None)) is not None
               and b.service_elapsed is not None for a in before.robots)
    releases=sorted(set(s.id for s in after.samples)-set(s.id for s in before.samples))
    return {'started':starts,'eligible':sum(r['eligibility'] is None for r in rows),
            'completed':sum(r['finished'] == 1 for r in rows),
            'continued':sum(r['finished'] == 0 for r in rows),
            'released_stages':releases,'labels':rows}


def trace_policy(policy, observation, plans, arm, geometry):
    """Capture original in-call V4 network logits without a duplicate forward pass."""
    import torch
    from business_transport.session_events import eligible_sessions
    from experiments.world_model_integration.build_dataset import label_for
    from lab_transport.torch_structured_adapter import ProgressAwareRule
    trace=[]
    original_outer=policy.model.predict_next
    inner=getattr(policy.model,'model',None)
    original_inner=getattr(inner,'predict_next',None) if arm=='V4' else None
    conditional=[]
    rule=ProgressAwareRule(geometry)
    if arm=='V4':
        original_forward=inner.network.forward
        def intercepted_inner(before,actions):
            qr=eligible_sessions(before,actions)
            viable=[q for q,reason in qr if reason is None]
            capture=[]
            probe=random.Random(); probe.setstate(inner.rng.getstate())
            uniforms=[probe.random() for _ in viable]
            def forward(x,mask,query):
                logits=original_forward(x,mask,query)
                capture.extend(torch.sigmoid(logits.detach()).cpu().tolist())
                return logits
            inner.network.forward=forward
            try:
                after=original_inner(before,actions)
            finally:
                inner.network.forward=original_forward
            if len(capture)!=len(viable):
                raise AssertionError('actual per-session forward did not match eligible query rows')
            rule_after=rule.predict_next(before,actions) if viable else None
            queries=[]
            for q,p,u in zip(viable,capture,uniforms):
                actual,reason=label_for(before,after,q)
                rule_event,rule_reason=label_for(before,rule_after,q)
                queries.append({'query':q,'p_v4':p,'u':u,'sampled_finish':u<p,
                                'v4_transition_finished':actual,'v4_label_reason':reason,
                                'rule_on_identical_v4_input_finished':rule_event,
                                'rule_label_reason':rule_reason,
                                'p_diff_vs_rule_on_same_input':abs(p-float(rule_event)) if rule_event is not None else None,
                                'event_diff_vs_rule_on_same_input':(int(u<p)!=rule_event) if rule_event is not None else None})
            conditional.append({'queries_considered':len(qr),'queries_eligible':len(viable),
                                'fallback_reasons':dict(Counter(reason for q,reason in qr if reason)),
                                'queries':queries})
            return after
        inner.predict_next=intercepted_inner
    def intercepted_outer(before,actions):
        prev=len(conditional)
        after=original_outer(before,actions)
        if arm=='V4' and len(conditional)!=prev+1:
            raise AssertionError('expected one inner V4 inference path for one outer forecast step')
        trace.append({'before':_snapshot(before),'after':_snapshot(after),
                      'actions':[asdict(a) for a in actions],
                      'events':_transition_events(before,after),
                      'session':conditional[-1] if arm=='V4' else None})
        return after
    policy.model.predict_next=intercepted_outer
    started=time.process_time()
    try:
        records=policy.evaluate(observation,plans)
    finally:
        policy.model.predict_next=original_outer
        if arm=='V4':inner.predict_next=original_inner
    n=policy.horizon
    expected=len(plans)*policy.particles*n
    if len(trace)!=expected:raise AssertionError(f'wrong traced steps {len(trace)} != {expected}')
    chunks=[];index=0
    for candidate in range(len(plans)):
        particles=[]
        for particle in range(policy.particles):
            particles.append(trace[index:index+n]);index+=n
        chunks.append(particles)
    return {'arm':arm,'horizon':n,'particles':policy.particles,
            'cpu_seconds':time.process_time()-started,
            'actual_model_forward_batches':inner.service_calls if arm=='V4' else 0,
            'actual_model_query_calls':inner.query_calls if arm=='V4' else 0,
            'fallback_reasons':dict(inner.fallback_reasons) if arm=='V4' else {},
            'records':records,'traces':chunks}


def compare(rule,v4):
    if len(rule['records'])!=len(v4['records']):raise AssertionError('candidate count changed')
    counters=Counter()
    detail=[]
    for i,(ra,va) in enumerate(zip(rule['traces'],v4['traces'])):
        for p,(rp,vp) in enumerate(zip(ra,va)):
            for step,(r,v) in enumerate(zip(rp,vp)):
                d={'candidate':i,'particle':p,'step':step,'tick':r['before']['tick']}
                for part in ('robots','samples','stations'):
                    d[part+'_different']=r['after'][part]!=v['after'][part]
                    counters[part+'_different']+=d[part+'_different']
                d['successor_releases_different']=r['events']['released_stages']!=v['events']['released_stages']
                d['service_completion_count_different']=r['events']['completed']!=v['events']['completed']
                counters['successor_releases_different']+=d['successor_releases_different']
                counters['service_completion_count_different']+=d['service_completion_count_different']
                session=v['session']
                counters['service_queries_considered']+=session['queries_considered']
                counters['service_queries_eligible']+=session['queries_eligible']
                for q in session['queries']:
                    if q['p_diff_vs_rule_on_same_input'] is not None:
                        counters['different_probabilities_on_identical_v4_state']+=(q['p_diff_vs_rule_on_same_input']>1e-9)
                    if q['event_diff_vs_rule_on_same_input'] is not None:
                        counters['sampled_event_diff_vs_rule_on_identical_v4_state']+=q['event_diff_vs_rule_on_same_input']
                detail.append(d)
    for r,v in zip(rule['records'],v4['records']):
        counters['candidate_score_tuple_changed']+=(r['score']!=v['score'])
        for key in r['component_means']:
            if key in v['component_means']:
                counters['component_'+key+'_changed']+=(abs(r['component_means'][key]-v['component_means'][key])>1e-8)
    rank=lambda run:sorted(range(len(run['records'])),key=lambda i:(run['records'][i]['score'],i))
    rrank,vrank=rank(rule),rank(v4)
    counters['ranking_changed']=rrank!=vrank
    counters['action_changed']=rrank[0]!=vrank[0]
    return {'counts':dict(counters),'per_step_differences':detail,'rule_rank':rrank,'v4_rank':vrank}


def run(archive_path,checkpoint,output):
    import torch
    from business_transport.observation import BusinessObserver
    from business_transport.policy import WorkflowPolicy
    from business_transport.session_dynamics import SessionRuleDynamics
    from business_transport.simulation import BusinessSimulator
    from business_transport.dispatch import BusinessCPSAT
    from experiments.business_constraints.run import rich_rows
    from experiments.world_model_integration.build_dataset import case_for
    from experiments.world_model_integration.build_dataset_v4 import CONFIG
    from lab_transport.decision_value import candidates_for
    from lab_transport.torch_structured_adapter import ProgressAwareRule
    torch.set_num_threads(1)
    if sha(checkpoint)!=V4_SHA:raise ValueError('wrong frozen V4 checkpoint')
    archive=json.loads(Path(archive_path).read_text())
    if not archive.get('only_frozen_precommitted_test') or len(archive['cases'])!=3:
        raise ValueError('wrong fixed source archive')
    cfg=json.loads(CONFIG.read_text())
    report={'label':'development_deterministic_cpu_replay_not_independent_confirmation',
            'source_head':HEAD,'checkpoint_sha256':sha(checkpoint),
            'original_dispatch_archive':str(archive_path),'steps':(5,10,20),'cases':[]}
    for case in archive['cases']:
        spec=next(s for s in cfg['groups']['independent_in'] if s['seed']==case['seed'])
        business,geometry=case_for(spec,cfg)
        def make_policy(arm,horizon):
            model=(ProgressAwareRule(geometry) if arm=='Rule' else
                   SessionRuleDynamics(geometry,business.rules,checkpoint,experimental=True,
                                       seed=1810,feature_version='v4'))
            return WorkflowPolicy('cpsat_rule' if arm=='Rule' else 'cpsat_session',geometry,business.rules,
                                  BusinessObserver(),model,seed=1810,horizon=horizon,
                                  particles=3,budget=4,solver_seconds=.5,model_aware=True,risk_weight=0.)
        events=[]
        actual=make_policy('Rule',5)
        BusinessSimulator(business,'world',case['seed'],collect_metrics=True).run(
            dispatch_policy=actual,on_observation=actual.observer.observe,on_transition=events.append)
        states={row.observation.tick:row.observation for row in rich_rows(events)}
        entry={'seed':case['seed'],'windows':{}}
        for previous in case['fixed_states']:
            tick=previous['tick']
            if tick not in states:raise AssertionError('frozen observation not recovered')
            observation=states[tick]
            plans=BusinessCPSAT(geometry,business.rules,candidate_limit=4,
                                time_limit_seconds=.5,seed=1810).proposals(observation,candidates_for(observation))
            normalized=[[asdict(a) for a in plan] for plan in plans]
            if normalized!=previous['candidates']:
                raise AssertionError(f'fixed candidate set changed for seed={case["seed"]},tick={tick}')
            for horizon in (5,10,20):
                records={arm:trace_policy(make_policy(arm,horizon),observation,plans,arm,geometry)
                         for arm in ('Rule','V4')}
                parity={}
                if horizon==5:
                    for arm in ('Rule','V4'):
                        saved=previous['predictions'][arm]['score']
                        got=[list(r['score']) for r in records[arm]['records']]
                        parity[arm]=len(saved)==len(got) and all(
                            len(a)==len(b) and all(abs(x-y)<1e-6 for x,y in zip(a,b))
                            for a,b in zip(saved,got))
                    if not all(parity.values()):
                        raise AssertionError('5-step replay failed to reproduce original scores')
                entry['windows'][str(horizon)]={'recorded_archive_5step_parity':parity,
                        'comparison':compare(records['Rule'],records['V4']),
                        'runs':records}
        report['cases'].append(entry)
    p=Path(output);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise FileExistsError('never overwrite development replay result')
    p.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps([{'seed':case['seed'],'summary':{
        h:data['comparison']['counts'] for h,data in case['windows'].items()}}
        for case in report['cases']],ensure_ascii=False))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dispatch',type=Path,required=True)
    p.add_argument('--v4',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.dispatch,a.v4,a.output)


if __name__=='__main__':main()
