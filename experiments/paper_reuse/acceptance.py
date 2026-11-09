"""Small CPU end-to-end acceptance on pre-frozen new synthetic scenarios."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import torch

from experiments.paper_reuse.config import REGIMES, SEEDS, audit
from experiments.service_value.counterfactual import load_pr9_model, rich_transitions
from lab_transport.decision_value import nearest_batch
from lab_transport.decision_value_round2 import make_round2_cases
from lab_transport.dynamics_experiment import compare, eligible_windows
from lab_transport.dynamics_model import Geometry
from lab_transport.multisite_experiment import discrete_safety_violations
from lab_transport.paper_reuse.contracts import check_transition
from lab_transport.paper_reuse.pets_service import PETSRuleDynamics, ServiceDistribution
from lab_transport.paper_reuse.rollout_dispatch import NearestRuleDispatch, RolloutDispatch
from lab_transport.paper_reuse.service_dataset import extract_episode, service_features
from lab_transport.service_dynamics import RuleMotionNeuralService, ThresholdDynamics
from lab_transport.service_fault_data import source_bases
from lab_transport.service_learning import service_keys
from lab_transport.simulation import Simulator
from lab_transport.state_v2 import ObservableHistory
from lab_transport.torch_structured_adapter import ProgressAwareRule, collate, encode_transition

METHODS=('nearest_rule','cpsat_rule','cpsat_pets')
MODEL_NAMES=('rule','structured','structured_rule_motion','pets')


def quantile_duration(params,p):
    lo,hi=0.,1.
    while 1-ServiceDistribution.survival(params,hi)<p and hi<1e6:
        hi*=2
    for _ in range(50):
        mid=(lo+hi)/2
        if 1-ServiceDistribution.survival(params,mid)<p:lo=mid
        else:hi=mid
    return (lo+hi)/2


def service_metrics(rows,geometry,distribution,network):
    """Factual sessions and stepwise labels; errors aggregated per scenario."""
    episodes=type('FactualEpisode',(),{'rows':rows})()
    examples,censored=extract_episode(episodes)
    errors=[]; rule_errors=[]; nll=[]; covered=[]
    for session in examples:
        params=distribution.parameters(session['features'])
        actual=session['duration_active_ticks']
        errors.append(abs(distribution.mean(params)-actual))
        rule_errors.append(abs(10*session['features'][1]-actual))
        nll.append(distribution.nll(params,actual))
        covered.append(quantile_duration(params,.05)<=actual<=quantile_duration(params,.95))
    squared=defaultdict(list)
    with torch.inference_mode():
        for row in rows:
            keys=service_keys(row)
            if not keys:continue
            items=[v for v in encode_transition(geometry,row.observation,row.assignments,row.next_observation)
                   if v['kind']=='service']
            if len(items)!=len(keys):raise ValueError('service item ordering mismatch')
            inputs,labels=collate(items)
            scores=torch.sigmoid(network(inputs)['service']).cpu().tolist()
            robots={r.id:r for r in row.observation.robots}
            samples={s.id:s for s in row.observation.samples}
            stations={s.id:s for s in row.observation.stations}
            for i,(rid,sid,phase,station_id) in enumerate(keys):
                robot,sample,station=robots[rid],samples[sid],stations[station_id]
                progress=robot.service_active_elapsed or 0
                params=distribution.parameters(service_features(row.observation,robot,sample,phase,station))
                p=distribution.finish_probability(params,progress)
                expected=(station.observed_effective_service_mean if station.observed_effective_service_mean is not None
                          else station.expected_service_ticks)
                rule=float(progress+1>=math.ceil(expected))
                actual=float(labels['service'][i])
                for name,prob in [('rule',rule),('structured',scores[i]),('pets',p)]:
                    squared[name].append((prob-actual)**2)
    mean=lambda xs:statistics.mean(xs) if xs else None
    return {'completed_sessions':len(examples),'censoring':censored,
            'duration_mae':{'rule':mean(rule_errors),'pets':mean(errors),
                            'structured':None},
            'duration_nll':{'pets':mean(nll),'rule':None,'structured':None},
            'interval90_coverage':{'pets':mean(covered),'rule':None,'structured':None},
            'brier':{name:mean(values) for name,values in squared.items()},
            'event_opportunities':{name:len(values) for name,values in squared.items()}}


def state_metrics(rows,geometry,distribution,network,threshold,seed):
    result={}
    for horizon in (1,3,5):
        windows,coverage=eligible_windows(rows,horizon)
        windows=windows[::5]
        totals={name:{'correct':0,'windows':0} for name in MODEL_NAMES}
        for wi,window in enumerate(windows):
            initial,actual=window[0].observation,window[-1].next_observation
            models={'rule':ProgressAwareRule(geometry),
                'structured':ThresholdDynamics(geometry,network,torch.device('cpu'),threshold),
                'structured_rule_motion':RuleMotionNeuralService(geometry,network,torch.device('cpu'),threshold),
                'pets':PETSRuleDynamics(geometry,distribution,seed=seed+horizon*10000+wi)}
            for name,model in models.items():
                state=initial
                for index,row in enumerate(window):
                    nxt=model.predict_next(state,row.assignments if index==0 else ())
                    check_transition(state,nxt,geometry)
                    state=nxt
                counts=compare(initial,actual,state)
                totals[name]['correct']+=counts['joint_correct']
                totals[name]['windows']+=counts['joint_total']
        result[str(horizon)]={'coverage':coverage,'selected_every_nth':5,'models':{
            name:{**v,'joint_accuracy':v['correct']/v['windows'] if v['windows'] else None}
            for name,v in totals.items()}}
    return result


def run_policy(case,seed,pauses,name,distribution):
    geometry=Geometry(case.grid)
    observer=ObservableHistory()
    if name=='nearest_rule':
        policy=NearestRuleDispatch(geometry,ProgressAwareRule(geometry),observer,horizon=5)
    else:
        dynamics=(ProgressAwareRule(geometry) if name=='cpsat_rule' else
                  PETSRuleDynamics(geometry,distribution,seed=seed))
        policy=RolloutDispatch(geometry,dynamics,observer,horizon=5,particles=3,seed=seed,
                               solver_seconds=.5,proposals=4)
    raw=[]
    result=Simulator(case,'world',seed,motion_pauses=pauses,collect_metrics=True).run(
        on_transition=raw.append if name=='nearest_rule' else None,
        dispatch_policy=policy,on_observation=observer.observe)
    summary=result['summary']
    flows=[x['completed_tick']-x['release_tick'] for x in result['samples'] if x['completed_tick'] is not None]
    action_events=[e for e in result['events'] if e['type']=='assigned']
    pairs=[(e['tick'],e['robot_id']) for e in action_events]
    task_ids=[e['sample_id'] for e in action_events]
    duplicates=len(pairs)-len(set(pairs)) + len(task_ids)-len(set(task_ids))
    elapsed=[x['cpu_seconds'] for x in policy.log]
    solver=getattr(policy,'solver',None)
    metrics={'status':summary['status'],'completed':summary['completed_count'],'samples':summary['sample_count'],
        'deadline_defined':summary['deadline_sample_count'],'late_count':summary['late_count'],
        'flow_mean_completed':statistics.mean(flows) if flows else None,
        'decision_count':len(elapsed),'decision_cpu_mean_ms':statistics.mean(elapsed)*1000 if elapsed else None,
        'decision_cpu_total_ms':sum(elapsed)*1000,'rollout_steps':policy.rollout_steps,
        'neural_model_forward_calls':policy.model_calls,
        'neural_transition_calls':sum(x['neural_transition_calls'] for x in policy.log),
        'neural_scoring_calls':sum(x['neural_scoring_calls'] for x in policy.log),
        'sampled_completion_events':sum(x['sampled_completion_events'] for x in policy.log),
        'model_forward_calls_per_decision':policy.model_calls/len(elapsed) if elapsed else 0,
        'prediction_fallbacks':policy.fallbacks,'solver_fallbacks':solver.fallbacks if solver else 0,
        'solver_partial_truncations':solver.truncated if solver else 0,
        'solver_calls':solver.calls if solver else 0,'duplicate_assignments':duplicates,
        'discrete_safety_violations':discrete_safety_violations(result),
        'station_unavailable_wait_ticks':summary.get('station_unavailable_wait_ticks',0),
        'sample_outcomes':[{k:s[k] for k in ('id','release_tick','deadline_tick','assigned_tick','picked_tick','completed_tick')}
                           for s in result['samples']],
        'decisions':policy.log, 'solver_log':solver.log if solver else []}
    return metrics,rich_transitions(raw) if raw else None


def summarize(cases):
    result={}
    for method in METHODS:
        rows=[c['methods'][method] for c in cases]
        result[method]={'cases':len(rows),'completed':sum(r['completed'] for r in rows),
          'tasks':sum(r['samples'] for r in rows),'cases_completed':sum(r['status']=='completed' for r in rows),
          'flow_mean':statistics.mean(r['flow_mean_completed'] for r in rows if r['flow_mean_completed'] is not None),
          'deadline_defined':sum(r['deadline_defined'] or 0 for r in rows),
          'late_count':sum(r['late_count'] or 0 for r in rows) if any(r['deadline_defined'] for r in rows) else None,
          'decision_cpu_mean_ms':sum(r['decision_cpu_total_ms'] for r in rows)/max(1,sum(r['decision_count'] for r in rows)),
          'neural_model_forward_calls':sum(r['neural_model_forward_calls'] for r in rows),
          'neural_transition_calls':sum(r['neural_transition_calls'] for r in rows),
          'neural_scoring_calls':sum(r['neural_scoring_calls'] for r in rows),
          'sampled_completion_events':sum(r['sampled_completion_events'] for r in rows),
          'model_calls_per_decision':sum(r['neural_model_forward_calls'] for r in rows)/max(1,sum(r['decision_count'] for r in rows)),
          'rollout_steps':sum(r['rollout_steps'] for r in rows),'prediction_fallbacks':sum(r['prediction_fallbacks'] for r in rows),
          'solver_fallbacks':sum(r['solver_fallbacks'] for r in rows),
          'solver_partial_truncations':sum(r['solver_partial_truncations'] for r in rows),
          'duplicate_assignments':sum(r['duplicate_assignments'] for r in rows),
          'discrete_safety_violations':sum(r['discrete_safety_violations'] for r in rows)}
    return result


def evaluate(output, checkpoint, profile, limit=None):
    if output.exists() and any(output.iterdir()):raise FileExistsError(f'refuse overwrite {output}')
    output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    bases,source_hashes=source_bases()
    proof=audit(bases)
    network,info=load_pr9_model()
    distribution=ServiceDistribution.load(checkpoint)
    cases=[]
    for layout,seeds in SEEDS[profile].items():
        for seed in seeds:
            options,_=make_round2_cases(bases[layout],layout,seed)
            for regime in REGIMES:
                if limit is not None and len(cases)>=limit:break
                case,pauses,incident=options[regime]
                methods={}; factual=None
                for method in METHODS:
                    methods[method],rows=run_policy(case,seed,pauses,method,distribution)
                    if rows is not None:factual=rows
                if factual is None:raise ValueError('missing factual trajectory')
                if methods['cpsat_pets']['neural_model_forward_calls']<=0:
                    raise ValueError('PETS did not participate in rollout')
                geo=Geometry(case.grid)
                service=service_metrics(factual,geo,distribution,network)
                states=state_metrics(factual,geo,distribution,network,info['threshold'],seed)
                cases.append({'layout':layout,'seed':seed,'regime':regime,'incident':incident,
                              'methods':methods,'service_prediction':service,'state_prediction':states})
                print(f'{layout} seed={seed} {regime} completed',flush=True)
    result={'scope':'synthetic CPU acceptance; factual labels only; no clinical performance claim',
      'profile':profile,'seeds':SEEDS[profile],'regimes':REGIMES,'schedule_audit':proof,'source_sha256':source_hashes,
      'pets_checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
      'pr9_checkpoint':info,'cases':cases,'summary':summarize(cases)}
    (output/'raw.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (output/'summary.json').write_text(json.dumps(result['summary'],ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result['summary'],ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--profile',choices=tuple(SEEDS),required=True)
    p.add_argument('--checkpoint',type=Path,default=Path('examples/evidence/paper-reuse-integration-20261109/pets-service-cpu.pt'))
    p.add_argument('--limit',type=int,help='bounded development smoke only; omit for full acceptance')
    args=p.parse_args()
    evaluate(args.output,args.checkpoint,args.profile,args.limit)


if __name__=='__main__':main()
