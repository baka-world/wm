"""Strictly limited, gate-controlled factual scheduling test for chosen v3.

No model retraining or choice based on the new results. If the independent
probabilities do not beat frozen simple baselines, only one legal rollout is
run and business-value claims are blocked.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import time
import torch

from business_transport.observation import BusinessObserver
from business_transport.policy import WorkflowPolicy
from business_transport.session_dynamics import SessionRuleDynamics
from business_transport.workflow_transition import check_business_transition
from business_transport.simulation import BusinessSimulator
from business_transport.dispatch import BusinessCPSAT
from experiments.business_constraints.run import summarize_result,rich_rows,replay_branch
from experiments.world_model_integration.build_dataset import case_for
from experiments.world_model_integration.v3_independent import CONFIG
from lab_transport.decision_value import candidates_for
from lab_transport.torch_structured_adapter import ProgressAwareRule

CFG={'seed':1810,'horizon':5,'particles':3,'candidate_budget':4,
     'solver_seconds':0.5,'risk_weight':0.,'sample_stride':13,
     'max_states_per_group':2}

def make_policy(name,business,geometry,v2,v3):
    if name=='Rule':model=ProgressAwareRule(geometry)
    elif name=='V2':model=SessionRuleDynamics(geometry,business.rules,v2,
                                               experimental=True,seed=CFG['seed'])
    elif name=='V3':model=SessionRuleDynamics(geometry,business.rules,v3,
                                               experimental=True,seed=CFG['seed'],
                                               feature_version='v3')
    else:raise ValueError('unrecognized frozen policy')
    method='cpsat_rule' if name=='Rule' else 'cpsat_session'
    return WorkflowPolicy(method,geometry,business.rules,BusinessObserver(),model,
        seed=CFG['seed'],horizon=CFG['horizon'],particles=CFG['particles'],
        budget=CFG['candidate_budget'],solver_seconds=CFG['solver_seconds'],
        model_aware=True,risk_weight=0)

def run(holdout_path,v2_path,v3_path,output):
    output=Path(output)
    if output.exists():raise FileExistsError('v3 decision evidence cannot overwrite')
    hold=json.loads(Path(holdout_path).read_text())
    config=json.loads(CONFIG.read_text())
    if hold['protocol']!=config['protocol'] or hold.get('no_tuning_on_holdout') is not True:
        raise ValueError('unverified independent decision evidence')
    if hold['selected_checkpoint_sha256']!=__import__('hashlib').sha256(Path(v3_path).read_bytes()).hexdigest():
        raise ValueError('checkpoint differs from one-shot independent evaluation')
    # Precommitted basic reliability gate: must beat both global and conditional
    # empirical baseline on entirely untouched new scenario families.
    gate=hold['independent_gate']['v3_brier_below_B0'] and hold['independent_gate']['v3_brier_below_B2']
    torch.set_num_threads(1)
    result={'gate_pass':bool(gate),'config':CFG,
         'comparison':['Rule','V2','V3'],'cases':[],
         'note':'only research experimental policy, never production',
         'no_hidden_future_used':True}
    if not gate:
        from experiments.business_constraints.tests.test_business import fixture
        from experiments.business_constraints.run import rich_rows
        from lab_transport.decision_value import nearest_batch
        from lab_transport.dynamics_model import Geometry
        base,cfg,business=fixture(1);geo=Geometry(base.grid)
        raws=[]
        BusinessSimulator(business,'world',1).run(
            on_transition=raws.append,
            dispatch_policy=lambda o,c:nearest_batch(o,c,geo))
        obs=rich_rows(raws)[0].observation
        v3_model=make_policy('V3',business,geo,v2_path,v3_path).model
        action=nearest_batch(obs,candidates_for(obs),geo)
        pred=v3_model.predict_next(obs,action)
        check_business_transition(obs,pred,geo,business.rules,action)
        result['stop_reason']='independent_brier_not_better_than_B0_and_B2'
        result['smoke']={'steps':1,'valid':True,'model_online_verified':False}
    else:
        for spec in config['groups']:
            business,geo=case_for(spec,config)
            group={'seed':spec['seed'],'layout':spec['layout'],'arms':{},'fixed':[]}
            reference_raw=None;reference_result=None
            for arm in ('Rule','V2','V3'):
                policy=make_policy(arm,business,geo,v2_path,v3_path)
                raw=[]
                start=time.perf_counter()
                actual=BusinessSimulator(business,'world',spec['seed'],
                       collect_metrics=True).run(dispatch_policy=policy,
                       on_observation=policy.observer.observe,
                       on_transition=raw.append if arm=='Rule' else None)
                outcome=summarize_result(actual,business,policy)
                group['arms'][arm]={
                    'completed':outcome['jobs_completed'],
                    'restricted_mean_flow':outcome['restricted_mean_flow'],
                    'mean_wait':outcome['mean_stage_wait_per_job'],
                    'resource_wait_ticks':outcome['resource']['capacity_wait_ticks'],
                    'mean_decision_ms':outcome['decision_wall_mean_ms'],
                    'p95_decision_ms':outcome['decision_wall_p95_ms'],
                    'model_calls':policy.prediction_calls,
                    'model_fallbacks':policy.fallbacks,
                    'capacity_violations':len(outcome['resource']['violations']),
                    'lifecycle_violations':len(outcome['lifecycle_violations']),
                    'motion_violations':outcome['discrete_safety_violations'],
                    'wall_seconds':time.perf_counter()-start}
                if arm=='Rule':
                    reference_raw,reference_result=raw,actual
            factual=rich_rows(reference_raw)
            candidate_states=[r.observation for r in factual if len(candidates_for(r.observation))>=2]
            selected=candidate_states[::CFG['sample_stride']][:CFG['max_states_per_group']]
            for obs in selected:
                plans=BusinessCPSAT(geo,business.rules,
                   candidate_limit=CFG['candidate_budget'],
                   time_limit_seconds=CFG['solver_seconds'],
                   seed=CFG['seed']).proposals(obs,candidates_for(obs))
                if len(plans)<2:continue
                actual_branches=[replay_branch(business,spec['seed'],reference_result,
                               reference_raw,obs.tick,p) for p in plans]
                scores={}
                for arm in ('Rule','V2','V3'):
                    policy=make_policy(arm,business,geo,v2_path,v3_path)
                    estimates=policy.evaluate(obs,plans)
                    ranks=sorted(range(len(plans)),key=lambda i:(estimates[i]['score'],i))
                    best_truth=min((r['flow'] for r in actual_branches
                                    if r['flow'] is not None),default=None)
                    choice=ranks[0];selected_truth=actual_branches[choice]['flow']
                    scores[arm]={'chosen':choice,'ranks':ranks,
                       'scores':[r['score'] for r in estimates],
                       'true_regret':selected_truth-best_truth
                           if selected_truth is not None and best_truth is not None else None}
                group['fixed'].append({'tick':obs.tick,'plans':[[asdict(a) for a in p] for p in plans],
                    'truth':[r['flow'] for r in actual_branches],'comparison':scores,
                    'model_changes_cost_but_not_choice':scores['V3']['scores']!=scores['Rule']['scores']
                          and scores['V3']['chosen']==scores['Rule']['chosen']})
            result['cases'].append(group)
        fixed=[x for g in result['cases'] for x in g['fixed']]
        result['fixed_comparison']={'count':len(fixed),
          'v3_action_changes':sum(s['comparison']['V3']['chosen']!=
                                  s['comparison']['Rule']['chosen'] for s in fixed),
          'v3_cost_changes_same_action':sum(s['model_changes_cost_but_not_choice'] for s in fixed)}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'gate':gate,'stop_reason':result.get('stop_reason'),
       'fixed':result.get('fixed_comparison'),
       'cases':len(result['cases'])},ensure_ascii=False))
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--holdout',type=Path,required=True)
    p.add_argument('--frozen-v2',type=Path,required=True)
    p.add_argument('--selected',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.holdout,a.frozen_v2,a.selected,a.output)
if __name__=='__main__':main()
