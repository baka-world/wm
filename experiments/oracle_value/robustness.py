"""Predesignated post-decision keyed-service robustness, separate from oracle selection.

Run ONLY after the initial fixed confirmation. Neither candidate generation nor
Rule/PETS/Structured predictions are changed. Factual prefixes and all non-service
exogenous events remain original; each future-service seed gives the SAME potential
service duration to each candidate for a given task/phase/station.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

from business_transport.lacam3_integration import RecoveryObserver
from experiments.business_constraints.benchmark import make_case, read_config
from experiments.service_value.counterfactual import load_pr9_model
from experiments.workflow_prediction.evaluate import replay_branch, run_method,rich_rows
from lab_transport.dynamics_model import Geometry
from lab_transport.transition_schema import Assignment
from .analysis import select_oracle
from .evaluate import CONFIG, settings


def run(raw_path,out,pibt,lacam):
    root=Path(out)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError('refusing to overwrite robustness outputs')
    root.mkdir(parents=True,exist_ok=True)
    with gzip.open(raw_path,'rt',encoding='utf-8') as h:
        first=json.load(h)
    if first['profile']!='confirmation' or first['limited_pilot']:
        raise ValueError('robustness requires the full frozen confirmation run')
    cfg=json.loads(CONFIG.read_text(encoding='utf-8'))
    if hashlib.sha256(CONFIG.read_bytes()).hexdigest()!=first['frozen_config_sha256']:
        raise ValueError('frozen config changed since initial confirmation')
    control=cfg['robust_replays_if_large_gap']
    threshold=control['threshold_steps']
    extra_seeds=control['additional_simulation_seeds']
    if len(set(extra_seeds))!=len(extra_seeds) or not extra_seeds:
        raise ValueError('additional service seeds must be distinct and nonempty')
    torch.set_num_threads(1)
    net,info=load_pr9_model()
    benchmark=read_config()
    selected=set(settings(cfg,'confirmation'))
    cases=first['cases']
    assert len(cases)==24 and len(selected)==24
    started=time.perf_counter()
    output=[]
    failed=[]
    for case in cases:
        layout,seed,regime=case['layout'],case['seed'],case['regime']
        assert (layout,seed,regime) in selected
        qualifying=[]
        for d in case['decisions']:
            if len(d['branches'])<2 or any('error' in b for b in d['branches']):
                failed.append({'case':[layout,seed,regime],'tick':d['tick'],
                               'error':'nominal candidate set incomplete'})
                continue
            if any(b['jobs_completed']!=b['jobs'] for b in d['branches']):
                continue
            flows=[b['restricted_mean_flow'] for b in d['branches']]
            if max(flows)-min(flows)>=threshold-1e-9:
                qualifying.append(d)
        if not qualifying:
            continue
        business,_=make_case(benchmark,layout,seed,regime)
        geometry=Geometry(business.scenario.grid)
        factual,rows,_=run_method(business,seed,'nearest',net,info['threshold'],
                                  pibt,lacam,cfg)
        if hashlib.sha256(json.dumps(factual['history'],sort_keys=True,ensure_ascii=False).encode()).hexdigest()!=case['factual_history_sha256']:
            # The original hash comes from the exact same helper; changing the
            # serialization here would accidentally reject correct reproductions.
            raise ValueError('factual history diverged before robustness')
        rich={r.observation.tick:r for r in rich_rows(rows)}
        for d in qualifying:
            obs=rich[d['tick']].observation
            if json.loads(json.dumps(asdict(obs),ensure_ascii=False))!=d['observable_state']:
                raise ValueError('decision observation differs from primary run')
            plans=[tuple(Assignment(**p) for p in plan) for plan in d['plans']]
            series=[];all_noise=[]
            for alt_seed in extra_seeds:
                branches=[]
                for plan in plans:
                    began=time.perf_counter()
                    try:
                        b=replay_branch(business,seed,rows,factual,d['tick'],
                                        plan,pibt,lacam,cfg,future_service_seed=alt_seed)
                        branches.append({'jobs_completed':b['jobs_completed'],
                                         'jobs':b['jobs'],
                                         'restricted_mean_flow':b['restricted_mean_flow'],
                                         'capacity_wait_ticks':b['capacity_wait_ticks'],
                                         'event_count':len(b['events']),
                                         'wall_seconds':time.perf_counter()-began})
                    except Exception as e:
                        branches.append({'error':type(e).__name__+': '+str(e)})
                series.append({'future_service_seed':alt_seed,'branches':branches})
                if any('error' in b for b in branches):
                    failed.append({'case':[layout,seed,regime],'tick':d['tick'],
                                   'seed':alt_seed,'error':'candidate replay failed'})
            # Independently replay a different-noise candidate for deterministic reproducibility.
            if series and not any('error' in b for b in series[0]['branches']):
                repeat=replay_branch(business,seed,rows,factual,d['tick'],
                                     plans[0],pibt,lacam,cfg,
                                     future_service_seed=extra_seeds[0])
                if (repeat['jobs_completed']!=series[0]['branches'][0]['jobs_completed'] or
                    abs(repeat['restricted_mean_flow']-
                        series[0]['branches'][0]['restricted_mean_flow'])>1e-9):
                    failed.append({'case':[layout,seed,regime],'tick':d['tick'],
                                   'error':'same seed/plan not reproducible'})
            if any('error' in b for s in series for b in s['branches']):
                outcome={'error':'incomplete robustness branches'}
            else:
                average=[statistics.mean(s['branches'][i]['restricted_mean_flow']
                          for s in series) for i in range(len(plans))]
                complete=[all(s['branches'][i]['jobs_completed']==s['branches'][i]['jobs']
                              for s in series) for i in range(len(plans))]
                if not all(complete):
                    outcome={'error':'robustness has unfinished jobs; no scalar regret'}
                else:
                    best=min(range(len(average)),key=lambda i:(average[i],i))
                    original=select_oracle(d['branches'])
                    rule=d['methods']['Rule-5']['choice']
                    outcome={'mean_flows_by_plan':average,
                             'mean_seed_optimal_plan':best,
                             'nominal_single_seed_oracle_plan':original,
                             'rule_plan':rule,
                             'nominal_best_still_robust_best':average[original]<=min(average)+1e-9,
                             'robust_rule_opportunity':average[rule]-min(average),
                             'robust_nominal_oracle_gain_vs_rule':average[rule]-average[original],
                             'robust_flow_spread':max(average)-min(average)}
            output.append({'layout':layout,'seed':seed,'regime':regime,'tick':d['tick'],
                           'nominal_flow_spread':max(b['restricted_mean_flow'] for b in d['branches'])-
                                                 min(b['restricted_mean_flow'] for b in d['branches']),
                           'plans':d['plans'],'random_seeds':series,'result':outcome})
            print('robust',layout,seed,regime,d['tick'],
                  'plans',len(plans),'seeds',len(extra_seeds),flush=True)
    valid=[d for d in output if 'error' not in d['result']]
    by_cluster={}
    for d in valid:
        by_cluster.setdefault((d['layout'],d['seed']),[]).append(d['result'])
    cluster_means=[{'layout':k[0],'seed':k[1],
                    'mean_robust_rule_opportunity':statistics.mean(
                        r['robust_rule_opportunity'] for r in rows),
                    'n_decisions':len(rows)} for k,rows in sorted(by_cluster.items())]
    summary={'scope':'PREDECLARED gap>=3 only; three extra future service noise seeds with identical factual prefix',
             'confirmation_decisions':sum(len(c['decisions']) for c in cases),
             'qualifying_decisions':len(output),
             'valid_robust_decisions':len(valid),
             'failed_replays':len(failed),
             'extra_noise_seeds':extra_seeds,
             'candidate_replays':sum(len(s['branches']) for d in output for s in d['random_seeds']),
             'nominal_oracle_remains_mean_best':sum(d['result']['nominal_best_still_robust_best']
                                                   for d in valid),
             'mean_robust_rule_opportunity_qualifying_states':
                statistics.mean(d['result']['robust_rule_opportunity'] for d in valid) if valid else None,
             'mean_robust_nominal_oracle_gain_qualifying_states':
                statistics.mean(d['result']['robust_nominal_oracle_gain_vs_rule']
                                for d in valid) if valid else None,
             'cluster_means':cluster_means,
             'mean_of_qualifying_cluster_means':
                statistics.mean(c['mean_robust_rule_opportunity'] for c in cluster_means)
                if cluster_means else None,
             'wall_seconds':time.perf_counter()-started,
             'limitation':'post-decision perturbation excludes effects of altered past service draws; hindsight averages still not online-deployable, only gap>=3 selected; not comparable to all-48 main mean'
            }
    (root/'raw.json').write_text(json.dumps({'config':control,'states':output,'failures':failed},
                                            ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',
                                      encoding='utf-8')
    if failed:
        raise RuntimeError('robustness fairness/legality failures: '+str(len(failed)))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--raw',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pibt-library',type=Path,required=True)
    parser.add_argument('--lacam-worker',type=Path,required=True)
    a=parser.parse_args()
    print(json.dumps(run(a.raw,a.output,a.pibt_library,a.lacam_worker),
                     ensure_ascii=False,indent=2))
