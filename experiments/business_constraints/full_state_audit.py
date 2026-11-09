"""Post-confirmation diagnostic: audit every decision in four factual policies.

This is an exploratory follow-up, not a new independent test and not an online
policy. Each differing candidate is separately replayed after an identical
factual prefix; no hidden state is used to rank candidates.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import statistics

import torch

from business_transport.dispatch import BusinessCPSAT
from business_transport.observation import BusinessObserver
from business_transport.policy import BusinessPolicy
from experiments.business_constraints.benchmark import make_case,read_config
from experiments.business_constraints.run import make_model,replay_branch,rich_rows,run_policy
from experiments.service_value.counterfactual import load_pr9_model
from lab_transport.decision_value import batch_candidates,candidates_for
from lab_transport.dynamics_model import Geometry

ORIGINS=('nearest','cpsat_rule','cpsat_pets','cpsat_structured')
MODELS=('rule','pets','structured')


def evaluate(reference,output):
    if output.exists():raise FileExistsError('refuse to overwrite diagnostic')
    output.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    ref=json.loads(reference.read_text(encoding='utf-8'))
    if ref['profile']!='confirmation':raise ValueError('diagnostic expects frozen confirmation data')
    config=read_config();network,info=load_pr9_model();records=[];scope=[]
    for case in ref['cases']:
        layout,seed,regime=case['layout'],case['seed'],case['regime']
        business,_=make_case(config,layout,seed,regime);geo=Geometry(business.scenario.grid)
        if case['config']['jobs']!=make_case(config,layout,seed,regime)[1]['jobs']:
            raise ValueError('scenario task configuration changed')
        for origin in ORIGINS:
            factual,rows,metrics=run_policy(business,seed,origin,network,info['threshold'],capture=True,
                                           policy_seed=config['planner_seed'])
            expected=case['methods'][origin]
            chosen=lambda m:[(x['tick'],x['selected']) for x in m['decisions']]
            if (metrics['job_outcomes']!=expected['job_outcomes'] or chosen(metrics)!=chosen(expected) or
                    metrics['resource']!=expected['resource']):
                raise ValueError(f'factual episode did not reproduce frozen result: {layout} {seed} {regime} {origin}')
            enriched=rich_rows(rows);count=0
            for index,row in enumerate(rows):
                obs=enriched[index].observation;current=candidates_for(obs)
                if not row.assignments or len(current)<2:continue
                solver=BusinessCPSAT(geo,business.rules,seed=config['planner_seed'],candidate_limit=4,time_limit_seconds=.5)
                plans=solver.proposals(obs,current)
                if solver.truncated or solver.fallbacks:
                    raise ValueError('same-state candidate solver truncated or fell back')
                near,_=batch_candidates(obs,current,geo,5);near=tuple(p for p in near if p)
                predictions={}
                for name in MODELS:
                    policy=BusinessPolicy('cpsat_'+name,geo,business.rules,BusinessObserver(),
                        make_model(name,geo,network,info['threshold'],config['planner_seed']),seed=config['planner_seed'])
                    scored=policy.evaluate(obs,plans)
                    choice=min(range(len(plans)),key=lambda i:(scored[i]['score'],i))
                    predictions[name]={'choice':choice,'scores':[r['score'] for r in scored]}
                choices={x['choice'] for x in predictions.values()}
                # Branches are necessary only when model choice differs; all
                # candidate scores and choices remain recorded for all states.
                branches={}
                if len(choices)>1:
                    for chosen_index in sorted(choices):
                        branches[str(chosen_index)]=replay_branch(business,seed,factual,rows,obs.tick,plans[chosen_index])
                normalize=lambda p:tuple(sorted((a.robot_id,a.sample_id) for a in p))
                left={normalize(p) for p in near};right={normalize(p) for p in plans}
                records.append({'layout':layout,'seed':seed,'regime':regime,'origin':origin,'tick':obs.tick,
                   'observed_state':asdict(obs),'candidates':[[asdict(a) for a in p] for p in plans],
                   'candidate_intersection':len(left&right),'candidate_union':len(left|right),
                   'solver_statuses':solver.log[-1]['statuses'],'forecasts':predictions,'branches':branches})
                count+=1
            scope.append({'layout':layout,'seed':seed,'regime':regime,'origin':origin,'decision_states':count})
            print(layout,seed,regime,origin,'states',count,flush=True)
    groups=defaultdict(list)
    for record in records:
        key=(record['layout'],record['seed'],record['regime'],record['tick'],
             json.dumps(record['observed_state'],sort_keys=True,ensure_ascii=False,separators=(',',':')))
        groups[key].append(record)
    unique=[];ambiguous=0
    for group in groups.values():
        first=group[0]
        if any((other['forecasts']!=first['forecasts'] or other['branches']!=first['branches'])
               for other in group[1:]):
            # Same observed state may hide different execution histories; do
            # not merge such cases or use private state to rank candidates.
            ambiguous+=1;unique.extend(group)
        else:unique.append(first)
    counts={};deltas={};unique_counts={}
    for model in ('pets','structured'):
        result=Counter();values=[]
        for record in records:
            chosen=record['forecasts'][model]['choice'];rule=record['forecasts']['rule']['choice']
            if chosen==rule:
                result['same']+=1;continue
            result['different']+=1
            a,b=record['branches'][str(chosen)]['flow'],record['branches'][str(rule)]['flow']
            if a is None or b is None:result['censored']+=1
            else:
                values.append(a-b)
                result['better' if a<b-1e-9 else 'worse' if a>b+1e-9 else 'equal_flow']+=1
        counts[model+'_vs_rule']=dict(result)
        deltas[model+'_vs_rule_mean_flow_on_differences']=statistics.mean(values) if values else None
        distinct=Counter()
        for record in unique:
            chosen=record['forecasts'][model]['choice'];rule=record['forecasts']['rule']['choice']
            if chosen==rule:
                distinct['same']+=1;continue
            distinct['different']+=1
            a,b=record['branches'][str(chosen)]['flow'],record['branches'][str(rule)]['flow']
            if a is None or b is None:distinct['censored']+=1
            else:distinct['better' if a<b-1e-9 else 'worse' if a>b+1e-9 else 'equal_flow']+=1
        unique_counts[model+'_vs_rule']=dict(distinct)
    summary={'scope':'post-confirmation exploratory all-decision diagnostic; same scenario and origins are correlated',
       'factual_episodes':len(scope),'same_state_decisions':len(records),'counts':counts,'flow_difference':deltas,
       'unique_observable_states':len(unique),'duplicate_origin_records_removed':len(records)-len(unique),
       'duplicate_observation_outcome_disagreement_groups':ambiguous,'unique_state_counts':unique_counts,
       'candidate_overlap_intersection':sum(r['candidate_intersection'] for r in records),
       'candidate_overlap_union':sum(r['candidate_union'] for r in records)}
    result={'summary':summary,'episodes':scope,'records':records}
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();evaluate(a.reference,a.output)


if __name__=='__main__':main()
