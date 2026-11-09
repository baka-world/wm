"""Scenario-clustered summaries of immutable acceptance output."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

from lab_transport.decision_value_round2 import bootstrap_paired


def mean(values):
    values=[x for x in values if x is not None]
    return statistics.mean(values) if values else None


def analyze(data):
    cases=data['cases']
    outcome={}
    for left,right in [('cpsat_rule','nearest_rule'),('cpsat_pets','cpsat_rule')]:
        differences=[(c['layout'],c['seed'],c['methods'][left]['flow_mean_completed']-
                      c['methods'][right]['flow_mean_completed']) for c in cases
                     if c['methods'][left]['flow_mean_completed'] is not None and
                        c['methods'][right]['flow_mean_completed'] is not None]
        outcome[left+'_minus_'+right]={'by_layout_seed_cluster':bootstrap_paired(differences),
            'scenario_wins':sum(d < -1e-9 for _,_,d in differences),
            'scenario_losses':sum(d > 1e-9 for _,_,d in differences),
            'scenario_ties':sum(abs(d)<=1e-9 for _,_,d in differences)}
    probabilities={}
    for measure,names in [('duration_mae',('rule','pets','structured')),
                          ('duration_nll',('rule','pets','structured')),
                          ('interval90_coverage',('rule','pets','structured')),
                          ('brier',('rule','structured','pets'))]:
        probabilities[measure]={name:mean(c['service_prediction'][measure].get(name) for c in cases) for name in names}
    states={}
    for horizon in ('1','3','5'):
        states[horizon]={}
        for model in ('rule','structured','structured_rule_motion','pets'):
            entries=[c['state_prediction'][horizon]['models'][model] for c in cases]
            states[horizon][model]={'mean_scenario_joint_accuracy':mean(x['joint_accuracy'] for x in entries),
                'correct_windows':sum(x['correct'] for x in entries), 'windows':sum(x['windows'] for x in entries)}
    per_regime={}
    for regime in sorted({c['regime'] for c in cases}):
        selected=[c for c in cases if c['regime']==regime]
        per_regime[regime]={name:mean(c['methods'][name]['flow_mean_completed'] for c in selected)
                            for name in ('nearest_rule','cpsat_rule','cpsat_pets')}
    return {'scope':'each scenario weighted equally; paired intervals resample layout/seed clusters, not individual ticks',
        'independent_layout_seed_clusters':len({(c['layout'],c['seed']) for c in cases}),
        'scenarios':len(cases),'closed_loop':data['summary'],'paired_flow_differences':outcome,
        'per_regime_mean_flow':per_regime,'mean_scenario_service_metrics':probabilities,
        'service_completed_sessions':sum(c['service_prediction']['completed_sessions'] for c in cases),
        'service_event_opportunities':sum(c['service_prediction']['event_opportunities'].get('pets',0) for c in cases),
        'joint_state_prediction':states}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('refuse to overwrite analysis')
    data=json.loads(a.input.read_text(encoding='utf-8'))
    result=analyze(data)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
