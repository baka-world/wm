"""One-time, precommitted novel-map factual holdout. Never selects model."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import torch

from business_transport.session_events import eligible_sessions
from business_transport.state_features import build_business_features
from business_transport.dynamics import WorkflowDynamics
from experiments.world_model_integration.build_dataset import case_for,rows_from_factual,label_for
from experiments.world_model_integration.empirical_baselines import EmpiricalHazards
from experiments.world_model_integration.compare_event_models import structured_probability
from experiments.business_constraints.run import make_model
from experiments.service_value.counterfactual import load_pr9_model
from lab_transport.business_session_learning import load_session_model,probabilities as old_probs
from lab_transport.business_session_v3 import load_v3,probs as new_probs,stats,grouped
from lab_transport.torch_structured_adapter import ProgressAwareRule
from lab_transport.paper_reuse.service_dataset import service_features

CONFIG=Path(__file__).with_name("v3_independent_config.json")
OLD_SEEDS=(941,947,21071,21073,18011,18013,18017,18019,18023,18029,18031,18037,
           19041,19043,19047,19049,20053,20059,20061,20063)

def run(source,old_v2,trained,output):
    output=Path(output)
    if output.exists():raise FileExistsError('independent evaluation is write-once')
    cfgbytes=CONFIG.read_bytes();cfg=json.loads(cfgbytes)
    if cfg.get('protocol')!='pr18-v3-independent-first-use' or not cfg.get('one_time_evaluation'):
        raise ValueError('independent protocol not frozen')
    specs=cfg['groups'];seeds=[x['seed'] for x in specs]
    if len(seeds)!=len(set(seeds)) or set(seeds)&set(OLD_SEEDS) or any(x['layout'] not in (3,4) for x in specs):
        raise ValueError('independent seed or map reused from previous work')
    torch.set_num_threads(1)
    data=json.loads(Path(source).read_text())
    train=EmpiricalHazards(data['train'])
    original,old_meta=load_session_model(old_v2,experimental=True)
    candidate,meta=load_v3(trained,experimental=True)
    if meta['variant'] not in ('V3-A','V3-B','V3-C') or meta['online_verified']:
        raise ValueError('unexpected selected model gate')
    network,info=load_pr9_model();network.eval()
    paired={name:[] for name in cfg['comparison']}
    counts=Counter();groups={}
    for spec in specs:
        business,geo=case_for(spec,cfg)
        rows,summary=rows_from_factual(business,geo,spec['seed'])
        rule=WorkflowDynamics(ProgressAwareRule(geo),business.rules)
        pets=make_model('pets',geo,network,info['threshold'],spec['seed']).distribution
        group=f"new-map{spec['layout']}-seed{spec['seed']}"
        events=[];omitted=Counter()
        for record in rows:
            before,actions,actual=record.observation,record.assignments,record.next_observation
            queries=eligible_sessions(before,actions)
            if not queries:continue
            # Predictor sees ONLY the same factual BEFORE state and executed action.
            factual=None
            for query,reason in queries:
                counts['total_considered']+=1
                if reason is not None:
                    counts['excluded_'+reason]+=1;omitted[reason]+=1;continue
                y,event=label_for(before,actual,query)
                if y is None:
                    counts['excluded_'+event]+=1;omitted[event]+=1;continue
                features=build_business_features(before,actions,geo,business.rules)
                observed={'label':y,'event':event,'query':query,'features':features,
                          'trajectory_group':group,'id':group+':'+str(before.tick)+':'+query['robot_id'],
                          'tick':before.tick,'seed':spec['seed'],'station':query['station_id'],
                          'phase':query['phase']}
                if factual is None:factual=rule.predict_next(before,actions)
                sample={s.id:s for s in before.samples}[query['sample_id']]
                target='in_transit' if query['phase']=='to_pickup' else 'completed'
                rprob=float({s.id:s for s in factual.samples}[sample.id].status==target)
                robot={r.id:r for r in before.robots}[query['robot_id']]
                station={s.id:s for s in before.stations}[query['station_id']]
                params=pets.parameters(service_features(before,robot,sample,query['phase'],station))
                p_pets=pets.finish_probability(params,robot.service_active_elapsed)
                from business_transport.session_features_v3 import encode_query_v3
                encode_query_v3(features,query) # enforce v3 target contract even for frozen baselines
                values={'B0':train.predict_one(observed,'B0'),
                        'B1':train.predict_one(observed,'B1'),
                        'B2':train.predict_one(observed,'B2'),
                        'Rule':rprob,'PETS':float(p_pets),
                        'Structured':float(structured_probability(network,before,actions,query)),
                        'V2 frozen':old_probs(original,[observed])[0],
                        'Validation-selected V3':new_probs(candidate,[observed],meta['input_schema'])[0]}
                for name,p in values.items():
                    if not 0<=p<=1:raise ValueError('invalid holdout predicted probability')
                    paired[name].append((observed,p))
                events.append(observed)
                counts['valid']+=1;counts['positive']+=y
        if not events:
            raise ValueError('independent map had zero valid started service queries')
        groups[group]={'seed':spec['seed'],'layout':spec['layout'],
                       'factual_steps':len(rows),'positive':sum(r['label'] for r in events),
                       'negative':len(events)-sum(r['label'] for r in events),
                       'omitted':dict(omitted),
                       'outage':spec['outage'],'capacity':spec['resource_capacity_A']}
    out={'protocol':'pr18-v3-independent-first-use',
        'config_sha256':hashlib.sha256(cfgbytes).hexdigest(),
        'source_train_data_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        'selected_checkpoint_sha256':hashlib.sha256(Path(trained).read_bytes()).hexdigest(),
        'selected_variant':meta['variant'],
        'source_old_v2_sha256':hashlib.sha256(Path(old_v2).read_bytes()).hexdigest(),
        'factual_counts':dict(counts),'trajectory_groups':groups,'models':{},
        'no_tuning_on_holdout':True,
        'warnings':['only synthetic BusinessSimulator data',
                    'paused / non-started service not a valid negative event',
                    'full-trajectory groups, not independent adjacent events']}
    for name,values in paired.items():
        records=[r for r,p in values];p=[p for r,p in values]
        out['models'][name]={'overall':stats(records,p),
            'trajectory':grouped(records,p,'trajectory'),
            'station':grouped(records,p,'station'),
            'phase':grouped(records,p,'phase'),
            'effective_elapsed':grouped(records,p,'elapsed')}
    baseline=out['models']['Rule']['trajectory']
    v3=out['models']['Validation-selected V3']['trajectory']
    out['per_trajectory_vs_rule']={g:{
        'brier_v3_minus_rule':v3[g]['brier']-baseline[g]['brier'],
        'n':v3[g]['count']} for g in groups}
    out['independent_gate']={
        'v3_brier_below_B0':out['models']['Validation-selected V3']['overall']['brier']<
                            out['models']['B0']['overall']['brier'],
        'v3_brier_below_B2':out['models']['Validation-selected V3']['overall']['brier']<
                            out['models']['B2']['overall']['brier'],
        'v3_brier_below_Rule':out['models']['Validation-selected V3']['overall']['brier']<
                            out['models']['Rule']['overall']['brier']}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'models':{k:{'brier':v['overall']['brier'],
         'log_loss':v['overall']['log_loss'],
         'ece':v['overall']['ece5'],
         'precision':v['overall']['precision'],
         'recall':v['overall']['recall']} for k,v in out['models'].items()},
         'gate':out['independent_gate'],'counts':out['factual_counts']},ensure_ascii=False))
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--frozen-v2',type=Path,required=True)
    p.add_argument('--selected',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.source,a.frozen_v2,a.selected,a.output)
if __name__=='__main__':main()
