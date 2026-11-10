"""PR18 isolated CPU tick-zero candidate cost-to-go identification experiment.

Offline truth ALWAYS comes from BusinessSimulator restart via rollout_value.run_branch,
never from model features. Grouped Train/Val/Test, no selected test-case filtering.
Run: python -m experiments.world_model_integration.value_learnability.run --v4 /path/v4-data.pt --out /path/out
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import time
import numpy as np

from business_transport.observation import BusinessObserver
from business_transport.scenario import compile_scenario
from business_transport.simulation import BusinessSimulator
from experiments.world_model_integration.build_dataset import case_for
from experiments.world_model_integration.rollout_value.evaluate import run_branch
from experiments.world_model_integration.v4_dispatch import policy_for
from lab_transport.decision_value import batch_candidates, candidates_for, nearest_batch

PROTOCOL = Path(__file__).with_name('protocol.json')
BASE_KEYS = ('n_robot','n_visible','n_busy','n_active','n_occupied','n_capacity',
             'mean_prior','tick','n_assign','sum_approach','sum_trip','max_approach',
             'sum_service','capacity_pressure','blocked','remaining_stages',
             'queue_proxy','min_idle_to_pickup','assign_x_pressure','distance_x_pressure')
ROLLOUT_KEYS = ('unfinished_job_ticks','remaining_distance','estimated_remaining_service',
                'resource_contention_proxy','terminal_work')

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def frozen_specs(p):
    """Entire splits, maps, configs, seeds are determined BEFORE reading any outcomes."""
    groups={}
    for split,n in p['split_counts'].items():
        layouts=p['split_layouts'][split]
        seq=[]
        for index in range(n):
            seed=p['split_seed_bases'][split]+index*37
            rng=random.Random(seed)
            jobs=2+(index%3)
            releases=([0,0,0,0] if index%3==0 else
                      [0,0,3,7] if index%3==1 else [0,2,6,10])[:jobs]
            seq.append({'seed':seed,'layout':layouts[index%len(layouts)],
                'resource_capacity_A':1+(index%2), 'robot_count':1+(index%3),
                'workflow_variant':'three' if index%4==0 else 'two',
                'workflow_depth':'single' if index%5==0 else 'multiple',
                'station_service_ticks':{k:rng.randint(2,8) for k in ('A0','A1','B0','B1','C')},
                'jitter_ticks':0 if index%4==0 else 1+(index%2),
                'outage':bool(index%3==0), 'outage_window':[13,21],
                'outage_station':'C', 'job_workflows':[('w0' if j%2==0 else 'w1') for j in range(jobs)],
                'job_release_ticks':releases, 'max_ticks':115})
        groups[split]=seq
    seeds=[s['seed'] for specs in groups.values() for s in specs]
    assert len(seeds)==len(set(seeds)) and not set(seeds)&{43011,43015,43019}
    seen={k:set(v['layout'] for v in items) for k,items in groups.items()}
    assert not (seen['train']&seen['validation'] or seen['train']&seen['test'] or seen['validation']&seen['test'])
    return groups

def scenario_for(spec):
    business,geo=case_for(spec,spec)
    if spec['workflow_depth']=='single':
        rules=business.rules
        cfg={'resources':[{'id':r.id,'stations':list(r.stations),'capacity':r.capacity} for r in rules.resources],
             'workflows':{key:[{'pickup':route[0][0],'dropoff':route[0][1]}] for key,route in rules.workflows.items()},
             'jobs':[{'id':j['id'],'workflow':j['workflow'],'release_tick':j['release_tick']} for j in business.jobs],
             'max_ticks':spec['max_ticks']}
        business=compile_scenario(replace(business.scenario,samples=()),cfg)
    return business,geo

def initial_state(business,geo,seed):
    """Only tick0, before stochastic service starts; never inject private state."""
    obs=BusinessObserver(); captured=[]
    def on_observation(o):
        rich=obs.observe(o)
        if o.tick==0:captured.append((o,rich))
    BusinessSimulator(business,'world',seed,collect_metrics=False).run(
        dispatch_policy=lambda o,c:nearest_batch(o,c,geo),on_observation=on_observation)
    if len(captured)!=1 or captured[0][0].tick!=0 or captured[0][1].tick!=0:
        raise AssertionError('tick0 observation capture failed')
    return captured[0]

def base_features(obs,plan,geo,rules):
    """Only public observation, selected legal action, geometry, and public rules."""
    robots={r.id:r for r in obs.robots}
    samples={s.id:s for s in obs.samples}
    stations={s.id:s for s in obs.stations}
    if len({a.robot_id for a in plan})!=len(plan) or len({a.sample_id for a in plan})!=len(plan):
        raise ValueError('not a distinct robot-sample matching')
    if any(a.robot_id not in robots or a.sample_id not in samples or
           robots[a.robot_id].task_id is not None or samples[a.sample_id].status!='waiting' for a in plan):
        raise ValueError('candidate not legal under current observation')
    occupied=sum(int(getattr(st,'resource_occupied',0)) for st in obs.stations)
    active=sum(r.service_elapsed is not None for r in obs.robots)
    resource_demand={r.id:0 for r in rules.resources}
    approaches=[];trips=[];services=[];blocked=0;remaining=0
    for a in plan:
        r=robots[a.robot_id];s=samples[a.sample_id]
        st1,st2=stations[s.pickup],stations[s.dropoff]
        d1=geo.distance(r.position,st1.position)
        d2=geo.distance(st1.position,st2.position)
        if d1>=geo.unreachable or d2>=geo.unreachable:raise ValueError('unreachable candidate')
        approaches.append(d1);trips.append(d2)
        services.append(float(st1.expected_service_ticks)+float(st2.expected_service_ticks))
        resource_demand[rules.station_resources[s.pickup].id]+=1
        blocked+=int(not getattr(st1,'device_available',st1.available))
        workflow=rules.workflows.get(getattr(s,'workflow',''),())
        remaining+=max(0,len(workflow)-getattr(s,'stage_index',0)-1)
    pressure=sum(max(0,n-next(r.capacity for r in rules.resources if r.id==rid)) for rid,n in resource_demand.items())
    queued=sum(int(s.status=='waiting') for s in obs.samples)
    idle=[r for r in obs.robots if r.task_id is None]
    waiting=[s for s in obs.samples if s.status=='waiting']
    min_dist=min((geo.distance(r.position,stations[s.pickup].position) for r in idle for s in waiting),default=0)
    fields=[len(obs.robots),len(obs.samples),sum(r.task_id is not None for r in obs.robots),
            active,occupied,sum(r.capacity for r in rules.resources),
            statistics.mean(float(s.expected_service_ticks) for s in obs.stations),obs.tick,
            len(plan),sum(approaches),sum(trips),max(approaches,default=0),sum(services),
            pressure,blocked,remaining,queued,min_dist,len(plan)*pressure,sum(approaches)*pressure]
    assert len(fields)==len(BASE_KEYS) and all(np.isfinite(x) for x in fields)
    return [float(x) for x in fields]

def heuristic(x):
    d=dict(zip(BASE_KEYS,x))
    return (-100*d['n_assign'] + d['sum_approach']+0.5*d['sum_trip']+
            0.5*d['sum_service']+8*d['capacity_pressure']+8*d['blocked'])

def predict_features(business,geo,obs,plans,v4_path):
    """Frozen five-step/three-particle Rule and V4; NO real future as features."""
    answer={};timings={};queries=0
    for arm in ('Rule','V4'):
        pol=policy_for(arm,business,geo,None,v4_path)
        if pol.horizon!=5 or pol.particles!=3:raise AssertionError('changed frozen model budget')
        started=time.process_time()
        records=pol.evaluate(obs,plans)
        timings[arm]=time.process_time()-started
        if len(records)!=len(plans):raise AssertionError('candidate cardinality changed')
        answer[arm]=records
        if arm=='V4':
            inner=getattr(pol.model,'model',None)
            queries=int(getattr(inner,'query_calls',0))
    return answer,timings,queries

def collect_group(split,spec,p,v4_path,deadline):
    start=time.monotonic()
    business,geo=scenario_for(spec)
    raw_initial,obs=initial_state(business,geo,spec['seed'])
    batches,total=batch_candidates(obs,candidates_for(obs),geo,budget=p['candidate_limit']+1)
    plans=list(dict.fromkeys(tuple(plan) for plan in batches if plan))[:p['candidate_limit']]
    if not plans: raise AssertionError('no legal dispatch candidate')
    feature_started=time.process_time()
    public=[base_features(obs,plan,geo,business.rules) for plan in plans]
    base_feature_seconds=time.process_time()-feature_started
    pred,times,queries=predict_features(business,geo,obs,plans,v4_path)
    # All truth is computed AFTER model features have been completed.
    offsets=p['future_seed_offsets'] if spec['jitter_ticks'] else [0]
    outcomes=[]
    for ix,plan in enumerate(plans):
        flows=[];visible_flows=[];runtime=[]
        for offset in offsets:
            if time.monotonic()>deadline:raise TimeoutError('30-minute CPU budget exhausted before test')
            result=run_branch(business,geo,spec['seed']+offset,plan,'P1_nearest',raw_initial)
            flows.append(float(result['restricted_mean_flow']))
            visible_jobs={getattr(x,'job_id','') for x in obs.samples}
            visible=[flow for job,flow in zip(business.jobs,result['per_job_flow']) if job['id'] in visible_jobs]
            visible_flows.append(float(statistics.mean(visible)) if visible else None)
            runtime.append(float(result['cpu_wall_seconds']))
            if result['safety_capacity_violations']:raise AssertionError('capacity violation')
            if ix==0 and offset==0:
                duplicate=run_branch(business,geo,spec['seed'],plan,'P1_nearest',raw_initial)
                if any(result[k]!=duplicate[k] for k in ('history_sha256','events_sha256','restricted_mean_flow')):
                    raise AssertionError('same-seed replay not deterministic')
        outcomes.append({'values':flows,'mean':statistics.mean(flows),
                         'visible_job_values':visible_flows,
                         'visible_job_mean':statistics.mean(x for x in visible_flows if x is not None) if any(x is not None for x in visible_flows) else None,
                         'branch_wall_seconds':runtime})
    y=[x['mean'] for x in outcomes]
    rows=[]
    for ix,plan in enumerate(plans):
        feats=public[ix]
        rows.append({'candidate_index':ix,'action':[asdict(a) for a in plan],
            'base_features':feats,'rule_rollout':[float(pred['Rule'][ix]['component_means'][k]) for k in ROLLOUT_KEYS],
            'v4_rollout':[float(pred['V4'][ix]['component_means'][k]) for k in ROLLOUT_KEYS],
            'rule_score':list(pred['Rule'][ix]['score']),'v4_score':list(pred['V4'][ix]['score']),
            'heuristic_score':heuristic(feats),'full_future':outcomes[ix],
            'label_centered':y[ix]-statistics.mean(y)})
    group={'split':split,'seed':spec['seed'],'layout':spec['layout'],'spec':spec,
        'state_tick':0,'candidate_count':len(plans),'available_batches_including_wait':total,
        'distinct_action_count':len({json.dumps(r['action'],sort_keys=True) for r in rows}),
        'oracle_cost_spread':max(y)-min(y),
        'active_service_sessions_at_state':sum(r.service_elapsed is not None for r in obs.robots),
        'resource_contested':spec['resource_capacity_A']==1 and len(spec['job_workflows'])>=2,
        'v4_effective_queries':queries, 'model_cpu_seconds':times,
        'base_feature_cpu_seconds':base_feature_seconds,
        'ground_truth_scope':'offline complete future all configured jobs (not online features)',
        'futures_per_candidate':len(offsets),'rows':rows,
        'group_wall_seconds':time.monotonic()-start}
    return group

def raw_vector(row,variant):
    x=row['base_features']
    if variant=='current':return x
    return x+row[variant+'_rollout']

def center_matrix(groups,variant):
    matrices=[];ys=[];weights=[]
    for g in groups:
        a=np.asarray([raw_vector(r,variant) for r in g['rows']],dtype=float)
        matrices.append(a-a.mean(axis=0,keepdims=True))
        ys.extend([r['label_centered'] for r in g['rows']])
        weights.extend([1/len(a)]*len(a))
    return np.vstack(matrices),np.asarray(ys),np.asarray(weights)

def fit(groups,variant,alpha):
    tic=time.process_time()
    x,y,w=center_matrix(groups,variant)
    std=x.std(axis=0);std=np.where(std<1e-9,1.,std)
    z=x/std
    design=z*np.sqrt(w[:,None]);target=y*np.sqrt(w)
    if alpha==0:coef=np.linalg.lstsq(design,target,rcond=None)[0]
    else:coef=np.linalg.solve(design.T@design+alpha*np.eye(design.shape[1]),design.T@target)
    return {'variant':variant,'alpha':alpha,'std':std.tolist(),'coef':coef.tolist(),
            'dim':len(coef),'train_groups':len(groups),'train_rows':len(y),
            'cpu_fit_seconds':time.process_time()-tic,
            'feature_names':list(BASE_KEYS)+([] if variant=='current' else list(ROLLOUT_KEYS))}

def model_predict(g,model):
    a=np.asarray([raw_vector(r,model['variant']) for r in g['rows']],dtype=float)
    x=(a-a.mean(axis=0,keepdims=True))/np.asarray(model['std'])
    return (x@np.asarray(model['coef'])).tolist()

def score_order(scores):return sorted(range(len(scores)),key=lambda i:(scores[i],i))

def measure(groups,models):
    by_group=[];latency={k:[] for k in ('B0','B1','B2','B3','B4','B5')}
    for g in groups:
        actual=[r['full_future']['mean'] for r in g['rows']];minimum=min(actual)
        methods={'B0':[tuple(r['rule_score']) for r in g['rows']],
                 'B1':[r['heuristic_score'] for r in g['rows']],
                 'B3':[tuple(r['v4_score']) for r in g['rows']]}
        for arm,variant in (('B2','current'),('B4','rule'),('B5','v4')):
            tic=time.process_time()
            methods[arm]=model_predict(g,models[variant]);
            latency[arm].append(time.process_time()-tic+ (g['model_cpu_seconds']['Rule'] if arm=='B4' else g['model_cpu_seconds']['V4'] if arm=='B5' else 0.)+g['base_feature_cpu_seconds'])
        for arm in ('B0','B1','B3'):
            latency[arm].append((g['model_cpu_seconds']['V4' if arm=='B3' else 'Rule'] if arm in ('B0','B3') else 0.)+g['base_feature_cpu_seconds'])
        measures={}
        for name,scores in methods.items():
            order=score_order(scores);best=score_order(actual)[0]
            pairs=[(i,j) for i in range(len(actual)) for j in range(i+1,len(actual)) if abs(actual[i]-actual[j])>1e-9]
            concord=sum((scores[i]<scores[j])==(actual[i]<actual[j]) for i,j in pairs)/len(pairs) if pairs else None
            gap=[abs((scores[i]-scores[j])-(actual[i]-actual[j])) for i,j in pairs] if name in ('B2','B4','B5') else []
            measures[name]={'chosen':order[0],'oracle_choice':best,'regret':actual[order[0]]-minimum,
                            'top1':int(abs(actual[order[0]]-minimum)<1e-9),
                            'pair_concordance':concord,
                            'relative_gap_mae':statistics.mean(gap) if gap else None,
                            'scores':[list(s) if isinstance(s,tuple) else float(s) for s in scores]}
        by_group.append({'seed':g['seed'],'layout':g['layout'],'candidate_count':len(actual),
                        'cost_spread':max(actual)-minimum,'contested':g['resource_contested'],
                        'workflow_depth':g['spec']['workflow_depth'],'v4_queries':g['v4_effective_queries'],
                        'methods':measures})
    overview={}
    for name in ('B0','B1','B2','B3','B4','B5'):
        entries=[g['methods'][name] for g in by_group]
        informative=[x for x,g in zip(entries,by_group) if g['cost_spread']>1e-9 and g['candidate_count']>=2]
        overview[name]={'groups':len(entries),'informative_groups':len(informative),
          'mean_regret':statistics.mean(x['regret'] for x in entries),
          'top1_all':statistics.mean(x['top1'] for x in entries),
          'top1_informative':statistics.mean(x['top1'] for x in informative) if informative else None,
          'pair_concordance':statistics.mean(x['pair_concordance'] for x in entries if x['pair_concordance'] is not None) if informative else None,
          'relative_gap_mae':statistics.mean(x['relative_gap_mae'] for x in entries if x['relative_gap_mae'] is not None) if name in ('B2','B4','B5') and any(x['relative_gap_mae'] is not None for x in entries) else None,
          'mean_scoring_cpu_seconds_per_state':statistics.mean(latency[name])}
    return {'groups':by_group,'summary':overview}

def select_models(train,validation,p):
    models={};tuning={}
    for variant in ('current','rule','v4'):
        candidates={}
        for a in (0.,*p['ridge_alphas']):
            m=fit(train,variant,a)
            validated=measure(validation,{variant:m,**{other:fit(train,other,1.) for other in ('current','rule','v4') if other!=variant}})
            name={'current':'B2','rule':'B4','v4':'B5'}[variant]
            metric=validated['summary'][name]
            candidates[str(a)]={'mean_regret':metric['mean_regret'],'gap_mae':metric['relative_gap_mae'],
                                'cpu_fit_seconds':m['cpu_fit_seconds']}
        selected_key=min(candidates,key=lambda key:(candidates[key]['mean_regret'],
            candidates[key]['gap_mae'] if candidates[key]['gap_mae'] is not None else float('inf'),float(key)))
        selected=float(selected_key)
        models[variant]=fit(train,variant,selected)
        tuning[variant]={'candidates':candidates,'selected_alpha':selected}
    return models,tuning

def paired_ci(by_group,left,right,seed=1908):
    d=np.asarray([g['methods'][left]['regret']-g['methods'][right]['regret'] for g in by_group])
    if len(d)<2:return None
    rng=np.random.default_rng(seed)
    means=np.mean(d[rng.integers(0,len(d),size=(2000,len(d)))],axis=1)
    return [float(np.quantile(means,.025)),float(np.quantile(means,.975))]

def verdict(development,test):
    d=test['summary'];g=test['groups']
    informative=sum(x['cost_spread']>1e-9 and x['candidate_count']>1 for x in g)
    coverage=sum(x['v4_queries']>0 for x in g)
    if informative<5:return {'value':'BLOCKED','v4':'BLOCKED','reason':'insufficient informative independent decision groups'}
    if d['B2']['mean_regret']>=d['B1']['mean_regret'] and d['B4']['mean_regret']>=d['B1']['mean_regret']:
        value='NO-GO'
    else:
        value='EXPLORATORY_ONLY'  # small independent sample cannot establish stable GO
    if coverage==0:v4='NO-GO-V4: no effective model event queries'
    elif d['B5']['mean_regret']>=d['B4']['mean_regret']:v4='NO-GO-V4'
    else:v4='EXPLORATORY_ONLY_NOT_GO_V4'
    return {'value':value,'v4':v4,'informative_groups':informative,'groups_with_v4_queries':coverage,
            'reason':'16 independent groups are exploratory; inspect paired CI; do not claim statistical proof'}

def write_once(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(data,f,indent=2,ensure_ascii=False);f.write('\n')

def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument('--v4',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--phase',choices=['preflight','full'],default='full')
    args=ap.parse_args(argv)
    p=json.loads(PROTOCOL.read_text());groups=frozen_specs(p)
    if sha(args.v4)!=p['v4_checkpoint_sha256']:raise ValueError('V4 checkpoint SHA mismatch')
    if args.out.exists():raise FileExistsError('refusing to overwrite evidence directory')
    args.out.mkdir(parents=True)
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    assert torch.version.cuda is None
    torch.set_num_threads(1)
    deadline=time.monotonic()+p['clock_budget_seconds'];start=time.monotonic()
    write_once(args.out/'frozen_groups.json',groups)
    write_once(args.out/'provenance.json',{'protocol_sha256':sha(PROTOCOL),'v4_sha256':sha(args.v4),
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,
                       'platform':platform.platform()},'initial_head':p['source_head'],
        'mode':args.phase,'cpu_only':True})
    if args.phase=='preflight':
        stats=[]
        for split in ('train','validation'):
            s=groups[split][0];item=collect_group(split,s,p,args.v4,deadline)
            stats.append({'seed':s['seed'],'candidate_count':item['candidate_count'],
                          'jitter':s['jitter_ticks'],'v4_queries':item['v4_effective_queries'],
                          'elapsed':item['group_wall_seconds'],'spread':item['oracle_cost_spread']})
        write_once(args.out/'preflight.json',stats)
        print(json.dumps({'preflight':stats},ensure_ascii=False),flush=True)
        return
    data={}
    for split in ('train','validation'):
        data[split]=[]
        for spec in groups[split]:
            if time.monotonic()>deadline:raise TimeoutError('CPU budget ended during development')
            g=collect_group(split,spec,p,args.v4,deadline)
            data[split].append(g)
            print(f"{split} {len(data[split])}/{len(groups[split])} candidates={g['candidate_count']} spread={g['oracle_cost_spread']:.5f} v4q={g['v4_effective_queries']}",flush=True)
        write_once(args.out/(split+'.json'),data[split])
    informative=sum(g['candidate_count']>1 and g['oracle_cost_spread']>1e-9 for g in data['train'])
    if informative<8:
        write_once(args.out/'verdict.json',{'value':'BLOCKED','reason':'train labels lack informative candidate gap','informative_train_groups':informative})
        return
    models,tuning=select_models(data['train'],data['validation'],p)
    for variant,m in models.items():write_once(args.out/('model_'+variant+'.json'),m)
    development=measure(data['validation'],models)
    write_once(args.out/'validation_metrics.json',development)
    write_once(args.out/'validation_selection.json',tuning)
    # Freeze and checksum models and selection BEFORE accessing Test scenario outcomes.
    lock={'protocol_sha256':sha(PROTOCOL),'test_group_ids':[s['seed'] for s in groups['test']],
          'candidate_rule':'first four distinct nonempty legal batch_candidates in stable order',
          'prediction_budget':'five steps three particles both arms',
          'models':{variant:sha(args.out/('model_'+variant+'.json')) for variant in models},
          'model_selection_sha256':sha(args.out/'validation_selection.json'),
          'test_read_count_allowed':1}
    write_once(args.out/'test_lock.json',lock)
    print('TEST_LOCK_SHA256',sha(args.out/'test_lock.json'),flush=True)
    test=[]
    for spec in groups['test']:
        if time.monotonic()>deadline:raise TimeoutError('CPU wall budget ended before test completed')
        g=collect_group('test',spec,p,args.v4,deadline)
        test.append(g)
        print(f"TEST {len(test)}/{len(groups['test'])} candidates={g['candidate_count']} v4q={g['v4_effective_queries']}",flush=True)
    write_once(args.out/'test.json',test)
    result=measure(test,models)
    result['paired_ci_regret_improvement']={
        'B2_vs_B1':paired_ci(result['groups'],'B1','B2'),
        'B4_vs_B2':paired_ci(result['groups'],'B2','B4'),
        'B5_vs_B4':paired_ci(result['groups'],'B4','B5')}
    write_once(args.out/'test_metrics.json',result)
    write_once(args.out/'verdict.json',verdict(development,result))
    print('FINAL',json.dumps({'summary':result['summary'],'verdict':verdict(development,result),
         'wall_seconds':time.monotonic()-start},ensure_ascii=False),flush=True)

if __name__=='__main__':main()