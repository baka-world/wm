"""Read-only audit of archived PR18 synthetic urgent workflow evidence.

No simulation/model imports or calls. This module only parses frozen JSON/JSON.gz.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import time

SOURCE_SHA = {
    'full_raw.json.gz':'85197b34d21a1bfe232086f78d6741a584ba710eed1f9517402270c2beaaef90',
    'oracle.json':'52985d64beeee42d2094cbde625c7280b30b648a794a65f22800610627e63b98',
    'selection_lock.json':'7bb8d9aae414407d56fd321b2538f646d8445fc1fdf51cbe961aa9100b9645d7',
    'paired.json':'2b3a9f20f6cd4e7eda6cb29f427daab88ae844cc889c8b682768e9879f62f1d7',
    'summary.json':'41225e243b6dde9a49386e276c8255e9e32ea6ef212b6ce3285aa5c77f8a537d',
}
ARMS=('P0','P1','P2','P3')
FIXED=(('903074:10',903074,10),('903293:10',903293,10),('903512:13',903512,13))

def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_archive(evidence):
    for name,expected in SOURCE_SHA.items():
        actual=sha_file(evidence/name)
        if actual!=expected:raise AssertionError(f'input SHA mismatch {name}: {actual}')
    with gzip.open(evidence/'full_raw.json.gz','rt',encoding='utf-8') as f: raw=json.load(f)
    oracle=json.loads((evidence/'oracle.json').read_text())
    lock=json.loads((evidence/'selection_lock.json').read_text())
    paired=json.loads((evidence/'paired.json').read_text())
    summary=json.loads((evidence/'summary.json').read_text())
    assert len(raw['groups'])==96 and len(oracle)==12 and len(lock['selected'])==12
    assert set(a['id'] for a in oracle)==set(x['id'] for x in lock['selected'])
    assert summary['groups']==24 and summary['oracle_valid']==12 and not summary['oracle_blocked']
    assert raw['protocol_sha256']==summary['protocol_sha256']
    cases={(x['spec']['seed'],x['arm']):x for x in raw['groups']}
    seeds=sorted({s for s,_ in cases})
    assert len(seeds)==24 and len(cases)==24*4
    for seed in seeds:
        assert set((seed,a) for a in ARMS) <= set(cases)
    return {'cases':cases,'seeds':seeds,'oracle':oracle,'lock':lock,'summary':summary,'paired':paired}

def pairs(plan):
    return tuple(sorted((a['robot_id'],a['sample_id']) for a in plan))

def rankings(scores):
    return tuple(sorted(range(len(scores)),key=lambda i:tuple(scores[i])))

def class_one(a,b):
    """Mutually exclusive C0-C5. Secondary masks retained separately.

    Compare only strictly identical current state, candidates and order.
    C0 all candidate business vectors same in both;
    C1 unique minimum missed (higher-level goal hard blocks ETA);
    C2 same eligible scorer values and winner;
    C3 eligible numeric values differ but ranking unchanged;
    C4 eligible ranking changed but winner preserved;
    C5 winner changed.
    """
    A=[p['score'] for p in a['plans']];B=[p['score'] for p in b['plans']]
    if not A:return 'SINGLE_OR_EMPTY'
    if len({tuple(s[:-1]) for s in A})==len({tuple(s[:-1]) for s in B})==1:
        return 'C0'
    minimum=min(x[0] for x in A)
    eligible=[i for i,x in enumerate(A) if x[0]==minimum]
    assert all(B[i][0]==A[i][0] for i in range(len(A)))
    if len(eligible)==1:return 'C1'
    wa=rankings([A[i] for i in eligible]);wb=rankings([B[i] for i in eligible])
    if wa[0]!=wb[0]:return 'C5'
    if wa!=wb:return 'C4'
    if any(A[i][1:-1]!=B[i][1:-1] for i in eligible):return 'C3'
    return 'C2'

def audit_scores(data):
    cases=data['cases'];seed_list=data['seeds']
    counts=Counter();coverage=defaultdict(set);examples=defaultdict(list)
    masks=Counter();mask_cov=defaultdict(set);rows=[]
    for seed in seed_list:
        a,b=cases[seed,'P2'],cases[seed,'P3']
        assert a['history_rows']==b['history_rows'],f'history diverged for {seed}'
        assert a['event_rows']==b['event_rows'],f'events diverged for {seed}'
        assert a['history_sha256']==b['history_sha256']
        assert a['events_sha256']==b['events_sha256']
        aa,bb=a['decisions'],b['decisions'];assert len(aa)==len(bb)
        counts['paired_scenarios']+=1
        for x,y in zip(aa,bb):
            assert x['tick']==y['tick']
            assert pairs(x['selected'])==pairs(y['selected'])
            assert len(x['plans'])==len(y['plans'])
            assert [pairs(p['matching']) for p in x['plans']]==[pairs(p['matching']) for p in y['plans']]
            counts['all_strict_paired_calls']+=1
            if len(x['plans'])<2:continue
            counts['multi_candidate_calls']+=1
            ca=class_one(x,y);counts[ca]+=1;coverage[ca].add(seed)
            A=[p['score'] for p in x['plans']];B=[p['score'] for p in y['plans']]
            assert len({pairs(p['matching']) for p in x['plans']})==len(A)
            minmiss=min(s[0] for s in A)
            eligible=[i for i in range(len(A)) if A[i][0]==minmiss]
            numeric=[i for i in range(len(A)) if A[i][:-1]!=B[i][:-1]]
            inside=[i for i in numeric if i in eligible]
            outside=[i for i in numeric if i not in eligible]
            assert all(s[0]==t[0] for s,t in zip(A,B))
            if len(set(s[0] for s in A))>1: masks['different_missed_values']+=1
            if len(eligible)==1:masks['unique_min_missed']+=1
            if numeric:masks['numeric_score_changed']+=1
            if inside:masks['changed_among_min_missed']+=1
            if outside:masks['changed_outside_min_missed']+=1
            if numeric and not inside:masks['score_changed_only_dominated']+=1
            if numeric and len(eligible)==1:masks['numeric_changed_but_first_layer_unique']+=1
            if len(eligible)>1:masks['multiple_min_missed']+=1
            if rankings(A)!=rankings(B):masks['full_candidate_rank_order_change']+=1
            if rankings(A)[0]!=rankings(B)[0]:masks['top_choice_change']+=1
            if any(A[i][1]!=B[i][1] for i in range(len(A))):masks['urgent_risk_changed']+=1
            if any(A[i][2]!=B[i][2] for i in range(len(A))):masks['routine_risk_changed']+=1
            if numeric:mask_cov['numeric_score_changed'].add(seed)
            if inside:mask_cov['changed_among_min_missed'].add(seed)
            if not inside and numeric:mask_cov['score_changed_only_dominated'].add(seed)
            row={'seed':seed,'tick':x['tick'],'class':ca,'candidate_count':len(A),
                 'missed': [s[0] for s in A], 'eligible_ids':eligible,
                 'p2_rank':rankings(A),'p3_rank':rankings(B),
                 'p2_chosen':rankings(A)[0],'p3_chosen':rankings(B)[0],
                 'numeric_changed_indices':numeric,'within_best_coverage_changed':inside,
                 'all_candidate_scores_p2':A,'all_candidate_scores_p3':B,
                 'candidate_matchings':[pairs(p['matching']) for p in x['plans']],
                 'source':'full_raw.json.gz/groups[%s]/decisions[tick=%s]'%(seed,x['tick'])}
            rows.append(row)
            if len(examples[ca])<5:examples[ca].append({'seed':seed,'tick':x['tick'],'score_p2':A,'score_p3':B,
                                                        'eligible_ids':eligible,'matching':row['candidate_matchings']})
    assert counts['all_strict_paired_calls']>=counts['multi_candidate_calls']
    assert sum(counts[k] for k in ('C0','C1','C2','C3','C4','C5'))==counts['multi_candidate_calls']
    return {'counts':dict(counts),'class_scenario_coverage':{k:len(coverage[k]) for k in ('C0','C1','C2','C3','C4','C5')},
            'mask_metrics':dict(masks),'mask_scenario_coverage':{k:len(v) for k,v in mask_cov.items()},
            'examples':dict(examples),'state_details':rows}

def job_status(row,horizon):
    if row['completed']:
        return 'on_time' if row['on_time'] else 'late_complete'
    return 'censored_not_yet_due' if row['deadline_tick']>=horizon else 'censored_overdue'

def audit_jobs(data):
    cases=data['cases'];by_arm={};transitions={};records=[];due_stage_checks={}
    for arm in ARMS:
        ct=Counter();details=[]
        for seed in data['seeds']:
            case=cases[seed,arm]
            horizon=case['spec']['max_ticks']
            for row in case['metrics']['jobs']:
                if row['priority_class']!='urgent':continue
                state=job_status(row,horizon);ct[state]+=1
                assert row['on_time']==(row['final_tick'] is not None and row['final_tick']<=row['deadline_tick'])
                assert row['restricted_flow']==(row['final_tick']-row['release_tick'] if row['completed'] else horizon-row['release_tick'])
                if row['censored']:assert row['final_tick'] is None
                details.append({'seed':seed,'job_id':row['job_id'],'state':state,
                                'deadline':row['deadline_tick'],'finish_tick':row['final_tick'],
                                'retest':row['retest'],'late_lower_bound':row['late_ticks_or_censored_lower_bound'],
                                'last_completed_stage_tick':max([s['completed_tick'] for s in row['stages'] if s['completed_tick'] is not None],default=None)})
        by_arm[arm]={'counts':dict(ct),'total_urgent':sum(ct.values())}
        assert sum(ct.values())==48
        due_stage_checks[arm]=details
    for arm in ('P1','P2','P3'):
        tr=Counter();changed=[]
        for seed in data['seeds']:
            a=cases[seed,'P0'];b=cases[seed,arm]
            p0={j['job_id']:j for j in a['metrics']['jobs'] if j['priority_class']=='urgent'}
            current={j['job_id']:j for j in b['metrics']['jobs'] if j['priority_class']=='urgent'}
            assert set(p0)==set(current)
            for jid,old in sorted(p0.items()):
                new=current[jid];key=(job_status(old,a['spec']['max_ticks']),job_status(new,b['spec']['max_ticks']))
                tr[' => '.join(key)]+=1
                if key[0]!=key[1]:
                    changed.append({'seed':seed,'job':jid,'from':key[0],'to':key[1],
                                    'deadline':old['deadline_tick'],'p0_final':old['final_tick'],
                                    'comp_final':new['final_tick'],'p0_retest':old['retest'],
                                    'comp_retest':new['retest'],'scenario_outage':b['spec']['outage'],
                                    'p0_late_bound':old['late_ticks_or_censored_lower_bound'],
                                    'comp_late_bound':new['late_ticks_or_censored_lower_bound']})
        transitions[arm]={'matrix':dict(tr),'changed_rows':changed}
    # Deadline expired before final-stage assignment: events are factual; this is observation, not unique cause.
    for arm in ARMS:
        late_assigned=[];completed_urgent=[];retest_urgent=[]
        for seed in data['seeds']:
            case=cases[seed,arm]
            jobs={j['job_id']:j for j in case['metrics']['jobs']}
            for e in case['event_rows']:
                if e['type']=='assigned':
                    sid=e['sample_id'];jid=sid.split('::stage')[0]
                    if jid in jobs and jobs[jid]['priority_class']=='urgent' and e['tick']>jobs[jid]['deadline_tick']:
                        late_assigned.append({'seed':seed,'job_id':jid,'tick':e['tick'],
                                               'deadline':jobs[jid]['deadline_tick'],'stage':sid})
            completed_urgent.extend([j['final_tick'] for j in jobs.values() if j['priority_class']=='urgent' and j['final_tick'] is not None]);
            retest_urgent.extend([{'seed':seed,'job_id':j['job_id'],'state':job_status(j,case['spec']['max_ticks'])}
                                 for j in jobs.values() if j['priority_class']=='urgent' and j['retest']])
        by_arm[arm].update({'urgent_stage_assignments_after_due':len(late_assigned),
                            'urgent_jobs_with_after_due_assignment':len({(v['seed'],v['job_id']) for v in late_assigned}),
                            'urgent_retest_count':len(retest_urgent),
                            'urgent_retest_statuses':dict(Counter(x['state'] for x in retest_urgent)),
                            'after_due_assignment_example':late_assigned[:5],
                            'urgent_due_min':min(x['deadline'] for x in due_stage_checks[arm]),
                            'urgent_due_max':max(x['deadline'] for x in due_stage_checks[arm])})
    return {'by_arm':by_arm,'P0_to_other_job_level_transitions':transitions,'urgent_job_details':due_stage_checks}

def audit_retest(data):
    per_arm={};cross_policy_match=True;exceptions=[]
    for arm in ARMS:
        c=Counter();known_per_seed=[];event_keys=[]
        for seed in data['seeds']:
            case=data['cases'][seed,arm]
            ev=case['event_rows'];dec=[e for e in ev if e['type']=='retest_decision'];created=[e for e in ev if e['type']=='retest_stage_created'];
            c['decisions']+=len(dec);c['created']+=len(created);c['declined']+=sum(not e['needs_retest'] for e in dec)
            c['urgent_created']+=sum(e['job_id'].split('-')[-1].isdigit() and
                                      any(j['job_id']==e['job_id'] and j['priority_class']=='urgent'
                                          for j in case['metrics']['jobs']) for e in created)
            c['decision_before_detection']+=sum(not any(z['type']=='delivered' and z.get('sample_id')==e['job_id']+'::stage3' and
                                              z['tick']<=e['tick'] for z in ev) for e in dec)
            c['candidate_retest_observed']+=sum(d['retest_visible']>0 for d in case['decisions'])
            known_per_seed.append({'seed':seed,'number_decisions':len(dec),'created':len(created),
                                   'retest_results':case['retest_results']})
            event_keys.extend((seed,e['job_id'],e['needs_retest']) for e in dec)
        per_arm[arm]={'counts':dict(c),'cases':known_per_seed}
    for seed in data['seeds']:
        truths=defaultdict(set)
        for arm in ARMS:
            for jid,truth in data['cases'][seed,arm]['retest_results'].items(): truths[jid].add(truth)
        if any(len(vals)>1 for vals in truths.values()):cross_policy_match=False;exceptions.append(seed)
    return {'per_arm':per_arm,'same_job_same_seed_truth_on_observed_decisions':cross_policy_match,
            'contradictions':exceptions,
            'law':'SHA256(private_seed,public_job_id) %100 < synthetic_retest_probability; policy order independent; seed hidden; no verified predictor prior to detection',
            'pre_retest_signal_verdict':'PRE_RETEST_PREDICTION_NOT_IDENTIFIABLE_FROM_CURRENT_OBSERVATION',
            'leakage_limitations':'Cannot identify pre-detection result from public job id or group-level rate; group-level base-rate estimate is possible, not job-specific risk prediction.'}

def audit_oracle(data):
    oracle={x['id']:x for x in data['oracle']};lock={x['id']:x for x in data['lock']['selected']}
    assert set(oracle)==set(lock)
    records=[]
    for oid,seed,tick in FIXED:
        o=oracle[oid];state=lock[oid];case=data['cases'][seed,'P0'];snap=case['history_rows'][tick]
        matching=[tuple((p['robot_id'],p['sample_id']) for p in cand['matching']) for cand in o['candidates']]
        assert len(matching)==len(set(tuple(sorted(p)) for p in matching))
        d=next(d for d in case['decisions'] if d['tick']==tick)
        assert [pairs(p['matching']) for p in d['plans']]==[pairs(p) for p in state['candidate_matchings']]
        # Snapshot was taken after the previous tick, *before* today's release event.
        # Apply only archived actual 'released' events at this tick, never future facts.
        today_released={e['sample_id'] for e in case['event_rows']
                        if e['type']=='released' and e['tick']==tick}
        visible=[]
        for row in snap['samples']:
            c=dict(row)
            if c['id'] in today_released and c['status']=='unreleased':
                c['status']='waiting'
            if c['status']!='unreleased':visible.append(c)
        waiting=[s for s in visible if s['status']=='waiting']
        assert len(waiting)==d['urgent_waiting']+d['routine_waiting'], (oid,'release reconstruction mismatch')
        known={s['id']:s for s in visible}
        present_events=[e for e in case['event_rows'] if e['tick']<tick]
        observed_retests=[e for e in case['event_rows'] if e['type']=='retest_decision' and e['tick']<=tick]
        current_resources={'POOL_CORE':{'capacity':case['spec']['capacity'],'stations':['B','C']},
                           'POOL_OUTER':{'capacity':case['spec']['capacity'],'stations':['A','D']}}
        pri={'urgent':[],'routine':[]}
        for s in waiting:
            jid=s['id'].split('::stage')[0]
            idx=int(jid.split('-')[-1]);key='urgent' if idx<case['spec']['urgent_count'] else 'routine'
            pri[key].append({'id':s['id'],'deadline':s['deadline_tick'],'stage':int(s['id'].split('stage')[-1])})
        candidates=[]
        for i,c in enumerate(o['candidates']):
            jobs={j['job_id']:j for j in c['metrics']['jobs']}
            candidates.append({'index':i,'matching':c['matching'],'urgent':c['metrics']['urgent'],
                'routine':c['metrics']['routine'],'resource_wait_ticks':c['resource_wait_ticks'],
                'end_job_states':[{'job_id':j['job_id'],'priority':j['priority_class'],
                      'status':job_status(j,case['spec']['max_ticks']),
                      'final':j['final_tick'],'deadline':j['deadline_tick'],
                      'retest':j['retest'],'stages':j['stages']}
                      for j in c['metrics']['jobs']],
                'retest_end_count':sum(j['retest'] for j in jobs.values()),
                'event_history_available':False})
        chosen=candidates[o['factual_p0_index']];best=candidates[o['best_within_candidates']]
        p0_events=[e for e in case['event_rows'] if e['tick']>=tick]
        future={'factual_resource_wait_events':sum(e['type']=='resource_wait' for e in p0_events),
                'factual_retest_decisions_after_tick':[{'job':e['job_id'],'tick':e['tick'],'result':e['needs_retest']}
                   for e in p0_events if e['type']=='retest_decision'],
                'factual_retest_created_after_tick':[{'job':e['job_id'],'tick':e['tick']}
                   for e in p0_events if e['type']=='retest_stage_created'],
                'factual_service_event_counts':dict(Counter(e['type'] for e in p0_events if e['type'] in
                                          ('service_started','service_finished','resource_wait','stage_ready')))}
        records.append({'id':oid,'seed':seed,'tick':tick,
            'source_lock':f'selection_lock.json/selected[id={oid}]',
            'source_oracle':f'oracle.json/[id={oid}]',
            'current':{'robots':[{'id':r['id'],'position':r['position'],'task_id':r['task_id'],'phase':r['phase']} for r in snap['robots']],
                       'stations':snap['stations'],'resources':current_resources,
                       'priority_waiting':pri,'visible_samples':visible,
                       'waiting_count':len(waiting),'urgent_count':len(pri['urgent']),'routine_count':len(pri['routine']),
                       'active_robots':sum(r['task_id'] is not None for r in snap['robots']),
                       'known_retest_decisions_so_far':observed_retests,
                       'candidate_resource_schedule':d['solver'].get('schedules'),
                       'observed_busy_estimates':d['solver'].get('observed_busy_estimates'),
                       'visible_capacity_blocked_station_count':d['capacity_blocked'],
                       'visible_device_unavailable_station_count':d['outage_visible']},
            'candidates':candidates,'factual_index':o['factual_p0_index'],
            'best_index':o['best_within_candidates'],'factual_vs_best':{
                 'urgent_incomplete_fraction_delta':chosen['urgent']['incomplete_rate']-best['urgent']['incomplete_rate'],
                 'urgent_late_lower_bound_delta':chosen['urgent']['late_mean_lower_bound']-best['urgent']['late_mean_lower_bound'],
                 'routine_restricted_flow_delta':chosen['routine']['restricted_mean_flow']-best['routine']['restricted_mean_flow'],
                 'capacity_wait_delta':chosen['resource_wait_ticks']-best['resource_wait_ticks']},
            'factual_after_tick_events':future,
            'alternative_event_timelines':'NOT_ARCHIVED; oracle.json contains metrics, no per-candidate event rows, only event/history SHA. Cannot isolate exact retest/resource-wait/outage timing in alternative.',
            'causal_attribution':'One changed assignment, fixed P0 continuation, same exogenous seed. Observed differences are total effect of changed first action; mediator (service, queue, future retest) contributions NOT separately identified.'})
    return records

def audit_service(data):
    return {'current_synthetic_duration':'D = public station.service_ticks + SHA256(seed,sample_id,phase,station_id) % (jitter+1); future seed private',
      'analytic_conditional_hazard':'for integer service effective progress e with not completed, P(D=e+1|D>e,x)=P(D=e+1|x)/P(D>e|x) when denominator nonzero, conditional on public duration/jitter and legitimate history',
      'p2_p3_full_workflow':'P3 convolves independent public discrete jitter for current publicly known future stages and compares expected positive lateness; it omits unknown future retest, queue dynamics and outage. Not a whole-system true oracle.',
      'v4_scope':'V4 predicts only active eligible service next-step completion, not future test retest decision, workstation outage or job-level on-time success probability. Prior independent event Brier Rule 0.06970, B2 0.06369, V4 0.06482.',
      'information_verdict':'NO_VERIFIED_V4_INCREMENT_OVER_ANALYTIC_OR_B2; no new inference performed',
      'analytic_formula_scope':'Cannot use hidden SHA/seed, private service_remaining, future outage plan as online input. A complex analytic whole-system closed form may not exist but this does not establish neural-exclusive conditional information.'}

def json_dump(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf8')

def json_gzip(path,data):
    # Deterministic gzip header; content is purely a function of frozen inputs.
    import io
    serialized=(json.dumps(data,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode('utf8')
    with open(path,'wb') as f:
        with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0,compresslevel=9) as gz:
            gz.write(serialized)

def build(evidence,out):
    start=time.process_time()
    data=read_archive(Path(evidence));out=Path(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('output must be empty')
    out.mkdir(parents=True,exist_ok=True)
    scores=audit_scores(data)
    jobs=audit_jobs(data)
    retest=audit_retest(data)
    oracle=audit_oracle(data)
    service=audit_service(data)
    # Reconcile archived anchor values exactly; no sampling or new simulator is invoked.
    assert scores['counts']['paired_scenarios']==24
    assert scores['counts']['multi_candidate_calls']==330
    assert scores['mask_metrics'].get('top_choice_change',0)==0
    assert jobs['by_arm']['P0']['counts']['on_time']==jobs['by_arm']['P1']['counts']['on_time']==22
    assert data['summary']['coverage']['urgent_vs_routine']==137
    assert len(oracle)==3
    inputs={k:{'sha256':v,'path':'evidence/'+k} for k,v in SOURCE_SHA.items()}
    output={'source_head':'9aa554bc541e5a68cb74089feeecbbe62ada1aa0',
            'input_provenance':inputs,'protocol_sha256':data['summary']['protocol_sha256'],
            'read_only':True,'simulator_calls':0,'model_calls':0,'training_calls':0,
            'original_scenarios':24,'original_arms':4,'new_trajectories':0}
    # Keep per-tick classifications, source locators and hashes; full candidate
    # score rows remain immutably archived in SOURCE evidence/full_raw.json.gz.
    compact_scores={**scores,'state_details':[{k:row[k] for k in
        ('seed','tick','class','candidate_count','missed','eligible_ids',
         'p2_rank','p3_rank','numeric_changed_indices',
         'within_best_coverage_changed','source')} for row in scores['state_details']]}
    compact_jobs={**jobs,'urgent_job_details':{'P0':jobs['urgent_job_details']['P0'],
                                             'P1':jobs['urgent_job_details']['P1']}}
    compact_retest={**retest,'per_arm':{arm:{'counts':x['counts']}
                                      for arm,x in retest['per_arm'].items()}}
    compact_oracle=[]
    for row in oracle:
        cc=[]
        for cand in row['candidates']:
            cc.append({**cand,'end_job_states':[{k:job[k] for k in
                  ('job_id','priority','status','final','deadline','retest')}
                  for job in cand['end_job_states']]})
        compact_oracle.append({**row,'candidates':cc,
             'current':{k:v for k,v in row['current'].items()
                        if k not in ('candidate_resource_schedule','visible_samples')}})
    json_gzip(out/'score_masking.json.gz',compact_scores)
    json_gzip(out/'job_transitions.json.gz',compact_jobs)
    json_dump(out/'retest_information.json',compact_retest)
    json_gzip(out/'three_fixed_oracles.json.gz',compact_oracle)
    json_dump(out/'service_information.json',service)
    output['cpu_process_seconds']=time.process_time()-start
    output['audit_source_sha256']=sha_file(__file__)
    output['score_masking_counts']=scores['counts']
    output['score_numeric_masks']=scores['mask_metrics']
    output['urgent_job_states']={k:v['counts'] for k,v in jobs['by_arm'].items()}
    output['fixed_oracle_ids']=[x['id'] for x in oracle]
    output['factual_retest_decisions']=retest['per_arm']['P0']['counts']['decisions']
    output['factual_retest_created']=retest['per_arm']['P0']['counts']['created']
    output['retest_information_verdict']=retest['pre_retest_signal_verdict']
    output['prediction_information_verdict']='BLOCKED_NO_INCREMENTAL_CONDITIONAL_SIGNAL_DEMONSTRATED'
    json_dump(out/'summary.json',output)
    manifest={p.name:sha_file(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='sha256.json'}
    json_dump(out/'sha256.json',manifest)
    return output

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(build(args.evidence,args.out),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
