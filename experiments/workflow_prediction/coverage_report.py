"""Derive auditable multi-stage window coverage from recorded visible traces."""
from collections import Counter
import json
from pathlib import Path
import sys

from experiments.workflow_prediction.config_audit import CONFIG


def key(sample):return sample.get('job_id') or sample['id']


def report_case(case,horizon):
    rows=case['factual_visible_transitions'];stride=json.loads(CONFIG.read_text())['prediction_window_stride']
    routes=case['assumptions']['workflows']
    counts=Counter(candidate_windows=max(0,len(rows)-horizon+1))
    selected_ticks={item['start_tick'] for item in case['state_prediction']['horizons'][str(horizon)]['dynamic_windows']}
    for index in range(max(0,len(rows)-horizon+1)):
        window=rows[index:index+horizon];initial=window[0]['observation']
        if any(row['assignments'] for row in window[1:]):continue
        jobs={key(s) for s in initial['samples']}
        if any(key(s) not in jobs for row in window for s in row['next_observation']['samples']):continue
        device={s['id']:s.get('device_available',s['available']) for s in initial['stations']}
        if any({s['id']:s.get('device_available',s['available']) for s in row['next_observation']['stations']}!=device
               for row in window):continue
        counts['eligible']+=1
        grouped={}
        for sample in initial['samples']:grouped.setdefault(key(sample),[]).append(sample)
        active_multistage=any(len(routes.get(samples[0].get('workflow'),[]))>1 and not any(
            s.get('stage_index',0)==len(routes[samples[0]['workflow']])-1 and s['status']=='completed' for s in samples)
            for samples in grouped.values())
        successor=False
        for row in window:
            old={s['id'] for s in row['observation']['samples']}
            successor|=any(s['id'] not in old and key(s) in jobs and s.get('stage_index',0)>0
                           for s in row['next_observation']['samples'])
        counts['eligible_active_multistage']+=active_multistage
        counts['eligible_with_successor']+=successor
        selected=index%stride==0 or successor
        if selected:
            assert initial['tick'] in selected_ticks
            counts['selected']+=1
            counts['selected_active_multistage']+=active_multistage
            counts['selected_with_successor']+=successor
    original=case['state_prediction']['horizons'][str(horizon)]['dynamic_coverage']
    for field in ('candidate_windows','eligible','selected','selected_with_successor'):
        assert counts[field]==original.get(field,0),(case['layout'],case['seed'],horizon,field)
    return dict(counts)


def report(source,destination):
    data=json.loads(Path(source).read_text());records=[];total={}
    for case in data['cases']:
        horizons={str(h):report_case(case,h) for h in (1,3,5)}
        records.append({k:case[k] for k in ('layout','seed','regime')}|{'horizons':horizons})
        for h,counts in horizons.items():total[h]=dict(Counter(total.get(h,{}))+Counter(counts))
    output=Path(destination)
    if output.exists():raise FileExistsError('refusing to overwrite coverage')
    output.write_text(json.dumps({'profile':data['profile'],'scope':'window counts overlap; not independent samples',
                                  'totals':total,'scenarios':records},ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':report(sys.argv[1],sys.argv[2])
