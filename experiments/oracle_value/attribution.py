"""Decision-relevant first-particle event timing and state mismatch attribution.

Uses original frozen candidate traces and model forecasts; never changes actions,
models or scores. Error counts are conditional on the five-step observation window.
"""
from collections import Counter
import gzip
import json
from pathlib import Path
import sys


METHODS=('Rule-5','PETS-5','Structured-5')


def read(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:
        return json.load(f)


def events(steps,kind):
    value={}
    for row in steps:
        for event in row[kind]:
            key=tuple(event) if isinstance(event,list) else event
            if key in value:raise ValueError('one event appeared twice in the same short horizon')
            value[key]=row['tick']
    return value


def summarize(raw):
    stats={}
    late=Counter()
    selected_examples=[]
    decisions=0
    for case in raw['cases']:
        for d in case['decisions']:
            decisions+=1
            tick=d['tick']
            for b in d['branches']:
                if 'error' in b:continue
                for e in b['events']:
                    if tick+5 < e['tick'] <= tick+20 and e['type'] in ('service_finished','stage_ready','resource_wait'):
                        late[e['type']]+=1
            for name in METHODS:
                method=d['methods'][name]
                if method.get('choice') is None:
                    continue
                counter=stats.setdefault(name,Counter())
                for candidate in method.get('candidate_diagnostics',[]):
                    if 'error' in candidate:
                        counter['failed_diagnostic_branch']+=1
                        continue
                    counter['candidate_comparisons']+=1
                    steps=candidate['steps']
                    counter['steps']+=len(steps)
                    counter.update(candidate['first_particle_mismatches'])
                    for real_key,pred_key,term in (
                        ('service_true','service_predicted','service'),
                        ('release_true','release_predicted','stage_release')
                    ):
                        actual=events(steps,real_key)
                        predicted=events(steps,pred_key)
                        overlap=actual.keys()&predicted.keys()
                        counter[term+'_true_events']+=len(actual)
                        counter[term+'_predicted_events']+=len(predicted)
                        counter[term+'_matched_events']+=len(overlap)
                        counter[term+'_missed_events']+=len(actual.keys()-predicted.keys())
                        counter[term+'_spurious_events']+=len(predicted.keys()-actual.keys())
                        counter[term+'_timing_abs_error_ticks']+=sum(
                            abs(actual[e]-predicted[e]) for e in overlap)
                        counter[term+'_timing_nonzero']+=sum(
                            actual[e]!=predicted[e] for e in overlap)
    return {'scope':'first prediction particle, five-step event-window alignment only; nested candidates are NOT independent samples',
            'decisions':decisions,
            'models':{n:dict(c) for n,c in stats.items()},
            'outside_5_inside_20_events':dict(late),
            'caveats':['A missing short-window event is NOT an absolute time error.',
                       'Path and station mismatches do not imply service prediction caused them.',
                       'All-model selections and full outcomes are in original confirmation data.']}


def run(raw,out):
    summary=summarize(read(raw))
    Path(out).write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return summary


if __name__=='__main__':
    run(*sys.argv[1:3])
