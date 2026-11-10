"""Explain frozen fixed-candidate score changes without retuning the score."""
from collections import Counter
import json
from pathlib import Path
import sys


def run(source,destination):
    data=json.loads(Path(source).read_text());counts=Counter();records=[]
    for case in data['cases']:
        for decision in case['same_state']:
            forecasts=decision['forecasts']
            old=forecasts['old_transition_old_score']['evaluations']
            rescored=forecasts['old_transition_new_score']['evaluations']
            new=forecasts['new_transition_new_score']['evaluations']
            deltas=[round(b['score'][0]-a['score'][0],8) for a,b in zip(old,rescored)]
            dynamics_different=any(a['score']!=b['score'] for a,b in zip(rescored,new))
            counts['decision_states']+=1
            counts['score_primary_shift_constant_across_candidates']+=len(set(deltas))<=1
            counts['score_primary_shift_varies_across_candidates']+=len(set(deltas))>1
            counts['new_transition_changes_any_candidate_score']+=dynamics_different
            records.append({'layout':case['layout'],'seed':case['seed'],'regime':case['regime'],
                'tick':decision['tick'],'candidate_count':len(decision['plans']),
                'scoring_primary_deltas':deltas,'new_transition_changes_any_score':dynamics_different,
                'choices':{key:value['choice'] for key,value in forecasts.items()},
                'counterfactual_restricted_mean_flows':[b['restricted_mean_flow'] for b in decision['branches']]})
    destination=Path(destination)
    if destination.exists():raise FileExistsError('refusing to overwrite diagnostics')
    destination.write_text(json.dumps({'profile':data['profile'],'counts':dict(counts),'states':records,
        'note':'decision states are nested within scenarios and are not independent statistical samples'},
        ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':run(sys.argv[1],sys.argv[2])
