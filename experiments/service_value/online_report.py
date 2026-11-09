"""Paired online comparisons against the matching predictor and score."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from lab_transport.decision_value_round2 import bootstrap_paired

PAIRS = {
    'rule_new_minus_rule_old':('rule_workload','rule_original'),
    'hybrid_new_minus_hybrid_old':('hybrid_workload','hybrid_original'),
    'hybrid_old_minus_rule_old':('hybrid_original','rule_original'),
    'hybrid_new_minus_rule_new':('hybrid_workload','rule_workload'),
}


def summarize(data):
    by_regime = {}
    for regime in ('all','normal','burst','outage_processing','outage_arrival'):
        cases = [c for c in data['cases'] if regime == 'all' or c['regime'] == regime]
        pairs = {}
        for label,(left,right) in PAIRS.items():
            values = [(c['layout'],c['seed'],c['methods'][left]['mean_flow_complete_episode'] -
                       c['methods'][right]['mean_flow_complete_episode']) for c in cases
                      if c['methods'][left]['mean_flow_complete_episode'] is not None
                      and c['methods'][right]['mean_flow_complete_episode'] is not None]
            pairs[label] = {'cases':len(values),'difference':bootstrap_paired(values),
                            'wins':sum(v < -1e-9 for _,_,v in values),
                            'losses':sum(v > 1e-9 for _,_,v in values),
                            'ties':sum(abs(v) <= 1e-9 for _,_,v in values)}
        by_regime[regime] = pairs
    return {'profile':data['profile'], 'scope':'Paired full synthetic episodes, clustered by layout/seed.',
            'paired':by_regime}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'refusing to overwrite: {args.output}')
    result = summarize(json.loads(args.input.read_text(encoding='utf-8')))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
