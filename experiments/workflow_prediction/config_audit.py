"""Audit frozen task signatures before using the confirmation cases."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.business_constraints.benchmark import make_case,read_config

CONFIG=Path('experiments/workflow_prediction/config.json')


def signature(config,layout,seed,regime):
    _,business=make_case(config,layout,seed,regime)
    return tuple((job['workflow'],job['release_tick']) for job in business['jobs'])


def audit():
    fixed=json.loads(CONFIG.read_text());old=read_config();prior={key:set() for key in old['layouts']}
    historic=[]
    for mapping in list(old['seeds'].values())+[old['previous_confirmation_pilot'],old['previous_shared_rng_confirmation']]:
        for layout,seeds in mapping.items():
            for seed in seeds:
                for regime in old['regimes']:
                    prior[layout].add(signature(old,layout,seed,regime))
                    historic.append((layout,seed,regime))
    clusters={};records=[]
    for profile in ('development','confirmation'):
        for group in fixed[profile]:
            layout,seed=group['layout'],group['seed']
            for regime in group['regimes']:
                sig=signature(old,layout,seed,regime)
                key=(layout,seed,regime)
                if key in clusters:raise ValueError('duplicate configured scene')
                clusters[key]=profile
                duplicate=sig in prior[layout]
                if profile=='confirmation' and duplicate:
                    raise ValueError('confirmation task signature overlaps prior data')
                records.append({'profile':profile,'layout':layout,'seed':seed,'regime':regime,
                                'signature':sig,'overlaps_pr12_history':duplicate})
    # Same task table across outage/capacity variants within a seed is intended.
    for a in records:
        for b in records:
            if (a['profile']=='development' and b['profile']=='confirmation' and
                    a['layout']==b['layout'] and a['signature']==b['signature']):
                raise ValueError('confirmation task table duplicates development')
    return {'config_sha256':hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
            'historic_scene_entries_checked':len(historic),'scenes':records,
            'confirmation_cross_seed_task_overlap_with_pr12':0,
            'map_note':'both public maps were used before; this is new workload/seed confirmation, not unseen-map validation',
            'prior_pr8_pr11_note':'PR12 benchmark audit records these map hashes differ from PR8–11 hospital/clinic source maps'}


if __name__=='__main__':
    print(json.dumps(audit(),ensure_ascii=False,indent=2))
