"""Freeze new independent synthetic map/seed/incident configurations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.service_value.counterfactual import SEEDS as PR10_SEEDS
from lab_transport.decision_value import case_config
from lab_transport.decision_value_round2 import burst_case, make_round2_cases
from lab_transport.dynamics_revision import SPLITS_V2
from lab_transport.multisite_experiment import make_case
from lab_transport.service_fault_data import select_seeds,source_bases
from lab_transport.training_data import _task_hash

SEEDS={'development':{'tartu':(41000,41002),'clinic':(42000,42001)},
       'confirmation':{'tartu':(49002,49003),'clinic':(50000,50001)}}
PREVIOUS_PILOT={'tartu':(45001,45004),'clinic':(46000,46002)}
REGIMES=('normal','burst','outage_processing','outage_arrival')


def audit(bases):
    old = set()
    def signatures(layout,seed):
        return {_task_hash(make_case(bases[layout],layout,seed,'normal'),layout),
                _task_hash(burst_case(bases[layout],layout,seed),layout)}
    for _,(layout,seeds) in SPLITS_V2.items():
        for seed in seeds:
            old.add(_task_hash(make_case(bases[layout],layout,seed,'normal'),layout))
    for _,(layout,seeds) in select_seeds(bases)[0].items():
        for seed in seeds: old.update(signatures(layout,seed))
    for maps in PR10_SEEDS.values():
        for layout,seeds in maps.items():
            for seed in seeds: old.update(signatures(layout,seed))
    for layout,seeds in PREVIOUS_PILOT.items():
        for seed in seeds: old.update(signatures(layout,seed))
    seen=set()
    for profile,maps in SEEDS.items():
        for layout,seeds in maps.items():
            for seed in seeds:
                sigs=signatures(layout,seed)
                if len(sigs)!=2 or sigs.intersection(old) or sigs.intersection(seen):
                    raise ValueError('independent task schedule signature overlaps previous or new cases')
                seen.update(sigs)
    return {'previous_unique_patterns':len(old),'new_unique_patterns':len(seen),
            'cross_group_duplicates':0,'method':'layout plus pickup/dropoff/release task signature; before evaluation'}


def build():
    bases,sources=source_bases()
    proof=audit(bases)
    cases=[]
    for profile,maps in SEEDS.items():
        for layout,seeds in maps.items():
            for seed in seeds:
                options,design=make_round2_cases(bases[layout],layout,seed)
                for regime in REGIMES:
                    case,pauses,incident=options[regime]
                    cases.append({'profile':profile,'layout':layout,'seed':seed,'regime':regime,'design':design,
                                  'incident':incident,'config':case_config(case,pauses)})
    return {'scope':'fixed synthetic scenarios only; config contains offline future events and is never model input',
            'seeds':SEEDS,'previous_pilot':PREVIOUS_PILOT,'regimes':REGIMES,'horizon':5,'particles':3,
            'solver_time_limit_seconds':.5,'solver_deterministic_time_limit':.05,
            'solver_candidates':4,'schedule_audit':proof,'source_sha256':sources,'cases':cases}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('refuse to overwrite scenario config')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    data=build()
    args.output.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Frozen',len(data['cases']),'scenarios;',data['schedule_audit'])


if __name__=='__main__':main()
