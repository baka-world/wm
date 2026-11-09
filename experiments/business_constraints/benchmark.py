"""Load pinned public MAPF maps and fixed synthetic business assumptions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

from business_transport.scenario import compile_scenario
from lab_transport.domain import Grid,RobotSpec,Scenario,Station

CONFIG=Path('business_transport/scenarios/benchmark.json')


def read_config():
    return json.loads(CONFIG.read_text(encoding='utf-8'))


def read_map(layout):
    path=Path(layout['map']);raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=layout['sha256']:
        raise ValueError('public MAPF map checksum mismatch')
    lines=raw.decode('ascii').splitlines()
    if len(lines)<5 or lines[0]!='type octile' or lines[3]!='map':raise ValueError('invalid MAPF map')
    height=int(lines[1].split()[1]);width=int(lines[2].split()[1]);rows=lines[4:]
    if len(rows)!=height or any(len(r)!=width for r in rows):raise ValueError('invalid MAPF dimensions')
    if any(c not in ('.','@','T','O') for row in rows for c in row):raise ValueError('unsupported map symbol')
    return Grid(width,height,frozenset((x,y) for y,row in enumerate(rows) for x,c in enumerate(row) if c!='.'))


def make_case(config,layout_name,seed,regime_name):
    layout=config['layouts'][layout_name];regime=config['regimes'][regime_name];grid=read_map(layout)
    stations={}
    for sid,xy in layout['stations'].items():
        if not grid.contains(tuple(xy)):raise ValueError('station not on map')
        outage=regime.get('outage')
        outages=((outage['start'],outage['end']),) if outage and outage['station']==sid else ()
        stations[sid]=Station(sid,tuple(xy),config['station_service_ticks'][sid],
            config['station_prior_ticks'][sid],outages,config['duration_jitter'])
    robots=tuple(RobotSpec(f'R{i+1}',tuple(p),tuple(h)) for i,(p,h) in enumerate(
        zip(layout['robot_positions'],layout['robot_homes'])))
    if len({r.position for r in robots})!=len(robots) or any(not grid.contains(r.position) for r in robots):
        raise ValueError('invalid robot positions')
    base=Scenario(f'{layout_name} public benchmark',grid,stations,robots,(),config['duration_jitter'],config['max_ticks'],
       metadata={'source':'Moving AI map via LaCAM3, pinned by hash','assumptions':config['scope']})
    first=config['simple_first_two'] if regime['single_stage_only'] else config['multistage_first_two']
    allowed=['single_0','single_1'] if regime['single_stage_only'] else config['later_templates']
    rng=random.Random(seed)
    types=list(first)+[rng.choice(allowed) for _ in config['release_ticks'][2:]]
    jobs=[{'id':f'S{i+1:02}','workflow':name,'release_tick':release,'deadline_tick':None}
          for i,(name,release) in enumerate(zip(types,config['release_ticks']))]
    business={'name':f'{layout_name} {regime_name} seed={seed} (synthetic roles)',
        'resources':[{'id':'collection','stations':['A0','A1'],'capacity':regime['capacity_A']},
                     {'id':'processing','stations':['B0','B1'],'capacity':regime['capacity_B']}],
        'workflows':config['workflows'],'jobs':jobs,'max_ticks':config['max_ticks']}
    return compile_scenario(base,business),business


def audit(config):
    signatures={}
    for profile,maps in config['seeds'].items():
        for layout,seeds in maps.items():
            for seed in seeds:
                # Serial, parallel and outage share the same task table by design.
                for regime in ('single_compatible','workflow_serial'):
                    _,business=make_case(config,layout,seed,regime)
                    key=(layout,tuple((j['workflow'],j['release_tick']) for j in business['jobs']))
                    signatures.setdefault(key,[]).append((profile,seed,regime))
    conflicts=[v for v in signatures.values() if len(v)>1]
    if conflicts:raise ValueError(f'duplicate new task patterns: {conflicts}')
    prior_files=('examples/evidence/service-dynamics-20261109/dataset-manifest.json',
                 'examples/evidence/service-decision-value-20261109/scenario-config.json',
                 'examples/evidence/paper-reuse-integration-20261109/scenario-config.json')
    prior=set()
    for path in prior_files:
        prior.update(json.loads(Path(path).read_text(encoding='utf-8')).get('source_sha256',{}).values())
    if any(layout['sha256'] in prior for layout in config['layouts'].values()):
        raise ValueError('benchmark map hash overlaps previous data source')
    return {'unique_layout_job_patterns':len(signatures),'cross_seed_duplicates':0,
            'previous_source_hash_overlap':0,
            'old_pr8_pr11_note':'PR #8–#11 used hospital/clinic maps; these exact benchmark maps differ.'}
