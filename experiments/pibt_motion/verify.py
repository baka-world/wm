"""Check frozen historical reproduction, safety and actual native execution."""
from __future__ import annotations
import ast
import gzip
import json
from pathlib import Path
import re
import sys


def verify(root):
    root=Path(root)
    legacy=json.loads((root/'hospital-legacy.json').read_text())
    pibt=json.loads((root/'hospital-pibt.json').read_text())
    archive=Path('examples/evidence/business-capacity-20261109/public-hospital-three-robot-pilot.log.gz')
    expected={}
    for line in gzip.decompress(archive.read_bytes()).decode().splitlines():
        match=re.fullmatch(r'(tartu|clinic) (\{.*?\}) (\{.*\})',line)
        if not match:raise ValueError('cannot parse archived PR #12 pilot')
        expected[match[1]]=ast.literal_eval(match[2])
    for a,b in zip(legacy['cases'],pibt['cases']):
        assert a['layout']==b['layout'] and a['config']==b['config']
        assert a['summary']==expected[a['layout']], 'legacy pilot changed relative to archive'
        assert a['topology']==b['topology'] and a['initial_robots']==b['initial_robots']
        assert a['station_configuration']==b['station_configuration']
        assert b['pibt_calls']>0
        for case in (a,b):
            assert not case['resources']['violations'] and not case['lifecycle_violations']
            assert not case['duplicate_assignments']
            assert not any(case['safety'][k] for k in ('vertex_collisions','edge_swaps','invalid_edges'))
    joint=json.loads((root/'joint-diagnostic.json').read_text())
    assert len(joint)==2 and all(row['found_witness'] for row in joint)
    alignment=json.loads((root/'model-alignment.json').read_text())
    for name in ('rule','pets','structured'):
        parity=alignment['parity'][name];loop=alignment['closed_loop'][name]
        assert parity['factual_transitions']>0 and parity['current_paused_robot_observations']>0
        assert parity['matching_one_step_robot_configurations']==parity['factual_transitions']
        assert parity['native_prediction_calls']>0 and loop['prediction_native_calls']>0
        assert loop['status']=='completed' and loop['execution_native_calls']>0
        assert not loop['model_fallbacks'] and not loop['capacity_violations'] and not loop['discrete_safety_violations']
    assert alignment['closed_loop']['pets']['pets_forward_calls']>0
    assert alignment['closed_loop']['structured']['structured_service_calls']>0
    data=json.loads((root/'grid-benchmark.json').read_text())
    assert len(data['cases'])==20
    assert sum(x['robots']==3 for x in data['cases'])==16
    assert sum(x['robots']==5 for x in data['cases'])==4
    assert any(c['methods']['pibt']['jobs_completed']==c['methods']['pibt']['jobs']
               for c in data['cases'] if c['robots']==3)
    for case in data['cases']:
        for name,m in case['methods'].items():
            assert m['jobs_completed']+m['jobs_unfinished']==m['jobs']
            assert not m['resource_violations'] and not m['lifecycle_violations']
            assert not m['duplicate_assignments']
            assert not any(m['safety'][k] for k in ('vertex_collisions','edge_swaps','invalid_edges'))
            if name=='pibt':assert m['pibt_native_calls']>0
    print('Verified archived hospital replay, both bounded joint witnesses, native PIBT calls, '
          '16 original 3-robot and four 5-robot cases, lifecycle and discrete safety')


if __name__=='__main__':verify(sys.argv[1])
