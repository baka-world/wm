"""Verify frozen PR13 replay, offline gate, safety and prediction branches."""
import gzip
import json
from pathlib import Path
import sys


ARCHIVE = Path('examples/evidence/pibt-motion-20261109')


def read_archive(name):
    with gzip.open(ARCHIVE / name, 'rt') as handle:
        return json.load(handle)


def verify(root):
    root = Path(root)
    gate = json.loads((root / 'clinic-gate.json').read_text())
    assert len(gate['states']) == 5
    assert all(row['result']['status'] == 'solved' and len(row['result']['path']) > 1
               for row in gate['states'])
    data = json.loads((root / 'paired.json').read_text())
    cases = data['cases']
    assert len(cases) == 22
    assert sum(case['regime'] == 'hospital_pilot' for case in cases) == 2
    assert sum(case['regime'] != 'hospital_pilot' and case['robots'] == 3 for case in cases) == 16
    assert sum(case['robots'] == 5 for case in cases) == 4
    for case in cases:
        assert set(case['methods']) == {'legacy', 'pibt', 'hybrid'}
        for name, method in case['methods'].items():
            assert method['jobs_completed'] + method['jobs_unfinished'] == method['jobs']
            assert not any(method['safety'][k] for k in ('vertex_collisions', 'edge_swaps', 'invalid_edges'))
            assert not any(method[k] for k in ('locked_moves', 'locked_crossings', 'resource_violations',
                                               'duplicate_assignments', 'lifecycle_violations'))
            if name != 'legacy':
                assert method['pibt_native_calls'] > 0
    tartu = next(c for c in cases if c['layout'] == 'tartu')
    assert not tartu['lacam3_topology_support']['supported']
    assert tartu['lacam3_topology_support']['one_way_edges'] > 0
    assert tartu['methods']['hybrid']['official_solver_calls'] == 0
    for method in ('legacy', 'pibt'):
        old = read_archive(f'hospital-{method}.json.gz')['cases']
        for prior in old:
            current = next(c for c in cases if c['layout'] == prior['layout'] and
                           c['regime'] == 'hospital_pilot')['methods'][method]
            for key in ('history', 'events', 'job_outcomes'):
                assert current[key] == prior[key], (prior['layout'], method, key)
    old_grid = read_archive('grid-benchmark.json.gz')['cases']
    assert len(old_grid) == 20
    for prior in old_grid:
        identifier = tuple(prior[k] for k in ('layout', 'seed', 'regime', 'robots'))
        current = next(c for c in cases if tuple(c[k] for k in ('layout', 'seed', 'regime', 'robots')) == identifier)
        for method in ('legacy', 'pibt'):
            for key in ('history', 'events', 'job_outcomes'):
                assert current['methods'][method][key] == prior['methods'][method][key], (identifier, method, key)
    alignment = json.loads((root / 'model-alignment.json').read_text())
    assert alignment['factual_status'] == 'completed' and alignment['actual_cache_invalidations']
    for name in ('rule', 'pets', 'structured'):
        model = alignment['models'][name]
        assert model['one_step_positions_match'] == model['factual_transitions']
        assert model['conditional_motion_match'] == model['matching_service_state_transitions']
        assert model['conditional_snapshot_match'] == model['matching_service_state_transitions']
        assert all(model['cached_multistep_positions_match']) and model['branch_isolation']
        loop = alignment['cp_sat_closed_loop'][name]
        assert loop['status'] == 'completed' and loop['stages_completed'] == loop['stages']
        assert loop['prediction_pibt_calls'] > 0 and loop['execution_pibt_calls'] > 0
        assert not loop['model_fallbacks']
    assert alignment['cp_sat_closed_loop']['pets']['pets_forward_calls'] > 0
    assert alignment['cp_sat_closed_loop']['structured']['structured_service_calls'] > 0
    ablation = json.loads((root / 'ablation.json').read_text())
    assert ablation['layout'] == 'clinic' and ablation['outcome']['jobs'] > 0
    print('Verified five official full plans, 44 historical replays, safety, directed-map restriction, '
          'independent model snapshots and real CP-SAT/model calls.')


if __name__ == '__main__':
    verify(sys.argv[1])
