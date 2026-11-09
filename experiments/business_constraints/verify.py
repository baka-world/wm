"""Verify safety, scenario separation, observability and model participation."""
from __future__ import annotations
import json
from pathlib import Path
import sys


def verify(root):
    root=Path(root)
    for profile,expected in [('development',8),('confirmation',16)]:
        data=json.loads((root/profile/'raw.json').read_text(encoding='utf-8'))
        assert data['profile']==profile and len(data['cases'])==expected
        assert data['scenario_audit']['cross_seed_duplicates']==0
        assert data['scenario_audit']['previous_source_hash_overlap']==0
        assert data['scenario_audit']['planner_seed_equals_environment'] is False
        for case in data['cases']:
            for name,m in case['methods'].items():
                assert m['jobs_completed']==m['jobs'] and m['status']=='completed'
                assert m['resource']['violations']==[] and m['discrete_safety_violations']==0
                assert m['duplicate_assignments']==0 and m['lifecycle_violations']==[]
                assert m['solver_fallbacks']==0 and m['solver_truncations']==0 and m['model_fallbacks']==0
                assert len(m['decision_wall_ms'])==m['decision_count']
            for decision in case['same_state']:
                obs=decision['observed_state']
                assert all(s['release_tick']<=obs['tick'] for s in obs['samples'])
                assert not decision['solver_truncated']
                assert all(b['status']=='completed' and b['flow'] is not None for b in decision['branches'])
        assert data['summary']['methods']['cpsat_pets']['pets_forward_calls']>0
        assert data['summary']['methods']['cpsat_structured']['structured_service_calls']>0
        assert data['summary']['same_state_decisions']>0
    full=json.loads((root/'full-state-confirmation.json').read_text(encoding='utf-8'))
    assert full['summary']['factual_episodes']==64
    assert full['summary']['same_state_decisions']==len(full['records'])
    for row in full['records']:
        state=row['observed_state']
        assert all(s['release_tick']<=state['tick'] for s in state['samples'])
        choices={entry['choice'] for entry in row['forecasts'].values()}
        if len(choices)>1:
            assert set(row['branches'])=={str(choice) for choice in choices}
            assert all(b['status']=='completed' for b in row['branches'].values())
    print('Verified complete staged jobs, capacity, lifecycle, current-only observations, '
          'all three model interfaces and paired candidate replays, including exploratory audit')


if __name__=='__main__':verify(sys.argv[1])
