"""Structural checks for CPU experiment artifacts, not assertions of positive results."""
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
expected_profile = sys.argv[2]
result = json.loads((root / 'decision-value.json').read_text(encoding='utf-8'))
summary = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
assert result['profile'] == summary['profile'] == expected_profile
assert len(result['cases']) == summary['case_count'] == 12
assert result['horizon'] == 20 and result['candidate_budget'] == 8
assert result['prior_split_schedule_audit']['cross_split_task_duplicate_groups'] == 0
assert result['transformed_schedule_audit']['cross_split_duplicate_groups'] == 0
assert summary['decision_count'] >= 2 and summary['comparable_decisions'] >= 2
assert summary['prediction_errors'] == 0
assert summary['discrete_safety_violations'] == 0
assert result['cpu_seconds'] < 180, 'pilot exceeds bounded CPU budget'
for case in result['cases']:
    assert set(case['policies']) == {'nearest', 'world', 'rule', 'rollout', 'heuristic', 'learned'}
    assert len(case['config']['samples']) == 6
    for entry in case['policies'].values():
        assert 0 <= entry['completed'] <= entry['total'] == 6
    for decision in case['offline_decisions']:
        assert 2 <= decision['candidate_evaluated'] <= 8
        assert decision['candidate_count'] >= decision['candidate_evaluated']
        assert len(decision['candidates']) == decision['candidate_evaluated']
        assert decision['candidates'][0]['batch'] == []
        initial = decision['initial_observation']
        assert initial['tick'] == decision['tick']
        known = {s['id'] for s in initial['samples']}
        for candidate in decision['candidates']:
            assert candidate['full_episode']['discrete_safety_violations'] == 0
            if candidate['reference_complete']:
                assert set(candidate['reference_states']) == {'1', '5', '20'}
                for prediction in candidate['predictions'].values():
                    assert set(prediction['states']) == {'1', '5', '20'}
                    assert all({s['id'] for s in state['samples']} == known
                               for state in prediction['states'].values())
        for index in decision['choices'].values():
            assert 0 <= index < decision['candidate_evaluated']
print(f"validated {expected_profile}: {len(result['cases'])} cases, {summary['comparable_decisions']} comparable decisions; no efficacy claim")
