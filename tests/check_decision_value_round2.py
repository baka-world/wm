"""Predeclared validity gate. Does not require a favorable model result."""
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
profile = sys.argv[2]
d = json.loads((root / 'round2.json').read_text(encoding='utf-8'))
s = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
assert d['profile'] == s['profile'] == profile
assert len(d['cases']) == (10 if profile == 'pilot' else 40)
assert d['schedule_audit']['cross_split_duplicate_groups'] == 0
assert d['horizon'] == 20 and d['candidate_budget'] == 8
assert d['cpu_seconds'] < (240 if profile == 'pilot' else 600)
assert s['comparable_decisions'] > 0
assert s['paired_learned_minus_heuristic_flow']['clusters'] == (2 if profile == 'pilot' else 8)
assert s['paired_true_short_minus_heuristic_flow']['clusters'] == (2 if profile == 'pilot' else 8)
assert s['reference_vs_rule_decision_counts']['full_flow_reference']['worse'] == 0
ordinary = near = 0
meaningful = 0

def short_primary(components, initial):
    deadlines = {s['id']: s['deadline_tick'] for s in initial['samples']}
    ids = components['sample_ids']
    assert ids == sorted(deadlines)
    assert len(components['steps']) == 20
    score = 0
    for tick, statuses in components['steps']:
        assert len(statuses) == len(ids)
        for sid, status in zip(ids, statuses):
            if status != 'completed':
                score += 1 + 4 * int(deadlines[sid] is not None and tick > deadlines[sid])
    return score

for c in d['cases']:
    assert set(c['policies']) == {'nearest', 'world', 'rule', 'rollout', 'heuristic', 'learned'}
    for policy in c['policies'].values():
        assert policy['metrics']['discrete_safety_violations'] == 0
        trace = policy['post_run_trace']
        assert len(trace['history']) == policy['metrics']['summary']['ticks'] + 1
        assert 'never policy/training input' in trace['scope']
        incident = policy['incident']
        if incident:
            assert policy['paired_no_fault_trace'] is not None
            assert incident['service_wait_ticks'] == policy['metrics']['summary']['station_unavailable_wait_ticks']
    for x in c['decisions']:
        assert x['candidate_evaluated'] <= 8
        assert x['candidate_count'] >= x['candidate_evaluated'] >= 2
        assert all(y['discrete_safety_violations'] == 0 for y in x['candidates'])
        for candidate in x['candidates']:
            samples = candidate['full_sample_outcomes']
            recomputed = sum((s['completed_tick'] if s['completed_tick'] is not None else c['config']['max_ticks'])
                             - s['release_tick'] for s in samples) / len(samples)
            assert abs(recomputed - candidate['full_restricted_mean_flow_ticks']) < 1e-9
            if candidate['short_reference_complete']:
                assert short_primary(candidate['true_short_components'], x['initial_observation']) == candidate['true_short_score'][0]
                for prediction in candidate['predictions'].values():
                    assert short_primary(prediction['short_components'], x['initial_observation']) == prediction['short_score'][0]
        if not x['fully_comparable']:
            continue
        best = x['choices']['full_flow_reference']
        assert x['candidates'][best]['full_restricted_mean_flow_ticks'] == min(
            y['full_restricted_mean_flow_ticks'] for y in x['candidates'])
        ordinary += 'ordinary' in x['sampling_labels'].split(',')
        near += any(y in x['sampling_labels'].split(',') for y in ('before','during','after'))
        meaningful += x['full_flow_gap_ticks'] >= d['meaningful_mean_flow_gap_ticks']
assert ordinary >= 2 and near >= 2 and meaningful >= 1
if profile == 'pilot':
    for layout in ('tartu', 'clinic'):
        processing = next(c for c in d['cases'] if c['layout'] == layout and c['regime'] == 'outage_processing')
        arrival = next(c for c in d['cases'] if c['layout'] == layout and c['regime'] == 'outage_arrival')
        for c in (processing, arrival):
            impact = c['policies']['nearest']['incident']
            assert impact['outage_or_pause_duration_ticks'] == 20
            assert impact['service_wait_ticks'] > 0
            assert impact['affected_service_sample_count'] > 0
            assert impact['affected_extra_delay_sample_count'] > 0
            assert c['policies']['nearest']['contention']['occupied_target_nonmoving_ticks'] > 0
        assert processing['policies']['nearest']['incident']['interrupted_sessions'] > 0
        assert arrival['policies']['nearest']['incident']['dropoff_arrivals_during_outage'] > 0
    assert s['paired_learned_minus_heuristic_flow']['ci95'] is None
print(f'{profile}: validity gate passed; episodes={len(d["cases"])} ordinary={ordinary} near={near} meaningful={meaningful}; no model superiority required')
