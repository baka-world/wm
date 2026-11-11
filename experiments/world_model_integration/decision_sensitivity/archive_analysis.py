"""Read-only PR18 V4 decision archive audit. Never executes a model or simulator.

Inputs are immutable GitHub Actions artifact 13765795897. The factual event
rows belong to nearest-batch trajectories, NOT the saved fixed-candidate rollouts.
Do not interpret their coverage as observed rollout query/event counts.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json

ORIGINAL_HEAD = 'f9ba33515406ae077494e0cd9e5faf7c39708d3e'
V4_SHA256 = '60c8a6c3d832da3e6519072688ea099f64ae120ba19d3bd2ec705f7f1a11ac0c'
SOURCE_RUN = 42816481249
SOURCE_ARTIFACT = 13765795897
SEEDS = (43011, 43015, 43019)


def analyze(dispatch, factual=None):
    if not dispatch.get('only_frozen_precommitted_test') or not dispatch.get('gate_pass'):
        raise ValueError('expected completed frozen one-shot V4 dispatch archive')
    if dispatch['config']['horizon'] != 5 or dispatch['config']['particles'] != 3 or dispatch['config']['candidate_budget'] != 4:
        raise ValueError('unexpected original dispatch budget')
    rows = None if factual is None else factual['independent_in']
    out = {
        'scope': 'read_only_archive_development_diagnosis_not_independent_confirmation',
        'original_head': ORIGINAL_HEAD,
        'v4_sha256': V4_SHA256,
        'source': {'workflow_run': SOURCE_RUN, 'artifact_id': SOURCE_ARTIFACT,
                   'dispatch_file': 'v4-dispatch.json',
                   'factual_file': 'v4-independent-in-data.json'},
        'limits': {'factual_event_rows_are_separate_nearest_batch_trajectories': True,
                   'saved_fixed_rollout_queries': None,
                   'saved_fixed_rollout_probabilities': None,
                   'saved_fixed_rollout_event_differences': None,
                   'saved_fixed_rollout_state_differences': None,
                   'saved_fixed_rollout_component_means': None,
                   'note': 'Null means not archived, never zero. Replay is required for F1-F3 or 5/10/20 sensitivity.'},
        'states': [],
    }
    if [c['seed'] for c in dispatch['cases']] != list(SEEDS):
        raise ValueError('unexpected fixed case identities')
    for case in dispatch['cases']:
        states = case['fixed_states']
        if len(states) != 1:
            raise ValueError('original evidence did not contain exactly one state for this case')
        state = states[0]
        predictions = state['predictions']
        if not all(k in predictions for k in ('Rule', 'V3-C', 'V4')):
            raise ValueError('missing one frozen comparison arm')
        n = len(state['candidates'])
        if any(len(predictions[k]['score']) != n for k in predictions):
            raise ValueError('score/candidate length mismatch')
        truth = state['factual_counterfactual_flow']
        if len(truth) != n:
            raise ValueError('factual outcome length mismatch')
        if not any(x is not None for x in truth):
            raise ValueError('cannot compute observed counterfactual regret')
        optimal_cost = min(x for x in truth if x is not None)
        rank_rule = predictions['Rule']['rank']
        first, second = rank_rule[:2]
        a, b = predictions['Rule']['score'][first], predictions['Rule']['score'][second]
        first_different = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
        margin = 0.0 if first_different is None else b[first_different] - a[first_different]
        v4_delta = [[v-r for v, r in zip(predictions['V4']['score'][i],predictions['Rule']['score'][i])]
                    for i in range(n)]
        value = {
            'seed': case['seed'], 'tick': state['tick'], 'candidate_count': n,
            'candidates': state['candidates'],
            'archived_scores': {k: predictions[k]['score'] for k in ('Rule','V3-C','V4')},
            'archived_rank': {k: predictions[k]['rank'] for k in ('Rule','V3-C','V4')},
            'chosen': {k: predictions[k]['chosen'] for k in ('Rule','V3-C','V4')},
            'score_delta_v4_minus_rule': v4_delta,
            'score_delta_candidate_differential_primary': [d[0]-v4_delta[first][0] for d in v4_delta],
            'rule_leader': first, 'rule_runner_up': second,
            'rule_primary_gap': b[0]-a[0],
            'rule_lexicographic_first_distinguishing_component': first_different,
            'rule_first_distinguishing_component_gap': margin,
            'rule_full_score_tie': first_different is None,
            'primary_cost_change_to_reverse_rank':
                {'kind': 'strictly_positive_in_favor_of_runner_up' if b[0] == a[0] else 'strictly_greater_than_gap',
                 'value': b[0]-a[0]},
            'counterfactual_flow': truth,
            'ex_post_best_legal_candidates': [i for i,x in enumerate(truth) if x==optimal_cost],
            'ex_post_best_flow': optimal_cost,
            'rule_regret': truth[predictions['Rule']['chosen']]-optimal_cost,
            'v4_regret': truth[predictions['V4']['chosen']]-optimal_cost,
            'v3c_regret': truth[predictions['V3-C']['chosen']]-optimal_cost,
            'v4_probabilities_at_candidate_rollout_queries': None,
            'candidate_rollout_query_count': None,
            'candidate_rollout_event_differences': None,
            'candidate_rollout_state_differences': None,
            'candidate_rollout_score_component_means': None,
            'F_classification': {
                'F1': 'undetermined_without_per_candidate_query_replay',
                'F2': 'undetermined_without_sampled_event_replay',
                'F3': 'undetermined_without_state_and_component_replay',
                'F4': 'not_applicable_no_score_difference_for_V4',
                'F5': 'not_applicable_no_rank_change_for_V4',
                'F6': 'not_proven_candidate_cost_ties_may_coexist_with_different_factual_outcomes',
            },
        }
        if rows is not None:
            matched = [r for r in rows if r['seed'] == case['seed']]
            if not matched:
                raise ValueError('no separate factual records for seed')
            horizon = {}
            for h in (5, 10, 20):
                window = [r for r in matched if state['tick'] <= r['tick'] < state['tick']+h]
                horizon[str(h)] = {'eligible_factual_queries': len(window),
                    'factual_completed_service_events': sum(r['label'] for r in window),
                    'IMPORTANT': 'NOT candidate rollout, no scoring or CPU timing claim'}
            completed = [r['tick'] for r in matched if r['label']]
            value['separate_factual_trajectory_only'] = {
                'source': 'one-shot factual nearest_batch trajectories; not the fixed CP-SAT candidates',
                'first_eligible_tick': min(r['tick'] for r in matched),
                'first_completed_service_tick': min(completed) if completed else None,
                'coverage_by_absolute_horizon': horizon,
            }
        out['states'].append(value)
    out['summary'] = {
        'fixed_states': len(out['states']),
        'candidates_total': sum(s['candidate_count'] for s in out['states']),
        'v4_vs_rule_score_changed_candidates': sum(any(d != 0 for d in delta)
            for s in out['states'] for delta in s['score_delta_v4_minus_rule']),
        'v4_vs_rule_rank_changed_states': sum(s['archived_rank']['V4'] != s['archived_rank']['Rule'] for s in out['states']),
        'v4_vs_rule_action_changed_states': sum(s['chosen']['V4'] != s['chosen']['Rule'] for s in out['states']),
        'v3c_vs_rule_score_changed_candidates': sum(s['archived_scores']['V3-C'][i] != s['archived_scores']['Rule'][i]
            for s in out['states'] for i in range(s['candidate_count'])),
        'rule_and_v4_ex_post_regret_by_state': [s['rule_regret'] for s in out['states']],
    }
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dispatch', type=Path,required=True)
    p.add_argument('--factual',type=Path,default=None)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    data=analyze(json.loads(a.dispatch.read_text()),json.loads(a.factual.read_text()) if a.factual else None)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(data['summary'],ensure_ascii=False))

if __name__ == '__main__':
    main()
