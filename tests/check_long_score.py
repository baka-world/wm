"""Validate a complete new run and compare preregistered summary quantities."""
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab_transport.long_score_experiment import SEEDS, summarize


def main(root, profile):
    raw = json.loads((root / 'raw.json').read_text(encoding='utf-8'))
    summary = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
    archived = json.loads((Path('examples/evidence/structured-world-model') /
                           f'long-score-{profile}-summary.json').read_text(encoding='utf-8'))
    assert raw['profile'] == profile
    assert raw['schedule_audit']['cross_split_duplicate_groups'] == 0
    assert len(raw['cases']) == 20 and summary['comparable_decisions'] > 0
    assert {c['regime'] for c in raw['cases']} == {'normal', 'burst', 'outage_processing', 'outage_arrival', 'motion_delay'}
    for case in raw['cases']:
        assert case['seed'] in SEEDS[profile][case['layout']]
        for decision in case['decisions']:
            assert len(decision['candidates']) <= 8
            if not decision['fully_comparable']:
                continue
            for candidate in decision['candidates']:
                assert candidate['reference_complete'] and candidate['episode_complete']
                assert candidate['discrete_safety_violations'] == 0
                component = candidate['true_components']
                initial = {s['id']: s for s in decision['initial']['samples']}
                old_cost = 0
                for tick, statuses in component['steps']:
                    for sid, status in zip(component['sample_ids'], statuses):
                        deadline = initial[sid]['deadline_tick']
                        if status != 'completed':
                            old_cost += 1 + 4 * int(deadline is not None and tick > deadline)
                assert old_cost == candidate['true_original20'][0]
    regenerated = summarize(raw['cases'])
    assert summary == regenerated
    for metric in ('original20', 'long5', 'long20', 'rule_long20', 'linear_long20', 'progress_rule_long20'):
        current = summary['mean_regret_ticks'][metric]['mean']
        reference = archived['mean_regret_ticks'][metric]['mean']
        assert math.isclose(current, reference, rel_tol=1e-8, abs_tol=1e-8), (metric, current, reference)
    delta = summary['long20_minus_original20_regret_ticks']['mean']
    assert math.isclose(delta, archived['long20_minus_original20_regret_ticks']['mean'], rel_tol=1e-8, abs_tol=1e-8)
    print(f'{profile}: {len(raw["cases"])} cases, {summary["comparable_decisions"]} comparable decisions, '
          'replay scores and independent seed audit verified')


if __name__ == '__main__':
    main(Path(sys.argv[1]), sys.argv[2])
