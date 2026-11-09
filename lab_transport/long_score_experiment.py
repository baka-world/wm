"""Predeclared finite-candidate score comparison; CPU and offline replay only."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

from .clinic_map import make_clinic_scenario
from .decision_value import (PARAMETERS, batch_candidates, candidates_for, case_config, encode_batch,
    factual_run, incident_end, metrics, predict_trace, record_observation, replay_branch, scenario_case,
    terminal_distance, trace_score, truth_trace)
from .decision_value_round2 import (PRIOR_SEEDS, REGIMES, SEEDS as ROUND2_SEEDS, bootstrap_paired,
    burst_case, chosen_decisions, make_round2_cases, pairwise_loss, select_index, short_score_components, spearman)
from .dynamics_experiment import compare
from .dynamics_model import Geometry, HybridDynamicsModel
from .dynamics_revision import RevisedDynamicsModel, SPLITS_V2
from .hospital_case import make_scenario
from .multisite_experiment import make_case, prepare_tartu
from .state_v2 import ObservableHistory
from .torch_structured_adapter import ProgressAwareRule
from .training_data import SOURCES, download_sources

SEEDS = {'development': {'tartu': (15000, 15002), 'clinic': (16000, 16001)},
         'test': {'tartu': (17000, 17001), 'clinic': (18000, 18001)}}


def audit_schedules(bases):
    groups = defaultdict(set)
    def add(group, layout, case):
        groups[(layout, tuple((s.pickup, s.dropoff, s.release_tick) for s in case.samples))].add(group)
    for split, (layout, seeds) in SPLITS_V2.items():
        for seed in seeds:
            add('training_' + split, layout, make_case(bases[layout], layout, seed, 'normal'))
    for profile, maps in PRIOR_SEEDS.items():
        for layout, seeds in maps.items():
            for seed in seeds:
                for regime in ('normal', 'burst', 'urgent', 'contention'):
                    add('round1_' + profile, layout, scenario_case(bases[layout], layout, seed, regime)[0])
    for profile, maps in ROUND2_SEEDS.items():
        for layout, seeds in maps.items():
            for seed in seeds:
                add('round2_' + profile, layout, make_case(bases[layout], layout, seed, 'normal'))
                add('round2_' + profile, layout, burst_case(bases[layout], layout, seed))
    for profile, maps in SEEDS.items():
        for layout, seeds in maps.items():
            for seed in seeds:
                add(profile, layout, make_case(bases[layout], layout, seed, 'normal'))
                add(profile, layout, burst_case(bases[layout], layout, seed))
    overlap = [sorted(v) for v in groups.values() if len(v) > 1]
    return {'unique_patterns': len(groups), 'cross_split_duplicate_groups': len(overlap),
            'overlapping_split_names': overlap}


def long_score(initial, trace, geometry):
    if not trace:
        raise ValueError('trace must be nonempty')
    known = {s.id for s in initial.samples}
    cost = 0
    for state in trace:
        samples = {s.id: s for s in state.samples}
        if not known.issubset(samples):
            raise ValueError('known task disappeared')
        for sid in known:
            sample = samples[sid]
            if sample.status != 'completed':
                cost += 1 + 4 * int(sample.deadline_tick is not None and state.tick > sample.deadline_tick)
    terminal = trace[-1]
    distance = terminal_distance(terminal, known, geometry)
    stations = {s.id: s for s in terminal.stations}
    service = 0.0
    for sample in terminal.samples:
        if sample.id not in known or sample.status == 'completed':
            continue
        service += stations[sample.dropoff].expected_service_ticks
        if sample.status in {'waiting', 'assigned'}:
            service += stations[sample.pickup].expected_service_ticks
    return (cost + distance + service, distance)


def enriched_at(rows, target):
    observer = ObservableHistory()
    for row in rows:
        rich = observer.observe(row.observation)
        if row.observation.tick == target:
            return rich
    raise ValueError('decision tick missing')


def analyze(case, seed, pauses, factual, rows, row, labels, parameters):
    obs = row.observation
    geometry = Geometry(case.grid)
    batches, total = batch_candidates(obs, candidates_for(obs), geometry)
    models = {'rule': (HybridDynamicsModel(geometry, mode='heuristic'), obs),
              'linear': (RevisedDynamicsModel(geometry, parameters), obs),
              'progress_rule': (ProgressAwareRule(geometry), enriched_at(rows, obs.tick))}
    predictions = {}
    times = {}
    for name, (model, initial) in models.items():
        start = time.process_time()
        predictions[name] = [predict_trace(model, initial, action, geometry) for action in batches]
        times[name] = time.process_time() - start
    candidates = []
    for index, action in enumerate(batches):
        outcome, branch = replay_branch(case, seed, pauses, factual, rows, obs.tick, action)
        truth, complete = truth_trace(obs, branch)
        data = {'batch': encode_batch(action), 'reference_complete': complete,
                'episode_complete': outcome['summary']['status'] == 'completed',
                'full_mean_flow': metrics(outcome, case.max_ticks, None, incident_end(case, pauses))['restricted_mean_flow_ticks'],
                'discrete_safety_violations': metrics(outcome, case.max_ticks, None, incident_end(case, pauses))['discrete_safety_violations'],
                'sample_outcomes': [{k: s[k] for k in ('id', 'release_tick', 'assigned_tick', 'picked_tick', 'completed_tick', 'deadline_tick')}
                                    for s in outcome['samples']]}
        if complete:
            data['true_original20'] = list(trace_score(obs, truth, geometry))
            data['true_long5'] = list(long_score(obs, truth[:5], geometry))
            data['true_long20'] = list(long_score(obs, truth, geometry))
            data['true_components'] = short_score_components(obs, truth)
            data['true_states'] = {str(h): record_observation(truth[h-1]) for h in (1, 5, 20)}
            data['predictions'] = {}
            for name, traces in predictions.items():
                pred = traces[index]
                data['predictions'][name] = {'original20': list(trace_score(obs, pred, geometry)),
                    'long5': list(long_score(obs, pred[:5], geometry)), 'long20': list(long_score(obs, pred, geometry)),
                    'joint_counts': {str(h): compare(obs, truth[h-1], pred[h-1]) for h in (1, 5, 20)},
                    'states': {str(h): record_observation(pred[h-1]) for h in (1, 5, 20)}}
        candidates.append(data)
    result = {'tick': obs.tick, 'sampling_labels': labels, 'initial': record_observation(obs),
              'candidate_total': total, 'cpu_seconds': times, 'candidates': candidates,
              'fully_comparable': all(c['reference_complete'] and c['episode_complete'] for c in candidates)}
    if not result['fully_comparable']:
        return result
    flow = [c['full_mean_flow'] for c in candidates]
    score_sets = {'original20': [tuple(c['true_original20']) for c in candidates],
                  'long5': [tuple(c['true_long5']) for c in candidates],
                  'long20': [tuple(c['true_long20']) for c in candidates]}
    for name in models:
        score_sets[name + '_long20'] = [tuple(c['predictions'][name]['long20']) for c in candidates]
    best = select_index(flow, batches)
    result['full_reference_index'] = best
    result['flow_gap'] = max(flow) - min(flow)
    result['choices'] = {name: select_index(scores, batches) for name, scores in score_sets.items()}
    result['regret'] = {name: flow[index] - flow[best] for name, index in result['choices'].items()}
    result['rank_error'] = {name: pairwise_loss(scores, flow) for name, scores in score_sets.items()}
    result['spearman'] = {name: spearman(scores, flow) for name, scores in score_sets.items()}
    return result


def summarize(cases):
    keys = ('original20', 'long5', 'long20', 'rule_long20', 'linear_long20', 'progress_rule_long20')
    counts = {k: [] for k in keys}
    gaps, differences = [], []
    by_regime = defaultdict(lambda: defaultdict(list))
    decision_count = 0
    for case in cases:
        ds = [d for d in case['decisions'] if d['fully_comparable']]
        if not ds:
            continue
        decision_count += len(ds)
        layout, seed = case['layout'], case['seed']
        gaps.append((layout, seed, statistics.mean(d['flow_gap'] for d in ds)))
        for key in keys:
            value = statistics.mean(d['regret'][key] for d in ds)
            counts[key].append((layout, seed, value))
            by_regime[case['regime']][key].append(value)
        differences.append((layout, seed, statistics.mean(d['regret']['long20'] - d['regret']['original20'] for d in ds)))
    return {'cases': len(cases), 'comparable_decisions': decision_count,
            'mean_candidate_gap_ticks': bootstrap_paired(gaps),
            'mean_regret_ticks': {key: bootstrap_paired(values) for key, values in counts.items()},
            'long20_minus_original20_regret_ticks': bootstrap_paired(differences),
            'by_regime_case_mean_regret_ticks': {regime: {key: statistics.mean(v) for key, v in values.items()}
                                                 for regime, values in by_regime.items()},
            'note': 'Equal decision weighting within case, equal case weighting within seed cluster; layout-stratified seed bootstrap.'}


def run(profile):
    started = time.process_time()
    src = download_sources()
    bases = {'tartu': prepare_tartu(make_scenario(src['hospital_map'], seconds_per_tick=10, tiago_config=src['tiago_config'])),
             'clinic': make_clinic_scenario(src['clinic_map'], src['clinic_config'])}
    audit = audit_schedules(bases)
    if audit['cross_split_duplicate_groups']:
        raise ValueError('predeclared cross-split schedules overlap: ' + repr(audit))
    parameters = json.loads(PARAMETERS.read_text(encoding='utf-8'))
    cases = []
    for layout, seeds in SEEDS[profile].items():
        for seed in seeds:
            variants, design = make_round2_cases(bases[layout], layout, seed)
            for regime in REGIMES:
                case, pauses, incident = variants[regime]
                factual, rows = factual_run(case, seed, pauses)
                decisions = [analyze(case, seed, pauses, factual, rows, row, labels, parameters)
                             for row, labels in chosen_decisions(rows, incident)]
                cases.append({'layout': layout, 'seed': seed, 'regime': regime,
                              'config': case_config(case, pauses), 'incident': incident,
                              'design': design, 'decisions': decisions})
                print(f'{profile} {layout} {seed} {regime} decisions={len(decisions)}', flush=True)
    return {'profile': profile, 'scope': 'synthetic CPU, finite offline replay; no future information in online model',
            'source_sha256': {k: v[1] for k, v in SOURCES.items()},
            'linear_parameters_sha256': hashlib.sha256(PARAMETERS.read_bytes()).hexdigest(),
            'schedule_audit': audit, 'cases': cases, 'cpu_seconds': time.process_time() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=tuple(SEEDS), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
            raise FileExistsError(f'refusing to overwrite nonempty output: {args.output}')
        result = run(args.profile)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'raw.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        summary = summarize(result['cases'])
        (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return 0
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        print(f'long-score experiment failed: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
