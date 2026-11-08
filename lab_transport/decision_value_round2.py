"""Second CPU pilot: disturbance exposure, action gaps and objective alignment.

Scenarios are pre-specified simulations, not clinical data. Exogenous incident
windows are derived from a no-incident nearest-policy probe before comparisons.
Offline future outcomes never enter a normal online decision.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import time

from .clinic_map import make_clinic_scenario
from .decision_value import (HORIZON, BUDGET, ModelPolicy, batch_candidates, candidates_for,
    case_config, encode_batch, incident_end, metrics, predict_trace, record_observation,
    replay_branch, trace_score, truth_trace)
from .dynamics_experiment import compare
from .dynamics_model import Geometry, HybridDynamicsModel
from .dynamics_revision import RevisedDynamicsModel, SPLITS_V2
from .hospital_case import make_scenario
from .multisite_experiment import make_case, prepare_tartu
from .simulation import Simulator
from .training_data import SOURCES, download_sources

ROOT = Path(__file__).resolve().parents[1]
PARAMETERS = ROOT / 'examples/evidence/dynamics-revision/dynamics-parameters.json'
PRIOR_SEEDS = {'pilot': {'tartu': (11000,), 'clinic': (12000,)},
               'confirm': {'tartu': (11001,), 'clinic': (12001,)}}
SEEDS = {'pilot': {'tartu': (13000,), 'clinic': (14000,)},
         'confirm': {'tartu': (13001, 13002, 13003, 13004),
                     'clinic': (14001, 14002, 14003, 14004)}}
REGIMES = ('normal', 'burst', 'outage_processing', 'outage_arrival', 'motion_delay')
POLICIES = ('nearest', 'world', 'rule', 'rollout', 'heuristic', 'learned')
FAULT_TICKS = 20
PAUSE_TICKS = 10
MEANINGFUL_FLOW_GAP = 1.0
BOOTSTRAP_SEED = 20261109
BOOTSTRAP_SAMPLES = 2000


def burst_case(base, layout, seed):
    case = make_case(base, layout, seed, 'normal')
    samples = list(case.samples)
    # Explicit synthetic competition: the first two samples share endpoints.
    samples[1] = replace(samples[1], pickup=samples[0].pickup, dropoff=samples[0].dropoff)
    samples = [replace(s, release_tick=t) for s, t in zip(samples, (0, 0, 2, 4, 8, 12))]
    return replace(case, name=f'{layout} round2 burst seed={seed} (synthetic)', samples=tuple(samples))


def probe_windows(burst, seed):
    """Deterministic, predeclared no-incident nearest reference; fixed before outcome comparisons."""
    probe = Simulator(burst, 'nearest', seed).run()
    pickup = next((e for e in probe['events'] if e['type'] == 'service_started' and e['phase'] == 'to_pickup'), None)
    drop = next((e for e in probe['events'] if e['type'] == 'service_started' and e['phase'] == 'to_dropoff'), None)
    positions = [{r['id']: tuple(r['position']) for r in h['robots']} for h in probe['history']]
    first_move = next((t for t in range(len(positions)-1) if positions[t]['R1'] != positions[t+1]['R1']), None)
    if pickup is None or drop is None or first_move is None:
        raise ValueError('reference probe lacks a pickup, dropoff or R1 movement')
    return {'pickup': {'station_id': pickup['station_id'], 'start': pickup['tick'] + 1,
                       'end': pickup['tick'] + 1 + FAULT_TICKS, 'probe_service_start': pickup['tick']},
            'arrival': {'station_id': drop['station_id'], 'start': max(0, drop['tick'] - 1),
                        'end': max(0, drop['tick'] - 1) + FAULT_TICKS, 'probe_service_start': drop['tick']},
            'delay': {'robot_id': 'R1', 'start': first_move, 'end': first_move + PAUSE_TICKS},
            'source': 'no-incident nearest reference; predeclared timing rule, not clinical data'}


def make_round2_cases(base, layout, seed):
    normal = make_case(base, layout, seed, 'normal')
    burst = burst_case(base, layout, seed)
    design = probe_windows(burst, seed)
    cases = {'normal': (normal, {}, None), 'burst': (burst, {}, None)}
    for regime, key in (('outage_processing', 'pickup'), ('outage_arrival', 'arrival')):
        cfg = design[key]
        stations = dict(burst.stations)
        stations[cfg['station_id']] = replace(stations[cfg['station_id']], outages=((cfg['start'], cfg['end']),))
        cases[regime] = (replace(burst, name=f'{layout} {regime} seed={seed} (synthetic)', stations=stations),
                         {}, {'kind': 'station_outage', **cfg})
    delay = design['delay']
    cases['motion_delay'] = (replace(burst, name=f'{layout} motion_delay seed={seed} (synthetic)'),
                             {'R1': ((delay['start'], delay['end']),)}, {'kind': 'motion_pause', **delay})
    return cases, design


def audit_schedules(bases):
    groups = {}
    def add(split, layout, case):
        key = (layout, tuple((s.pickup, s.dropoff, s.release_tick) for s in case.samples))
        groups.setdefault(key, set()).add(split)
    for split, (layout, seeds) in SPLITS_V2.items():
        for seed in seeds:
            add('old_' + split, layout, make_case(bases[layout], layout, seed, 'normal'))
    for profile, maps in PRIOR_SEEDS.items():
        for layout, seeds in maps.items():
            for seed in seeds:
                for regime in ('normal', 'burst', 'urgent', 'contention'):
                    from .decision_value import scenario_case
                    add('old_' + profile, layout, scenario_case(bases[layout], layout, seed, regime)[0])
    for profile, maps in SEEDS.items():
        for layout, seeds in maps.items():
            for seed in seeds:
                add(profile, layout, make_case(bases[layout], layout, seed, 'normal'))
                add(profile, layout, burst_case(bases[layout], layout, seed))
    overlaps = [sorted(v) for v in groups.values() if len(v) > 1]
    return {'unique_patterns': len(groups), 'cross_split_duplicate_groups': len(overlaps),
            'overlapping_split_names': overlaps}


def run_policy(case, seed, pauses, name, parameters, capture=False):
    geometry = Geometry(case.grid)
    controller = None
    if name == 'heuristic':
        controller = TieAwareModelPolicy(HybridDynamicsModel(geometry, mode='heuristic'), geometry)
    elif name == 'learned':
        controller = TieAwareModelPolicy(RevisedDynamicsModel(geometry, parameters), geometry)
    rows = []
    start = time.process_time()
    sim = Simulator(case, 'world' if controller else name, seed, motion_pauses=pauses, collect_metrics=True)
    result = sim.run(on_transition=rows.append if capture else None, dispatch_policy=controller)
    elapsed = time.process_time() - start
    return result, rows, controller, elapsed


def post_run_trace(result):
    """Reproducible offline evidence; never an online policy or training input.

    The factual history includes tasks before their release, so this record
    is stored only after execution. Position keys allow consistency checks
    without publishing the source map's coordinate list or graph.
    """
    def key(position):
        return hashlib.sha256(f'{position[0]},{position[1]}'.encode()).hexdigest()[:16]
    history = [{'tick': snap['tick'],
                'robots': [[r['id'], key(r['position']), r['task_id'], r['phase']] for r in snap['robots']],
                'samples': [[s['id'], s['status'], s['robot_id']] for s in snap['samples']],
                'stations': [[s['id'], s['available']] for s in snap['stations']]} for snap in result['history']]
    allowed = {'tick', 'type', 'robot_id', 'sample_id', 'station_id', 'phase',
               'observed_ticks', 'deadline_tick', 'steps', 'explored_states'}
    events = [{k: v for k, v in e.items() if k in allowed} for e in result['events']]
    return {'scope': 'offline factual history; contains unreleased task states; never policy/training input',
            'robot_columns': ['id', 'position_key', 'task_id', 'phase'],
            'sample_columns': ['id', 'status', 'robot_id'],
            'station_columns': ['id', 'available'], 'history': history, 'events': events}


def select_index(scores, batches):
    """On an exact tie prefer a feasible assignment over indefinite waiting."""
    if not scores or len(scores) != len(batches):
        raise ValueError('scores and candidate batches must be nonempty and aligned')
    return min(range(len(scores)), key=lambda i: (scores[i], not bool(batches[i]), i))


class TieAwareModelPolicy(ModelPolicy):
    """The same no-op tie rule for both predictors; wait remains available."""
    def __call__(self, obs, candidates):
        start = time.process_time()
        batches, total = batch_candidates(obs, candidates, self.geometry)
        records = []
        try:
            for batch in batches:
                trace = predict_trace(self.model, obs, batch, self.geometry)
                self.model_steps += len(trace)
                records.append({'batch': encode_batch(batch), 'score': list(trace_score(obs, trace, self.geometry))})
            chosen = select_index([entry['score'] for entry in records], batches)
        except (ValueError, RuntimeError, TypeError, KeyError, StopIteration, OverflowError) as exc:
            self.fallbacks += 1
            chosen = 1 if len(batches) > 1 else 0
            self.decisions.append({'tick': obs.tick, 'candidate_count': total, 'evaluated': records,
                                   'fallback': type(exc).__name__, 'selected': encode_batch(batches[chosen])})
            self.cpu_seconds += time.process_time() - start
            return batches[chosen]
        self.decisions.append({'tick': obs.tick, 'candidate_count': total, 'evaluated': records,
                               'fallback': None, 'selected': encode_batch(batches[chosen])})
        self.cpu_seconds += time.process_time() - start
        return batches[chosen]


def service_sessions(result, case, incident):
    """Count actual ticks with an unavailable target during a service session."""
    finished = {(e['robot_id'], e['sample_id'], e['station_id'], e['phase']): e['tick']
                for e in result['events'] if e['type'] == 'service_finished'}
    sessions = []
    for e in result['events']:
        if e['type'] != 'service_started':
            continue
        key = (e['robot_id'], e['sample_id'], e['station_id'], e['phase'])
        end = finished.get(key, result['summary']['ticks'])
        unavailable = [t for t in range(e['tick'], end) if not case.stations[e['station_id']].available(t)]
        interrupted = bool(incident and incident['kind'] == 'station_outage'
            and e['station_id'] == incident['station_id'] and e['tick'] < incident['start'] < end
            and case.stations[e['station_id']].available(incident['start'] - 1))
        arrival_down = bool(incident and incident['kind'] == 'station_outage'
            and e['station_id'] == incident['station_id'] and e['phase'] == 'to_dropoff'
            and incident['start'] <= e['tick'] < incident['end'])
        sessions.append({'sample_id': e['sample_id'], 'robot_id': e['robot_id'],
                         'station_id': e['station_id'], 'phase': e['phase'],
                         'service_start': e['tick'], 'service_end': finished.get(key),
                         'right_censored': key not in finished, 'unavailable_wait_ticks': len(unavailable),
                         'interrupted_after_service_start': interrupted,
                         'dropoff_arrived_during_outage': arrival_down})
    actual_wait = sum(s['unavailable_wait_ticks'] for s in sessions)
    if actual_wait != result['summary']['station_unavailable_wait_ticks']:
        raise ValueError('service-session wait ticks do not match simulator instrumentation')
    return sessions


def contention_proxy(result, case):
    """Occupied station plus another task seeking it; not a FIFO queue measurement."""
    ticks, still_ticks, tasks = 0, 0, set()
    history = result['history']
    for t in range(len(history) - 1):
        old, new = history[t], history[t+1]
        samples = {s['id']: s for s in old['samples']}
        robots = {r['id']: r for r in old['robots']}
        nxt = {r['id']: r for r in new['robots']}
        for r in robots.values():
            sid, phase = r['task_id'], r['phase']
            if sid is None or phase not in ('to_pickup', 'to_dropoff'):
                continue
            sample = samples[sid]
            # history omits endpoints, use scenario's immutable sample specification
            spec = next(s for s in case.samples if s.id == sample['id'])
            station = case.stations[spec.pickup if phase == 'to_pickup' else spec.dropoff]
            occupied = any(other['id'] != r['id'] and tuple(other['position']) == station.position
                           for other in robots.values())
            if occupied and tuple(r['position']) != station.position:
                ticks += 1
                tasks.add(sid)
                still_ticks += r['position'] == nxt[r['id']]['position']
    return {'occupied_target_robot_ticks': ticks, 'occupied_target_nonmoving_ticks': still_ticks,
            'tasks_seeking_occupied_target': len(tasks),
            'scope': 'occupancy/intent proxy, not measured FIFO queue or isolated cause of nonmovement'}


def pending_at(result, case, tick):
    return sum(s.release_tick <= tick and
        (next(x for x in result['samples'] if x['id'] == s.id)['completed_tick'] is None or
         next(x for x in result['samples'] if x['id'] == s.id)['completed_tick'] > tick)
        for s in case.samples)


def paired_incident(result, baseline, case, incident):
    if not incident:
        return None
    normal = {s['id']: s for s in baseline['samples']}
    changes = []
    censored = []
    for item in result['samples']:
        other = normal[item['id']]
        if item['completed_tick'] is None or other['completed_tick'] is None:
            censored.append(item['id'])
        else:
            changes.append({'sample_id': item['id'], 'completion_delta_ticks': item['completed_tick'] - other['completed_tick']})
    sessions = service_sessions(result, case, incident)
    return {'outage_or_pause_duration_ticks': incident['end'] - incident['start'],
            'service_sessions': sessions,
            'affected_service_sample_count': len({s['sample_id'] for s in sessions if s['unavailable_wait_ticks']}),
            'service_wait_ticks': sum(s['unavailable_wait_ticks'] for s in sessions),
            'interrupted_sessions': sum(s['interrupted_after_service_start'] for s in sessions),
            'dropoff_arrivals_during_outage': sum(s['dropoff_arrived_during_outage'] for s in sessions),
            'fault_vs_no_fault_completion_deltas': changes, 'completion_censored_samples': censored,
            'completed_paired_samples': len(changes),
            'affected_extra_delay_sample_count': sum(x['completion_delta_ticks'] > 0 for x in changes),
            'mean_signed_completion_delta_ticks': statistics.mean(x['completion_delta_ticks'] for x in changes) if changes else None,
            'pending_at_recovery': pending_at(result, case, incident['end']),
            'pending_at_recovery_no_fault': pending_at(baseline, case, incident['end']),
            'note': 'same seed and policy; signed outcome differences include changed subsequent interactions, not isolated causal attribution for an individual sample'}


def chosen_decisions(rows, incident):
    """Offline sampling rule fixed before results; no selection affects online policy."""
    eligible = [r for r in rows if candidates_for(r.observation)]
    if not incident:
        return [(r, 'ordinary') for r in eligible[:2]]
    start, end = incident['start'], incident['end']
    ordinary = [r for r in eligible if r.observation.tick < start - 10 or r.observation.tick > end + 10][:2]
    before = [r for r in eligible if max(0, start-20) <= r.observation.tick < start]
    during = [r for r in eligible if start <= r.observation.tick < end]
    after = [r for r in eligible if end <= r.observation.tick <= end+20]
    selections = [(r, 'ordinary') for r in ordinary]
    if before: selections.append((before[-1], 'before'))
    if during: selections.append((during[0], 'during'))
    if after: selections.append((after[0], 'after'))
    dedup = {}
    for row, label in selections:
        if row.observation.tick not in dedup: dedup[row.observation.tick] = [row, []]
        dedup[row.observation.tick][1].append(label)
    return [(x[0], ','.join(x[1])) for _, x in sorted(dedup.items())]


def ranks(values):
    ordered = sorted(range(len(values)), key=lambda i: values[i])
    output = [0.0] * len(values)
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and values[ordered[j]] == values[ordered[i]]: j += 1
        rank = (i + j - 1) / 2.0
        for k in ordered[i:j]: output[k] = rank
        i = j
    return output


def spearman(first, second):
    if len(first) != len(second) or len(first) < 2: return None
    a, b = ranks(first), ranks(second)
    ma, mb = statistics.mean(a), statistics.mean(b)
    numerator = sum((x-ma)*(y-mb) for x, y in zip(a, b))
    denominator = math.sqrt(sum((x-ma)**2 for x in a) * sum((y-mb)**2 for y in b))
    return numerator / denominator if denominator else None


def pairwise_loss(predicted, truth):
    wrong = comparable = 0.0
    for i in range(len(truth)):
        for j in range(i+1, len(truth)):
            if truth[i] == truth[j]: continue
            comparable += 1
            if predicted[i] == predicted[j]: wrong += 0.5
            elif (predicted[i] < predicted[j]) != (truth[i] < truth[j]): wrong += 1
    return {'discordance': wrong, 'comparable_pairs': int(comparable),
            'rate': wrong/comparable if comparable else None}


def analyze_decision(case, seed, pauses, factual, rows, row, labels, parameters):
    obs = row.observation
    geo = Geometry(case.grid)
    batches, total = batch_candidates(obs, candidates_for(obs), geo, BUDGET)
    models = {'heuristic': HybridDynamicsModel(geo, mode='heuristic'),
              'learned': RevisedDynamicsModel(geo, parameters)}
    predictions = {name: [predict_trace(model, obs, batch, geo) for batch in batches]
                   for name, model in models.items()}
    candidates = []
    for i, batch in enumerate(batches):
        outcome, branch_rows = replay_branch(case, seed, pauses, factual, rows, obs.tick, batch)
        trace, complete = truth_trace(obs, branch_rows)
        record = {'batch': encode_batch(batch), 'short_reference_complete': complete,
                  'full_episode_completed': outcome['summary']['status'] == 'completed',
                  'full_restricted_mean_flow_ticks': metrics(outcome, case.max_ticks, None, incident_end(case, pauses))['restricted_mean_flow_ticks'],
                  'discrete_safety_violations': metrics(outcome, case.max_ticks, None, incident_end(case, pauses))['discrete_safety_violations']}
        if complete:
            record['true_short_score'] = list(trace_score(obs, trace, geo))
            record['true_states'] = {str(h): record_observation(trace[h-1]) for h in (1,5,20)}
            record['future_arrivals'] = len({s.id for o in trace for s in o.samples} - {s.id for s in obs.samples})
            record['future_station_change'] = any({s.id:s.available for s in o.stations} !=
                                                  {s.id:s.available for s in obs.stations} for o in trace)
            record['predictions'] = {}
            for name, traces in predictions.items():
                pred = traces[i]
                record['predictions'][name] = {'short_score': list(trace_score(obs, pred, geo)),
                    'short_primary_abs_error': abs(trace_score(obs, pred, geo)[0] - record['true_short_score'][0]),
                    'joint_counts': {str(h): compare(obs, trace[h-1], pred[h-1]) for h in (1,5,20)},
                    'states': {str(h): record_observation(pred[h-1]) for h in (1,5,20)}}
        candidates.append(record)
    result = {'tick': obs.tick, 'sampling_labels': labels, 'initial_observation': record_observation(obs),
              'candidate_count': total, 'candidate_evaluated': len(batches), 'candidates': candidates,
              'fully_comparable': all(c['short_reference_complete'] and c['full_episode_completed'] for c in candidates)}
    if not result['fully_comparable']:
        result['reason'] = 'at least one short reference or full candidate episode is censored'
        return result
    flows = [c['full_restricted_mean_flow_ticks'] for c in candidates]
    true_short = [tuple(c['true_short_score']) for c in candidates]
    best = select_index(flows, batches)
    short = select_index(true_short, batches)
    choices = {'full_flow_reference': best, 'true_short_reference': short}
    rankings = {'true_short_vs_full_spearman': spearman(true_short, flows)}
    maes = {}
    for name in models:
        predicted = [tuple(c['predictions'][name]['short_score']) for c in candidates]
        choices[name] = select_index(predicted, batches)
        rankings[name + '_vs_true_short_pairwise'] = pairwise_loss(predicted, true_short)
        rankings[name + '_vs_full_spearman'] = spearman(predicted, flows)
        maes[name] = statistics.mean(c['predictions'][name]['short_primary_abs_error'] for c in candidates)
    gap = max(flows) - min(flows)
    regret = {k: flows[v] - flows[best] for k, v in choices.items()}
    true_short_regret = regret['true_short_reference']
    hr = rankings['heuristic_vs_true_short_pairwise']['rate']
    lr = rankings['learned_vs_true_short_pairwise']['rate']
    result.update({'choices': choices, 'full_flow_gap_ticks': gap, 'full_flow_regret_ticks': regret,
        'rankings': rankings, 'short_primary_score_mae': maes,
        'diagnostics': {'A_near_equal_candidates': gap < MEANINGFUL_FLOW_GAP,
            'B_predicted_short_ranking_has_errors': gap >= MEANINGFUL_FLOW_GAP and ((hr or 0) > 0 or (lr or 0) > 0),
            'B_heuristic_scores_all_equal': gap >= MEANINGFUL_FLOW_GAP and len({tuple(c['predictions']['heuristic']['short_score']) for c in candidates}) == 1,
            'B_learned_scores_all_equal': gap >= MEANINGFUL_FLOW_GAP and len({tuple(c['predictions']['learned']['short_score']) for c in candidates}) == 1,
            'C_true_short_choice_full_regret': gap >= MEANINGFUL_FLOW_GAP and true_short_regret >= MEANINGFUL_FLOW_GAP,
            'D_learning_improves_short_error_rank_and_flow': hr is not None and lr is not None and
                maes['learned'] < maes['heuristic'] and lr < hr and
                regret['learned'] + MEANINGFUL_FLOW_GAP <= regret['heuristic']}})
    return result


def quantile(sorted_values, p):
    if len(sorted_values) == 1: return sorted_values[0]
    pos = (len(sorted_values)-1)*p
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_values[lo] + (sorted_values[hi]-sorted_values[lo])*(pos-lo)


def bootstrap_paired(case_values):
    """Resample independent layout/seed clusters, keeping their scenarios and paired methods together."""
    clusters = defaultdict(list)
    for layout, seed, value in case_values:
        clusters[(layout, seed)].append(value)
    by_map = defaultdict(list)
    for (layout, seed), values in sorted(clusters.items()):
        by_map[layout].append(statistics.mean(values))
    values = [v for group in by_map.values() for v in group]
    if not values: return {'clusters': 0, 'mean': None, 'ci95': None}
    if len(values) < 2 or any(len(group) < 2 for group in by_map.values()):
        return {'clusters': len(values), 'mean': statistics.mean(values), 'ci95': None,
                'reason': 'fewer than two seed clusters in a map stratum'}
    rng = random.Random(BOOTSTRAP_SEED)
    draws = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [rng.choice(group) for group in by_map.values() for _ in group]
        draws.append(statistics.mean(sample))
    draws.sort()
    return {'clusters': len(values), 'mean': statistics.mean(values),
            'ci95': [quantile(draws, .025), quantile(draws, .975)],
            'method': 'stratified bootstrap of layout/seed clusters; episodes and decision points kept together',
            'resamples': BOOTSTRAP_SAMPLES, 'seed': BOOTSTRAP_SEED}


def summarize(result):
    cases = result['cases']
    decisions = [d for c in cases for d in c['decisions'] if d['fully_comparable']]
    diagnostics = {k: sum(d['diagnostics'][k] for d in decisions) for k in (
        'A_near_equal_candidates', 'B_predicted_short_ranking_has_errors',
        'B_heuristic_scores_all_equal', 'B_learned_scores_all_equal',
        'C_true_short_choice_full_regret', 'D_learning_improves_short_error_rank_and_flow')}
    paired_flows, paired_regret, paired_rank_error, paired_primary_mae = [], [], [], []
    short_vs_rule, full_vs_rule = [], []
    reference_outcomes = {name: {'better': 0, 'equal': 0, 'worse': 0}
                          for name in ('true_short_reference', 'full_flow_reference', 'learned')}
    gap_by_episode, short_regret_by_episode, rho_by_episode = [], [], []
    joint_by_episode = {name: {h: [] for h in ('1', '5', '20')} for name in ('heuristic', 'learned')}
    for c in cases:
        h, l = c['policies']['heuristic']['metrics'], c['policies']['learned']['metrics']
        paired_flows.append((c['layout'], c['seed'], l['restricted_mean_flow_ticks'] - h['restricted_mean_flow_ticks']))
        ds = [d for d in c['decisions'] if d['fully_comparable']]
        if ds:
            paired_regret.append((c['layout'], c['seed'], statistics.mean(
                d['full_flow_regret_ticks']['learned'] - d['full_flow_regret_ticks']['heuristic'] for d in ds)))
            gap_by_episode.append((c['layout'], c['seed'], statistics.mean(d['full_flow_gap_ticks'] for d in ds)))
            short_regret_by_episode.append((c['layout'], c['seed'], statistics.mean(
                d['full_flow_regret_ticks']['true_short_reference'] for d in ds)))
            paired_primary_mae.append((c['layout'], c['seed'], statistics.mean(
                d['short_primary_score_mae']['learned'] - d['short_primary_score_mae']['heuristic'] for d in ds)))
            short_vs_rule.append((c['layout'], c['seed'], statistics.mean(
                d['full_flow_regret_ticks']['true_short_reference'] - d['full_flow_regret_ticks']['heuristic'] for d in ds)))
            full_vs_rule.append((c['layout'], c['seed'], statistics.mean(
                d['full_flow_regret_ticks']['full_flow_reference'] - d['full_flow_regret_ticks']['heuristic'] for d in ds)))
            for d in ds:
                for name in reference_outcomes:
                    delta = d['full_flow_regret_ticks'][name] - d['full_flow_regret_ticks']['heuristic']
                    reference_outcomes[name]['better' if delta < 0 else 'worse' if delta > 0 else 'equal'] += 1
            paired_ranks = [d['rankings']['learned_vs_true_short_pairwise']['rate'] -
                            d['rankings']['heuristic_vs_true_short_pairwise']['rate'] for d in ds
                            if d['rankings']['learned_vs_true_short_pairwise']['rate'] is not None and
                               d['rankings']['heuristic_vs_true_short_pairwise']['rate'] is not None]
            if paired_ranks:
                paired_rank_error.append((c['layout'], c['seed'], statistics.mean(paired_ranks)))
            correlations = [d['rankings']['true_short_vs_full_spearman'] for d in ds
                            if d['rankings']['true_short_vs_full_spearman'] is not None]
            if correlations:
                rho_by_episode.append((c['layout'], c['seed'], statistics.mean(correlations)))
            for name in ('heuristic', 'learned'):
                for h in ('1', '5', '20'):
                    counts = [candidate['predictions'][name]['joint_counts'][h]
                              for d in ds for candidate in d['candidates']]
                    total = sum(x['joint_total'] for x in counts)
                    if total:
                        joint_by_episode[name][h].append((c['layout'], c['seed'],
                                                         sum(x['joint_correct'] for x in counts) / total))
    by_regime = {}
    for regime in REGIMES:
        subset = [c for c in cases if c['regime'] == regime]
        ds = [d for c in subset for d in c['decisions'] if d['fully_comparable']]
        gap_cases, rho_cases, ranking_h, ranking_l, mae_h, mae_l, regime_regret = [], [], [], [], [], [], []
        for c in subset:
            valid = [d for d in c['decisions'] if d['fully_comparable']]
            if not valid:
                continue
            identity = (c['layout'], c['seed'])
            gap_cases.append((*identity, statistics.mean(d['full_flow_gap_ticks'] for d in valid)))
            regime_regret.append((*identity, statistics.mean(
                d['full_flow_regret_ticks']['learned'] - d['full_flow_regret_ticks']['heuristic'] for d in valid)))
            for key, target in (('true_short_vs_full_spearman', rho_cases),):
                vals = [d['rankings'][key] for d in valid if d['rankings'][key] is not None]
                if vals: target.append((*identity, statistics.mean(vals)))
            for name, rank_target, mae_target in (('heuristic', ranking_h, mae_h), ('learned', ranking_l, mae_l)):
                vals = [d['rankings'][name + '_vs_true_short_pairwise']['rate'] for d in valid
                        if d['rankings'][name + '_vs_true_short_pairwise']['rate'] is not None]
                if vals: rank_target.append((*identity, statistics.mean(vals)))
                mae_target.append((*identity, statistics.mean(d['short_primary_score_mae'][name] for d in valid)))
        paired_regime_flow = [(c['layout'], c['seed'],
            c['policies']['learned']['metrics']['restricted_mean_flow_ticks'] -
            c['policies']['heuristic']['metrics']['restricted_mean_flow_ticks']) for c in subset]
        incident_deltas = [(c['layout'], c['seed'], c['policies']['nearest']['incident']['mean_signed_completion_delta_ticks'])
                           for c in subset if c['policies']['nearest']['incident'] is not None
                           and c['policies']['nearest']['incident']['mean_signed_completion_delta_ticks'] is not None]
        exposed = [c['policies']['nearest']['incident'] for c in subset if c['policies']['nearest']['incident'] is not None]
        by_regime[regime] = {'episodes': len(subset), 'comparable_decisions': len(ds),
            'meaningful_gap_decisions': sum(d['full_flow_gap_ticks'] >= MEANINGFUL_FLOW_GAP for d in ds),
            'mean_gap_ticks_episode_weighted': bootstrap_paired(gap_cases),
            'near_event_decisions': sum(any(x in d['sampling_labels'].split(',') for x in ('before','during','after')) for d in ds),
            'true_short_vs_full_spearman_episode_weighted': bootstrap_paired(rho_cases),
            'heuristic_short_rank_error': bootstrap_paired(ranking_h),
            'learned_short_rank_error': bootstrap_paired(ranking_l),
            'heuristic_short_primary_mae': bootstrap_paired(mae_h),
            'learned_short_primary_mae': bootstrap_paired(mae_l),
            'paired_learned_minus_heuristic_flow': bootstrap_paired(paired_regime_flow),
            'paired_learned_minus_heuristic_decision_regret': bootstrap_paired(regime_regret),
            'nearest_fault_minus_no_fault_completion_delta': bootstrap_paired(incident_deltas),
            'nearest_exposure': {'interrupted_sessions': sum(x['interrupted_sessions'] for x in exposed),
                'dropoff_arrivals_during_outage': sum(x['dropoff_arrivals_during_outage'] for x in exposed),
                'service_wait_ticks': sum(x['service_wait_ticks'] for x in exposed),
                'affected_service_samples': sum(x['affected_service_sample_count'] for x in exposed),
                'positive_completion_deltas': sum(x['affected_extra_delay_sample_count'] for x in exposed),
                'censored_samples': sum(len(x['completion_censored_samples']) for x in exposed),
                'pending_at_recovery': sum(x['pending_at_recovery'] for x in exposed),
                'pending_at_recovery_no_fault': sum(x['pending_at_recovery_no_fault'] for x in exposed)},
            'nearest_competition_nonmoving_ticks': sum(c['policies']['nearest']['contention']['occupied_target_nonmoving_ticks']
                                                        for c in subset)}
    return {'profile': result['profile'], 'episodes': len(cases), 'comparable_decisions': len(decisions),
            'candidate_branches': sum(d['candidate_evaluated'] for d in decisions),
            'truncated_decisions': sum(d['candidate_count'] > d['candidate_evaluated'] for d in decisions),
            'diagnostics': diagnostics, 'by_regime': by_regime,
            'mean_candidate_flow_gap': bootstrap_paired(gap_by_episode),
            'true_short_choice_flow_regret': bootstrap_paired(short_regret_by_episode),
            'true_short_vs_full_spearman': bootstrap_paired(rho_by_episode),
            'paired_learned_minus_heuristic_rank_error': bootstrap_paired(paired_rank_error),
            'paired_learned_minus_heuristic_short_primary_mae': bootstrap_paired(paired_primary_mae),
            'reference_vs_rule_decision_counts': reference_outcomes,
            'paired_true_short_minus_heuristic_flow': bootstrap_paired(short_vs_rule),
            'paired_full_reference_minus_heuristic_flow': bootstrap_paired(full_vs_rule),
            'joint_accuracy_episode_weighted': {name: {h: bootstrap_paired(rows) for h, rows in values.items()}
                                                for name, values in joint_by_episode.items()},
            'paired_learned_minus_heuristic_flow': bootstrap_paired(paired_flows),
            'paired_learned_minus_heuristic_decision_regret': bootstrap_paired(paired_regret),
            'cpu_seconds': result['cpu_seconds']}


def run_experiment(profile):
    begin = time.process_time()
    src = download_sources()
    bases = {'tartu': prepare_tartu(make_scenario(src['hospital_map'], seconds_per_tick=10, tiago_config=src['tiago_config'])),
             'clinic': make_clinic_scenario(src['clinic_map'], src['clinic_config'])}
    audit = audit_schedules(bases)
    if audit['cross_split_duplicate_groups']:
        raise ValueError('predeclared task schedules overlap old or new splits: ' + repr(audit))
    params = json.loads(PARAMETERS.read_text(encoding='utf-8'))
    cases_out = []
    for layout, seeds in SEEDS[profile].items():
        for seed in seeds:
            cases, design = make_round2_cases(bases[layout], layout, seed)
            for regime in REGIMES:
                case, pauses, incident = cases[regime]
                policies = {}
                source_result = source_rows = None
                for name in POLICIES:
                    result, rows, controller, elapsed = run_policy(case, seed, pauses, name, params, capture=name=='world')
                    base_case = replace(case, stations={sid: replace(s, outages=()) for sid, s in case.stations.items()})
                    if incident:
                        no_fault, _, _, _ = run_policy(base_case, seed, {}, name, params)
                    else:
                        no_fault = result
                    entry = {'metrics': metrics(result, case.max_ticks, elapsed, incident_end(case, pauses)),
                             'contention': contention_proxy(result, case),
                             'incident': paired_incident(result, no_fault, case, incident),
                             'post_run_trace': post_run_trace(result),
                             'paired_no_fault_trace': post_run_trace(no_fault) if incident else None}
                    if controller:
                        entry['model_steps'] = controller.model_steps
                        entry['fallbacks'] = controller.fallbacks
                        entry['decision_cpu_seconds'] = controller.cpu_seconds
                        entry['online_decisions'] = controller.decisions
                    policies[name] = entry
                    if name == 'world': source_result, source_rows = result, rows
                assert source_result is not None and source_rows is not None
                selected = chosen_decisions(source_rows, incident)
                analyses = [analyze_decision(case, seed, pauses, source_result, source_rows, row, label, params)
                            for row, label in selected]
                config = case_config(case, pauses)
                config['station_position_keys'] = {sid: hashlib.sha256(
                    f'{station.position[0]},{station.position[1]}'.encode()).hexdigest()[:16]
                    for sid, station in case.stations.items()}
                cases_out.append({'layout': layout, 'seed': seed, 'regime': regime, 'config': config,
                                  'incident': incident, 'design_basis': design, 'policies': policies, 'decisions': analyses})
                print(f'{layout} seed={seed} {regime}: decisions={len(analyses)} ' +
                      (f"world_station_wait={policies['world']['incident']['service_wait_ticks']} " if incident else '') +
                      f"world_competition={policies['world']['contention']['occupied_target_robot_ticks']}", flush=True)
    return {'profile': profile, 'scope': 'synthetic CPU simulation; offline references privileged, finite candidates and fixed continuation',
            'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                            'github_sha': os.environ.get('GITHUB_SHA'), 'github_run_id': os.environ.get('GITHUB_RUN_ID')},
            'source_sha256': {k:v[1] for k,v in SOURCES.items()}, 'parameters_path': str(PARAMETERS.relative_to(ROOT)),
            'parameters_sha256': hashlib.sha256(PARAMETERS.read_bytes()).hexdigest(),
            'schedule_audit': audit, 'horizon': HORIZON, 'candidate_budget': BUDGET,
            'meaningful_mean_flow_gap_ticks': MEANINGFUL_FLOW_GAP, 'cases': cases_out,
            'cpu_seconds': time.process_time()-begin}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=tuple(SEEDS), default='pilot')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
            raise FileExistsError(f'refusing to overwrite nonempty output: {args.output}')
        result = run_experiment(args.profile)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output/'round2.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
        summary = summarize(result)
        (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(summary,ensure_ascii=False),flush=True)
        return 0
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        print(f'round2 failed: {exc}', file=sys.stderr)
        return 2

if __name__ == '__main__':
    raise SystemExit(main())
