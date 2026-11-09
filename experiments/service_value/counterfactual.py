"""CPU-only, candidate-specific offline service-duration counterfactuals."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import torch

from lab_transport.decision_value import (factual_run, record_observation, replay_branch, truth_trace)
from lab_transport.decision_value_round2 import (bootstrap_paired, burst_case, chosen_decisions, make_round2_cases,
                                    pairwise_loss, quantile)
from lab_transport.dynamics_experiment import check_predicted_step, compare
from lab_transport.dynamics_model import Geometry, _intent
from lab_transport.dynamics_revision import SPLITS_V2
from lab_transport.multisite_experiment import discrete_safety_violations, make_case
from lab_transport.service_dynamics import RuleMotionNeuralService
from lab_transport.service_evaluation import candidates_for
from lab_transport.service_fault_data import source_bases, select_seeds
from .scoring import workload_components, workload_score
from lab_transport.service_learning import contract_hashes
from lab_transport.state_v2 import ObservableHistory, ObservationV2
from lab_transport.torch_structured_adapter import ProgressAwareRule, build_model
from lab_transport.training_data import _task_hash
from lab_transport.transition_schema import Transition

SEEDS = {'development': {'tartu': (31002, 31005), 'clinic': (32000, 32001)},
         'confirmation': {'tartu': (33000, 33003), 'clinic': (34000, 34001)}}
REGIMES = ('normal', 'burst', 'outage_processing', 'outage_arrival')
HORIZONS = (3, 5, 10, 20)
BUDGET = 8
CHECKPOINT_DIR = Path('examples/evidence/service-dynamics-20261109')


def audit_schedules(bases):
    seen = defaultdict(set)
    def add(group, layout, case):
        seen[_task_hash(case, layout)].add(group)
    for _, (layout, seeds) in SPLITS_V2.items():
        for seed in seeds:
            add('pr8', layout, make_case(bases[layout], layout, seed, 'normal'))
    old, _ = select_seeds(bases)
    for _, (layout, seeds) in old.items():
        for seed in seeds:
            add('pr9', layout, make_case(bases[layout], layout, seed, 'normal'))
            add('pr9', layout, burst_case(bases[layout], layout, seed))
    for group, layouts in SEEDS.items():
        for layout, seeds in layouts.items():
            for seed in seeds:
                add(group, layout, make_case(bases[layout], layout, seed, 'normal'))
                add(group, layout, burst_case(bases[layout], layout, seed))
    overlaps = [sorted(groups) for groups in seen.values() if len(groups) > 1]
    return {'unique_task_patterns':len(seen), 'cross_group_duplicates':len(overlaps), 'overlaps':overlaps}


def load_pr9_model():
    """Load only the frozen PR #9 checkpoint after integrity verification."""
    path = CHECKPOINT_DIR/'sequence.pt'
    report = json.loads((CHECKPOINT_DIR/'training.json').read_text(encoding='utf-8'))
    manifest_hash = hashlib.sha256((CHECKPOINT_DIR/'dataset-manifest.json').read_bytes()).hexdigest()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != report['variants']['sequence']['checkpoint_sha256']:
        raise ValueError('PR #9 checkpoint checksum mismatch')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if (checkpoint['variant'] != 'sequence' or checkpoint['dataset_manifest_sha256'] != manifest_hash
            or checkpoint['contract_code_sha256'] != contract_hashes()):
        raise ValueError('PR #9 checkpoint or source contract mismatch')
    model = build_model()
    model.load_state_dict(checkpoint['model_state'])
    threshold = checkpoint['calibration']['threshold']
    if threshold != report['variants']['sequence']['calibration']['threshold']:
        raise ValueError('PR #9 threshold mismatch')
    return model.eval(), {'checkpoint_sha256':digest, 'dataset_manifest_sha256':manifest_hash,
                          'threshold':threshold, 'best_epoch':checkpoint['best_epoch']}


def rich_observations(rows):
    observer = ObservableHistory()
    return {row.observation.tick:observer.observe(row.observation) for row in rows}


def rich_transitions(rows):
    observer = ObservableHistory()
    enriched = []
    for row in rows:
        current = observer.observe(row.observation)
        future = observer.observe(row.next_observation)
        enriched.append(Transition(current, row.assignments, future, row.next_boundary, row.termination))
    return enriched


def actual_durations(result, rows, known):
    """Recover completed service durations from this branch's events/history.

    Only an offline reference may use this. No simulator countdown is read.
    """
    observed = {r.observation.tick:{s.id:s.available for s in r.observation.stations} for r in rows}
    starts = {}
    durations, missing = {}, []
    for event in result['events']:
        if event.get('sample_id') not in known or event['type'] not in ('service_started', 'service_finished'):
            continue
        key = (event['sample_id'], event['phase'], event['station_id'])
        if event['type'] == 'service_started':
            if key in starts:
                raise ValueError('duplicate service start in branch')
            starts[key] = event['tick']
        elif key in starts:
            start, end = starts.pop(key), event['tick']
            if any(t not in observed or key[2] not in observed[t] for t in range(start, end)):
                missing.append(key)
                continue
            active = sum(observed[t][key[2]] for t in range(start, end))
            if active < 1:
                raise ValueError('completed service has no active duration')
            durations[key] = active
    missing.extend(starts)
    return durations, missing


class OracleUnavailable(ValueError):
    pass


class OfflineServiceOracle:
    """Privileged, per-branch actual service durations; not an online policy."""
    def __init__(self, geometry, durations):
        self.rule = ProgressAwareRule(geometry)
        self.durations = durations

    def predict_next(self, observation, assignments):
        if not isinstance(observation, ObservationV2):
            raise ValueError('oracle requires an observable history-derived state')
        intents = _intent(observation, assignments)
        samples = {s.id:s for s in observation.samples}
        stations = {s.id:s for s in observation.stations}
        overrides = {}
        for robot in observation.robots:
            phase, sample_id = intents[robot.id]
            if sample_id is None:
                continue
            sample = samples[sample_id]
            station_id = sample.pickup if phase == 'to_pickup' else sample.dropoff
            if robot.position == stations[station_id].position:
                key = (sample_id, phase, station_id)
                if key not in self.durations:
                    raise OracleUnavailable('branch has no observed complete duration for this service')
                overrides[station_id] = self.durations[key]
        adjusted = replace(observation, stations=tuple(replace(s, expected_service_ticks=overrides[s.id],
            observed_effective_service_mean=float(overrides[s.id])) if s.id in overrides else s
            for s in observation.stations))
        predicted = self.rule.predict_next(adjusted, assignments)
        old = {s.id:s for s in observation.stations}
        # No privileged duration is included in the scored state's prior or history.
        restored = tuple(replace(s, expected_service_ticks=old[s.id].expected_service_ticks,
                                 observed_effective_service_mean=old[s.id].observed_effective_service_mean)
                         for s in predicted.stations)
        return replace(predicted, stations=restored)


def original_score(initial, action, terminal, geometry):
    robot = next(r for r in terminal.robots if r.id == action.robot_id)
    sample = next(s for s in terminal.samples if s.id == action.sample_id)
    stations = {s.id:s.position for s in terminal.stations}
    if sample.status == 'completed':
        return 0
    if sample.status == 'in_transit':
        return geometry.distance(robot.position, stations[sample.dropoff])
    return (geometry.distance(robot.position, stations[sample.pickup]) +
            geometry.distance(stations[sample.pickup], stations[sample.dropoff]))


def event_set(initial, trace):
    known = {s.id:s.status for s in initial.samples}
    events = set()
    before = dict(known)
    for state in trace:
        samples = {s.id:s for s in state.samples}
        for sid in known:
            after = samples[sid].status
            if before[sid] in ('waiting', 'assigned') and after == 'in_transit':
                events.add((sid, 'to_pickup', state.tick))
            if before[sid] == 'in_transit' and after == 'completed':
                events.add((sid, 'to_dropoff', state.tick))
            before[sid] = after
    return events


def rollout(model, initial, action, geometry):
    trace, cumulative = [], {}
    current = initial
    started = time.process_time()
    for i in range(max(HORIZONS)):
        nxt = model.predict_next(current, (action,) if i == 0 else ())
        check_predicted_step(current, nxt, geometry)
        if {s.id for s in nxt.samples} != {s.id for s in initial.samples}:
            raise ValueError('model changed the set of observed tasks')
        trace.append(nxt)
        current = nxt
        if i + 1 in HORIZONS:
            cumulative[str(i + 1)] = time.process_time() - started
    return trace, cumulative


def evaluate_decision(case, seed, pauses, factual, rows, initial, rich, label, network, threshold):
    geo = Geometry(case.grid)
    actions = candidates_for(initial)
    if not actions:
        raise ValueError('decision has no feasible assignment')
    actions = actions[:BUDGET]
    known = {s.id for s in initial.samples}
    candidates = []
    for action in actions:
        result, branch = replay_branch(case, seed, pauses, factual, rows, initial.tick, (action,))
        truth, valid = truth_trace(initial, rich_transitions(branch))
        full = all(s['completed_tick'] is not None for s in result['samples'])
        flow = (statistics.mean(s['completed_tick'] - s['release_tick'] for s in result['samples']) if full else None)
        durations, missing = actual_durations(result, branch, known)
        item = {'action':asdict(action), 'actual_full_complete':full, 'actual_mean_flow':flow,
                'actual_discrete_safety_violations':discrete_safety_violations(result),
                'actual_sample_outcomes':[{k:s[k] for k in ('id','release_tick','assigned_tick','picked_tick','completed_tick')}
                                          for s in result['samples']],
                'actual_short_complete':valid, 'offline_service_duration_count':len(durations),
                'offline_duration_missing':[list(x) for x in missing]}
        if valid:
            item['actual_events'] = {str(h):sorted([list(x) for x in event_set(initial, truth[:h])]) for h in HORIZONS}
            item['actual_original_scores'] = {str(h):original_score(initial, action, truth[h-1], geo) for h in HORIZONS}
            item['actual_workload_scores'] = {str(h):list(workload_score(initial, truth[:h], geo)) for h in HORIZONS}
            item['actual_workload_components'] = {str(h):workload_components(initial, truth[:h], geo) for h in HORIZONS}
            models = {'rule':ProgressAwareRule(geo),
                      'hybrid':RuleMotionNeuralService(geo, network, torch.device('cpu'), threshold),
                      'service_oracle':OfflineServiceOracle(geo, durations)}
            item['predictions'] = {}
            for name, model in models.items():
                try:
                    trace, elapsed = rollout(model, rich, action, geo)
                except OracleUnavailable as exc:
                    if name != 'service_oracle':
                        raise
                    item['predictions'][name] = {'available':False, 'reason':str(exc)}
                    continue
                horizons = {}
                for h in HORIZONS:
                    actual_events = event_set(initial, truth[:h])
                    predicted_events = event_set(initial, trace[:h])
                    started = time.process_time()
                    score = original_score(initial, action, trace[h-1], geo)
                    score_cpu = time.process_time() - started
                    started = time.process_time()
                    new_score = workload_score(initial, trace[:h], geo)
                    new_cpu = time.process_time() - started
                    horizons[str(h)] = {'original_score':score, 'rollout_cpu_seconds':elapsed[str(h)],
                        'score_cpu_seconds':score_cpu, 'workload_score':list(new_score),
                        'workload_components':workload_components(initial, trace[:h], geo),
                        'workload_score_cpu_seconds':new_cpu,
                        'known_entity_counts':compare(initial, truth[h-1], trace[h-1]),
                        'service_events':{'actual':len(actual_events), 'predicted':len(predicted_events),
                                          'correct':len(actual_events & predicted_events)},
                        'terminal_state':record_observation(trace[h-1])}
                item['predictions'][name] = {'available':True, 'horizons':horizons}
            item['actual_states'] = {str(h):record_observation(truth[h-1]) for h in HORIZONS}
        candidates.append(item)
    # Waiting is a legal simulator action but is not scored by PR #9.
    waited, _ = replay_branch(case, seed, pauses, factual, rows, initial.tick, ())
    wait_full = all(s['completed_tick'] is not None for s in waited['samples'])
    wait_flow = statistics.mean(s['completed_tick'] - s['release_tick'] for s in waited['samples']) if wait_full else None
    original_total = len(candidates_for(initial))
    entry = {'tick':initial.tick, 'sampling':label, 'initial':record_observation(initial),
             'candidate_total':original_total, 'candidate_evaluated':len(actions),
             'budget_truncated':original_total > len(actions), 'candidates':candidates,
             'wait_reference':{'full_complete':wait_full, 'mean_flow':wait_flow},
             'comparable':all(c['actual_full_complete'] and c['actual_short_complete'] and
                              c['predictions']['service_oracle']['available'] for c in candidates)}
    if not entry['comparable']:
        return entry
    flows = [c['actual_mean_flow'] for c in candidates]
    entry['gap'] = max(flows)-min(flows)
    entry['wait_minus_best_assignment'] = wait_flow-min(flows) if wait_full else None
    entry['horizons'] = {}
    for h in HORIZONS:
        key = str(h)
        scores = {name:[c['predictions'][name]['horizons'][key]['original_score'] for c in candidates]
                  for name in ('rule','hybrid','service_oracle')}
        scores['actual_short_state'] = [c['actual_original_scores'][key] for c in candidates]
        best = min(range(len(flows)), key=lambda i:(flows[i],i))
        choices = {name:min(range(len(values)), key=lambda i:(values[i],i)) for name,values in scores.items()}
        entry['horizons'][key] = {'choices':choices,
            'regret':{name:flows[index]-flows[best] for name,index in choices.items()},
            'ranking_error':{name:pairwise_loss(values,flows) for name,values in scores.items()},
            'best_score_tied':{name:sum(x == min(values) for x in values)>1 for name,values in scores.items()},
            'all_scores_equal':{name:len(set(values)) == 1 for name,values in scores.items()}}
        workload = {name:[tuple(c['predictions'][name]['horizons'][key]['workload_score']) for c in candidates]
                    for name in ('rule','hybrid','service_oracle')}
        workload['actual_short_state'] = [tuple(c['actual_workload_scores'][key]) for c in candidates]
        selected = {name:min(range(len(values)), key=lambda i:(values[i],i)) for name,values in workload.items()}
        entry['horizons'][key]['workload'] = {'choices':selected,
            'regret':{name:flows[index]-flows[best] for name,index in selected.items()},
            'ranking_error':{name:pairwise_loss(values,flows) for name,values in workload.items()},
            'best_score_tied':{name:sum(x == min(values) for x in values)>1 for name,values in workload.items()},
            'all_scores_equal':{name:len(set(values)) == 1 for name,values in workload.items()}}
    return entry


def summarize(cases):
    models = ('rule','hybrid','service_oracle','actual_short_state')
    grouped = defaultdict(list)
    metrics = {str(h):{name:[] for name in models} for h in HORIZONS}
    workload_metrics = {str(h):{name:[] for name in models} for h in HORIZONS}
    decisions = [d for c in cases for d in c['decisions'] if d['comparable']]
    for case in cases:
        ds = [d for d in case['decisions'] if d['comparable']]
        if not ds:
            continue
        for h in HORIZONS:
            for name in models:
                metrics[str(h)][name].append((case['layout'],case['seed'],
                    statistics.mean(d['horizons'][str(h)]['regret'][name] for d in ds)))
                workload_metrics[str(h)][name].append((case['layout'],case['seed'],
                    statistics.mean(d['horizons'][str(h)]['workload']['regret'][name] for d in ds)))
    summaries, workload_summaries = {}, {}
    for h in HORIZONS:
        key = str(h)
        detail = {}
        for name in models:
            valid_errors = [d['horizons'][key]['ranking_error'][name]['rate'] for d in decisions
                            if d['horizons'][key]['ranking_error'][name]['rate'] is not None]
            detail[name] = {'mean_regret':bootstrap_paired(metrics[key][name]),
                'selected_flow_optimal':sum(d['horizons'][key]['regret'][name] < 1e-9 for d in decisions),
                'mean_pairwise_error_descriptive':statistics.mean(valid_errors) if valid_errors else None,
                'best_tie_points':sum(d['horizons'][key]['best_score_tied'][name] for d in decisions),
                'all_equal_points':sum(d['horizons'][key]['all_scores_equal'][name] for d in decisions),
                'choice_differs_from_rule':sum(d['horizons'][key]['choices'][name] != d['horizons'][key]['choices']['rule'] for d in decisions)}
        summaries[key] = detail
        new_detail = {}
        for name in models:
            valid_errors = [d['horizons'][key]['workload']['ranking_error'][name]['rate'] for d in decisions
                            if d['horizons'][key]['workload']['ranking_error'][name]['rate'] is not None]
            new_detail[name] = {'mean_regret':bootstrap_paired(workload_metrics[key][name]),
                'selected_flow_optimal':sum(d['horizons'][key]['workload']['regret'][name] < 1e-9 for d in decisions),
                'mean_pairwise_error_descriptive':statistics.mean(valid_errors) if valid_errors else None,
                'best_tie_points':sum(d['horizons'][key]['workload']['best_score_tied'][name] for d in decisions),
                'all_equal_points':sum(d['horizons'][key]['workload']['all_scores_equal'][name] for d in decisions),
                'choice_differs_from_rule':sum(d['horizons'][key]['workload']['choices'][name] !=
                                               d['horizons'][key]['workload']['choices']['rule'] for d in decisions)}
        workload_summaries[key] = new_detail
    return {'cases':len(cases), 'decisions_total':sum(len(c['decisions']) for c in cases),
            'comparable_decisions':len(decisions), 'candidate_branches':sum(len(d['candidates']) for d in decisions),
            'candidate_budget_truncated_points':sum(d['budget_truncated'] for d in decisions),
            'gap_at_least_one':sum(d['gap'] >= 1 for d in decisions),
            'wait_better_than_all_assignments':sum(d['wait_minus_best_assignment'] is not None and
                                                    d['wait_minus_best_assignment'] < -1e-9 for d in decisions),
            'horizons':summaries, 'workload_horizons':workload_summaries}


def run(profile):
    torch.set_num_threads(1)
    began = time.process_time()
    bases, source_hashes = source_bases()
    audit = audit_schedules(bases)
    if audit['cross_group_duplicates']:
        raise ValueError('predeclared task schedules overlap: ' + repr(audit))
    network, model_info = load_pr9_model()
    cases = []
    for layout, seeds in SEEDS[profile].items():
        for seed in seeds:
            scenarios, design = make_round2_cases(bases[layout],layout,seed)
            for regime in REGIMES:
                case, pauses, incident = scenarios[regime]
                factual, rows = factual_run(case,seed,pauses)
                rich = rich_observations(rows)
                analyses = [evaluate_decision(case,seed,pauses,factual,rows,row.observation,
                                               rich[row.observation.tick],label,network,model_info['threshold'])
                            for row,label in chosen_decisions(rows,incident)]
                cases.append({'layout':layout,'seed':seed,'regime':regime,'incident':incident,
                              'design':design,'decisions':analyses})
                print(f'{profile} {layout} seed={seed} {regime}: decisions={len(analyses)}',flush=True)
    return {'profile':profile, 'scope':'Independent candidate replay; service oracle and short truth are offline only.',
            'source_sha256':source_hashes,'task_schedule_audit':audit,'checkpoint':model_info,
            'horizons':HORIZONS,'candidate_budget':BUDGET,'cases':cases,'cpu_seconds':time.process_time()-began}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=tuple(SEEDS), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
            raise FileExistsError(f'refusing to overwrite output: {args.output}')
        result = run(args.profile)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output/'raw.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        summary = summarize(result['cases'])
        (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(summary,ensure_ascii=False),flush=True)
        return 0
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        print(f'counterfactual experiment failed: {exc}',file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
