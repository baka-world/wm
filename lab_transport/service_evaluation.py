"""Paired factual prediction, same-state action and closed-loop CPU evaluation."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import torch

from .decision_value_round2 import (bootstrap_paired, chosen_decisions, contention_proxy, make_round2_cases,
                                    pending_at, service_sessions)
from .dynamics_experiment import (check_predicted_step, compare, eligible_windows, merge_counts, rates, zero_counts)
from .dynamics_model import Geometry
from .multisite_experiment import discrete_safety_violations
from .service_dynamics import RuleMotionNeuralService, ThresholdDynamics
from .service_fault_data import REGIMES, source_bases
from .service_learning import contract_hashes, read_episodes, service_keys
from .simulation import Simulator
from .state_v2 import ObservableHistory
from .torch_rollout import RolloutPolicy
from .torch_structured_adapter import ProgressAwareRule, build_model
from .transition_schema import Assignment

MODELS = ('rule', 'original', 'weighted', 'calibrated', 'sequence_uncalibrated', 'improved', 'hybrid')
HORIZON, BUDGET = 3, 8


def load_networks(dataset, training, device):
    report = json.loads((training/'training.json').read_text(encoding='utf-8'))
    digest = hashlib.sha256((dataset/'manifest.json').read_bytes()).hexdigest()
    if report['dataset_manifest_sha256'] != digest:
        raise ValueError('training dataset hash differs')
    if report['contract_code_sha256'] != contract_hashes():
        raise ValueError('training contract source hash differs')
    networks = {}
    for name in ('original', 'weighted', 'sequence'):
        path = training/f'{name}.pt'
        if hashlib.sha256(path.read_bytes()).hexdigest() != report['variants'][name]['checkpoint_sha256']:
            raise ValueError('training checkpoint checksum differs')
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        if (checkpoint['dataset_manifest_sha256'] != digest or checkpoint['variant'] != name or
                checkpoint['contract_code_sha256'] != report['contract_code_sha256']):
            raise ValueError('checkpoint metadata differs')
        network = build_model().to(device)
        network.load_state_dict(checkpoint['model_state'])
        networks[name] = network.eval()
    return report, networks


def models_for(geometry, networks, report, device):
    weighted_threshold = report['variants']['weighted']['calibration']['threshold']
    sequence_threshold = report['variants']['sequence']['calibration']['threshold']
    return {'rule': ProgressAwareRule(geometry),
            'original': ThresholdDynamics(geometry, networks['original'], device, .5),
            'weighted': ThresholdDynamics(geometry, networks['weighted'], device, .5),
            'calibrated': ThresholdDynamics(geometry, networks['weighted'], device, weighted_threshold),
            'sequence_uncalibrated': ThresholdDynamics(geometry, networks['sequence'], device, .5),
            'improved': ThresholdDynamics(geometry, networks['sequence'], device, sequence_threshold),
            'hybrid': RuleMotionNeuralService(geometry, networks['sequence'], device, sequence_threshold)}


def new_event_counts():
    return {'opportunities': 0, 'actual': 0, 'predicted': 0, 'tp': 0}


def add_events(counts, row, predicted):
    actual = {s.id:s for s in row.next_observation.samples}
    forecast = {s.id:s for s in predicted.samples}
    for _, sid, phase, _ in service_keys(row):
        target = 'in_transit' if phase == 'to_pickup' else 'completed'
        a, p = actual[sid].status == target, forecast[sid].status == target
        counts['opportunities'] += 1
        counts['actual'] += int(a)
        counts['predicted'] += int(p)
        counts['tp'] += int(a and p)


def prediction_episode(ep, geometry, networks, report, device):
    models = models_for(geometry, networks, report, device)
    one = {name:zero_counts() for name in MODELS}
    service = {name:new_event_counts() for name in MODELS}
    paired_events = {name:{'both_correct':0, 'only_model_correct':0,
                           'only_rule_correct':0, 'both_wrong':0} for name in MODELS if name != 'rule'}
    for row in ep.rows:
        outputs = {}
        for name, model in models.items():
            predicted = model.predict_next(row.observation, row.assignments)
            check_predicted_step(row.observation, predicted, geometry)
            merge_counts(one[name], compare(row.observation, row.next_observation, predicted))
            add_events(service[name], row, predicted)
            outputs[name] = {s.id:s for s in predicted.samples}
        actual = {s.id:s for s in row.next_observation.samples}
        for _, sid, phase, _ in service_keys(row):
            target = 'in_transit' if phase == 'to_pickup' else 'completed'
            truth = actual[sid].status == target
            rule_correct = (outputs['rule'][sid].status == target) == truth
            for name in paired_events:
                model_correct = (outputs[name][sid].status == target) == truth
                label = ('both_correct' if model_correct and rule_correct else
                         'only_model_correct' if model_correct else
                         'only_rule_correct' if rule_correct else 'both_wrong')
                paired_events[name][label] += 1
    multi = {}
    for horizon in (3, 5):
        windows, coverage = eligible_windows(ep.rows, horizon)
        windows = windows[::5]
        coverage['evaluated'] = len(windows)
        totals = {name:zero_counts() for name in MODELS}
        for window in windows:
            initial, actual = window[0].observation, window[-1].next_observation
            for name, model in models.items():
                predicted = initial
                for index, row in enumerate(window):
                    nxt = model.predict_next(predicted, row.assignments if index == 0 else ())
                    check_predicted_step(predicted, nxt, geometry)
                    predicted = nxt
                merge_counts(totals[name], compare(initial, actual, predicted))
        multi[str(horizon)] = {'coverage':coverage, 'counts':totals}
    return {'split':ep.split, 'layout':ep.layout, 'seed':ep.seed, 'regime':ep.regime,
            'behavior_policy':ep.policy, 'one':one, 'service':service,
            'paired_service_events':paired_events, 'multi':multi}


def prediction_summary(cases):
    groups = defaultdict(list)
    for case in cases:
        groups[(case['split'], 'all')].append(case)
        groups[(case['split'], case['regime'])].append(case)
    result = {}
    for (split, regime), items in groups.items():
        table = {}
        for name in MODELS:
            counts = zero_counts()
            events = new_event_counts()
            paired = defaultdict(int)
            multi = {str(h): zero_counts() for h in (3, 5)}
            cover = {str(h): defaultdict(int) for h in (3, 5)}
            for item in items:
                merge_counts(counts, item['one'][name])
                for k,v in item['service'][name].items():
                    events[k] += v
                if name != 'rule':
                    for k,v in item['paired_service_events'][name].items():
                        paired[k] += v
                for h in ('3', '5'):
                    merge_counts(multi[h], item['multi'][h]['counts'][name])
                    for k,v in item['multi'][h]['coverage'].items():
                        cover[h][k] += v
            recall = events['tp']/events['actual'] if events['actual'] else None
            precision = events['tp']/events['predicted'] if events['predicted'] else None
            table[name] = {'one_step_counts':counts, 'one_step_rates':rates(counts),
                           'service_event_counts':events, 'service_recall':recall, 'service_precision':precision,
                           'paired_service_vs_rule':dict(paired) if name != 'rule' else None,
                           'multi_step': {h:{'counts':multi[h], 'rates':rates(multi[h]), 'coverage':dict(cover[h])}
                                          for h in ('3','5')}}
        result[f'{split}/{regime}'] = {'episodes':len(items), 'models':table}
    return result


class TimedPolicy(RolloutPolicy):
    def __init__(self, dynamics, geometry, observer=None):
        super().__init__(dynamics, geometry, HORIZON, BUDGET, observer)
        self.cpu_seconds = 0.

    def __call__(self, observation, candidates):
        started = time.process_time()
        result = super().__call__(observation, candidates)
        self.cpu_seconds += time.process_time() - started
        return result


def candidates_for(observation):
    robots = sorted((r for r in observation.robots if r.task_id is None), key=lambda r:r.id)
    samples = sorted((s for s in observation.samples if s.status == 'waiting'), key=lambda s:(s.release_tick,s.id))
    return tuple(Assignment(r.id, s.id) for r in robots for s in samples)


def same_state_comparison(episodes, geometries, networks, report, device, coverage):
    incidents = {(c['layout'], c['seed'], c['regime']):c['incident'] for c in coverage}
    output = []
    for ep in episodes:
        if ep.split not in ('test_tartu','test_clinic') or ep.policy != 'nearest':
            continue
        incident = incidents[(ep.layout, ep.seed, ep.regime)]
        models = models_for(geometries[ep.layout], networks, report, device)
        for row, label in chosen_decisions(ep.rows, incident):
            candidates = candidates_for(row.observation)
            if not candidates:
                continue
            selected = {}
            for name, model in models.items():
                policy = TimedPolicy(model, geometries[ep.layout])
                choice = policy(row.observation, candidates)
                selected[name] = {'robot_id':choice[0].robot_id, 'sample_id':choice[0].sample_id,
                                  'fallbacks':policy.fallbacks, 'model_steps':policy.model_steps,
                                  'cpu_seconds':policy.cpu_seconds, 'decision':policy.log[-1]}
            output.append({'split':ep.split, 'layout':ep.layout, 'seed':ep.seed, 'regime':ep.regime,
                           'tick':row.observation.tick, 'sampling':label,
                           'candidates':[{'robot_id':c.robot_id,'sample_id':c.sample_id} for c in candidates],
                           'selected':selected})
    return output


def closed_loop(bases, splits, networks, report, device):
    cases = []
    for split in ('test_tartu', 'test_clinic'):
        layout, seeds = splits[split]
        for seed in seeds:
            scenarios, _ = make_round2_cases(bases[layout], layout, seed)
            for regime in REGIMES:
                case, pauses, incident = scenarios[regime]
                geometry = Geometry(case.grid)
                models = models_for(geometry, networks, report, device)
                variants = {}
                for name, model in models.items():
                    observer = ObservableHistory()
                    policy = TimedPolicy(model, geometry, observer)
                    result = Simulator(case, 'world', seed, motion_pauses=pauses, collect_metrics=True).run(
                        dispatch_policy=policy, on_observation=observer.observe)
                    sessions = service_sessions(result, case, incident)
                    variants[name] = {'summary':result['summary'],
                        'station_wait_ticks':sum(s['unavailable_wait_ticks'] for s in sessions),
                        'affected_tasks':len({s['sample_id'] for s in sessions if s['unavailable_wait_ticks']}),
                        'pending_at_recovery':pending_at(result, case, incident['end']) if incident else None,
                        'competition':contention_proxy(result, case),
                        'safety_violations':discrete_safety_violations(result),
                        'dispatch_calls':len(policy.log), 'model_steps':policy.model_steps,
                        'fallbacks':policy.fallbacks, 'decision_cpu_seconds':policy.cpu_seconds,
                        'decision_mean_ms':1000*policy.cpu_seconds/len(policy.log) if policy.log else None,
                        'decisions':policy.log,
                        'sample_outcomes':[{k:s[k] for k in ('id','release_tick','assigned_tick','picked_tick','completed_tick')}
                                           for s in result['samples']]}
                cases.append({'split':split,'layout':layout,'seed':seed,'regime':regime,
                              'incident':incident,'models':variants})
                print(f'closed-loop {split} {seed} {regime} completed', flush=True)
    return cases


def summarize_closed_loop(cases):
    output = {}
    for regime in ('all', *REGIMES):
        selected = [case for case in cases if regime == 'all' or case['regime'] == regime]
        table = {}
        for model in MODELS:
            values = [c['models'][model] for c in selected]
            complete = [v for v in values if v['summary']['status'] == 'completed']
            flow = [v['summary']['mean_completion_ticks'] for v in complete]
            times = [v['decision_mean_ms'] for v in values if v['decision_mean_ms'] is not None]
            paired = [(c['layout'],c['seed'],c['models'][model]['summary']['mean_completion_ticks'] -
                       c['models']['rule']['summary']['mean_completion_ticks']) for c in selected
                      if c['models'][model]['summary']['status'] == c['models']['rule']['summary']['status'] == 'completed']
            diffs = [x[2] for x in paired]
            table[model] = {'episodes':len(values), 'complete':len(complete),
                            'mean_flow_completed_only':statistics.mean(flow) if flow else None,
                            'mean_decision_cpu_ms':statistics.mean(times) if times else None,
                            'station_wait_ticks':sum(v['station_wait_ticks'] for v in values),
                            'safety_violations':sum(v['safety_violations'] for v in values),
                            'fallbacks':sum(v['fallbacks'] for v in values),
                            'paired_flow_minus_rule':bootstrap_paired(paired),
                            'paired_wins':sum(x < -1e-9 for x in diffs),
                            'paired_losses':sum(x > 1e-9 for x in diffs),
                            'paired_ties':sum(abs(x) <= 1e-9 for x in diffs)}
        output[regime] = table
    return output


def run(dataset: Path, training: Path, output: Path, device_name='cpu'):
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'refusing to overwrite evaluation output: {output}')
    if device_name not in ('cpu','cuda') or (device_name == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('requested device unavailable')
    torch.set_num_threads(1)
    device = torch.device(device_name)
    manifest, geometries, episodes, checked = read_episodes(dataset)
    report, networks = load_networks(dataset, training, device)
    coverage = json.loads((dataset/'coverage.json').read_text(encoding='utf-8'))
    prediction_cases = []
    for ep in episodes:
        if ep.split in ('test_tartu','test_clinic'):
            prediction_cases.append(prediction_episode(ep, geometries[ep.layout], networks, report, device))
            print(f'prediction {ep.split} {ep.seed} {ep.regime} {ep.policy}', flush=True)
    same_state = same_state_comparison(episodes, geometries, networks, report, device, coverage)
    bases, source_hashes = source_bases()
    if source_hashes != manifest['source_sha256']:
        raise ValueError('public source hashes differ from dataset')
    cases = closed_loop(bases, manifest['schedule_audit']['seed_selection'], networks, report, device)
    summary = {'scope':'Synthetic factual test and paired closed loop. No future input; no clinical inference.',
               'device':str(device), 'data_check':checked, 'horizon':HORIZON, 'candidate_budget':BUDGET,
               'factual_prediction':prediction_summary(prediction_cases),
               'closed_loop':summarize_closed_loop(cases), 'same_state_decisions':len(same_state)}
    output.mkdir(parents=True, exist_ok=True)
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (output/'prediction-cases.json').write_text(json.dumps(prediction_cases,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (output/'same-state.json').write_text(json.dumps(same_state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (output/'closed-loop-cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    args = parser.parse_args()
    try:
        result = run(args.dataset,args.training,args.output,args.device)
        print(json.dumps({'same_state_decisions':result['same_state_decisions'],
                          'closed_loop':result['closed_loop']['all']},ensure_ascii=False),flush=True)
        return 0
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        print(f'service evaluation failed: {exc}',file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
