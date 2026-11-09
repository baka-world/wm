"""Exploratory replay with the same continuation rule on predicted states.

Runs only after the predeclared experiment; no model or score is selected from
this retrospective diagnostic. Candidate-specific reference data stay offline.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import statistics
import sys
import time

import torch

from lab_transport.decision_value import candidates_for, factual_run, nearest_batch, record_observation, replay_branch, truth_trace
from lab_transport.decision_value_round2 import make_round2_cases, pairwise_loss
from lab_transport.dynamics_experiment import check_predicted_step, compare
from lab_transport.dynamics_model import Geometry
from .counterfactual import (HORIZONS, OfflineServiceOracle, OracleUnavailable, actual_durations,
    event_set, load_pr9_model, original_score, rich_observations, rich_transitions, summarize)
from .scoring import workload_components, workload_score
from lab_transport.service_dynamics import RuleMotionNeuralService
from lab_transport.service_fault_data import source_bases
from lab_transport.torch_structured_adapter import ProgressAwareRule
from lab_transport.transition_schema import Assignment


def aligned_rollout(model, initial, action, geometry):
    trace, cumulative = [], {}
    state = initial
    started = time.process_time()
    for step in range(max(HORIZONS)):
        actions = (action,) if step == 0 else nearest_batch(state, candidates_for(state), geometry)
        nxt = model.predict_next(state, actions)
        check_predicted_step(state, nxt, geometry)
        if {s.id for s in nxt.samples} != {s.id for s in initial.samples}:
            raise ValueError('predicted future introduced a new task')
        trace.append(nxt)
        state = nxt
        if step + 1 in HORIZONS:
            cumulative[str(step+1)] = time.process_time() - started
    return trace, cumulative


def replace_scores(decision):
    candidates = decision['candidates']
    decision['comparable'] = all(c['actual_full_complete'] and c['actual_short_complete'] and
                                  c['predictions']['service_oracle']['available'] for c in candidates)
    if not decision['comparable']:
        decision.pop('horizons',None)
        return
    flows = [c['actual_mean_flow'] for c in candidates]
    best = min(range(len(flows)),key=lambda i:(flows[i],i))
    decision['gap'] = max(flows)-min(flows)
    decision['horizons'] = {}
    for horizon in HORIZONS:
        key = str(horizon)
        original = {name:[c['predictions'][name]['horizons'][key]['original_score'] for c in candidates]
                    for name in ('rule','hybrid','service_oracle')}
        original['actual_short_state'] = [c['actual_original_scores'][key] for c in candidates]
        workload = {name:[tuple(c['predictions'][name]['horizons'][key]['workload_score']) for c in candidates]
                    for name in ('rule','hybrid','service_oracle')}
        workload['actual_short_state'] = [tuple(c['actual_workload_scores'][key]) for c in candidates]
        def assessment(scores):
            choices = {name:min(range(len(values)),key=lambda i:(values[i],i)) for name,values in scores.items()}
            return {'choices':choices, 'regret':{name:flows[index]-flows[best] for name,index in choices.items()},
                    'ranking_error':{name:pairwise_loss(values,flows) for name,values in scores.items()},
                    'best_score_tied':{name:sum(x == min(values) for x in values)>1 for name,values in scores.items()},
                    'all_scores_equal':{name:len(set(values)) == 1 for name,values in scores.items()}}
        decision['horizons'][key] = assessment(original)
        decision['horizons'][key]['workload'] = assessment(workload)


def run(base):
    torch.set_num_threads(1)
    bases, source_hashes = source_bases()
    if source_hashes != base['source_sha256']:
        raise ValueError('source hashes differ from registered experiment')
    network, info = load_pr9_model()
    if info != base['checkpoint']:
        raise ValueError('checkpoint differs from registered experiment')
    result = copy.deepcopy(base)
    result['scope'] = ('Exploratory: simulated nearest matching on predicted states; '
                       'candidate-specific service duration oracle remains offline.')
    began = time.process_time()
    scenarios = {}
    for case in result['cases']:
        layout, seed, regime = case['layout'],case['seed'],case['regime']
        group = (layout,seed)
        if group not in scenarios:
            scenarios[group] = make_round2_cases(bases[layout],layout,seed)[0]
        scenario, pauses, _ = scenarios[group][regime]
        factual, rows = factual_run(scenario,seed,pauses)
        rich = rich_observations(rows)
        original_rows = {r.observation.tick:r for r in rows}
        geometry = Geometry(scenario.grid)
        for decision in case['decisions']:
            tick = decision['tick']
            initial = original_rows[tick].observation
            for item in decision['candidates']:
                action = Assignment(**item['action'])
                outcome, branch = replay_branch(scenario,seed,pauses,factual,rows,tick,(action,))
                if not all(s['completed_tick'] is not None for s in outcome['samples']):
                    raise ValueError('previously complete branch became censored')
                flow = statistics.mean(s['completed_tick']-s['release_tick'] for s in outcome['samples'])
                if abs(flow-item['actual_mean_flow']) > 1e-9:
                    raise ValueError('independent replay changed the recorded branch outcome')
                truth, valid = truth_trace(initial,rich_transitions(branch))
                if not valid:
                    raise ValueError('previously valid branch became censored')
                durations, _ = actual_durations(outcome,branch,{s.id for s in initial.samples})
                models = {'rule':ProgressAwareRule(geometry),
                          'hybrid':RuleMotionNeuralService(geometry,network,torch.device('cpu'),info['threshold']),
                          'service_oracle':OfflineServiceOracle(geometry,durations)}
                for name, model in models.items():
                    try:
                        trace, elapsed = aligned_rollout(model,rich[tick],action,geometry)
                    except OracleUnavailable as exc:
                        if name != 'service_oracle':
                            raise
                        item['predictions'][name] = {'available':False,'reason':str(exc)}
                        continue
                    horizons = {}
                    for h in HORIZONS:
                        true_events = event_set(initial,truth[:h])
                        predicted_events = event_set(initial,trace[:h])
                        started = time.process_time()
                        old_score = original_score(initial,action,trace[h-1],geometry)
                        old_time = time.process_time()-started
                        started = time.process_time()
                        new_score = workload_score(initial,trace[:h],geometry)
                        new_time = time.process_time()-started
                        horizons[str(h)] = {'original_score':old_score,'rollout_cpu_seconds':elapsed[str(h)],
                            'score_cpu_seconds':old_time,'workload_score':list(new_score),
                            'workload_components':workload_components(initial,trace[:h],geometry),
                            'workload_score_cpu_seconds':new_time,
                            'known_entity_counts':compare(initial,truth[h-1],trace[h-1]),
                            'service_events':{'actual':len(true_events),'predicted':len(predicted_events),
                                              'correct':len(true_events & predicted_events)},
                            'terminal_state':record_observation(trace[h-1])}
                    item['predictions'][name] = {'available':True,'horizons':horizons}
            replace_scores(decision)
        print(f"{base['profile']} aligned {layout} seed={seed} {regime}",flush=True)
    result['cpu_seconds'] = time.process_time()-began
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    try:
        if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
            raise FileExistsError(f'refusing to overwrite: {args.output}')
        base = json.loads(args.input.read_text(encoding='utf-8'))
        result = run(base)
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/'raw.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        summary = summarize(result['cases'])
        (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(summary,ensure_ascii=False),flush=True)
        return 0
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        print(f'aligned continuation diagnostic failed: {exc}',file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
