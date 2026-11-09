"""CPU-only online paired evaluation. Privileged references cannot be policies."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import sys
import time

import torch

from lab_transport.decision_value import case_config
from lab_transport.decision_value_round2 import bootstrap_paired, contention_proxy, make_round2_cases, pending_at, quantile, service_sessions
from lab_transport.dynamics_experiment import check_predicted_step
from lab_transport.dynamics_model import Geometry
from lab_transport.multisite_experiment import discrete_safety_violations
from .counterfactual import (BUDGET, REGIMES, SEEDS, OfflineServiceOracle, audit_schedules,
                                     load_pr9_model, original_score)
from .scoring import workload_score
from lab_transport.service_dynamics import RuleMotionNeuralService
from lab_transport.service_fault_data import source_bases
from lab_transport.simulation import Simulator
from lab_transport.state_v2 import ObservableHistory
from lab_transport.torch_structured_adapter import ProgressAwareRule

HORIZON = 20
METHODS = ('rule_original','rule_workload','hybrid_original','hybrid_workload')


class OnlinePolicy:
    """Candidate budgets and current/predicted states only; no simulator handle."""
    def __init__(self, model, geometry, observer, score):
        if isinstance(model, OfflineServiceOracle):
            raise ValueError('offline privileged service information is forbidden in an online policy')
        if score not in ('original','workload'):
            raise ValueError('unknown scoring function')
        self.model, self.geometry, self.observer, self.score = model, geometry, observer, score
        self.decisions, self.times, self.fallbacks, self.model_steps = [], [], 0, 0

    def __call__(self, observation, candidates):
        started = time.process_time()
        observation = self.observer.observe(observation)
        records = []
        try:
            for action in candidates[:BUDGET]:
                state, trace = observation, []
                for i in range(HORIZON):
                    nxt = self.model.predict_next(state, (action,) if i == 0 else ())
                    self.model_steps += 1
                    check_predicted_step(state, nxt, self.geometry)
                    trace.append(nxt)
                    state = nxt
                score = (original_score(observation,action,state,self.geometry) if self.score == 'original'
                         else workload_score(observation,trace,self.geometry))
                records.append({'action':asdict(action),'score':score})
            selected = min(range(len(records)),key=lambda i:(records[i]['score'],i))
        except (ValueError, RuntimeError, TypeError, KeyError, StopIteration, OverflowError) as exc:
            self.fallbacks += 1
            self.times.append(time.process_time()-started)
            self.decisions.append({'tick':observation.tick,'fallback':type(exc).__name__,
                                   'candidate_count':len(candidates),'evaluated':records,'selected':asdict(candidates[0])})
            return [candidates[0]]
        self.times.append(time.process_time()-started)
        self.decisions.append({'tick':observation.tick,'fallback':None,'candidate_count':len(candidates),
                               'budget_truncated':len(candidates)>BUDGET,'evaluated':records,
                               'selected':asdict(candidates[selected])})
        return [candidates[selected]]


def metrics(result, case, incident, policy):
    samples = result['samples']
    complete = result['summary']['status'] == 'completed'
    flows = [s['completed_tick'] - s['release_tick'] for s in samples if s['completed_tick'] is not None]
    waits = [s['assigned_tick'] - s['release_tick'] for s in samples if s['assigned_tick'] is not None]
    backlog = [sum(s['status'] in ('waiting','assigned','in_transit') for s in state['samples'])
               for state in result['history']]
    waiting = [sum(s['status'] == 'waiting' for s in state['samples']) for state in result['history']]
    durations = sorted(policy.times)
    sessions = service_sessions(result,case,incident)
    return {'completed_count':result['summary']['completed_count'],'sample_count':result['summary']['sample_count'],
            'status':result['summary']['status'], 'mean_flow_complete_episode':statistics.mean(flows) if complete else None,
            'mean_assignment_wait_observed':statistics.mean(waits) if waits else None,
            'station_unavailable_wait_ticks':sum(s['unavailable_wait_ticks'] for s in sessions),
            'max_released_unfinished':max(backlog,default=0), 'max_unassigned_waiting':max(waiting,default=0),
            'pending_at_recovery':pending_at(result,case,incident['end']) if incident else None,
            'resource_contention_proxy':contention_proxy(result,case),
            'decision_count':len(durations), 'decision_cpu_mean_ms':1000*statistics.mean(durations) if durations else None,
            'decision_cpu_p95_ms':1000*quantile(durations,.95) if durations else None,
            'decision_cpu_samples_ms':[1000*t for t in policy.times],
            'fallbacks':policy.fallbacks, 'model_steps':policy.model_steps,
            'discrete_safety_violations':discrete_safety_violations(result),
            'sample_outcomes':[{k:s[k] for k in ('id','release_tick','assigned_tick','picked_tick','completed_tick')}
                               for s in samples], 'decisions':policy.decisions}


def summarize(cases):
    result = {}
    for regime in ('all',*REGIMES):
        selected = [c for c in cases if regime == 'all' or c['regime'] == regime]
        table = {}
        for method in METHODS:
            values = [c['methods'][method] for c in selected]
            paired = [(c['layout'],c['seed'],c['methods'][method]['mean_flow_complete_episode'] -
                       c['methods']['rule_original']['mean_flow_complete_episode']) for c in selected
                      if c['methods'][method]['mean_flow_complete_episode'] is not None
                      and c['methods']['rule_original']['mean_flow_complete_episode'] is not None]
            flows = [v['mean_flow_complete_episode'] for v in values if v['mean_flow_complete_episode'] is not None]
            waits = [v['mean_assignment_wait_observed'] for v in values if v['mean_assignment_wait_observed'] is not None]
            means = [v['decision_cpu_mean_ms'] for v in values if v['decision_cpu_mean_ms'] is not None]
            p95 = [v['decision_cpu_p95_ms'] for v in values if v['decision_cpu_p95_ms'] is not None]
            pooled_decisions = sorted(t for v in values for t in v['decision_cpu_samples_ms'])
            table[method] = {'cases':len(values), 'completed':sum(v['status'] == 'completed' for v in values),
                'task_completion_rate':sum(v['completed_count'] for v in values)/sum(v['sample_count'] for v in values),
                'mean_flow_completed_only':statistics.mean(flows) if flows else None,
                'mean_assignment_wait':statistics.mean(waits) if waits else None,
                'total_station_unavailable_wait_ticks':sum(v['station_unavailable_wait_ticks'] for v in values),
                'mean_max_released_unfinished':statistics.mean(v['max_released_unfinished'] for v in values),
                'mean_competing_robot_ticks':statistics.mean(v['resource_contention_proxy']['occupied_target_robot_ticks'] for v in values),
                'mean_decision_cpu_ms':statistics.mean(means) if means else None,
                'mean_case_p95_decision_cpu_ms':statistics.mean(p95) if p95 else None,
                'pooled_decision_cpu_p95_ms':quantile(pooled_decisions,.95) if pooled_decisions else None,
                'fallbacks':sum(v['fallbacks'] for v in values),
                'safety_violations':sum(v['discrete_safety_violations'] for v in values),
                'paired_flow_minus_rule_original':bootstrap_paired(paired),
                'wins':sum(v < -1e-9 for _,_,v in paired), 'losses':sum(v > 1e-9 for _,_,v in paired),
                'ties':sum(abs(v) <= 1e-9 for _,_,v in paired)}
        result[regime] = table
    return result


def run(profile):
    torch.set_num_threads(1)
    began = time.process_time()
    bases, sources = source_bases()
    audit = audit_schedules(bases)
    if audit['cross_group_duplicates']:
        raise ValueError('task schedules overlap: ' + repr(audit))
    network, model_info = load_pr9_model()
    cases = []
    for layout, seeds in SEEDS[profile].items():
        for seed in seeds:
            scenarios, design = make_round2_cases(bases[layout],layout,seed)
            for regime in REGIMES:
                case, pauses, incident = scenarios[regime]
                geometry = Geometry(case.grid)
                methods = {}
                for name in METHODS:
                    observer = ObservableHistory()
                    model = (ProgressAwareRule(geometry) if name.startswith('rule_') else
                             RuleMotionNeuralService(geometry,network,torch.device('cpu'),model_info['threshold']))
                    policy = OnlinePolicy(model,geometry,observer,name.split('_',1)[1])
                    result = Simulator(case,'world',seed,motion_pauses=pauses,collect_metrics=True).run(
                        dispatch_policy=policy,on_observation=observer.observe)
                    methods[name] = metrics(result,case,incident,policy)
                cases.append({'layout':layout,'seed':seed,'regime':regime,'incident':incident,
                              'design':design,'config':case_config(case,pauses),'methods':methods})
                print(f'{profile} online {layout} seed={seed} {regime}',flush=True)
    return {'profile':profile, 'scope':'Synthetic online rule and frozen neural policies only; no oracle input.',
            'source_sha256':sources,'schedule_audit':audit,'checkpoint':model_info,
            'horizon':HORIZON,'budget':BUDGET,'cases':cases,'cpu_seconds':time.process_time()-began}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=tuple(SEEDS),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    try:
        if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
            raise FileExistsError(f'refusing to overwrite output: {args.output}')
        result = run(args.profile)
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/'raw.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        summary = summarize(result['cases'])
        (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(summary,ensure_ascii=False),flush=True)
        return 0
    except (OSError,ValueError,TypeError,KeyError,RuntimeError) as exc:
        print(f'online experiment failed: {exc}',file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
