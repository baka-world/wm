"""Summarize paired counterfactuals without treating candidate branches as independent."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from lab_transport.decision_value_round2 import bootstrap_paired, quantile
from .counterfactual import HORIZONS

NAMES = ('rule','hybrid','service_oracle','actual_short_state')


def paired_difference(cases, horizon, left, right):
    values = []
    for case in cases:
        ds = [d for d in case['decisions'] if d['comparable']]
        if not ds:
            continue
        differences = []
        for d in ds:
            h = d['horizons'][str(horizon)]
            a = h['regret'][left] if left in NAMES else h['workload']['regret'][left.removeprefix('workload_')]
            b = h['regret'][right] if right in NAMES else h['workload']['regret'][right.removeprefix('workload_')]
            differences.append(a-b)
        values.append((case['layout'],case['seed'],statistics.mean(differences)))
    return bootstrap_paired(values)


def run(data):
    cases = data['cases']
    ds = [d for case in cases for d in case['decisions'] if d['comparable']]
    results = {}
    for horizon in HORIZONS:
        key = str(horizon)
        model_metrics = {}
        for name in ('rule','hybrid','service_oracle'):
            predictions = [c['predictions'][name]['horizons'][key] for d in ds for c in d['candidates']]
            actual = sum(p['service_events']['actual'] for p in predictions)
            predicted = sum(p['service_events']['predicted'] for p in predictions)
            correct = sum(p['service_events']['correct'] for p in predictions)
            joint_total = sum(p['known_entity_counts']['joint_total'] for p in predictions)
            joint_correct = sum(p['known_entity_counts']['joint_correct'] for p in predictions)
            elapsed = sorted(p['rollout_cpu_seconds']*1000 for p in predictions)
            original_score = [p['score_cpu_seconds']*1000 for p in predictions]
            workload_score = [p['workload_score_cpu_seconds']*1000 for p in predictions]
            model_metrics[name] = {'service_actual':actual,'service_predicted':predicted,'service_correct':correct,
                'service_recall':correct/actual if actual else None,
                'service_precision':correct/predicted if predicted else None,
                'joint_state_accuracy':joint_correct/joint_total if joint_total else None,
                'candidate_rollout_cpu_mean_ms':statistics.mean(elapsed) if elapsed else None,
                'candidate_rollout_cpu_p95_ms':quantile(elapsed,.95) if elapsed else None,
                'original_score_cpu_mean_ms':statistics.mean(original_score) if original_score else None,
                'workload_score_cpu_mean_ms':statistics.mean(workload_score) if workload_score else None}
        results[key] = {'model_metrics_descriptive':model_metrics,
            'oracle_minus_rule_original_regret':paired_difference(cases,horizon,'service_oracle','rule'),
            'oracle_minus_rule_workload_regret':paired_difference(cases,horizon,'workload_service_oracle','workload_rule'),
            'workload_minus_original_regret':{name:paired_difference(cases,horizon,'workload_'+name,name) for name in NAMES},
            'new_score_vs_old_points':{name:{
                'better':sum(d['horizons'][key]['workload']['regret'][name] < d['horizons'][key]['regret'][name]-1e-9 for d in ds),
                'equal':sum(abs(d['horizons'][key]['workload']['regret'][name]-d['horizons'][key]['regret'][name])<=1e-9 for d in ds),
                'worse':sum(d['horizons'][key]['workload']['regret'][name] > d['horizons'][key]['regret'][name]+1e-9 for d in ds)}
                for name in NAMES}}
    return {'scope':'Offline candidate/event/latency counts are descriptive; paired effects cluster by layout/seed.',
            'profile':data['profile'],'decisions':len(ds),'horizons':results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'refusing to overwrite: {args.output}')
    data = json.loads(args.input.read_text(encoding='utf-8'))
    result = run(data)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
