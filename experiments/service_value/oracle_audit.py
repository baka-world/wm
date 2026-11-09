"""Reconstruct and verify each candidate's offline-only observed service durations."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import sys

from lab_transport.decision_value import factual_run, replay_branch, truth_trace
from lab_transport.decision_value_round2 import make_round2_cases
from lab_transport.service_fault_data import source_bases
from lab_transport.transition_schema import Assignment

from .counterfactual import HORIZONS, actual_durations, event_set


def audit(data):
    bases, sources = source_bases()
    if sources != data['source_sha256']:
        raise ValueError('source hashes changed')
    scenarios, output = {}, []
    for case in data['cases']:
        layout,seed,regime = case['layout'],case['seed'],case['regime']
        group = (layout,seed)
        if group not in scenarios:
            scenarios[group] = make_round2_cases(bases[layout],layout,seed)[0]
        scenario,pauses,_ = scenarios[group][regime]
        factual, rows = factual_run(scenario,seed,pauses)
        by_tick = {row.observation.tick:row for row in rows}
        for decision in case['decisions']:
            initial = by_tick[decision['tick']].observation
            known = {s.id for s in initial.samples}
            for candidate in decision['candidates']:
                action = Assignment(**candidate['action'])
                outcome,branch = replay_branch(scenario,seed,pauses,factual,rows,initial.tick,(action,))
                if all(s['completed_tick'] is not None for s in outcome['samples']):
                    flow = statistics.mean(s['completed_tick']-s['release_tick'] for s in outcome['samples'])
                    if abs(flow-candidate['actual_mean_flow']) > 1e-9:
                        raise ValueError('candidate branch flow changed')
                durations,missing = actual_durations(outcome,branch,known)
                if len(durations) != candidate['offline_service_duration_count']:
                    raise ValueError('oracle duration count changed')
                truth,valid = truth_trace(initial,branch)
                if valid != candidate['actual_short_complete']:
                    raise ValueError('short reference censoring changed')
                if valid:
                    for h in HORIZONS:
                        actual = sorted([list(x) for x in event_set(initial,truth[:h])])
                        if actual != candidate['actual_events'][str(h)]:
                            raise ValueError('candidate short service events changed')
                relevant = []
                for event in outcome['events']:
                    if event['type'] in ('service_started','service_finished') and event.get('sample_id') in known:
                        relevant.append({k:event[k] for k in ('tick','type','sample_id','robot_id','station_id','phase','observed_ticks') if k in event})
                output.append({'layout':layout,'seed':seed,'regime':regime,'decision_tick':initial.tick,
                               'candidate':asdict(action), 'actual_mean_flow':candidate['actual_mean_flow'],
                               'observed_completed_service_events':relevant,
                               'privileged_active_service_durations':[{'sample_id':k[0],'phase':k[1],
                                   'station_id':k[2],'active_ticks':v} for k,v in sorted(durations.items())],
                               'missing':[list(x) for x in missing]})
        print(f"{data['profile']} oracle replay audit {layout} seed={seed} {regime}",flush=True)
    return {'profile':data['profile'],
            'scope':'Offline-only branch-derived full-episode service events and active durations; never online inputs.',
            'candidates':output}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise FileExistsError(f'refusing to overwrite: {args.output}')
        data = json.loads(args.input.read_text(encoding='utf-8'))
        result = audit(data)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(f"Verified {len(result['candidates'])} independent candidate service-duration records",flush=True)
        return 0
    except (OSError,ValueError,TypeError,KeyError,RuntimeError) as exc:
        print(f'oracle audit failed: {exc}',file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
