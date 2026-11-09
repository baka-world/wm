"""Export predeclared factual service-interruption data; never model inputs from metadata."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

from .clinic_map import make_clinic_scenario
from .decision_value import case_config
from .decision_value_round2 import burst_case, contention_proxy, make_round2_cases, pending_at, service_sessions
from .dynamics_revision import SPLITS_V2
from .hospital_case import make_scenario
from .multisite_experiment import make_case, prepare_tartu
from .simulation import Simulator
from .training_data import (_bytes, _digest, _task_hash, _topology, download_sources, validate_dataset)
from .transition_schema import SCHEMA_VERSION

POOLS = {'train': ('tartu', 21000, 4), 'validation': ('tartu', 23000, 2),
         'test_tartu': ('tartu', 25000, 2), 'test_clinic': ('clinic', 26000, 2)}
REGIMES = ('normal', 'burst', 'outage_processing', 'outage_arrival')
POLICIES = ('nearest', 'world')


def source_bases():
    src = download_sources()
    return ({'tartu': prepare_tartu(make_scenario(src['hospital_map'], seconds_per_tick=10,
                                                   tiago_config=src['tiago_config'])),
             'clinic': make_clinic_scenario(src['clinic_map'], src['clinic_config'])},
            {k: _digest(v) for k, v in src.items()})


def select_seeds(bases):
    """Select on task signatures alone, including all PR #8 dataset splits."""
    used = set()
    for _, (layout, seeds) in SPLITS_V2.items():
        for seed in seeds:
            used.add(_task_hash(make_case(bases[layout], layout, seed, 'normal'), layout))
    selected, rejected = {}, {}
    for split, (layout, start, count) in POOLS.items():
        chosen, skipped = [], []
        for seed in range(start, start + 200):
            a = _task_hash(make_case(bases[layout], layout, seed, 'normal'), layout)
            b = _task_hash(burst_case(bases[layout], layout, seed), layout)
            if a in used or b in used or a == b:
                skipped.append(seed)
                continue
            used.update((a, b))
            chosen.append(seed)
            if len(chosen) == count:
                break
        if len(chosen) != count:
            raise ValueError(f'not enough unique task patterns for {split}')
        selected[split], rejected[split] = (layout, tuple(chosen)), skipped
    return selected, rejected


def export(output: Path, bases, hashes):
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'refusing to overwrite dataset: {output}')
    splits, skipped = select_seeds(bases)
    output.mkdir(parents=True, exist_ok=True)
    maps = {}
    for layout in sorted({x[0] for x in splits.values()}):
        relative = f'maps/{layout}.json'
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = _bytes(_topology(bases[layout]))
        path.write_bytes(raw)
        maps[layout] = {'file': relative, 'sha256': _digest(raw)}
    entries, coverage = [], []
    for split, (layout, seeds) in splits.items():
        for seed in seeds:
            cases, design = make_round2_cases(bases[layout], layout, seed)
            for regime in REGIMES:
                case, pauses, incident = cases[regime]
                for policy in POLICIES:
                    rows = []
                    result = Simulator(case, policy, seed, motion_pauses=pauses, collect_metrics=True).run(on_transition=rows.append)
                    rows = [row for row in rows if row.next_boundary == 'before_dispatch']
                    relative = f'episodes/{split}/{layout}-{seed}-{regime}-{policy}.jsonl'
                    path = output / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    digest = hashlib.sha256()
                    with path.open('xb') as handle:
                        for row in rows:
                            line = _bytes(asdict(row))
                            handle.write(line)
                            digest.update(line)
                    entry = {'split': split, 'layout': layout, 'seed': seed, 'regime': regime,
                             'behavior_policy': policy, 'file': relative, 'sha256': digest.hexdigest(),
                             'task_schedule_sha256': _task_hash(case, layout), 'transitions': len(rows),
                             'episode_status': result['summary']['status'],
                             'completed_tasks': result['summary']['completed_count'],
                             'scenario_sample_count': len(case.samples)}
                    entries.append(entry)
                    sessions = service_sessions(result, case, incident)
                    coverage.append({'split': split, 'layout': layout, 'seed': seed, 'regime': regime,
                                     'behavior_policy': policy, 'incident': incident, 'design': design,
                                     'config': case_config(case, pauses),
                                     'service_wait_ticks': sum(s['unavailable_wait_ticks'] for s in sessions),
                                     'affected_tasks': len({s['sample_id'] for s in sessions if s['unavailable_wait_ticks']}),
                                     'interrupted_sessions': sum(s['interrupted_after_service_start'] for s in sessions),
                                     'arrived_during_outage': sum(s['dropoff_arrived_during_outage'] for s in sessions),
                                     'pending_at_recovery': pending_at(result, case, incident['end']) if incident else None,
                                     'competition': contention_proxy(result, case),
                                     'episode_status': result['summary']['status']})
                print(f'{split} {layout} seed={seed} {regime} collected', flush=True)
    manifest = {'dataset_version': 1, 'observation_schema_version': SCHEMA_VERSION,
                'scope': 'synthetic factual simulation; never clinical records', 'maps': maps,
                'source_sha256': hashes, 'schedule_audit': {'cross_split_duplicate_groups': 0,
                    'cross_split_task_duplicate_groups': 0, 'seed_selection': splits, 'skipped_signatures': skipped},
                'input_boundary': 'Only current observation, current actions and static topology; no future, seeds or incident metadata.',
                'episodes': entries}
    (output / 'manifest.json').write_bytes(_bytes(manifest))
    (output / 'coverage.json').write_text(json.dumps(coverage, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    check = validate_dataset(output)
    return {'check': check, 'splits': splits, 'skipped_signatures': skipped,
            'source_sha256': hashes, 'episodes': len(entries), 'coverage': coverage}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        bases, hashes = source_bases()
        result = export(args.output, bases, hashes)
        print(json.dumps({k:v for k,v in result.items() if k != 'coverage'}, ensure_ascii=False), flush=True)
        return 0
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        print(f'fault-data export failed: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
