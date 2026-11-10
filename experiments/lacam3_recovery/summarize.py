"""Produce a compact machine-readable summary without discarding failures."""
from collections import Counter
import json
from pathlib import Path
import statistics
import sys


def summarize(source, destination):
    data = json.loads(Path(source).read_text())
    cases = data['cases']
    groups = {}
    for label, selected in (
        ('hospital', [c for c in cases if c['regime'] == 'hospital_pilot']),
        ('grid_3_robots', [c for c in cases if c['regime'] != 'hospital_pilot' and c['robots'] == 3]),
        ('grid_5_robots', [c for c in cases if c['robots'] == 5]),
    ):
        methods = {}
        for name in ('legacy', 'pibt', 'hybrid'):
            rows = [c['methods'][name] for c in selected]
            methods[name] = {
                'jobs_completed': sum(r['jobs_completed'] for r in rows),
                'jobs': sum(r['jobs'] for r in rows),
                'stages_completed': sum(r['stages_completed'] for r in rows),
                'stages': sum(r['stages'] for r in rows),
                'mean_of_scene_restricted_mean_flow': statistics.mean(r['restricted_mean_flow'] for r in rows),
                'mean_of_scene_motion_wall_mean_ms': statistics.mean(r['motion_wall_mean_ms'] for r in rows),
                'mean_of_scene_motion_wall_p95_ms': statistics.mean(r['motion_wall_p95_ms'] for r in rows),
                'recovery_triggers': sum(r['recovery_triggers'] for r in rows),
                'official_solver_calls': sum(r['official_solver_calls'] for r in rows),
                'recovery_status_counts': dict(sum((Counter(r['recovery_status_counts']) for r in rows), Counter())),
            }
        groups[label] = {'scenes': len(selected), 'methods': methods}
    omitted = {'motion_wall_ms', 'recovery_log', 'motion_audit', 'history', 'events', 'job_outcomes'}
    scene_rows = []
    for case in cases:
        scene_rows.append({key: case[key] for key in ('layout', 'seed', 'regime', 'robots', 'lacam3_topology_support')}
                          | {'methods': {name: {k: v for k, v in method.items() if k not in omitted}
                                         for name, method in case['methods'].items()}})
    result = {'definition': 'scene is the paired unit; means average per-scene means without excluding unfinished tasks',
              'groups': groups, 'scenes': scene_rows}
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('refusing to overwrite summary')
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    summarize(sys.argv[1], sys.argv[2])
