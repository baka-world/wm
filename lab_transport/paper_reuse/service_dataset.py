"""Create service-duration labels exclusively from completed factual sessions.

Only data present at service start is encoded. Incomplete or unobserved starts
are censored and never treated as completed durations. No simulator internals,
future schedules or hypothetical action branches are read.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

from lab_transport.dynamics_model import _intent
from lab_transport.service_learning import read_episodes

FEATURES = ('prior_ticks_scaled', 'observed_mean_scaled', 'history_count_scaled',
            'pickup_phase', 'station_available', 'waiting_pickups_scaled',
            'approaching_scaled')


def service_features(observation, robot, sample, phase, station):
    """Current observables only; service progress is conditioned on separately."""
    del observation, robot, sample
    mean = getattr(station, 'observed_effective_service_mean', None)
    count = getattr(station, 'observed_effective_service_count', 0)
    prior = station.expected_service_ticks
    return [float(prior) / 10., float(prior if mean is None else mean) / 10.,
            min(count, 20) / 20., float(phase == 'to_pickup'), float(station.available),
            min(getattr(station, 'waiting_pickups', 0), 8) / 8.,
            min(getattr(station, 'approaching_pickups', 0) +
                getattr(station, 'approaching_dropoffs', 0), 8) / 8.]


def extract_episode(episode):
    """Return completed service-start examples and censoring counts for one episode."""
    sessions, examples, counts = {}, [], Counter()
    for row in episode.rows:
        obs, nxt = row.observation, row.next_observation
        samples = {s.id:s for s in obs.samples}
        future = {s.id:s for s in nxt.samples}
        stations = {s.id:s for s in obs.stations}
        intents = _intent(obs, row.assignments)
        present = set()
        for robot in obs.robots:
            phase, sid = intents[robot.id]
            if sid is None or sid not in samples:
                continue
            sample = samples[sid]
            station_id = sample.pickup if phase == 'to_pickup' else sample.dropoff
            station = stations[station_id]
            if robot.position != station.position:
                continue
            key = (robot.id, sid, phase, station_id)
            present.add(key)
            if key not in sessions:
                sessions[key] = {'features':service_features(obs, robot, sample, phase, station),
                    'start_tick':obs.tick, 'active_ticks':0, 'paused_ticks':0,
                    'start_known':robot.service_elapsed is None}
            session = sessions[key]
            session['active_ticks'] += int(station.available)
            session['paused_ticks'] += int(not station.available)
            expected = 'in_transit' if phase == 'to_pickup' else 'completed'
            if sid in future and future[sid].status == expected:
                session = sessions.pop(key)
                if session['start_known'] and session['active_ticks'] > 0:
                    examples.append({'features':session['features'],
                        'duration_active_ticks':session['active_ticks'],
                        'paused_ticks_for_audit_only':session['paused_ticks'],
                        'sample_id':sid, 'phase':phase, 'station_id':station_id,
                        'start_tick':session['start_tick'], 'end_tick':nxt.tick})
                    counts['completed'] += 1
                else:
                    counts['excluded_unknown_start'] += 1
        # A discontinuous trajectory or an abandoned session is censored.
        for key in list(sessions):
            if key not in present:
                del sessions[key]
                counts['excluded_incomplete'] += 1
    counts['excluded_incomplete'] += len(sessions)
    return examples, dict(counts)


def export(dataset: Path, output: Path):
    if output.exists():
        raise FileExistsError(f'refusing to overwrite {output}')
    manifest, _, episodes, _ = read_episodes(dataset)
    examples, counts = [], Counter()
    for ep in episodes:
        if ep.split not in ('train', 'validation'):
            continue
        rows, excluded = extract_episode(ep)
        counts.update(excluded)
        for row in rows:
            examples.append({'split':ep.split,'layout':ep.layout,'seed':ep.seed,
                'regime':ep.regime,'policy':ep.policy, **row})
    if not examples or not any(row['split'] == 'validation' for row in examples):
        raise ValueError('requires nonempty existing train and validation sessions')
    result = {'schema':1, 'feature_names':FEATURES,
        'scope':'completed factual services, synthetic PR #9 trajectories; no test trajectories',
        'source_manifest_sha256':hashlib.sha256((dataset/'manifest.json').read_bytes()).hexdigest(),
        'source_episode_count':sum(e['split'] in ('train','validation') for e in manifest['episodes']),
        'counts':dict(counts), 'examples':examples}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as handle:
        with gzip.GzipFile(filename='',mode='wb',fileobj=handle,mtime=0,compresslevel=9) as compressed:
            compressed.write((json.dumps(result,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode())
    return {'counts':dict(counts),'examples':len(examples),
            'train':sum(x['split']=='train' for x in examples),
            'validation':sum(x['split']=='validation' for x in examples),
            'source_manifest_sha256':result['source_manifest_sha256']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.dataset,args.output),ensure_ascii=False))


if __name__ == '__main__':
    main()
