"""Extract prescribed clinic states directly from archived PR #13 trajectories."""
from __future__ import annotations
from collections import Counter
import gzip
import json
from pathlib import Path

ROOT=Path('examples/evidence/pibt-motion-20261109')


def archived(name):
    data=json.loads(gzip.decompress((ROOT/name).read_bytes()))
    return next(c for c in data['cases'] if c['layout']=='clinic')


def frozen_states():
    pibt=archived('hospital-pibt.json.gz');legacy=archived('hospital-legacy.json.gz')
    last=max(e['tick'] for e in pibt['events'] if e['type']=='delivered')
    history={h['tick']:h for h in pibt['history']};seen={};first=None
    def sig(t):
        return tuple((r['id'],tuple(r['position']),r['task_id'],r['phase'],tuple(r['target']),
                      r['serviced_this_tick'],r['motion_paused']) for r in t['robots'])
    for step in pibt['motion_trace']:
        if step['tick']<last:continue
        key=sig(step)
        if key in seen:first=(seen[key],step['tick']);break
        seen[key]=step['tick']
    if first is None:raise ValueError('no repeated state in frozen clinic trajectory')
    patterns=Counter(tuple(tuple(r['position']) for r in t['robots']) for t in pibt['motion_trace'][-100:])
    if len(patterns)!=3:raise ValueError('frozen terminal cycle differs from PR #13')
    selected=[('cycle_first',pibt['motion_trace'][first[0]],pibt)]
    for i,(pattern,count) in enumerate(patterns.most_common(),1):
        step=next(t for t in pibt['motion_trace'] if t['tick']>=last and
                  tuple(tuple(r['position']) for r in t['robots'])==pattern)
        selected.append((f'cycle_typical_{i}',step,pibt))
    selected.append(('legacy_final',legacy['motion_trace'][-1],legacy))
    results=[]
    for label,step,source in selected:
        snapshot=source['history'][step['tick']]
        samples={s['id']:s for s in snapshot['samples']};robots=[]
        for r in step['robots']:
            sample=samples.get(r['task_id'])
            robots.append({'id':r['id'],'position':r['position'],'target':r['target'],
                'task_id':r['task_id'],'phase':r['phase'],
                'release_tick':sample['release_tick'] if sample else None,
                'locked':bool(r['serviced_this_tick'] or r['motion_paused']),
                'serviced':r['serviced_this_tick'],'motion_paused':r['motion_paused'],
                'station_at_position':r['station_at_position'],'device_available':r['device_available'],
                'capacity_wait':r['capacity_wait'],'device_unavailable_wait':r['device_unavailable_wait']})
        results.append({'label':label,'tick':step['tick'],'robots':robots,
                        'resources':source['config']['resources']})
    return {'last_stage_completed_tick':last,'first_repeat_ticks':first,
            'terminal_state_counts_last_100':[{'positions':key,'count':value} for key,value in patterns.most_common()],
            'topology':pibt['topology'],'states':results}
