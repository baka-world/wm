"""Bounded offline witness search, never used by online motion control."""
from __future__ import annotations
from collections import defaultdict,deque
import argparse
import itertools
import json
from pathlib import Path


def inspect(case,max_states):
    adjacency=defaultdict(list)
    for left,right in case['topology']['edges']:adjacency[tuple(left)].append(tuple(right))
    agents=case['motion_trace'][-1]['robots']
    start=tuple(tuple(r['position']) for r in agents)
    goals=tuple(tuple(r['target']) for r in agents)
    target=next((i for i,r in enumerate(agents) if r['phase']=='to_dropoff'),0)
    queue=deque([start]);previous={start:None};found=None;limited=False
    while queue:
        state=queue.popleft()
        if state[target]==goals[target]:found=state;break
        # Conservatively freeze a task robot upon reaching its own target.
        # This can rule out paths requiring later service completion; failure is
        # not proof the real system has no feasible joint execution.
        choices=[([pos] if pos==goals[i] and agents[i]['task_id'] is not None
                  else [pos,*adjacency[pos]]) for i,pos in enumerate(state)]
        for nxt in itertools.product(*choices):
            if nxt==state or len(set(nxt))!=len(nxt):continue
            if any(state[i]!=state[j] and state[i]==nxt[j] and state[j]==nxt[i]
                   for i in range(len(state)) for j in range(i+1,len(state))):continue
            if nxt in previous:continue
            if len(previous)>=max_states:limited=True;break
            previous[nxt]=state;queue.append(nxt)
        if limited:break
    path=[]
    while found is not None:
        path.append(found);found=previous[found]
    path.reverse()
    return {'layout':case['layout'],'selected_robot':agents[target]['id'],
        'target':goals[target],'found_witness':bool(path),'steps':len(path)-1 if path else None,
        'searched_states':len(previous),'search_limited':limited,'queue_exhausted':not queue and not limited,
        'conditions':'static directed edges; no vertex conflict or edge swap; other robots freeze at their task goals',
        'path':path}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--max-states',type=int,default=450000)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('refusing to overwrite diagnostic')
    cases=json.loads(a.input.read_text())['cases']
    result=[inspect(case,a.max_states) for case in cases]
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in c.items() if k!='path'} for c in result],ensure_ascii=False))


if __name__=='__main__':main()
