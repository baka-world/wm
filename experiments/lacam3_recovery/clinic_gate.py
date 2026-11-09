"""Run official full LaCAM3 solve on frozen PR #13 clinic states."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from business_transport.lacam3_motion import LaCAM3Planner
from business_transport.lacam3_coordinator import priority_for
from experiments.lacam3_recovery.clinic_states import frozen_states
from lab_transport.domain import DirectedGraph
from lab_transport.dynamics_model import Geometry


def run(worker,output):
    if output.exists():raise FileExistsError('refusing to overwrite clinic gate evidence')
    data=frozen_states();topology=data['topology']
    graph=DirectedGraph({tuple(x['position']):tuple(x['source_coordinates']) for x in topology['coordinates']},
                        tuple((tuple(a),tuple(b)) for a,b in topology['edges']))
    planner=LaCAM3Planner(Geometry(graph),worker)
    records=[]
    for state in data['states']:
        robots=state['robots'];positions={r['id']:tuple(r['position']) for r in robots}
        goals={r['id']:tuple(r['target']) for r in robots};locked={r['id'] for r in robots if r['locked']}
        result=planner.plan(positions,goals,locked,priority_for(robots))
        records.append({'state':state,'result':asdict(result)})
        print(state['label'],state['tick'],result.status,'steps',max(0,len(result.path)-1),
              'goal_conflict',result.duplicate_business_goals,flush=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({'source':'frozen PR #13 archived trajectories',
        'last_stage_completed_tick':data['last_stage_completed_tick'],'first_repeat_ticks':data['first_repeat_ticks'],
        'terminal_state_counts_last_100':data['terminal_state_counts_last_100'],
        'solver':'official LaCAM3 solve(), first complete solution, single thread',
        'budget_ms':planner.budget_ms,'planner_seed':planner.seed,'states':records},ensure_ascii=False,indent=2)+'\n')
    if any(r['result']['status']!='solved' for r in records):
        raise SystemExit('clinic gate did not solve every required frozen state; stop further integration')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.worker,a.output)


if __name__=='__main__':main()
