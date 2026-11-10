"""Record deterministic three-stage rollout and capacity-relation regression."""
from dataclasses import asdict
import json
from pathlib import Path
import sys

from business_transport.dynamics import CapacityDynamics,WorkflowDynamics
from business_transport.workflow_transition import check_business_transition
from experiments.workflow_prediction.tests.test_workflow import case,factual
from lab_transport.decision_value import candidates_for,nearest_batch
from lab_transport.torch_structured_adapter import ProgressAwareRule


def run(destination):
    business=case(3);_,rows,geometry=factual(business)
    model=WorkflowDynamics(ProgressAwareRule(geometry),business.rules);state=rows[0].observation;trajectory=[]
    for tick,row in enumerate(rows):
        actions=nearest_batch(state,candidates_for(state),geometry)
        predicted=model.predict_next(state,actions)
        check_business_transition(state,predicted,geometry,business.rules,actions)
        predicted_samples=[(s.id,s.status,s.release_tick) for s in predicted.samples]
        factual_samples=[(s.id,s.status,s.release_tick) for s in row.next_observation.samples]
        predicted_robots=[(r.id,r.position,r.task_id,r.phase) for r in predicted.robots]
        factual_robots=[(r.id,r.position,r.task_id,r.phase) for r in row.next_observation.robots]
        if actions!=row.assignments or predicted_samples!=factual_samples or predicted_robots!=factual_robots:
            raise ValueError('deterministic factual trajectory diverged')
        trajectory.append({'tick':tick,'actions':[asdict(a) for a in actions],
                           'predicted_samples':predicted_samples,'factual_samples':factual_samples,
                           'predicted_robots':predicted_robots,'factual_robots':factual_robots})
        state=predicted
    if not any(s.id=='J1::stage3' and s.status=='completed' for s in state.samples):
        raise ValueError('three-stage controlled rollout did not finish')
    business=case(2,1,2);_,rows,geometry=factual(business)
    old=CapacityDynamics(ProgressAwareRule(geometry),business.rules)
    new=WorkflowDynamics(ProgressAwareRule(geometry),business.rules);corrections=[]
    for row in rows:
        a=old.predict_next(row.observation,row.assignments)
        b=new.predict_next(row.observation,row.assignments)
        check_business_transition(row.observation,b,geometry,business.rules,row.assignments)
        for previous,current in zip(a.stations,b.stations):
            if previous.service_session_active and not current.service_session_active:
                robot=next(r for r in b.robots if r.id==current.occupant_robot_id)
                corrections.append({'tick':row.observation.tick,'station_id':current.id,
                    'robot_id':robot.id,'robot_service_elapsed':robot.service_elapsed,
                    'old_service_task_id':previous.service_task_id,'old_service_session_active':previous.service_session_active,
                    'new_service_task_id':current.service_task_id,'new_service_session_active':current.service_session_active})
    if not corrections:raise ValueError('capacity-relation regression did not exercise its target')
    output=Path(destination)
    if output.exists():raise FileExistsError('refusing to overwrite edge diagnostics')
    output.write_text(json.dumps({'scope':'synthetic deterministic engineering control; not learned-model accuracy',
        'three_stage_steps':len(trajectory),'three_stage_trajectory':trajectory,'capacity_corrections':corrections},
        ensure_ascii=False,indent=2)+'\n')
    print('deterministic three-stage rollout:',len(trajectory),'steps; capacity relationship corrections:',len(corrections))


if __name__=='__main__':run(sys.argv[1])
