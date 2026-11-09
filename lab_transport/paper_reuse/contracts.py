"""Validate predicted entity identities, task transitions and legal motion."""
from lab_transport.dynamics_experiment import check_predicted_step


def check_transition(current, predicted, geometry):
    check_predicted_step(current,predicted,geometry)
    old = {s.id:s for s in current.samples}
    new = {s.id:s for s in predicted.samples}
    if set(old) != set(new):
        raise ValueError('prediction changed known sample IDs')
    allowed = {'waiting':{'waiting','assigned','in_transit'},
               'assigned':{'assigned','in_transit'},
               'in_transit':{'in_transit','completed'},'completed':{'completed'}}
    for sid,sample in old.items():
        if sample.status not in allowed or new[sid].status not in allowed[sample.status]:
            raise ValueError('invalid sample lifecycle transition')
        if (sample.pickup,sample.dropoff,sample.release_tick)!=(new[sid].pickup,new[sid].dropoff,new[sid].release_tick):
            raise ValueError('prediction changed immutable sample fields')
    robots = {r.id:r for r in predicted.robots}
    assigned = [r.task_id for r in predicted.robots if r.task_id is not None]
    if len(assigned) != len(set(assigned)):
        raise ValueError('multiple robots have the same sample')
    for robot in predicted.robots:
        if robot.task_id is None:
            if robot.phase != 'idle':
                raise ValueError('unassigned robot has a non-idle phase')
            continue
        sample = new.get(robot.task_id)
        expected = {'to_pickup':'assigned','to_dropoff':'in_transit'}.get(robot.phase)
        if sample is None or expected is None or sample.status != expected or sample.robot_id != robot.id:
            raise ValueError('predicted robot/sample assignment is inconsistent')
    for sample in predicted.samples:
        if sample.status in ('assigned','in_transit') and (
                sample.robot_id not in robots or robots[sample.robot_id].task_id != sample.id):
            raise ValueError('active sample lacks its assigned robot')
