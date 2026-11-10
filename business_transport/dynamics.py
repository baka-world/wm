"""Use existing world models with observable capacity admission."""
from __future__ import annotations

from dataclasses import replace
from business_transport.observation import BusinessStationV2,attach_prediction,capacity_view
from lab_transport.dynamics_model import _intent
from lab_transport.state_v2 import RobotV2,StationV2


class CapacityDynamics:
    def __init__(self,model,rules):
        self.model,self.rules=model,rules

    @property
    def distribution(self):
        return getattr(self.model,'distribution',None)

    def reset_seed(self,seed):
        if hasattr(self.model,'reset_seed'):self.model.reset_seed(seed)

    def predict_next(self,observation,assignments):
        prepared,allowed=capacity_view(observation,assignments,self.rules)
        states={r.id:r for r in observation.robots}
        intents=_intent(observation,tuple(assignments))
        samples={s.id:s for s in observation.samples}
        stations={s.id:s for s in observation.stations}
        blocked=set()
        for rid,(phase,sid) in intents.items():
            if sid is None or rid in allowed:continue
            sample=samples[sid]
            station=stations[sample.pickup if phase=='to_pickup' else sample.dropoff]
            if states[rid].position==station.position and states[rid].service_elapsed is None:
                blocked.add(rid)
        nxt=self.model.predict_next(prepared,tuple(assignments))
        # A robot denied a resource has not started service. Legacy models
        # otherwise represent any unavailable station as a paused session.
        robots=[]
        for robot in nxt.robots:
            if robot.id in blocked:
                robot=replace(robot,service_elapsed=None,service_active_elapsed=None,service_paused_elapsed=None)
            robots.append(robot)
        nxt=attach_prediction(observation,replace(nxt,robots=tuple(robots)))
        device={s.id:getattr(s,'device_available',s.available) for s in observation.stations}
        return capacity_view(nxt,(),self.rules,device)[0]


class WorkflowDynamics(CapacityDynamics):
    """Add only deterministic public successor release to existing models."""
    def predict_next(self,observation,assignments):
        from business_transport.workflow_transition import release_successors
        predicted=super().predict_next(observation,assignments)
        return release_successors(observation,predicted,self.rules)
