"""Optional shared PIBT movement for execution and observable prediction."""
from __future__ import annotations

from dataclasses import dataclass,replace
import time

from business_transport.observation import BusinessObserver
from business_transport.pibt_motion import PIBTMotion
from business_transport.simulation import BusinessSimulator
from lab_transport.dynamics_model import Geometry,_intent,_target
from lab_transport.state_v2 import rebuild
from lab_transport.transition_schema import ObservedRobot


@dataclass(frozen=True)
class MotionObservedRobot(ObservedRobot):
    operational_available: bool = True


class MotionObserver(BusinessObserver):
    """Expose only whether a robot is paused at the current observation tick."""
    def observe(self,observation):
        rich=super().observe(observation)
        raw={r.id:r for r in observation.robots}
        return replace(rich,robots=tuple(replace(r,operational_available=
            getattr(raw[r.id],'operational_available',True)) for r in rich.robots))


class CoordinatedBusinessSimulator(BusinessSimulator):
    def __init__(self,business,*args,motion_backend='legacy',pibt_library=None,
                 planner_seed=20261109,**kwargs):
        if motion_backend not in ('legacy','pibt'):raise ValueError('unknown motion backend')
        if motion_backend=='pibt' and not pibt_library:raise ValueError('PIBT library is required')
        self.motion_backend=motion_backend;self.motion_wall_seconds=[]
        super().__init__(business,*args,**kwargs)
        self.pibt=(PIBTMotion(Geometry(self.scenario.grid),pibt_library,planner_seed=planner_seed)
                   if motion_backend=='pibt' else None)

    def _observable_state(self,tick):
        observation=super()._observable_state(tick)
        robots=[]
        for robot in observation.robots:
            paused=any(a<=tick<b for a,b in self.motion_pauses.get(robot.id,()))
            robots.append(MotionObservedRobot(**{k:getattr(robot,k) for k in ObservedRobot.__dataclass_fields__},
                                              operational_available=not paused))
        return replace(observation,robots=tuple(robots))

    def _move(self,tick,serviced):
        start=time.perf_counter()
        if self.motion_backend=='legacy':
            super()._move(tick,serviced)
        else:
            paused={rid for rid,intervals in self.motion_pauses.items()
                    if any(a<=tick<b for a,b in intervals)}
            if self.collect_metrics:self.metrics['motion_pause_ticks']+=len(paused-serviced)
            positions={rid:r.position for rid,r in self.robots.items()}
            goals={rid:self._target(r) for rid,r in self.robots.items()}
            updated=self.pibt.step(positions,goals,serviced|paused,tick)
            for rid,pos in updated.items():self.robots[rid].position=pos
        self.motion_wall_seconds.append(time.perf_counter()-start)


class PIBTMovementModel:
    """Reuse existing service/lifecycle prediction with the same PIBT step.

    Current service or a currently observed pause locks the robot. Unknown
    future pause changes are not forecast and cannot be treated as facts.
    """
    def __init__(self,model,geometry,pibt):
        self.model=model;self.geometry=geometry;self.pibt=pibt

    @property
    def distribution(self):return getattr(self.model,'distribution',None)

    @property
    def service_calls(self):return getattr(self.model,'service_calls',0)

    def reset_seed(self,seed):
        if hasattr(self.model,'reset_seed'):self.model.reset_seed(seed)

    def predict_next(self,observation,assignments):
        assignments=tuple(assignments);intents=_intent(observation,assignments)
        samples={s.id:s for s in observation.samples}
        stations={s.id:s.position for s in observation.stations}
        positions={r.id:r.position for r in observation.robots}
        goals={r.id:_target(r,intents[r.id],samples,stations) for r in observation.robots}
        locked={r.id for r in observation.robots if
                (intents[r.id][1] is not None and r.position==goals[r.id]) or
                getattr(r,'operational_available',None) is False}
        predicted=self.model.predict_next(observation,assignments)
        moved=self.pibt.step(positions,goals,locked,observation.tick)
        before={r.id:r for r in observation.robots}
        robots=tuple(replace(r,position=moved[r.id],operational_available=
                    getattr(before[r.id],'operational_available',None)) for r in predicted.robots)
        return rebuild(predicted,robots)
