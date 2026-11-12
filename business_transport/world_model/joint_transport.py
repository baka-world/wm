"""Optional V4 service + published pgmpy AGV-risk joint world-model forecast.

Method sources:
- Tang & Wang (2025), Bayesian AGV transport uncertainty.
- pgmpy 1.0.0 BayesianEstimator / VariableElimination (MIT).
- Chua et al. (2018), PETS trajectory-sampling concept.
- Existing RHWM-v1 frozen V4 + lawful WorkflowDynamics.

Two explicit modes:
  shadow: attach learned transport risk for candidate actions/forecasted legs,
          leaving all physical/business future states identical to V4.
  sampled_hold: experimental event-time adapter that draws ONE delay event
          at each newly observed leg start and holds motion for rounded
          training positive-delay mean. Motion safety and workflow legality
          are checked, and rejected holds remain visible. This is an
          engineering adapter, NOT a reproduction of any paper's motion
          dynamics or a claimed calibrated per-tick delay process.

No simulator oracle, private countdown, future policy actions or online labels.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import random

from business_transport.observation import capacity_view
from business_transport.workflow_transition import check_business_transition
from lab_transport.dynamics_model import _intent
from lab_transport.state_v2 import rebuild
from .contracts import ForecastBatch, ForecastParticle, ForecastStep
from .hybrid import HybridSystemWorldModel, _distribution, _seed


RELATION_FIELDS=(
    "occupant_robot_id","service_task_id",
    "service_session_active","service_in_progress",
    "waiting_pickups","approaching_pickups","approaching_dropoffs",
)


@dataclass(frozen=True)
class TransportRisk:
    tick: int
    robot_id: str
    sample_id: str
    phase: str
    p_delay: float
    sampled_extra_ticks: int
    mode: str


@dataclass(frozen=True)
class JointForecast:
    forecast: ForecastBatch
    transport_risks: tuple[TransportRisk, ...]
    applied_motion_holds: int
    rejected_unsafe_holds: int
    mode: str

    @property
    def distributions(self):
        return self.forecast.distributions

    @property
    def particles(self):
        return self.forecast.particles


def _public_leg_query(state, robot, sample, phase, geometry):
    stations={s.id:s for s in state.stations}
    target=sample.pickup if phase=="to_pickup" else sample.dropoff
    station=stations[target]
    distance=geometry.distance(robot.position,station.position)
    if distance>=geometry.unreachable:
        raise ValueError("unreachable public transport target")
    return {
        "phase":phase,"distance":distance,
        "other_near_target":sum(geometry.distance(other.position,station.position)<=2
                               for other in state.robots if other.id!=robot.id),
        "other_active":sum(other.task_id is not None
                           for other in state.robots if other.id!=robot.id),
        "target_capacity":station.resource_capacity,
        "target_occupied":station.resource_occupied,
        "path_only_baseline_ticks":distance+int(phase=="to_dropoff" and distance>0),
    }


def _with_stalled_robot_positions(before, proposed, robot_ids, *, geometry, rules):
    """Postpone a legal move without breaking resource occupancy or task state.

    If any other predicted robot occupies the old cell, that particular stall
    is unsafe and rejected, preserving the original collision-free plan.
    """
    if not robot_ids:return proposed,0,0
    previous={r.id:r for r in before.robots}
    next_robots={r.id:r for r in proposed.robots}
    applied=rejected=0
    for rid in sorted(robot_ids):
        old=previous[rid]
        new=next_robots[rid]
        if new.position==old.position:
            continue
        if any(other.id!=rid and other.position==old.position
               for other in next_robots.values()):
            rejected+=1
            continue
        next_robots[rid]=replace(new,position=old.position)
        applied+=1
    if applied==0:return proposed,0,rejected
    robots=tuple(next_robots[r.id] for r in proposed.robots)
    # Recompute public entity relationships, keeping the existing business
    # station metadata / service histories intact.
    rebuilt=rebuild(replace(proposed,robots=robots),robots)
    station_by_id={s.id:s for s in rebuilt.stations}
    stations=tuple(replace(st,**{k:getattr(station_by_id[st.id],k)
                                  for k in RELATION_FIELDS})
                   for st in proposed.stations)
    changed=replace(proposed,robots=robots,stations=stations)
    devices={s.id:s.device_available for s in before.stations}
    changed,_=capacity_view(changed,(),rules,device_status=devices)
    check_business_transition(before,changed,geometry,rules)
    return changed,applied,rejected


class JointServiceTransportWorldModel:
    """Couple official pgmpy event risk with frozen V4 legal state trajectories."""

    def __init__(self,geometry,rules,*,v4_checkpoint,transport_model):
        if transport_model is None or not callable(getattr(
                transport_model,"predict_delay_probability",None)):
            raise ValueError("must supply fitted official pgmpy transport reference")
        self.geometry,self.rules=geometry,rules
        self.transport=transport_model
        self.service=HybridSystemWorldModel(
            geometry,rules,kind="v4",checkpoint=v4_checkpoint,experimental=True)

    def rollout(self,public_state,first_action,*,continuation_policy,
                horizon_ticks=5,particles=3,sampling_seed=1810,
                continuation_policy_name="nearest_public",mode="shadow"):
        if mode not in ("shadow","sampled_hold"):
            raise ValueError("only explicit shadow/sampled_hold modes are supported")
        if not 1<=horizon_ticks<=self.service.MAX_HORIZON:
            raise ValueError("RHWM forecast horizon unsupported")
        if not 1<=particles<=self.service.MAX_PARTICLES:
            raise ValueError("RHWM particle count unsupported")
        first_action=tuple(first_action)
        output=[]
        risks=[]
        applied=rejected=0
        for p in range(particles):
            state=public_state
            # Initial in-transit legs began before this forecast anchor; their
            # delay event is already partially realized and must NOT be drawn
            # as a fresh start. Newly assigned / progressed legs are sampled.
            seen={(r.id,r.task_id,r.phase) for r in state.robots if r.task_id}
            pending={}
            trace=[]
            rng=random.Random(_seed(sampling_seed,p,state.tick))
            for k in range(horizon_ticks):
                actions=first_action if k==0 else tuple(continuation_policy(state))
                intents=_intent(state,actions)
                robots={r.id:r for r in state.robots}
                samples={s.id:s for s in state.samples}
                for rid,(phase,sid) in sorted(intents.items()):
                    if sid is None:continue
                    key=(rid,sid,phase)
                    if key in seen:continue
                    seen.add(key)
                    row=_public_leg_query(state,robots[rid],samples[sid],
                                          phase,self.geometry)
                    if row["distance"]==0:continue
                    p_delay=self.transport.predict_delay_probability(row)
                    extra=(max(1,round(self.transport.positive_delay_ticks))
                           if rng.random()<p_delay else 0)
                    pending[key]=extra
                    risks.append(TransportRisk(
                        state.tick,rid,sid,phase,p_delay,extra,mode))
                step=self.service.predict_next(
                    state,actions,sampling_seed=_seed(sampling_seed,p,state.tick))
                new_state=step.next_state
                if mode=="sampled_hold":
                    stall_candidates=[]
                    for rid,(phase,sid) in intents.items():
                        key=(rid,sid,phase)
                        if sid is not None and pending.get(key,0)>0:
                            stall_candidates.append(rid)
                    new_state,done,unsafe=_with_stalled_robot_positions(
                        state,new_state,stall_candidates,
                        geometry=self.geometry,rules=self.rules)
                    applied+=done
                    rejected+=unsafe
                    for rid in stall_candidates:
                        old=robots[rid]
                        candidate=next(r for r in new_state.robots if r.id==rid)
                        prior=next(r for r in step.next_state.robots if r.id==rid)
                        if candidate.position==old.position and prior.position!=old.position:
                            key=(rid,intents[rid][1],intents[rid][0])
                            pending[key]-=1
                check_business_transition(state,new_state,self.geometry,self.rules,actions)
                trace.append(replace(step,next_state=new_state,model_version=
                                     "rhwm-joint-"+mode))
                state=new_state
            output.append(ForecastParticle(p,tuple(trace)))
        frames=tuple(output)
        forecast=ForecastBatch(
            "rhwm-joint-"+mode,public_state,first_action,
            continuation_policy_name,frames,
            tuple(_distribution(frames,j) for j in range(horizon_ticks)))
        return JointForecast(forecast,tuple(risks),applied,rejected,mode)
