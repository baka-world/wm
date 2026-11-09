"""Derived observable state, without changing the archived v1 wire format.

Progress is recoverable only when service start and station availability were
observed continuously.  No simulator private state or future event is read.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from .transition_schema import Observation, ObservedRobot, ObservedSample, ObservedStation


@dataclass(frozen=True)
class RobotV2(ObservedRobot):
    service_active_elapsed: int | None = None
    service_paused_elapsed: int | None = None
    operational_available: bool | None = None


@dataclass(frozen=True)
class SampleV2(ObservedSample):
    unassigned_wait_ticks: int | None = None
    priority: int | None = None


@dataclass(frozen=True)
class StationV2(ObservedStation):
    occupant_robot_id: str | None = None
    service_task_id: str | None = None
    service_session_active: bool = False
    service_in_progress: bool = False
    waiting_pickups: int = 0
    approaching_pickups: int = 0
    approaching_dropoffs: int = 0
    observed_effective_service_count: int = 0
    observed_effective_service_mean: float | None = None
    history_complete_since_start: bool = False


@dataclass(frozen=True)
class ObservationV2(Observation):
    pass


def rebuild(observation: Observation, robots: tuple[RobotV2, ...],
            history: dict[str, tuple[int, float]] | None = None,
            history_complete: bool | None = None) -> ObservationV2:
    """Recompute visible relations. Counts are demand proxies, not a FIFO queue."""
    samples = tuple(SampleV2(**{k: getattr(s, k) for k in ObservedSample.__dataclass_fields__},
                             unassigned_wait_ticks=max(0, observation.tick - s.release_tick)
                             if s.status == "waiting" else None,
                             priority=getattr(s, "priority", None)) for s in observation.samples)
    by_sample = {s.id: s for s in samples}
    stations = []
    for s in observation.stations:
        at_station = [r for r in robots if r.position == s.position]
        servicing = [r for r in at_station if r.task_id in by_sample and r.service_elapsed is not None
                     and ((r.phase == "to_pickup" and by_sample[r.task_id].pickup == s.id)
                          or (r.phase == "to_dropoff" and by_sample[r.task_id].dropoff == s.id))]
        if history is None:
            count = getattr(s, "observed_effective_service_count", 0)
            mean = getattr(s, "observed_effective_service_mean", None)
        else:
            count, total = history.get(s.id, (0, 0.0))
            mean = total / count if count else None
        stations.append(StationV2(**{k: getattr(s, k) for k in ObservedStation.__dataclass_fields__},
                                  occupant_robot_id=at_station[0].id if at_station else None,
                                  service_task_id=servicing[0].task_id if servicing else None,
                                  service_session_active=bool(servicing),
                                  service_in_progress=bool(servicing) and s.available,
                                  waiting_pickups=sum(x.pickup == s.id and x.status == "waiting" for x in samples),
                                  approaching_pickups=sum(x.pickup == s.id and x.status == "assigned" for x in samples),
                                  approaching_dropoffs=sum(x.dropoff == s.id and x.status == "in_transit" for x in samples),
                                  observed_effective_service_count=count, observed_effective_service_mean=mean,
                                  history_complete_since_start=(getattr(s, 'history_complete_since_start', False)
                                      if history_complete is None else history_complete)))
    return ObservationV2(2, observation.tick, robots, samples, tuple(stations))


class ObservableHistory:
    """Collect current observations in order; incomplete past progress stays unknown.

    The active-time inference assumes station availability implies service
    progresses in this simulator. Real deployments need equivalent telemetry.
    """
    def __init__(self):
        self.availability: dict[int, dict[str, bool]] = {}
        self.effective_history: dict[str, tuple[int, float]] = {}
        self.last_raw: Observation | None = None
        self.last_rich: ObservationV2 | None = None
        self.history_complete = True

    def observe(self, observation: Observation) -> ObservationV2:
        if self.last_raw is None and observation.tick != 0:
            self.history_complete = False
        if self.last_raw is not None:
            if observation.tick < self.last_raw.tick:
                raise ValueError("observations must be chronological")
            if observation.tick == self.last_raw.tick:
                if observation != self.last_raw:
                    raise ValueError("conflicting observations at the same tick")
                assert self.last_rich is not None
                return self.last_rich
            if observation.tick != self.last_raw.tick + 1:
                self.history_complete = False
            # A completed service is a factual transition, not a counterfactual.
            if observation.tick == self.last_raw.tick + 1:
                next_robots = {r.id: r for r in observation.robots}
                old_samples = {s.id: s for s in self.last_raw.samples}
                next_samples = {s.id: s for s in observation.samples}
                assert self.last_rich is not None
                for old in self.last_rich.robots:
                    if old.task_id not in old_samples or old.task_id not in next_samples:
                        continue
                    sample = old_samples[old.task_id]
                    station_id = sample.pickup if old.phase == "to_pickup" else sample.dropoff
                    station = next((s for s in self.last_raw.stations if s.id == station_id), None)
                    finished_status = "in_transit" if old.phase == "to_pickup" else "completed"
                    if (station is not None and old.position == station.position and station.available
                            and next_samples[old.task_id].status == finished_status
                            and old.id in next_robots):
                        # A first-tick completion has no previous service_elapsed.
                        active = 0 if old.service_elapsed is None else old.service_active_elapsed
                        if active is not None:
                            count, total = self.effective_history.get(station_id, (0, 0.0))
                            self.effective_history[station_id] = (count + 1, total + active + 1)
        self.availability[observation.tick] = {s.id: s.available for s in observation.stations}
        samples = {s.id: s for s in observation.samples}
        robots = []
        for robot in observation.robots:
            active = paused = None
            if robot.service_elapsed is not None and robot.task_id in samples:
                sample = samples[robot.task_id]
                station_id = sample.pickup if robot.phase == "to_pickup" else sample.dropoff
                ticks = range(observation.tick - robot.service_elapsed, observation.tick)
                known = all(t in self.availability and station_id in self.availability[t] for t in ticks)
                if known:
                    active = sum(self.availability[t][station_id] for t in ticks)
                    paused = robot.service_elapsed - active
            robots.append(RobotV2(**{k: getattr(robot, k) for k in ObservedRobot.__dataclass_fields__},
                                  service_active_elapsed=active, service_paused_elapsed=paused))
        rich = rebuild(observation, tuple(robots), self.effective_history, self.history_complete)
        self.last_raw, self.last_rich = observation, rich
        return rich


def predicted_progress(before: ObservationV2, predicted: Observation,
                       assignments) -> ObservationV2:
    """Update observable progress under a forecast, assuming no new outage."""
    from .dynamics_model import _intent
    intents = _intent(before, assignments)
    old = {r.id: r for r in before.robots}
    samples = {s.id: s for s in before.samples}
    stations = {s.id: s for s in before.stations}
    robots = []
    for r in predicted.robots:
        previous = old[r.id]
        phase, sid = intents[r.id]
        active = paused = wall = None
        if sid is not None:
            sample = samples[sid]
            station = stations[sample.pickup if phase == "to_pickup" else sample.dropoff]
            if previous.position == station.position and r.task_id == sid and r.phase == phase:
                wall = (previous.service_elapsed or 0) + 1
                old_active = 0 if previous.service_elapsed is None else previous.service_active_elapsed
                old_paused = 0 if previous.service_elapsed is None else previous.service_paused_elapsed
                active = None if old_active is None else old_active + int(station.available)
                paused = None if old_paused is None else old_paused + int(not station.available)
        robots.append(RobotV2(**{k: (wall if k == "service_elapsed" else getattr(r, k))
                                  for k in ObservedRobot.__dataclass_fields__},
                              service_active_elapsed=active, service_paused_elapsed=paused))
    return rebuild(predicted, tuple(robots))


def progress_adjusted(observation: ObservationV2) -> ObservationV2:
    """Feed measured effective progress to v1 rules; zero is conservative when unknown."""
    return replace(observation, robots=tuple(replace(r, service_elapsed=(
        r.service_active_elapsed if r.service_active_elapsed is not None else
        (0 if r.service_elapsed is not None else None))) for r in observation.robots))
