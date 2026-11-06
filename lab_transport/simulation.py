"""Discrete-time sample transport simulation with conservative grid safety."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any

from .domain import Position, Sample, Scenario
from .world_model import PublicStation, WorldModel


@dataclass
class RobotState:
    id: str
    position: Position
    home: Position
    task_id: str | None = None
    phase: str = "idle"
    service_remaining: int | None = None
    service_started: int | None = None


@dataclass
class SampleState:
    spec: Sample
    status: str = "unreleased"
    robot_id: str | None = None
    assigned_tick: int | None = None
    picked_tick: int | None = None
    completed_tick: int | None = None
    predicted_completion_tick: float | None = None


class Simulator:
    """Owns real simulation state; forecasts never receive future outage data."""

    def __init__(self, scenario: Scenario, policy: str = "world", seed: int = 0):
        if policy not in {"world", "nearest"}:
            raise ValueError("policy must be 'world' or 'nearest'")
        self.scenario = scenario
        self.policy = policy
        self.seed = seed
        self.robots = {r.id: RobotState(r.id, r.position, r.home) for r in scenario.robots}
        self.samples = {s.id: SampleState(s) for s in scenario.samples}
        self.model = WorldModel(scenario.grid, {
            sid: PublicStation(station.position, station.prior_ticks)
            for sid, station in scenario.stations.items()
        })
        self.events: list[dict[str, Any]] = []
        self.history: list[dict[str, Any]] = []
        self.prevented_moves = 0
        self._has_run = False

    def _event(self, tick: int, kind: str, **fields: Any) -> None:
        self.events.append({"tick": tick, "type": kind, **fields})

    def _service_duration(self, sample_id: str, phase: str, station_id: str) -> int:
        """Use keyed exogenous noise so the same task has the same draw across policies."""
        key = f"{self.seed}\0{sample_id}\0{phase}\0{station_id}".encode("utf-8")
        draw = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
        jitter = draw % (self.scenario.duration_jitter + 1)
        return self.scenario.stations[station_id].service_ticks + jitter

    def _current_station_busy(self, tick: int) -> dict[str, int]:
        """Estimate remaining occupancy from presence, elapsed time and past observations.

        The actual sampled service_remaining is simulator-internal future
        information and must not be handed to the predictor.
        """
        result: dict[str, int] = {}
        for sid, station in self.scenario.stations.items():
            estimates = []
            for robot in self.robots.values():
                if robot.position != station.position or robot.task_id is None:
                    continue
                if robot.service_started is not None:
                    elapsed = max(0, tick - robot.service_started)
                    estimates.append(max(1, math.ceil(self.model.expected_service(sid) - elapsed)))
                elif self._target(robot) == station.position:
                    estimates.append(math.ceil(self.model.expected_service(sid)))
                else:
                    estimates.append(1)
            result[sid] = max(estimates, default=0)
        return result

    def _dispatch(self, tick: int) -> None:
        while True:
            idle = sorted((r for r in self.robots.values() if r.task_id is None), key=lambda r: r.id)
            waiting = sorted((s for s in self.samples.values() if s.status == "waiting"),
                             key=lambda s: (s.spec.release_tick, s.spec.id))
            if not idle or not waiting:
                return
            forecasts: list[dict[str, Any]] = []
            for robot in idle:
                for sample in waiting:
                    if self.policy == "world":
                        forecast = self.model.predict(tick, robot.id, robot.position,
                                                      sample.spec, self._current_station_busy(tick))
                        key = (forecast["score"], sample.spec.release_tick, sample.spec.id, robot.id)
                    else:
                        path = self.scenario.grid.shortest_path(
                            robot.position, self.scenario.stations[sample.spec.pickup].position)
                        assert path is not None  # scenario connectivity was validated
                        forecast = {"robot_id": robot.id, "sample_id": sample.spec.id,
                                    "distance_to_pickup": len(path)}
                        key = (len(path), sample.spec.release_tick, sample.spec.id, robot.id)
                    forecasts.append({"key": key, "forecast": forecast})
            chosen = min(forecasts, key=lambda entry: entry["key"])["forecast"]
            robot = self.robots[chosen["robot_id"]]
            sample = self.samples[chosen["sample_id"]]
            robot.task_id, robot.phase = sample.spec.id, "to_pickup"
            sample.status, sample.robot_id, sample.assigned_tick = "assigned", robot.id, tick
            sample.predicted_completion_tick = chosen.get("expected_completion_tick")
            self._event(tick, "assigned", robot_id=robot.id, sample_id=sample.spec.id,
                        policy=self.policy, selected=chosen,
                        candidates=[entry["forecast"] for entry in forecasts])

    def _target(self, robot: RobotState) -> Position:
        if robot.task_id is None:
            return robot.home
        sample = self.samples[robot.task_id].spec
        station_id = sample.pickup if robot.phase == "to_pickup" else sample.dropoff
        return self.scenario.stations[station_id].position

    def _handle_station(self, tick: int, robot: RobotState) -> bool:
        if robot.task_id is None or robot.position != self._target(robot):
            return False
        sample = self.samples[robot.task_id]
        station_id = sample.spec.pickup if robot.phase == "to_pickup" else sample.spec.dropoff
        station = self.scenario.stations[station_id]
        if robot.service_remaining is None:
            robot.service_remaining = self._service_duration(sample.spec.id, robot.phase, station_id)
            robot.service_started = tick
            self._event(tick, "service_started", robot_id=robot.id, sample_id=sample.spec.id,
                        station_id=station_id, phase=robot.phase)
        if not station.available(tick):
            return True
        robot.service_remaining -= 1
        if robot.service_remaining > 0:
            return True
        assert robot.service_started is not None
        observed_duration = tick + 1 - robot.service_started
        self.model.observe_service(station_id, observed_duration)
        self._event(tick + 1, "service_finished", robot_id=robot.id, sample_id=sample.spec.id,
                    station_id=station_id, phase=robot.phase, observed_ticks=observed_duration)
        robot.service_remaining, robot.service_started = None, None
        if robot.phase == "to_pickup":
            sample.status, sample.picked_tick, robot.phase = "in_transit", tick + 1, "to_dropoff"
            self._event(tick + 1, "picked_up", robot_id=robot.id, sample_id=sample.spec.id,
                        station_id=station_id)
        else:
            sample.status, sample.completed_tick = "completed", tick + 1
            self._event(tick + 1, "delivered", robot_id=robot.id, sample_id=sample.spec.id,
                        station_id=station_id, deadline_tick=sample.spec.deadline_tick)
            robot.task_id, robot.phase = None, "idle"
        return True

    def _move(self, tick: int, serviced: set[str]) -> None:
        before = {rid: robot.position for rid, robot in self.robots.items()}
        occupied = frozenset(before.values())
        reserved: set[Position] = set()
        ids = sorted(self.robots)
        offset = tick % len(ids)
        for rid in ids[offset:] + ids[:offset]:
            robot = self.robots[rid]
            if rid in serviced or robot.position == self._target(robot):
                continue
            blocked = (occupied - {robot.position}) | reserved
            target = self._target(robot)
            # Plan toward an occupied destination, but never enter it while it
            # is occupied. Otherwise two robots at each other's destinations
            # can both wait forever without even leaving their starting cells.
            path = self.scenario.grid.shortest_path(robot.position, target, frozenset(blocked - {target}))
            if not path or path[0] in blocked:
                self.prevented_moves += 1
                continue
            robot.position = path[0]
            reserved.add(robot.position)
        after = {rid: robot.position for rid, robot in self.robots.items()}
        if len(set(after.values())) != len(after):
            raise RuntimeError("grid safety invariant failed: two robots share a cell")
        for index, first in enumerate(ids):
            for second in ids[index + 1:]:
                if before[first] == after[second] and before[second] == after[first]:
                    raise RuntimeError("grid safety invariant failed: robots swapped an edge")

    def _snapshot(self, tick: int) -> None:
        self.history.append({
            "tick": tick,
            "robots": [{"id": r.id, "position": list(r.position), "task_id": r.task_id,
                        "phase": r.phase, "service_remaining": r.service_remaining}
                       for r in sorted(self.robots.values(), key=lambda item: item.id)],
            "samples": [{"id": s.spec.id, "status": s.status, "robot_id": s.robot_id,
                         "release_tick": s.spec.release_tick, "deadline_tick": s.spec.deadline_tick}
                        for s in sorted(self.samples.values(), key=lambda item: item.spec.id)],
            "stations": [{"id": station.id, "available": station.available(tick)}
                         for station in self.scenario.stations.values()],
        })

    def run(self) -> dict[str, Any]:
        if self._has_run:
            raise RuntimeError("a Simulator instance can only be run once")
        self._has_run = True
        self._snapshot(0)
        for tick in range(self.scenario.max_ticks):
            for sample in self.samples.values():
                if sample.status == "unreleased" and sample.spec.release_tick <= tick:
                    sample.status = "waiting"
                    self._event(tick, "released", sample_id=sample.spec.id)
            self._dispatch(tick)
            serviced = {r.id for r in self.robots.values() if self._handle_station(tick, r)}
            self._move(tick, serviced)
            self._snapshot(tick + 1)
            if all(sample.status == "completed" for sample in self.samples.values()):
                break
        complete = [s for s in self.samples.values() if s.completed_tick is not None]
        errors = [abs(s.completed_tick - s.predicted_completion_tick) for s in complete
                  if s.predicted_completion_tick is not None and s.completed_tick is not None]
        late_completed = sum(s.completed_tick > s.spec.deadline_tick for s in complete)
        overdue_unfinished = sum(s.completed_tick is None and self.history[-1]["tick"] > s.spec.deadline_tick
                                 for s in self.samples.values())
        summary = {
            "status": "completed" if len(complete) == len(self.samples) else "timed_out",
            "sample_count": len(self.samples), "completed_count": len(complete),
            "late_count": late_completed + overdue_unfinished,
            "late_completed_count": late_completed, "overdue_unfinished_count": overdue_unfinished,
            "ticks": self.history[-1]["tick"],
            "mean_completion_ticks": round(sum(s.completed_tick - s.spec.release_tick for s in complete)
                                           / len(complete), 3) if complete else None,
            "forecast_mae_ticks": round(sum(errors) / len(errors), 3) if errors else None,
            "blocked_move_attempts": self.prevented_moves,
        }
        return {
            "scenario": {"name": self.scenario.name, "width": self.scenario.grid.width,
                         "height": self.scenario.grid.height,
                         "blocked": [list(p) for p in sorted(self.scenario.grid.blocked)],
                         "stations": [{"id": s.id, "position": list(s.position)}
                                      for s in self.scenario.stations.values()]},
            "policy": self.policy, "seed": self.seed, "summary": summary,
            "samples": [{"id": s.spec.id, "pickup": s.spec.pickup, "dropoff": s.spec.dropoff,
                         "release_tick": s.spec.release_tick, "deadline_tick": s.spec.deadline_tick,
                         "status": s.status, "robot_id": s.robot_id, "assigned_tick": s.assigned_tick,
                         "picked_tick": s.picked_tick, "completed_tick": s.completed_tick,
                         "predicted_completion_tick": s.predicted_completion_tick}
                        for s in sorted(self.samples.values(), key=lambda item: item.spec.id)],
            "station_observations": self.model.observations(), "events": self.events, "history": self.history,
        }
