"""Discrete-time sample transport simulation with conservative grid safety."""
from __future__ import annotations

from dataclasses import dataclass, replace
from collections import deque
import hashlib
from itertools import product
import math
from typing import Any, Callable

from .domain import DirectedGraph, Position, RobotSpec, Sample, Scenario, Station
from .world_model import PublicStation, WorldModel
from .learned_model import LearnedWorldModel
from .transition_schema import (SCHEMA_VERSION, Assignment, DispatchPolicy, Observation, ObservedRobot,
                                ObservedSample, ObservedStation, Transition)


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

    def __init__(self, scenario: Scenario, policy: str = "world", seed: int = 0,
                 learned_parameters: dict | None = None, *, motion_pauses: dict | None = None,
                 collect_metrics: bool = False):
        if policy not in {"world", "nearest", "rule", "learned", "learned_rule", "rollout"}:
            raise ValueError("unknown dispatch policy")
        if policy in {"learned", "learned_rule"} and learned_parameters is None:
            raise ValueError("learned policy requires trained parameters")
        self.scenario = scenario
        self.policy = policy
        self.seed = seed
        self.robots = {r.id: RobotState(r.id, r.position, r.home) for r in scenario.robots}
        self.motion_pauses = {}
        for rid, intervals in (motion_pauses or {}).items():
            if rid not in self.robots or not isinstance(intervals, (tuple, list)):
                raise ValueError("motion pause names an unknown robot or invalid intervals")
            checked = []
            for interval in intervals:
                if (not isinstance(interval, (tuple, list)) or len(interval) != 2 or
                        any(not isinstance(x, int) or isinstance(x, bool) for x in interval) or
                        interval[0] < 0 or interval[1] <= interval[0]):
                    raise ValueError("motion pause must be a nonnegative [start, end) interval")
                checked.append(tuple(interval))
            self.motion_pauses[rid] = tuple(checked)
        self.collect_metrics = collect_metrics
        self.metrics = {"station_unavailable_wait_ticks": 0, "motion_pause_ticks": 0,
                        "path_search_calls": 0, "blocked_station_target_attempts": 0}
        self.samples = {s.id: SampleState(s) for s in scenario.samples}
        stations = {
            sid: PublicStation(station.position, station.prior_ticks)
            for sid, station in scenario.stations.items()
        }
        self.model = (LearnedWorldModel(scenario.grid, stations, learned_parameters)
                      if policy in {"learned", "learned_rule"} else WorldModel(scenario.grid, stations))
        self.events: list[dict[str, Any]] = []
        self.history: list[dict[str, Any]] = []
        self.prevented_moves = 0
        self._recovery: list[tuple[tuple[Position, ...], tuple[Position, ...]]] = []
        self._recovery_goal: tuple[str, str | None, str] | None = None
        self._has_run = False

    def _event(self, tick: int, kind: str, **fields: Any) -> None:
        self.events.append({"tick": tick, "type": kind, **fields})

    def _service_duration(self, sample_id: str, phase: str, station_id: str) -> int:
        """Use keyed exogenous noise so the same task has the same draw across policies."""
        key = f"{self.seed}\0{sample_id}\0{phase}\0{station_id}".encode("utf-8")
        draw = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
        station = self.scenario.stations[station_id]
        jitter_limit = self.scenario.duration_jitter if station.jitter_ticks is None else station.jitter_ticks
        jitter = draw % (jitter_limit + 1)
        return station.service_ticks + jitter

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

    def _observable_rollout(self, tick: int, robot: RobotState, sample: SampleState,
                            horizon: int = 120) -> dict[str, Any]:
        """Evaluate one assignment on a bounded copy of the *observed* state.

        Future releases and outages, RNG draws and actual service countdowns
        never enter the copy. Current downtime is assumed to last one more
        tick; subsequent services use rounded observed means. The same motion
        rules and two-robot recovery as the live simulator are used.
        """
        if horizon < 1:
            raise ValueError("rollout horizon must be positive")
        visible = [s for s in self.samples.values() if s.status in {"waiting", "assigned", "in_transit"}]
        stations = {}
        for sid, station in self.scenario.stations.items():
            expected = self.model.expected_service(sid)
            stations[sid] = Station(sid, station.position, max(1, math.ceil(expected)), expected,
                                    () if station.available(tick) else ((0, 1),), 0)
        shadow_specs = tuple(replace(s.spec, release_tick=0,
                                     deadline_tick=None if s.spec.deadline_tick is None
                                     else s.spec.deadline_tick - tick) for s in visible)
        shadow_robots = tuple(RobotSpec(r.id, r.position, r.home) for r in self.robots.values())
        shadow_scenario = Scenario("observable rollout", self.scenario.grid, stations, shadow_robots,
                                   shadow_specs, 0, horizon)
        shadow = Simulator(shadow_scenario, "world", 0)
        for sid, values in self.model.observations().items():
            for value in values:
                shadow.model.observe_service(sid, value)
        for old in visible:
            new = shadow.samples[old.spec.id]
            new.status, new.robot_id = old.status, old.robot_id
        for old in self.robots.values():
            new = shadow.robots[old.id]
            new.task_id, new.phase = old.task_id, old.phase
            if old.task_id is not None and old.service_started is not None:
                target_sample = self.samples[old.task_id].spec
                sid = target_sample.pickup if old.phase == "to_pickup" else target_sample.dropoff
                elapsed = max(0, tick - old.service_started)
                new.service_started = -elapsed
                new.service_remaining = max(1, math.ceil(self.model.expected_service(sid) - elapsed))
        proposed = shadow.samples[sample.spec.id]
        proposed.status, proposed.robot_id = "assigned", robot.id
        shadow.robots[robot.id].task_id, shadow.robots[robot.id].phase = sample.spec.id, "to_pickup"
        for step in range(horizon):
            shadow._dispatch(step)
            serviced = {r.id for r in shadow.robots.values() if shadow._handle_station(step, r)}
            # The live motion priority rotates with absolute tick parity.
            shadow._move(tick + step, serviced)
            if proposed.completed_tick is not None:
                break
        completed = proposed.completed_tick is not None
        completion = tick + proposed.completed_tick if completed else None
        pickup = tick + proposed.picked_tick if proposed.picked_tick is not None else None
        late = (max(0, completion - sample.spec.deadline_tick)
                if completed and sample.spec.deadline_tick is not None else None)
        # A horizon-censored candidate receives the horizon as a lower-bound
        # score; this is a ranking convention, not an infeasibility proof.
        score = completion + 4 * (late or 0) if completed else tick + horizon
        return {"robot_id": robot.id, "sample_id": sample.spec.id,
                "expected_pickup_tick": pickup, "expected_completion_tick": completion,
                "predicted_final_sample_status": "completed" if completed else "unknown",
                "predicted_lateness": late, "score": score, "predictor": "observable_rollout",
                "rollout_horizon": horizon, "rollout_horizon_reached": not completed}

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
                    if self.policy == "rollout":
                        forecast = self._observable_rollout(tick, robot, sample)
                        key = (forecast["score"], sample.spec.release_tick, sample.spec.id, robot.id)
                    elif self.policy in {"world", "rule", "learned", "learned_rule"}:
                        # Only the current observed status and positions are
                        # shared. Future releases, outages and RNG are hidden.
                        context = {
                            "other_positions": tuple(r.position for r in self.robots.values() if r.id != robot.id),
                            "other_active": sum(r.task_id is not None for r in self.robots.values() if r.id != robot.id),
                            "waiting_count": len(waiting),
                            "pickup_unavailable_now": int(not self.scenario.stations[sample.spec.pickup].available(tick)),
                            "dropoff_unavailable_now": int(not self.scenario.stations[sample.spec.dropoff].available(tick)),
                        }
                        forecast = self.model.predict(tick, robot.id, robot.position,
                                                      sample.spec, self._current_station_busy(tick), context)
                        if self.policy in {"rule", "learned_rule"}:
                            # Earliest known deadline first, including already
                            # infeasible tasks; use prediction to choose a robot.
                            due = sample.spec.deadline_tick if sample.spec.deadline_tick is not None else float("inf")
                            key = (due, forecast["score"], sample.spec.release_tick, sample.spec.id, robot.id)
                        else:
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

    def _dispatch_external(self, tick: int, observation: Observation, policy: DispatchPolicy) -> None:
        """Validate an injected policy's whole assignment batch before applying it."""
        idle = sorted((r for r in self.robots.values() if r.task_id is None), key=lambda r: r.id)
        waiting = sorted((s for s in self.samples.values() if s.status == "waiting"),
                         key=lambda s: (s.spec.release_tick, s.spec.id))
        candidates = tuple(Assignment(r.id, s.spec.id) for r in idle for s in waiting)
        if not candidates:
            return
        actions = policy(observation, candidates)
        if not isinstance(actions, (tuple, list)) or any(not isinstance(a, Assignment) for a in actions):
            raise ValueError("injected policy must return a list or tuple of Assignment objects")
        if any(action not in candidates for action in actions):
            raise ValueError("injected policy selected an unavailable candidate")
        if len({a.robot_id for a in actions}) != len(actions) or len({a.sample_id for a in actions}) != len(actions):
            raise ValueError("injected assignments must use distinct robots and samples")
        candidate_records = [{"robot_id": a.robot_id, "sample_id": a.sample_id} for a in candidates]
        for action in actions:
            robot, sample = self.robots[action.robot_id], self.samples[action.sample_id]
            robot.task_id, robot.phase = sample.spec.id, "to_pickup"
            sample.status, sample.robot_id, sample.assigned_tick = "assigned", robot.id, tick
            sample.predicted_completion_tick = None
            selected = {"robot_id": robot.id, "sample_id": sample.spec.id}
            self._event(tick, "assigned", robot_id=robot.id, sample_id=sample.spec.id,
                        policy="external", selected=selected, candidates=candidate_records)

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
            if self.collect_metrics:
                self.metrics["station_unavailable_wait_ticks"] += 1
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
        paused = {rid for rid, intervals in self.motion_pauses.items()
                  if any(start <= tick < end for start, end in intervals)}
        immobilized = serviced | paused
        if self._follow_recovery(tick, before, immobilized):
            return
        occupied = frozenset(before.values())
        reserved: set[Position] = set()
        ids = sorted(self.robots)
        offset = tick % len(ids)
        for rid in ids[offset:] + ids[:offset]:
            robot = self.robots[rid]
            if rid in serviced or robot.position == self._target(robot):
                continue
            if rid in paused:
                if self.collect_metrics:
                    self.metrics["motion_pause_ticks"] += 1
                continue
            blocked = (occupied - {robot.position}) | reserved
            target = self._target(robot)
            # Plan toward an occupied destination, but never enter it while it
            # is occupied. Otherwise two robots at each other's destinations
            # can both wait forever without even leaving their starting cells.
            if self.collect_metrics:
                self.metrics["path_search_calls"] += 1
            path = self.scenario.grid.shortest_path(robot.position, target, frozenset(blocked - {target}))
            if not path or path[0] in blocked:
                self.prevented_moves += 1
                if self.collect_metrics and target in blocked and any(
                        station.position == target for station in self.scenario.stations.values()):
                    self.metrics["blocked_station_target_attempts"] += 1
                continue
            robot.position = path[0]
            reserved.add(robot.position)
        after = {rid: robot.position for rid, robot in self.robots.items()}
        if before == after and isinstance(self.scenario.grid, DirectedGraph):
            self._plan_recovery(tick, before, immobilized)
            if self._follow_recovery(tick, before, immobilized):
                return
        if len(set(after.values())) != len(after):
            raise RuntimeError("grid safety invariant failed: two robots share a cell")
        for index, first in enumerate(ids):
            for second in ids[index + 1:]:
                if before[first] == after[second] and before[second] == after[first]:
                    raise RuntimeError("grid safety invariant failed: robots swapped an edge")

    def _plan_recovery(self, tick: int, before: dict[str, Position], serviced: set[str]) -> None:
        """Search joint waypoint states for a local way out of a complete standstill.

        This limited two-robot fallback is not a continuous-space safety or
        completeness guarantee. It gives already-loaded jobs preference.
        """
        if len(self.robots) != 2:
            return
        ids = tuple(sorted(self.robots))
        candidates = [r for r in self.robots.values()
                      if r.task_id is not None and r.id not in serviced and r.position != self._target(r)]
        if not candidates:
            return
        chosen = min(candidates, key=lambda r: (r.phase != "to_dropoff", r.id))
        target, target_index = self._target(chosen), ids.index(chosen.id)
        graph = self.scenario.grid
        assert isinstance(graph, DirectedGraph)
        start = tuple(before[rid] for rid in ids)
        queue = deque([start])
        previous: dict[tuple[Position, ...], tuple[Position, ...] | None] = {start: None}
        found = None
        while queue and len(previous) <= 10000:
            state = queue.popleft()
            if state[target_index] == target:
                found = state
                break
            options = []
            for index, rid in enumerate(ids):
                current = state[index]
                choices = [current]
                if rid not in serviced:
                    choices.extend(b for a, b in graph.edges if a == current)
                options.append(choices)
            for nxt in product(*options):
                if nxt == state or nxt[0] == nxt[1] or nxt[0] == state[1] or nxt[1] == state[0]:
                    continue
                if nxt not in previous:
                    previous[nxt] = state
                    queue.append(nxt)
        if found is None:
            return
        states = [found]
        while states[-1] != start:
            parent = previous[states[-1]]
            assert parent is not None
            states.append(parent)
        states.reverse()
        self._recovery = list(zip(states, states[1:]))
        self._recovery_goal = (chosen.id, chosen.task_id, chosen.phase)
        self._event(tick, "recovery_planned", robot_id=chosen.id, sample_id=chosen.task_id,
                    steps=len(self._recovery), explored_states=len(previous))

    def _follow_recovery(self, tick: int, before: dict[str, Position], serviced: set[str]) -> bool:
        if not self._recovery or self._recovery_goal is None:
            return False
        ids = tuple(sorted(self.robots))
        rid, task, phase = self._recovery_goal
        robot = self.robots[rid]
        expected, nxt = self._recovery[0]
        if ((robot.task_id, robot.phase) != (task, phase) or tuple(before[x] for x in ids) != expected
                or any(name in serviced and before[name] != nxt[i] for i, name in enumerate(ids))):
            self._recovery, self._recovery_goal = [], None
            self._event(tick, "recovery_cancelled", robot_id=rid)
            return False
        for i, name in enumerate(ids):
            self.robots[name].position = nxt[i]
        self._recovery.pop(0)
        if not self._recovery:
            self._recovery_goal = None
        return True

    def _snapshot(self, tick: int) -> None:
        self.history.append({
            "tick": tick,
            "robots": [{"id": r.id, "position": list(r.position), "task_id": r.task_id,
                        "phase": r.phase, "service_remaining": r.service_remaining}
                       for r in sorted(self.robots.values(), key=lambda item: item.id)],
            "samples": [{"id": s.spec.id, "status": s.status, "robot_id": s.robot_id,
                         "release_tick": s.spec.release_tick, "deadline_tick": s.spec.deadline_tick,
                         "specimen_type": s.spec.specimen_type}
                        for s in sorted(self.samples.values(), key=lambda item: item.spec.id)],
            "stations": [{"id": station.id, "available": station.available(tick)}
                         for station in self.scenario.stations.values()],
        })

    def _observable_state(self, tick: int) -> Observation:
        """Build a current-state observation, excluding unreleased and hidden state.

        This is called by the run loop at its current tick only, never as an
        arbitrary-time query against the simulator's future outage schedule.
        """
        return Observation(
            SCHEMA_VERSION, tick,
            tuple(ObservedRobot(r.id, r.position, r.home, r.task_id, r.phase,
                                None if r.service_started is None else max(0, tick - r.service_started))
                  for r in sorted(self.robots.values(), key=lambda item: item.id)),
            tuple(ObservedSample(s.spec.id, s.spec.pickup, s.spec.dropoff, s.status, s.robot_id,
                                 s.spec.release_tick, s.spec.deadline_tick, s.spec.specimen_type,
                                 s.spec.packaging, s.spec.temperature_condition, s.spec.rule_profile)
                  for s in sorted(self.samples.values(), key=lambda item: item.spec.id)
                  if s.status != "unreleased"),
            tuple(ObservedStation(sid, station.position, station.available(tick), self.model.expected_service(sid))
                  for sid, station in sorted(self.scenario.stations.items())),
        )

    def run(self, on_transition: Callable[[Transition], None] | None = None,
            dispatch_policy: DispatchPolicy | None = None,
            on_observation: Callable[[Observation], Any] | None = None) -> dict[str, Any]:
        if self._has_run:
            raise RuntimeError("a Simulator instance can only be run once")
        self._has_run = True
        self._snapshot(0)
        pending: tuple[Observation, tuple[Assignment, ...]] | None = None
        for tick in range(self.scenario.max_ticks):
            for sample in self.samples.values():
                if sample.status == "unreleased" and sample.spec.release_tick <= tick:
                    sample.status = "waiting"
                    self._event(tick, "released", sample_id=sample.spec.id)
            current = self._observable_state(tick) if (on_transition is not None or dispatch_policy is not None
                                                      or on_observation is not None) else None
            if on_observation is not None:
                assert current is not None
                on_observation(current)
            if on_transition is not None and pending is not None:
                assert current is not None
                on_transition(Transition(pending[0], pending[1], current, "before_dispatch", None))
            events_before_dispatch = len(self.events) if on_transition is not None else 0
            if dispatch_policy is None:
                self._dispatch(tick)
            else:
                assert current is not None
                self._dispatch_external(tick, current, dispatch_policy)
            assignments = (tuple(Assignment(e["robot_id"], e["sample_id"])
                                 for e in self.events[events_before_dispatch:] if e["type"] == "assigned")
                           if on_transition is not None else ())
            serviced = {r.id for r in self.robots.values() if self._handle_station(tick, r)}
            self._move(tick, serviced)
            self._snapshot(tick + 1)
            completed = all(sample.status == "completed" for sample in self.samples.values())
            final_step = completed or tick + 1 == self.scenario.max_ticks
            if on_transition is not None:
                assert current is not None
                if final_step:
                    termination = "completed" if completed else "time_limit"
                    on_transition(Transition(current, assignments, self._observable_state(tick + 1),
                                             "end_of_run", termination))
                else:
                    pending = (current, assignments)
            if completed:
                break
        complete = [s for s in self.samples.values() if s.completed_tick is not None]
        errors = [abs(s.completed_tick - s.predicted_completion_tick) for s in complete
                  if s.predicted_completion_tick is not None and s.completed_tick is not None]
        deadlines = [s for s in self.samples.values() if s.spec.deadline_tick is not None]
        late_completed = sum(s.completed_tick is not None and s.completed_tick > s.spec.deadline_tick
                             for s in deadlines)
        overdue_unfinished = sum(s.completed_tick is None and self.history[-1]["tick"] > s.spec.deadline_tick
                                 for s in deadlines)
        summary = {
            "status": "completed" if len(complete) == len(self.samples) else "timed_out",
            "sample_count": len(self.samples), "completed_count": len(complete),
            "deadline_sample_count": len(deadlines),
            "late_count": late_completed + overdue_unfinished if deadlines else None,
            "late_completed_count": late_completed if deadlines else None,
            "overdue_unfinished_count": overdue_unfinished if deadlines else None,
            "ticks": self.history[-1]["tick"],
            "mean_completion_ticks": round(sum(s.completed_tick - s.spec.release_tick for s in complete)
                                           / len(complete), 3) if complete else None,
            "forecast_mae_ticks": round(sum(errors) / len(errors), 3) if errors else None,
            "blocked_move_attempts": self.prevented_moves,
        }
        if self.collect_metrics:
            summary.update(self.metrics)
            summary["recovery_plans"] = sum(e["type"] == "recovery_planned" for e in self.events)
        scenario_info: dict[str, Any] = {"name": self.scenario.name, "width": self.scenario.grid.width,
                         "height": self.scenario.grid.height,
                         "blocked": [list(p) for p in sorted(self.scenario.grid.blocked)],
                         "stations": [{"id": s.id, "position": list(s.position)}
                                      for s in self.scenario.stations.values()]}
        if isinstance(self.scenario.grid, DirectedGraph):
            scenario_info["graph"] = {
                "nodes": [{"position": list(pos), "x": xy[0], "y": xy[1]}
                          for pos, xy in self.scenario.grid.coordinates.items()],
                "edges": [[list(a), list(b)] for a, b in self.scenario.grid.edges],
            }
        if self.scenario.metadata:
            scenario_info["metadata"] = dict(self.scenario.metadata)
        if self.policy in {"learned", "learned_rule"} and dispatch_policy is None:
            scenario_info["prediction_model"] = {
                "kind": "ridge_residual",
                "training_scope": self.model.parameters.get("training_scope", "unspecified"),
                "training_records": self.model.parameters.get("training_records"),
                "note": "学习参数来自仿真；跨地图及真实环境的有效性尚未验证。",
            }
        return {
            "scenario": scenario_info,
            "policy": "external" if dispatch_policy is not None else self.policy, "seed": self.seed, "summary": summary,
            "samples": [{"id": s.spec.id, "pickup": s.spec.pickup, "dropoff": s.spec.dropoff,
                         "release_tick": s.spec.release_tick, "deadline_tick": s.spec.deadline_tick,
                         "specimen_type": s.spec.specimen_type, "collected_at": s.spec.collected_at,
                         "packaging": s.spec.packaging, "temperature_condition": s.spec.temperature_condition,
                         "rule_profile": s.spec.rule_profile,
                         "status": s.status, "robot_id": s.robot_id, "assigned_tick": s.assigned_tick,
                         "picked_tick": s.picked_tick, "completed_tick": s.completed_tick,
                         "predicted_completion_tick": s.predicted_completion_tick}
                        for s in sorted(self.samples.values(), key=lambda item: item.spec.id)],
            "station_observations": self.model.observations(), "events": self.events, "history": self.history,
        }
