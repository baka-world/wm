"""Small CPU hybrid model of observable, action-conditioned state transitions.

Two learned linear classifiers choose robot moves and service completions.
Static topology, assignment validity and discrete movement constraints are
explicit rules. Future arrivals and station disruptions are not predicted.
"""
from __future__ import annotations

from collections import deque
from dataclasses import replace
import math
from typing import Any

from .domain import DirectedGraph, Grid, Position
from .transition_schema import Assignment, Observation, ObservedRobot, ObservedSample

MOTION_FEATURES = ("stay", "route_progress", "remaining_distance", "at_goal", "idle_progress",
                   "pickup_progress", "dropoff_progress", "adjacent_other", "occupied_goal_stay")
SERVICE_FEATURES = ("bias", "elapsed", "expected", "elapsed_ratio", "pickup", "dropoff", "new_assignment")
EPOCHS, MOTION_LR, SERVICE_LR, L2 = 35, 0.2, 0.15, 0.001


class Geometry:
    """Known static topology only; no simulator state or future disruptions."""

    def __init__(self, grid: Grid | DirectedGraph):
        self.grid = grid
        if isinstance(grid, DirectedGraph):
            positions = tuple(sorted(grid.coordinates))
            adjacency = {p: [] for p in positions}
            for a, b in grid.edges:
                if b not in adjacency[a]:
                    adjacency[a].append(b)
        else:
            positions = tuple((x, y) for y in range(grid.height) for x in range(grid.width) if grid.contains((x, y)))
            adjacency = {}
            for x, y in positions:
                adjacency[(x, y)] = [p for p in ((x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1))
                                      if grid.contains(p)]
        self.adjacency = {p: tuple(values) for p, values in adjacency.items()}
        self.reverse: dict[Position, list[Position]] = {p: [] for p in positions}
        for a, values in self.adjacency.items():
            for b in values:
                self.reverse[b].append(a)
        self.unreachable = len(positions) + 1
        self._distances: dict[Position, dict[Position, int]] = {}

    def distance(self, start: Position, goal: Position) -> int:
        if goal not in self.reverse or start not in self.adjacency:
            raise ValueError("position is not on the known map")
        if goal not in self._distances:
            distances, queue = {goal: 0}, deque([goal])
            while queue:
                current = queue.popleft()
                for previous in self.reverse[current]:
                    if previous not in distances:
                        distances[previous] = distances[current] + 1
                        queue.append(previous)
            self._distances[goal] = distances
        return self._distances[goal].get(start, self.unreachable)

    def shortest_path(self, start: Position, goal: Position, blocked: set[Position]) -> list[Position] | None:
        if start not in self.adjacency or goal not in self.adjacency or goal in blocked:
            return None
        previous: dict[Position, Position | None] = {start: None}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            if current == goal:
                path = []
                while current != start:
                    path.append(current)
                    parent = previous[current]
                    assert parent is not None
                    current = parent
                return list(reversed(path))
            for nxt in self.adjacency[current]:
                if nxt not in blocked and nxt not in previous:
                    previous[nxt] = current
                    queue.append(nxt)
        return None


def _dot(weights: list[float], values: list[float]) -> float:
    return sum(w * value for w, value in zip(weights, values))


def validate_training_examples(moves, services, motion_width: int) -> None:
    if not moves or not services:
        raise ValueError("training needs both motion and service examples")

    def valid_vector(values, length):
        return (isinstance(values, (list, tuple)) and len(values) == length and all(
            isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in values))

    for options, label in moves:
        if (not isinstance(options, (list, tuple)) or not options or not isinstance(label, int)
                or isinstance(label, bool) or not 0 <= label < len(options)
                or any(not valid_vector(vector, motion_width) for vector in options)):
            raise ValueError("invalid motion training example")
    for values, label in services:
        if not valid_vector(values, len(SERVICE_FEATURES)) or label not in (0, 1) or isinstance(label, bool):
            raise ValueError("invalid service training example")


def _intent(observation: Observation, assignments: tuple[Assignment, ...]) -> dict[str, tuple[str, str | None]]:
    robots = {r.id: r for r in observation.robots}
    samples = {s.id: s for s in observation.samples}
    intents = {r.id: (r.phase, r.task_id) for r in observation.robots}
    used_robots, used_samples = set(), set()
    for assignment in assignments:
        robot, sample = robots.get(assignment.robot_id), samples.get(assignment.sample_id)
        if (robot is None or sample is None or robot.task_id is not None or sample.status != "waiting"
                or robot.id in used_robots or sample.id in used_samples):
            raise ValueError("assignment must select a distinct idle robot and currently waiting sample")
        intents[robot.id] = ("to_pickup", sample.id)
        used_robots.add(robot.id)
        used_samples.add(sample.id)
    for robot_id, (phase, sample_id) in intents.items():
        if phase not in {"idle", "to_pickup", "to_dropoff"}:
            raise ValueError(f"unknown robot phase for {robot_id}")
        if (phase == "idle") != (sample_id is None) or (sample_id is not None and sample_id not in samples):
            raise ValueError("observation has inconsistent robot task and phase")
    return intents


def _target(robot: ObservedRobot, intent: tuple[str, str | None],
            samples: dict[str, ObservedSample], station_positions: dict[str, Position]) -> Position:
    phase, sample_id = intent
    if sample_id is None:
        return robot.home
    sample = samples[sample_id]
    return station_positions[sample.pickup if phase == "to_pickup" else sample.dropoff]


def _motion_features(geometry: Geometry, robot: ObservedRobot, phase: str, goal: Position,
                     candidate: Position, others: tuple[Position, ...]) -> list[float]:
    before, after = geometry.distance(robot.position, goal), geometry.distance(candidate, goal)
    progress = max(-2, min(2, before - after)) / 2.0
    stay = float(candidate == robot.position)
    return [stay, progress, -min(after, 64) / 64.0, float(candidate == goal),
            progress * (phase == "idle"), progress * (phase == "to_pickup"),
            progress * (phase == "to_dropoff"),
            float(any(candidate in geometry.adjacency[p] for p in others)),
            stay * (goal in others)]


def _service_features(robot: ObservedRobot, phase: str, expected: float, new_assignment: bool) -> list[float]:
    elapsed = robot.service_elapsed or 0
    return [1.0, min(elapsed, 30) / 10.0, min(expected, 30) / 10.0,
            min(elapsed / max(expected, 0.1), 4.0) / 4.0,
            float(phase == "to_pickup"), float(phase == "to_dropoff"), float(new_assignment)]


def _motion_options(geometry: Geometry, robot: ObservedRobot, phase: str, goal: Position,
                    occupied: set[Position], reserved: set[Position]) -> list[Position]:
    # A robot receiving station service cannot move during the same step.
    if phase != "idle" and robot.position == goal:
        return [robot.position]
    blocked = (occupied - {robot.position}) | reserved
    return [robot.position] + [p for p in geometry.adjacency[robot.position] if p not in blocked]


def training_examples(geometry: Geometry, observation: Observation, assignments: tuple[Assignment, ...],
                      actual: Observation, motion_features_fn=None
                      ) -> tuple[list[tuple[list[list[float]], int]], list[tuple[list[float], int]], int]:
    """Extract supervised examples only from a factual, executed transition."""
    intents = _intent(observation, assignments)
    samples, actual_samples = {s.id: s for s in observation.samples}, {s.id: s for s in actual.samples}
    actual_robots = {r.id: r for r in actual.robots}
    stations = {s.id: s for s in observation.stations}
    station_positions = {s.id: s.position for s in observation.stations}
    robots = {r.id: r for r in observation.robots}
    ids = sorted(robots)
    if set(ids) != set(actual_robots):
        raise ValueError("robot set changed within a transition")
    offset, reserved = observation.tick % len(ids), set()
    occupied = {r.position for r in observation.robots}
    moves, services, excluded = [], [], 0
    assigned_ids = {a.sample_id for a in assignments}
    features_fn = motion_features_fn or _motion_features
    for rid in ids[offset:] + ids[:offset]:
        robot, (phase, sample_id) = robots[rid], intents[rid]
        goal = _target(robot, (phase, sample_id), samples, station_positions)
        options = _motion_options(geometry, robot, phase, goal, occupied, reserved)
        others = tuple(r.position for r in observation.robots if r.id != rid)
        actual_position = actual_robots[rid].position
        if actual_position not in options:
            excluded += 1
        elif len(options) > 1:
            moves.append(([features_fn(geometry, robot, phase, goal, p, others) for p in options],
                          options.index(actual_position)))
        if actual_position != robot.position:
            reserved.add(actual_position)
        if sample_id is not None and robot.position == goal:
            sample = samples[sample_id]
            station = stations[sample.pickup if phase == "to_pickup" else sample.dropoff]
            if station.available and sample_id in actual_samples:
                next_status = actual_samples[sample_id].status
                advanced = next_status == ("in_transit" if phase == "to_pickup" else "completed")
                services.append((_service_features(robot, phase, station.expected_service_ticks,
                                                   sample_id in assigned_ids), int(advanced)))
    return moves, services, excluded


def train(moves: list[tuple[list[list[float]], int]], services: list[tuple[list[float], int]]) -> dict[str, Any]:
    validate_training_examples(moves, services, len(MOTION_FEATURES))
    positive = sum(label for _, label in services)
    negative = len(services) - positive
    if not positive or not negative:
        raise ValueError("service training requires completed and ongoing examples")
    motion_weights, service_weights = [0.0] * len(MOTION_FEATURES), [0.0] * len(SERVICE_FEATURES)
    class_weight = {0: len(services) / (2.0 * negative), 1: len(services) / (2.0 * positive)}
    for _ in range(EPOCHS):
        gradient = [0.0] * len(motion_weights)
        for options, label in moves:
            scores = [_dot(motion_weights, option) for option in options]
            highest = max(scores)
            terms = [math.exp(score - highest) for score in scores]
            denominator = sum(terms)
            for option, weight in zip(options, terms):
                for i, value in enumerate(option):
                    gradient[i] += weight / denominator * value
            for i, value in enumerate(options[label]):
                gradient[i] -= value
        for i in range(len(motion_weights)):
            motion_weights[i] -= MOTION_LR * (gradient[i] / len(moves) + L2 * motion_weights[i])
        gradient = [0.0] * len(service_weights)
        for features, label in services:
            score = max(-40.0, min(40.0, _dot(service_weights, features)))
            error = (1.0 / (1.0 + math.exp(-score)) - label) * class_weight[label]
            for i, value in enumerate(features):
                gradient[i] += error * value
        for i in range(len(service_weights)):
            service_weights[i] -= SERVICE_LR * (gradient[i] / len(services) + L2 * service_weights[i])
    return {"schema_version": 1, "kind": "hybrid_linear_dynamics", "motion_features": list(MOTION_FEATURES),
            "service_features": list(SERVICE_FEATURES), "motion_weights": motion_weights,
            "service_weights": service_weights, "epochs": EPOCHS, "motion_learning_rate": MOTION_LR,
            "service_learning_rate": SERVICE_LR, "l2": L2,
            "training_motion_examples": len(moves), "training_service_examples": len(services),
            "training_service_completions": positive}


def validate(parameters: dict[str, Any]) -> None:
    if not isinstance(parameters, dict) or parameters.get("schema_version") != 1 or parameters.get("kind") != "hybrid_linear_dynamics":
        raise ValueError("invalid dynamics model schema")
    for feature_key, weight_key, features in (("motion_features", "motion_weights", MOTION_FEATURES),
                                              ("service_features", "service_weights", SERVICE_FEATURES)):
        weights = parameters.get(weight_key)
        if parameters.get(feature_key) != list(features) or not isinstance(weights, list) or len(weights) != len(features):
            raise ValueError("invalid dynamics feature schema")
        if any(not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x) for x in weights):
            raise ValueError("invalid dynamics model weights")


class HybridDynamicsModel:
    """Predict known entities one step; external arrivals/outages remain unknown."""

    def __init__(self, geometry: Geometry, parameters: dict[str, Any] | None = None, mode: str = "learned"):
        if mode not in {"learned", "heuristic", "persistence"}:
            raise ValueError("unknown dynamics prediction mode")
        if mode == "learned":
            validate(parameters)
        self.geometry, self.parameters, self.mode = geometry, parameters, mode

    def motion_features(self, robot: ObservedRobot, phase: str, goal: Position,
                        candidate: Position, others: tuple[Position, ...]) -> list[float]:
        return _motion_features(self.geometry, robot, phase, goal, candidate, others)

    def predict_next(self, observation: Observation, assignments: tuple[Assignment, ...]) -> Observation:
        if self.mode == "persistence":
            _intent(observation, assignments)
            return replace(observation, tick=observation.tick + 1)
        intents = _intent(observation, assignments)
        samples = {s.id: s for s in observation.samples}
        stations = {s.id: s for s in observation.stations}
        station_positions = {s.id: s.position for s in observation.stations}
        robots = {r.id: r for r in observation.robots}
        new_assignments = {a.sample_id: a.robot_id for a in assignments}
        new_samples = dict(samples)
        robot_updates: dict[str, tuple[str, str | None, int | None]] = {}
        for rid, robot in robots.items():
            phase, sample_id = intents[rid]
            if sample_id is None:
                robot_updates[rid] = ("idle", None, None)
                continue
            sample = samples[sample_id]
            station = stations[sample.pickup if phase == "to_pickup" else sample.dropoff]
            status = "assigned" if sample_id in new_assignments else sample.status
            new_samples[sample_id] = replace(sample, status=status, robot_id=rid)
            elapsed = None
            if robot.position == station.position:
                current_elapsed = robot.service_elapsed or 0
                advance = False
                if station.available:
                    if self.mode == "heuristic":
                        advance = current_elapsed + 1 >= math.ceil(station.expected_service_ticks)
                    else:
                        assert self.parameters is not None
                        features = _service_features(robot, phase, station.expected_service_ticks,
                                                     sample_id in new_assignments)
                        advance = _dot(self.parameters["service_weights"], features) >= 0
                if advance and phase == "to_pickup":
                    new_samples[sample_id] = replace(new_samples[sample_id], status="in_transit")
                    robot_updates[rid] = ("to_dropoff", sample_id, None)
                elif advance:
                    new_samples[sample_id] = replace(new_samples[sample_id], status="completed")
                    robot_updates[rid] = ("idle", None, None)
                else:
                    elapsed = current_elapsed + 1
                    robot_updates[rid] = (phase, sample_id, elapsed)
            else:
                robot_updates[rid] = (phase, sample_id, elapsed)
        ids = sorted(robots)
        if not ids:
            raise ValueError("observation requires at least one robot")
        offset, reserved, positions = observation.tick % len(ids), set(), {}
        occupied = {r.position for r in observation.robots}
        for rid in ids[offset:] + ids[:offset]:
            robot, (phase, sample_id) = robots[rid], intents[rid]
            goal = _target(robot, (phase, sample_id), samples, station_positions)
            options = _motion_options(self.geometry, robot, phase, goal, occupied, reserved)
            others = tuple(r.position for r in observation.robots if r.id != rid)
            if self.mode == "heuristic":
                blocked = (occupied - {robot.position}) | reserved
                path = self.geometry.shortest_path(robot.position, goal, blocked - {goal})
                next_pos = path[0] if path and path[0] in options else robot.position
            else:
                assert self.parameters is not None
                next_pos = max(options, key=lambda p: _dot(self.parameters["motion_weights"],
                               self.motion_features(robot, phase, goal, p, others)))
            positions[rid] = next_pos
            if next_pos != robot.position:
                reserved.add(next_pos)
        next_robots = tuple(replace(r, position=positions[r.id], phase=robot_updates[r.id][0],
                                    task_id=robot_updates[r.id][1], service_elapsed=robot_updates[r.id][2])
                            for r in observation.robots)
        return replace(observation, tick=observation.tick + 1, robots=next_robots,
                       samples=tuple(new_samples[s.id] for s in observation.samples))
