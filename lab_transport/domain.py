"""Scenario definitions and grid routing for the sample-transport prototype."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

Position = tuple[int, int]


@dataclass(frozen=True)
class Grid:
    width: int
    height: int
    blocked: frozenset[Position]

    def contains(self, position: Position) -> bool:
        x, y = position
        return 0 <= x < self.width and 0 <= y < self.height and position not in self.blocked

    def shortest_path(self, start: Position, goal: Position,
                      occupied: frozenset[Position] = frozenset()) -> list[Position] | None:
        """Return steps after start, or None if no path exists under current occupancy."""
        if not self.contains(start) or not self.contains(goal) or goal in occupied:
            return None
        queue = deque([start])
        previous: dict[Position, Position | None] = {start: None}
        while queue:
            current = queue.popleft()
            if current == goal:
                path: list[Position] = []
                while current != start:
                    path.append(current)
                    parent = previous[current]
                    assert parent is not None
                    current = parent
                return list(reversed(path))
            x, y = current
            for nxt in ((x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1)):
                if self.contains(nxt) and nxt not in occupied and nxt not in previous:
                    previous[nxt] = current
                    queue.append(nxt)
        return None


@dataclass(frozen=True)
class DirectedGraph:
    """Navigation lanes at source waypoint resolution, not a geometric safety model."""

    # Internal positions identify the upstream waypoint (index, 0); coordinates
    # are retained separately for visualization and do not imply travel time.
    coordinates: dict[Position, tuple[float, float]]
    edges: tuple[tuple[Position, Position], ...]

    @property
    def width(self) -> int:
        return 0

    @property
    def height(self) -> int:
        return 0

    @property
    def blocked(self) -> frozenset[Position]:
        return frozenset()

    def contains(self, position: Position) -> bool:
        return position in self.coordinates

    def shortest_path(self, start: Position, goal: Position,
                      occupied: frozenset[Position] = frozenset()) -> list[Position] | None:
        if not self.contains(start) or not self.contains(goal) or goal in occupied:
            return None
        previous: dict[Position, Position | None] = {start: None}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            if current == goal:
                path: list[Position] = []
                while current != start:
                    path.append(current)
                    parent = previous[current]
                    assert parent is not None
                    current = parent
                return list(reversed(path))
            for source, destination in self.edges:
                if source == current and destination not in occupied and destination not in previous:
                    previous[destination] = current
                    queue.append(destination)
        return None


@dataclass(frozen=True)
class Station:
    id: str
    position: Position
    service_ticks: int
    prior_ticks: float
    outages: tuple[tuple[int, int], ...]

    def available(self, tick: int) -> bool:
        return all(not (start <= tick < end) for start, end in self.outages)


@dataclass(frozen=True)
class RobotSpec:
    id: str
    position: Position
    home: Position


@dataclass(frozen=True)
class Sample:
    id: str
    pickup: str
    dropoff: str
    release_tick: int
    deadline_tick: int | None
    specimen_type: str | None = None
    collected_at: str | None = None
    packaging: str | None = None
    temperature_condition: str | None = None
    rule_profile: str | None = None


@dataclass(frozen=True)
class Scenario:
    name: str
    grid: Grid | DirectedGraph
    stations: dict[str, Station]
    robots: tuple[RobotSpec, ...]
    samples: tuple[Sample, ...]
    duration_jitter: int
    max_ticks: int
    metadata: dict[str, Any] | None = None


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _position(value: Any, label: str) -> Position:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{label} must be [x, y]")
    return (_integer(value[0], f"{label}.x"), _integer(value[1], f"{label}.y"))


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def load_scenario(path: str | Path) -> Scenario:
    """Load a JSON scenario and reject ambiguous, invalid or disconnected inputs."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("scenario must be a JSON object")
    grid_data = data["grid"]
    width = _integer(grid_data["width"], "grid.width", 1)
    height = _integer(grid_data["height"], "grid.height", 1)
    blocked = frozenset(_position(p, "grid.blocked") for p in grid_data.get("blocked", []))
    if any(x >= width or y >= height for x, y in blocked):
        raise ValueError("blocked position outside grid")
    grid = Grid(width, height, blocked)

    stations: dict[str, Station] = {}
    for item in data["stations"]:
        sid = _identifier(item["id"], "station.id")
        if sid in stations:
            raise ValueError(f"duplicate station id: {sid}")
        position = _position(item["position"], f"station {sid} position")
        if not grid.contains(position):
            raise ValueError(f"station {sid} is outside the walkable grid")
        service = _integer(item["service_ticks"], f"station {sid} service_ticks", 1)
        prior = item.get("prior_ticks", service)
        if not isinstance(prior, (int, float)) or isinstance(prior, bool) or not 0 < prior < float("inf"):
            raise ValueError(f"station {sid} prior_ticks must be a finite positive number")
        outages: list[tuple[int, int]] = []
        for interval in item.get("outages", []):
            if not isinstance(interval, list) or len(interval) != 2:
                raise ValueError(f"station {sid} outage must be [start, end]")
            start, end = (_integer(interval[0], "outage start"), _integer(interval[1], "outage end"))
            if end <= start:
                raise ValueError(f"station {sid} outage end must be after start")
            outages.append((start, end))
        stations[sid] = Station(sid, position, service, float(prior), tuple(outages))
    if not stations or len({s.position for s in stations.values()}) != len(stations):
        raise ValueError("stations must be non-empty and occupy distinct positions")

    robots: list[RobotSpec] = []
    for item in data["robots"]:
        rid = _identifier(item["id"], "robot.id")
        position = _position(item["position"], f"robot {rid} position")
        home = _position(item.get("home", item["position"]), f"robot {rid} home")
        if not grid.contains(position) or not grid.contains(home):
            raise ValueError(f"robot {rid} position or home is not walkable")
        robots.append(RobotSpec(rid, position, home))
    if not robots or len({r.id for r in robots}) != len(robots):
        raise ValueError("robot ids must be non-empty and unique")
    if len({r.position for r in robots}) != len(robots) or len({r.home for r in robots}) != len(robots):
        raise ValueError("robots must start at and have distinct home positions")
    if {r.home for r in robots} & {s.position for s in stations.values()}:
        raise ValueError("robot home positions must not be station positions")

    samples: list[Sample] = []
    for item in data["samples"]:
        sid = _identifier(item["id"], "sample.id")
        pickup = _identifier(item["pickup"], f"sample {sid} pickup")
        dropoff = _identifier(item["dropoff"], f"sample {sid} dropoff")
        release = _integer(item["release_tick"], f"sample {sid} release_tick")
        deadline_value = item.get("deadline_tick")
        deadline = None if deadline_value is None else _integer(deadline_value, f"sample {sid} deadline_tick")
        if pickup not in stations or dropoff not in stations or pickup == dropoff:
            raise ValueError(f"sample {sid} must name two distinct existing stations")
        if deadline is not None and deadline < release:
            raise ValueError(f"sample {sid} deadline precedes release")
        attributes = {}
        for key in ("specimen_type", "collected_at", "packaging", "temperature_condition", "rule_profile"):
            value = item.get(key)
            attributes[key] = None if value is None else _identifier(value, f"sample {sid} {key}")
        samples.append(Sample(sid, pickup, dropoff, release, deadline, **attributes))
    if not samples or len({s.id for s in samples}) != len(samples):
        raise ValueError("sample ids must be non-empty and unique")
    important = [r.position for r in robots] + [r.home for r in robots] + [s.position for s in stations.values()]
    if any(grid.shortest_path(important[0], p) is None for p in important):
        raise ValueError("all robot and station positions must be connected")
    return Scenario(
        _identifier(data.get("name", "示例实验室"), "scenario.name"), grid, stations,
        tuple(robots), tuple(samples), _integer(data.get("duration_jitter", 0), "duration_jitter"),
        _integer(data.get("max_ticks", 300), "max_ticks", 1),
    )
