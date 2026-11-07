"""Observable transition contracts for future CPU dynamics models.

Only current, available information belongs here. The live simulator retains
additional private state which must not be supplied as model input.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, Sequence

from .domain import Position

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ObservedRobot:
    id: str
    position: Position
    home: Position
    task_id: str | None
    phase: str
    service_elapsed: int | None


@dataclass(frozen=True)
class ObservedSample:
    id: str
    pickup: str
    dropoff: str
    status: str
    robot_id: str | None
    release_tick: int
    deadline_tick: int | None
    specimen_type: str | None
    packaging: str | None
    temperature_condition: str | None
    rule_profile: str | None


@dataclass(frozen=True)
class ObservedStation:
    id: str
    position: Position
    available: bool
    expected_service_ticks: float


@dataclass(frozen=True)
class Observation:
    schema_version: int
    tick: int
    robots: tuple[ObservedRobot, ...]
    samples: tuple[ObservedSample, ...]
    stations: tuple[ObservedStation, ...]


@dataclass(frozen=True)
class Assignment:
    """One observed dispatch assignment, not a robot's low-level motion."""

    robot_id: str
    sample_id: str


@dataclass(frozen=True)
class Transition:
    """One factual step; no outcome for unexecuted candidates is implied."""

    observation: Observation
    assignments: tuple[Assignment, ...]
    next_observation: Observation
    next_boundary: Literal["before_dispatch", "end_of_run"]
    termination: Literal["completed", "time_limit"] | None


class DynamicsModel(Protocol):
    """One-step contract for a partially observable dynamics predictor.

    A model can be initialized separately with known static map information.
    Observations are partial and future arrivals/outages are unobserved.
    """

    def predict_next(self, observation: Observation, assignments: tuple[Assignment, ...]) -> Observation:
        ...


class DispatchPolicy(Protocol):
    """Choose zero or more compatible assignments from current candidates."""

    def __call__(self, observation: Observation, candidates: tuple[Assignment, ...]) -> Sequence[Assignment]:
        ...
