"""Explicit public-state forecast contracts, distinct from any simulator truth.

Every predicted state is the complete Observable V2 business snapshot, not a
neural feature vector or a dispatch score. The finite particle distribution is
a simulation approximation; it is not a calibrated joint trajectory posterior.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from lab_transport.state_v2 import ObservationV2
from lab_transport.transition_schema import Assignment


UNKNOWN_EXOGENOUS = (
    "future_unreleased_jobs",
    "future_device_outages",
    "future_conditional_retests",
)


@dataclass(frozen=True)
class EventForecast:
    robot_id: str
    sample_id: str
    phase: str
    station_id: str
    p_finish_next: float
    sampled_finished: bool


@dataclass(frozen=True)
class ForecastStep:
    before: ObservationV2
    assignments: tuple[Assignment, ...]
    next_state: ObservationV2
    service_events: tuple[EventForecast, ...]
    model_version: str
    unknown_exogenous: tuple[str, ...] = UNKNOWN_EXOGENOUS


@dataclass(frozen=True)
class ForecastParticle:
    seed: int
    steps: tuple[ForecastStep, ...]

    @property
    def final_state(self) -> ObservationV2:
        if not self.steps:
            raise ValueError("a forecast particle has no steps")
        return self.steps[-1].next_state


@dataclass(frozen=True)
class StepDistribution:
    tick: int
    # Fraction of particles that have reached each visible status by this tick.
    sample_status_probabilities: Mapping[str, Mapping[str, float]]
    robot_task_probabilities: Mapping[str, Mapping[str, float]]
    station_service_probability: Mapping[str, float]
    resource_occupancy_mean: Mapping[str, float]
    expected_waiting_samples: float
    expected_completed_samples: float
    expected_in_transit_samples: float


@dataclass(frozen=True)
class ForecastBatch:
    model_version: str
    initial_state: ObservationV2
    first_action: tuple[Assignment, ...]
    continuation_policy_name: str
    particles: tuple[ForecastParticle, ...]
    distributions: tuple[StepDistribution, ...]
    unknown_exogenous: tuple[str, ...] = UNKNOWN_EXOGENOUS
