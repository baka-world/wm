"""Versioned *scoring* adapter for observed conditional service work.

Legacy neural inputs are never enlarged. Future service sessions have no
observed progress, so v1 estimates use public rule priors for those sessions.
This API expresses epistemic fallback explicitly, not as a zero observation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from experiments.service_value.scoring import _prior


@dataclass(frozen=True)
class ServiceWorkEstimate:
    prior_steps: float
    estimated_steps: float
    usable: bool
    source: str
    uncertainty: dict | None = None
    fallback_reason: str | None = None

    def record(self):
        return asdict(self)


class ServiceWorkEstimator:
    schema_version = 1

    def __init__(self, dynamics=None):
        # CapacityDynamics -> RecoveryMovementModel -> PETSRuleDynamics.
        current = dynamics
        seen = set()
        for _ in range(4):
            if current is None or id(current) in seen:
                break
            seen.add(id(current))
            if callable(getattr(current, "_parameters", None)) and getattr(current, "distribution", None) is not None:
                break
            current = getattr(current, "model", None)
        self.pets = current if callable(getattr(current, "_parameters", None)) and getattr(current, "distribution", None) is not None else None

    def estimate(self, observation, station, *, sample=None, robot=None, phase=None):
        prior = float(_prior(station))
        if self.pets is None:
            return ServiceWorkEstimate(prior, prior, False, "rule_prior",
                                       fallback_reason="no_verified_remaining_duration_distribution")
        if robot is None or sample is None or phase not in ("to_pickup", "to_dropoff"):
            return ServiceWorkEstimate(prior, prior, False, "rule_prior",
                                       fallback_reason="future_or_unstarted_service")
        if robot.task_id != sample.id or robot.phase != phase or robot.position != station.position or robot.service_elapsed is None:
            return ServiceWorkEstimate(prior, prior, False, "rule_prior",
                                       fallback_reason="no_observed_service_session")
        progress = getattr(robot, "service_active_elapsed", None)
        if progress is None:
            return ServiceWorkEstimate(prior, prior, False, "rule_prior",
                                       fallback_reason="unknown_effective_progress")
        if not isinstance(progress, int) or progress < 0:
            raise ValueError("invalid observed effective service progress")
        from lab_transport.paper_reuse.service_dataset import service_features
        features = service_features(observation, robot, sample, phase, station)
        if len(features) != 7 or any(not math.isfinite(x) for x in features):
            raise ValueError("legacy PETS feature contract violation")
        parameters = self.pets._parameters(features, transition=False)
        expected = float(self.pets.distribution.expected_remaining_steps(parameters, progress))
        if not math.isfinite(expected) or expected < 0:
            raise ValueError("invalid PETS conditional remaining estimate")
        # Member-specific conditional expectation is descriptive spread, not a CI.
        members = [
            float(self.pets.distribution.expected_remaining_steps([member], progress))
            for member in parameters
        ]
        return ServiceWorkEstimate(prior, expected, True, "pets_conditional_v1",
                                   uncertainty={"type": "ensemble_member_spread_not_calibrated",
                                                "member_remaining_steps": members,
                                                "min": min(members), "max": max(members),
                                                "mean": sum(members) / len(members),
                                                "cap_active_ticks": 60,
                                                "future_device_failure_risk": "unknown"})
