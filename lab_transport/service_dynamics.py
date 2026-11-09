"""Thresholded service model and deterministic-motion ablation."""
from __future__ import annotations

from dataclasses import replace
import math

import torch

from .state_v2 import ObservationV2, rebuild
from .torch_structured_adapter import ProgressAwareRule, StructuredDynamics, make_observer


class ThresholdDynamics(StructuredDynamics):
    """The unchanged structured network, with a frozen validation threshold."""
    def __init__(self, geometry, network, device, threshold=0.5):
        if not 0 < threshold < 1:
            raise ValueError('service threshold must be strictly between 0 and 1')
        super().__init__(geometry, network, device)
        self.threshold = float(threshold)
        self.logit_threshold = math.log(threshold / (1 - threshold))

    def service_advances(self, features):
        values = torch.tensor(features, dtype=torch.float32, device=self.device)
        score = self.network.service(torch.cat((values, self._context))).item()
        if not math.isfinite(score):
            raise ValueError('model returned a non-finite service score')
        return score >= self.logit_threshold


class RuleMotionNeuralService(ThresholdDynamics):
    """Retain the rule's exact movement and use the network for service only.

    Both transitions start from the same observation/actions. In the existing
    simulator contract service is handled before movement, so the service
    result and deterministic movement can be composed without future data.
    """
    def __init__(self, geometry, network, device, threshold=0.5):
        super().__init__(geometry, network, device, threshold)
        self.rule = ProgressAwareRule(geometry)

    def score_motion(self, features):
        # Do not evaluate the neural movement head. Its temporary choice is
        # discarded and replaced by the rule's collision-checked positions.
        return [0.0] * len(features)

    def predict_next(self, observation, assignments):
        if not isinstance(observation, ObservationV2):
            observation = make_observer().observe(observation)
        motion = self.rule.predict_next(observation, assignments)
        service = super().predict_next(observation, assignments)
        positions = {r.id: r.position for r in motion.robots}
        return rebuild(service, tuple(replace(r, position=positions[r.id]) for r in service.robots))
