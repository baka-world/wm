"""Action-conditioned entity-pooling dynamics plus explicit simulator rules.

Tokens contain only current observations and the candidate assignment. Factual
next observations are used solely to create supervised labels.
"""
from __future__ import annotations

import math
import time

import torch
from torch import nn
from torch.nn import functional as F

from .dynamics_model import (Geometry, HybridDynamicsModel, SERVICE_FEATURES, _intent,
                             _motion_options, _service_features, _target)
from .dynamics_revision import FEATURES_V2, motion_features_v2
from .state_v2 import ObservableHistory, ObservationV2, predicted_progress, progress_adjusted

ADAPTER_VERSION = 1
ENTITY_WIDTH = 26


def make_observer():
    return ObservableHistory()


def _clip(value, scale=30):
    return max(-2.0, min(2.0, float(value) / scale))


def entity_tokens(observation, assignments):
    """Permutation-invariant typed entities; identifiers are used for joins only."""
    _intent(observation, assignments)
    assigned_robots = {a.robot_id for a in assignments}
    assigned_samples = {a.sample_id for a in assignments}
    tokens = []
    for r in observation.robots:
        active, paused = getattr(r, "service_active_elapsed", None), getattr(r, "service_paused_elapsed", None)
        tokens.append([1., 0., 0., float(r.phase == "idle"), float(r.phase == "to_pickup"),
                       float(r.phase == "to_dropoff"), 0., 0., 0., 0., 0., 0., 0.,
                       0., 0., _clip(r.service_elapsed or 0), _clip(active or 0), _clip(paused or 0),
                       float(active is not None), 0., 0., 0., 0., float(r.id in assigned_robots), 0., 0.])
    for s in observation.samples:
        tokens.append([0., 1., 0., 0., 0., 0., float(s.status == "waiting"), float(s.status == "assigned"),
                       float(s.status == "in_transit"), float(s.status == "completed"), 0., 0., 0.,
                       _clip(max(0, observation.tick - s.release_tick) if s.status == "waiting" else 0),
                       _clip(s.deadline_tick - observation.tick) if s.deadline_tick is not None else 0.,
                       0., 0., 0., 0., 0., 0., 0., 0., float(s.id in assigned_samples),
                       float(s.deadline_tick is not None), 0.])
    for s in observation.stations:
        mean = getattr(s, "observed_effective_service_mean", None)
        tokens.append([0., 0., 1., 0., 0., 0., 0., 0., 0., 0., float(s.available),
                       float(getattr(s, "occupant_robot_id", None) is not None),
                       float(getattr(s, "service_in_progress", False)), 0., 0., 0., 0., 0., 0.,
                       _clip(mean if mean is not None else s.expected_service_ticks),
                       _clip(getattr(s, "waiting_pickups", 0), 10),
                       _clip(getattr(s, "approaching_pickups", 0), 10),
                       _clip(getattr(s, "approaching_dropoffs", 0), 10), 0., 0., float(mean is not None)])
    if not tokens or any(len(row) != ENTITY_WIDTH for row in tokens):
        raise ValueError("invalid entity tokens")
    return tokens


class StructuredNetwork(nn.Module):
    """Shared typed-entity MLP and masked pooling, with two action heads."""
    def __init__(self):
        super().__init__()
        self.entity = nn.Sequential(nn.Linear(ENTITY_WIDTH, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.motion = nn.Sequential(nn.Linear(len(FEATURES_V2) + 32, 32), nn.ReLU(), nn.Linear(32, 1))
        self.service = nn.Sequential(nn.Linear(len(SERVICE_FEATURES) + 32, 32), nn.ReLU(), nn.Linear(32, 1))

    def context(self, entities, mask):
        encoded = self.entity(entities) * mask.unsqueeze(-1)
        return encoded.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)

    def forward(self, inputs):
        mc = self.context(inputs["motion_entities"], inputs["motion_entities_mask"])
        motion_context = mc.unsqueeze(1).expand(-1, inputs["motion"].shape[1], -1)
        scores = self.motion(torch.cat((inputs["motion"], motion_context), dim=-1)).squeeze(-1)
        scores = scores.masked_fill(~inputs["motion_mask"], float("-inf"))
        sc = self.context(inputs["service_entities"], inputs["service_entities_mask"])
        service = self.service(torch.cat((inputs["service"], sc), dim=-1)).squeeze(-1)
        return {"motion": scores, "service": service}


def build_model():
    return StructuredNetwork()


def encode_transition(geometry, observation, assignments, actual):
    if not isinstance(observation, ObservationV2):
        observation = make_observer().observe(observation)
    tokens = entity_tokens(observation, assignments)
    adjusted = progress_adjusted(observation)
    intents = _intent(adjusted, assignments)
    samples = {s.id: s for s in adjusted.samples}
    stations = {s.id: s for s in adjusted.stations}
    positions = {s.id: s.position for s in adjusted.stations}
    next_robots = {r.id: r for r in actual.robots}
    next_samples = {s.id: s for s in actual.samples}
    assigned = {a.sample_id for a in assignments}
    occupied = {r.position for r in adjusted.robots}
    items = []
    for robot in adjusted.robots:
        phase, sample_id = intents[robot.id]
        goal = _target(robot, (phase, sample_id), samples, positions)
        options = _motion_options(geometry, robot, phase, goal, occupied, set())
        others = tuple(r.position for r in adjusted.robots if r.id != robot.id)
        if len(options) > 1 and next_robots[robot.id].position in options:
            items.append({"kind": "motion", "entities": tokens,
                          "input": [motion_features_v2(geometry, robot, phase, goal, p, others) for p in options],
                          "target": options.index(next_robots[robot.id].position)})
        if sample_id is not None and robot.position == goal:
            sample = samples[sample_id]
            station = stations[sample.pickup if phase == "to_pickup" else sample.dropoff]
            if station.available and sample_id in next_samples:
                # The corrected observed effective mean is preferred; the old
                # wall-time prior is a documented fallback, never hidden truth.
                expected = station.observed_effective_service_mean
                if expected is None:
                    expected = station.expected_service_ticks
                status = "in_transit" if phase == "to_pickup" else "completed"
                items.append({"kind": "service", "entities": tokens,
                              "input": _service_features(robot, phase, expected, sample_id in assigned),
                              "target": int(next_samples[sample_id].status == status)})
    return items


def collate(items):
    moves = [i for i in items if i["kind"] == "motion"]
    services = [i for i in items if i["kind"] == "service"]
    width = max((len(i["input"]) for i in moves), default=1)
    motion = torch.zeros((len(moves), width, len(FEATURES_V2)))
    motion_mask = torch.zeros((len(moves), width), dtype=torch.bool)
    for index, item in enumerate(moves):
        n = len(item["input"])
        motion[index, :n] = torch.tensor(item["input"])
        motion_mask[index, :n] = True

    def entities(group):
        n = max((len(i["entities"]) for i in group), default=1)
        values = torch.zeros((len(group), n, ENTITY_WIDTH))
        mask = torch.zeros((len(group), n), dtype=torch.bool)
        for index, item in enumerate(group):
            count = len(item["entities"])
            values[index, :count] = torch.tensor(item["entities"])
            mask[index, :count] = True
        return values, mask

    me, mm = entities(moves)
    se, sm = entities(services)
    inputs = {"motion": motion, "motion_mask": motion_mask,
              "motion_entities": me, "motion_entities_mask": mm,
              "service": torch.tensor([i["input"] for i in services]).reshape(-1, len(SERVICE_FEATURES)),
              "service_entities": se, "service_entities_mask": sm}
    labels = {"motion": torch.tensor([i["target"] for i in moves], dtype=torch.long),
              "service": torch.tensor([i["target"] for i in services], dtype=torch.float32)}
    return inputs, labels


def loss(outputs, labels):
    count = labels["motion"].numel() + labels["service"].numel()
    if not count:
        raise ValueError("empty batch")
    total = torch.zeros((), device=outputs["motion"].device)
    if labels["motion"].numel():
        total = total + F.cross_entropy(outputs["motion"], labels["motion"], reduction="sum")
    if labels["service"].numel():
        total = total + F.binary_cross_entropy_with_logits(outputs["service"], labels["service"], reduction="sum")
    return total / count, count


class StructuredDynamics(HybridDynamicsModel):
    def __init__(self, geometry, network, device):
        self.geometry, self.parameters, self.mode = geometry, None, "learned"
        self.network, self.device = network.to(device).eval(), device
        self._context = None
        self.process_cpu_seconds = 0.0

    def predict_next(self, observation, assignments):
        if not isinstance(observation, ObservationV2):
            observation = make_observer().observe(observation)
        started = time.process_time()
        with torch.inference_mode():
            values = torch.tensor([entity_tokens(observation, assignments)], dtype=torch.float32, device=self.device)
            self._context = self.network.context(values, torch.ones(values.shape[:2], dtype=torch.bool, device=self.device))[0]
            adjusted = progress_adjusted(observation)
            # Use only observed completed-service means. Fall back to v1 prior.
            from dataclasses import replace
            adjusted = replace(adjusted, stations=tuple(replace(s, expected_service_ticks=(
                s.observed_effective_service_mean if s.observed_effective_service_mean is not None
                else s.expected_service_ticks)) for s in adjusted.stations))
            result = super().predict_next(adjusted, assignments)
        self.process_cpu_seconds += time.process_time() - started
        return predicted_progress(observation, result, assignments)

    def motion_features(self, robot, phase, goal, candidate, others):
        return motion_features_v2(self.geometry, robot, phase, goal, candidate, others)

    def score_motion(self, features):
        values = torch.tensor(features, dtype=torch.float32, device=self.device)
        context = self._context.expand(values.shape[0], -1)
        return self.network.motion(torch.cat((values, context), dim=-1)).squeeze(-1).cpu().tolist()

    def service_advances(self, features):
        values = torch.tensor(features, dtype=torch.float32, device=self.device)
        score = self.network.service(torch.cat((values, self._context))).item()
        if not math.isfinite(score):
            raise ValueError("model returned an invalid service score")
        return score >= 0


class ProgressAwareRule(HybridDynamicsModel):
    """Rule baseline with the same observable effective-progress correction."""
    def __init__(self, geometry):
        super().__init__(geometry, mode="heuristic")

    def predict_next(self, observation, assignments):
        from dataclasses import replace
        if not isinstance(observation, ObservationV2):
            observation = make_observer().observe(observation)
        adjusted = progress_adjusted(observation)
        adjusted = replace(adjusted, stations=tuple(replace(s, expected_service_ticks=(
            s.observed_effective_service_mean if s.observed_effective_service_mean is not None
            else s.expected_service_ticks)) for s in adjusted.stations))
        return predicted_progress(observation, super().predict_next(adjusted, assignments), assignments)


def build_dynamics(geometry: Geometry, network, device):
    return StructuredDynamics(geometry, network, device)


def build_baselines(geometry: Geometry):
    return {'progress_rule': ProgressAwareRule(geometry)}
