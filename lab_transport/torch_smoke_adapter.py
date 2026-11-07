"""Replaceable, deliberately small PyTorch dynamics adapter for plumbing tests.

This adapter is not the thesis' final model. It learns one-step movement and
station-service outcomes for currently known entities. It does not predict new
sample arrivals or future changes to station availability.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .dynamics_model import (Geometry, HybridDynamicsModel, SERVICE_FEATURES, _intent,
                             _motion_options, _service_features, _target)
from .dynamics_revision import FEATURES_V2, motion_features_v2

ADAPTER_VERSION = 1


class SmokeNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.motion = nn.Sequential(nn.Linear(len(FEATURES_V2), 16), nn.ReLU(), nn.Linear(16, 1))
        self.service = nn.Sequential(nn.Linear(len(SERVICE_FEATURES), 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        scores = self.motion(inputs["motion"]).squeeze(-1)
        scores = scores.masked_fill(~inputs["motion_mask"], float("-inf"))
        return {"motion": scores, "service": self.service(inputs["service"]).squeeze(-1)}


def build_model() -> SmokeNetwork:
    return SmokeNetwork()


def encode_transition(geometry, observation, assignments, actual) -> list[dict[str, Any]]:
    """Construct inputs from current information, then attach factual labels.

    Options are formed independently for each robot from *current* occupancy;
    future robot movements are used only as labels, never as option masks.
    Runtime rollout handles inter-robot reservations using its own predictions.
    """
    intents = _intent(observation, assignments)
    samples = {s.id: s for s in observation.samples}
    stations = {s.id: s for s in observation.stations}
    station_positions = {s.id: s.position for s in observation.stations}
    next_robots = {r.id: r for r in actual.robots}
    next_samples = {s.id: s for s in actual.samples}
    assigned = {a.sample_id for a in assignments}
    occupied = {r.position for r in observation.robots}
    items = []
    for robot in observation.robots:
        phase, sample_id = intents[robot.id]
        goal = _target(robot, (phase, sample_id), samples, station_positions)
        options = _motion_options(geometry, robot, phase, goal, occupied, set())
        others = tuple(r.position for r in observation.robots if r.id != robot.id)
        features = [motion_features_v2(geometry, robot, phase, goal, p, others) for p in options]
        actual_position = next_robots[robot.id].position
        if len(options) > 1 and actual_position in options:
            items.append({"kind": "motion", "input": features, "target": options.index(actual_position)})
        if sample_id is not None and robot.position == goal:
            sample = samples[sample_id]
            station = stations[sample.pickup if phase == "to_pickup" else sample.dropoff]
            if station.available and sample_id in next_samples:
                features = _service_features(robot, phase, station.expected_service_ticks, sample_id in assigned)
                status = "in_transit" if phase == "to_pickup" else "completed"
                items.append({"kind": "service", "input": features,
                              "target": int(next_samples[sample_id].status == status)})
    return items


def collate(items: list[dict[str, Any]]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    moves = [item for item in items if item["kind"] == "motion"]
    services = [item for item in items if item["kind"] == "service"]
    width = max((len(item["input"]) for item in moves), default=1)
    motion = torch.zeros((len(moves), width, len(FEATURES_V2)), dtype=torch.float32)
    mask = torch.zeros((len(moves), width), dtype=torch.bool)
    for index, item in enumerate(moves):
        count = len(item["input"])
        motion[index, :count] = torch.tensor(item["input"], dtype=torch.float32)
        mask[index, :count] = True
    service = torch.tensor([item["input"] for item in services], dtype=torch.float32).reshape(-1, len(SERVICE_FEATURES))
    inputs = {"motion": motion, "motion_mask": mask, "service": service}
    labels = {"motion": torch.tensor([item["target"] for item in moves], dtype=torch.long),
              "service": torch.tensor([item["target"] for item in services], dtype=torch.float32)}
    return inputs, labels


def loss(outputs: dict[str, torch.Tensor], labels: dict[str, torch.Tensor]) -> tuple[torch.Tensor, int]:
    count = labels["motion"].numel() + labels["service"].numel()
    if not count:
        raise ValueError("empty batch")
    total = torch.zeros((), device=outputs["motion"].device)
    if labels["motion"].numel():
        total = total + F.cross_entropy(outputs["motion"], labels["motion"], reduction="sum")
    if labels["service"].numel():
        total = total + F.binary_cross_entropy_with_logits(outputs["service"], labels["service"], reduction="sum")
    return total / count, count


class TorchDynamicsModel(HybridDynamicsModel):
    """One-step adapter; static rules still enforce discrete movement constraints."""
    def __init__(self, geometry: Geometry, network: SmokeNetwork, device: torch.device):
        self.geometry, self.parameters, self.mode = geometry, None, "learned"
        self.network, self.device = network.to(device).eval(), device

    def motion_features(self, robot, phase, goal, candidate, others):
        return motion_features_v2(self.geometry, robot, phase, goal, candidate, others)

    def score_motion(self, features):
        with torch.inference_mode():
            values = torch.tensor(features, dtype=torch.float32, device=self.device)
            return self.network.motion(values).squeeze(-1).cpu().tolist()

    def service_advances(self, features):
        with torch.inference_mode():
            values = torch.tensor(features, dtype=torch.float32, device=self.device)
            score = self.network.service(values).item()
            if not math.isfinite(score):
                raise ValueError("model returned invalid service score")
            return bool(score >= 0)


def build_dynamics(geometry: Geometry, network: SmokeNetwork, device: torch.device) -> TorchDynamicsModel:
    return TorchDynamicsModel(geometry, network, device)
