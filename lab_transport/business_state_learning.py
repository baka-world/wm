"""Separate business-observable v1 Deep Sets service-event model.

This never changes or loads PR #8/#9/#11 weights. One model predicts one
stochastic next-step service-completion event; business/workflow rules still
own the legal state transition. An unvalidated model cannot be used online.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch import nn

from business_transport.state_features import SCHEMA_VERSION, LABEL_VERSION

TOKEN_WIDTH = 40
MODEL_VERSION = "business-sets-service-event-v1"


def _field(row, name, scale=1.):
    item = row.get(name, {"known": False, "value": None})
    if not isinstance(item, dict) or "known" not in item or "value" not in item:
        raise ValueError("missing explicit known/value mask")
    if not item["known"]:
        if item["value"] is not None:
            raise ValueError("unknown feature contains a fabricated value")
        return 0., 0.
    value = item["value"]
    if value is None or not isinstance(value, (int, float, bool)) or not math.isfinite(value):
        raise ValueError("known scalar must be finite")
    return max(-4., min(4., float(value) / scale)), 1.


def _put(row, typ, fields, extra=()):
    vector = [0.] * TOKEN_WIDTH
    vector[typ] = 1.
    pos = 4
    for field, scale in fields:
        value, mask = _field(row, field, scale)
        vector[pos], vector[pos+1] = value, mask
        pos += 2
    for value in extra:
        if pos >= TOKEN_WIDTH:
            raise ValueError("business entity vector too long")
        vector[pos] = float(value)
        pos += 1
    return vector


def encode_tokens(payload):
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("label_version") != LABEL_VERSION:
        raise ValueError("business feature/label schema mismatch")
    observed = payload["observed_facts"]; tokens = []
    for robot in observed["robots"]:
        features = [("service_elapsed", 30), ("service_active_elapsed", 30),
                    ("service_paused_elapsed", 30), ("operational_available", 1)]
        position = robot["position"]
        extra = [float(robot["phase"] == phase) for phase in ("idle", "to_pickup", "to_dropoff")]
        extra += [position[0] / 30., position[1] / 30., float(bool(robot["candidate_distances"]))]
        tokens.append(_put(robot, 0, features, extra))
    for sample in observed["samples"]:
        features = [("deadline_tick", 60), ("priority", 10), ("public_stages_total", 5)]
        extra = [sample["stage_index"] / 5., sample["release_tick"] / 60.,
                 len(sample["completed_stage_indices"]) / 5.,
                 len(sample["public_remaining_route"]) / 5.]
        extra += [float(sample["status"] == st) for st in ("waiting", "assigned", "in_transit", "completed")]
        tokens.append(_put(sample, 1, features, extra))
    for station in observed["stations"]:
        fields = [("device_available", 1), ("service_admission_available", 1),
                  ("resource_capacity", 10), ("resource_occupied", 10),
                  ("capacity_blocked", 1), ("service_session_active", 1),
                  ("service_in_progress", 1), ("waiting_pickups", 10),
                  ("approaching_pickups", 10), ("approaching_dropoffs", 10),
                  ("observed_effective_service_mean", 10),
                  ("observed_effective_service_count", 20),
                  ("history_complete_since_start", 1), ("expected_service_ticks_public", 10)]
        tokens.append(_put(station, 2, fields))
    for action in payload["candidate_action"]:
        fields = [("previous_task", 1)]
        # Previous task is a categorical ID, not a numeric scalar.
        fields = []
        extra = [action["robot_to_pickup_distance"] / 30.,
                 action["pickup_to_dropoff_distance"] / 30.,
                 float(action["replaces_running_task"]),
                 float(action["competes_with_running_service"])]
        tokens.append(_put(action, 3, fields, extra))
    if not tokens:
        raise ValueError("empty business observation")
    if any(len(token) != TOKEN_WIDTH for token in tokens):
        raise ValueError("invalid business token width")
    return torch.tensor(tokens, dtype=torch.float32)


class BusinessSetsServiceEvent(nn.Module):
    """Masked mean pooling; versioned separately from old Structured weights."""
    def __init__(self):
        super().__init__()
        self.entity = nn.Sequential(nn.Linear(TOKEN_WIDTH, 32), nn.ReLU(),
                                    nn.Linear(32, 32), nn.ReLU())
        self.service_event = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, tokens, mask):
        if tokens.ndim != 3 or tokens.shape[-1] != TOKEN_WIDTH:
            raise ValueError("wrong business v1 feature shape")
        pooled = (self.entity(tokens) * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return self.service_event(pooled).squeeze(-1)


def encode_batch(rows):
    values = [encode_tokens(row) for row in rows]
    n = max(item.shape[0] for item in values)
    x = torch.zeros((len(rows), n, TOKEN_WIDTH), dtype=torch.float32)
    mask = torch.zeros((len(rows), n), dtype=torch.float32)
    for i, value in enumerate(values):
        x[i, :len(value)] = value; mask[i, :len(value)] = 1.
    return x, mask


def train_cpu(dataset_path, checkpoint_path, *, seed=1810, epochs=5, seconds=60.):
    """Train only explicitly sourced *factual* next-step events on CPU."""
    dataset_path = Path(dataset_path); checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists():
        raise FileExistsError("refuse checkpoint overwrite")
    if seconds <= 0 or not 1 <= epochs <= 50:
        raise ValueError("invalid predeclared CPU budget or epoch count")
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    if (dataset.get("schema_version") != SCHEMA_VERSION or
        dataset.get("label_version") != LABEL_VERSION or
        dataset.get("source_type") != "factual_observed_transition"):
        raise ValueError("unsupported provenance or training schema")
    splits = {}
    identifiers = set()
    for name in ("train", "validation", "test"):
        rows = dataset.get(name)
        if not isinstance(rows, list) or not rows:
            raise ValueError("missing disjoint factual split: " + name)
        for row in rows:
            if row["id"] in identifiers:
                raise ValueError("duplicate factual transition across splits")
            identifiers.add(row["id"])
            if row.get("label") not in (0, 1):
                raise ValueError("binary next-step service event label required")
            encode_tokens(row["features"])
        splits[name] = rows
    torch.set_num_threads(1); torch.manual_seed(seed)
    model = BusinessSetsServiceEvent().cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    started = time.monotonic(); losses = []
    for epoch in range(epochs):
        if time.monotonic() - started > seconds:
            raise TimeoutError("CPU training time budget exceeded; checkpoint not saved")
        model.train()
        optimizer.zero_grad()
        x, mask = encode_batch([r["features"] for r in splits["train"]])
        labels = torch.tensor([r["label"] for r in splits["train"]], dtype=torch.float32)
        logits = model(x, mask)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite service loss")
        loss.backward(); optimizer.step()
        losses.append(float(loss.item()))
    def metrics(rows):
        model.eval()
        with torch.inference_mode():
            x, mask = encode_batch([r["features"] for r in rows])
            logits = model(x, mask)
            labels = torch.tensor([r["label"] for r in rows], dtype=torch.float32)
            return {"bce": float(nn.functional.binary_cross_entropy_with_logits(logits, labels)),
                    "accuracy_at_0_5": float(((logits >= 0) == (labels >= .5)).float().mean())}
    metadata = {"model_version": MODEL_VERSION, "schema_version": SCHEMA_VERSION,
                "label_version": LABEL_VERSION, "source_type": dataset["source_type"],
                "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
                "seed": seed, "epochs": epochs, "seconds_budget": seconds,
                "train_loss": losses, "validation": metrics(splits["validation"]),
                "test_frozen": metrics(splits["test"]),
                "split_sizes": {k: len(v) for k, v in splits.items()},
                "online_verified": False,
                "note": "training alone does not validate rollout or decision quality"}
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": metadata, "model_state": model.state_dict()}, checkpoint_path)
    return metadata


def load_online(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    metadata = saved["metadata"]
    if (metadata.get("model_version") != MODEL_VERSION or
        metadata.get("schema_version") != SCHEMA_VERSION or
        metadata.get("label_version") != LABEL_VERSION or
        metadata.get("online_verified") is not True):
        raise ValueError("business v1 model is not independently verified for online service")
    model = BusinessSetsServiceEvent()
    model.load_state_dict(saved["model_state"])
    return model.eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1810)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seconds", type=float, default=60)
    args = parser.parse_args()
    print(json.dumps(train_cpu(args.dataset, args.checkpoint, seed=args.seed,
                               epochs=args.epochs, seconds=args.seconds), indent=2))


if __name__ == "__main__":
    main()
