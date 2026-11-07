"""Small action-conditioned outcome model, trained on simulated executions.

Two ridge regressions learn residuals over the observable-state baseline for
pickup and completion times. No GPU or third-party package is required.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

from .domain import Sample
from .world_model import WorldModel

FEATURES = (
    "distance_to_pickup", "distance_to_dropoff", "pickup_service_est", "dropoff_service_est",
    "pickup_busy_est", "dropoff_busy_est", "other_on_route", "other_active", "waiting_count",
    "pickup_unavailable_now", "dropoff_unavailable_now",
)


def _vector(features: Mapping[str, Any]) -> list[float]:
    if not isinstance(features, Mapping) or set(features) != set(FEATURES):
        raise ValueError("learned model features do not match schema")
    values = []
    for name in FEATURES:
        value = features[name]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"invalid learned feature: {name}")
        values.append(float(value))
    return values


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Gaussian elimination with pivoting for the small regularized system."""
    a = [row[:] + [value] for row, value in zip(matrix, rhs)]
    n = len(rhs)
    for col in range(n):
        pivot = max(range(col, n), key=lambda i: abs(a[i][col]))
        if abs(a[pivot][col]) < 1e-12:
            raise ValueError("singular regression system")
        a[col], a[pivot] = a[pivot], a[col]
        scale = a[col][col]
        for j in range(col, n + 1):
            a[col][j] /= scale
        for row in range(n):
            if row == col:
                continue
            factor = a[row][col]
            for j in range(col, n + 1):
                a[row][j] -= factor * a[col][j]
    return [a[i][-1] for i in range(n)]


def train(records: list[dict[str, Any]], ridge_lambda: float = 4.0) -> dict[str, Any]:
    if len(records) < 2 or not math.isfinite(ridge_lambda) or ridge_lambda <= 0:
        raise ValueError("training requires at least two records and positive ridge penalty")
    raw = [_vector(record["features"]) for record in records]
    count, width = len(raw), len(FEATURES)
    means = [sum(row[i] for row in raw) / count for i in range(width)]
    scales = [math.sqrt(sum((row[i] - means[i]) ** 2 for row in raw) / count) for i in range(width)]
    scales = [s if s > 1e-9 else 1.0 for s in scales]
    vectors = [[1.0] + [(row[i] - means[i]) / scales[i] for i in range(width)] for row in raw]
    size = width + 1
    matrix = [[sum(x[i] * x[j] for x in vectors) + (ridge_lambda if i == j and i > 0 else 0.0)
               for j in range(size)] for i in range(size)]
    weights = {}
    for target in ("pickup_residual", "completion_residual"):
        outcomes = []
        for record in records:
            value = record.get(target)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"invalid {target}")
            outcomes.append(float(value))
        rhs = [sum(x[i] * y for x, y in zip(vectors, outcomes)) for i in range(size)]
        weights[target] = _solve(matrix, rhs)
    return {"schema_version": 1, "features": list(FEATURES), "means": means, "scales": scales,
            "pickup_weights": weights["pickup_residual"], "completion_weights": weights["completion_residual"],
            "ridge_lambda": ridge_lambda, "training_records": count,
            "training_scope": "synthetic simulated episodes; not clinical measurements"}


def validate(parameters: dict[str, Any]) -> None:
    if not isinstance(parameters, dict) or parameters.get("schema_version") != 1 or parameters.get("features") != list(FEATURES):
        raise ValueError("invalid learned model schema or feature order")
    for key, length in (("means", len(FEATURES)), ("scales", len(FEATURES)),
                        ("pickup_weights", len(FEATURES) + 1), ("completion_weights", len(FEATURES) + 1)):
        value = parameters.get(key)
        if not isinstance(value, list) or len(value) != length or any(
                not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x) for x in value):
            raise ValueError(f"invalid learned model {key}")
    if any(x <= 0 for x in parameters["scales"]):
        raise ValueError("learned model scales must be positive")


def corrected_forecast(baseline: dict[str, Any], tick: int, deadline: int | None,
                       parameters: dict[str, Any]) -> dict[str, Any]:
    validate(parameters)
    features = baseline["training_features"]
    raw = _vector(features)
    vector = [1.0] + [(x - mean) / scale for x, mean, scale in zip(raw, parameters["means"], parameters["scales"])]
    pickup_delta = sum(x * w for x, w in zip(vector, parameters["pickup_weights"]))
    completion_delta = sum(x * w for x, w in zip(vector, parameters["completion_weights"]))
    pickup = max(tick + features["distance_to_pickup"] + 1,
                 baseline["expected_pickup_tick"] + pickup_delta)
    completion = max(pickup + features["distance_to_dropoff"] + 1,
                     baseline["expected_completion_tick"] + completion_delta)
    late = None if deadline is None else max(0.0, completion - deadline)
    return {**baseline, "baseline_expected_pickup_tick": baseline["expected_pickup_tick"],
            "baseline_expected_completion_tick": baseline["expected_completion_tick"],
            "expected_pickup_tick": round(pickup, 3), "expected_completion_tick": round(completion, 3),
            "predicted_lateness": round(late, 3) if late is not None else None,
            "score": round(completion + 4 * (late or 0), 3), "predictor": "learned_ridge",
            "wait_estimate_note": "pickup/dropoff wait fields are the baseline estimates before learned residual correction"}


class LearnedWorldModel(WorldModel):
    """Predict candidate outcomes using trained residuals and live observations."""

    def __init__(self, grid, stations, parameters: dict[str, Any]):
        super().__init__(grid, stations)
        validate(parameters)
        self.parameters = parameters

    def predict(self, tick: int, robot_id: str, position, sample: Sample,
                station_busy: Mapping[str, int] | None = None, context: Mapping | None = None) -> dict:
        baseline = super().predict(tick, robot_id, position, sample, station_busy, context)
        return corrected_forecast(baseline, tick, sample.deadline_tick, self.parameters)
