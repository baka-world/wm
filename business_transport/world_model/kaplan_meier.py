"""Adapter for the published Kaplan-Meier estimator; not a new ML algorithm.

Kaplan & Meier (1958), DOI 10.1080/01621459.1958.10501452.
Implementation from MIT-licensed lifelines 0.30.0, KaplanMeierFitter.
For an active service at effective age t the discrete completion hazard is
1 - S(t+1)/S(t). Only current, publicly observed progress enters inference.

The bridge that applies the sampled event to ProgressAwareRule follows the
existing SessionRuleDynamics contract. No model weights / private service
remaining time are read from the live simulator during prediction.
"""
from __future__ import annotations

from dataclasses import replace
import math
import random

from business_transport.session_events import eligible_sessions
from lab_transport.state_v2 import ObservationV2
from lab_transport.torch_structured_adapter import ProgressAwareRule


class KaplanMeierServiceModel:
    """Thin wrapper around lifelines' actual, upstream KaplanMeierFitter.

    This is a per-station nonparametric baseline. No graph, action-dependent
    delay predictor, or model claim beyond the published KM survival estimate.
    """

    def __init__(self, sessions):
        from lifelines import KaplanMeierFitter

        per_station = {}
        for row in sessions:
            station_id = row["station_id"]
            duration = row["duration"]
            observed = row["event_observed"]
            if not isinstance(station_id, str) or not station_id:
                raise ValueError("service training row is missing station ID")
            if type(duration) is not int or duration < 1 or type(observed) is not bool:
                raise ValueError("invalid duration or censoring flag")
            per_station.setdefault(station_id, []).append((duration, observed))
        if not per_station:
            raise ValueError("Kaplan-Meier requires factual, grouped service observations")
        self.models = {}
        self.group_counts = {}
        for station, rows in sorted(per_station.items()):
            kmf = KaplanMeierFitter()
            kmf.fit([duration for duration, _ in rows],
                    event_observed=[int(observed) for _, observed in rows],
                    label=station)
            self.models[station] = kmf
            self.group_counts[station] = len(rows)

    def p_finish_next(self, station_id: str, active_elapsed: int, fallback_mean: float) -> float:
        if type(active_elapsed) is not int or active_elapsed < 0:
            raise ValueError("effective elapsed service duration must be observed")
        if not math.isfinite(fallback_mean) or fallback_mean <= 0:
            raise ValueError("fallback mean must be positive and finite")
        model = self.models.get(station_id)
        if model is None:
            # No facts exist for station: explicitly revert to the fixed
            # public-duration rule rather than invent a learned probability.
            return float(active_elapsed + 1 >= math.ceil(fallback_mean))
        # Use upstream lifelines' stepwise survival curve and standard
        # conditional probability identity, never self-implemented KM fit.
        surv = model.survival_function_at_times([active_elapsed, active_elapsed + 1])
        before, after = (float(v) for v in surv.to_numpy())
        if before <= 0:
            # Unseen, beyond fitted support. Conservative deterministic public
            # rule fallback, reported via counts on the adapter.
            return float(active_elapsed + 1 >= math.ceil(fallback_mean))
        return max(0.0, min(1.0, 1.0 - after / before))


class KaplanMeierRuleDynamics:
    """Map a source-library probability to the existing legal state transition."""

    def __init__(self, geometry, rules, service_model: KaplanMeierServiceModel, *, seed=1810):
        if not isinstance(service_model, KaplanMeierServiceModel):
            raise TypeError("service_model must be fitted upstream lifelines Kaplan-Meier")
        self.geometry, self.rules = geometry, rules
        self.rule = ProgressAwareRule(geometry)
        self.service_model = service_model
        self.rng = random.Random(seed)
        self.last_service_events = ()
        self.query_calls = 0

    def reset_seed(self, seed):
        self.rng.seed(seed)

    def predict_next(self, observation, assignments):
        if not isinstance(observation, ObservationV2):
            raise ValueError("requires public, history-derived ObservationV2")
        assignments = tuple(assignments)
        self.last_service_events = ()
        viable = [q for q, reason in eligible_sessions(observation, assignments)
                  if reason is None]
        by_robot = {r.id: r for r in observation.robots}
        by_station = {s.id: s for s in observation.stations}
        overrides, events = {}, []
        for q in viable:
            robot = by_robot[q["robot_id"]]
            station = by_station[q["station_id"]]
            age = robot.service_active_elapsed
            if age is None:
                raise ValueError("missing observed effective service progress")
            fallback = (station.observed_effective_service_mean
                        if station.observed_effective_service_mean is not None
                        else station.expected_service_ticks)
            p = self.service_model.p_finish_next(q["station_id"], age, fallback)
            outcome = self.rng.random() < p
            events.append({
                "query": dict(q), "p_finish": float(p),
                "sampled_finished": bool(outcome),
            })
            # Reuse the existing legal service transition rule; modify only
            # temporary public service duration for this step.
            overrides[q["station_id"]] = age + (1 if outcome else 2)
        self.query_calls += len(events)
        self.last_service_events = tuple(events)
        if overrides:
            adjusted = replace(observation, stations=tuple(
                replace(station,
                        expected_service_ticks=overrides[station.id],
                        observed_effective_service_mean=float(overrides[station.id]))
                if station.id in overrides else station
                for station in observation.stations))
        else:
            adjusted = observation
        predicted = self.rule.predict_next(adjusted, assignments)
        previous = {s.id: s for s in observation.stations}
        return replace(predicted, stations=tuple(
            replace(s,
                    expected_service_ticks=previous[s.id].expected_service_ticks,
                    observed_effective_service_mean=previous[s.id].observed_effective_service_mean)
            for s in predicted.stations))
