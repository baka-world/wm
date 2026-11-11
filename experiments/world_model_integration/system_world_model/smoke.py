"""CPU RHWM-v1 preview: one public state, two distinct lawful first actions.

Only synthetic business test fixtures are used; never a hidden simulator
future or the PR18 selected posthoc flip states.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from business_transport.observation import BusinessObserver
from business_transport.simulation import BusinessSimulator
from business_transport.world_model import HybridSystemWorldModel
from experiments.business_constraints.tests.test_business import fixture
from lab_transport.decision_value import candidates_for, nearest_batch
from lab_transport.dynamics_model import Geometry


def run(kind="rule", checkpoint=None, *, horizon=5, particles=3):
    _, _, business = fixture(capacity=1)
    geo = Geometry(business.scenario.grid)
    raw = []
    BusinessSimulator(business, "world", 11).run(on_transition=raw.append)
    public = BusinessObserver().observe(raw[0].observation)
    legal = candidates_for(public)
    if len(legal) < 2:
        raise AssertionError("synthetic smoke requires two different legal first actions")
    model = HybridSystemWorldModel(
        geo, business.rules, kind=kind,
        checkpoint=checkpoint, experimental=(kind == "v4"))
    out = []
    for first in ([legal[0]], [legal[-1]]):
        prediction = model.rollout(
            public, first,
            continuation_policy=lambda state: nearest_batch(
                state, candidates_for(state), geo),
            continuation_policy_name="nearest_public",
            horizon_ticks=horizon, particles=particles, sampling_seed=1810)
        out.append({
            "first_action": [{"robot_id":a.robot_id, "sample_id":a.sample_id} for a in first],
            "future_summary": [{
                "tick": s.tick,
                "expected_waiting_samples": s.expected_waiting_samples,
                "expected_completed_samples": s.expected_completed_samples,
                "expected_in_transit_samples": s.expected_in_transit_samples,
                "station_service_probability": dict(s.station_service_probability),
                "resource_occupancy_mean": dict(s.resource_occupancy_mean),
                "sample_status_probabilities": dict(s.sample_status_probabilities),
                "robot_task_probabilities": dict(s.robot_task_probabilities),
            } for s in prediction.distributions],
            "query_count": sum(len(step.service_events) for particle in prediction.particles
                               for step in particle.steps),
            "legal_transitions": horizon * particles,
        })
    return {
        "schema": "rhwm-v1-observable-full-state",
        "world_model": model.model_version,
        "initial_tick": public.tick,
        "public_sample_ids": [s.id for s in public.samples],
        "unknown_exogenous": list(prediction.unknown_exogenous),
        "particle_count": particles,
        "forecast_horizon": horizon,
        "candidate_forecasts": out,
        "ground_truth_future_used_for_model": False,
        "note": "Synthetic example; predicted states, NOT actual counterfactual outcomes",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=("rule", "v4"), default="rule")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--particles", type=int, default=3)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("RHWM smoke outputs must be write-once")
    output = run(args.kind, args.checkpoint, horizon=args.horizon,
                 particles=args.particles)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "world_model": output["world_model"],
        "candidates": len(output["candidate_forecasts"]),
        "horizon": output["forecast_horizon"],
        "queries": [c["query_count"] for c in output["candidate_forecasts"]],
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
