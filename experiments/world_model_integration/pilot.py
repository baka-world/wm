"""Small, fixed CPU integration pilot. Not an independent confirmation set.

Executes four online business workflows with frozen checkpoints, and
evaluates identical factual states under 5/10/20 step model rollouts.
No training, no GPU, no hindsight oracle, and no tuning from these results.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import time

import torch

from business_transport.observation import BusinessObserver
from business_transport.policy import WorkflowPolicy
from business_transport.simulation import BusinessSimulator
from business_transport.dispatch import BusinessCPSAT
from experiments.business_constraints.run import (make_model, summarize_result,
    replay_branch, rich_rows)
from experiments.business_constraints.tests.test_business import fixture
from experiments.world_model_integration.prediction_metrics import evaluate_predictions
from experiments.service_value.counterfactual import load_pr9_model
from lab_transport.decision_value import candidates_for
from lab_transport.dynamics_model import Geometry
from lab_transport.decision_value_round2 import quantile


ARMS = (
    ("A_rule_legacy", "rule", False),
    ("B_pets_legacy", "pets", False),
    ("C_pets_model", "pets", True),
    ("D_structured_model", "structured", True),
)
WINDOWS = (5, 10, 20)
SEED = 1810
PARTICLES = 3
CANDIDATES = 4


def run(output: Path):
    if output.exists():
        raise FileExistsError("pilot refuses to overwrite outputs")
    torch.set_num_threads(1)
    base, cfg, business = fixture(1)
    geometry = Geometry(base.grid)
    network, info = load_pr9_model()
    arm_results = {}
    factual = factual_rows = None
    for arm, model_type, model_aware in ARMS:
        model = make_model(model_type, geometry, network, info["threshold"], SEED)
        observer = BusinessObserver()
        policy = WorkflowPolicy("cpsat_" + model_type, geometry, business.rules,
                                observer, model, seed=SEED, horizon=5,
                                particles=PARTICLES, budget=CANDIDATES,
                                model_aware=model_aware,
                                max_prediction_seconds=120.)
        rows = []
        started = time.perf_counter()
        sim = BusinessSimulator(business, "world", 1, collect_metrics=True)
        result = sim.run(on_transition=rows.append,
                         dispatch_policy=policy, on_observation=observer.observe)
        metrics = summarize_result(result, business, policy)
        arm_results[arm] = {
            "jobs_completed": metrics["jobs_completed"],
            "status": metrics["status"],
            "restricted_mean_flow": metrics["restricted_mean_flow"],
            "resource_capacity_violations": len(metrics["resource"]["violations"]),
            "discrete_safety_violations": metrics["discrete_safety_violations"],
            "lifecycle_violations": len(metrics["lifecycle_violations"]),
            "model_fallbacks": metrics["model_fallbacks"],
            "decision_wall_mean_ms": metrics["decision_wall_mean_ms"],
            "decision_wall_p95_ms": metrics["decision_wall_p95_ms"],
            "decision_count": metrics["decision_count"],
            "pets_forward_calls": metrics["pets_forward_calls"],
            "structured_service_calls": metrics["structured_service_calls"],
            "elapsed_cpu_pilot_wall_seconds": time.perf_counter()-started,
            "sample_decision_logs": metrics["decisions"][:2],
        }
        if arm == "A_rule_legacy":
            factual, factual_rows = result, rows
    rich = rich_rows(factual_rows)
    # Equal candidate sets / factual prefix. Do not feed the factual future to
    # prediction. Replay is used only afterwards as an outcome measurement.
    sampled = []
    found = [(i, row) for i, row in enumerate(rich)
             if len(candidates_for(row.observation)) >= 2]
    if found:
        index, row = found[len(found)//2]
        observation = row.observation
        plans = BusinessCPSAT(geometry, business.rules, candidate_limit=CANDIDATES,
                              time_limit_seconds=.5, seed=SEED).proposals(
                                  observation, candidates_for(observation))
        branches = [replay_branch(business, 1, factual, factual_rows, observation.tick, plan)
                    for plan in plans]
        scored = {}
        for horizon in WINDOWS:
            scored[str(horizon)] = {}
            for arm, model_type, aware in ARMS:
                policy = WorkflowPolicy("cpsat_" + model_type, geometry, business.rules,
                                        BusinessObserver(), make_model(model_type, geometry,
                                        network, info["threshold"], SEED),
                                        seed=SEED, horizon=horizon, particles=PARTICLES,
                                        budget=CANDIDATES, model_aware=aware,
                                        max_prediction_seconds=120.)
                start = time.perf_counter()
                records = policy.evaluate(observation, plans)
                choice = min(range(len(records)), key=lambda i:(records[i]["score"], i))
                scored[str(horizon)][arm] = {
                    "choice": choice,
                    "scores": [r["score"] for r in records],
                    "model_service_work": [r.get("component_means",{}).get("model_service") for r in records],
                    "rule_prior_service_work": [r.get("component_means",{}).get("rule_prior_service") for r in records],
                    "candidate_deltas": [r.get("component_means",{}).get("model_minus_prior") for r in records],
                    "prediction_calls": policy.prediction_calls,
                    "prediction_seconds": policy.prediction_wall_seconds,
                    "wall_seconds": time.perf_counter()-start,
                    "fallbacks": policy.fallbacks,
                    "particle_spreads": [r.get("cost_distribution") for r in records],
                }
        sampled.append({"tick": observation.tick,
                        "candidates": [[asdict(a) for a in plan] for plan in plans],
                        "outcome_only_counterfactual_branches": branches,
                        "horizons": scored})
    prediction_metrics=evaluate_predictions(rich,business,geometry,network,info["threshold"],seed=SEED)
    result = {
        "prediction_accuracy_development":prediction_metrics,
        "status": "development_cpu_pilot_not_independent_confirmation",
        "seed": SEED, "simulation_seed": 1, "layout": "synthetic_fixture_7x5",
        "particles": PARTICLES, "candidate_budget": CANDIDATES,
        "horizons": list(WINDOWS), "risk_weight": 0.,
        "legacy_checkpoint_files_unmodified": True,
        "new_business_v1_state_model": "UNTRAINED_NOT_AVAILABLE",
        "future_equipment_failure_risk": "UNOBSERVED",
        "arms": arm_results, "same_state_diagnostics": sampled,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({"arm_metrics": {k: {f: v for f, v in row.items() if
           f != "sample_decision_logs"} for k,row in result["arms"].items()},
           "same_state": result["same_state_diagnostics"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
