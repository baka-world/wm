"""Limited held-out *synthetic* confirmation on frozen new task/map/seed cluster.

No fitting or threshold selection is permitted here. This experiment reports
both wins and losses; it never changes the scorer or the checkpoints.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

from business_transport.observation import BusinessObserver
from business_transport.policy import WorkflowPolicy
from business_transport.scenario import compile_scenario
from business_transport.simulation import BusinessSimulator
from experiments.business_constraints.run import make_model, summarize_result
from experiments.business_constraints.tests.test_business import fixture
from experiments.service_value.counterfactual import load_pr9_model
from experiments.world_model_integration.pilot import ARMS
from lab_transport.domain import Grid
from lab_transport.dynamics_model import Geometry

CONFIG = Path(__file__).with_name("confirmation_config.json")


def make_holdout(configuration):
    base, cfg, _ = fixture(1)
    spec = configuration["map"]
    blocked = frozenset(tuple(position) for position in spec["blocked"])
    grid = Grid(spec["width"], spec["height"], blocked)
    dx,dy = spec["station_shift"]
    rx,ry = spec["robot_shift"]
    new_stations = {
        sid: replace(station, position=(station.position[0]+dx,station.position[1]+dy),
                     jitter_ticks=configuration["station_service_jitter_ticks"])
        for sid,station in base.stations.items()
    }
    new_robots = tuple(
        replace(robot,
                position=(robot.position[0]+rx,robot.position[1]+ry),
                home=(robot.home[0]+rx,robot.home[1]+ry))
        for robot in base.robots)
    revised = replace(base, grid=grid, stations=new_stations, robots=new_robots)
    return compile_scenario(revised, {**cfg, "jobs": configuration["jobs"],
                                    "max_ticks": configuration["max_ticks"]})


def run(output: Path):
    if output.exists():
        raise FileExistsError("refusing to overwrite held-out confirmation")
    config_bytes = CONFIG.read_bytes()
    configuration = json.loads(config_bytes)
    if (configuration.get("status") != "predeclared-before-confirmation-evaluation" or
        configuration["arms"] != [arm for arm,_,_ in ARMS] or
        configuration["risk_weight"] != 0):
        raise ValueError("confirmation preregistration mismatch")
    torch.set_num_threads(1)
    business = make_holdout(configuration)
    geometry = Geometry(business.scenario.grid)
    network, info = load_pr9_model()
    cases = []
    for seed in configuration["execution_seeds"]:
        results = {}
        for arm,kind,aware in ARMS:
            observer = BusinessObserver()
            model = make_model(kind,geometry,network,info["threshold"],configuration["planner_seed"])
            policy = WorkflowPolicy("cpsat_" + kind,geometry,business.rules,observer,model,
                                    seed=configuration["planner_seed"],
                                    horizon=configuration["horizon"],
                                    particles=configuration["particles"],
                                    budget=configuration["candidate_budget"],
                                    solver_seconds=configuration["solver_seconds"],
                                    model_aware=aware,
                                    max_prediction_seconds=configuration["max_prediction_seconds"],
                                    risk_weight=configuration["risk_weight"])
            started=time.perf_counter()
            actual=BusinessSimulator(business,"world",seed,collect_metrics=True).run(
                dispatch_policy=policy,on_observation=observer.observe)
            metrics=summarize_result(actual,business,policy)
            if (metrics["resource"]["violations"] or metrics["discrete_safety_violations"] or
                metrics["lifecycle_violations"]):
                raise ValueError("held-out simulator produced unsafe result")
            results[arm]={
                "jobs_completed": metrics["jobs_completed"],
                "status": metrics["status"],
                "restricted_mean_flow": metrics["restricted_mean_flow"],
                "decision_wall_mean_ms": metrics["decision_wall_mean_ms"],
                "decision_wall_p95_ms": metrics["decision_wall_p95_ms"],
                "decision_count": metrics["decision_count"],
                "model_fallbacks": metrics["model_fallbacks"],
                "pets_forward_calls": metrics["pets_forward_calls"],
                "structured_service_calls": metrics["structured_service_calls"],
                "resource_violations": len(metrics["resource"]["violations"]),
                "motion_violations": metrics["discrete_safety_violations"],
                "cpu_wall_seconds": time.perf_counter()-started,
            }
        cases.append({"seed":seed,"arms":results,
                     "difference_C_minus_A": results["C_pets_model"]["restricted_mean_flow"]-
                                           results["A_rule_legacy"]["restricted_mean_flow"],
                     "difference_C_minus_B": results["C_pets_model"]["restricted_mean_flow"]-
                                           results["B_pets_legacy"]["restricted_mean_flow"]})
    outcome={
        "scope": "limited_independent_synthetic_holdout_not_real_world",
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "checkpoint_retrained":False,
        "map_obstacles": configuration["map"]["blocked"],
        "seed_count": len(cases),
        "cases": cases,
        "mean_difference_C_minus_A": statistics.mean(c["difference_C_minus_A"] for c in cases),
        "mean_difference_C_minus_B": statistics.mean(c["difference_C_minus_B"] for c in cases),
        "small_sample_no_claim_of_significance":True,
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(outcome,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    return outcome


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(run(args.output),ensure_ascii=False))


if __name__=="__main__":
    main()
