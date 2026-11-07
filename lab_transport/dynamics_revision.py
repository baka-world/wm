"""Validation-selected CPU revision of the hybrid state-transition model."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

from .clinic_map import make_clinic_scenario
from .dynamics_experiment import (BEHAVIOR_POLICIES, REGIMES, aggregate, audit_task_schedules, check_predicted_step, collect_episodes,
                                  compare, evaluate_episode, merge_counts, zero_counts)
from .dynamics_model import (L2, MOTION_FEATURES, SERVICE_FEATURES, Geometry, HybridDynamicsModel, _dot,
                             _motion_features, training_examples, validate_training_examples)
from .hospital_case import make_scenario
from .multisite_experiment import prepare_tartu

SPLITS_V2 = {"train": ("tartu", tuple(range(3000, 3012))),
             "validation": ("tartu", tuple(range(4000, 4004))),
             "test_tartu": ("tartu", tuple(range(9000, 9004))),
             "test_clinic": ("clinic", tuple(range(10000, 10004)))}
FEATURES_V2 = MOTION_FEATURES + ("current_goal_stay",)
CHECKPOINTS, LEARNING_RATE = (35, 90, 180), 1.0


def motion_features_v2(geometry, robot, phase, goal, candidate, others):
    return _motion_features(geometry, robot, phase, goal, candidate, others) + [
        float(robot.position == goal and candidate == robot.position)]


def validate_v2(parameters: dict[str, Any]) -> None:
    if not isinstance(parameters, dict) or parameters.get("schema_version") != 2 or parameters.get("kind") != "hybrid_linear_dynamics_v2":
        raise ValueError("invalid revised dynamics model schema")
    for feature_key, weight_key, features in (("motion_features", "motion_weights", FEATURES_V2),
                                              ("service_features", "service_weights", SERVICE_FEATURES)):
        weights = parameters.get(weight_key)
        if parameters.get(feature_key) != list(features) or not isinstance(weights, list) or len(weights) != len(features):
            raise ValueError("invalid revised dynamics feature schema")
        if any(not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x) for x in weights):
            raise ValueError("invalid revised dynamics weights")


class RevisedDynamicsModel(HybridDynamicsModel):
    def __init__(self, geometry: Geometry, parameters: dict[str, Any], mode: str = "learned"):
        if mode != "learned":
            raise ValueError("revised model is only used for learned predictions")
        validate_v2(parameters)
        self.geometry, self.parameters, self.mode = geometry, parameters, mode

    def motion_features(self, robot, phase, goal, candidate, others):
        return motion_features_v2(self.geometry, robot, phase, goal, candidate, others)


def train_checkpoints(moves, services) -> list[dict[str, Any]]:
    validate_training_examples(moves, services, len(FEATURES_V2))
    positive = sum(label for _, label in services)
    negative = len(services) - positive
    if not positive or not negative:
        raise ValueError("service training requires completed and ongoing examples")
    motion, service = [0.0] * len(FEATURES_V2), [0.0] * len(SERVICE_FEATURES)
    class_weight = {0: len(services) / (2.0 * negative), 1: len(services) / (2.0 * positive)}
    checkpoints = []
    for epoch in range(1, max(CHECKPOINTS) + 1):
        gradient = [0.0] * len(motion)
        for options, label in moves:
            scores = [_dot(motion, option) for option in options]
            highest = max(scores)
            terms = [math.exp(score - highest) for score in scores]
            total = sum(terms)
            for option, weight in zip(options, terms):
                for index, value in enumerate(option):
                    gradient[index] += weight / total * value
            for index, value in enumerate(options[label]):
                gradient[index] -= value
        for index in range(len(motion)):
            motion[index] -= LEARNING_RATE * (gradient[index] / len(moves) + L2 * motion[index])
        gradient = [0.0] * len(service)
        for features, label in services:
            score = max(-40.0, min(40.0, _dot(service, features)))
            error = (1.0 / (1.0 + math.exp(-score)) - label) * class_weight[label]
            for index, value in enumerate(features):
                gradient[index] += error * value
        for index in range(len(service)):
            service[index] -= LEARNING_RATE * (gradient[index] / len(services) + L2 * service[index])
        if epoch in CHECKPOINTS:
            checkpoints.append({"schema_version": 2, "kind": "hybrid_linear_dynamics_v2",
                "motion_features": list(FEATURES_V2), "service_features": list(SERVICE_FEATURES),
                "motion_weights": motion[:], "service_weights": service[:], "epoch": epoch,
                "learning_rate": LEARNING_RATE, "l2": L2, "training_motion_examples": len(moves),
                "training_service_examples": len(services), "training_service_completions": positive})
            print(f"Trained checkpoint {epoch}/{max(CHECKPOINTS)}", flush=True)
    return checkpoints


def validation_score(episodes, geometries, parameters) -> tuple[float, dict[str, Any]]:
    counts = zero_counts()
    for episode in episodes:
        if episode.split != "validation":
            continue
        geometry = geometries[episode.layout]
        model = RevisedDynamicsModel(geometry, parameters)
        for row in episode.rows:
            predicted = model.predict_next(row.observation, row.assignments)
            check_predicted_step(row.observation, predicted, geometry)
            merge_counts(counts, compare(row.observation, row.next_observation, predicted))
    if not counts["joint_total"]:
        raise ValueError("validation has no transitions")
    joint = counts["joint_correct"] / counts["joint_total"]
    denominator = counts["service_actual_finishes"] + counts["service_predicted_finishes"]
    service_f1 = 2 * counts["service_finish_exact_true_positive"] / denominator if denominator else 0.0
    score = 0.5 * (joint + service_f1)
    return score, {"epoch": parameters["epoch"], "selection_score": round(score, 8),
                   "joint_accuracy": round(joint, 8), "service_finish_f1": round(service_f1, 8),
                   "counts": counts}


def run_experiment(tartu, clinic) -> tuple[dict[str, Any], dict[str, Any]]:
    seed_sets = [set(seeds) for _, seeds in SPLITS_V2.values()]
    if any(a & b for index, a in enumerate(seed_sets) for b in seed_sets[index + 1:]):
        raise ValueError("scenario seed splits overlap")
    bases = {"tartu": prepare_tartu(tartu), "clinic": clinic}
    geometries = {layout: Geometry(base.grid) for layout, base in bases.items()}
    schedule_audit = audit_task_schedules(bases, SPLITS_V2)
    if schedule_audit["cross_split_duplicate_groups"] or schedule_audit["cross_split_task_duplicate_groups"]:
        raise ValueError("identical task or task/outage schedules occur across data splits")
    print(f"Schedule audit: {schedule_audit['unique_task_patterns']} unique task patterns, no cross-split duplicates", flush=True)
    episodes = collect_episodes(bases, SPLITS_V2)
    moves, services, excluded = [], [], 0
    for episode in episodes:
        if episode.split != "train":
            continue
        for row in episode.rows:
            m, s, skipped = training_examples(geometries[episode.layout], row.observation, row.assignments,
                                               row.next_observation, motion_features_v2)
            moves.extend(m)
            services.extend(s)
            excluded += skipped
    start = time.process_time()
    checkpoints = train_checkpoints(moves, services)
    train_seconds = time.process_time() - start
    scores = [validation_score(episodes, geometries, parameters) for parameters in checkpoints]
    best = max(range(len(checkpoints)), key=lambda i: (scores[i][0], -checkpoints[i]["epoch"]))
    parameters = checkpoints[best]
    parameters["selection"] = {"selected_epoch": parameters["epoch"], "validation_candidates": [detail for _, detail in scores],
                               "rule": "0.5 * validation known-entity joint accuracy + 0.5 * validation service-finish F1; earlier epoch breaks ties"}
    parameters["provenance"] = {"layout": "tartu", "train_seeds": list(SPLITS_V2["train"][1]),
                                "validation_seeds": list(SPLITS_V2["validation"][1]),
                                "regimes": list(REGIMES), "behavior_policies": list(BEHAVIOR_POLICIES),
                                "scope": "synthetic simulated transitions, not clinical measurements",
                                "excluded_motion_examples": excluded}
    print(f"Validation selected epoch {parameters['epoch']}; training examples: motion={len(moves)} service={len(services)}", flush=True)
    evaluations = [evaluate_episode(e, geometries[e.layout], parameters, RevisedDynamicsModel)
                   for e in episodes if e.split != "train"]
    results = {"description": "基于验证集选择的 CPU 混合状态预测；任务和扰动为合成仿真。",
               "training": {"episodes": sum(e.split == "train" for e in episodes),
                            "transitions": sum(len(e.rows) for e in episodes if e.split == "train"),
                            "timed_out_episodes": sum(e.status != "completed" for e in episodes if e.split == "train"),
                            "completed_tasks": sum(e.completed for e in episodes if e.split == "train"),
                            "motion_examples": len(moves), "service_examples": len(services),
                            "service_finishes": parameters["training_service_completions"],
                            "excluded_motion_examples": excluded, "cpu_seconds": train_seconds},
               "selection": parameters["selection"],
               "schedule_audit": schedule_audit,
               "splits": {name: {"layout": layout, "seeds": list(seeds), "regimes": list(REGIMES),
                                  "behavior_policies": list(BEHAVIOR_POLICIES)}
                          for name, (layout, seeds) in SPLITS_V2.items()},
               "aggregates": {split: aggregate(evaluations, split) for split in ("validation", "test_tartu", "test_clinic")},
               "cases": evaluations}
    return parameters, results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("hospital-map", "tiago-config", "clinic-map", "clinic-config", "output"):
        parser.add_argument("--" + argument, required=True)
    args = parser.parse_args()
    try:
        output = Path(args.output)
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
        tartu = make_scenario(Path(args.hospital_map).read_bytes(), seconds_per_tick=10,
                              tiago_config=Path(args.tiago_config).read_bytes())
        clinic = make_clinic_scenario(Path(args.clinic_map).read_bytes(), Path(args.clinic_config).read_bytes())
        parameters, results = run_experiment(tartu, clinic)
        output.mkdir(parents=True, exist_ok=True)
        (output / "dynamics-parameters.json").write_text(json.dumps(parameters, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (output / "dynamics-results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"training": results["training"], "selection": results["selection"],
                          "aggregates": results["aggregates"]}, ensure_ascii=False), flush=True)
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"状态转移复验失败：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
