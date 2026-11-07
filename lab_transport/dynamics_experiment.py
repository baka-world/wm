"""Train and evaluate a small CPU hybrid state-transition model on fixed cases."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any

from .clinic_map import make_clinic_scenario
from .dynamics_model import Geometry, HybridDynamicsModel, train, training_examples
from .hospital_case import make_scenario
from .multisite_experiment import make_case, prepare_tartu
from .simulation import Simulator
from .transition_schema import Observation, Transition

SPLITS = {"train": ("tartu", tuple(range(3000, 3012))),
          "validation": ("tartu", tuple(range(4000, 4004))),
          "test_tartu": ("tartu", tuple(range(5000, 5004))),
          "test_clinic": ("clinic", tuple(range(6000, 6004)))}
REGIMES = ("normal", "short_outage")
BEHAVIOR_POLICIES = ("nearest", "world")
PREDICTORS = ("persistence", "heuristic", "learned")
HORIZONS = (3, 5)


@dataclass
class Episode:
    split: str
    layout: str
    seed: int
    regime: str
    policy: str
    status: str
    completed: int
    rows: list[Transition]


def audit_task_schedules(bases: dict[str, Any], splits=SPLITS) -> dict[str, Any]:
    """Detect identical nominal task/outage schedules across split names.

    Different seeds may still produce different hidden service-time draws;
    this audit intentionally checks the observable workload pattern itself.
    Policies share one schedule by design and are not counted twice.
    """
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    task_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    count = 0
    for split, (layout, seeds) in splits.items():
        for seed in seeds:
            for regime in REGIMES:
                scenario = make_case(bases[layout], layout, seed, regime)
                tasks = (layout, tuple((s.pickup, s.dropoff, s.release_tick) for s in scenario.samples))
                signature = (tasks,
                             tuple((sid, station.outages) for sid, station in sorted(scenario.stations.items())
                                   if station.outages))
                case = {"split": split, "seed": seed, "regime": regime}
                groups.setdefault(signature, []).append(case)
                task_groups.setdefault(tasks, []).append(case)
                count += 1
    overlaps = [cases for cases in groups.values() if len({case["split"] for case in cases}) > 1]
    task_overlaps = [cases for cases in task_groups.values() if len({case["split"] for case in cases}) > 1]
    return {"scenario_patterns": count, "unique_patterns": len(groups),
            "cross_split_duplicate_groups": len(overlaps), "cross_split_duplicates": overlaps,
            "unique_task_patterns": len(task_groups), "cross_split_task_duplicate_groups": len(task_overlaps),
            "cross_split_task_duplicates": task_overlaps}


def collect_episodes(bases: dict[str, Any], splits=SPLITS) -> list[Episode]:
    episodes = []
    for split, (layout, seeds) in splits.items():
        for seed in seeds:
            for regime in REGIMES:
                scenario = make_case(bases[layout], layout, seed, regime)
                for policy in BEHAVIOR_POLICIES:
                    rows: list[Transition] = []
                    result = Simulator(scenario, policy, seed).run(on_transition=rows.append)
                    episodes.append(Episode(split, layout, seed, regime, policy, result["summary"]["status"],
                                            result["summary"]["completed_count"],
                                            [r for r in rows if r.next_boundary == "before_dispatch"]))
        print(f"Collected {split}: {sum(e.split == split for e in episodes)} episodes", flush=True)
    return episodes


COUNT_KEYS = ("robot_total", "robot_position_correct", "robot_actual_moves", "robot_actual_moves_correct",
              "robot_predicted_moves", "robot_move_detected", "sample_total", "sample_status_correct",
              "sample_actual_changes", "sample_actual_changes_correct", "sample_predicted_changes",
              "sample_change_exact_true_positive", "service_actual_finishes", "service_predicted_finishes",
              "service_finish_exact_true_positive", "joint_total", "joint_correct", "new_arrivals", "station_changes")


def zero_counts() -> dict[str, int]:
    return {key: 0 for key in COUNT_KEYS}


def _service_advance(before: str, after: str) -> bool:
    return ((before in {"waiting", "assigned"} and after == "in_transit") or
            (before == "in_transit" and after == "completed"))


def compare(current: Observation, actual: Observation, predicted: Observation) -> dict[str, int]:
    counts = zero_counts()
    before_robots, true_robots, predicted_robots = ({r.id: r for r in state.robots}
                                                    for state in (current, actual, predicted))
    before_samples, true_samples, predicted_samples = ({s.id: s for s in state.samples}
                                                         for state in (current, actual, predicted))
    if set(before_robots) != set(true_robots) or set(before_robots) != set(predicted_robots):
        raise ValueError("robot set differs during prediction")
    if set(predicted_samples) != set(before_samples) or not set(before_samples).issubset(true_samples):
        raise ValueError("prediction must preserve exactly the currently known samples")
    joint = True
    for rid, old in before_robots.items():
        truth, estimate = true_robots[rid], predicted_robots[rid]
        correct = truth.position == estimate.position
        moved, predicted_move = truth.position != old.position, estimate.position != old.position
        counts["robot_total"] += 1
        counts["robot_position_correct"] += correct
        counts["robot_actual_moves"] += moved
        counts["robot_actual_moves_correct"] += moved and correct
        counts["robot_predicted_moves"] += predicted_move
        counts["robot_move_detected"] += moved and predicted_move
        joint = joint and correct
    for sid, old in before_samples.items():
        truth, estimate = true_samples[sid], predicted_samples[sid]
        correct = truth.status == estimate.status
        changed, predicted_change = truth.status != old.status, estimate.status != old.status
        actual_finish = _service_advance(old.status, truth.status)
        predicted_finish = _service_advance(old.status, estimate.status)
        counts["sample_total"] += 1
        counts["sample_status_correct"] += correct
        counts["sample_actual_changes"] += changed
        counts["sample_actual_changes_correct"] += changed and correct
        counts["sample_predicted_changes"] += predicted_change
        counts["sample_change_exact_true_positive"] += changed and predicted_change and correct
        counts["service_actual_finishes"] += actual_finish
        counts["service_predicted_finishes"] += predicted_finish
        counts["service_finish_exact_true_positive"] += actual_finish and predicted_finish and correct
        joint = joint and correct
    old_stations, true_stations = ({s.id: s for s in state.stations} for state in (current, actual))
    counts["station_changes"] = sum(old_stations[sid].available != true_stations[sid].available for sid in old_stations)
    counts["new_arrivals"] = len(set(true_samples) - set(before_samples))
    counts["joint_total"], counts["joint_correct"] = 1, int(joint)
    return counts


def check_predicted_step(current: Observation, predicted: Observation, geometry: Geometry) -> None:
    """Check the model's explicit discrete constraints, not physical safety."""
    before = {r.id: r.position for r in current.robots}
    after = {r.id: r.position for r in predicted.robots}
    if predicted.tick != current.tick + 1 or set(before) != set(after) or len(after) != len(set(after.values())):
        raise ValueError("predicted step violates the discrete robot state constraints")
    for rid, position in before.items():
        if after[rid] != position and after[rid] not in geometry.adjacency[position]:
            raise ValueError("predicted step is not a map edge")
    ids = sorted(before)
    if any(before[a] == after[b] and before[b] == after[a] for i, a in enumerate(ids) for b in ids[i + 1:]):
        raise ValueError("predicted robots swapped an edge")


def merge_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for key in COUNT_KEYS:
        target[key] += source[key]


def rates(counts: dict[str, int]) -> dict[str, float | None]:
    def ratio(numerator: str, denominator: str) -> float | None:
        return round(counts[numerator] / counts[denominator], 6) if counts[denominator] else None
    return {"robot_position_accuracy": ratio("robot_position_correct", "robot_total"),
            "moving_robot_exact_accuracy": ratio("robot_actual_moves_correct", "robot_actual_moves"),
            "robot_move_detection_recall": ratio("robot_move_detected", "robot_actual_moves"),
            "sample_status_accuracy": ratio("sample_status_correct", "sample_total"),
            "sample_change_exact_recall": ratio("sample_actual_changes_correct", "sample_actual_changes"),
            "sample_change_exact_precision": ratio("sample_change_exact_true_positive", "sample_predicted_changes"),
            "service_finish_exact_recall": ratio("service_finish_exact_true_positive", "service_actual_finishes"),
            "service_finish_exact_precision": ratio("service_finish_exact_true_positive", "service_predicted_finishes"),
            "known_entity_joint_accuracy": ratio("joint_correct", "joint_total")}


def eligible_windows(rows: list[Transition], horizon: int) -> tuple[list[list[Transition]], dict[str, int]]:
    """Use factual hindsight only to label conditional diagnostic windows."""
    counts = {"candidate_windows": 0, "future_assignments": 0, "new_arrivals": 0,
              "station_changes": 0, "eligible": 0}
    windows = []
    for i in range(max(0, len(rows) - horizon + 1)):
        window = rows[i:i + horizon]
        counts["candidate_windows"] += 1
        if any(row.assignments for row in window[1:]):
            counts["future_assignments"] += 1
            continue
        source = window[0].observation
        sample_ids = {s.id for s in source.samples}
        availability = {s.id: s.available for s in source.stations}
        if any({s.id for s in row.next_observation.samples} != sample_ids for row in window):
            counts["new_arrivals"] += 1
        elif any({s.id: s.available for s in row.next_observation.stations} != availability for row in window):
            counts["station_changes"] += 1
        else:
            counts["eligible"] += 1
            windows.append(window)
    return windows, counts


def evaluate_episode(episode: Episode, geometry: Geometry, parameters: dict[str, Any],
                     learned_factory=HybridDynamicsModel) -> dict[str, Any]:
    models = {"persistence": HybridDynamicsModel(geometry, mode="persistence"),
              "heuristic": HybridDynamicsModel(geometry, mode="heuristic"),
              "learned": learned_factory(geometry, parameters, "learned")}
    one_step = {name: zero_counts() for name in PREDICTORS}
    for row in episode.rows:
        for name, model in models.items():
            estimate = model.predict_next(row.observation, row.assignments)
            check_predicted_step(row.observation, estimate, geometry)
            merge_counts(one_step[name], compare(row.observation, row.next_observation, estimate))
    multi_step = {}
    for horizon in HORIZONS:
        windows, coverage = eligible_windows(episode.rows, horizon)
        totals = {name: zero_counts() for name in PREDICTORS}
        for window in windows:
            start, actual = window[0].observation, window[-1].next_observation
            for name, model in models.items():
                predicted = start
                for index, row in enumerate(window):
                    next_prediction = model.predict_next(predicted, row.assignments if index == 0 else ())
                    check_predicted_step(predicted, next_prediction, geometry)
                    predicted = next_prediction
                merge_counts(totals[name], compare(start, actual, predicted))
        multi_step[str(horizon)] = {"coverage": coverage, "counts": totals}
    return {"split": episode.split, "layout": episode.layout, "seed": episode.seed,
            "regime": episode.regime, "behavior_policy": episode.policy,
            "episode_status": episode.status, "completed": episode.completed,
            "one_step": one_step, "multi_step": multi_step}


def aggregate(evaluations: list[dict[str, Any]], split: str) -> dict[str, Any]:
    cases = [case for case in evaluations if case["split"] == split]
    one_step = {name: zero_counts() for name in PREDICTORS}
    per_episode_joint = {name: [] for name in PREDICTORS}
    multi = {str(h): {"coverage": {"candidate_windows": 0, "future_assignments": 0, "new_arrivals": 0,
                                    "station_changes": 0, "eligible": 0},
                       "counts": {name: zero_counts() for name in PREDICTORS}} for h in HORIZONS}
    for case in cases:
        for name in PREDICTORS:
            counts = case["one_step"][name]
            merge_counts(one_step[name], counts)
            if counts["joint_total"]:
                per_episode_joint[name].append(counts["joint_correct"] / counts["joint_total"])
        for horizon in HORIZONS:
            key = str(horizon)
            for field, count in case["multi_step"][key]["coverage"].items():
                multi[key]["coverage"][field] += count
            for name in PREDICTORS:
                merge_counts(multi[key]["counts"][name], case["multi_step"][key]["counts"][name])
    for key in multi:
        multi[key]["rates"] = {name: rates(multi[key]["counts"][name]) for name in PREDICTORS}
    return {"episodes": len(cases), "timed_out_episodes": sum(case["episode_status"] != "completed" for case in cases),
            "completed_tasks": sum(case["completed"] for case in cases),
            "one_step": {name: {"counts": one_step[name], "rates": rates(one_step[name]),
                         "mean_episode_joint_accuracy": round(sum(per_episode_joint[name]) / len(per_episode_joint[name]), 6)
                         if per_episode_joint[name] else None} for name in PREDICTORS}, "multi_step": multi}


def run_experiment(tartu: Any, clinic: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    bases = {"tartu": prepare_tartu(tartu), "clinic": clinic}
    geometries = {layout: Geometry(base.grid) for layout, base in bases.items()}
    seed_sets = [set(seeds) for _, seeds in SPLITS.values()]
    if any(a & b for index, a in enumerate(seed_sets) for b in seed_sets[index + 1:]):
        raise ValueError("scenario seed splits overlap")
    episodes = collect_episodes(bases)
    motions, services, excluded = [], [], 0
    for episode in episodes:
        if episode.split != "train":
            continue
        for row in episode.rows:
            moves, service, skipped = training_examples(geometries[episode.layout], row.observation,
                                                         row.assignments, row.next_observation)
            motions.extend(moves)
            services.extend(service)
            excluded += skipped
    start = time.process_time()
    parameters = train(motions, services)
    train_seconds = time.process_time() - start
    parameters["provenance"] = {"layout": "tartu", "seeds": list(SPLITS["train"][1]),
                                 "regimes": list(REGIMES), "behavior_policies": list(BEHAVIOR_POLICIES),
                                 "scope": "synthetic simulated transitions, not clinical measurements",
                                 "excluded_motion_examples": excluded}
    print(f"Trained: {len(motions)} motion and {len(services)} service examples; excluded motion={excluded}", flush=True)
    evaluations = []
    for episode in episodes:
        if episode.split == "train":
            continue
        evaluations.append(evaluate_episode(episode, geometries[episode.layout], parameters))
    results = {"description": "CPU 混合状态预测；公开地图上的任务与扰动是合成仿真，不是临床记录。",
               "training": {"episodes": sum(e.split == "train" for e in episodes),
                            "transitions": sum(len(e.rows) for e in episodes if e.split == "train"),
                            "timed_out_episodes": sum(e.status != "completed" for e in episodes if e.split == "train"),
                            "completed_tasks": sum(e.completed for e in episodes if e.split == "train"),
                            "motion_examples": len(motions), "service_examples": len(services),
                            "service_finishes": parameters["training_service_completions"],
                            "excluded_motion_examples": excluded, "cpu_seconds": train_seconds},
               "splits": {name: {"layout": layout, "seeds": list(seeds), "regimes": list(REGIMES),
                                  "behavior_policies": list(BEHAVIOR_POLICIES)}
                          for name, (layout, seeds) in SPLITS.items()},
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
        print(json.dumps({"training": results["training"], "aggregates": results["aggregates"]}, ensure_ascii=False), flush=True)
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"状态转移实验失败：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
