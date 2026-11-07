"""Reproducible CPU experiment for a learned, action-conditioned predictor.

All training labels and disturbances are simulated; this is not clinical data.
See docs/world-model-experiment.md for the fixed split and limitations.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from .domain import Sample, Scenario
from .hospital_case import SOURCE_COMMIT, make_scenario
from .learned_model import corrected_forecast, train
from .simulation import Simulator

TRAIN_SEEDS = tuple(range(40))
TEST_SEEDS = tuple(range(100, 110))
SHIFT_SEEDS = tuple(range(200, 210))
POLICIES = ("nearest", "world", "learned")


def make_case(base: Scenario, seed: int, regime: str) -> Scenario:
    if regime not in {"normal", "short_outage", "shifted"}:
        raise ValueError("unknown experiment regime")
    rng = random.Random(1729 + seed)
    release = 0
    samples = []
    for i in range(6):
        if i:
            release += rng.choice((4, 8, 12, 16))
        samples.append(Sample(f"S{i + 1:02}", "ICU取样点", "实验室交付点", release, None))
    stations = dict(base.stations)
    if regime != "normal":
        target = rng.choice(("ICU取样点", "实验室交付点"))
        start = rng.randint(10, 55)
        duration = rng.randint(35, 50) if regime == "shifted" else rng.randint(6, 14)
        stations[target] = replace(stations[target], outages=((start, start + duration),))
    if regime == "shifted":
        stations["ICU取样点"] = replace(stations["ICU取样点"], jitter_ticks=6)
        stations["实验室交付点"] = replace(stations["实验室交付点"], jitter_ticks=3)
    return replace(base, name=f"合成任务测试 {regime} / {seed}", stations=stations,
                   samples=tuple(samples), max_ticks=350)


def training_cases() -> list[tuple[str, int]]:
    return [("normal" if seed % 2 == 0 else "short_outage", seed) for seed in TRAIN_SEEDS]


def testing_cases() -> list[tuple[str, int]]:
    return [(regime, seed) for regime in ("normal", "short_outage") for seed in TEST_SEEDS] + [
        ("shifted", seed) for seed in SHIFT_SEEDS]


def observations(result: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    """Collect executed decisions only, excluding unfinished outcomes."""
    samples = {s["id"]: s for s in result["samples"]}
    records = []
    for event in result["events"]:
        if event["type"] != "assigned":
            continue
        sample = samples[event["sample_id"]]
        chosen = event["selected"]
        if sample["picked_tick"] is None or sample["completed_tick"] is None:
            continue
        if "training_features" not in chosen:
            raise ValueError("training data requires baseline prediction features")
        tick = event["tick"]
        records.append({"features": chosen["training_features"], "tick": tick,
                        "baseline": chosen,
                        "actual_pickup": sample["picked_tick"], "actual_completion": sample["completed_tick"],
                        "pickup_residual": sample["picked_tick"] - chosen["expected_pickup_tick"],
                        "completion_residual": sample["completed_tick"] - chosen["expected_completion_tick"]})
    return records, len(result["samples"]) - len(records)


def outcome(result: dict[str, Any], max_ticks: int, cpu_seconds: float) -> dict[str, Any]:
    samples = result["samples"]
    completed = [s for s in samples if s["completed_tick"] is not None]
    # Time to termination for unfinished jobs is a right-censored lower bound.
    restricted = sum((s["completed_tick"] if s["completed_tick"] is not None else max_ticks)
                     - s["release_tick"] for s in samples) / len(samples)
    # An injected callback can examine the candidate batch an unknown number
    # of times; counting each event's displayed batch would misreport work.
    forecasts = (None if result["policy"] == "external" else
                 sum(len(e["candidates"]) for e in result["events"] if e["type"] == "assigned"))
    return {"completed": len(completed), "total": len(samples),
            "timed_out": result["summary"]["status"] != "completed",
            "restricted_mean_flow_ticks": restricted,
            "mean_completed_flow_ticks": (sum(s["completed_tick"] - s["release_tick"] for s in completed)
                                           / len(completed) if completed else None),
            "candidate_evaluations": forecasts, "cpu_seconds": cpu_seconds,
            "assignments": [[e["tick"], e["robot_id"], e["sample_id"]]
                            for e in result["events"] if e["type"] == "assigned"]}


def aggregate(rows: list[dict[str, Any]], offline: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for regime in ("normal", "short_outage", "shifted"):
        cases = [r for r in rows if r["regime"] == regime]
        fixed = [r for r in offline if r["regime"] == regime]
        per_policy = {}
        for policy in POLICIES:
            entries = [r["policies"][policy] for r in cases]
            per_policy[policy] = {
                "completed": sum(x["completed"] for x in entries),
                "total": sum(x["total"] for x in entries),
                "timed_out_episodes": sum(x["timed_out"] for x in entries),
                "mean_restricted_flow_ticks": round(statistics.mean(x["restricted_mean_flow_ticks"] for x in entries), 3),
                "mean_episode_cpu_seconds": round(statistics.mean(x["cpu_seconds"] for x in entries), 4),
                "candidate_evaluations": sum(x["candidate_evaluations"] for x in entries),
            }
        errors = {}
        for policy in ("world", "learned"):
            pickup = [abs(r[f"{policy}_pickup"] - r["actual_pickup"]) for r in fixed]
            delivery = [abs(r[f"{policy}_completion"] - r["actual_completion"]) for r in fixed]
            errors[policy] = {"pickup_mae_ticks": round(statistics.mean(pickup), 3) if pickup else None,
                              "completion_mae_ticks": round(statistics.mean(delivery), 3) if delivery else None}
        delta = [r["policies"]["learned"]["restricted_mean_flow_ticks"]
                 - r["policies"]["world"]["restricted_mean_flow_ticks"] for r in cases]
        output[regime] = {"episodes": len(cases), "policies": per_policy,
                          "offline_same_decisions": {"count": len(fixed), "errors": errors},
                          "same_assignment_trace_world_and_learned": sum(
                              r["policies"]["world"]["assignments"] == r["policies"]["learned"]["assignments"]
                              for r in cases),
                          "learned_minus_world_restricted_flow_ticks": {
                              "mean": round(statistics.mean(delta), 3), "median": round(statistics.median(delta), 3),
                              "lower": round(min(delta), 3), "upper": round(max(delta), 3)}}
    return output


def run_experiment(base: Scenario) -> tuple[dict[str, Any], dict[str, Any]]:
    training, training_censored = [], 0
    for regime, seed in training_cases():
        result = Simulator(make_case(base, seed, regime), "world", seed).run()
        records, missing = observations(result)
        training.extend(records)
        training_censored += missing
    parameters = train(training, ridge_lambda=4.0)
    parameters["provenance"] = {"source_commit": SOURCE_COMMIT, "training_seeds": list(TRAIN_SEEDS),
                                 "training_episodes": len(TRAIN_SEEDS), "censored_excluded": training_censored,
                                 "scenario": "synthetic tasks and disruptions on a source-backed estimated-time graph"}
    rows, offline = [], []
    offline_censored = {"normal": 0, "short_outage": 0, "shifted": 0}
    for regime, seed in testing_cases():
        scenario = make_case(base, seed, regime)
        row = {"regime": regime, "seed": seed, "policies": {}}
        for policy in POLICIES:
            start = time.process_time()
            result = Simulator(scenario, policy, seed, parameters if policy == "learned" else None).run()
            elapsed = time.process_time() - start
            row["policies"][policy] = outcome(result, scenario.max_ticks, elapsed)
            if policy == "world":
                records, missing = observations(result)
                offline_censored[regime] += missing
                for record in records:
                    predicted = corrected_forecast(record["baseline"], record["tick"], None, parameters)
                    offline.append({"regime": regime, "actual_pickup": record["actual_pickup"],
                                    "actual_completion": record["actual_completion"],
                                    "world_pickup": record["baseline"]["expected_pickup_tick"],
                                    "world_completion": record["baseline"]["expected_completion_tick"],
                                    "learned_pickup": predicted["expected_pickup_tick"],
                                    "learned_completion": predicted["expected_completion_tick"]})
        rows.append(row)
    results = {"description": "模拟训练与测试；不是临床运输记录或通用性能结论。",
               "training_episodes": len(TRAIN_SEEDS), "training_records": len(training),
               "training_censored_excluded": training_censored, "offline_censored_excluded": offline_censored,
               "test_episodes": len(rows), "aggregates": aggregate(rows, offline), "cases": rows}
    return parameters, results


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU 学习预测器与对照实验；任务和扰动为仿真设置")
    parser.add_argument("--map-file", required=True)
    parser.add_argument("--tiago-config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        output = Path(args.output)
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
        base = make_scenario(Path(args.map_file).read_bytes(), seconds_per_tick=10,
                             tiago_config=Path(args.tiago_config).read_bytes())
        parameters, results = run_experiment(base)
        output.mkdir(parents=True, exist_ok=True)
        (output / "learned-parameters.json").write_text(json.dumps(parameters, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (output / "experiment-results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: results[k] for k in ("description", "training_episodes", "training_records",
                                                  "training_censored_excluded", "offline_censored_excluded",
                                                  "test_episodes", "aggregates")}, ensure_ascii=False))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"实验失败：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
