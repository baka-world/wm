"""Fixed, CPU-only, multi-location evaluation on two pinned public maps.

Sample roles, jobs and disruptions are simulation assumptions; the clinic
handoff is not claimed to be a laboratory. No training occurs in this module.
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

from .clinic_map import make_clinic_scenario
from .domain import Sample, Scenario
from .experiment import TRAIN_SEEDS, observations, outcome
from .hospital_case import make_scenario
from .learned_model import corrected_forecast, validate
from .simulation import Simulator

LAYOUTS = ("tartu", "clinic")
REGIMES = ("normal", "short_outage")
SEEDS = {"tartu": tuple(range(1000, 1005)), "clinic": tuple(range(2000, 2005))}
POLICIES = ("nearest", "world", "learned", "rollout")
PICKUPS = {"tartu": ("ICU取样点", "ICU2_假设取样"),
           "clinic": ("诊疗左_假设取样", "诊疗右_假设取样")}
DROPOFFS = {"tartu": ("实验室交付点",), "clinic": ("护理左_假设交接", "护理右_假设交接")}


def prepare_tartu(base: Scenario) -> Scenario:
    stations = dict(base.stations)
    reference = stations["ICU取样点"]
    # Existing waypoint icu_2, repurposed only as an explicit simulation assumption.
    stations["ICU2_假设取样"] = replace(reference, id="ICU2_假设取样", position=(178, 0))
    positions = [r.position for r in base.robots] + [s.position for s in stations.values()]
    if any(base.grid.shortest_path(a, b) is None for a in positions for b in positions):
        raise ValueError("Tartu experiment waypoints are not mutually reachable")
    return replace(base, stations=stations)


def make_case(base: Scenario, layout: str, seed: int, regime: str) -> Scenario:
    if layout not in LAYOUTS or regime not in REGIMES:
        raise ValueError("unknown layout or regime")
    rng = random.Random(4919 + seed)
    sources = list(PICKUPS[layout]) * 3
    destinations = list(DROPOFFS[layout]) * (6 // len(DROPOFFS[layout]))
    rng.shuffle(sources)
    rng.shuffle(destinations)
    samples = []
    release = 0
    for index, (pickup, dropoff) in enumerate(zip(sources, destinations), start=1):
        if index > 1:
            release += rng.choice((4, 8, 12, 16))
        samples.append(Sample(f"S{index:02}", pickup, dropoff, release, None))
    stations = {sid: replace(station, outages=()) for sid, station in base.stations.items()}
    if regime == "short_outage":
        target = rng.choice(sorted(stations))
        start = rng.randint(10, 55)
        duration = rng.randint(6, 14)
        stations[target] = replace(stations[target], outages=((start, start + duration),))
    return replace(base, name=f"{layout} synthetic multi-location / {regime} / {seed}",
                   stations=stations, samples=tuple(samples), max_ticks=350)


def discrete_safety_violations(result: dict[str, Any]) -> int:
    """Count vertex sharing and edge swaps in recorded states, not physical collisions."""
    violations = 0
    previous = None
    for snapshot in result["history"]:
        current = {robot["id"]: tuple(robot["position"]) for robot in snapshot["robots"]}
        violations += len(current) - len(set(current.values()))
        if previous is not None:
            ids = sorted(current)
            violations += sum(previous[a] == current[b] and previous[b] == current[a]
                              and previous[a] != current[a]
                              for index, a in enumerate(ids) for b in ids[index + 1:])
        previous = current
    return violations


def aggregate(rows: list[dict[str, Any]], offline: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    for layout in LAYOUTS:
        summary[layout] = {}
        for regime in REGIMES:
            cases = [r for r in rows if r["layout"] == layout and r["regime"] == regime]
            matched = [r for r in offline if r["layout"] == layout and r["regime"] == regime]
            policies = {}
            for policy in POLICIES:
                entries = [r["policies"][policy] for r in cases]
                policies[policy] = {
                    "completed": sum(x["completed"] for x in entries),
                    "total": sum(x["total"] for x in entries),
                    "timed_out_episodes": sum(x["timed_out"] for x in entries),
                    "mean_restricted_flow_ticks": round(statistics.mean(x["restricted_mean_flow_ticks"] for x in entries), 3),
                    "mean_episode_cpu_seconds": round(statistics.mean(x["cpu_seconds"] for x in entries), 4),
                    "candidate_evaluations": sum(x["candidate_evaluations"] for x in entries),
                    "discrete_safety_violations": sum(x["discrete_safety_violations"] for x in entries),
                }
            errors = {}
            for policy in ("world", "learned"):
                errors[policy] = {
                    f"{kind}_mae_ticks": round(statistics.mean(abs(r[f"{policy}_{kind}"] - r[f"actual_{kind}"])
                                                              for r in matched), 3) if matched else None
                    for kind in ("pickup", "completion")}
            differences = {}
            for policy in ("learned", "rollout"):
                delta = [r["policies"][policy]["restricted_mean_flow_ticks"] -
                         r["policies"]["world"]["restricted_mean_flow_ticks"] for r in cases]
                differences[policy] = {"mean": round(statistics.mean(delta), 3),
                                       "min": round(min(delta), 3), "max": round(max(delta), 3)}
            summary[layout][regime] = {
                "episodes": len(cases), "policies": policies,
                "offline_same_world_decisions": {"completed_records": len(matched),
                    "censored_excluded": sum(r["world_censored"] for r in cases), "errors": errors},
                "same_assignment_trace_world_and_learned": sum(
                    r["policies"]["world"]["assignments"] == r["policies"]["learned"]["assignments"]
                    for r in cases),
                "paired_restricted_flow_difference_vs_world_ticks": differences,
            }
    return summary


def run_experiment(tartu: Scenario, clinic: Scenario, parameters: dict[str, Any]) -> dict[str, Any]:
    validate(parameters)
    provenance = parameters.get("provenance", {})
    recorded_seeds = provenance.get("training_seeds")
    if recorded_seeds is None or set(recorded_seeds) & {seed for values in SEEDS.values() for seed in values}:
        raise ValueError("model training seeds missing or overlapping the fixed test seeds")
    if set(recorded_seeds) != set(TRAIN_SEEDS):
        raise ValueError("expected the fixed previously trained model")
    bases = {"tartu": prepare_tartu(tartu), "clinic": clinic}
    rows, offline = [], []
    for layout in LAYOUTS:
        for regime in REGIMES:
            for seed in SEEDS[layout]:
                scenario = make_case(bases[layout], layout, seed, regime)
                row = {"layout": layout, "regime": regime, "seed": seed,
                       "samples": [{"id": s.id, "pickup": s.pickup, "dropoff": s.dropoff,
                                    "release_tick": s.release_tick} for s in scenario.samples],
                       "outages": {sid: list(station.outages) for sid, station in scenario.stations.items()
                                   if station.outages}, "policies": {}}
                for policy in POLICIES:
                    start = time.process_time()
                    result = Simulator(scenario, policy, seed, parameters if policy == "learned" else None).run()
                    elapsed = time.process_time() - start
                    metrics = outcome(result, scenario.max_ticks, elapsed)
                    metrics["discrete_safety_violations"] = discrete_safety_violations(result)
                    if policy == "rollout":
                        metrics["censored_candidate_evaluations"] = sum(
                            forecast["rollout_horizon_reached"] for event in result["events"]
                            if event["type"] == "assigned" for forecast in event["candidates"])
                    row["policies"][policy] = metrics
                    if policy == "world":
                        records, row["world_censored"] = observations(result)
                        for record in records:
                            predicted = corrected_forecast(record["baseline"], record["tick"], None, parameters)
                            offline.append({"layout": layout, "regime": regime,
                                "actual_pickup": record["actual_pickup"], "actual_completion": record["actual_completion"],
                                "world_pickup": record["baseline"]["expected_pickup_tick"],
                                "world_completion": record["baseline"]["expected_completion_tick"],
                                "learned_pickup": predicted["expected_pickup_tick"],
                                "learned_completion": predicted["expected_completion_tick"]})
                rows.append(row)
                print(f"{layout} {regime} seed={seed} " +
                      " ".join(f"{p}:{row['policies'][p]['completed']}/6" for p in POLICIES), flush=True)
    return {"description": "公开地图上的合成任务；站点角色和扰动是假设，非临床运输记录。",
            "test_episodes": len(rows), "rollout_horizon_ticks": 120,
            "model_training_seeds": sorted(recorded_seeds),
            "aggregates": aggregate(rows, offline), "cases": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("hospital-map", "tiago-config", "clinic-map", "clinic-config", "model", "output"):
        parser.add_argument("--" + argument, required=True)
    args = parser.parse_args()
    try:
        output = Path(args.output)
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
        tartu = make_scenario(Path(args.hospital_map).read_bytes(), seconds_per_tick=10,
                              tiago_config=Path(args.tiago_config).read_bytes())
        clinic = make_clinic_scenario(Path(args.clinic_map).read_bytes(), Path(args.clinic_config).read_bytes())
        parameters = json.loads(Path(args.model).read_text(encoding="utf-8"))
        results = run_experiment(tartu, clinic, parameters)
        output.mkdir(parents=True, exist_ok=True)
        (output / "experiment-results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(results["aggregates"], ensure_ascii=False), flush=True)
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"实验失败：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
