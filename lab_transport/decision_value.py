"""Bounded CPU decision-value pilot. All workloads and disruptions are simulated.

Online policies see only Observation, candidates and static Geometry. Counterfactual
replays are performed after the factual episode and never become policy inputs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
from itertools import combinations
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any

from .clinic_map import make_clinic_scenario
from .domain import Scenario
from .dynamics_experiment import audit_task_schedules, check_predicted_step, compare
from .dynamics_model import Geometry, HybridDynamicsModel
from .dynamics_revision import RevisedDynamicsModel, SPLITS_V2
from .experiment import outcome
from .hospital_case import make_scenario
from .multisite_experiment import discrete_safety_violations, make_case, prepare_tartu
from .simulation import Simulator
from .training_data import download_sources, SOURCES
from .transition_schema import Assignment, Observation

REGIMES = ("normal", "short_outage", "burst", "urgent", "contention", "motion_delay")
SEEDS = {"pilot": {"tartu": (11000,), "clinic": (12000,)},
         "confirm": {"tartu": (11001,), "clinic": (12001,)}}
HORIZON, BUDGET = 20, 8
ROOT = Path(__file__).resolve().parents[1]
PARAMETERS = ROOT / "examples/evidence/dynamics-revision/dynamics-parameters.json"


def candidates_for(obs: Observation) -> tuple[Assignment, ...]:
    return tuple(Assignment(r.id, s.id) for r in obs.robots if r.task_id is None
                 for s in obs.samples if s.status == "waiting")


def batch_candidates(obs: Observation, candidates: tuple[Assignment, ...], geometry: Geometry,
                     budget: int = BUDGET) -> tuple[list[tuple[Assignment, ...]], int]:
    """Wait plus bounded distinct valid matchings; no hidden scenario data."""
    if budget < 2:
        raise ValueError("budget must allow waiting and one assignment")
    robots = {r.id: r for r in obs.robots}
    samples = {s.id: s for s in obs.samples}
    stations = {s.id: s for s in obs.stations}
    items = tuple(sorted(candidates, key=lambda a: (a.robot_id, a.sample_id)))
    maximum = min(len({a.robot_id for a in items}), len({a.sample_id for a in items}))
    batches = []
    for size in range(1, maximum + 1):
        for batch in combinations(items, size):
            if len({x.robot_id for x in batch}) != size or len({x.sample_id for x in batch}) != size:
                continue
            distances = [geometry.distance(robots[a.robot_id].position, stations[samples[a.sample_id].pickup].position)
                         for a in batch]
            if any(distance >= geometry.unreachable for distance in distances):
                continue
            distance = sum(distances)
            key = (-size, distance, tuple((a.robot_id, a.sample_id) for a in batch))
            batches.append((key, batch))
    batches.sort(key=lambda entry: entry[0])
    return [()] + [b for _, b in batches[:budget - 1]], len(batches) + 1


def nearest_batch(obs: Observation, candidates: tuple[Assignment, ...], geometry: Geometry) -> tuple[Assignment, ...]:
    batches, _ = batch_candidates(obs, candidates, geometry)
    return batches[1] if len(batches) > 1 else ()


def terminal_distance(obs: Observation, initial_ids: set[str], geometry: Geometry) -> int:
    robots = {r.id: r for r in obs.robots}
    stations = {s.id: s.position for s in obs.stations}
    total = 0
    for sample in obs.samples:
        if sample.id not in initial_ids or sample.status == "completed":
            continue
        if sample.status == "in_transit":
            if sample.robot_id not in robots:
                raise ValueError("in-transit sample has no robot")
            total += geometry.distance(robots[sample.robot_id].position, stations[sample.dropoff])
        elif sample.status in {"waiting", "assigned"}:
            if sample.status == "assigned":
                if sample.robot_id not in robots:
                    raise ValueError("assigned sample has no robot")
                start = geometry.distance(robots[sample.robot_id].position, stations[sample.pickup])
            else:
                start = min(geometry.distance(r.position, stations[sample.pickup]) for r in robots.values())
            total += start + geometry.distance(stations[sample.pickup], stations[sample.dropoff])
        else:
            raise ValueError("unexpected sample status in observable state")
    return total


def trace_score(initial: Observation, trace: list[Observation], geometry: Geometry) -> tuple[int, int]:
    if len(trace) != HORIZON:
        raise ValueError("incomplete prediction or reference horizon")
    known = {s.id for s in initial.samples}
    cost = 0
    for obs in trace:
        samples = {s.id: s for s in obs.samples}
        if not known.issubset(samples):
            raise ValueError("a known sample disappeared")
        for sid in known:
            s = samples[sid]
            if s.status != "completed":
                cost += 1 + 4 * int(s.deadline_tick is not None and obs.tick > s.deadline_tick)
    return cost, terminal_distance(trace[-1], known, geometry)


def predict_trace(model: Any, obs: Observation, batch: tuple[Assignment, ...], geometry: Geometry) -> list[Observation]:
    trace = []
    current = obs
    for step in range(HORIZON):
        actions = batch if step == 0 else nearest_batch(current, candidates_for(current), geometry)
        nxt = model.predict_next(current, actions)
        check_predicted_step(current, nxt, geometry)
        if {s.id for s in nxt.samples} != {s.id for s in obs.samples}:
            raise ValueError("predictor changed the set of currently known samples")
        trace.append(nxt)
        current = nxt
    return trace


def encode_batch(batch: tuple[Assignment, ...]) -> list[dict[str, str]]:
    return [asdict(a) for a in batch]


class ModelPolicy:
    """Controlled online policy: no future scenario or simulator reference."""
    def __init__(self, model: Any, geometry: Geometry):
        self.model, self.geometry = model, geometry
        self.decisions: list[dict[str, Any]] = []
        self.model_steps = 0
        self.fallbacks = 0
        self.cpu_seconds = 0.0

    def __call__(self, obs: Observation, candidates: tuple[Assignment, ...]):
        start = time.process_time()
        batches, total = batch_candidates(obs, candidates, self.geometry)
        records = []
        try:
            for batch in batches:
                trace = predict_trace(self.model, obs, batch, self.geometry)
                self.model_steps += len(trace)
                records.append({"batch": encode_batch(batch), "score": list(trace_score(obs, trace, self.geometry))})
            chosen = min(range(len(records)), key=lambda i: (records[i]["score"], i))
        except (ValueError, RuntimeError, TypeError, KeyError, StopIteration, OverflowError) as exc:
            self.fallbacks += 1
            chosen = 1 if len(batches) > 1 else 0
            self.decisions.append({"tick": obs.tick, "candidate_count": total,
                                   "evaluated": records, "fallback": type(exc).__name__,
                                   "selected": encode_batch(batches[chosen])})
            self.cpu_seconds += time.process_time() - start
            return batches[chosen]
        self.decisions.append({"tick": obs.tick, "candidate_count": total, "evaluated": records,
                               "fallback": None, "selected": encode_batch(batches[chosen])})
        self.cpu_seconds += time.process_time() - start
        return batches[chosen]


def scenario_case(base: Scenario, layout: str, seed: int, regime: str) -> tuple[Scenario, dict]:
    if regime not in REGIMES:
        raise ValueError("unknown regime")
    case = make_case(base, layout, seed, "short_outage" if regime == "short_outage" else "normal")
    samples = list(case.samples)
    pauses = {}
    if regime in {"burst", "contention"}:
        samples = [replace(s, release_tick=tick) for s, tick in zip(samples, (0, 0, 0, 4, 4, 4))]
    if regime == "contention":
        target = "实验室交付点" if layout == "tartu" else "护理左_假设交接"
        samples = [replace(s, dropoff=target) for s in samples]
    if regime == "urgent":
        s = samples[-1]
        geometry = Geometry(case.grid)
        pickup, dropoff = case.stations[s.pickup], case.stations[s.dropoff]
        lower = (min(geometry.distance(r.position, pickup.position) for r in case.robots)
                 + geometry.distance(pickup.position, dropoff.position)
                 + math.ceil(pickup.prior_ticks) + math.ceil(dropoff.prior_ticks))
        samples[-1] = replace(s, release_tick=8, deadline_tick=8 + lower + 5)
    if regime == "motion_delay":
        pauses = {"R1": ((10, 16),)}
    return replace(case, name=f"{layout} {regime} seed={seed} (synthetic)", samples=tuple(samples)), pauses


def case_config(case: Scenario, pauses: dict) -> dict:
    return {"name": case.name, "max_ticks": case.max_ticks,
            "robots": [{"id": r.id} for r in case.robots],
            "samples": [{"id": s.id, "pickup": s.pickup, "dropoff": s.dropoff,
                         "release_tick": s.release_tick, "deadline_tick": s.deadline_tick} for s in case.samples],
            "outages": {sid: list(s.outages) for sid, s in case.stations.items() if s.outages},
            "motion_pauses": pauses, "duration_jitter": case.duration_jitter,
            "station_service": {sid: {"service_ticks": s.service_ticks, "prior_ticks": s.prior_ticks,
                                       "jitter_ticks": s.jitter_ticks} for sid, s in case.stations.items()},
            "source": case.metadata or {}, "note": "workload, station roles and disruptions are simulation assumptions"}


def metrics(result: dict, max_ticks: int, elapsed: float | None, incident_end: int | None) -> dict:
    out = outcome(result, max_ticks, elapsed)
    samples = result["samples"]
    completed_waits = [s["assigned_tick"] - s["release_tick"] for s in samples if s["assigned_tick"] is not None]
    restricted_waits = [(s["assigned_tick"] if s["assigned_tick"] is not None else max_ticks) - s["release_tick"]
                        for s in samples]
    out.update({"mean_assignment_wait_completed": statistics.mean(completed_waits) if completed_waits else None,
                "restricted_mean_assignment_wait": statistics.mean(restricted_waits),
                "summary": result["summary"], "discrete_safety_violations": discrete_safety_violations(result)})
    recovery = {"status": "not_applicable", "ticks": None}
    if incident_end is not None and incident_end < len(result["history"]):
        state = result["history"][incident_end]
        if any(s["status"] != "completed" for s in state["samples"]):
            events = [e["tick"] for e in result["events"]
                      if e["type"] in {"picked_up", "delivered"} and e["tick"] >= incident_end]
            recovery = ({"status": "observed", "ticks": min(events) - incident_end} if events
                        else {"status": "right_censored", "ticks": result["summary"]["ticks"] - incident_end})
    out["post_incident_next_progress"] = recovery
    return out


def incident_end(case: Scenario, pauses: dict) -> int | None:
    ends = [end for station in case.stations.values() for _, end in station.outages]
    ends.extend(end for intervals in pauses.values() for _, end in intervals)
    return max(ends) if ends else None


def audit_transformed_schedules(bases: dict) -> dict:
    """Compare nominal and modified task schedules without inspecting outcomes."""
    signatures = {}
    for split, (layout, seeds) in SPLITS_V2.items():
        for seed in seeds:
            case = make_case(bases[layout], layout, seed, "normal")
            key = (layout, tuple((s.pickup, s.dropoff, s.release_tick) for s in case.samples))
            signatures.setdefault(key, set()).add(split)
    for profile, maps in SEEDS.items():
        for layout, seeds in maps.items():
            for seed in seeds:
                for regime in REGIMES:
                    case, _ = scenario_case(bases[layout], layout, seed, regime)
                    key = (layout, tuple((s.pickup, s.dropoff, s.release_tick) for s in case.samples))
                    signatures.setdefault(key, set()).add(profile)
    overlaps = [sorted(groups) for groups in signatures.values() if len(groups) > 1]
    return {"unique_task_patterns": len(signatures), "cross_split_duplicate_groups": len(overlaps),
            "duplicate_split_names": overlaps}


def factual_run(case: Scenario, seed: int, pauses: dict):
    rows = []
    result = Simulator(case, "world", seed, motion_pauses=pauses, collect_metrics=True).run(on_transition=rows.append)
    return result, rows


def replay_branch(case: Scenario, seed: int, pauses: dict, original: dict, rows: list,
                  target_tick: int, batch: tuple[Assignment, ...]):
    """Replay factual actions, then force a batch; never expose truth to policy."""
    source = {r.observation.tick: r for r in rows}
    geometry = Geometry(case.grid)
    branch_rows = []
    def policy(obs, candidates):
        if obs.tick <= target_tick:
            if obs.tick not in source or obs != source[obs.tick].observation:
                raise ValueError("replay observation differs from factual prefix")
            return source[obs.tick].assignments if obs.tick < target_tick else batch
        return nearest_batch(obs, candidates, geometry)
    result = Simulator(case, "world", seed, motion_pauses=pauses, collect_metrics=True).run(
        on_transition=branch_rows.append, dispatch_policy=policy)
    if result["history"][:target_tick + 1] != original["history"][:target_tick + 1]:
        raise ValueError("replay hidden-state history differs from factual prefix")
    before = lambda events: [e for e in events if e["tick"] < target_tick and e["type"] != "assigned"]
    if before(result["events"]) != before(original["events"]):
        raise ValueError("replay service/recovery events differ from factual prefix")
    return result, branch_rows


def truth_trace(initial: Observation, rows: list) -> tuple[list[Observation], bool]:
    start = initial.tick
    relevant = [r for r in rows if start <= r.observation.tick < start + HORIZON]
    trace = [r.next_observation for r in relevant]
    if len(trace) < HORIZON:
        if relevant and relevant[-1].termination == "completed":
            last = trace[-1]
            while len(trace) < HORIZON:
                last = replace(last, tick=last.tick + 1)
                trace.append(last)
        else:
            return trace, False
    return trace, True


def decision_analysis(case: Scenario, seed: int, pauses: dict, original: dict, rows: list, parameters: dict) -> list[dict]:
    geometry = Geometry(case.grid)
    selected = []
    for row in rows:
        options = candidates_for(row.observation)
        batches, total = batch_candidates(row.observation, options, geometry)
        if options and len(batches) >= 2:
            selected.append((row.observation, batches, total))
        if len(selected) == 2:
            break
    records = []
    for initial, batches, total in selected:
        models = {"heuristic": HybridDynamicsModel(geometry, mode="heuristic"),
                  "learned": RevisedDynamicsModel(geometry, parameters)}
        predictions: dict[str, list] = {}
        errors = {}
        for name, model in models.items():
            try:
                predictions[name] = [predict_trace(model, initial, batch, geometry) for batch in batches]
            except (ValueError, RuntimeError, TypeError, KeyError, StopIteration, OverflowError) as exc:
                errors[name] = type(exc).__name__
        candidate_records = []
        for idx, batch in enumerate(batches):
            actual, branch = replay_branch(case, seed, pauses, original, rows, initial.tick, batch)
            trace, valid = truth_trace(initial, branch)
            item = {"batch": encode_batch(batch), "reference_complete": valid,
                    "full_episode": metrics(actual, case.max_ticks, None, incident_end(case, pauses))}
            if valid:
                item["reference_score"] = list(trace_score(initial, trace, geometry))
                item["future_new_samples"] = len({s.id for o in trace for s in o.samples} - {s.id for s in initial.samples})
                item["future_station_change"] = any(
                    {s.id: s.available for s in o.stations} != {s.id: s.available for s in initial.stations}
                    for o in trace)
                item["predictions"] = {}
                for name, traces in predictions.items():
                    predicted = traces[idx]
                    counts = {}
                    for horizon in (1, 5, 20):
                        counts[str(horizon)] = compare(initial, trace[horizon - 1], predicted[horizon - 1])
                    item["predictions"][name] = {"score": list(trace_score(initial, predicted, geometry)),
                                                  "horizon_counts": counts}
            candidate_records.append(item)
        valid_all = all(x["reference_complete"] for x in candidate_records)
        choice = {}
        if valid_all:
            oracle = min(range(len(batches)), key=lambda i: (candidate_records[i]["reference_score"], i))
            choice["offline_reference"] = oracle
            for name in predictions:
                choice[name] = min(range(len(batches)),
                    key=lambda i: (candidate_records[i]["predictions"][name]["score"], i))
        records.append({"tick": initial.tick, "known_samples": len(initial.samples), "candidate_count": total,
                        "candidate_evaluated": len(batches), "prediction_errors": errors,
                        "choices": choice, "candidates": candidate_records})
    return records


def run_experiment(profile: str) -> dict:
    started = time.process_time()
    raw = download_sources()
    bases = {"tartu": prepare_tartu(make_scenario(raw["hospital_map"], seconds_per_tick=10,
                                                   tiago_config=raw["tiago_config"])),
             "clinic": make_clinic_scenario(raw["clinic_map"], raw["clinic_config"])}
    splits = dict(SPLITS_V2)
    for part, maps in SEEDS.items():
        for layout, seeds in maps.items():
            splits[f"{part}_{layout}"] = (layout, seeds)
    audit = audit_task_schedules(bases, splits)
    if audit["cross_split_task_duplicate_groups"] or audit["cross_split_duplicate_groups"]:
        raise ValueError("fixed nominal schedules overlap prior model data")
    transformed_audit = audit_transformed_schedules(bases)
    if transformed_audit["cross_split_duplicate_groups"]:
        raise ValueError("modified task schedules overlap prior model data or pilot/confirm splits")
    parameters = json.loads(PARAMETERS.read_text(encoding="utf-8"))
    cases = []
    for layout, seeds in SEEDS[profile].items():
        for seed in seeds:
            for regime in REGIMES:
                case, pauses = scenario_case(bases[layout], layout, seed, regime)
                original, rows = factual_run(case, seed, pauses)
                geometry = Geometry(case.grid)
                policies = {}
                for name in ("nearest", "world", "rule", "rollout", "heuristic", "learned"):
                    model_policy = None
                    if name == "heuristic":
                        model_policy = ModelPolicy(HybridDynamicsModel(geometry, mode="heuristic"), geometry)
                    elif name == "learned":
                        model_policy = ModelPolicy(RevisedDynamicsModel(geometry, parameters), geometry)
                    begin = time.process_time()
                    if name == "world":
                        result = original
                    else:
                        result = Simulator(case, "world" if model_policy else name, seed,
                                           motion_pauses=pauses, collect_metrics=True).run(dispatch_policy=model_policy)
                    elapsed = time.process_time() - begin
                    entry = metrics(result, case.max_ticks, elapsed, incident_end(case, pauses))
                    if name == "world":
                        # Factual run occurred before the timing block; do not report a fake near-zero measurement.
                        begin = time.process_time()
                        timed = Simulator(case, "world", seed, motion_pauses=pauses, collect_metrics=True).run()
                        entry["cpu_seconds"] = time.process_time() - begin
                        if timed["history"] != original["history"]:
                            raise ValueError("factual timing replay differs")
                    if model_policy:
                        entry["candidate_evaluations"] = sum(len(x["evaluated"]) for x in model_policy.decisions)
                        entry["model_steps"] = model_policy.model_steps
                        entry["fallbacks"] = model_policy.fallbacks
                        entry["decision_cpu_seconds"] = model_policy.cpu_seconds
                        entry["decisions"] = model_policy.decisions
                    policies[name] = entry
                decisions = decision_analysis(case, seed, pauses, original, rows, parameters)
                cases.append({"layout": layout, "seed": seed, "regime": regime,
                              "config": case_config(case, pauses), "policies": policies,
                              "offline_decisions": decisions})
                print(f"{layout} {regime} seed={seed}: decisions={len(decisions)} " +
                      " ".join(f"{p}={policies[p]['completed']}/{policies[p]['total']}" for p in policies), flush=True)
    return {"profile": profile, "scope": "CPU; synthetic workload; offline reference is privileged and bounded",
            "environment": {"python": platform.python_version(), "platform": platform.platform(),
                            "github_sha": os.environ.get("GITHUB_SHA"), "github_run_id": os.environ.get("GITHUB_RUN_ID")},
            "learned_parameters_sha256": hashlib.sha256(PARAMETERS.read_bytes()).hexdigest(),
            "source_sha256": {key: expected for key, (_, expected) in SOURCES.items()},
            "prior_split_schedule_audit": {k: audit[k] for k in ("unique_task_patterns", "cross_split_task_duplicate_groups", "cross_split_duplicate_groups")},
            "transformed_schedule_audit": transformed_audit,
            "horizon": HORIZON, "candidate_budget": BUDGET, "cases": cases,
            "cpu_seconds": time.process_time() - started}


def summarize(result: dict) -> dict:
    """Compact descriptive counts; decision/candidate rows are correlated, not independent trials."""
    cases = result["cases"]
    decisions = [d for case in cases for d in case["offline_decisions"]]
    comparable = [d for d in decisions if d["choices"]]
    summary = {"profile": result["profile"], "case_count": len(cases),
               "decision_count": len(decisions), "comparable_decisions": len(comparable),
               "candidate_branches": sum(d["candidate_evaluated"] for d in decisions),
               "truncated_candidate_decisions": sum(d["candidate_count"] > d["candidate_evaluated"] for d in decisions),
               "prediction_errors": sum(len(d["prediction_errors"]) for d in decisions),
               "discrete_safety_violations": sum(p["discrete_safety_violations"]
                                                   for c in cases for p in c["policies"].values()),
               "model_comparison": {}, "per_case": [], "cpu_seconds": result["cpu_seconds"]}
    for name in ("heuristic", "learned"):
        selected = [d for d in comparable if name in d["choices"]]
        counts = {"n": len(selected), "same_reference_score": 0, "strictly_worse_primary_score": 0,
                  "same_reference_index": 0, "joint_accuracy": {}}
        for d in selected:
            oracle, chosen = d["choices"]["offline_reference"], d["choices"][name]
            reference = d["candidates"][oracle]["reference_score"]
            observed = d["candidates"][chosen]["reference_score"]
            counts["same_reference_score"] += observed == reference
            counts["same_reference_index"] += oracle == chosen
            counts["strictly_worse_primary_score"] += observed[0] > reference[0]
        for h in ("1", "5", "20"):
            entries = [candidate["predictions"][name]["horizon_counts"][h]
                       for d in selected for candidate in d["candidates"]
                       if name in candidate.get("predictions", {})]
            denominator = sum(x["joint_total"] for x in entries)
            counts["joint_accuracy"][h] = {"correct": sum(x["joint_correct"] for x in entries),
                                           "total": denominator}
        summary["model_comparison"][name] = counts
    both = [d for d in comparable if "heuristic" in d["choices"] and "learned" in d["choices"]]
    summary["different_model_choices"] = sum(d["choices"]["heuristic"] != d["choices"]["learned"] for d in both)
    summary["paired_counterfactual_flow"] = {"learned_better": 0, "equal": 0, "learned_worse": 0}
    for d in both:
        h = d["candidates"][d["choices"]["heuristic"]]["full_episode"]["restricted_mean_flow_ticks"]
        l = d["candidates"][d["choices"]["learned"]]["full_episode"]["restricted_mean_flow_ticks"]
        summary["paired_counterfactual_flow"]["learned_better" if l < h else "learned_worse" if l > h else "equal"] += 1
    for case in cases:
        summary["per_case"].append({"layout": case["layout"], "regime": case["regime"], "seed": case["seed"],
            "policies": {name: {key: p.get(key) for key in ("completed", "total", "restricted_mean_flow_ticks", "cpu_seconds", "fallbacks")}
                         for name, p in case["policies"].items()},
            "station_outage_exposure": {name: p["summary"]["station_unavailable_wait_ticks"] for name, p in case["policies"].items()},
            "motion_delay_exposure": {name: p["summary"]["motion_pause_ticks"] for name, p in case["policies"].items()}})
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(SEEDS), default="pilot")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
            raise FileExistsError(f"refusing to overwrite nonempty output directory: {args.output}")
        result = run_experiment(args.profile)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "decision-value.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (args.output / "summary.json").write_text(json.dumps(summarize(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"profile": args.profile, "cases": len(result["cases"]),
                          "decisions": sum(len(c["offline_decisions"]) for c in result["cases"]),
                          "cpu_seconds": result["cpu_seconds"]}, ensure_ascii=False), flush=True)
        return 0
    except (OSError, ValueError, TypeError, KeyError, ImportError, RuntimeError) as exc:
        print(f"decision-value experiment failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
