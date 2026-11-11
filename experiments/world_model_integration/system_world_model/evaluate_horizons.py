"""Frozen group-level RHWM-v1 forecast-versus-factual-state evaluation.

New scenario seeds, fixed predeclared anchors, two model arms, shared factual
nearest-policy episodes. The ONLY simulator-private information used is to
generate ground-truth outcomes after all prediction inputs/actions are saved.
The world model sees public BusinessObserver snapshots and legal assignments.
No oracle future control actions, hidden service duration or future arrivals.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import argparse
import hashlib
import json
from pathlib import Path
import time

from business_transport.observation import BusinessObserver
from business_transport.scenario import compile_scenario
from business_transport.simulation import BusinessSimulator
from business_transport.world_model import HybridSystemWorldModel
from business_transport.session_events import eligible_sessions
from experiments.business_constraints.tests.test_business import fixture
from lab_transport.decision_value import nearest_batch, candidates_for
from lab_transport.dynamics_model import Geometry

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "generalization_protocol.json"
METRICS = (
    "sample_status_error", "robot_task_error", "robot_phase_error",
    "robot_position_l1", "station_busy_brier", "resource_occupancy_mae",
    "waiting_count_mae", "completed_count_mae", "in_transit_count_mae",
)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def group_scenario(spec, *, max_ticks):
    """Only static, declared, synthetic variants of the shared business fixture."""
    base, config, _ = fixture(capacity=spec["capacity"])
    if spec["jobs"] == 3:
        config = {**config, "jobs": [
            *config["jobs"], {"id": "J3", "workflow": "w0", "release_tick": 0}
        ]}
    # The initial factual queue is closed: no new unknown jobs after tick zero.
    assert all(job["release_tick"] == 0 for job in config["jobs"])
    station_overrides = {
        sid: replace(
            st, service_ticks=st.service_ticks + spec["service_add"],
            prior_ticks=st.prior_ticks + spec["service_add"],
            jitter_ticks=spec["jitter"])
        for sid, st in base.stations.items()
    }
    base = replace(base, stations=station_overrides,
                   robots=base.robots[:spec["robot_count"]],
                   duration_jitter=spec["jitter"], max_ticks=max_ticks,
                   name="rhwm-independent-" + spec["id"])
    cfg = {**config, "max_ticks": max_ticks,
           "name": "rhwm-independent-" + spec["id"]}
    business = compile_scenario(base, cfg)
    return business, Geometry(base.grid)


def factual_episode(business, geometry, seed):
    """Public observations and actions, with simulator used only as truth."""
    observed = {}
    previous = BusinessObserver()
    transitions = []

    def observe(raw):
        observed[raw.tick] = previous.observe(raw)

    def public_policy(raw, choices):
        selected = nearest_batch(raw, choices, geometry)
        if any(a not in choices for a in selected):
            raise AssertionError("factual policy chose an unavailable action")
        return selected

    sim = BusinessSimulator(business, "world", seed)
    result = sim.run(dispatch_policy=public_policy, on_observation=observe,
                     on_transition=transitions.append)
    if transitions and transitions[-1].next_boundary == "end_of_run":
        final = transitions[-1].next_observation
        observed[final.tick] = previous.observe(final)
    actions = {item.observation.tick: item.assignments for item in transitions}
    if len(actions) != len(transitions):
        raise AssertionError("duplicated factual transition tick")
    if any(t not in observed or (t + 1) not in observed for t in actions):
        raise AssertionError("missing chronological public observation")
    if any(observed[t].tick != t for t in observed):
        raise AssertionError("inconsistent public history clock")
    if result["summary"]["status"] != "completed":
        raise AssertionError("factual synthetic scenario did not complete")
    return observed, actions, result


def _event_label(before, after, query):
    old = next(s for s in before.samples if s.id == query["sample_id"])
    new = next(s for s in after.samples if s.id == query["sample_id"])
    target = "in_transit" if query["phase"] == "to_pickup" else "completed"
    if old.status == target:
        raise AssertionError("active service was already complete in public state")
    return int(new.status == target)


def score(prediction, truth, horizon):
    """Scenario-wise expected disagreement across finite forecast particles.

    All task and entity identifiers align by visible IDs, not by list index.
    Each probabilistic error is measured against one factual sample path; this
    does NOT establish calibration or a counterfactual expected utility.
    """
    states = [p.steps[horizon - 1].next_state for p in prediction.particles]
    if not states or any(st.tick != truth.tick for st in states):
        raise AssertionError("incorrect forecast horizon/clock")
    n = len(states)
    true_samples = {s.id: s.status for s in truth.samples}
    pred_samples = [{s.id: s.status for s in st.samples} for st in states]
    ids = set(true_samples)
    for sample in pred_samples:
        ids.update(sample)
    if not ids:
        raise AssertionError("no sample state to compare")
    status_mismatch = sum(
        sum((sample.get(sid, "not_yet_observed") !=
             true_samples.get(sid, "not_yet_observed")) for sample in pred_samples)
        for sid in ids) / (len(ids) * n)

    true_robots = {r.id: r for r in truth.robots}
    expected_robots = set(true_robots)
    robot_task = robot_phase = position_l1 = 0
    for st in states:
        robots = {r.id: r for r in st.robots}
        if set(robots) != expected_robots:
            raise AssertionError("predicted robot identities differ")
        for rid, actual in true_robots.items():
            guess = robots[rid]
            robot_task += guess.task_id != actual.task_id
            robot_phase += guess.phase != actual.phase
            position_l1 += sum(abs(a - b) for a,b in zip(guess.position, actual.position))
    robot_count = len(true_robots) * n

    stations = {s.id: s for s in truth.stations}
    sqerr = 0
    for sid, actual in stations.items():
        p = sum(next(s.service_in_progress for s in st.stations
                     if s.id == sid) for st in states) / n
        sqerr += (p - int(actual.service_in_progress)) ** 2

    resource_actual = {}
    for station in truth.stations:
        rid = station.resource_id
        if rid in resource_actual and resource_actual[rid] != station.resource_occupied:
            raise AssertionError("truth resource group occupied inconsistently")
        resource_actual[rid] = station.resource_occupied
    occupancy = 0
    for rid, actual in resource_actual.items():
        predicted = prediction.distributions[horizon - 1].resource_occupancy_mean[rid]
        occupancy += abs(predicted - actual)

    def count_error(attr):
        x = prediction.distributions[horizon - 1]
        p = getattr(x, attr)
        field = {
            "expected_waiting_samples": "waiting",
            "expected_completed_samples": "completed",
            "expected_in_transit_samples": "in_transit",
        }[attr]
        return abs(p - sum(s.status == field for s in truth.samples))

    return {
        "sample_status_error": status_mismatch,
        "robot_task_error": robot_task / robot_count,
        "robot_phase_error": robot_phase / robot_count,
        "robot_position_l1": position_l1 / robot_count,
        "station_busy_brier": sqerr / len(stations),
        "resource_occupancy_mae": occupancy / len(resource_actual),
        "waiting_count_mae": count_error("expected_waiting_samples"),
        "completed_count_mae": count_error("expected_completed_samples"),
        "in_transit_count_mae": count_error("expected_in_transit_samples"),
    }


def group_forecast(spec, protocol, checkpoint):
    business, geo = group_scenario(spec, max_ticks=protocol["max_ticks"])
    truth_states, first_actions, result = factual_episode(business, geo, spec["seed"])
    arms = {}
    for kind in protocol["world_model_branches"]:
        model = HybridSystemWorldModel(
            geo, business.rules, kind=kind,
            checkpoint=(checkpoint if kind == "v4" else None),
            experimental=(kind == "v4"))
        outputs = []
        calls = 0
        for anchor in protocol["anchors"]:
            if anchor not in first_actions:
                continue  # fixed-design anchor is right-censored by episode end
            available_h = [h for h in protocol["horizons"] if anchor + h in truth_states]
            if not available_h:
                continue
            horizon = max(available_h)
            initial = truth_states[anchor]
            actions = first_actions[anchor]
            if any(a not in candidates_for(initial) for a in actions):
                raise AssertionError("factual first action not reproducible from public observation")
            started = time.process_time()
            forecast = model.rollout(
                initial, actions,
                continuation_policy=lambda public: nearest_batch(
                    public, candidates_for(public), geo),
                continuation_policy_name=protocol["continuation_policy"],
                horizon_ticks=horizon,
                particles=protocol["particles"],
                sampling_seed=protocol["sampling_seed"] + spec["seed"] * 101 + anchor)
            elapsed = time.process_time() - started
            calls += horizon * protocol["particles"]
            if calls > protocol["max_total_rollout_steps"]:
                raise AssertionError("fixed per-group rollout budget exceeded")
            for h in available_h:
                metric = score(forecast, truth_states[anchor + h], h)
                outputs.append({
                    "anchor": anchor, "horizon": h, "metrics": metric,
                    "model_cpu_seconds_full_forecast": elapsed,
                    "factual_tick": anchor + h,
                    "factual_sample_ids": sorted(s.id for s in truth_states[anchor + h].samples),
                    "predicted_sample_ids": sorted(forecast.distributions[h - 1].sample_status_probabilities),
                })
        if not outputs:
            raise AssertionError("no eligible anchors in frozen evaluation group")
        arms[kind] = {
            "rows": outputs,
            "rollout_steps": calls,
            "query_calls": getattr(model.transition.model, "query_calls", 0),
        }
    return {
        "id": spec["id"], "split": spec["split"], "seed": spec["seed"],
        "truth_status": result["summary"]["status"],
        "truth_ticks": result["summary"]["ticks"],
        "true_episode_sha256": hashlib.sha256(
            json.dumps(result["events"], sort_keys=True).encode()).hexdigest(),
        "arms": arms,
    }


def aggregate(groups, horizons):
    """Equal weight per independent scenario, never count particles as N."""
    result = {}
    for h in horizons:
        rows = {}
        for arm in ("rule", "v4"):
            per_group = []
            for group in groups:
                selected = [r["metrics"] for r in group["arms"][arm]["rows"]
                            if r["horizon"] == h]
                if not selected:
                    continue
                per_group.append({k: sum(v[k] for v in selected) / len(selected)
                                  for k in METRICS})
            if not per_group:
                raise AssertionError("missing group-level horizon evaluation")
            rows[arm] = {
                "scenario_groups": len(per_group),
                "metrics": {k: sum(g[k] for g in per_group) / len(per_group)
                            for k in METRICS},
            }
        result[str(h)] = {
            **rows,
            "v4_minus_rule": {
                k: rows["v4"]["metrics"][k] - rows["rule"]["metrics"][k]
                for k in METRICS
            },
        }
    return result


def evaluate(checkpoint, *, protocol=PROTOCOL, cpu_budget=300):
    config = json.loads(Path(protocol).read_text())
    if (config["protocol"] != "RHWM_SYSTEM_STATE_H1_H5_H10_20261111_V1" or
            config["horizons"] != [1, 5, 10] or
            config["particles"] != 3 or
            config["anchors"] != [0, 5, 12, 20, 30]):
        raise ValueError("frozen evaluation contract was modified")
    if file_sha(checkpoint) != config["checkpoint_sha256"]:
        raise ValueError("frozen V4 checkpoint SHA256 mismatch")
    if type(cpu_budget) is not int or cpu_budget < 1 or cpu_budget > config["max_cpu_process_seconds"]:
        raise ValueError("CPU budget cannot exceed the frozen limit")
    from torch import set_num_threads
    set_num_threads(1)
    start = time.process_time()
    groups = []
    for spec in config["groups"]:
        if time.process_time() - start > cpu_budget:
            raise TimeoutError("CPU budget exceeded before next independent group")
        groups.append(group_forecast(spec, config, checkpoint))
    splits = {}
    for split in sorted(set(g["split"] for g in groups)):
        members = [g for g in groups if g["split"] == split]
        splits[split] = {
            "group_count": len(members),
            "by_horizon": aggregate(members, config["horizons"]),
        }
    return {
        "protocol": config["protocol"],
        "protocol_sha256": file_sha(protocol),
        "checkpoint_sha256": file_sha(checkpoint),
        "world_model_versions": ["rhwm-v1-rule", "rhwm-v1-hybrid-v4"],
        "groups": groups,
        "summary": {
            "by_horizon": aggregate(groups, config["horizons"]),
            "splits": splits,
            "groups": len(groups),
            "forecasts": sum(len(a["rows"]) for g in groups for a in g["arms"].values()),
            "model_steps": sum(a["rollout_steps"] for g in groups
                               for a in g["arms"].values()),
            "cpu_process_seconds": time.process_time() - start,
            "gpu": 0, "trainings": 0,
            "independent_unit": "synthetic scenario/seed group",
            "future_policy":"same public nearest_batch on actual/predicted states separately",
            "caveat": "One factual realization per synthetic group. Event/anchor/particle rows are dependent.",
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cpu-budget", type=int, default=300)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError("frozen holdout report is write-once")
    results = evaluate(args.checkpoint, cpu_budget=args.cpu_budget)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    summary = results["summary"]
    print(json.dumps({
        "groups": summary["groups"], "forecasts": summary["forecasts"],
        "model_steps": summary["model_steps"],
        "cpu_process_seconds": summary["cpu_process_seconds"],
        "by_horizon": summary["by_horizon"],
        "gpu": 0, "trainings": 0,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
