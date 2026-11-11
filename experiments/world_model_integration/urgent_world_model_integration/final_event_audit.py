"""ONE final bounded PR18 CPU-only evidence run: factual service event predictions.

Five PRE-SELECTED known decision states, all 12 pre-frozen legal candidate futures.
No new scenarios, weights, stochastic seeds, strategy tuning, or GPU.
Only records p(finish at t+1 | public facts) vs the true paired transition.
These correlated synthetic futures do NOT provide a generalization estimate.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time
from zipfile import ZipFile

import torch

from business_transport.dispatch import BusinessCPSAT
from business_transport.dynamics import WorkflowDynamics
from business_transport.session_events import eligible_sessions
from business_transport.state_features import build_business_features
from experiments.business_constraints.run import lifecycle_audit
from experiments.world_model_integration.build_dataset import label_for
from experiments.world_model_integration.empirical_baselines import EmpiricalHazards
from experiments.world_model_integration.urgent_workflow_feasibility.core import (
    ConditionalRetestSimulator, IsolatedPolicy, PriorityObserver, PROTOCOL,
    digest, job_metrics, judge_score, public_spec, scene_for, unique_plans)
from experiments.world_model_integration.urgent_world_model_integration.fixed_state_experiment import (
    PROBE_PROTOCOL, _assignment, _matches, _stored_matches, read_archive, sha)
from business_transport.simulation import resource_audit
from lab_transport.multisite_experiment import discrete_safety_violations
from lab_transport.business_session_v3 import encode_rows
from lab_transport.torch_structured_adapter import ProgressAwareRule
from lab_transport.transition_schema import Assignment
from experiments.world_model_integration.train_v4 import load_v4


def replay_factual(item, original, config, choice):
    spec = public_spec(item["index"], item["seed"], config["scenario_max_ticks"])
    business, geo = scene_for(spec)
    sim = ConditionalRetestSimulator(
        business, spec["seed"], spec["retest_rate_percent"], collect_metrics=True)
    current = PriorityObserver()
    follow = IsolatedPolicy("P1", geo, config, spec["jitter"])
    archived = {int(d["tick"]): d for d in original["decisions"]}
    states, rules, transitions = {}, {}, []
    captured = {"seen": False}

    def on_observation(raw):
        state = current.observe(raw)
        follow.observer.observe(raw)
        states[state.tick] = state
        rules[state.tick] = sim.business.rules

    def on_transition(transition):
        transitions.append(transition)

    def dispatch(raw, legal):
        obs = states[raw.tick]
        t = obs.tick
        if t < item["tick"]:
            old = archived.get(t)
            if old is None or digest(asdict(obs)) != old["observation_sha256"]:
                raise AssertionError("PREFIX_OBSERVATION_DRIFT")
            chosen = _assignment(old["selected"])
        elif t == item["tick"]:
            old = archived.get(t)
            if old is None or digest(asdict(obs)) != item["observation_sha256"]:
                raise AssertionError("TARGET_OBSERVATION_DRIFT")
            solver = BusinessCPSAT(
                geo, sim.business.rules, candidate_limit=config["max_legal_candidates"],
                time_limit_seconds=config["cpsat_seconds"],
                deterministic_time_limit=config["deterministic_cpsat_seconds"], seed=1810)
            plans = unique_plans(solver.proposals(obs, legal), legal)
            if [_matches(p) for p in plans] != [
                _stored_matches(p["matching"]) for p in old["plans"]]:
                raise AssertionError("LEGAL_CANDIDATE_ORDER_DRIFT")
            if len(plans) != item["candidate_count"] or not 0 <= choice < len(plans):
                raise AssertionError("CANDIDATE_COUNT_DRIFT")
            chosen = plans[choice]
            captured["seen"] = True
        else:
            return follow(raw, legal, sim)
        if any(a not in legal for a in chosen):
            raise AssertionError("ILLEGAL_FORCED_ACTION")
        return chosen

    result = sim.run(
        dispatch_policy=dispatch, on_observation=on_observation,
        on_transition=on_transition)
    if not captured["seen"]:
        raise AssertionError("NEVER_REACHED_TARGET")
    if resource_audit(result, sim.business.rules)["violations"]:
        raise AssertionError("CAPACITY_VIOLATION")
    if discrete_safety_violations(result) or lifecycle_audit(result["history"]):
        raise AssertionError("SIMULATOR_SAFETY_VIOLATION")

    rows, exclusions = [], Counter()
    for transition in transitions:
        tick = transition.observation.tick
        if tick < item["tick"]:
            continue
        if tick not in states or tick + 1 not in states:
            exclusions["terminal_transition_without_next_public_observation"] += 1
            continue
        before, after = states[tick], states[tick+1]
        if after.tick != before.tick + 1:
            raise AssertionError("NONADJACENT_FACTUAL_TRANSITION")
        assignments = tuple(transition.assignments)
        for query, reason in eligible_sessions(before, assignments):
            if reason is not None:
                exclusions["not_eligible_" + reason] += 1
                continue
            label, label_reason = label_for(before, after, query)
            if label is None:
                exclusions["invalid_label_" + label_reason] += 1
                continue
            features = build_business_features(before, assignments, geo, rules[tick])
            rows.append({
                "seed": item["seed"], "candidate": choice, "tick": tick,
                "query": query, "features": features, "label": label,
                "before": before, "assignments": assignments, "rules": rules[tick]})
    return {
        "rows": rows, "exclusions": dict(exclusions),
        "metrics": job_metrics(result, sim.business, config["scenario_max_ticks"]),
        "events_sha256": digest(result["events"]),
        "history_sha256": digest(result["history"]),
        "target_observation_sha256": item["observation_sha256"],
        "candidate": choice,
        "seed": item["seed"],
        "target_tick": item["tick"],
    }, geo


def evaluate(model, schema, b2, replay, geo):
    rows = replay["rows"]
    result = []
    if not rows:
        return result
    encoded = [{"features": r["features"], "query": r["query"]} for r in rows]
    # Crucially: model input is built from before(t), never after(t+1).
    with torch.inference_mode():
        tokens, masks, queries = encode_rows(encoded, schema)
        ps = torch.sigmoid(model(tokens, masks, queries)).tolist()
    if len(ps) != len(rows):
        raise AssertionError("INFERENCE_ROW_COUNT_DRIFT")
    for row, p in zip(rows, ps):
        rule = WorkflowDynamics(ProgressAwareRule(geo), row["rules"])
        predicted_next = rule.predict_next(row["before"], row["assignments"])
        rule_label, reason = label_for(row["before"], predicted_next, row["query"])
        if rule_label is None:
            raise AssertionError("RULE_PREDICTION_UNLABELABLE: " + reason)
        data = {"features": row["features"], "query": row["query"]}
        p2 = b2.predict_one(data, "B2")
        for v in (p, p2, rule_label):
            if not math.isfinite(v) or not 0 <= v <= 1:
                raise AssertionError("INVALID_PREDICTION_PROBABILITY")
        result.append({
            "seed": row["seed"], "candidate": row["candidate"],
            "tick": row["tick"], "query": row["query"],
            "true_finished_next_tick": row["label"],
            "probabilities": {"V4": p, "B2": p2, "Rule": float(rule_label)},
            "observed_active_elapsed": next(
                r.service_active_elapsed for r in row["before"].robots
                if r.id == row["query"]["robot_id"]),
            "station_public_expected_ticks": next(
                s.expected_service_ticks for s in row["before"].stations
                if s.id == row["query"]["station_id"]),
        })
    return result


def summarize(rows):
    methods = ("V4", "B2", "Rule")
    if not rows:
        return {"n": 0}
    groups = {}
    for name in methods:
        groups[name] = {
            "brier": sum((r["probabilities"][name] -
                          r["true_finished_next_tick"]) ** 2 for r in rows) / len(rows),
            "mean_probability": sum(r["probabilities"][name] for r in rows) / len(rows),
        }
    return {
        "n": len(rows),
        "positives": sum(r["true_finished_next_tick"] for r in rows),
        "predictors": groups,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmation-zip", type=Path, required=True)
    parser.add_argument("--new-full-zip", type=Path, required=True)
    parser.add_argument("--v4-archive", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=float, default=180.0)
    a = parser.parse_args()
    if torch.version.cuda is not None:
        raise AssertionError("CPU_ONLY_REQUIRED")
    torch.set_num_threads(1)
    start = time.process_time()
    if a.output.exists():
        raise FileExistsError("FROZEN_EVIDENCE_OUTPUT_IS_WRITE_ONCE")
    protocol = json.loads(PROBE_PROTOCOL.read_text())
    config = json.loads(PROTOCOL.read_text())
    if sha(a.checkpoint.read_bytes()) != protocol["checkpoint_sha256"]:
        raise AssertionError("FROZEN_V4_CHECKPOINT_DRIFT")
    with ZipFile(a.v4_archive) as archive:
        development = json.loads(archive.read("v4-development.json"))
    from experiments.world_model_integration.build_dataset_v4 import CONFIG
    if development["source_config_sha256"] != sha(CONFIG.read_bytes()):
        raise AssertionError("V4_TRAIN_DATA_PROTOCOL_DRIFT")
    if development["part"] != "development":
        raise AssertionError("TRAINING_ONLY_FIT_REQUIRED")
    b2 = EmpiricalHazards(development["train"])
    model, metadata = load_v4(a.checkpoint, experimental=True)
    if metadata["input_schema"] != "business-service-query-input-v3":
        raise AssertionError("NETWORK_FEATURE_SCHEMA_DRIFT")
    archives = {
        "confirmation": read_archive(
            a.confirmation_zip, protocol["archives"]["confirmation"]["sha256"],
            "confirmation"),
        "new_full": read_archive(
            a.new_full_zip, protocol["archives"]["new_full"]["sha256"],
            "new-full"),
    }
    records, events = [], []
    for item in protocol["fixed_states"]:
        factual = archives[item["source"]][(item["seed"], "P5")]
        if factual["index"] != item["index"]:
            raise AssertionError("SOURCE_SCENARIO_DRIFT")
        for choice in range(item["candidate_count"]):
            true, geo = replay_factual(item, factual, config, choice)
            predicted = evaluate(model, metadata["input_schema"], b2, true, geo)
            events.extend(predicted)
            records.append({
                "seed": item["seed"], "tick": item["tick"],
                "candidate": choice, "factual_event_queries": len(predicted),
                "summary": summarize(predicted),
                "exclusions": true["exclusions"],
                "judge_score": judge_score(true["metrics"]),
                "events_sha256": true["events_sha256"],
                "history_sha256": true["history_sha256"],
            })
            if time.process_time() - start > a.budget_seconds:
                raise TimeoutError("ONE_FINAL_AUDIT_CPU_LIMIT")
    if len(records) != 12:
        raise AssertionError("FROZEN_CANDIDATE_SCOPE_CHANGED")
    by_seed = {}
    for item in protocol["fixed_states"]:
        subset = [r for r in events if r["seed"] == item["seed"]]
        by_seed[str(item["seed"])] = summarize(subset)
    result = {
        "protocol": "PR18_FINAL_NO_MORE_MECHANISM_PROBES_V1",
        "scope": "Five post-hoc synthetic states, 12 already-frozen candidate P1 futures",
        "model_checkpoint_sha256": sha(a.checkpoint.read_bytes()),
        "original_archives_sha256": {
            "confirmation": sha(a.confirmation_zip.read_bytes()),
            "new_full": sha(a.new_full_zip.read_bytes())},
        "training_artifact_sha256": sha(a.v4_archive.read_bytes()),
        "models": {
            "V4": "frozen CPU deep-sets conditional service hazard",
            "Rule": "deterministic one-step business Rule",
            "B2": "fitted once on original V4 Train split, never on these futures",
        },
        "aggregate_descriptive_only": summarize(events),
        "by_synthetic_seed": by_seed,
        "candidate_futures": records,
        "factual_transition_rows": events,
        "cpu_process_seconds": time.process_time() - start,
        "training_runs": 0, "gpu": 0,
        "decision": "NO_MORE_MICRO_AUDITS; switch to system engineering",
        "limitations": [
            "posthoc picked states; no independent confirmation",
            "12 branch continuations share the same five world seeds and prior histories",
            "repeated event rows across paired futures are correlated",
            "a Brier difference here is diagnostic only and not evidence of deployment value",
            "V4/B2 source training distribution differs from urgent workflow states",
            "this cannot identify 5-step causal error contribution or conditional expected regret",
            "no retraining, no selection or new action policy is justified by these data",
        ],
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    print(json.dumps({
        "aggregate_descriptive_only": result["aggregate_descriptive_only"],
        "by_synthetic_seed": result["by_synthetic_seed"],
        "futures": len(records), "rows": len(events),
        "cpu_process_seconds": result["cpu_process_seconds"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
