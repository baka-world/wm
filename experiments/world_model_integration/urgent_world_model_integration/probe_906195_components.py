"""PR18 isolated CPU probe: whether frozen V4 decisions depend on the terminal value proxy.

Post-hoc mechanism audit for 906195:138 only. Replays the archived P5 prefix,
checks the original V4 particle scores, and keeps every candidate, model weight,
random key and H5 transition model fixed when varying terminal computation.
No training, no GPU, no production scheduler changes. Not a blind evaluation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import torch

from business_transport.observation import capacity_view
from business_transport.session_events import eligible_sessions
from business_transport.state_features import build_business_features
from lab_transport.business_session_v3 import encode_rows
from experiments.world_model_integration.urgent_workflow_feasibility.core import PROTOCOL, public_spec
from experiments.world_model_integration.urgent_world_model_integration.core import UrgentWorldPolicy
from experiments.world_model_integration.urgent_world_model_integration.fixed_state_experiment import (
    PROBE_PROTOCOL, _safe_scores, read_archive, replay_one, sha,
)

TARGET_SEED = 906195
TARGET_TICK = 138
TARGET_OBS = "1f3573019d375d8457c8ce7d3eee079c3f45ab2aa220e1c9570463ff3858f1ab"
MODES = ("optimistic", "budgeted_queue", "rule_continuation")
ARMS = ("P4", "P5")


def candidate_queries(observation, plans, geometry, rules, checkpoint, config):
    """V4 p(y=finished | public current eligible session, candidate); no truth labels."""
    policy = UrgentWorldPolicy(
        "P5", geometry, rules, config, checkpoint, seed=TARGET_SEED,
        random_coupling="keyed", terminal_mode="budgeted_queue")
    model = policy.prediction.model
    output = []
    for index, plan in enumerate(plans):
        prepared, _ = capacity_view(observation, plan, rules)
        viable = [q for q, reason in eligible_sessions(prepared, plan) if reason is None]
        probabilities = []
        if viable:
            features = build_business_features(prepared, plan, geometry, rules)
            rows = [{"features": features, "query": q} for q in viable]
            with torch.inference_mode():
                tokens, masks, queries = encode_rows(rows, model.query_input_schema)
                probabilities = torch.sigmoid(model.network(tokens, masks, queries)).tolist()
        output.append({
            "candidate": index,
            "eligible_service_queries": len(viable),
            "probabilities": [
                {"query": q, "v4_completion_probability": float(p)}
                for q, p in zip(viable, probabilities)
            ],
            "label_available": False,
            "note": "These are current eligible query probabilities, not a Brier/calibration test."
        })
    return output


def score_candidates(arm, mode, seed, observation, plans, geometry, rules, config, checkpoint):
    policy = UrgentWorldPolicy(
        arm, geometry, rules, config, checkpoint if arm == "P5" else None,
        seed=seed, random_coupling="keyed", terminal_mode=mode, evaluation_mode="full")
    policy._synchronize_public_rules(rules)
    scores = []
    particles = []
    for i, plan in enumerate(plans):
        score, samples = policy._score(observation, plan, rules, i)
        scores.append(score)
        particles.append(samples)
    rank = sorted(range(len(scores)), key=lambda i: scores[i])
    first_differing = next(
        (j for j, (a, b) in enumerate(zip(scores[rank[0]], scores[rank[1]]))
         if a != b), None)
    return {
        "arm": arm, "terminal_mode": mode, "sampling_seed": seed,
        "choice": rank[0], "rank": rank,
        "scores": scores, "particles": particles,
        "first_differing_component": first_differing,
        "winning_margin": (scores[rank[1]][first_differing] -
                           scores[rank[0]][first_differing]
                           if first_differing is not None else None),
        "service_queries": policy.query_calls,
        "predicted_steps": policy.predict_steps,
        "terminal_steps": policy.terminal_steps,
    }


def run(archive, checkpoint, output, cpu_budget):
    start = time.process_time()
    protocol = json.loads(PROBE_PROTOCOL.read_text())
    config = json.loads(PROTOCOL.read_text())
    if sha(checkpoint.read_bytes()) != protocol["checkpoint_sha256"]:
        raise AssertionError("CHECKPOINT_SHA_DRIFT")
    item = next(
        x for x in protocol["fixed_states"]
        if x["seed"] == TARGET_SEED and x["tick"] == TARGET_TICK)
    if item["observation_sha256"] != TARGET_OBS or item["candidate_count"] != 2:
        raise AssertionError("TARGET_STATE_PROTOCOL_DRIFT")
    cases = read_archive(archive, protocol["archives"]["new_full"]["sha256"], "new-full")
    original = cases[(TARGET_SEED, "P5")]
    if original["index"] != item["index"]:
        raise AssertionError("TARGET_SCENE_DRIFT")
    record = next(x for x in original["decisions"] if x["tick"] == TARGET_TICK)
    if record["observation_sha256"] != TARGET_OBS:
        raise AssertionError("ORIGINAL_OBSERVATION_DRIFT")
    spec = public_spec(item["index"], TARGET_SEED, config["scenario_max_ticks"])
    _, captured = replay_one(
        spec, config, original, TARGET_TICK,
        int(record["selected_candidate_index"]),
        verify_candidate_list=True, retain_state=True)
    observation, rules, geometry, plans = (
        captured["observation"], captured["rules"],
        captured["geometry"], captured["plans"])
    if len(plans) != 2:
        raise AssertionError("TARGET_CANDIDATES_DRIFT")

    seeds = [TARGET_SEED, *protocol["independent_rng_seeds"]]
    results = []
    for mode in MODES:
        for seed in seeds:
            for arm in ARMS:
                result = score_candidates(
                    arm, mode, seed, observation, plans, geometry, rules, config, checkpoint)
                if arm == "P5" and mode == "budgeted_queue" and seed == TARGET_SEED:
                    for index, score in enumerate(result["scores"]):
                        old = record["plans"][index]
                        if not _safe_scores(score, old["score"]):
                            raise AssertionError("ORIGINAL_V4_SCORE_DRIFT")
                        if len(result["particles"][index]) != len(old["particle_costs"]):
                            raise AssertionError("ORIGINAL_V4_PARTICLE_COUNT_DRIFT")
                        for now, past in zip(result["particles"][index], old["particle_costs"]):
                            if not _safe_scores(now, past):
                                raise AssertionError("ORIGINAL_V4_PARTICLES_DRIFT")
                    if result["choice"] != record["selected_candidate_index"]:
                        raise AssertionError("ORIGINAL_V4_ACTION_DRIFT")
                results.append(result)
                if time.process_time() - start > cpu_budget:
                    raise TimeoutError("CPU_PROCESS_BUDGET_EXCEEDED")
    summary = {}
    for mode in MODES:
        summary[mode] = {}
        for arm in ARMS:
            rows = [r for r in results if r["terminal_mode"] == mode and r["arm"] == arm]
            counts = Counter(r["choice"] for r in rows)
            summary[mode][arm] = {
                "choice_counts": {str(i): counts[i] for i in range(len(plans))},
                "service_queries": sum(r["service_queries"] for r in rows),
                "terminal_steps": sum(r["terminal_steps"] for r in rows),
            }
    base_v4 = summary["budgeted_queue"]["P5"]["choice_counts"]
    if base_v4 != {"0": 8, "1": 0}:
        raise AssertionError("EXISTING_8_SEED_STABILITY_NOT_REPRODUCED")
    result = {
        "protocol": "PR18_906195_FIXED_H5_TERMINAL_DECOMPOSITION_V1",
        "target": {"seed": TARGET_SEED, "tick": TARGET_TICK,
                   "observation_sha256": TARGET_OBS, "candidates": len(plans)},
        "checkpoint_sha256": sha(checkpoint.read_bytes()),
        "source_archive_sha256": sha(archive.read_bytes()),
        "seeds": seeds, "arms": list(ARMS), "terminal_modes": list(MODES),
        "current_observable_query_probabilities": candidate_queries(
            observation, plans, geometry, rules, checkpoint, config),
        "scores": results, "summary": summary,
        "cpu_process_seconds": time.process_time() - start,
        "gpu": 0, "training": 0,
        "inference_scope": "frozen V4 checkpoint, selected known synthetic state, CPU",
        "limitation": [
            "post-hoc selected state, no blind/generalization claim",
            "terminal-mode change alters a proxy, not the real simulator dynamics",
            "P4/P5 have different H5 transitions, not only different terminal scores",
            "single synthetic true future does not identify expected action value",
            "reported current-query probabilities have no paired true event labels",
            "this experiment cannot isolate prediction calibration error from workflow transition error"
        ]
    }
    if output.exists():
        raise FileExistsError("WRITE_ONCE_OUTPUT_ALREADY_EXISTS")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"summary": summary,
        "current_observable_query_probabilities": result["current_observable_query_probabilities"],
        "cpu_process_seconds": result["cpu_process_seconds"]}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-new-full", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-budget", type=float, default=300.0)
    args = parser.parse_args()
    if torch.version.cuda is not None:
        raise AssertionError("CPU_ONLY_REQUIRED")
    torch.set_num_threads(1)
    run(args.archive_new_full, args.checkpoint, args.output, args.cpu_budget)


if __name__ == "__main__":
    main()
