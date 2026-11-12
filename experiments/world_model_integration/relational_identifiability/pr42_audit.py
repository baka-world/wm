"""PR42: public action-relation identifiability audit for frozen V4-H20.

This is a DEVELOPMENT diagnostic, using previously tested PR30 synthetic seeds.
No retraining, policy changes, unknown future events, or oracle feature inputs.
Factual counterfactual branches are used only as offline labels AFTER forecasts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from business_transport.session_events import eligible_sessions
from business_transport.session_features_v3 import encode_query_v3
from business_transport.state_features import build_business_features
from experiments.world_model_integration.horizon_generalization.pr29_evaluate import (
    forecast_pair, paired_truth,
)
from experiments.world_model_integration.online_horizon.pr30_run import PROTOCOL as PR30
from experiments.world_model_integration.system_world_model.evaluate_horizons import (
    file_sha, score,
)
from experiments.world_model_integration.train_v4 import load_v4
from lab_transport.business_state_learning import encode_tokens

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "relation_protocol.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sorted_token_multiset(tokens):
    """Compare sets exactly, not their incidental action or entity row order."""
    return tuple(sorted(tuple(float(v) for v in row) for row in tokens.tolist()))


def candidate_links(state, plan):
    """Ground-truth PUBLIC action joins, not private simulator outcomes."""
    samples = {s.id: s for s in state.samples}
    return tuple(sorted(
        (a.robot_id, a.sample_id, samples[a.sample_id].pickup,
         samples[a.sample_id].dropoff) for a in plan
    ))


def network_view(model, features, state, plan):
    """Record precisely what the frozen network sees and its global embedding.

    Per-session probabilities are only defined for eligible, already-started
    services. The absence of a service query is NOT a zero probability.
    """
    tokens = encode_tokens(features)
    with torch.inference_mode():
        pooled = model.entity(tokens).mean(0)
        probs = {}
        for query, reason in eligible_sessions(state, plan):
            if reason is not None:
                continue
            key = (query["robot_id"], query["sample_id"],
                   query["station_id"], query["phase"])
            q = torch.tensor([encode_query_v3(features, query)],
                             dtype=torch.float32)
            logits = model(tokens.unsqueeze(0),
                           torch.ones((1, len(tokens))), q)
            probs["|".join(key)] = float(torch.sigmoid(logits)[0])
    return {
        "tokens": sorted_token_multiset(tokens),
        "context": tuple(float(x) for x in pooled.tolist()),
        "eligible_event_probabilities": probs,
    }


def compare_views(left, right, left_links, right_links):
    """Separate raw relation identity, numeric tokens, embedding, and logits."""
    if left_links == right_links:
        raise ValueError("two candidate plans do not differ in public relations")
    keys = sorted(left["eligible_event_probabilities"].keys() &
                  right["eligible_event_probabilities"].keys())
    return {
        "raw_relationships_different": True,
        "numeric_token_multisets_equal": left["tokens"] == right["tokens"],
        "pooled_context_linf": max(
            abs(a-b) for a, b in zip(left["context"], right["context"])),
        "same_query_count": len(keys),
        "shared_event_probability_linf": (
            max(abs(left["eligible_event_probabilities"][k] -
                    right["eligible_event_probabilities"][k]) for k in keys)
            if keys else None
        ),
        "no_eligible_shared_query": not bool(keys),
    }


def decision_config(cfg):
    solver = {"limit": 4, "wall_seconds": 0.08,
              "deterministic_seconds": 0.04, "seed": 1810}
    return {"max_ticks": cfg["max_ticks"],
            "first_action_pair": {"solver": solver},
            "prediction": {"particles": 3}}


def frozen_groups(cfg):
    original = json.loads(PR30.read_text())
    if original["protocol"] != cfg["source_protocol"]:
        raise ValueError("source PR30 protocol no longer matches")
    groups = {g["id"]: g for g in original["groups"]}
    ids = cfg["source_case_ids"]
    if len(ids) != len(set(ids)) or any(i not in groups for i in ids):
        raise ValueError("invalid or duplicate frozen PR30 ids")
    if len(ids) != 8 or any(
        sum(groups[i]["regime"] == regime for i in ids) != 2
        for regime in ("light", "medium", "heavy", "shift")
    ):
        raise ValueError("frozen development diagnostic requires 2 per regime")
    return [groups[i] for i in ids]


def run_one(spec, cfg, checkpoint, network):
    factual = paired_truth(spec, decision_config(cfg))
    first = factual[0]
    state = first["first_state"]
    plans = first["first_plans"][:2]
    if len(factual) != len(plans):
        raise AssertionError("paired truth did not match evaluated candidates")
    features = [build_business_features(
        state, plan, first["geometry"], first["business"].rules)
        for plan in plans]
    links = [candidate_links(state, plan) for plan in plans]
    views = [network_view(network, f, state, p)
             for f, p in zip(features, plans)]
    comparison = (compare_views(views[0], views[1], links[0], links[1])
                  if len(views) == 2 else None)

    start = time.process_time()
    predictions = forecast_pair(
        first, decision_config(cfg), checkpoint, arm="v4", horizon=20)
    prediction_cpu = time.process_time() - start
    if len(predictions) != len(factual):
        raise AssertionError("H20 candidate prediction count differs")

    rows = []
    for i, (branch, forecast) in enumerate(zip(factual, predictions)):
        truth = branch["public_observations"].get(20)
        # Early termination is missing observation, not invented future truth.
        error = score(forecast["forecast"], truth, 20) if truth is not None else None
        dist = forecast["forecast"].distributions[-1]
        first_events = forecast["forecast"].particles[0].steps[0].service_events
        rows.append({
            "candidate_index": i,
            "public_relation_edges": [list(edge) for edge in links[i]],
            "public_feature_sha256": hashlib.sha256(
                json.dumps(features[i], sort_keys=True, default=str).encode()
            ).hexdigest(),
            "encoded_tokens_sha256": hashlib.sha256(
                repr(views[i]["tokens"]).encode()
            ).hexdigest(),
            "eligible_initial_service_queries": len(
                views[i]["eligible_event_probabilities"]),
            "first_step_service_events": [
                {"robot": e.robot_id, "sample": e.sample_id,
                 "station": e.station_id, "p_finish": e.p_finish_next}
                for e in first_events
            ],
            "predicted_h20": {
                "completed": float(dist.expected_completed_samples),
                "waiting": float(dist.expected_waiting_samples),
                "in_transit": float(dist.expected_in_transit_samples),
                "resource_occupancy_mean": dict(dist.resource_occupancy_mean),
            },
            "h20_state_error_when_truth_observed": error,
            "truth_tick20_available": truth is not None,
            "factual_outcome_offline_only": {
                "makespan_ticks": branch["outcomes"]["end_tick"],
                "late_jobs": branch["outcomes"]["late_jobs"],
                "completed_jobs": branch["outcomes"]["completed_jobs"],
                "censored": branch["outcomes"]["censored"],
                "resource_wait_ticks": branch["outcomes"]["resource_wait_ticks"],
                "safety_violations": branch["outcomes"]["safety_violations"],
            },
        })
    complete = all(not row["factual_outcome_offline_only"]["censored"]
                   for row in rows)
    pred_delta = (rows[1]["predicted_h20"]["completed"] -
                  rows[0]["predicted_h20"]["completed"]
                  if len(rows) == 2 else None)
    true_delta = (rows[1]["factual_outcome_offline_only"]["makespan_ticks"] -
                  rows[0]["factual_outcome_offline_only"]["makespan_ticks"]
                  if len(rows) == 2 and complete else None)
    return {
        "id": spec["id"], "seed": spec["seed"], "regime": spec["regime"],
        "initial_public_tick": state.tick,
        "distinct_legal_candidates": len(plans),
        "representation_comparison": comparison,
        "predicted_completed_second_minus_first": pred_delta,
        "factual_makespan_second_minus_first": true_delta,
        "candidate_predictions_tied": (
            abs(pred_delta) <= 1e-9 if pred_delta is not None else None),
        "true_makespan_different": (
            abs(true_delta) > 1e-9 if true_delta is not None else None),
        "h20_forecast_cpu_seconds": prediction_cpu,
        "candidates": rows,
    }


def summarize(rows):
    eligible = [r for r in rows if r["representation_comparison"] is not None]
    misses = [r for r in eligible if r["true_makespan_different"]]
    return {
        "cases": len(rows),
        "two_candidate_cases": len(eligible),
        "different_raw_relationships": len(eligible),
        "exact_numeric_token_collisions": sum(
            r["representation_comparison"]["numeric_token_multisets_equal"]
            for r in eligible),
        "pool_context_identical_1e_minus_7": sum(
            r["representation_comparison"]["pooled_context_linf"] <= 1e-7
            for r in eligible),
        "shared_query_coverage": sum(
            r["representation_comparison"]["same_query_count"] for r in eligible),
        "h20_completed_ties": sum(r["candidate_predictions_tied"] for r in eligible),
        "factual_sensitive_cases": len(misses),
        "sensitive_but_h20_tied": sum(
            r["candidate_predictions_tied"] for r in misses),
        "per_regime": {
            regime: {
                "cases": sum(r["regime"] == regime for r in rows),
                "two_candidate": sum(r["regime"] == regime for r in eligible),
                "token_collision": sum(
                    r["regime"] == regime and
                    r["representation_comparison"]["numeric_token_multisets_equal"]
                    for r in eligible),
            } for regime in ("light", "medium", "heavy", "shift")
        },
        "promotion": "NONE: development-only mechanism audit",
        "gpu": 0, "new_training": 0,
    }


def run(checkpoint, output):
    cfg = json.loads(PROTOCOL.read_text())
    if cfg["protocol"] != "PR42_RELATION_IDENTIFIABILITY_DEVELOPMENT_V1":
        raise ValueError("not the frozen audit")
    if sha(checkpoint) != cfg["frozen_v4_sha256"]:
        raise ValueError("unrecognized original V4 checkpoint")
    if output.exists():
        raise FileExistsError("do not overwrite evidence")
    torch.set_num_threads(1)
    network, meta = load_v4(checkpoint, experimental=True)
    network.eval()
    rows = []
    started = time.process_time()
    for spec in frozen_groups(cfg):
        if time.process_time() - started > cfg["cpu_process_budget_seconds"]:
            raise TimeoutError("PR42 CPU diagnostic budget exceeded")
        rows.append(run_one(spec, cfg, checkpoint, network))
    report = {
        "protocol": cfg["protocol"],
        "evidence_scope": "reused_PR30_synthetic_development_not_independent",
        "base_source_ref": cfg["source_commit"],
        "source_pr30_sha256": sha(PR30),
        "protocol_sha256": sha(PROTOCOL),
        "frozen_v4_sha256": sha(checkpoint),
        "model_version": meta["model_version"],
        "original_scenarios_reused": True,
        "oracle_features_prohibited": True,
        "decision_policy_changed": False,
        "summary": summarize(rows),
        "groups": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report["summary"], ensure_ascii=False))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.checkpoint, args.out)


if __name__ == "__main__":
    main()
