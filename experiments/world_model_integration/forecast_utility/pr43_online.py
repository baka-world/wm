"""PR43: trajectory-state utility transfer into frozen RHWM-H20 online control.

No network/weights/solver/simulator modifications. Original PR30 policy is the
baseline; PR27's published-in-repo particle trajectory cost is reused verbatim.
All online decisions are made before reading any factual future outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from business_transport.observation import BusinessObserver
from business_transport.dispatch import BusinessCPSAT
from business_transport.simulation import BusinessSimulator, resource_audit
from business_transport.world_model import HybridSystemWorldModel
from experiments.business_constraints.run import lifecycle_audit
from experiments.world_model_integration.dynamic_identifiability.pr28_experiment import build_business
from experiments.world_model_integration.online_horizon.pr30_run import (
    OnlineHorizonPolicy, _facts, _unique_plans,
)
from experiments.world_model_integration.decision_value.pr27_evaluate import _trace_cost
from experiments.world_model_integration.system_world_model.evaluate_horizons import file_sha
from lab_transport.decision_value import candidates_for
from lab_transport.multisite_experiment import discrete_safety_violations

PROTOCOL = Path(__file__).with_name("pr43_protocol.json")
ARMS = ("cp_sat", "v4_completed", "v4_tie_trajectory", "v4_trajectory")
REPO_ROOT = Path(__file__).resolve().parents[3]


def select_scores(scores, method):
    """No fitted weights. Identical score ties prefer first CP-SAT proposal."""
    if not scores:
        raise ValueError("cannot select from empty candidate scores")
    if method == "v4_completed":
        best = max(row["completed"] for row in scores)
        return next(i for i, row in enumerate(scores)
                    if best - row["completed"] <= 1e-9)
    if method == "v4_tie_trajectory":
        best = max(row["completed"] for row in scores)
        eligible = [i for i, row in enumerate(scores)
                    if best - row["completed"] <= 1e-9]
        return min(eligible, key=lambda i: (
            scores[i]["unfinished_late_cost"], scores[i]["terminal_distance"], i))
    if method == "v4_trajectory":
        return min(range(len(scores)), key=lambda i: (
            scores[i]["unfinished_late_cost"], scores[i]["terminal_distance"], i))
    raise ValueError("unknown frozen score method")


def score_batch(initial, batch, geometry):
    """Existing PR30 end-state signal + existing PR27 PETS-style trace utility."""
    progress = float(batch.distributions[-1].expected_completed_samples)
    unfinished, terminal = _trace_cost(initial, batch, geometry)
    return {
        "completed": progress,
        "unfinished_late_cost": float(unfinished),
        "terminal_distance": float(terminal),
    }


def tied_completed(scores):
    return len(scores) > 1 and any(
        abs(a["completed"] - b["completed"]) <= 1e-9
        for i, a in enumerate(scores) for b in scores[i+1:])


def h20_state_distinct(forecasts):
    """All modeled distribution fields, not only final completed count."""
    return len(forecasts) > 1 and any(
        f.distributions[-1] != forecasts[0].distributions[-1]
        for f in forecasts[1:])


class StateAwarePolicy(OnlineHorizonPolicy):
    """Adapt PR30 policy only at the action-value readout, not world dynamics."""

    def __init__(self, arm, geometry, rules, protocol, *, checkpoint, seed):
        if arm not in ("v4_tie_trajectory", "v4_trajectory"):
            raise ValueError("state-aware arm required")
        super().__init__("v4_h20", geometry, rules, protocol,
                         checkpoint=checkpoint, seed=seed)
        self.arm = arm
        self.n_tie_decisions = 0
        self.n_tied_but_state_distinct = 0
        self.n_tie_resolved = 0

    def __call__(self, raw, legal_candidates):
        started = time.process_time()
        observed = self.observer.observe(raw)
        plans = _unique_plans(self.solver, observed, legal_candidates)
        self.n_calls += 1
        self.n_multi += len(plans) > 1
        chosen = plans[0]
        scores = []
        models = []
        timed_out = False
        fallback_reason = None
        if len(plans) > 1:
            seed = 1810 + self.seed*17 + observed.tick*101
            # Identical continuation algorithm applied only to predicted state.
            continuation = lambda state: _unique_plans(
                self.forecast_solver, state, candidates_for(state))[0]
            for plan in plans:
                if time.process_time()-started >= self.protocol["decision_cpu_budget_seconds"]:
                    timed_out = True
                    fallback_reason = "before_candidate"
                    break
                try:
                    batch = self.model.rollout(
                        observed, plan, continuation_policy=continuation,
                        continuation_policy_name="CP_SAT_predicted_state",
                        horizon_ticks=self.horizon,
                        particles=self.protocol["particles"],
                        sampling_seed=seed)
                    val = score_batch(observed, batch, self.geometry)
                    scores.append(val)
                    models.append(batch)
                    self.n_model_queries += 1
                    self.n_model_steps += self.horizon*self.protocol["particles"]
                    if time.process_time()-started >= self.protocol["decision_cpu_budget_seconds"]:
                        timed_out = True
                        fallback_reason = "after_candidate"
                        break
                except (RuntimeError, ValueError, AssertionError) as exc:
                    self.n_errors += 1
                    fallback_reason = "prediction_error: " + str(exc)[:160]
                    break

            # Ranking is atomic: never act on a partial set of proposals.
            if len(scores) == len(plans) and not timed_out:
                if tied_completed(scores):
                    self.n_tie_decisions += 1
                    self.n_tied_but_state_distinct += int(h20_state_distinct(models))
                idx = select_scores(scores, self.arm)
                original_idx = select_scores(scores, "v4_completed")
                chosen = plans[idx]
                if self.arm == "v4_tie_trajectory" and idx != original_idx:
                    self.n_tie_resolved += 1
            else:
                self.n_fallback += 1
                if fallback_reason is None:
                    fallback_reason = "incomplete_candidate_batch"
        # Count every exceeded CPU-process limit, including post-scoring overhead.
        elapsed = time.process_time() - started
        if elapsed >= self.protocol["decision_cpu_budget_seconds"] and len(plans)>1:
            self.n_budget_exceeded += 1
            if chosen != plans[0]:
                chosen = plans[0]
                self.n_fallback += 1
                fallback_reason = "post_ranking_budget"
        self.n_flip += int(chosen != plans[0])
        elapsed = time.process_time() - started
        self.cpu_decision_seconds += elapsed
        self.decisions.append({
            "tick": observed.tick, "legal_plans": len(plans),
            "selected_index": plans.index(chosen),
            "first_proposal": str(plans[0]),
            "selected": str(chosen),
            "score_details": scores,
            "completed_tie": tied_completed(scores) if len(scores)==len(plans) else None,
            "future_state_different": h20_state_distinct(models) if len(models)==len(plans) else None,
            "fallback_reason": fallback_reason,
            "process_cpu_seconds": elapsed,
        })
        return chosen


def one(spec, proto, arm, checkpoint):
    business, geometry = build_business(spec, max_ticks=proto["max_ticks"])
    init_start = time.process_time()
    if arm in ("cp_sat", "v4_completed"):
        original_arm = "cp_sat" if arm == "cp_sat" else "v4_h20"
        policy = OnlineHorizonPolicy(original_arm, geometry, business.rules,
                                     proto, checkpoint=checkpoint, seed=spec["seed"])
    else:
        policy = StateAwarePolicy(arm, geometry, business.rules,
                                  proto, checkpoint=checkpoint, seed=spec["seed"])
    init_cpu = time.process_time() - init_start
    start = time.process_time()
    factual = BusinessSimulator(
        business, "world", spec["seed"], collect_metrics=True).run(
        dispatch_policy=policy, on_observation=policy.observe)
    execution_cpu = time.process_time() - start
    audit = resource_audit(factual, business.rules)
    motion = discrete_safety_violations(factual)
    lifecycle = lifecycle_audit(factual["history"])
    if audit["violations"] or motion or lifecycle:
        raise AssertionError("safety or lifecycle violation")
    return {
        "arm": arm, "regime": spec["regime"], "seed": spec["seed"],
        "status": factual["summary"]["status"],
        "makespan_ticks": factual["summary"]["ticks"],
        "job_metrics": _facts(factual, business),
        "resource_wait_ticks": audit["capacity_wait_ticks"],
        "blocked_moves": factual["summary"]["blocked_move_attempts"],
        "policy_calls": policy.n_calls, "multiple_candidates": policy.n_multi,
        "changed_plan_calls": policy.n_flip,
        "model_prediction_calls": policy.n_model_queries,
        "model_prediction_steps": policy.n_model_steps,
        "decision_budget_exceeded": policy.n_budget_exceeded,
        "fallbacks": policy.n_fallback, "prediction_errors": policy.n_errors,
        "model_init_cpu_seconds": init_cpu,
        "full_episode_cpu_seconds": init_cpu + execution_cpu,
        "policy_decision_cpu_seconds": policy.cpu_decision_seconds,
        "solver_fallbacks": policy.solver.fallbacks,
        "complete_score_tie_decisions": getattr(policy, "n_tie_decisions", None),
        "tied_but_state_distinct": getattr(policy, "n_tied_but_state_distinct", None),
        "tie_resolved_changed_action": getattr(policy, "n_tie_resolved", None),
        "safety_violations": 0,
        "factual_event_sha256": hashlib.sha256(
            json.dumps(factual["events"], sort_keys=True).encode()).hexdigest(),
        "decision_records": policy.decisions,
    }


def _mean(vals):
    return sum(vals) / len(vals) if vals else None


def summarize(groups):
    out = {}
    for regime in ("light", "medium", "heavy", "shift"):
        own = [g for g in groups if g["regime"] == regime]
        arms = {}
        paired = {}
        for arm in ARMS:
            rows = [g["arms"][arm] for g in own]
            arms[arm] = {
                "mean_makespan": _mean([v["makespan_ticks"] for v in rows]),
                "mean_late_jobs": _mean([v["job_metrics"]["late_jobs"] for v in rows]),
                "mean_wait_ticks": _mean([v["resource_wait_ticks"] for v in rows]),
                "mean_policy_cpu_seconds": _mean([v["policy_decision_cpu_seconds"] for v in rows]),
                "budget_exceeded": sum(v["decision_budget_exceeded"] for v in rows),
                "fallbacks": sum(v["fallbacks"] for v in rows),
                "action_flips": sum(v["changed_plan_calls"] for v in rows),
                "models_called": sum(v["model_prediction_calls"] for v in rows),
                "completed_episodes": sum(not v["job_metrics"]["censored"] for v in rows),
                "tie_decisions": sum(v["complete_score_tie_decisions"] or 0 for v in rows),
                "tie_but_state_distinct": sum(v["tied_but_state_distinct"] or 0 for v in rows),
                "tie_resolutions": sum(v["tie_resolved_changed_action"] or 0 for v in rows),
            }
            if arm != "v4_completed":
                diffs = [
                    g["arms"][arm]["makespan_ticks"] -
                    g["arms"]["v4_completed"]["makespan_ticks"] for g in own
                ]
                paired[arm] = {
                    "mean_delta_vs_v4_completed": _mean(diffs),
                    "wins": sum(d < 0 for d in diffs),
                    "ties": sum(d == 0 for d in diffs),
                    "losses": sum(d > 0 for d in diffs),
                    "late_jobs_delta": _mean([
                        g["arms"][arm]["job_metrics"]["late_jobs"] -
                        g["arms"]["v4_completed"]["job_metrics"]["late_jobs"]
                        for g in own]),
                }
        out[regime] = {"groups": len(own), "arms": arms,
                       "paired_vs_v4_completed": paired}
    return out


def check_protocol(proto, checkpoint):
    if (proto["protocol"] != "RHWM_PR43_FROZEN_FORECAST_TRAJECTORY_UTILITY_20261112_V1"
            or proto["arms"] != list(ARMS) or len(proto["groups"]) != 24
            or proto["horizon"] != 20 or proto["particles"] != 3
            or proto["candidate_limit"] != 4
            or proto["decision_cpu_budget_seconds"] != .5
            or proto["source_pr42_sha"] !=
            "f2fd9c29079aba8eadd09bb478bc6b8f2c3ff1a6"):
        raise ValueError("changed PR43 precommitted evaluation protocol")
    if file_sha(checkpoint) != proto["frozen_v4_sha256"]:
        raise ValueError("original V4 checkpoint SHA changed")
    if len(set(x["seed"] for x in proto["groups"])) != 24:
        raise ValueError("duplicate PR43 seed")
    for regime in ("light", "medium", "heavy", "shift"):
        if sum(g["regime"] == regime for g in proto["groups"]) != 6:
            raise ValueError("incorrect preregistered stratum")
    # Prior testing had seeds around 1.1m-1.4m; these high seeds are disjoint.
    existing = json.loads(Path(__file__).resolve().parents[1].joinpath(
        "online_horizon/pr30_protocol.json").read_text())
    if set(g["seed"] for g in existing["groups"]) & set(g["seed"] for g in proto["groups"]):
        raise ValueError("test seed used by PR30")


def run(checkpoint, output):
    proto = json.loads(PROTOCOL.read_text())
    check_protocol(proto, checkpoint)
    if output.exists():
        raise FileExistsError("PR43 source evidence cannot overwrite")
    torch.set_num_threads(1)
    start = time.process_time()
    groups = []
    for spec in proto["groups"]:
        if time.process_time() - start >= proto["total_cpu_process_budget_seconds"]:
            raise TimeoutError("PR43 total CPU budget exhausted")
        groups.append({
            "id": spec["id"], "regime": spec["regime"], "seed": spec["seed"],
            "arms": {arm: one(spec, proto, arm, checkpoint) for arm in ARMS},
        })
    elapsed = time.process_time()-start
    if elapsed >= proto["total_cpu_process_budget_seconds"]:
        raise TimeoutError("PR43 total CPU budget exceeded")
    record = {
        "schema": "rhwm-pr43-independent-online-forecast-utility-v1",
        "protocol": proto["protocol"], "protocol_sha256": file_sha(PROTOCOL),
        "frozen_v4_sha256": file_sha(checkpoint),
        "training": 0, "gpu": 0, "model_architecture_unchanged": True,
        "evidence_scope": "one-shot_new_synthetic_independent_full_online",
        "method_provenance": "reuse_PR27_trajectory_cost_and_PR30_CP_SAT_H20",
        "groups": groups,
        "summary": {"groups": len(groups), "cpu_process_seconds": elapsed,
                    "by_regime": summarize(groups),
                    "limitations": "Synthetic cases; CPU-process-time gate is not hard wall-clock preemption"},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2)+"\n")
    print("PR43_RESULT", json.dumps(record["summary"], ensure_ascii=False))
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    run(args.checkpoint, args.out)


if __name__ == "__main__":
    main()
