"""CPU-only PR16 finite-candidate hindsight value, with explicit observability boundaries.

Extends PR15's branch replay only to *return* true short-run observations.
No online model is given those observations; scoring and diagnostics use them offline.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

from business_transport.dispatch import BusinessCPSAT
from business_transport.lacam3_integration import RecoveryObserver
from business_transport.policy import WorkflowPolicy
from business_transport.workflow_scoring import workflow_score
from business_transport.workflow_transition import check_business_transition
from experiments.business_constraints.benchmark import make_case, read_config
from experiments.service_value.counterfactual import load_pr9_model
from experiments.workflow_prediction.config_audit import CONFIG as PR15_CONFIG
from experiments.workflow_prediction.evaluate import (
    make_adapter, replay_branch, rich_rows, run_method, service_events,
    successor_events, robot_signature, station_signature, sample_signature
)
from lab_transport.decision_value import candidates_for, nearest_batch
from lab_transport.dynamics_model import Geometry

from .analysis import analyze, outcome_key, select_oracle

CONFIG = Path("experiments/oracle_value/config.json")
PR15_RAW = Path("examples/evidence/workflow-prediction-20261110/confirmation/raw.json.gz")
NAMES = {"Rule-5": "rule", "PETS-5": "pets", "Structured-5": "structured"}
EPS = 1e-9


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def dumps(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def settings(config, profile):
    if profile == "history":
        reference = json.loads(PR15_CONFIG.read_text())["confirmation"]
        return [(r["layout"], r["seed"], regime)
                for r in reference for regime in r["regimes"]]
    return [(r["layout"], seed, regime)
            for r in config["confirmation"] for seed in r["seeds"]
            for regime in r["regimes"]]


def preflight(config, historical_config, prior_benchmark):
    old = [(r["layout"], r["seed"], regime)
           for part in ("development", "confirmation")
           for r in historical_config[part] for regime in r["regimes"]]
    for grouping in list(prior_benchmark["seeds"].values()) + [
        prior_benchmark["previous_confirmation_pilot"],
        prior_benchmark["previous_shared_rng_confirmation"]
    ]:
        for layout, seeds in grouping.items():
            old.extend((layout, seed, regime) for seed in seeds
                       for regime in prior_benchmark["regimes"])
    def sig(layout, seed, regime):
        _, conf = make_case(prior_benchmark, layout, seed, regime)
        return sha([(j["workflow"], j["release_tick"]) for j in conf["jobs"]])
    prior = {(layout, sig(layout,seed,regime)) for layout,seed,regime in old}
    rows = []
    seen = {}
    for layout, seed, regime in settings(config, "confirmation"):
        _, business = make_case(prior_benchmark, layout,seed,regime)
        fingerprint = sig(layout,seed,regime)
        record = {"layout":layout, "seed":seed, "regime":regime,
                  "fingerprint":fingerprint, "jobs":business["jobs"],
                  "overlaps_prior_task_table":(layout,fingerprint) in prior,
                  "same_table_as_other_confirmed_clusters":False}
        key=(layout,fingerprint)
        if key in seen and seen[key] != seed:
            record["same_table_as_other_confirmed_clusters"] = True
        else:
            seen[key] = seed
        rows.append(record)
    return {"frozen_config_sha256":hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
            "historical_config_sha256":hashlib.sha256(PR15_CONFIG.read_bytes()).hexdigest(),
            "scenes":rows,
            "prior_duplicates":sum(r["overlaps_prior_task_table"] for r in rows),
            "cross_seed_duplicates":sum(r["same_table_as_other_confirmed_clusters"] for r in rows),
            "duplicate_note":"The same seed has intentionally identical task tables across its three regimes.",
            "comparison_scope":"Previously used maps; new synthetic task tables are not novel physical layouts."}


def plan_records(plans):
    return [[asdict(action) for action in plan] for plan in plans]


def future_metrics(observation, plans, method, geometry, rules, network, threshold, pibt, lacam, config, truth):
    """PR15 model rollout with identical public state and first-action plan.

    One particle's state sequence is used ONLY for mismatch attribution.
    Decision selection remains PR15's unchanged three-particle evaluation.
    """
    model = make_adapter(NAMES[method], geometry, rules, network, threshold,
                         pibt, lacam, config["planner_seed"])
    policy = WorkflowPolicy("cpsat_" + NAMES[method],geometry,rules,RecoveryObserver(),
                            model,seed=config["planner_seed"],horizon=5,
                            particles=config["particles"],budget=config["candidate_budget"],
                            solver_seconds=config["cp_sat_time_limit_seconds"])
    try:
        evaluations = policy.evaluate(observation,plans)
        scores = [list(e["score"]) for e in evaluations]
        choice = min(range(len(scores)), key=lambda i:(scores[i],i))
        diagnostics = []
        for plan, actual in zip(plans, truth):
            if actual is None:
                diagnostics.append({"error":"candidate replay failed"})
                continue
            policy.model.reset_seed(config["planner_seed"]+observation.tick*1009)
            current = observation
            mismatches = Counter()
            per_step = []
            for t in range(min(5,len(actual))):
                actions = plan if t == 0 else nearest_batch(current,candidates_for(current),geometry)
                nxt = policy.model.predict_next(current,actions)
                check_business_transition(current,nxt,geometry,rules,actions)
                true_before = observation if t == 0 else actual[t-1]
                true_after = actual[t]
                true_service = service_events(true_before,true_after,observation)
                pred_service = service_events(current,nxt,observation)
                true_release = successor_events(true_before,true_after,observation)
                pred_release = successor_events(current,nxt,observation)
                mismatches.update({
                    "service_event_disagreement":int(true_service!=pred_service),
                    "successor_release_disagreement":int(true_release!=pred_release),
                    "robot_position_task_disagreement":int(robot_signature(nxt)!=robot_signature(true_after)),
                    "station_capacity_disagreement":int(station_signature(nxt)!=station_signature(true_after)),
                    "sample_state_disagreement":int(sample_signature(nxt,observation)!=
                                                    sample_signature(true_after,observation))
                })
                per_step.append({
                    "tick":observation.tick+t+1,
                    "service_true":sorted(true_service),
                    "service_predicted":sorted(pred_service),
                    "release_true":sorted(true_release),
                    "release_predicted":sorted(pred_release),
                    "robot_true":robot_signature(true_after),
                    "robot_predicted":robot_signature(nxt),
                    "station_true":station_signature(true_after),
                    "station_predicted":station_signature(nxt)
                })
                current = nxt
            diagnostics.append({"first_particle_mismatches":dict(mismatches),"steps":per_step})
        return {"choice":choice,"scores":scores,"evaluations":evaluations,
                "candidate_diagnostics":diagnostics,"prediction_calls":policy.prediction_calls,
                "prediction_wall_seconds":policy.prediction_wall_seconds}
    except (ValueError, RuntimeError, TypeError, KeyError, StopIteration, OverflowError) as exc:
        # No silent online fallback is allowed to masquerade as valid evaluation.
        return {"error":type(exc).__name__+": "+str(exc), "choice":None,"scores":None}


def factual_short_score(observation, traces, horizon, geometry, rules):
    scores = []
    truncated = []
    for trace in traces:
        if trace is None or len(trace) < horizon:
            truncated.append(len(trace) if trace is not None else None)
            scores.append(None)
        else:
            truncated.append(None)
            scores.append(list(workflow_score(observation, trace[:horizon],geometry,rules)))
    if any(value is None for value in scores):
        return {"error":"full actual horizon unavailable; no synthetic padding",
                "available_steps":truncated,"choice":None,"scores":scores}
    return {"choice":min(range(len(scores)),key=lambda i:(scores[i],i)),
            "scores":scores,"available_steps":[horizon]*len(scores)}


def event_counters(events):
    types = Counter(e["type"] for e in events)
    return {"service_started":types["service_started"],
            "service_finished":types["service_finished"],
            "stage_ready":types["stage_ready"],
            "resource_wait":types["resource_wait"],
            "station_unavailable_wait":types["station_unavailable_wait"]}


def get_prior_reference():
    with gzip.open(PR15_RAW,"rt",encoding="utf-8") as file:
        data = json.load(file)
    return {(c["layout"],c["seed"],c["regime"]):c for c in data["cases"]}


def historical_match(layout,seed,regime,decisions,reference):
    old = reference[(layout,seed,regime)]["same_state"]
    issues = []
    if len(decisions)!=len(old):
        issues.append(f"decision count {len(decisions)} != {len(old)}")
    by_tick={d["tick"]:d for d in old}
    labels={"Rule-5":"new_transition_new_score","PETS-5":"pets_new",
            "Structured-5":"structured_new"}
    for row in decisions:
        truth=by_tick.get(row["tick"])
        if truth is None:
            issues.append(f"tick {row['tick']} missing from PR15");continue
        if row["plans"] != truth["plans"]:
            issues.append(f"tick {row['tick']} candidate plans diverge")
        for method,label in labels.items():
            newer=row["methods"][method]
            old_selection=truth["forecasts"][label]["choice"]
            if newer["choice"] != old_selection:
                issues.append(f"tick {row['tick']} {method} choice diverges")
        for i,(new,previous) in enumerate(zip(row["branches"],truth["branches"])):
            if "error" in new:
                issues.append(f"tick {row['tick']} branch {i} replay failed")
            else:
                for field in ("jobs_completed","restricted_mean_flow"):
                    if abs(new[field]-previous[field]) > EPS:
                        issues.append(f"tick {row['tick']} branch {i} {field} diverges")
    return issues


def evaluate_decisions(business,seed,factual,rows,geometry,network,threshold,pibt,lacam,config):
    rich=rich_rows(rows)
    eligible=[i for i,r in enumerate(rich) if r.assignments and
              len(candidates_for(r.observation))>=2]
    indices=sorted(set([eligible[0],eligible[len(eligible)//2]])) if eligible else []
    decisions = []
    for index in indices:
        begun=time.perf_counter(); obs=rich[index].observation
        possible=candidates_for(obs)
        solver=BusinessCPSAT(geometry,business.rules,seed=config["planner_seed"],
                            candidate_limit=config["candidate_budget"],
                            time_limit_seconds=config["cp_sat_time_limit_seconds"])
        plans=solver.proposals(obs,possible)
        decision={"tick":obs.tick,"observable_state":asdict(obs),
                  "eligible_assignment_pairs":[asdict(a) for a in possible],
                  "plans":plan_records(plans),"solver":solver.log[-1],
                  "branches":[],"methods":{},"replay_checks":{}}
        traces=[]
        for candidate_index,plan in enumerate(plans):
            t0=time.perf_counter()
            try:
                result,truth,meta = replay_branch(
                    business,seed,rows,factual,obs.tick,plan,pibt,lacam,config,trace_limit=20)
                result["replay_wall_seconds"] = time.perf_counter()-t0
                result["event_counts"] = event_counters(result["events"])
                result["motion_recovery"] = meta
                result["trace_first20"] = [asdict(s) for s in truth]
                result["trace_length"] = len(truth)
                decision["branches"].append(result)
                traces.append(truth)
            except Exception as exc:
                decision["branches"].append({"error":type(exc).__name__+": "+str(exc),
                                             "replay_wall_seconds":time.perf_counter()-t0})
                traces.append(None)
        # Repeating the first original candidate is an independent full simulator run.
        if plans and "error" not in decision["branches"][0]:
            try:
                first, repeated_trace,meta = replay_branch(
                    business,seed,rows,factual,obs.tick,plans[0],pibt,lacam,config,trace_limit=20)
                left={k:v for k,v in decision["branches"][0].items() if k in first}
                decision["replay_checks"]["first_candidate_reproduced"] = left == first and [
                    asdict(s) for s in repeated_trace] == decision["branches"][0]["trace_first20"]
                decision["replay_checks"]["prefix_equal"] = bool(meta["complete_factual_prefix"])
                if not decision["replay_checks"]["first_candidate_reproduced"]:
                    decision["replay_checks"]["error"]="same candidate returned different results"
            except Exception as exc:
                decision["replay_checks"]["error"]=type(exc).__name__+": "+str(exc)
                decision["replay_checks"]["first_candidate_reproduced"]=False
        if len(plans)<2:
            decision["error"]="fewer than two distinct CP-SAT plans; retained without replacement"
        # Hindsight alone may look at actual futures. Public predictors never do.
        if all(trace is not None for trace in traces) and len(plans)>=2:
            for horizon in (5,20):
                decision["methods"][f"Perfect-State-{horizon}"]=factual_short_score(
                    obs,traces,horizon,geometry,business.rules)
        else:
            for horizon in (5,20):
                decision["methods"][f"Perfect-State-{horizon}"]={
                    "error":"no complete set of replayed candidates","choice":None,"scores":None}
        for method in NAMES:
            decision["methods"][method] = future_metrics(
                obs,plans,method,geometry,business.rules,network,threshold,pibt,lacam,config,traces)
        if len(plans)>=2 and all("error" not in b for b in decision["branches"]):
            best=select_oracle(decision["branches"])
            decision["methods"]["Full-outcome Oracle"]={"choice":best,"scores":None,
                        "meaning":"hindsight best ONLY among these original candidates"}
        else:
            decision["methods"]["Full-outcome Oracle"]={"choice":None,"scores":None,
                       "error":"incomplete or invalid finite candidate set"}
        decision["candidate_diversity"]={
            "represented_pair_count":len({(p["robot_id"],p["sample_id"]) for plan in
                                          decision["plans"] for p in plan}),
            "visible_eligible_pair_count":len(possible),
            "unrepresented_eligible_pairs":[asdict(a) for a in possible if
                (a.robot_id,a.sample_id) not in
                {(p["robot_id"],p["sample_id"]) for plan in decision["plans"] for p in plan}],
            "warning":"Eligibility of a single pair is NOT proof that another full batch is feasible."}
        decision["wall_seconds"]=time.perf_counter()-begun
        decisions.append(decision)
    return decisions,{"eligible_pair_ticks":len(eligible),"selected_ticks":len(indices),
                      "no_eligible_decision_ticks":not eligible}


def run(profile,output,pibt,lacam,limit=None):
    output=Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    config=json.loads(CONFIG.read_text())
    frozen_hash=hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    baseline=read_config()
    pre=preflight(config,json.loads(PR15_CONFIG.read_text()),baseline)
    (output/"preflight.json").write_text(dumps(pre),encoding="utf-8")
    print("PREFLIGHT",profile, "prior_duplicates",pre["prior_duplicates"],
          "cross_seed_duplicates",pre["cross_seed_duplicates"],flush=True)
    if limit is not None:
        print("LIMITED pilot: not confirmation evidence",limit,flush=True)
    network, checkpoint = load_pr9_model()
    all_settings=settings(config,profile)
    if limit is not None:all_settings=all_settings[:limit]
    reference=get_prior_reference() if profile=="history" else None
    cases=[]
    started=time.perf_counter();cpu0=time.process_time();history_issues=[]
    for layout,seed,regime in all_settings:
        scene_start=time.perf_counter()
        business,assumptions=make_case(baseline,layout,seed,regime)
        geometry=Geometry(business.scenario.grid)
        factual,rows,metrics=run_method(business,seed,"nearest",network,checkpoint["threshold"],
                                       pibt,lacam,config)
        expected_factual=sha(factual["history"])
        decisions,sampling=evaluate_decisions(
            business,seed,factual,rows,geometry,network,checkpoint["threshold"],pibt,lacam,config)
        factual_again,_,_=run_method(business,seed,"nearest",network,checkpoint["threshold"],
                                    pibt,lacam,config)
        unchanged=(sha(factual["history"])==expected_factual==
                   sha(factual_again["history"]) and factual_again["events"]==factual["events"])
        if not unchanged:
            history_issues.append(f"{layout}/{seed}/{regime}: factual run changed after replay")
        if reference is not None:
            history_issues.extend(historical_match(layout,seed,regime,decisions,reference))
        cases.append({"layout":layout,"seed":seed,"regime":regime,
                      "assumptions":assumptions,"factual_metrics":metrics,
                      "factual_history_sha256":expected_factual,
                      "factual_repeat_unchanged":unchanged,
                      "sampling":sampling,"decisions":decisions,
                      "scene_wall_seconds":time.perf_counter()-scene_start})
        print(profile,layout,seed,regime,"selected",len(decisions),
              "plans",sum(len(d["plans"]) for d in decisions),
              "elapsed_s",round(time.perf_counter()-scene_start,3),flush=True)
        # Incremental crash-safe evidence: write each completed scene before the next.
        (output/"progress.json").write_text(dumps({"completed_cases":len(cases),
            "configured_cases":len(all_settings),"wall_seconds":time.perf_counter()-started,
            "history_issues":history_issues}),encoding="utf-8")
    summary=analyze(cases)
    data={"profile":profile,"limited_pilot":limit is not None,"frozen_config_sha256":frozen_hash,
          "checkpoint":checkpoint,"preflight":pre,
          "history_validation_issues":history_issues,
          "cpu_seconds":time.process_time()-cpu0,
          "wall_seconds":time.perf_counter()-started,
          "cases":cases}
    with gzip.open(output/"raw.json.gz","wt",encoding="utf-8") as stream:
        json.dump(data,stream,ensure_ascii=False,sort_keys=True)
    (output/"summary.json").write_text(dumps(summary),encoding="utf-8")
    (output/"cpu.json").write_text(dumps({
        "cpu_seconds":data["cpu_seconds"],"wall_seconds":data["wall_seconds"],
        "scenes":len(cases),
        "approximate_remaining_scenes_to_24":max(0,24-len(cases)) if profile=="confirmation" else 24
    }),encoding="utf-8")
    if history_issues:
        (output/"historical_validation_failures.json").write_text(
            dumps(history_issues),encoding="utf-8")
        raise RuntimeError(f"mechanism acceptance failed: {len(history_issues)} issues; no automatic confirmation")
    if any(not c["factual_repeat_unchanged"] or
           any(d["replay_checks"].get("first_candidate_reproduced") is False
               for d in c["decisions"]) for c in cases):
        raise RuntimeError("counterfactual fairness invariant failed")
    print("SUMMARY",profile,json.dumps({k:summary[k] for k in
          ("scenarios","clusters","actual_decisions","candidate_total","failed_decisions")},
          ensure_ascii=False),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile",choices=["history","confirmation"],required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--pibt-library",type=Path,required=True)
    parser.add_argument("--lacam-worker",type=Path,required=True)
    parser.add_argument("--limit",type=int,help="smoke test only; never report as confirmation")
    args=parser.parse_args()
    run(args.profile,args.output,args.pibt_library,args.lacam_worker,args.limit)


if __name__=="__main__":
    main()
