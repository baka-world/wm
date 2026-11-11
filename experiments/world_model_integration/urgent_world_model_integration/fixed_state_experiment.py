"""Frozen-decision Monte Carlo stability and paired exogenous counterfactuals.

No changes to the production world model, business simulator, CP-SAT or
original trajectories. Reconstruct exactly one *public* state from archived
selected action prefixes; freeze its legal candidate list; resample V4
predicted service events with independent hashing seeds; compare each legal
candidate's actual future using the original simulator and an identical
P1 urgent-first continuation after the forced action.

Public observations and candidate scores are predictors. True future outcomes
are accessed only in the separate post-hoc evaluation stage.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from zipfile import ZipFile

import torch

from business_transport.dispatch import BusinessCPSAT
from business_transport.simulation import resource_audit
from experiments.business_constraints.run import lifecycle_audit
from experiments.world_model_integration.urgent_workflow_feasibility.core import (
    ConditionalRetestSimulator, IsolatedPolicy, PriorityObserver, PROTOCOL,
    digest, job_metrics, judge_score, public_spec, scene_for, unique_plans,
    synthetic_retest,
)
from experiments.world_model_integration.urgent_world_model_integration.core import (
    UrgentWorldPolicy,
)
from lab_transport.multisite_experiment import discrete_safety_violations
from lab_transport.transition_schema import Assignment

ROOT=Path(__file__).resolve().parent
PROBE_PROTOCOL=ROOT/"fixed_state_protocol.json"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_archive(path, expected_hash, prefix):
    data=path.read_bytes()
    if sha(data)!=expected_hash:
        raise AssertionError("SOURCE_ARCHIVE_SHA256_MISMATCH: "+path.name)
    with ZipFile(path) as z:
        manifest=json.loads(z.read(prefix+"/sha256.json"))
        for name, hash_value in manifest.items():
            if sha(z.read(prefix+"/"+name))!=hash_value:
                raise AssertionError("INNER_ARTIFACT_SHA256_MISMATCH: "+name)
        cases=json.loads(gzip.decompress(z.read(prefix+"/cases.json.gz")))
    indexed={(int(c["seed"]),c["arm"]):c for c in cases}
    if len(indexed)!=len(cases):
        raise AssertionError("DUPLICATE_ARCHIVE_CASE")
    return indexed


def _matches(plan):
    return tuple(sorted((a.robot_id,a.sample_id) for a in plan))


def _stored_matches(items):
    return tuple(sorted((a["robot_id"],a["sample_id"]) for a in items))


def _assignment(items):
    return tuple(Assignment(a["robot_id"],a["sample_id"]) for a in items)


def replay_one(spec, config, original, target_tick, target_choice, *,
               verify_candidate_list, retain_state=False):
    """Re-run original P5 action prefix, force one legal target action, then P1.

    Stochastic real service duration and conditional retest are both keyed by
    the same synthetic scenario/task identity across forced alternatives.
    This paired-future diagnostic is not a proof of expected performance.
    """
    business,geo=scene_for(spec)
    sim=ConditionalRetestSimulator(
        business,spec["seed"],spec["retest_rate_percent"],collect_metrics=True)
    current_observer=PriorityObserver()
    follow=IsolatedPolicy("P1",geo,config,spec["jitter"])
    stored={int(d["tick"]):d for d in original["decisions"]}
    captured={}
    entered={"target":False}
    started=time.process_time()

    def on_observation(raw):
        current_observer.observe(raw)
        follow.observer.observe(raw)

    def dispatch(raw,legal):
        obs=current_observer.observe(raw)
        tick=obs.tick
        if tick<target_tick:
            old=stored.get(tick)
            if old is None or digest(asdict(obs))!=old["observation_sha256"]:
                raise AssertionError("ORIGINAL_FACTUAL_PREFIX_STATE_DRIFT")
            chosen=_assignment(old["selected"])
        elif tick==target_tick:
            old=stored.get(tick)
            if old is None or digest(asdict(obs))!=old["observation_sha256"]:
                raise AssertionError("FROZEN_TARGET_OBSERVATION_DRIFT")
            solver=BusinessCPSAT(geo,sim.business.rules,
                  candidate_limit=config["max_legal_candidates"],
                  time_limit_seconds=config["cpsat_seconds"],
                  deterministic_time_limit=config["deterministic_cpsat_seconds"],seed=1810)
            plans=unique_plans(solver.proposals(obs,legal),legal)
            if verify_candidate_list:
                if [_matches(p) for p in plans]!=[
                        _stored_matches(x["matching"]) for x in old["plans"]]:
                    raise AssertionError("FROZEN_CPSAT_CANDIDATE_SET_OR_ORDER_DRIFT")
            if target_choice<0 or target_choice>=len(plans):
                raise AssertionError("FORCED_CANDIDATE_INDEX_OUT_OF_RANGE")
            chosen=plans[target_choice]
            if retain_state:
                captured["observation"]=obs
                captured["rules"]=sim.business.rules
                captured["geometry"]=geo
                captured["plans"]=plans
                captured["archived_record"]=old
            entered["target"]=True
        else:
            # Same public P1 continuation after the one forced decision.
            return follow(raw,legal,sim)
        if any(action not in legal for action in chosen):
            raise AssertionError("FROZEN_PREFIX_OR_FORCED_ACTION_NOT_LEGAL")
        return chosen

    result=sim.run(dispatch_policy=dispatch,on_observation=on_observation)
    if not entered["target"]:
        raise AssertionError("TARGET_TICK_WAS_NOT_REACHED")
    capacity=resource_audit(result,sim.business.rules)
    motion=discrete_safety_violations(result)
    lifecycle=lifecycle_audit(result["history"])
    if capacity["violations"] or motion or lifecycle:
        raise AssertionError("REAL_COUNTERFACTUAL_SAFETY_FAILURE")
    metric=job_metrics(result,sim.business,config["scenario_max_ticks"])
    if metric["job_count"]!=spec["jobs"]:
        raise AssertionError("COUNTERFACTUAL_CHANGED_CLINICAL_JOB_COUNT")
    for jid,flag in sim.retest_results.items():
        if flag!=synthetic_retest(spec["seed"],jid,spec["retest_rate_percent"]):
            raise AssertionError("EXOGENOUS_RETEST_CHANGED")
    return {"metrics":metric,
            "judge_score":judge_score(metric),
            "history_sha256":digest(result["history"]),
            "events_sha256":digest(result["events"]),
            "retest_results":sim.retest_results,
            "replay_cpu_seconds":time.process_time()-started},captured


def _safe_scores(a,b):
    if len(a)!=len(b):return False
    for x,y in zip(a,b):
        if not math.isclose(x,y,rel_tol=1e-10,abs_tol=1e-9):
            return False
    return True


def _candidate_value_scores(observation,rules,geo,spec,config,checkpoint,
                            proposals, original_record, resampling_seeds):
    """Frozen state: same candidates, same scoring, independent RNG seed groups."""
    p5=UrgentWorldPolicy("P5",geo,rules,config,checkpoint,seed=spec["seed"],
                         random_coupling="keyed",terminal_mode="budgeted_queue",
                         evaluation_mode="full")
    p5._synchronize_public_rules(rules)
    p4=UrgentWorldPolicy("P4",geo,rules,config,None,seed=spec["seed"],
                         random_coupling="keyed",terminal_mode="budgeted_queue",
                         evaluation_mode="full")
    p4._synchronize_public_rules(rules)
    rule_scores=[]
    for i,plan in enumerate(proposals):
        score,_=p4._score(observation,plan,rules,i)
        rule_scores.append(score)
    repetitions=[]
    for k,pred_seed in enumerate(resampling_seeds):
        p5.seed=int(pred_seed)     # changes only keyed Monte Carlo draw identity
        scores=[]
        particle_details=[]
        before=p5.query_calls
        for i,plan in enumerate(proposals):
            value, particles=p5._score(observation,plan,rules,i)
            scores.append(value)
            particle_details.append(particles)
            if k==0:
                recorded=original_record["plans"][i]
                if not _safe_scores(value,recorded["score"]):
                    raise AssertionError("ORIGINAL_FROZEN_V4_PREDICTION_DRIFT")
                if any(not _safe_scores(x,y)
                       for x,y in zip(particles,recorded["particle_costs"])):
                    raise AssertionError("ORIGINAL_V4_PARTICLES_DIFFER")
        selected=min(range(len(scores)),key=lambda i:scores[i])
        repetitions.append({
            "rng_seed":pred_seed,
            "chosen_candidate_index":selected,
            "scores":scores,
            "particle_costs":particle_details,
            "V4_service_event_queries":p5.query_calls-before,
        })
    original_chosen=int(original_record["selected_candidate_index"])
    if repetitions[0]["chosen_candidate_index"]!=original_chosen:
        raise AssertionError("ARCHIVED_V4_SELECTED_CANDIDATE_DRIFT")
    return rule_scores,repetitions


def audit_one(item, original, config, checkpoint, sample_seeds):
    spec=public_spec(item["index"],item["seed"],config["scenario_max_ticks"])
    tick=item["tick"]
    record=next((r for r in original["decisions"] if r["tick"]==tick),None)
    if record is None:
        raise AssertionError("FROZEN_DECISION_NOT_IN_ARCHIVED_TRAJECTORY")
    if record["observation_sha256"]!=item["observation_sha256"]:
        raise AssertionError("FROZEN_OBSERVATION_IDENTITY_PROTOCOL_MISMATCH")
    if record["candidate_count"]!=item["candidate_count"]:
        raise AssertionError("FROZEN_CANDIDATE_COUNT_PROTOCOL_MISMATCH")
    original_choice=int(record["selected_candidate_index"])
    if len(record["plans"])<2:
        raise AssertionError("TARGET_REQUIRES_MULTIPLE_CANDIDATES")
    # One replay obtains lawful state, frozen candidate set and original
    # chosen-action outcome under the same P1 continuation as alternatives.
    initial,captured=replay_one(spec,config,original,tick,original_choice,
                               verify_candidate_list=True,retain_state=True)
    plans=captured["plans"]
    if len(plans)!=len(record["plans"]):
        raise AssertionError("FROZEN_CANDIDATE_COUNT_MISMATCH")
    trial_seeds=[spec["seed"],*sample_seeds]
    rule_scores,reps=_candidate_value_scores(
        captured["observation"],captured["rules"],captured["geometry"],
        spec,config,checkpoint,plans,record,trial_seeds)
    futures=[]
    for i in range(len(plans)):
        if i==original_choice:
            out=initial
        else:
            out,_=replay_one(spec,config,original,tick,i,
                             verify_candidate_list=True,retain_state=False)
        futures.append({"candidate_index":i,
                        "matching":[asdict(a) for a in plans[i]],
                        "judge_score":out["judge_score"],
                        "metrics":out["metrics"],
                        "history_sha256":out["history_sha256"],
                        "events_sha256":out["events_sha256"],
                        "retest_results":out["retest_results"],
                        "cpu_seconds":out["replay_cpu_seconds"]})
    best=min(range(len(futures)),key=lambda i:tuple(futures[i]["judge_score"])+(i,))
    counts=Counter(rep["chosen_candidate_index"] for rep in reps)
    costs={i:future["judge_score"] for i,future in enumerate(futures)}
    best_hits=sum(tuple(costs[r["chosen_candidate_index"]])==tuple(costs[best]) for r in reps)
    for rep in reps:
        outcome=costs[rep["chosen_candidate_index"]]
        optimal=costs[best]
        rep["posthoc_true_optimal_for_fixed_P1_continuation"]=(tuple(outcome)==tuple(optimal))
        rep["oracle_difference_componentwise"]=[
            a-b for a,b in zip(outcome,optimal)]
    return {
        "seed":spec["seed"],"index":spec["index"],"target_tick":tick,
        "source":item["source"],
        "source_observation_sha256":record["observation_sha256"],
        "frozen_candidates":len(plans),
        "original_selected_index":original_choice,
        "rule_scores":rule_scores,
        "rule_selected_index":min(range(len(rule_scores)),key=lambda i:rule_scores[i]),
        "frozen_V4_sampling":reps,
        "independent_resampling_count":len(sample_seeds),
        "selection_counts":dict(sorted(counts.items())),
        "selection_consistency_with_original":counts[original_choice]/len(reps),
        "empirical_selection_entropy_bits":-sum((v/len(reps))*math.log2(v/len(reps))
                                                 for v in counts.values()),
        "real_futures_with_same_prefix_and_P1_continuation":futures,
        "posthoc_optimal_candidate_index":best,
        "posthoc_best_selection_frequency":best_hits/len(reps),
        "counterfactual_scope":"fixed external task/station service keyed draws and retest; fixed P1 continuation, single simulator future, NOT expected advantage",
        "original_prefix_observation_reproduction":True,
        "candidate_identity_verified":True
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--archive-confirmation",type=Path,required=True)
    p.add_argument("--archive-new-full",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--cpu-budget",type=float,default=600.)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError("EVIDENCE_PATH_ALREADY_EXISTS")
    config=json.loads(PROTOCOL.read_text())
    protocol=json.loads(PROBE_PROTOCOL.read_text())
    if sha(a.checkpoint.read_bytes())!=protocol["checkpoint_sha256"]:
        raise AssertionError("FROZEN_V4_CHECKPOINT_MISMATCH")
    if torch.version.cuda is not None:
        raise AssertionError("CPU_ONLY_REQUIRED")
    torch.set_num_threads(1)
    cases={
        "confirmation":read_archive(a.archive_confirmation,
            protocol["archives"]["confirmation"]["sha256"],"confirmation"),
        "new_full":read_archive(a.archive_new_full,
            protocol["archives"]["new_full"]["sha256"],"new-full")
    }
    work=[]
    begun=time.process_time()
    for item in protocol["fixed_states"]:
        case=cases[item["source"]][(item["seed"],"P5")]
        if case["index"]!=item["index"]:
            raise AssertionError("ARCHIVED_SCENARIO_INDEX_DRIFT")
        res=audit_one(item,case,config,a.checkpoint,protocol["independent_rng_seeds"])
        work.append(res)
        print("FIXED_STATE",item["seed"],item["tick"],
              "selected_counts",res["selection_counts"],
              "fixed_future_best",res["posthoc_optimal_candidate_index"],
              "best_hit_rate",res["posthoc_best_selection_frequency"],
              "elapsed_cpu",round(time.process_time()-begun,3),flush=True)
        if time.process_time()-begun>a.cpu_budget:
            raise TimeoutError("FROZEN_CPU_BUDGET_EXCEEDED")
    if len(work)!=len(protocol["fixed_states"]):
        raise AssertionError("FROZEN_STATE_COUNT_DRIFT")
    total=len(work)
    mean_original=statistics.mean(x["selection_consistency_with_original"] for x in work)
    mean_best=statistics.mean(x["posthoc_best_selection_frequency"] for x in work)
    summary={
        "scenes":total,
        "candidate_count_sum":sum(x["frozen_candidates"] for x in work),
        "prediction_seed_repetitions_per_state":1+len(protocol["independent_rng_seeds"]),
        "mean_agreement_with_original":mean_original,
        "mean_posthoc_optimal_selection_frequency":mean_best,
        "completely_stable_states":sum(len(x["selection_counts"])==1 for x in work),
        "states_with_original_V4_choice_not_posthoc_best":sum(
            x["original_selected_index"]!=x["posthoc_optimal_candidate_index"]
            for x in work),
        "states_where_P4_choice_not_posthoc_best":sum(
            x["rule_selected_index"]!=x["posthoc_optimal_candidate_index"]
            for x in work),
        "CPU_process_seconds":time.process_time()-begun,
        "gpu":0,"trainings":0,"new_model_weights":0,
        "future_sampling":"real keyed service and retest, same scenario seed; model sampled separately",
        "not_independent_test":True,
        "disclaimer":"five states deliberately selected after seeing action flips; no external/causal generalization"
    }
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({
        "protocol_sha256":sha(PROBE_PROTOCOL.read_bytes()),
        "checkpoint_sha256":sha(a.checkpoint.read_bytes()),
        "summary":summary,"states":work
    },ensure_ascii=False,sort_keys=True,indent=2)+"\n")
    print("FINAL",json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
