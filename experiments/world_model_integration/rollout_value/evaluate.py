"""PR18 offline paired full-horizon candidate value audit. NO training / no online oracle.
Restart the ACTUAL simulator from tick zero, not from a reconstructed visible
Observation. Future seeds are admissible ONLY because all fixed decisions are at
tick zero, before any service noise is realized. All outcome access is offline.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from business_transport.dispatch import BusinessCPSAT
from business_transport.observation import BusinessObserver
from business_transport.simulation import BusinessSimulator, resource_audit
from experiments.business_constraints.run import replay_branch, rich_rows
from experiments.world_model_integration.build_dataset import case_for
from experiments.world_model_integration.build_dataset_v4 import CONFIG
from experiments.world_model_integration.v4_dispatch import policy_for
from lab_transport.decision_value import candidates_for, nearest_batch

V4_SHA="60c8a6c3d832da3e6519072688ea099f64ae120ba19d3bd2ec705f7f1a11ac0c"
SOURCE_SEEDS=(43011,43015,43019)
# Frozen BEFORE reading any new multi-seed results. No search for favorable seeds.
SEED_OFFSETS=(0,101,203,307,409,503,607,709)
SOLVER_SECONDS=0.5
CPU_WALL_BUDGET_SECONDS=1500
KNOWN_HORIZONS=(5,10,20)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def restricted_flow(result,business):
    by_id={s["id"]:s for s in result["samples"]}
    cutoff=result["summary"]["ticks"]
    flows=[]
    waits=[]
    for job in business.jobs:
        stages=[by_id[sid] for sid in job["stages"]]
        final=stages[-1]["completed_tick"]
        flows.append((final if final is not None else cutoff)-job["release_tick"] if
                     (final is not None or cutoff>=job["release_tick"]) else 0)
        waits.append(sum(s["assigned_tick"]-s["release_tick"] for s in stages if s["assigned_tick"] is not None))
    return {
      "restricted_mean_flow":statistics.mean(flows),
      "completed_all":all(by_id[j["stages"][-1]]["completed_tick"] is not None for j in business.jobs),
      "mean_stage_wait_per_job":statistics.mean(waits),
      "capacity_wait_ticks":resource_audit(result,business.rules)["capacity_wait_ticks"],
      "ticks":cutoff, "per_job_flow":flows,
      "history_sha256":hashlib.sha256(json.dumps(result["history"],sort_keys=True).encode()).hexdigest(),
      "events_sha256":hashlib.sha256(json.dumps(result["events"],sort_keys=True).encode()).hexdigest()
    }


def run_branch(business, geometry, seed, first, continuation, expected_initial, *, solver_seed=1810):
    """No simulator cloning or private state injection. Only legal observation->actions."""
    assert continuation in ("P1_nearest","P2_receding_cpsat")
    observer=BusinessObserver()
    solver=BusinessCPSAT(geometry,business.rules,candidate_limit=4,
                         time_limit_seconds=SOLVER_SECONDS,seed=solver_seed) if continuation.startswith("P2") else None
    first_seen=[False]
    def observe(o):
        return observer.observe(o)
    def dispatch(o,candidates):
        if o.tick==0:
            if o!=expected_initial:
                raise AssertionError("SEED-CONDITIONAL-STATE GATE: tick0 observation mismatch")
            if first_seen[0]:raise AssertionError("tick zero was dispatched twice")
            first_seen[0]=True
            return tuple(first)
        if continuation=="P1_nearest":
            return nearest_batch(o,candidates,geometry)
        # CP-SAT sees public BusinessObserver history only, not scenario future.
        current=observer.observer  # lifecycle already updated by on_observation
        del current
        # Store last rich observed state via on_observation wrapper below.
        rich=rich_obs[0]
        if rich.tick!=o.tick:raise AssertionError("stale decision observation")
        return solver.proposals(rich,candidates)[0]
    rich_obs=[None]
    def observed(o):
        rich_obs[0]=observe(o)
    started=time.perf_counter()
    result=BusinessSimulator(business,"world",seed,collect_metrics=True).run(
        dispatch_policy=dispatch,on_observation=observed)
    if not first_seen[0]:raise AssertionError("never executed frozen first action")
    val=restricted_flow(result,business)
    val.update(seed=seed,continuation=continuation,
               cpu_wall_seconds=time.perf_counter()-started,
               solver_calls=solver.calls if solver is not None else 0,
               solver_fallbacks=solver.fallbacks if solver is not None else 0,
               safety_capacity_violations=len(resource_audit(result,business.rules)["violations"]))
    if val["safety_capacity_violations"]:
        raise AssertionError("unsafe capacity branch")
    return val


def rank(costs):
    return sorted(range(len(costs)),key=lambda i:(costs[i],i))


def ranks_from_scores(scores):
    return sorted(range(len(scores)),key=lambda i:(tuple(scores[i]),i))


def gap_error(model_costs, ref_costs):
    # Model primary score and restricted mean flow do NOT have the same
    # intercept/units, so compare pairwise centered *differences*, not raw MAE.
    return statistics.mean(abs((a-model_costs[0])-(b-ref_costs[0]))
                           for a,b in zip(model_costs,ref_costs))


def score_comparison(saved,reference):
    ranks={}
    n=len(reference)
    b=min(reference)
    for horizon in KNOWN_HORIZONS:
        w=saved["windows"][str(horizon)]
        for arm in ("Rule","V4"):
            key=arm+"_h"+str(horizon)
            records=w["runs"][arm]["records"]
            scores=[r["score"] for r in records]
            vals=[float(s[0]) for s in scores]
            ordered=ranks_from_scores(scores)
            ranks[key]={"rank":ordered,"choice":ordered[0],"regret_vs_reference":reference[ordered[0]]-b,
                        "rank_exact":ordered==rank(reference),
                        "pairwise_gap_mae_nominal_only":gap_error(vals,reference),
                        "primary_scores":vals,"score_tuples":scores,
                        "forecast_steps":n*3*horizon,
                        "forecast_wall_instrumented_seconds":w["runs"][arm]["cpu_seconds"]}
    return ranks


def prepare_case(spec,config,archive_case):
    business,geo=case_for(spec,config)
    original_policy=policy_for("Rule",business,geo,None,None)
    history=[]
    factual=BusinessSimulator(business,"world",spec["seed"],collect_metrics=True).run(
       dispatch_policy=original_policy,on_observation=original_policy.observer.observe,
       on_transition=history.append)
    raw=rich_rows(history)
    states={r.observation.tick:r.observation for r in raw}
    assert len(archive_case["fixed_states"])==1, "unexpected number of fixed states"
    frozen=archive_case["fixed_states"][0]
    assert frozen["tick"]==0
    obs=states[0]
    plans=BusinessCPSAT(geo,business.rules,candidate_limit=4,
      time_limit_seconds=SOLVER_SECONDS,seed=1810).proposals(obs,candidates_for(obs))
    if [[asdict(a) for a in p] for p in plans]!=frozen["candidates"]:
        raise AssertionError("FROZEN CANDIDATE PARITY FAILED")
    return business,geo,factual,history,obs,plans,frozen


def run(dispatch_path,replay_path,output,phase):
    from experiments.world_model_integration.decision_sensitivity.postprocess_replay import summarize
    dispatch=json.loads(Path(dispatch_path).read_text())
    with gzip.open(replay_path,"rt") if str(replay_path).endswith(".gz") else open(replay_path) as f:
        replay=json.load(f)
    verified=summarize(replay)  # checks original 5-step parity, samples/RNG, calls
    if verified["checkpoint_sha256"]!=V4_SHA:raise AssertionError("changed model evidence")
    if [c["seed"] for c in dispatch["cases"]]!=list(SOURCE_SEEDS):
        raise AssertionError("changed frozen case seeds")
    config=json.loads(CONFIG.read_text())
    started=time.monotonic()
    report={"scope":"DEVELOPMENT_OFFLINE_COUNTERFACTUAL_NOT_NEW_CONFIRMATION",
       "phase":phase,"objective":"full_business_restricted_mean_flow_all_configured_jobs",
       "policies":["P1_nearest","P2_receding_cpsat"],
       "seed_offsets":SEED_OFFSETS,"max_external_samples":8,
       "v4_sha256":V4_SHA,
       "dispatch_source_sha256":digest(dispatch_path),
       "replay_source_sha256":digest(replay_path),
       "config_sha256":digest(CONFIG),
       "model_cost_source":"FROZEN_PR18_5_10_20_ROLLOUT_NO_NEW_NETWORK_PREDICTIONS",
       "max_wall_seconds":CPU_WALL_BUDGET_SECONDS,
       "cases":[]}
    for ix,archive_case in enumerate(dispatch["cases"]):
        seed=archive_case["seed"]
        spec=next(x for x in config["groups"]["independent_in"] if x["seed"]==seed)
        business,geo,factual,raw,obs,plans,frozen=prepare_case(spec,config,archive_case)
        # Exact original historical P1 branch oracle reproduction, using the
        # existing restart-and-prefix-check replay_branch implementation.
        archive_costs=frozen["factual_counterfactual_flow"]
        gate=[]
        for ci,p in enumerate(plans):
            prior=replay_branch(business,seed,factual,raw,0,p)
            got=prior["flow"]
            if got is None or archive_costs[ci] is None or abs(got-archive_costs[ci])>1e-9:
                raise AssertionError(f"historical first-seed P1 differs: {seed}/{ci} {got} vs {archive_costs[ci]}")
            gate.append({"candidate":ci,"legacy_flow":got,"archive_flow":archive_costs[ci],
                         "legacy_status":prior["status"]})
        # P1 & P2 first seed preflight; deterministic twin replay for candidate
        # zero demonstrates full internal-history and event reproducibility.
        first={}
        for policy in ("P1_nearest","P2_receding_cpsat"):
            first[policy]=[]
            for ci,p in enumerate(plans):
                v=run_branch(business,geo,seed,p,policy,raw[0].observation)
                first[policy].append(v)
                if ci==0:
                    duplicate=run_branch(business,geo,seed,p,policy,raw[0].observation)
                    keys=("history_sha256","events_sha256","restricted_mean_flow",
                          "capacity_wait_ticks","ticks")
                    if any(duplicate[k]!=v[k] for k in keys):
                        raise AssertionError(f"same-seed deterministic restart failed {seed}/{policy}")
                if policy=="P1_nearest" and abs(v["restricted_mean_flow"]-archive_costs[ci])>1e-9:
                    raise AssertionError(f"new P1 implementation deviates from legacy {seed}/{ci}")
        jitter=max(s.jitter_ticks or 0 for s in business.scenario.stations.values())
        # When the configured service jitter is exactly zero there is no
        # seed-driven future randomness; don't claim 8 effective samples.
        use_offsets=(0,) if jitter==0 else SEED_OFFSETS
        item={"seed":seed,"tick":0,"jitter_max":jitter,"effective_seed_offsets":use_offsets,
              "frozen_candidates":frozen["candidates"],
              "historical_p1_gate":gate,"full_internal_replay_duplicate_passed":True,
              "policies":{},"source_original_factual_flow":archive_costs}
        for policy in ("P1_nearest","P2_receding_cpsat"):
            rows=[{"candidate":i,"samples":[first[policy][i]]} for i in range(len(plans))]
            if phase=="full":
                for offset in use_offsets[1:]:
                    for ci,p in enumerate(plans):
                        if time.monotonic()-started>CPU_WALL_BUDGET_SECONDS:
                            raise TimeoutError("predeclared 1500s counterfactual budget exceeded")
                        rows[ci]["samples"].append(run_branch(business,geo,seed+offset,p,policy,raw[0].observation))
            means=[statistics.mean(x["restricted_mean_flow"] for x in row["samples"]) for row in rows]
            dispersions=[statistics.pstdev(x["restricted_mean_flow"] for x in row["samples"])
                         if len(row["samples"])>1 else 0.0 for row in rows]
            model=saved=replay["cases"][ix]
            item["policies"][policy]={"candidate_values":rows,"mean_flow":means,
                 "population_std_flow":dispersions,"rank":rank(means),
                 "mean_stage_wait":[statistics.mean(x["mean_stage_wait_per_job"] for x in r["samples"]) for r in rows],
                 "mean_capacity_wait":[statistics.mean(x["capacity_wait_ticks"] for x in r["samples"]) for r in rows],
                 "model_scores_vs_reference":score_comparison(model,means),
                 "candidate_count":len(plans),"samples_per_candidate":len(rows[0]["samples"]),
                 "reference_regret_5step_rule":means[0]-min(means)}
        p1=item["policies"]["P1_nearest"]
        p2=item["policies"]["P2_receding_cpsat"]
        item["continuation_policy_rank_changed"]=p1["rank"]!=p2["rank"]
        item["baseline_policy_best_changed"]=p1["rank"][0]!=p2["rank"][0]
        report["cases"].append(item)
        print(json.dumps({"seed":seed,"phase":phase,
             "p1_cost":p1["mean_flow"],"p2_cost":p2["mean_flow"],
             "p1_rank":p1["rank"],"p2_rank":p2["rank"],
             "samples":p1["samples_per_candidate"],
             "elapsed_seconds":round(time.monotonic()-started,2)}),flush=True)
    report["elapsed_wall_seconds"]=time.monotonic()-started
    out=Path(output)
    if out.exists():raise FileExistsError("Do not overwrite offline results")
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    return report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dispatch",required=True,type=Path)
    ap.add_argument("--replay",required=True,type=Path)
    ap.add_argument("--output",required=True,type=Path)
    ap.add_argument("--phase",choices=("preflight","full"),required=True)
    a=ap.parse_args()
    run(a.dispatch,a.replay,a.output,a.phase)


if __name__=="__main__":
    main()
