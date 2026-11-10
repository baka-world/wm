"""PR18 nonzero-tick dynamic dispatch: observable opportunities, prefix gates,
frozen Rule/V4 mechanism and single-realization counterfactuals only.

NEVER feed simulator result, hidden service countdown, future outage or future
unreleased tasks to the policy. Source cases are the original tick0 development
Train/Validation specifications, not the 16 viewed independent Test groups.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time

import torch
from business_transport.dispatch import BusinessCPSAT
from business_transport.observation import BusinessObserver
from business_transport.session_events import eligible_sessions
from business_transport.simulation import BusinessSimulator, resource_audit
from experiments.business_constraints.run import rich_rows, lifecycle_audit
from experiments.world_model_integration.value_learnability.run import frozen_specs, scenario_for
from experiments.world_model_integration.v4_dispatch import policy_for
from experiments.world_model_integration.decision_sensitivity.replay_trace import trace_policy, compare
from experiments.world_model_integration.rollout_value.evaluate import restricted_flow
from lab_transport.decision_value import candidates_for, nearest_batch
from lab_transport.multisite_experiment import discrete_safety_violations

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "protocol.json"
VALUE_PROTOCOL = ROOT.parent / "value_learnability" / "protocol.json"
PRIOR_TEST = ROOT.parent / "value_learnability" / "evidence" / "test_metrics.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()


def write_once(path,data):
    p=Path(path)
    if p.exists():
        raise FileExistsError("immutable evidence exists: "+str(p))
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")


def norm(plan):
    return tuple(sorted((a.robot_id,a.sample_id) for a in plan))


def legal_unique(plans,candidates):
    permitted=set((a.robot_id,a.sample_id) for a in candidates)
    result=[];seen=set()
    for plan in plans:
        identity=norm(plan)
        if not identity or identity in seen:
            continue
        if len(identity)!=len(set(r for r,_ in identity)) or len(identity)!=len(set(s for _,s in identity)):
            raise AssertionError("invalid repeated robot/sample matching")
        if not set(identity)<=permitted:
            raise AssertionError("CP-SAT returned candidate outside actual action space")
        seen.add(identity)
        result.append(tuple(plan))
    return result


def classify(active,multiple,competition,recovery):
    if active and multiple:
        return "S4" if recovery else "S3" if competition else "S2"
    if active:
        return "S1"
    return "S0"


def frozen_development():
    p=json.loads(VALUE_PROTOCOL.read_text())
    splits=frozen_specs(p)
    assert sorted(splits)==["test","train","validation"]
    assert len(splits["train"])==48 and len(splits["validation"])==12
    disallowed={x["seed"] for x in splits["test"]}|{43011,43015,43019}
    seen=set()
    cases=[]
    for split in ("train","validation"):
        for spec in splits[split]:
            assert spec["seed"] not in disallowed and spec["seed"] not in seen
            seen.add(spec["seed"])
            cases.append((split,spec))
    return cases


def factual_for(spec):
    business,geo=scenario_for(spec)
    raw=[]
    observer=BusinessObserver()
    result=BusinessSimulator(business,"world",spec["seed"],collect_metrics=True).run(
        dispatch_policy=lambda o,c:nearest_batch(o,c,geo),
        on_observation=observer.observe,on_transition=raw.append)
    rich=rich_rows(raw)
    assert len(rich)==len(raw)
    if resource_audit(result,business.rules)["violations"]:
        raise AssertionError("invalid factual station capacity")
    if discrete_safety_violations(result) or lifecycle_audit(result["history"]):
        raise AssertionError("invalid factual path/lifecycle")
    return business,geo,result,raw,rich


def scan_one(split,spec):
    business,geo,factual,raw,rich=factual_for(spec)
    counter=Counter()
    state_rows=[]
    from lab_transport.paper_reuse.ortools_dispatch import cp_model
    if cp_model is None:
        raise RuntimeError("CP-SAT unavailable; cannot certify current candidate set")
    prior=None
    stage_ready_by_tick=Counter(e["tick"] for e in factual["events"] if e["type"]=="stage_ready")
    resource_wait_by_tick=Counter(e["tick"] for e in factual["events"] if e["type"]=="resource_wait")
    for r in rich:
        obs=r.observation
        if obs.tick==0:
            prior=obs
            continue
        active=[x for x in obs.robots if x.service_elapsed is not None]
        moving=[x for x in obs.robots if x.task_id is not None and x.service_elapsed is None]
        available=[x for x in obs.robots if x.task_id is None]
        waiting=[x for x in obs.samples if x.status=="waiting"]
        occupied={s.resource_id:s.resource_occupied for s in obs.stations}
        blocked=[s.id for s in obs.stations if s.capacity_blocked]
        recovery=bool(prior is not None and any(
            (not getattr(old,"device_available",old.available)) and
            getattr(new,"device_available",new.available)
            for old,new in zip(prior.stations,obs.stations)))
        unavailable=[s.id for s in obs.stations if not getattr(s,"device_available",s.available)]
        allowed=tuple(candidates_for(obs))
        plans=[];solver_status=[];fallback=None;truncation=False
        if len(allowed)>=2 and available and waiting:
            solver=BusinessCPSAT(geo,business.rules,candidate_limit=4,time_limit_seconds=.5,seed=1810)
            plans=legal_unique(solver.proposals(obs,allowed),allowed)
            last=solver.log[-1]
            solver_status=last.get("statuses",[])
            fallback=last.get("fallback")
            truncation=bool(last.get("partial_truncation",False))
        multiple=len(plans)>=2
        competing=bool(blocked)
        cat=classify(bool(active),multiple,competing,recovery)
        eligible=eligible_sessions(obs,())
        queries=sum(reason is None for _,reason in eligible)
        row={
          "id":str(spec["seed"])+":"+str(obs.tick),
          "seed":spec["seed"],"split":split,"tick":obs.tick,
          "category":cat,"active":len(active),"moving":len(moving),
          "idle":len(available),"waiting":len(waiting),
          "robot_count":len(obs.robots),"visible_task_count":len(obs.samples),
          "service_progress":[{"robot":a.id,"wall":a.service_elapsed,
                "active":a.service_active_elapsed,"paused":a.service_paused_elapsed}
                for a in active],
          "resource_occupied":occupied,"capacity_blocked":blocked,
          "resource_wait_observed_previous_tick":resource_wait_by_tick.get(obs.tick-1,0),
          "unavailable_stations":unavailable,"recovery_at_tick":recovery,
          "prior_stage_ready_observed":stage_ready_by_tick.get(obs.tick-1,0),
          "raw_legal_pairs":len(allowed),
          "distinct_legal_plans":len(plans),
          "candidate_matchings":[[{"robot_id":a.robot_id,"sample_id":a.sample_id} for a in p] for p in plans],
          "solver_statuses":solver_status,"solver_fallback":fallback,"solver_truncated":truncation,
          "eligible_current_v4_queries":queries,
          "actual_potential_decision":multiple,
          "observed_contention":competing,
        }
        state_rows.append(row)
        counter[cat]+=1
        counter["nonzero_states"]+=1
        counter["dynamic_proposals"]+=int(multiple)
        counter["active_and_multiple"]+=int(bool(active) and multiple)
        counter["active_and_competing_and_multiple"]+=int(bool(active) and multiple and competing)
        counter["recovery_and_multiple"]+=int(bool(active) and multiple and recovery)
        counter["active_and_no_choice"]+=int(bool(active) and not multiple)
        counter["active_and_valid_session_query"]+=int(bool(active) and queries>0)
        counter["nonzero_any_choice"]+=int(multiple)
        prior=obs
    return {"source_seed":spec["seed"],"split":split,
            "spec_sha256":digest(spec),"factual_history_sha256":digest(factual["history"]),
            "factual_events_sha256":digest(factual["events"]),
            "factual_status":factual["summary"]["status"],
            "counter":dict(counter),"states":state_rows}


def select_states(cases,p):
    rows=[x for c in cases for x in c["states"]
          if x["tick"]>0 and x["category"] in ("S2","S3","S4")
          and x["distinct_legal_plans"]>=2]
    priority={"S4":0,"S3":1,"S2":2}
    rows.sort(key=lambda x:(priority[x["category"]],x["seed"],x["tick"]))
    count=Counter();chosen=[]
    for row in rows:
        if count[row["seed"]]>=p["max_states_per_scenario"]:
            continue
        count[row["seed"]]+=1
        chosen.append(row)
        if len(chosen)>=p["max_selected_states"]:
            break
    return chosen,{"eligible_count":len(rows),"selected_count":len(chosen),
       "eligible_by_class":dict(Counter(x["category"] for x in rows)),
       "selected_by_class":dict(Counter(x["category"] for x in chosen)),
       "unselected_count":len(rows)-len(chosen),
       "source_scenarios_selected":len(set(x["seed"] for x in chosen))}


def run_scan(output):
    p=json.loads(PROTOCOL.read_text())
    assert p["source_head"]=="08fba80bf73f6c07097d04bb40218efe7bb20366"
    assert not PRIOR_TEST.exists() or PRIOR_TEST.is_file()  # no read: disallow looking at old Test data
    start=time.monotonic()
    cases=[]
    for index,(split,spec) in enumerate(frozen_development(),1):
        case=scan_one(split,spec)
        cases.append(case)
        print("scan",index,split,spec["seed"],case["counter"],flush=True)
        if time.monotonic()-start>520:
            raise TimeoutError("CPU scan budget exhausted")
    selected,selection_stats=select_states(cases,p)
    selection_lock={
       "protocol_sha256":sha(PROTOCOL),"value_development_protocol_sha256":sha(VALUE_PROTOCOL),
       "selected":[{k:x[k] for k in ("id","seed","split","tick","category","candidate_matchings",
                                        "distinct_legal_plans","eligible_current_v4_queries")}
                   for x in selected],
       "selection_stats":selection_stats,
       "selection_rule":"category S4>S3>S2 then (seed,tick), max 2 per scenario",
       "not_selected_on_counterfactual_cost":True,"not_independent_test":True,
       "stage":"SEALED_BEFORE_V4_PREDICTION_OR_FUTURE_COST"
    }
    summary=Counter()
    for case in cases:summary.update(case["counter"])
    scan={"scope":"development_existing_train_validation_spec_replay",
          "protocol_sha256":sha(PROTOCOL),"sources":len(cases),
          "all_nonzero_states":sum(len(c["states"]) for c in cases),
          "total_by_class":{k:summary[k] for k in ("S0","S1","S2","S3","S4")},
          "counters":dict(summary),"cases":cases,
          "scan_elapsed_seconds":time.monotonic()-start}
    write_once(output/"scan.json",scan)
    write_once(output/"selection_lock.json",selection_lock)
    print("SCAN_SUMMARY",json.dumps({"total":scan["all_nonzero_states"],
           "classes":scan["total_by_class"],"selection":selection_stats}),flush=True)


def replay_exact(business,geo,seed,factual,raw,rich,target,plan):
    """Rebuild entire simulator, never clone/inject hidden internal state."""
    source={r.observation.tick:r for r in raw}
    richer={r.observation.tick:r.observation for r in rich}
    seen=[]
    observer=BusinessObserver()
    def observation_guard(o):
        r=observer.observe(o)
        if o.tick<=target:
            if o.tick not in source or o!=source[o.tick].observation:
                raise AssertionError("RAW_OBSERVATION_PREFIX_CHANGED")
            if r!=richer[o.tick]:
                raise AssertionError("ENRICHED_OBSERVATION_PREFIX_CHANGED")
            seen.append(o.tick)
    def action(o,legal):
        if o.tick<target:
            return tuple(source[o.tick].assignments)
        if o.tick==target:
            if any(a not in legal for a in plan):
                raise AssertionError("CANDIDATE_ILLEGAL_AT_FROZEN_TICK")
            return tuple(plan)
        return nearest_batch(o,legal,geo)
    result=BusinessSimulator(business,"world",seed,collect_metrics=True).run(
        dispatch_policy=action,on_observation=observation_guard)
    if len(seen)!=target+1 or sorted(seen)!=list(range(target+1)):
        raise AssertionError("PREFIX_TICKS_NOT_ALL_OBSERVED")
    if result["history"][:target+1]!=factual["history"][:target+1]:
        raise AssertionError("HIDDEN_HISTORY_PREFIX_NOT_IDENTICAL")
    original_events=[e for e in factual["events"] if e["tick"]<target]
    replayed_events=[e for e in result["events"] if e["tick"]<target]
    if original_events!=replayed_events:
        raise AssertionError("EVENT_PREFIX_NOT_IDENTICAL")
    capacity=resource_audit(result,business.rules)
    if capacity["violations"] or discrete_safety_violations(result) or lifecycle_audit(result["history"]):
        raise AssertionError("ILLEGAL_COUNTERFACTUAL_SAFETY")
    by_id={x["id"]:x for x in result["samples"]}
    for job in business.jobs:
        for left,right in zip(job["stages"],job["stages"][1:]):
            if by_id[right]["assigned_tick"] is not None and (by_id[left]["completed_tick"] is None or
                    by_id[right]["assigned_tick"]<by_id[left]["completed_tick"]):
                raise AssertionError("BROKEN_WORKFLOW_STAGE_DEPENDENCY")
    val=restricted_flow(result,business)
    visible_ids={getattr(s,"job_id","") for s in richer[target].samples}
    visible=[x for x,j in zip(val["per_job_flow"],business.jobs) if j["id"] in visible_ids]
    val.update({
       "current_visible_restricted_mean_flow":statistics.mean(visible) if visible else None,
       "visible_jobs":len(visible),
       "configured_jobs":len(business.jobs),
       "completed_jobs":sum(by_id[j["stages"][-1]]["completed_tick"] is not None for j in business.jobs),
       "event_prefix_sha256":digest(original_events),
       "prefix_history_sha256":digest(factual["history"][:target+1]),
       "source_seed":seed
    })
    return val


def probe_resampling(business,geo,source_seed,factual,raw,rich,target,plan,offset):
    # A different initial seed may change already observed/hidden service history.
    # This is a STRICT equality probe, never an alternative label generator.
    try:
        data=replay_exact(business,geo,source_seed+offset,factual,raw,rich,target,plan)
    except (AssertionError,ValueError) as exc:
        return {"offset":offset,"prefix_identical":False,"reason":str(exc)[:120],
                "usable_as_conditional_future":False}
    return {"offset":offset,"prefix_identical":True,
            "reason":"identical captured prefix for this probe; sampling contract not established",
            "usable_as_conditional_future":False,"status":"NOT_ADMITTED_AS_LABEL"}


def forecast_for(business,geo,obs,plans,checkpoint):
    results={}
    for arm in ("Rule","V4"):
        pol=policy_for(arm,business,geo,None,checkpoint)
        if pol.horizon!=5 or pol.particles!=3 or len(plans)>4:
            raise AssertionError("frozen forecast contract changed")
        results[arm]=trace_policy(pol,obs,plans,arm,geo)
    delta=compare(results["Rule"],results["V4"])
    queries=[]
    for ci,particles in enumerate(results["V4"]["traces"]):
        for pi,steps in enumerate(particles):
            for step_i,step in enumerate(steps):
                q=step["session"]
                queries.append({"candidate":ci,"particle":pi,"step":step_i,
                    "tick":step["before"]["tick"],"considered":q["queries_considered"],
                    "eligible":q["queries_eligible"],"queried":q["queries"]})
    # Preserve per-step mechanisms; never input offline outcomes into these.
    return {"rule_scores":[list(x["score"]) for x in results["Rule"]["records"]],
            "v4_scores":[list(x["score"]) for x in results["V4"]["records"]],
            "rule_components":[x["component_means"] for x in results["Rule"]["records"]],
            "v4_components":[x["component_means"] for x in results["V4"]["records"]],
            "rule_cpu_seconds":results["Rule"]["cpu_seconds"],
            "v4_cpu_seconds":results["V4"]["cpu_seconds"],
            "v4_actual_query_calls":results["V4"]["actual_model_query_calls"],
            "v4_actual_forward_batches":results["V4"]["actual_model_forward_batches"],
            "v4_fallback_reasons":results["V4"]["fallback_reasons"],
            "layered_diff":delta,"query_details":queries}


def run_evaluate(output,checkpoint):
    p=json.loads(PROTOCOL.read_text())
    if sha(checkpoint)!=p["v4_checkpoint_sha256"]:
        raise AssertionError("FROZEN_V4_SHA_MISMATCH")
    lock=json.loads((output/"selection_lock.json").read_text())
    scan=json.loads((output/"scan.json").read_text())
    assert lock["protocol_sha256"]==sha(PROTOCOL)==scan["protocol_sha256"]
    assert lock["value_development_protocol_sha256"]==sha(VALUE_PROTOCOL)
    assert lock["not_selected_on_counterfactual_cost"]
    cases_by_seed={c["source_seed"]:c for c in scan["cases"]}
    selected=lock["selected"]
    assert len(selected)<=12
    start=time.monotonic()
    outputs=[];failures=[];probes=[]
    torch.set_num_threads(1)
    for i,entry in enumerate(selected,1):
        seed=entry["seed"]
        source=cases_by_seed[seed]
        assert source["split"] in ("train","validation")
        spec=next(z for split,z in frozen_development() if z["seed"]==seed)
        business,geo,factual,raw,rich=factual_for(spec)
        if digest(factual["history"])!=source["factual_history_sha256"] or digest(factual["events"])!=source["factual_events_sha256"]:
            failures.append({"id":entry["id"],"gate":"FACTUAL_REPLAY_OR_SOURCE_DRIFT"});continue
        target=entry["tick"]
        row={r.observation.tick:r.observation for r in rich}.get(target)
        if row is None:failures.append({"id":entry["id"],"gate":"MISSING_FROZEN_TICK"});continue
        from lab_transport.transition_schema import Assignment
        plans=[tuple(Assignment(a["robot_id"],a["sample_id"]) for a in arr)
               for arr in entry["candidate_matchings"]]
        if len(plans)<2 or len(plans)>4 or len(set(norm(p) for p in plans))!=len(plans):
            failures.append({"id":entry["id"],"gate":"NO_DISTINCT_LEGAL_PLANS"});continue
        solver=BusinessCPSAT(geo,business.rules,candidate_limit=4,time_limit_seconds=.5,seed=1810)
        again=legal_unique(solver.proposals(row,candidates_for(row)),candidates_for(row))
        if [norm(x) for x in again]!=[norm(x) for x in plans]:
            failures.append({"id":entry["id"],"gate":"CANDIDATE_SET_DRIFT"});continue
        raw_gates=[]
        try:
            for ci,plan in enumerate(plans):
                truth=replay_exact(business,geo,seed,factual,raw,rich,target,plan)
                if ci==0:
                    twin=replay_exact(business,geo,seed,factual,raw,rich,target,plan)
                    for key in ("history_sha256","events_sha256","restricted_mean_flow"):
                        if truth[key]!=twin[key]:
                            raise AssertionError("DUPLICATE_SAME_SEED_NOT_DETERMINISTIC")
                raw_gates.append(truth)
        except (AssertionError,ValueError) as exc:
            failures.append({"id":entry["id"],"gate":"PREFIX_OR_COUNTERFACTUAL_FAILURE",
                            "error":str(exc)[:200]})
            continue
        probe=probe_resampling(business,geo,seed,factual,raw,rich,target,plans[0],p["probe_offset"])
        probes.append({"id":entry["id"],**probe})
        # Scores and tracing are read only after the candidate list is frozen.
        try:
            predicted=forecast_for(business,geo,row,plans,checkpoint)
        except (AssertionError,ValueError,RuntimeError,TypeError) as exc:
            failures.append({"id":entry["id"],"gate":"MODEL_FORECAST_FAILURE",
                            "error":str(exc)[:200]})
            continue
        flows=[x["restricted_mean_flow"] for x in raw_gates]
        best=min(flows)
        rule_choice=predicted["layered_diff"]["rule_rank"][0]
        v4_choice=predicted["layered_diff"]["v4_rank"][0]
        predictions=predicted
        outputs.append({
           "id":entry["id"],"seed":seed,"split":entry["split"],"tick":target,
           "category":entry["category"],"candidate_count":len(plans),
           "current_eligible_v4_query_count":sum(r is None for _,r in eligible_sessions(row,())),
           "current_active_sessions":sum(r.service_elapsed is not None for r in row.robots),
           "plans":entry["candidate_matchings"],"same_seed_single_realization":True,
           "model":predictions,
           "single_realization_candidate_costs":raw_gates,
           "candidate_flow_costs":flows,
           "rank_from_actual_finite_realization":sorted(range(len(flows)),key=lambda j:(flows[j],j)),
           "rule_choice":rule_choice,"v4_choice":v4_choice,
           "rule_regret":flows[rule_choice]-best,"v4_regret":flows[v4_choice]-best,
           "paired_regret_rule_minus_v4":flows[rule_choice]-flows[v4_choice],
           "counterfactual_label_scope":"single-realization counterfactual, not conditional expectation",
        })
        print("AUDIT_STATE",i,entry["id"],"rule",rule_choice,"v4",v4_choice,
              "queries",predicted["v4_actual_query_calls"],
              "realized_costs",flows,flush=True)
        if time.monotonic()-start>950:
            raise TimeoutError("dynamic audit finite CPU evaluation budget exceeded")
    counters=Counter()
    for x in outputs:
        for key in ("v4_actual_query_calls","v4_actual_forward_batches"):
            counters[key]+=x["model"][key]
        for key,val in x["model"]["layered_diff"]["counts"].items():
            if isinstance(val,(int,float)):counters[key]+=val
    changes=sum(x["rule_choice"]!=x["v4_choice"] for x in outputs)
    improvements=[x["paired_regret_rule_minus_v4"] for x in outputs]
    by_seed=defaultdict(list)
    for x in outputs:by_seed[str(x["seed"])].append(x["paired_regret_rule_minus_v4"])
    if not selected:
        status="BLOCKED_NO_DYNAMIC_OPPORTUNITIES"
    elif not outputs:
        status="BLOCKED_NO_VALID_PREFIX_OR_MODEL"
    elif any(x["rule_choice"]!=x["v4_choice"] and x["paired_regret_rule_minus_v4"]>0 for x in outputs):
        status="GO_MECHANISM_ONLY_NOT_INDEPENDENT_EVIDENCE"
    elif changes==0:
        status="NO_GO_DEVELOPMENT_NO_CHANGED_ACTION"
    else:
        status="NO_GO_DEVELOPMENT_NO_POSITIVE_REALIZED_VALUE"
    summary={
       "source":"existing_train_validation_only","selected":len(selected),
       "valid_evaluated_states":len(outputs),"rejected_states":len(failures),
       "rule_v4_different_actions":changes,
       "same_action_fraction":1-changes/len(outputs) if outputs else None,
       "total_actual_v4_queries":counters["v4_actual_query_calls"],
       "total_actual_v4_batches":counters["v4_actual_forward_batches"],
       "mechanism_counts":dict(counters),
       "realization_regret_rule_mean":statistics.mean(x["rule_regret"] for x in outputs) if outputs else None,
       "realization_regret_v4_mean":statistics.mean(x["v4_regret"] for x in outputs) if outputs else None,
       "paired_rule_minus_v4_by_original_scenario":{k:statistics.mean(v) for k,v in by_seed.items()},
       "conditioning_probes":probes,
       "legitimate_multiple_futures_estimated":False,
       "development_only":True,
       "not_independent_test":True,
       "cpu_evaluate_elapsed_seconds":time.monotonic()-start,
       "verdict":status
    }
    write_once(output/"dynamic_detail.json",outputs)
    write_once(output/"replay_gate_failures.json",failures)
    write_once(output/"summary.json",summary)
    print("DYNAMIC_SUMMARY",json.dumps(summary),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--phase",required=True,choices=("scan","evaluate"))
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--v4",type=Path)
    a=parser.parse_args()
    if a.phase=="scan":
        run_scan(a.out)
    else:
        if a.v4 is None:raise ValueError("--v4 required for frozen prediction")
        run_evaluate(a.out,a.v4)


if __name__=="__main__":
    main()
