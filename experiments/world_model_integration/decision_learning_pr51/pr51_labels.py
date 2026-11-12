"""PR51: same-public-state, same-exogenous-world counterfactual actions.

Actual simulator policy is CP-SAT proposal[0] before and after one
predeclared branch tick. At the branch tick choose *each* distinct lawful
CP-SAT proposal. No oracle state or private timetable enters features.
Simulator hidden future is allowed ONLY to generate offline outcome labels.

This is supervised DATA GENERATION, not yet learned decision-aware dynamics.
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import time

from business_transport.observation import BusinessObserver
from business_transport.dispatch import BusinessCPSAT
from business_transport.simulation import BusinessSimulator,resource_audit
from experiments.business_constraints.run import lifecycle_audit
from experiments.world_model_integration.stress_pr48.pr48_scenarios import (
    PROTOCOL as PR48_PROTOCOL,build_business
)
from experiments.world_model_integration.online_horizon.pr30_run import (
    _unique_plans,_facts
)
from experiments.world_model_integration.decision_learning_pr51.pr51_contract import (
    PROTOCOL,validate_protocol
)
from lab_transport.multisite_experiment import discrete_safety_violations

def _fingerprint(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def _plan(plan):
    return tuple((a.robot_id,a.sample_id) for a in plan)

def _public_summary(raw):
    public=raw
    return {
        "tick":public.tick,
        "completed_stage_count":sum(x.status=="completed" for x in public.samples),
        "waiting_stage_count":sum(x.status=="waiting" for x in public.samples),
        "in_transit_count":sum(x.status=="in_transit" for x in public.samples),
        "busy_resource_port_count":sum(bool(getattr(s,"resource_occupied",0)) for s in public.stations),
        "station_device_down_count":sum(not bool(getattr(s,"device_available",s.available))
                                         for s in public.stations),
    }

def public_relational_graph(obs,rules,plans):
    """Public graph + typed candidate action edges; no private schedule or seed."""
    robots=sorted(obs.robots,key=lambda x:x.id)
    stations=sorted(obs.stations,key=lambda x:x.id)
    tasks=sorted(obs.samples,key=lambda x:(x.release_tick,x.id))
    resources=sorted(rules.resources,key=lambda r:r.id)
    robot_id={r.id:f"robot:{i}" for i,r in enumerate(robots)}
    station_id={s.id:f"station:{i}" for i,s in enumerate(stations)}
    task_id={s.id:f"task:{i}" for i,s in enumerate(tasks)}
    resource_id={r.id:f"resource:{i}" for i,r in enumerate(resources)}
    nodes=[]
    edges=[]
    for i,r in enumerate(robots):
        nodes.append({"id":robot_id[r.id],"kind":"robot",
                      "pos":list(r.position),"phase":r.phase,
                      "busy":r.task_id is not None,
                      "service_elapsed":r.service_elapsed})
        if r.task_id in task_id:
            edges.append({"src":robot_id[r.id],"dst":task_id[r.task_id],
                          "kind":"currently_assigned"})
    for s in stations:
        nodes.append({"id":station_id[s.id],"kind":"station",
                      "pos":list(s.position),
                      "device_available":bool(getattr(s,"device_available",s.available)),
                      "resource_occupied":int(getattr(s,"resource_occupied",0) or 0),
                      "waiting_pickups":int(getattr(s,"waiting_pickups",0) or 0),
                      "approaching_pickups":int(getattr(s,"approaching_pickups",0) or 0)})
        r=rules.station_resources[s.id]
        edges.append({"src":station_id[s.id],"dst":resource_id[r.id],
                      "kind":"uses_resource"})
    for r in resources:
        nodes.append({"id":resource_id[r.id],"kind":"resource","capacity":r.capacity})
    seen_job={}
    for s in tasks:
        nodes.append({
            "id":task_id[s.id],"kind":"task",
            "status":s.status,"workflow":getattr(s,"workflow",""),
            "stage_index":int(getattr(s,"stage_index",0)),
            "released_age":max(0,obs.tick-s.release_tick),
            "deadline_slack":None if s.deadline_tick is None else s.deadline_tick-obs.tick,
        })
        edges.append({"src":task_id[s.id],"dst":station_id[s.pickup],"kind":"pickup_at"})
        edges.append({"src":task_id[s.id],"dst":station_id[s.dropoff],"kind":"dropoff_at"})
        j=getattr(s,"job_id",None)
        if j and j in seen_job:
            edges.append({"src":seen_job[j],"dst":task_id[s.id],
                          "kind":"same_workflow_successor"})
        if j:
            seen_job[j]=task_id[s.id]
    candidates=[]
    for plan in plans:
        candidate=[]
        for action in plan:
            candidate.append({"src":robot_id[action.robot_id],
                              "dst":task_id[action.sample_id],
                              "kind":"proposed_robot_task_assignment"})
        candidates.append(candidate)
    return {"tick":obs.tick,"nodes":nodes,"edges":edges,
            "candidate_actions":candidates,
            "source":"observed public state and CP-SAT legal actions, not simulator future"}

class TrueCPsatBranch:
    def __init__(self,geometry,rules,seed,protocol,*,expected=None,branch_index=0):
        self.observer=BusinessObserver()
        self.solver=BusinessCPSAT(
            geometry,rules,candidate_limit=protocol["counterfactual_max_candidates"],
            time_limit_seconds=0.08,deterministic_time_limit=0.04,seed=1810)
        self.rules=rules
        self.expected=expected
        self.branch_index=branch_index
        self.anchor=None
        self.anchor_fingerprint=None
        self.anchor_plan_fingerprint=None
        self.graph=None
        self.public_future={}
        self.trace=[]
        self.earliest_multi=None

    def observe(self,raw):
        rich=self.observer.observe(raw)
        if self.anchor is not None and raw.tick in (
            self.anchor["tick"]+5,self.anchor["tick"]+20):
            self.public_future[str(raw.tick-self.anchor["tick"])]=_public_summary(rich)

    def __call__(self,raw,legal):
        obs=self.observer.observe(raw)
        plans=_unique_plans(self.solver,obs,legal)
        snap=_fingerprint(asdict(obs))
        proposed=_fingerprint([_plan(p) for p in plans])
        if self.expected is None:
            if (len(plans)>1 and
                self.earliest_multi is None):
                self.earliest_multi=obs.tick
            if (len(plans)>1 and obs.tick>=7 and obs.tick<=70
                and self.anchor is None):
                self.anchor={"tick":obs.tick,"candidate_count":len(plans)}
                self.anchor_fingerprint=snap
                self.anchor_plan_fingerprint=proposed
                self.graph=public_relational_graph(obs,self.rules,plans)
        elif obs.tick<=self.expected["tick"]:
            if obs.tick==self.expected["tick"]:
                if snap!=self.expected["observation_sha256"] or proposed!=self.expected["plans_sha256"]:
                    raise AssertionError("counterfactual branch does not share exact public state AND candidates")
                if not 0<=self.branch_index<len(plans):
                    raise ValueError("unregistered branch index")
                self.anchor={"tick":obs.tick,"candidate_count":len(plans)}
        selected_index=(self.branch_index if self.expected is not None
                        and obs.tick==self.expected["tick"] else 0)
        self.trace.append({
            "tick":obs.tick,"selected_index":selected_index,
            "selected_plan":_plan(plans[selected_index]),
            "observation_sha256":snap,"plans_sha256":proposed
        })
        return plans[selected_index]

def _episode(spec,scenario_protocol,study_protocol,*,expected=None,index=0):
    business,geometry=build_business(spec,scenario_protocol)
    policy=TrueCPsatBranch(geometry,business.rules,spec["seed"],study_protocol,
                           expected=expected,branch_index=index)
    actual=BusinessSimulator(business,"world",spec["seed"],collect_metrics=True).run(
        dispatch_policy=policy,on_observation=policy.observe)
    safety=resource_audit(actual,business.rules)["violations"]
    if safety or discrete_safety_violations(actual) or lifecycle_audit(actual["history"]):
        raise AssertionError("true branch violated motion/capacity/workflow constraints")
    costs=_facts(actual,business)
    return policy,{
        "makespan_ticks":actual["summary"]["ticks"],
        "complete":actual["summary"]["status"]=="completed",
        "completed_jobs":costs["completed_jobs"],
        "late_jobs":costs["late_jobs"],
        "capacity_wait_ticks":resource_audit(actual,business.rules)["capacity_wait_ticks"],
        "true_future_public_snapshots":policy.public_future,
        "factual_event_sha256":_fingerprint(actual["events"]),
    }

def collect_case(spec,study,scenario_protocol):
    policy,ref=_episode(spec,scenario_protocol,study)
    if policy.anchor is None:
        return {
            "id":spec["id"],"seed":spec["seed"],"regime":spec["regime"],
            "mechanism":spec["mechanism"],"split":spec["split"],
            "labelable":False,"first_any_multi_tick":policy.earliest_multi,
            "reason":"no_legal_multiple_candidate_at_or_after_fault_tick7",
            "baseline_true_makespan_ticks":ref["makespan_ticks"]
        }
    anchor=policy.anchor
    selected_tick=anchor["tick"]
    expected={
        **anchor,"observation_sha256":policy.anchor_fingerprint,
        "plans_sha256":policy.anchor_plan_fingerprint
    }
    ref["index"]=0
    outcomes=[ref]
    prior=[x for x in policy.trace if x["tick"]<=selected_tick]
    if not prior or prior[-1]["tick"]!=selected_tick:
        raise AssertionError("missing actual anchor in baseline prefix")
    for index in range(1,anchor["candidate_count"]):
        alternate_policy,alternative=_episode(
            spec,scenario_protocol,study,expected=expected,index=index)
        observed=[x for x in alternate_policy.trace if x["tick"]<=selected_tick]
        if len(observed)!=len(prior):
            raise AssertionError("counterfactual prefix dispatch length changed")
        for baseline,new in zip(prior,observed):
            if (baseline["tick"]!=new["tick"] or
                baseline["observation_sha256"]!=new["observation_sha256"] or
                baseline["plans_sha256"]!=new["plans_sha256"] or
                (baseline["tick"]!=selected_tick and baseline["selected_plan"]!=new["selected_plan"])):
                raise AssertionError("same-prefix factual state or candidate list changed")
        alternative["index"]=index
        outcomes.append(alternative)
    vals=[v["makespan_ticks"] for v in outcomes]
    comparable=all(v["complete"] for v in outcomes)
    return {
        "id":spec["id"],"seed":spec["seed"],"regime":spec["regime"],
        "mechanism":spec["mechanism"],"split":spec["split"],
        "labelable":True,
        "anchor":expected,
        "public_graph":policy.graph,
        "candidate_count":len(outcomes),
        "true_factual_outcomes":outcomes,
        "paired_makespan_tick_deltas_vs_first":[x-vals[0] for x in vals],
        "nonzero_candidate_effect":len(set(vals))>1,
        "all_branches_completed":comparable,
        "first_proposal_baseline_tick":vals[0],
        "no_true_future_in_public_graph":True,
    }

def run(regime,out):
    study=json.loads(PROTOCOL.read_text())
    scenario=json.loads(PR48_PROTOCOL.read_text())
    validate_protocol(study)
    if out.exists():
        raise FileExistsError("counterfactual evidence must not overwrite")
    if regime not in ("congestion","multi_fault","staggered_arrivals"):
        raise ValueError("unregistered regime")
    began=time.process_time()
    cases=[]
    for spec in study["cohort"]:
        if spec["regime"]!=regime:
            continue
        if time.process_time()-began>=study["resources"]["phase0_cpu_seconds_per_regime"]:
            raise TimeoutError("precommitted CPU safety limit for factual label collection")
        # PR48 builder requires exactly these scenario features.
        spec={**spec,
              "secondary_station_id":"C" if regime=="multi_fault" else None,
              "job_count":14 if regime=="congestion" else 12}
        entry=collect_case(spec,study,scenario)
        cases.append(entry)
        print("PR51_LABEL_CASE",spec["id"],json.dumps({
            "labelable":entry["labelable"],
            "effects":entry.get("paired_makespan_tick_deltas_vs_first",[]),
        }),flush=True)
    if len(cases)!=24:
        raise AssertionError("24 completely enumerated cases per regime required")
    summary={
        "regime":regime,"scenarios":len(cases),
        "usable_counterfactual_groups":sum(c["labelable"] for c in cases),
        "nontrivial_ranked_groups":sum(c.get("nonzero_candidate_effect",False) for c in cases),
        "noncensored_groups":sum(c.get("all_branches_completed",False) for c in cases),
        "pairs":sum(c.get("candidate_count",0)-1 for c in cases),
        "by_split":{
            s:{"total":sum(c["split"]==s for c in cases),
               "labelable":sum(c["split"]==s and c["labelable"] for c in cases),
               "nontrivial":sum(c["split"]==s and c.get("nonzero_candidate_effect",False)
                                for c in cases)}
            for s in ("train","validation","test")
        },
        "cpu_process_seconds":time.process_time()-began,
        "new_gpu_training":0,
        "simulation_semantics":"Exact factual rerun with common public prefix & keyed exogenous noise, one first-action intervention, original CP-SAT continuation",
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({
        "schema":"pr51-paired-action-world-model-supervision-v1",
        "protocol_sha256":hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "regime":regime,"cases":cases,"summary":summary
    },indent=2,ensure_ascii=False)+"\n")
    print("PR51_GROUP_SUMMARY",json.dumps(summary,ensure_ascii=False),flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--regime",required=True,
                    choices=("congestion","multi_fault","staggered_arrivals"))
    ap.add_argument("--out",required=True,type=Path)
    opts=ap.parse_args()
    run(opts.regime,opts.out)

if __name__=="__main__":
    main()
