"""PR52 actual per-entity conditional future-state supervision.

Extends *already verified* PR51 same-prefix first-action interventions. Only
the true simulator sees exogenous future task releases and outage durations.
The network's graph contains only current public robots, released workflow
stages, stations, resources and legal candidate assignments.

Targets are aligned with graph nodes by current PUBLIC IDs inside the data
generator; these real names NEVER become model inputs. Newly visible tasks
get no fabricated node targets. Their future count is modeled separately.
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import asdict
import argparse
import hashlib
import json
import time

from business_transport.observation import BusinessObserver
from business_transport.simulation import BusinessSimulator,resource_audit
from experiments.business_constraints.run import lifecycle_audit
from experiments.world_model_integration.online_horizon.pr30_run import _facts
from experiments.world_model_integration.stress_pr48.pr48_scenarios import (
    PROTOCOL as PR48_PROTOCOL, build_business
)
from experiments.world_model_integration.decision_learning_pr51.pr51_labels import (
    TrueCPsatBranch,_fingerprint
)
from lab_transport.multisite_experiment import discrete_safety_violations

PROTOCOL=Path(__file__).with_name("pr52_protocol.json")
REGIMES=("congestion","multi_fault","staggered_arrivals")
SPLITS=("train","validation","test")
TARGET_DIMS=6


def validate_protocol(p):
    if p.get("protocol")!="PR52_ENTITY_CONDITIONAL_DECISION_TRAIN_20261112_V1":
        raise ValueError("wrong PR52 protocol")
    if p.get("parent_pr50_head")!="cae511bd214456ffe0e698725ae67b98c96faa10":
        raise ValueError("frozen PR50 parent changed")
    if p.get("candidate_limit")!=4 or p.get("event_horizons")!=[5,20]:
        raise ValueError("candidate/horizon changed")
    cases=p["cohort"]
    if len(cases)!=228 or len({c["seed"] for c in cases})!=228:
        raise ValueError("wrong cohort size or duplicate seeds")
    if any(s["seed"]<32000000 for s in cases):
        raise ValueError("reused older seed namespace")
    for regime in REGIMES:
        for split,n in (("train",48),("validation",12),("test",16)):
            group=[s for s in cases if s["regime"]==regime and s["split"]==split]
            if len(group)!=n:
                raise ValueError("wrong balance in scenario split")
            if sum(x["mechanism"]=="within_support" for x in group)!=n//2:
                raise ValueError("mechanism balance changed")
            if any(x["job_count"]!=(14 if regime=="congestion" else 12) or
                   x["secondary_station_id"]!=("C" if regime=="multi_fault" else None)
                   for x in group):
                raise ValueError("source scenario workload changed")
    return True


def anchor_ids(obs,rules):
    """Matches public_relational_graph ordering without any private fields."""
    keys=(
        [("robot",r.id) for r in sorted(obs.robots,key=lambda x:x.id)]+
        [("station",s.id) for s in sorted(obs.stations,key=lambda x:x.id)]+
        [("resource",r.id) for r in sorted(rules.resources,key=lambda x:x.id)]+
        [("task",s.id) for s in sorted(obs.samples,key=lambda x:(x.release_tick,x.id))]
    )
    # Original graph orders robots, stations, resources, tasks respectively.
    return keys


def _future_nodes(public,anchor):
    robots={r.id:r for r in public.robots}
    stations={s.id:s for s in public.stations}
    tasks={s.id:s for s in public.samples}
    group_occupancy={}
    group_wait={}
    group_approach={}
    for s in public.stations:
        key=getattr(s,"resource_id","")
        if key:
            group_occupancy[key]=max(group_occupancy.get(key,0),int(s.resource_occupied))
            group_wait[key]=group_wait.get(key,0)+int(s.waiting_pickups)
            group_approach[key]=group_approach.get(key,0)+int(s.approaching_pickups)
    resources={r.id:r for r in anchor["rules"].resources}
    features=[]
    masks=[]
    for kind,key in anchor["node_keys"]:
        v=[0.]*TARGET_DIMS
        mask=[0.]*TARGET_DIMS
        if kind=="robot":
            r=robots.get(key)
            if r is not None:
                x,y=list(r.position)
                v=[x/7.,y/5.,float(r.task_id is not None),
                   float(r.phase=="to_pickup"),float(r.phase=="to_dropoff"),
                   float(r.service_elapsed is not None)]
                mask=[1.]*TARGET_DIMS
        elif kind=="task":
            s=tasks.get(key)
            if s is not None:
                v=[float(s.status=="waiting"),float(s.status=="assigned"),
                   float(s.status=="in_transit"),float(s.status=="completed"),
                   float(s.robot_id is not None),min(1.,s.stage_index/4.)]
                mask=[1.]*TARGET_DIMS
        elif kind=="station":
            s=stations.get(key)
            if s is not None:
                cap=max(1,int(getattr(s,"resource_capacity",1)))
                v=[float(s.device_available),float(s.available),
                   min(1.,s.resource_occupied/cap),float(s.capacity_blocked),
                   min(1.,s.waiting_pickups/10.),min(1.,s.approaching_pickups/10.)]
                mask=[1.]*TARGET_DIMS
        elif kind=="resource":
            r=resources.get(key)
            if r is not None:
                cap=max(1,int(r.capacity))
                occ=min(1.,group_occupancy.get(key,0)/cap)
                v=[occ,1.-occ,min(1.,group_wait.get(key,0)/10.),
                   min(1.,group_approach.get(key,0)/10.),0.,0.]
                mask=[1.,1.,1.,1.,0.,0.]
        else:
            raise ValueError("unsupported public entity")
        features.append(v)
        masks.append(mask)
    if len(features)!=len(anchor["node_keys"]):
        raise AssertionError("lost source node keys")
    visible_tasks=set(tasks)
    already_known_tasks={name for kind,name in anchor["node_keys"] if kind=="task"}
    new_stage_names=visible_tasks-already_known_tasks
    completed_known=sum(tasks[k].status=="completed"
                        for k in already_known_tasks if k in tasks)
    total_completed=sum(t.status=="completed" for t in tasks.values())
    unknown_completed=total_completed-completed_known
    if unknown_completed<0:
        raise AssertionError("negative newly visible completed state")
    return {
        "node_target":features,"node_target_mask":masks,
        "global_total_completed_stage_count":total_completed,
        "global_newly_visible_stage_count":len(new_stage_names),
        "global_new_stage_completed_count":unknown_completed,
        "public_event_observed_tick":public.tick
    }


class EntityTruthBranch(TrueCPsatBranch):
    def __init__(self,geometry,rules,seed,protocol,*,expected=None,branch_index=0):
        super().__init__(geometry,rules,seed,protocol,expected=expected,
                         branch_index=branch_index)
        self.node_anchor=None
        self.future_nodes={}

    def __call__(self,raw,legal):
        selected=super().__call__(raw,legal)
        if self.anchor is not None and self.node_anchor is None:
            public=self.observer.observe(raw)
            self.node_anchor={
                "node_keys":anchor_ids(public,self.rules),
                "rules":self.rules
            }
            if self.graph is not None:
                expected_keys=[n["id"] for n in self.graph["nodes"]]
                order=[f"{kind}:{i}" for kind in ("robot","station","resource","task")
                       for i,(k,_) in enumerate([v for v in self.node_anchor["node_keys"]
                                                  if v[0]==kind])]
                if expected_keys!=order:
                    raise AssertionError("public graph node order and aligned labels diverge")
        return selected

    def observe(self,raw):
        super().observe(raw)
        if self.anchor is None or self.node_anchor is None:
            return
        dt=raw.tick-self.anchor["tick"]
        if dt in (5,20):
            public=self.observer.observe(raw)
            self.future_nodes[str(dt)]=_future_nodes(public,self.node_anchor)


def episode(spec,scenario,study,*,expected=None,branch_index=0):
    business,geometry=build_business(spec,scenario)
    policy=EntityTruthBranch(geometry,business.rules,spec["seed"],study,
                             expected=expected,branch_index=branch_index)
    world=BusinessSimulator(business,"world",spec["seed"],collect_metrics=True).run(
        dispatch_policy=policy,on_observation=policy.observe)
    resource=resource_audit(world,business.rules)
    if resource["violations"] or discrete_safety_violations(world) or lifecycle_audit(world["history"]):
        raise AssertionError("a factual branch is unsafe")
    done=_facts(world,business)
    return policy,{
        "branch_index":branch_index,
        "factual_makespan_ticks":world["summary"]["ticks"],
        "true_all_jobs_complete":world["summary"]["status"]=="completed",
        "late_jobs":done["late_jobs"],
        "future_entity_states":policy.future_nodes,
        "real_factual_events_sha256":_fingerprint(world["events"])
    }


def collect_case(spec,study,scenario):
    reference,ref=episode(spec,scenario,study)
    if reference.anchor is None:
        return {"id":spec["id"],"seed":spec["seed"],"regime":spec["regime"],
                "mechanism":spec["mechanism"],"split":spec["split"],
                "labelable":False,"reason":"no_multi_candidate_at_registered_tick"}
    expected={**reference.anchor,
        "observation_sha256":reference.anchor_fingerprint,
        "plans_sha256":reference.anchor_plan_fingerprint}
    original_prefix=[x for x in reference.trace if x["tick"]<=expected["tick"]]
    outcomes=[ref]
    for idx in range(1,expected["candidate_count"]):
        counter,other=episode(spec,scenario,study,expected=expected,branch_index=idx)
        replay_prefix=[x for x in counter.trace if x["tick"]<=expected["tick"]]
        if len(replay_prefix)!=len(original_prefix):
            raise AssertionError("different counterfactual prefix length")
        for a,b in zip(original_prefix,replay_prefix):
            if (a["tick"]!=b["tick"] or a["observation_sha256"]!=b["observation_sha256"]
                or a["plans_sha256"]!=b["plans_sha256"]
                or (a["tick"]!=expected["tick"] and a["selected_plan"]!=b["selected_plan"])):
                raise AssertionError("not identical actual state and legal action history")
        if counter.node_anchor["node_keys"]!=reference.node_anchor["node_keys"]:
            raise AssertionError("not identical public entity identity alignment")
        outcomes.append(other)
    dims=len(reference.graph["nodes"])
    if dims==0 or expected["candidate_count"]!=len(outcomes):
        raise AssertionError("source graph or true candidate count invalid")
    for o in outcomes:
        for horizon,v in o["future_entity_states"].items():
            if len(v["node_target"])!=dims or len(v["node_target_mask"])!=dims:
                raise AssertionError("future target graph nodes not aligned")
    ticks=[v["factual_makespan_ticks"] for v in outcomes]
    return {
        "id":spec["id"],"seed":spec["seed"],"regime":spec["regime"],
        "mechanism":spec["mechanism"],"split":spec["split"],
        "labelable":True,"anchor":expected,
        "public_graph":reference.graph,"candidate_count":len(outcomes),
        "true_factual_outcomes":outcomes,
        "nontrivial_rank_signal":len(set(ticks))>1,
        "all_branches_completed":all(v["true_all_jobs_complete"] for v in outcomes),
        "full_entity_horizon_labels_available":all(
            set(v["future_entity_states"])=={"5","20"} for v in outcomes),
        "future_job_timetables_excluded_from_public_graph":True,
    }


def run(regime,out):
    cfg=json.loads(PROTOCOL.read_text())
    validate_protocol(cfg)
    scenario=json.loads(PR48_PROTOCOL.read_text())
    if regime not in REGIMES:
        raise ValueError("unregistered regime")
    if out.exists():
        raise FileExistsError("PR52 label data immutable")
    start=time.process_time()
    cases=[]
    for s in cfg["cohort"]:
        if s["regime"]!=regime:
            continue
        # Only original PR48 factual scenario builds from these fields.
        case=collect_case(s,cfg,scenario)
        cases.append(case)
        if len(cases)%12==0:
            print("PR52_LABEL_PROGRESS",regime,len(cases),
                  "nontrivial",sum(c.get("nontrivial_rank_signal",False) for c in cases),
                  flush=True)
    if len(cases)!=76:
        raise AssertionError("expected 48train+12val+16test in each regime")
    summary={
        "regime":regime,"scenarios":len(cases),
        "labelable":sum(c["labelable"] for c in cases),
        "complete_horizon_labels":sum(c.get("full_entity_horizon_labels_available",False)
                                      for c in cases),
        "counterfactual_nontrivial":sum(c.get("nontrivial_rank_signal",False) for c in cases),
        "true_factual_branches":sum(c.get("candidate_count",0) for c in cases),
        "per_split":{
            split:{"groups":sum(c["split"]==split for c in cases),
                   "labelable":sum(c["split"]==split and c["labelable"] for c in cases)}
            for split in SPLITS},
        "cpu_process_seconds":time.process_time()-start,
        "training":0,"gpu":0
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({
        "schema":"pr52-true-entity-world-state-paired-actions-v1",
        "protocol_sha256":hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "regime":regime,"cases":cases,"summary":summary
    },indent=2,ensure_ascii=False)+"\n")
    print("PR52_LABELS_COMPLETED",json.dumps(summary,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--regime",choices=REGIMES,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    run(args.regime,args.out)


if __name__=="__main__":
    main()
