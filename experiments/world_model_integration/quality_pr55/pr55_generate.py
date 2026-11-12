"""PR55 quality-first real public world-state dataset pilot.

Full factual BusinessSimulator episodes are preserved. Training-only
curation marks rare action-relevant changes and negative controls; it does
NOT censor the original trajectory or touch any independent evaluation
split. There is NO model training, no GPU, no simulated metric invented.

All model input records use audited PR54 public t-3..t and actually selected
CP-SAT action. Changes are computed from t+1 LABELS for quality AUDIT only,
never retroactively inserted into the model's historical feature tensor.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import time

from experiments.world_model_integration.stress_pr48.pr48_scenarios import (
    PROTOCOL as STRESS_PROTOCOL, build_business
)
from experiments.world_model_integration.temporal_pr54.pr54_public_sequence import (
    PublicTransitionCollector,REGIMES
)
from experiments.world_model_integration.decision_learning_pr51.pr51_labels import TrueCPsatBranch
from business_transport.simulation import BusinessSimulator,resource_audit
from experiments.business_constraints.run import lifecycle_audit
from lab_transport.multisite_experiment import discrete_safety_violations
from lab_transport.dynamics_model import Geometry

PROTOCOL=Path(__file__).with_name("protocol.json")
RARE=("station_fault_onset","station_recovery","new_task_release",
      "capacity_transition")
POSITIVE=("assignment","task_status_change","task_completed",
          "robot_phase_change","queue_transition")
FLAGS=RARE+POSITIVE


def sha(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def validate(cfg):
    if cfg.get("protocol")!="PR55_QUALITY_FIRST_PUBLIC_WORLD_DATA_PILOT_V1_20261112":
        raise ValueError("wrong frozen data-quality protocol")
    if cfg.get("parent_pr_head")!="3a78396189720e844d79b28a2d04dc1a38100e6f":
        raise ValueError("source contract hash changed")
    samples=cfg["pilot_cases"]
    if len(samples)!=18 or len({x["seed"] for x in samples})!=18:
        raise ValueError("expected 18 new independent development scenarios")
    if any(x["split"]!="pilot_development_only" for x in samples):
        raise ValueError("not a blind test or preselected model validation cohort")
    for regime in REGIMES:
        group=[x for x in samples if x["regime"]==regime]
        if len(group)!=6 or len({x["profile"] for x in group})!=6:
            raise ValueError("quality profiles underrepresented")
        if sum(x["mechanism"]=="within_support" for x in group)!=3:
            raise ValueError("outage support balance changed")
        if regime=="multi_fault" and any(
            x["station_id"]==x["secondary_station_id"] for x in group):
            raise ValueError("multi-fault primary and secondary must be distinct")
        if regime!="multi_fault" and any(
            x["secondary_station_id"] is not None for x in group):
            raise ValueError("unexpected secondary outage in single-fault regime")
    if cfg["resource_budget"]["gpu"] or cfg["resource_budget"]["neural_training"]:
        raise ValueError("no GPU or model fitting authorized")
    if cfg["max_ticks"]!=260:
        raise ValueError("full-episode collection must not be truncated")
    return True


def build_quality_business(spec,stress):
    """Change actual FACTUAL simulator profiles, never public graph labels."""
    stress=json.loads(json.dumps(stress))
    regime=spec["regime"]
    profile=spec["profile"]
    if profile=="shifted_releases":
        # All release changes are simulator-private until public actual tick.
        stress["predefined_scenario_mechanisms"][regime]["release_ticks"]=(
            [0,2] if regime=="congestion" else
            [0,4,9] if regime=="multi_fault" else [0,3,11,23]
        )
    business,geo=build_business(spec,stress)
    scenario=business.scenario
    if profile=="capacity_relief":
        resources=tuple(replace(r,capacity=2) if r.id=="A" else r
                        for r in business.rules.resources)
        business=replace(business,rules=replace(business.rules,resources=resources))
    elif profile=="fewer_robots":
        scenario=replace(scenario,robots=tuple(scenario.robots[:2]))
        business=replace(business,scenario=scenario)
    elif profile=="slow_service":
        stations={
            sid:(replace(s,service_ticks=s.service_ticks+2,prior_ticks=s.prior_ticks+2)
                 if sid in ("B0","C") else s)
            for sid,s in scenario.stations.items()
        }
        scenario=replace(scenario,stations=stations)
        business=replace(business,scenario=scenario)
    elif profile not in ("baseline","higher_demand","shifted_releases"):
        raise ValueError("unsupported new synthetic mechanism")
    # Source remains exactly the unchanged 7x5 simulator geometry.
    if scenario.grid.width!=7 or scenario.grid.height!=5:
        raise AssertionError("geometry was unexpectedly modified")
    if len(business.scenario.robots) not in (2,3):
        raise AssertionError("unexpected robot cohort")
    return business,Geometry(business.scenario.grid)


def flag_transition(row):
    """Derive quality strata from factual targets, NEVER from prediction input."""
    inp=row["input"]
    truth=row["target"]
    nodes=inp["graph"]["nodes"]
    old=inp["node_persistence"]
    new=truth["node_target"]
    masks=truth["node_mask"]
    if not (len(nodes)==len(old)==len(new)==len(masks)):
        raise AssertionError("lost public entity label alignment")
    flags=set()
    if inp["graph"]["candidate_actions"][0]:
        flags.add("assignment")
    if truth["new_visible_task_stages"]>0:
        flags.add("new_task_release")
    eps=1e-8
    for n,prior,after,mask in zip(nodes,old,new,masks):
        kind=n["kind"]
        if kind=="station":
            if mask[0]:
                if prior[0]>0.5 and after[0]<0.5:
                    flags.add("station_fault_onset")
                elif prior[0]<0.5 and after[0]>0.5:
                    flags.add("station_recovery")
            if mask[2] and abs(prior[2]-after[2])>eps:
                flags.add("capacity_transition")
            if mask[4] and (abs(prior[4]-after[4])>eps or
                            abs(prior[5]-after[5])>eps):
                flags.add("queue_transition")
        elif kind=="task" and mask[3]:
            if any(abs(prior[j]-after[j])>eps for j in range(4)):
                flags.add("task_status_change")
            if prior[3]<0.5 and after[3]>0.5:
                flags.add("task_completed")
        elif kind=="robot" and mask[4]:
            if any(abs(prior[j]-after[j])>eps for j in (2,3,4,5)):
                flags.add("robot_phase_change")
            if any(abs(prior[j]-after[j])>eps for j in (0,1)):
                flags.add("robot_movement")
    return sorted(flags)


def semantic_fingerprint(row):
    """Ignore age/deadline clock drift to detect redundant PUBLIC inputs."""
    graph=row["input"]["graph"]
    useful=[]
    for n in graph["nodes"]:
        kind=n["kind"]
        if kind=="task":
            values=(n["status"],n["workflow"],n["stage_index"])
        elif kind=="robot":
            values=(tuple(n["pos"]),n["phase"],n["busy"])
        elif kind=="station":
            values=(n["device_available"],n["resource_occupied"],
                    n["waiting_pickups"],n["approaching_pickups"])
        else:
            values=(n["capacity"],)
        useful.append((kind,values))
    return sha({"nodes":useful,"edges":graph["edges"],
                "executed_action":graph["candidate_actions"][0]})


def spaced_pick(indices,num):
    if num<=0 or not indices:
        return []
    if len(indices)<=num:
        return indices
    if num==1:
        return [indices[len(indices)//2]]
    return [indices[(j*(len(indices)-1))//(num-1)] for j in range(num)]


def curated_indices(rows,cfg):
    """TRAIN-ONLY outcome-stratified preview; raw unbiased stream preserved."""
    n=len(rows)
    maximum=cfg["max_per_episode"]
    flagsets=[set(r["quality"]["flags"]) for r in rows]
    chosen=set()
    # Must cover rare cases across ENTIRE episode, not front-loaded tick0..56.
    for flag in RARE:
        pool=[i for i,v in enumerate(flagsets) if flag in v]
        chosen.update(spaced_pick(pool,6))
    for flag in POSITIVE:
        pool=[i for i,v in enumerate(flagsets) if flag in v]
        chosen.update(spaced_pick(pool,6))
    if len(chosen)>maximum:
        # Hard-predeclared priorities, does not optimize model metrics.
        chosen=set(sorted(chosen,key=lambda i:(-sum(f in flagsets[i] for f in RARE),i))
                   [:maximum])
    anchors=sorted(chosen)
    # Include nearest public context of true dynamic events when capacity allows.
    neighbors=[]
    for i in anchors:
        for delta in (-cfg["include_neighboring_ticks"],cfg["include_neighboring_ticks"]):
            j=i+delta
            if 0<=j<n and j not in chosen:
                neighbors.append(j)
    for i in dict.fromkeys(neighbors):
        if len(chosen)>=maximum:
            break
        chosen.add(i)
    # Realistic unchanged observations must remain as negative controls.
    stride=cfg["regular_negative_stride"]
    routine=[i for i in range(0,n,stride)
             if not (flagsets[i]&set(RARE+POSITIVE))]
    for i in routine:
        if len(chosen)>=maximum:
            break
        chosen.add(i)
    # Remaining quota samples diverse chronological intervals from full trajectory.
    pool=[i for i in range(n) if i not in chosen]
    for i in spaced_pick(pool,maximum-len(chosen)):
        chosen.add(i)
    return sorted(chosen)


def profile_episode(spec,cfg,stress):
    business,geometry=build_quality_business(spec,stress)
    policy=TrueCPsatBranch(
        geometry,business.rules,spec["seed"],
        {"counterfactual_max_candidates":4})
    # All up to 260 true transitions are retained until terminal. PR54 cap
    # was 56: avoid conditioning dataset only on the early normal period.
    collector=PublicTransitionCollector(
        business.rules,window=4,max_steps=cfg["max_ticks"])
    world=BusinessSimulator(
        business,"world",spec["seed"],collect_metrics=True).run(
            dispatch_policy=policy,on_transition=collector)
    if resource_audit(world,business.rules)["violations"]:
        raise AssertionError("resource safety violated")
    if discrete_safety_violations(world) or lifecycle_audit(world["history"]):
        raise AssertionError("robot or business workflow safety violated")
    if not collector.rows or collector.rows[-1]["input"]["public_tick"]+1!=world["summary"]["ticks"]:
        raise AssertionError("not full actual system history through terminal tick")
    rows=collector.rows
    for row in rows:
        row["quality"]={
            "flags":flag_transition(row),
            "semantic_public_input_sha256":semantic_fingerprint(row)
        }
    selected=curated_indices(rows,cfg["curated_training_preview"])
    for i,row in enumerate(rows):
        row["quality"]["training_preview_selected"]=i in selected
    flags=Counter(f for row in rows for f in row["quality"]["flags"])
    curated=[rows[i] for i in selected]
    flags_curated=Counter(f for row in curated for f in row["quality"]["flags"])
    key_all=[r["quality"]["semantic_public_input_sha256"] for r in rows]
    key_curated=[r["quality"]["semantic_public_input_sha256"] for r in curated]
    static=sum(not set(r["quality"]["flags"])&set(RARE+POSITIVE) for r in curated)
    true_late=sum(i>=56 for i,row in enumerate(rows)
                  if set(row["quality"]["flags"])&set(RARE+POSITIVE))
    return {
        "case":spec,"real_summary":world["summary"],
        "full_public_transitions":len(rows),
        "original_56_slice":min(56,len(rows)),
        "event_flags_full":dict(flags),
        "event_flags_curated":dict(flags_curated),
        "late_eventful_transitions_after_original_56":true_late,
        "curated_examples":len(curated),
        "curated_static_fraction":static/len(curated),
        "raw_unique_semantic_input_fraction":len(set(key_all))/len(key_all),
        "curated_unique_semantic_input_fraction":len(set(key_curated))/len(key_curated),
        "selected_ticks":[rows[i]["input"]["public_tick"] for i in selected],
        "raw_trajectories":rows,
        "actual_source":"original BusinessSimulator real t->t+1 public events and lawful selected CP-SAT actions",
    }


def run(regime,out):
    cfg=json.loads(PROTOCOL.read_text())
    validate(cfg)
    if regime not in REGIMES:
        raise ValueError("unregistered pilot stress regime")
    if out.exists():
        raise FileExistsError("never overwrite original quality raw truth")
    stress=json.loads(STRESS_PROTOCOL.read_text())
    start=time.process_time()
    episodes=[]
    for spec in cfg["pilot_cases"]:
        if spec["regime"]!=regime:
            continue
        episode=profile_episode(spec,cfg,stress)
        episodes.append(episode)
        print("PR55_REAL_QUALITY_EPISODE",spec["id"],
              json.dumps({
                  "raw":episode["full_public_transitions"],
                  "curated":episode["curated_examples"],
                  "events":episode["event_flags_full"],
                  "post56_events":episode["late_eventful_transitions_after_original_56"]
              }),flush=True)
    if len(episodes)!=6:
        raise AssertionError("not all six pilot design profiles executed")
    summary={
        "regime":regime,"development_only":True,"episodes":6,
        "all_actual_transitions":sum(x["full_public_transitions"] for x in episodes),
        "preselected_training_preview":sum(x["curated_examples"] for x in episodes),
        "real_event_coverage":dict(sum(
            (Counter(x["event_flags_full"]) for x in episodes),Counter())),
        "late_eventful_transitions_post56":sum(
            x["late_eventful_transitions_after_original_56"] for x in episodes),
        "GPU_used":0,"neural_training":0,"cpu_process_seconds":time.process_time()-start
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({
        "schema":"pr55-quality-first-true-public-world-dev-pilot-v1",
        "protocol_sha256":hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "original_simulator_profile":"same 7x5 grid; controlled workload/robot/capacity/service/release variation",
        "regime":regime,"episodes":episodes,"summary":summary
    },ensure_ascii=False)+"\n")
    print("PR55_SOURCE_FULL_REAL_WORLD_QUALITY",json.dumps(summary,ensure_ascii=False),
          flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--regime",choices=REGIMES,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    run(args.regime,args.out)


if __name__=="__main__":
    main()
