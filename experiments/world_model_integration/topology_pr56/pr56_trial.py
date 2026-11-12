"""PR56 ORIGINAL BusinessSimulator on 4 truly different known public Grid maps.

Each regime runs 4 PRE-REGISTERED development-only topology scene trials:
complete true public t->t+1 trace, original legal CP-SAT at every tick,
and exact-public-prefix FIRST dispatch action interventions (up to 4).
Public topology lives in the world-model INPUT (no private schedule);
H5/H20 realized future is a LABEL only. No neural training / no GPU.
"""
from __future__ import annotations
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import time

from business_transport.simulation import BusinessSimulator,resource_audit
from experiments.business_constraints.run import lifecycle_audit
from experiments.world_model_integration.temporal_pr54.pr54_public_sequence import PublicTransitionCollector
from experiments.world_model_integration.quality_pr55.pr55_generate import (
    flag_transition,semantic_fingerprint,RARE,POSITIVE
)
from experiments.world_model_integration.quality_pr55.pr55_action_quality import state_difference
from experiments.world_model_integration.entity_pr52.pr52_labels import EntityTruthBranch
from experiments.world_model_integration.decision_learning_pr51.pr51_labels import _fingerprint
from experiments.world_model_integration.online_horizon.pr30_run import _facts
from experiments.world_model_integration.stress_pr48.pr48_scenarios import PROTOCOL as STRESS_PROTOCOL
from experiments.world_model_integration.topology_pr56.pr56_public_map import (
    PROTOCOL,protocol,build_topology_business,augment_public_transition,
    public_topology_for_graph
)
from lab_transport.multisite_experiment import discrete_safety_violations

REGIMES=("congestion","multi_fault","staggered_arrivals")


def run_actual(spec,stress,cfg,*,expected=None,index=0,collect_all=False):
    business,geo=build_topology_business(spec,stress,cfg)
    policy=EntityTruthBranch(
        geo,business.rules,spec["seed"],
        {"counterfactual_max_candidates":cfg["actual_legal_alternatives"]["candidate_limit"]},
        expected=expected,branch_index=index)
    collector=(PublicTransitionCollector(
        business.rules,cfg["public_history_window"],cfg["max_ticks"])
               if collect_all else None)
    actual=BusinessSimulator(
        business,"world",spec["seed"],collect_metrics=True).run(
        dispatch_policy=policy,
        on_observation=policy.observe,
        on_transition=collector)
    if resource_audit(actual,business.rules)["violations"]:
        raise AssertionError("original business violated real resource capacity")
    if discrete_safety_violations(actual) or lifecycle_audit(actual["history"]):
        raise AssertionError("original business motion or lifecycle violated")
    if actual["summary"]["ticks"]>cfg["max_ticks"]:
        raise AssertionError("unregistered actual simulator time")
    stats=_facts(actual,business)
    info={
        "branch_index":index,
        "true_completed":actual["summary"]["status"]=="completed",
        "true_ticks":actual["summary"]["ticks"],
        "true_late_jobs":stats["late_jobs"],
        "H5H20_public_future":policy.future_nodes,
        "factual_event_sha256":_fingerprint(actual["events"]),
    }
    if not collect_all:
        return policy,info,None
    if collector is None or not collector.rows or len(collector.rows)!=actual["summary"]["ticks"]:
        raise AssertionError("full original public timeline not collected through terminal")
    for row in collector.rows:
        augment_public_transition(row,geo)
        row["quality"]={
            "flags":flag_transition(row),
            "semantic_input_state_sha256":semantic_fingerprint(row),
            "static_topology_digest":row["input"]["public_map"]["static_map_digest"]
        }
    return policy,info,collector.rows


def run_scene(spec,stress,cfg):
    base,b0,rows=run_actual(spec,stress,cfg,collect_all=True)
    business,geo=build_topology_business(spec,stress,cfg)
    expected_map=public_topology_for_graph(geo.grid,geo,rows[0]["input"]["graph"])
    static_digest=expected_map["static_map_digest"]
    if any(row["input"]["public_map"]["static_map_digest"]!=static_digest
           for row in rows):
        raise AssertionError("public static map changed in one original business episode")
    observed=dict(Counter(f for r in rows for f in r["quality"]["flags"]))
    if base.anchor is None:
        info={"labelable":False,"reason":"no multiple exact legal CP-SAT candidate actions after fault"}
        interventions=[]
        outcomes=[b0]
        anchor=None
    else:
        anchor={
            **base.anchor,
            "observation_sha256":base.anchor_fingerprint,
            "plans_sha256":base.anchor_plan_fingerprint
        }
        original_prefix=[x for x in base.trace if x["tick"]<=anchor["tick"]]
        outcomes=[b0]
        interventions=[]
        for i in range(1,anchor["candidate_count"]):
            counter,alt,_=run_actual(spec,stress,cfg,expected=anchor,index=i)
            branch_prefix=[x for x in counter.trace if x["tick"]<=anchor["tick"]]
            if len(branch_prefix)!=len(original_prefix):
                raise AssertionError("alternative legal branch has different original prefix length")
            for a,b in zip(original_prefix,branch_prefix):
                if (a["tick"]!=b["tick"] or
                    a["observation_sha256"]!=b["observation_sha256"] or
                    a["plans_sha256"]!=b["plans_sha256"] or
                    (a["tick"]!=anchor["tick"] and a["selected_plan"]!=b["selected_plan"])):
                    raise AssertionError("alternative public input / legal candidate history not identical")
            if counter.node_anchor["node_keys"]!=base.node_anchor["node_keys"]:
                raise AssertionError("original public future node labels unaligned")
            outcomes.append(alt)
            interventions.append({
                "alternative_index":i,
                "true_makespan_delta":alt["true_ticks"]-b0["true_ticks"],
                "H5_difference":state_difference(
                    b0["H5H20_public_future"],alt["H5H20_public_future"],"5"),
                "H20_difference":state_difference(
                    b0["H5H20_public_future"],alt["H5H20_public_future"],"20")
            })
        info={
            "labelable":True,
            "exact_same_original_public_prefix":True,
            "all_alternatives_safely_completed":all(x["true_completed"] for x in outcomes),
            "all_H5H20_labels":all(
                set(x["H5H20_public_future"])=={"5","20"} for x in outcomes),
            "H5_has_state_action_effect":any(
                v["H5_difference"] is not None and v["H5_difference"]["nonzero_true_public_state_effect"]
                for v in interventions),
            "H20_has_state_action_effect":any(
                v["H20_difference"] is not None and v["H20_difference"]["nonzero_true_public_state_effect"]
                for v in interventions),
            "makespan_action_effect":len({x["true_ticks"] for x in outcomes})>1,
            "candidate_count":len(outcomes),
            "anchor":anchor
        }
    return {
        "id":spec["id"],"case":spec,
        "public_map":expected_map,
        "real_full_episode_status":"completed" if b0["true_completed"] else "timed_out",
        "real_factual_ticks":b0["true_ticks"],
        "full_real_public_transitions":len(rows),
        "full_event_counts":observed,
        "late_eventful_transitions_after_original56":sum(
            i>=56 and bool(set(r["quality"]["flags"])&set(RARE+POSITIVE))
            for i,r in enumerate(rows)),
        "raw_actual_public_world_transitions":rows,
        "matched_legal_action_information":info,
        "real_factual_branches":outcomes,
        "true_action_state_differences":interventions,
        "no_private_future_in_graph_input":True,
        "no_future_label_in_t_input":True
    }


def run(regime,out):
    cfg=protocol()
    if regime not in REGIMES:
        raise ValueError("unknown preregistered pressure")
    if out.exists():
        raise FileExistsError("immutable first generation real topology source exists")
    stress=json.loads(STRESS_PROTOCOL.read_text())
    began=time.process_time()
    samples=[]
    for spec in cfg["cohort"]:
        if spec["regime"]!=regime:
            continue
        x=run_scene(spec,stress,cfg)
        samples.append(x)
        print("PR56_REAL_MAP_WORLD_SCENE",x["id"],
              json.dumps({
                  "map":x["case"]["layout"],
                  "source_ticks":x["full_real_public_transitions"],
                  "completed":x["real_full_episode_status"],
                  "different_maps_static_observed":bool(x["public_map"]["directed_map_edges"]),
                  "action":x["matched_legal_action_information"],
                  "late_events":x["late_eventful_transitions_after_original56"]
              },ensure_ascii=False),flush=True)
    if len(samples)!=4:
        raise AssertionError("one scene for each registered topology required")
    summary={
        "regime":regime,
        "development_only":True,"scenes":len(samples),
        "full_true_timestep_labels":sum(x["full_real_public_transitions"] for x in samples),
        "complete_factual_worlds":sum(x["real_full_episode_status"]=="completed" for x in samples),
        "actual_factual_branches":sum(len(x["real_factual_branches"]) for x in samples),
        "multiple_lawful_action_choice":sum(x["matched_legal_action_information"]["labelable"] for x in samples),
        "H20_action_state_effect":sum(x["matched_legal_action_information"].get("H20_has_state_action_effect",False) for x in samples),
        "makespan_action_effect":sum(x["matched_legal_action_information"].get("makespan_action_effect",False) for x in samples),
        "real_late_dynamic_events_after56":sum(x["late_eventful_transitions_after_original56"] for x in samples),
        "observed_fault_onset":sum(x["full_event_counts"].get("station_fault_onset",0) for x in samples),
        "observed_station_recovery":sum(x["full_event_counts"].get("station_recovery",0) for x in samples),
        "GPU":0,"neural_training":0,
        "cpu_process_seconds":time.process_time()-began
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({
        "schema":"pr56-original-real-topology-full-episode-and-action-pilot-v1",
        "protocol_sha256":hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "regime":regime,"scenes":samples,"summary":summary,
        "limit":"All scenarios development only; map tensor bridge exists but no fitted GPU or H20 recursive model; fixed environment seeds do not guarantee identical post-action draw order."
    },ensure_ascii=False)+"\n")
    print("PR56_CPU_TOPOLOGY_PILOT_RESULT",json.dumps(summary,ensure_ascii=False),flush=True)


def main():
    a=argparse.ArgumentParser()
    a.add_argument("--regime",choices=REGIMES,required=True)
    a.add_argument("--out",type=Path,required=True)
    args=a.parse_args()
    run(args.regime,args.out)


if __name__=="__main__":
    main()
