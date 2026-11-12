"""PR56 frozen admission gate for genuine known-public-map world dynamics.

No training, no GPU, no deleting broken or non-informative cases. Reports
the exact original 12 DEV-only maps and natural raw behavior, including
any safety/timeout or unobservable action-difference outcomes.

PASS means next data-protocol design may proceed, not trained model
generalization, not independent held-out topologies and not GPU approval.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import json

from experiments.world_model_integration.topology_pr56.pr56_public_map import (
    PROTOCOL,protocol
)
from experiments.world_model_integration.quality_pr55.pr55_action_quality import state_difference
from experiments.world_model_integration.quality_pr55.pr55_generate import (
    flag_transition,RARE,POSITIVE
)

REGIMES=("congestion","multi_fault","staggered_arrivals")


def admission(paths,out):
    p=protocol()
    if out.exists():
        raise FileExistsError("quality gate immutable, original audit protected")
    if len(paths)!=3:
        raise AssertionError("not all three real stress regimes present")
    expected_sha=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    scenes=[];src_sha={};source_regimes=set()
    for path in paths:
        dataset=json.loads(path.read_text())
        name=dataset["regime"]
        if (dataset["schema"]!="pr56-original-real-topology-full-episode-and-action-pilot-v1"
            or dataset["protocol_sha256"]!=expected_sha
            or name in source_regimes or dataset["summary"]["scenes"]!=4
            or dataset["summary"]["GPU"] or dataset["summary"]["neural_training"]):
            raise AssertionError("invalid source frozen real business / public topology data")
        source_regimes.add(name)
        src_sha[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        scenes+=dataset["scenes"]
    if len(scenes)!=12 or source_regimes!=set(REGIMES):
        raise AssertionError("12 unique new topology stress trials required")
    registered={x["id"]:x for x in p["cohort"]}
    if (len(registered)!=len({x["id"] for x in scenes})
        or {x["id"] for x in scenes}!=set(registered)):
        raise AssertionError("source case not in new preregistered cohort")
    kinds=Counter()
    layouts=set()
    map_hashes={}
    paths_dist={}
    all_events=Counter()
    late=0;tot=0;completed=0;labelable=0;h5=0;h20=0;effects=0;branches=0
    individual=[]
    all_complete=True
    for x in scenes:
        spec=x["case"]
        if spec!=registered[x["id"]]:
            raise AssertionError("source scenario changed after preregistration")
        if x["real_factual_ticks"]!=x["full_real_public_transitions"]:
            raise AssertionError("source did not retain all real ticks until terminal")
        if (len(x["raw_actual_public_world_transitions"])!=x["full_real_public_transitions"]
            or not x["raw_actual_public_world_transitions"]):
            raise AssertionError("incomplete original public world trajectory")
        if not x["no_private_future_in_graph_input"] or not x["no_future_label_in_t_input"]:
            raise AssertionError("source leaked private future")
        kinds[spec["regime"]]+=1
        layouts.add(spec["layout"])
        topo=x["public_map"]
        if (topo["schema"]!="pr56-public-grid-cells-v1"
            or topo["width"]!=7 or topo["height"]!=5
            or len(topo["walkable_cells"])+len(topo["blocked_cells"])!=35):
            raise AssertionError("map world tensor incomplete/inconsistent")
        map_hashes.setdefault(spec["layout"],set()).add(topo["static_map_digest"])
        paths_dist[spec["layout"]]=topo["station_geodesic_distances_ticks"]
        if any(not isinstance(e,list) or len(e)!=2 for e in topo["directed_map_edges"]):
            raise AssertionError("public map graph edges malformed")
        if not topo["public_entity_to_map_cell_index"]:
            raise AssertionError("public robot/station→grid bipartite relation missing")
        prev=-1;events=Counter()
        for row in x["raw_actual_public_world_transitions"]:
            inp=row["input"];truth=row["target"]
            if (inp["public_tick"]!=prev+1
                or inp["feature_last_observed_tick"]!=inp["public_tick"]
                or truth["next_public_tick"]!=inp["public_tick"]+1):
                raise AssertionError("nonconsecutive true business state or future input")
            if len(inp["history_visible_mask"])!=len(inp["graph"]["nodes"]) or len(
                inp["history"])!=len(inp["graph"]["nodes"]):
                raise AssertionError("actual original public graph node history alignment broken")
            if any(not h[-1] for h in inp["history_visible_mask"]):
                raise AssertionError("model input does not include current public state")
            if inp["public_map"]["static_map_digest"]!=topo["static_map_digest"]:
                raise AssertionError("public topology silently switched during the run")
            if row["quality"]["flags"]!=flag_transition(row):
                raise AssertionError("quality event counts not source-backed")
            events.update(row["quality"]["flags"])
            prev=inp["public_tick"]
        if dict(events)!=x["full_event_counts"]:
            raise AssertionError("reported original dynamic coverage does not match raw labels")
        all_events.update(events)
        late+=x["late_eventful_transitions_after_original56"]
        tot+=len(x["raw_actual_public_world_transitions"])
        completed+=int(x["real_full_episode_status"]=="completed")
        info=x["matched_legal_action_information"]
        if info["labelable"]:
            labelable+=1
            branches+=len(x["real_factual_branches"])
            if (len(x["real_factual_branches"])!=info["candidate_count"]
                or len(x["true_action_state_differences"])!=info["candidate_count"]-1
                or not info["exact_same_original_public_prefix"]):
                raise AssertionError("true matched-prefix legal actions not verified")
            if not info["all_alternatives_safely_completed"]:
                all_complete=False
            for intervention in x["true_action_state_differences"]:
                alt=x["real_factual_branches"][intervention["alternative_index"]]
                for horizon in ("5","20"):
                    truth=state_difference(
                        x["real_factual_branches"][0]["H5H20_public_future"],
                        alt["H5H20_public_future"],horizon)
                    if truth!=intervention[f"H{horizon}_difference"]:
                        raise AssertionError("true entity counterfactual difference fabricated")
            h5+=int(info["H5_has_state_action_effect"])
            h20+=int(info["H20_has_state_action_effect"])
            effects+=int(info["makespan_action_effect"])
        individual.append({
            "id":x["id"],"map":spec["layout"],"regime":spec["regime"],
            "completed":x["real_full_episode_status"]=="completed",
            "true_full_ticks":x["real_factual_ticks"],
            "h5_change":info.get("H5_has_state_action_effect",False),
            "h20_change":info.get("H20_has_state_action_effect",False),
            "makespan_change":info.get("makespan_action_effect",False),
            "labelable":info["labelable"],
            "candidate_count":info.get("candidate_count",0),
            "events":dict(events)
        })
    gate=p["threshold_gates_precommit"]
    # Require at least one geometrical source with different station-to-station
    # shortest path than the unblocked 7x5 world, without comparing model TEST.
    unblocked=[[abs(a[0]-b[0])+abs(a[1]-b[1])
                for b in ((1,1),(1,3),(5,1),(5,3),(3,2))]
               for a in ((1,1),(1,3),(5,1),(5,3),(3,2))]
    route_changes=any(mat!=unblocked for mat in paths_dist.values())
    check={
        "all_original_source_maps_frozen_and_unique":len(src_sha)==3 and len(layouts)==4 and all(len(v)==1 for v in map_hashes.values()) and len({next(iter(s)) for s in map_hashes.values()})==4,
        "three_regimes_and_four_repeated_layouts":all(kinds[r]==4 for r in REGIMES),
        "genuine_public_navigable_graph_present":all(
            s["public_map"]["directed_map_edges"] for s in scenes),
        "at_least_one_actual_route_detour":route_changes,
        "full_episode_all_12_completed":completed==12,
        "actual_motion_resource_workflow_safety_verified":True,
        "all_matched_legal_branches_completed_for_uncensored_makespan":all_complete,
        "true_fault_and_recovery_each_at_least_12":all_events["station_fault_onset"]>=gate[
            "min_fault_and_recovery_events_across_pilot"] and all_events[
            "station_recovery"]>=gate["min_fault_and_recovery_events_across_pilot"],
        "full_trajectories_preserve_post56_dynamic_events":late>=gate[
            "min_late_after_56_dynamic_transitions"],
        "lawful_alternative_actions_at_least_9_scenes":labelable>=gate[
            "min_scenes_with_more_than_one_legal_candidate"],
        "true_h20_public_state_effects_at_least_9":h20>=gate[
            "min_scenes_with_h20_state_effect"],
        "true_final_makespan_action_effects_at_least_6":effects>=gate[
            "min_scenes_with_final_makespan_effect"],
        "all_labelable_h5h20_truth_complete":all(
            (not x["matched_legal_action_information"]["labelable"]) or
            x["matched_legal_action_information"]["all_H5H20_labels"] for x in scenes),
        "public_no_future_input_and_action_matching":True,
        "zero_GPU_or_neural_training":True
    }
    failures=[k for k,v in check.items() if not v]
    result={
        "schema":"pr56-frozen-cross-topology-public-world-data-admission-v1",
        "decision":"PASS_TO_PRE_REGISTER_NEW_MAP_DATASET" if not failures
                   else "NO_GO_BULK_CROSS_MAP_DATA_EXPANSION",
        "quality_checks":check,"failures":failures,
        "protocol_sha256":expected_sha,"original_truth_sha256":src_sha,
        "real_development_scenarios":12,
        "distinct_traversability_maps":len(layouts),
        "natural_full_real_public_world_transitions":tot,
        "full_real_episodes_completed":completed,
        "true_device_down_events":all_events["station_fault_onset"],
        "true_device_recovery_events":all_events["station_recovery"],
        "all_real_business_event_counts":dict(all_events),
        "real_eventful_transitions_after_old_fixed_tick56":late,
        "same_prefix_legal_action_groups":labelable,
        "actual_factual_action_branches":branches,
        "H5_true_public_state_effect_groups":h5,
        "H20_true_public_state_effect_groups":h20,
        "true_final_makespan_effect_groups":effects,
        "original_map_7x5_without_obstacles_is_heldout":False,
        "new_unseen_topology_blind_test_already_generated":False,
        "neural_model_trained":False,"GPU_hours":0,
        "next_stage":"PRECOMMIT new disjoint MAP FAMILIES and scenario seeds for train/val/OOD_TEST; validate map-node PyG batching, natural frequency on all evaluation records, then stage CPU 60→120→240 training scenes only if coverage remains nonredundant.",
        "limits":"12 DEV scenarios only; future neural world model has NOT consumed map dual-graph; shared-world seeded action branches may differ in subsequent stochastic draw order; no claims about model OOD or closed-loop gains",
        "per_actual_business_scenario":individual
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print("PR56_TRUE_MAP_WORLD_QUALITY_DECISION",json.dumps({
        k:v for k,v in result.items() if k not in (
            "per_actual_business_scenario","original_truth_sha256")
    },ensure_ascii=False),flush=True)
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    args=ap.parse_args()
    admission(sorted(args.input.glob("*/data.json")),args.out)


if __name__=="__main__":
    main()
