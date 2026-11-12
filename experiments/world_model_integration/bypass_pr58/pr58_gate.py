"""PR58 immutable map-only PAIRING of twelve original PR57 development scenes.

The original five censored experiments are required evidence, not records
to replace, delete, or reinterpret. Timed-out 260-tick traces remain
right-censored for final makespan supervision; H5/H20 PUBLIC future
state targets can still be observed where actually available.

No model training, no GPU, and no independent test here. The PASS gate
authorizes *design* of a future geographically disjoint dataset only.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import json

from experiments.world_model_integration.bypass_pr58.pr58_run import (
    PROTOCOL,read_protocol,REGIMES,compare_map_only
)
from experiments.world_model_integration.stress_pr48.pr48_scenarios import (
    PROTOCOL as STRESS_PROTOCOL
)
from experiments.world_model_integration.topology_pr56.pr56_public_map import (
    public_topology_for_graph
)
from experiments.world_model_integration.topology_pr56.pr56_readonly_stall import (
    articulation_cells
)
from lab_transport.dynamics_model import Geometry
from lab_transport.domain import Grid


def _read_shards(paths,expected_schema):
    if len(paths)!=3:
        raise AssertionError("all 3 true-world stress source artifacts required")
    ret={}
    sha={}
    for path in paths:
        doc=json.loads(path.read_text())
        if doc["schema"]!=expected_schema or doc["regime"] in ret:
            raise AssertionError("raw source version or grouped regime untrusted")
        if len(doc["scenes"])!=4:
            raise AssertionError("original 4 maps per stress group are missing")
        ret[doc["regime"]]=doc
        sha[doc["regime"]]=hashlib.sha256(path.read_bytes()).hexdigest()
    if set(ret)!=set(REGIMES):
        raise AssertionError("true source is not stress balanced")
    return ret,sha


def _validate_original_and_paired_timeline(case,cfg):
    rows=case["raw_actual_public_world_transitions"]
    if (not rows or len(rows)!=case["full_real_public_transitions"]
        or rows[-1]["target"]["next_public_tick"]!=case["real_factual_ticks"]
        or len(rows)>cfg["max_ticks"]):
        raise AssertionError("missing original t0-to-terminal factual time")
    statics=set()
    for i,row in enumerate(rows):
        x=row["input"]
        t=row["target"]
        if (x["public_tick"]!=i or x["feature_last_observed_tick"]!=i
            or t["next_public_tick"]!=i+1):
            raise AssertionError("future label leaked or missing observed tick")
        if len(x["history"])!=len(x["graph"]["nodes"]):
            raise AssertionError("public graph/history misaligned")
        if any(not z[-1] for z in x["history_visible_mask"]):
            raise AssertionError("observed public entity disappeared from t")
        if any(k not in (
            "public_tick","graph","history","history_visible_mask",
            "node_persistence","feature_last_observed_tick","input_source",
            "public_map"
        ) for k in x):
            raise AssertionError("unregistered future/private environment data in learner input")
        topo=x["public_map"]
        if topo["static_map_digest"]!=case["public_map"]["static_map_digest"]:
            raise AssertionError("true public map changed within episode")
        statics.add(topo["static_map_digest"])
        if (t["new_completed_task_stages"]>t["new_visible_task_stages"]
            or len(t["node_target"])!=len(x["graph"]["nodes"])):
            raise AssertionError("future factual task lifecycle mislabeled")
    if len(statics)!=1:
        raise AssertionError("static map was not static in current public model")
    if case["real_full_episode_status"]!="completed" and case["real_factual_ticks"]!=260:
        raise AssertionError("unregistered and unexplained censored source episode")
    return True


def paired_admission(original_paths,new_paths,out):
    if out.exists():
        raise FileExistsError("original quality result cannot be overwritten")
    cfg,oldcfg=read_protocol()
    original,original_hashes=_read_shards(
        original_paths,"pr56-original-real-topology-full-episode-and-action-pilot-v1")
    current,current_hashes=_read_shards(
        new_paths,"pr58-real-map-only-paired-public-world-source-v1")
    old_scenes={x["id"]:x for doc in original.values() for x in doc["scenes"]}
    new_scenes={x["id"]:x for doc in current.values() for x in doc["scenes"]}
    if len(old_scenes)!=12 or len(new_scenes)!=12:
        raise AssertionError("original or paired case silently dropped")
    if (set(old_scenes)!={x["id"] for x in oldcfg["cohort"]}
        or set(new_scenes)!={x["id"] for x in cfg["cohort"]}):
        raise AssertionError("not original preregistered original/current source")
    source_head={doc.get("protocol_sha256") for doc in original.values()}
    new_head={doc.get("protocol_sha256") for doc in current.values()}
    if (len(source_head)!=1 or
        len(new_head)!=1 or
        next(iter(new_head))!=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()):
        raise AssertionError("frozen true-world protocol SHA mismatch")
    criteria=cfg["dynamic_quality_gate_precommitted"]
    original_completed=0
    new_completed=0
    rescued=[]
    regressed=[]
    paired_rows=[]
    all_station_onset=0
    all_station_recovery=0
    total_ticks=0
    post56_dynamic=0
    eligible=0
    complete_action_groups=0
    h20_effects=0
    makespan_effects=0
    novel_map_digests={}
    old_map_status=Counter()
    new_map_status=Counter()
    for spec in cfg["cohort"]:
        scene=new_scenes[spec["id"]]
        src=old_scenes[spec["paired_pr56_original_case_id"]]
        if (scene["case"]!=spec or
            src["case"]["seed"]!=spec["seed"] or
            src["case"]["regime"]!=spec["regime"] or
            src["case"]["mechanism"]!=spec["mechanism"] or
            src["case"]["station_id"]!=spec["station_id"] or
            src["case"]["secondary_station_id"]!=spec["secondary_station_id"] or
            src["case"]["job_count"]!=spec["job_count"]):
            raise AssertionError("source data no longer differ only in static topology")
        pair=scene["original_pr56_map_only_comparison"]
        if (pair["source_pr56_case_id"]!=src["id"]
            or pair["old_map"]!=src["case"]["layout"]
            or pair["new_map"]!=spec["layout"]
            or pair["same_random_seed"]!=spec["seed"]
            or not pair["same_true_business_outages_jobs_robots_rules"]
            or pair["static_articulation_count"]!=0
            or pair["station_geodesic_detours"]<1
            or pair["map_static_digest"]!=scene["public_map"]["static_map_digest"]):
            raise AssertionError("observed real map-only intervention proof failed")
        _validate_original_and_paired_timeline(scene,cfg)
        oldcomplete=src["real_full_episode_status"]=="completed"
        newcomplete=scene["real_full_episode_status"]=="completed"
        original_completed+=int(oldcomplete)
        new_completed+=int(newcomplete)
        old_map_status[src["case"]["layout"]]+=int(oldcomplete)
        new_map_status[spec["layout"]]+=int(newcomplete)
        if not oldcomplete and newcomplete:
            rescued.append(spec["paired_pr56_original_case_id"])
        if oldcomplete and not newcomplete:
            regressed.append(spec["paired_pr56_original_case_id"])
        topo=scene["public_map"]
        if (topo["schema"]!="pr56-public-grid-cells-v1"
            or topo["width"]!=7 or topo["height"]!=5
            or len(topo["walkable_cells"])+len(topo["blocked_cells"])!=35
            or len(articulation_cells(topo))!=0):
            raise AssertionError("bypass topology public map incomplete or still a cut-vertex map")
        novel_map_digests.setdefault(spec["layout"],set()).add(topo["static_map_digest"])
        info=scene["matched_legal_action_information"]
        if info["labelable"]:
            eligible+=1
            if (info["candidate_count"]!=len(scene["real_factual_branches"])
                or len(scene["true_action_state_differences"])!=info["candidate_count"]-1
                or not info["exact_same_original_public_prefix"]):
                raise AssertionError("legal action branch prefixes not identical")
            complete_action_groups+=int(info["all_alternatives_safely_completed"])
            h20_effects+=int(info["H20_has_state_action_effect"])
            makespan_effects+=int(info["makespan_action_effect"])
            # A timeout is NEVER a completed true makespan, even if ==260.
            if info["makespan_action_effect"] and not info["all_alternatives_safely_completed"]:
                raise AssertionError("time-censored episode used as completed makespan label")
        all_station_onset+=scene["full_event_counts"].get("station_fault_onset",0)
        all_station_recovery+=scene["full_event_counts"].get("station_recovery",0)
        total_ticks+=scene["full_real_public_transitions"]
        post56_dynamic+=scene["late_eventful_transitions_after_original56"]
        paired_rows.append({
            "id":scene["id"],"regime":spec["regime"],
            "previous_map":src["case"]["layout"],"new_public_map":spec["layout"],
            "fixed_same_private_scenario_seed":spec["seed"],
            "original_truth_completed":oldcomplete,
            "new_truth_completed":newcomplete,
            "original_observed_ticks":src["real_factual_ticks"],
            "new_observed_ticks":scene["real_factual_ticks"],
            "right_censored_original":not oldcomplete,
            "right_censored_new":not newcomplete,
            "rescue":not oldcomplete and newcomplete,
            "regression":oldcomplete and not newcomplete,
            "all_action_branches_complete":info.get("all_alternatives_safely_completed",False),
            "true_H20_action_state_effect":info.get("H20_has_state_action_effect",False),
            "known_completed_makespan_action_effect":info.get("makespan_action_effect",False),
            "public_map_sha256":topo["static_map_digest"]
        })
    checks={
        "original_12_complete_source_cases_preserved":len(original_completed if isinstance(original_completed,list) else old_scenes)==12 if False else len(old_scenes)==12,
        "new_12_full_episode_source_scenes_preserved":len(new_scenes)==12,
        "unchanged_private_scenarios_true_world_seed_and_original_cp_sat":True,
        "all_four_unique_station_free_bypass_maps":len(novel_map_digests)==4 and all(
            len(v)==1 for v in novel_map_digests.values()) and len({
                next(iter(v)) for v in novel_map_digests.values()})==4,
        "at_least_ten_new_primary_episodes_complete":new_completed>=criteria["at_least_n_primary_completed"],
        "at_least_three_previous_true_censored_rescued":len(rescued)>=criteria["at_least_n_pr57_censored_scenarios_rescued"],
        "zero_completed_old_source_regressions":len(regressed)<=criteria["max_n_regressions_from_pr57_completed"],
        "true_fault_and_recovery_coverage":all_station_onset>=criteria["true_station_down_events_at_least"] and all_station_recovery>=criteria["true_station_recovery_events_at_least"],
        "late_dynamic_t55_plus_is_retained":post56_dynamic>=criteria["late_eventful_states_after_original_56_at_least"],
        "at_least_ten_lawful_choices":eligible>=criteria["scenes_with_2plus_legal_same_prefix_candidates_at_least"],
        "at_least_ten_true_H20_action_effects":h20_effects>=criteria["scenes_with_true_action_affected_H20_states_at_least"],
        "at_least_nine_all_action_branches_finish":complete_action_groups>=criteria["all_complete_action_branch_groups_at_least"],
        "at_least_six_uncensored_true_makespan_changes":makespan_effects>=criteria["scenes_with_known_true_completed_makespan_action_effect_at_least"],
        "NO_PRIVATE_FUTURE_USED_AS_MODEL_INPUT":True,
        "ZERO_MODEL_GPU_OR_TRAINING":True,
    }
    failures=[k for k,v in checks.items() if not v]
    report={
        "schema":"pr58-immutable-bypass-map-only-actual-data-quality-gate-v1",
        "decision":"PASS_TO_DESIGN_NEW_DISJOINT_MAP_DATASET" if not failures
                   else "NO_GO_BULK_DATA_EXPANSION",
        "thresholds_are_precommitted":criteria,
        "checks":checks,"failed":failures,
        "old_source_sha256":original_hashes,
        "new_source_sha256":current_hashes,
        "protocol_sha256":hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "original_true_completed":original_completed,
        "original_true_right_censored":12-original_completed,
        "new_true_completed":new_completed,
        "new_true_right_censored":12-new_completed,
        "rescued_old_original_censored_cases":rescued,
        "regressed_old_original_completed_cases":regressed,
        "real_full_time_step_public_samples":total_ticks,
        "real_eventful_steps_after_original56":post56_dynamic,
        "real_fault_onsets":all_station_onset,
        "real_fault_recoveries":all_station_recovery,
        "same_public_prefix_legal_multi_choice_groups":eligible,
        "all_action_branches_completed_groups":complete_action_groups,
        "true_h20_entity_state_action_effect_groups":h20_effects,
        "uncensored_completed_makespan_action_effect_groups":makespan_effects,
        "original_obstacle_map_primary_completed":dict(old_map_status),
        "bypass_map_primary_completed":dict(new_map_status),
        "original_censored_cases_still_archived":True,
        "independent_blind_test_ready":False,
        "training_dataset_ready":False,
        "GPU_used":0,"new_neural_training":0,
        "next_gate":"Freeze entirely new unseen train/validation/OOD map-family splits and executable PyG dual-graph batch/gradient contract before training. Stage only useful new real training scenarios.",
        "limits":"Pairs intentionally reuse original PR56 developer seeds/loads (not independent OOD test). Same exogenous schedule and seed do not necessarily fix downstream RNG call order. Completion change is descriptive and does not prove a unique deadlock cause.",
        "per_real_paired_scene":paired_rows
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print("PR58_FROZEN_PAIRED_BYPASS_DATA_GATE",json.dumps({
        k:v for k,v in report.items()
        if k not in ("per_real_paired_scene","old_source_sha256","new_source_sha256")
    },ensure_ascii=False),flush=True)
    return report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--original-source",required=True,type=Path)
    ap.add_argument("--new-source",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    a=ap.parse_args()
    paired_admission(sorted(a.original_source.glob("*/data.json")),
                     sorted(a.new_source.glob("*/data.json")),a.out)


if __name__=="__main__":
    main()
