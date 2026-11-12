"""PR60 immutable marginal actual-data quality gate; no GPU, no neural fitting.

Verify all prior 24 TRAIN/12 natural VALIDATION source blobs and their
independent PASS verdict before assessing the NEW 36 extra TRAIN worlds.
A new stage must add previously absent dynamic state/action information:
counts, H20 action-difference MAGNITUDE and semantic signature novelty.

The true OOD TEST map-family remains PHYSICALLY UNGENERATED. Failing
episodes and time-censored makespan outcomes are retained, never deleted.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import json

from experiments.world_model_integration.expand_pr60.pr60_source import (
    PROTOCOL,ORIGINAL_PROTOCOL,protocol,REGIMES
)
from experiments.world_model_integration.quality_pr55.pr55_generate import flag_transition
from experiments.world_model_integration.sealed_pr59.pr59_source import validate_layout_business
from experiments.world_model_integration.stress_pr48.pr48_scenarios import PROTOCOL as STRESS_PROTOCOL


def evidence(paths,original_source,old_qa_file,out):
    if out.exists():
        raise FileExistsError("cannot overwrite original source admission decision")
    cfg,old=protocol()
    if len(paths)!=3:
        raise AssertionError("three original CPU stress source shards required")
    original_quality=json.loads(old_qa_file.read_text())
    if (original_quality["schema"]!="pr59-natural-map-family-physically-unopened-test-data-admission-v1"
        or original_quality["scientific_verdict"]!="PASS_QUALITY_TO_HANDOFF_GRAPH_TRAINING"
        or original_quality["protocol_sha256"]!=cfg["original_pr59_protocol_sha256"]
        or original_quality["BLIND_TEST_source_generated"] or original_quality["BLIND_TEST_source_read"]):
        raise AssertionError("frozen previous train+validation QA was not accepted")
    prev_paths=sorted(original_source.glob("*/data.json"))
    if len(prev_paths)!=2:
        raise AssertionError("original TRAIN and natural validation must both be available and no TEST")
    earlier={}
    for path in prev_paths:
        obj=json.loads(path.read_text())
        split=obj["split"]
        if (split not in ("train","validation") or split in earlier
            or obj["schema"]!="pr59-sha-sealed-natural-public-world-map-family-split-v1"
            or obj["source_protocol_sha256"]!=cfg["original_pr59_protocol_sha256"]
            or hashlib.sha256(path.read_bytes()).hexdigest()!=
                original_quality["source_split_sha256"][split]
            or obj["BLIND_TEST_RUN"]):
            raise AssertionError("original source raw data/sha mismatch")
        earlier[split]=obj
    if set(earlier)!={"train","validation"} or len(earlier["train"]["scenes"])!=24 or len(earlier["validation"]["scenes"])!=12:
        raise AssertionError("original natural 24+12 were not preserved")
    new_cases=[]
    source_hash={}
    for path in paths:
        src=json.loads(path.read_text())
        regime=src.get("regime")
        if (regime in source_hash or regime not in REGIMES or
            src["schema"]!="pr60-additional-real-public-map-train-source-v1"
            or src["split"]!="train" or src["BLIND_TEST_RUN"]
            or src["source_protocol_sha256"]!=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
            or src["parent_pr59_protocol_sha256"]!=cfg["original_pr59_protocol_sha256"]
            or len(src["scenes"])!=12):
            raise AssertionError("new unfiltered source not from frozen training-only protocol")
        source_hash[regime]=hashlib.sha256(path.read_bytes()).hexdigest()
        new_cases+=src["scenes"]
    if set(source_hash)!=set(REGIMES) or len(new_cases)!=36:
        raise AssertionError("missing original extra training case source")
    registered={s["id"]:s for s in cfg["cohort"]}
    if (len(registered)!=36 or len({x["id"] for x in new_cases})!=36
        or {x["id"] for x in new_cases}!=set(registered)):
        raise AssertionError("new training designs duplicated/omitted")
    old_train=earlier["train"]["scenes"]
    old_val=earlier["validation"]["scenes"]
    old_seeds={s["case"]["seed"] for s in old_train+old_val}
    if old_seeds&{s["case"]["seed"] for s in new_cases}:
        raise AssertionError("old training/validation random seeds reused")
    old_signature={
        (r["input"]["public_map"]["static_map_digest"],
         r["quality"]["semantic_input_state_sha256"])
        for e in old_train for r in e["raw_actual_public_world_transitions"]
    }
    if not old_signature:
        raise AssertionError("no old reference signature for marginal novelty")
    new_signature=set()
    old_overlap=0
    total_new_steps=0
    faults=Counter()
    map_counts=Counter()
    profile_counts=Counter()
    stress_counts=Counter()
    completed=0
    complete_branch=0
    H20=0
    substantial_H20=0
    post56=0
    interesting_action_effect_magnitudes=[]
    per_case=[]
    stress=json.loads(STRESS_PROTOCOL.read_text())
    for e in new_cases:
        c=e["case"]
        if c!=registered[e["id"]] or c["split"]!="train":
            raise AssertionError("new scene not the precommitted TRAIN-only developer data")
        topology,_=validate_layout_business(c,old,stress)
        if e["public_map"]["static_map_digest"]!=topology["static_map_digest"]:
            raise AssertionError("public route knowledge does not match actual business Grid")
        map_counts[c["layout"]]+=1
        profile_counts[c["profile"]]+=1
        stress_counts[c["regime"]]+=1
        rows=e["raw_actual_public_world_transitions"]
        if (len(rows)!=e["full_real_public_transitions"]
            or len(rows)!=e["real_factual_ticks"]
            or not rows or rows[-1]["target"]["next_public_tick"]!=e["real_factual_ticks"]):
            raise AssertionError("new source true factual time series truncated without flag")
        for tick,row in enumerate(rows):
            inp=row["input"];label=row["target"]
            if (inp["public_tick"]!=tick
                or inp["feature_last_observed_tick"]!=tick
                or label["next_public_tick"]!=tick+1
                or not all(v[-1] for v in inp["history_visible_mask"])):
                raise AssertionError("new source future leakage or damaged public history")
            if any(z not in (
                "public_tick","graph","history","history_visible_mask",
                "node_persistence","feature_last_observed_tick","input_source",
                "public_map"
            ) for z in inp):
                raise AssertionError("private environment state became learner input")
            if inp["public_map"]["static_map_digest"]!=topology["static_map_digest"]:
                raise AssertionError("true static public source topology changed mid-episode")
            if row["quality"]["flags"]!=flag_transition(row):
                raise AssertionError("true fault/action dynamics not grounded in target")
            key=(topology["static_map_digest"],row["quality"]["semantic_input_state_sha256"])
            old_overlap+=int(key in old_signature)
            new_signature.add(key)
        total_new_steps+=len(rows)
        faults.update(e["full_event_counts"])
        post56+=e["late_eventful_transitions_after_original56"]
        is_complete=e["real_full_episode_status"]=="completed"
        completed+=int(is_complete)
        info=e["matched_legal_action_information"]
        H20+=int(info.get("H20_has_state_action_effect",False))
        complete_branch+=int(info.get("all_alternatives_safely_completed",False))
        magnitudes=[
            float(v["H20_difference"]["mean_public_state_l1"])
            for v in e["true_action_state_differences"]
            if v.get("H20_difference") is not None
        ]
        meaningful=max(magnitudes) if magnitudes else 0.
        substantial_H20+=int(meaningful>=0.01)
        interesting_action_effect_magnitudes.append(meaningful)
        if info.get("makespan_action_effect",False) and not info.get(
            "all_alternatives_safely_completed",False):
            raise AssertionError("censored finished time falsely labelled")
        per_case.append({
            "id":e["id"],"map":c["layout"],"stress":c["regime"],
            "profile":c["profile"],"real_full_ticks":len(rows),
            "right_censored":not is_complete,
            "true_H20_nonzero_action_effect":info.get("H20_has_state_action_effect",False),
            "maximum_H20_common_future_state_l1":meaningful,
            "all_legal_actions_true_makespan_complete":info.get("all_alternatives_safely_completed",False)
        })
    if total_new_steps<=0:
        raise AssertionError("empty true-world dynamic source")
    novelty_fraction=1-old_overlap/total_new_steps
    within_new_uniqueness=len(new_signature)/total_new_steps
    criteria=cfg["high_quality_additional_train_precommitted_gate"]
    checks={
        "original_PR59_24train_12val_qa_and_SHA_verified":True,
        "all_original_36_pre_registered_train_worlds_retained":len(new_cases)==36,
        "original_BLIND_test_never_generated":cfg["new_blind_test_count"]==0,
        "six_real_business_profiles_stratified":all(profile_counts[p]==6 for p in old["profiles"]),
        "six_public_train_maps_stratified":all(map_counts[m]==6 for m in old["splits"]["train"]["maps"]),
        "three_stress_regimes_each_twelve":all(stress_counts[r]==12 for r in REGIMES),
        "genuine_fault_onset_and_recovery":faults["station_fault_onset"]>=criteria["minimum_36_true_FAULT_onsets"] and faults["station_recovery"]>=criteria["minimum_36_true_recovery_events"],
        "true_legal_action_H20_dynamics":H20>=criteria["minimum_36_valid_action_H20_state_effect_groups"],
        "all_legal_branches_completed_enough":complete_branch>=criteria["minimum_36_all_legal_branches_completed"],
        "not_just_minuscule_H20_state_differences":substantial_H20>=criteria["minimum_36_true_H20_mean_state_L1_at_least_0_01_scenes"],
        "late_business_events_represented":post56>=criteria["minimum_36_late_post56_dynamic_samples"],
        "limited_honest_time_censoring":(36-completed)/36<=criteria["maximum_36_censored_fraction"],
        "marginal_unique_new_public_state_information":novelty_fraction>=criteria["minimum_marginal_semantic_novelty_fraction_vs_original_train"],
        "new_raw_source_not_mostly_duplicate_ticks":within_new_uniqueness>=criteria["minimum_raw_semantic_unique_ratio_within_new"],
        "new_60train_and_12val_map_family_separation":not {x["case"]["map_family"] for x in old_val}&{x["case"]["map_family"] for x in new_cases},
        "no_PRIVATE_future_or_TEST_label":True,
        "source_0GPU_0new_model_training":True,
    }
    rejected=[k for k,v in checks.items() if not v]
    report={
        "schema":"pr60-frozen-60train-12val-untouched-blind-map-data-admission-v1",
        "decision":"PASS_60_TRAIN_TRUE_DYNAMICS_READY_FOR_GPU_AGENT" if not rejected
                   else "NO_GO_FURTHER_BULK_OR_GPU_TRAINING_DATA",
        "predeclared_criteria":criteria,"checks":checks,"failures":rejected,
        "PR59_training_source_SHA":original_quality["source_split_sha256"]["train"],
        "PR59_validation_source_SHA":original_quality["source_split_sha256"]["validation"],
        "new_three_source_SHA":source_hash,
        "protocol_sha256":hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "verified_training_scenarios":24+len(new_cases),
        "verified_validation_scenarios":len(old_val),
        "BLIND_TEST_source_scenes_generated":0,
        "BLIND_TEST_source_files_read":0,
        "true_new_train_public_transitions":total_new_steps,
        "original_train_true_public_transitions":sum(len(e["raw_actual_public_world_transitions"]) for e in old_train),
        "validation_true_public_transitions":sum(len(e["raw_actual_public_world_transitions"]) for e in old_val),
        "new_train_true_completed_scenes":completed,
        "new_train_right_censored_scenes":36-completed,
        "new_train_true_station_fault_onsets":faults["station_fault_onset"],
        "new_train_true_station_recoveries":faults["station_recovery"],
        "new_train_H20_action_diff_scenes":H20,
        "new_train_substantial_mean_public_H20_L1_at_least_0_01_scenes":substantial_H20,
        "new_train_complete_branch_labels":complete_branch,
        "new_train_late_eventful_post56_samples":post56,
        "new_train_raw_semantic_uniqueness_fraction":within_new_uniqueness,
        "new_train_marginal_novelty_vs_original_24_raw_public_ticks":novelty_fraction,
        "new_train_overlap_old_raw_public_ticks":old_overlap,
        "gpu_training_completed":False,
        "GPU_hours":0,
        "scientific_boundaries":"All map families are still 7x5 grid. 60 independent TRAIN scenarios, 12 natural disjoint-map VAL scenes, 12 entirely unopened horizontal-shelf BLIND scenario manifest. This source doesn't prove the neural model learns good OOD behavior.",
        "per_true_new_train_scenario":per_case
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print("PR60_REAL_MARGINAL_DATA_QUALITY_GATING",json.dumps({
        k:v for k,v in report.items() if k!="per_true_new_train_scenario"
    },ensure_ascii=False),flush=True)
    return report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--original",required=True,type=Path)
    ap.add_argument("--old-quality",required=True,type=Path)
    ap.add_argument("--new",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    x=ap.parse_args()
    evidence(sorted(x.new.glob("*/data.json")),x.original,x.old_quality,x.out)


if __name__=="__main__":
    main()
