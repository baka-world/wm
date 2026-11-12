"""PR60 additional 36 true-world high-information TRAIN-only CPU simulations.

Extend previously immutable 24 TRAIN scenarios to 60. No new
VALIDATION or BLIND TEST observation, and no network training/GPU here.
Use exactly original PR56 BusinessSimulator/CP-SAT and PR59 published
public map, 4-step causal history, true t+1 entity and H5/H20 lawful
action intervention source. Retain every factual original tick even
if the episode is time-limited/right-censored.
"""
from __future__ import annotations
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import time

from experiments.world_model_integration.sealed_pr59.pr59_source import (
    PROTOCOL as ORIGINAL_PROTOCOL,read_protocol as original_protocol,
    validate_layout_business
)
from experiments.world_model_integration.stress_pr48.pr48_scenarios import (
    PROTOCOL as STRESS_PROTOCOL
)
from experiments.world_model_integration.topology_pr56.pr56_trial import run_scene

PROTOCOL=Path(__file__).with_name("protocol.json")
IDENTITY="PR60_MARGINAL_TRUE_WORLD_TRAIN_BATCH_TO_GPU_HANDOFF_V1_20261112"
REGIMES=("congestion","multi_fault","staggered_arrivals")


def protocol():
    p=json.loads(PROTOCOL.read_text())
    old=original_protocol()
    if p.get("protocol")!=IDENTITY:
        raise AssertionError("unregistered true-world training expansion")
    if p["source_parent_pr59_head"]!="1040c0743bba6d75631c03c79510ad85aca98973":
        raise AssertionError("wrong exact prior PyG public-map model branch")
    if (p["original_pr59_source_actions_run"]!=43234653164 or
        p["original_pr59_source_head"]!="48acd54b19805c4582bd616bb93fc81c7c92b10d" or
        p["original_pr59_quality_actions_run"]!=43235876892 or
        p["original_pr59_quality_verdict"]!="PASS_QUALITY_TO_HANDOFF_GRAPH_TRAINING"):
        raise AssertionError("original frozen 24+12 actual quality data changed")
    if p["original_pr59_protocol_sha256"]!=hashlib.sha256(ORIGINAL_PROTOCOL.read_bytes()).hexdigest():
        raise AssertionError("original train/VAL OOD geography protocol modified")
    if p["GPU_use"] or not p["actual_GPU_training_delegated"]:
        raise AssertionError("this task is CPU true data collection only")
    if p["actual_legal_alternatives"]!=old["actual_legal_alternatives"]:
        raise AssertionError("true CP-SAT legal H20 candidate source was redefined")
    if p["public_history_window"]!=old["public_history_window"] or p["max_ticks"]!=old["max_ticks"]:
        raise AssertionError("temporal history and world physics time changed")
    if p["new_train_count"]!=36 or p["combined_train_count"]!=60 or len(p["cohort"])!=36:
        raise AssertionError("unregistered amount of true additional data")
    if p["new_validation_count"] or p["new_blind_test_count"]:
        raise AssertionError("TEST and validation cannot be regenerated during staged TRAIN growth")
    oldseeds={c["seed"] for c in old["cohort"]}
    seeds={c["seed"] for c in p["cohort"]}
    if len(seeds)!=36 or oldseeds&seeds:
        raise AssertionError("reused previous source or BLIND seeds")
    if {x["layout"] for x in p["cohort"]}!=set(old["splits"]["train"]["maps"]):
        raise AssertionError("source changed train MAP families")
    if set(p["validation_family_maps_fixed"])!=set(old["splits"]["validation"]["maps"]) or set(p["blind_test_family_maps_frozen_unexecuted"])!=set(old["splits"]["test"]["maps"]):
        raise AssertionError("validation/BLIND family were retuned after observations")
    for c in p["cohort"]:
        if (c["split"]!="train" or c["layout"] not in old["splits"]["train"]["maps"]
            or c["map_family"]!="vertical_blocked_route_training"):
            raise ValueError("not additional TRAIN source or unapproved map family")
        if c["regime"]=="multi_fault" and c["station_id"]==c["secondary_station_id"]:
            raise ValueError("illegal identical double-fault site")
    if any(sum(c["regime"]==r for c in p["cohort"])!=12 for r in REGIMES):
        raise ValueError("additional TRAIN stress group coverage not balanced")
    return p,old


def acquire(regime,out):
    p,old=protocol()
    if regime not in REGIMES:
        raise ValueError("unknown frozen stress regime")
    if out.exists():
        raise FileExistsError("original facts are write-once")
    scenarios=[x for x in p["cohort"] if x["regime"]==regime]
    if len(scenarios)!=12:
        raise AssertionError("source requires 12 registered scenarios per stress group")
    stress=json.loads(STRESS_PROTOCOL.read_text())
    start=time.process_time()
    episodes=[]
    unique=set()
    events=Counter()
    for c in scenarios:
        topo,detour=validate_layout_business(c,old,stress)
        truth=run_scene(c,stress,old)
        if truth["public_map"]["static_map_digest"]!=topo["static_map_digest"] or detour<1:
            raise AssertionError("true business route map not public to model")
        for row in truth["raw_actual_public_world_transitions"]:
            unique.add((topo["static_map_digest"],row["quality"]["semantic_input_state_sha256"]))
        events.update(truth["full_event_counts"])
        episodes.append(truth)
        info=truth["matched_legal_action_information"]
        print("PR60_ACTUAL_NEW_WORLD_TRAIN",json.dumps({
            "id":c["id"],"regime":c["regime"],"profile":c["profile"],
            "layout":c["layout"],
            "source_real_ticks":truth["real_factual_ticks"],
            "true_completed":truth["real_full_episode_status"]=="completed",
            "H20_actual_action_effect":info.get("H20_has_state_action_effect",False),
            "fully_completed_candidate_branches":info.get("all_alternatives_safely_completed",False)
        },ensure_ascii=False),flush=True)
    total=sum(x["full_real_public_transitions"] for x in episodes)
    summary={
        "regime":regime,"split":"train","scenarios":len(episodes),
        "full_real_public_t_to_next_state_transitions":total,
        "actual_episode_completed":sum(x["real_full_episode_status"]=="completed" for x in episodes),
        "right_censored_at_time_cap":sum(x["real_full_episode_status"]!="completed" for x in episodes),
        "true_H20_state_action_effect_scenes":sum(x["matched_legal_action_information"].get("H20_has_state_action_effect",False) for x in episodes),
        "all_legal_branches_completed_scenes":sum(x["matched_legal_action_information"].get("all_alternatives_safely_completed",False) for x in episodes),
        "real_late_dynamic_after56":sum(x["late_eventful_transitions_after_original56"] for x in episodes),
        "actual_fault_onsets":events["station_fault_onset"],
        "actual_recoveries":events["station_recovery"],
        "new_batch_semantic_input_uniqueness_ratio":len(unique)/max(1,total),
        "real_events":dict(events),
        "gpu":0,"neural_training":0,
        "cpu_process_seconds":time.process_time()-start
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({
        "schema":"pr60-additional-real-public-map-train-source-v1",
        "split":"train","regime":regime,
        "source_protocol_sha256":hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "parent_pr59_protocol_sha256":hashlib.sha256(ORIGINAL_PROTOCOL.read_bytes()).hexdigest(),
        "BLIND_TEST_RUN":False,
        "scenes":episodes,"summary":summary,
        "source_limit":"UNFILTERED t0-to-terminal public truth. No completed makespan of censored candidate is used, no CUDA neural inference or training."
    },ensure_ascii=False)+"\n")
    print("PR60_REAL_ADDITIONAL_TRAIN_SOURCE",json.dumps(summary,ensure_ascii=False),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--regime",choices=REGIMES,required=True)
    ap.add_argument("--out",required=True,type=Path)
    a=ap.parse_args()
    acquire(a.regime,a.out)


if __name__=="__main__":
    main()
