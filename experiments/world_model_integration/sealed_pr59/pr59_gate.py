"""PR59 immutable real TRAIN/VAL quality gate before *any* GPU training.

Checks the raw BusinessSimulator full t+1 public state and H5/H20 action
information (or honest censored label), exact map family isolation, true
robot/station-route geometry, late events and stale-state duplication.
All world outcomes are preserved. 12 BLIND TEST scenarios remain
protocol-only; absolutely no test outcomes are opened or generated.

A source PASS means data is fit for CPU model/batching preflight, not
that a neural model has been trained or generalized to unseen maps.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import json

from experiments.world_model_integration.sealed_pr59.pr59_source import (
    PROTOCOL,REGIMES,read_protocol,validate_layout_business
)
from experiments.world_model_integration.topology_pr56.pr56_readonly_stall import (
    articulation_cells
)
from experiments.world_model_integration.quality_pr55.pr55_generate import (
    RARE,POSITIVE,flag_transition
)
from experiments.world_model_integration.stress_pr48.pr48_scenarios import (
    PROTOCOL as STRESS_PROTOCOL
)


def source_quality(paths,out):
    if out.exists():
        raise FileExistsError("immutable PR59 data admission output already exists")
    p=read_protocol()
    sha=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    if len(paths)!=2:
        raise AssertionError("only natural TRAIN + validation source are authorized, not BLIND test")
    by_split={}
    source_hashes={}
    for path in paths:
        obj=json.loads(path.read_text())
        split=obj.get("split")
        if (split not in ("train","validation") or
            split in by_split or
            obj["schema"]!="pr59-sha-sealed-natural-public-world-map-family-split-v1"
            or obj["source_protocol_sha256"]!=sha or
            obj["BLIND_TEST_RUN"]):
            raise AssertionError("source corrupt or BLIND test unexpectedly consumed")
        by_split[split]=obj
        source_hashes[split]=hashlib.sha256(path.read_bytes()).hexdigest()
    if set(by_split)!={"train","validation"}:
        raise AssertionError("must have exactly two separately sourced natural distributions")
    stress=json.loads(STRESS_PROTOCOL.read_text())
    registered={c["id"]:c for c in p["cohort"]}
    all_profiles=Counter()
    results={}
    for split,src in by_split.items():
        cases=src["scenes"]
        expected=[x for x in registered.values() if x["split"]==split]
        if (len(cases)!=len(expected) or len(set(x["id"] for x in cases))!=len(cases)
            or {c["id"] for c in cases}!={c["id"] for c in expected}):
            raise AssertionError("source missing distinct independent scenarios")
        fault=Counter()
        by_regime=Counter()
        map_digest={}
        unique_sig=set()
        raw_tick=0
        h20=0
        completed=0
        label_complete=0
        labels_different=0
        after56=0
        all_scene=[]
        for e in cases:
            spec=e["case"]
            if spec!=registered[e["id"]]:
                raise AssertionError("source model case changed after frozen cohort")
            if spec["split"]!=split or spec["layout"] not in p["splits"][split]["maps"]:
                raise AssertionError("map topology split has leaked")
            all_profiles[spec["profile"]]+=1
            by_regime[spec["regime"]]+=1
            true_map,detours=validate_layout_business(spec,p,stress)
            source_map=e["public_map"]
            if (source_map["static_map_digest"]!=true_map["static_map_digest"]
                or bool(articulation_cells(source_map))
                or detours<1):
                raise AssertionError("source train/VAL public map not the actual simulator map")
            map_digest.setdefault(spec["layout"],set()).add(source_map["static_map_digest"])
            rows=e["raw_actual_public_world_transitions"]
            if (len(rows)!=e["full_real_public_transitions"]
                or len(rows)!=e["real_factual_ticks"]
                or not rows or rows[-1]["target"]["next_public_tick"]!=e["real_factual_ticks"]):
                raise AssertionError("real public world history is cropped / missing true termination")
            for i,row in enumerate(rows):
                x=row["input"]
                t=row["target"]
                if (x["public_tick"]!=i or x["feature_last_observed_tick"]!=i
                    or t["next_public_tick"]!=i+1
                    or x["public_map"]["static_map_digest"]!=source_map["static_map_digest"]
                    or not all(h[-1] for h in x["history_visible_mask"])
                    or len(x["history"])!=len(x["graph"]["nodes"])):
                    raise AssertionError("public current history/time/source topology corrupt")
                if any(key not in (
                    "public_tick","graph","history","history_visible_mask",
                    "node_persistence","feature_last_observed_tick","input_source",
                    "public_map"
                ) for key in x):
                    raise AssertionError("unregistered private future in model input")
                if (t["new_completed_task_stages"]>t["new_visible_task_stages"]
                    or len(t["node_target"])!=len(x["graph"]["nodes"])):
                    raise AssertionError("true next public state impossible or misaligned")
                if row["quality"]["flags"]!=flag_transition(row):
                    raise AssertionError("real event coverage reporting ungrounded")
                unique_sig.add((
                    source_map["static_map_digest"],
                    row["quality"]["semantic_input_state_sha256"]
                ))
                fault.update(row["quality"]["flags"])
            raw_tick+=len(rows)
            completed+=int(e["real_full_episode_status"]=="completed")
            info=e["matched_legal_action_information"]
            h20+=int(info.get("H20_has_state_action_effect",False))
            label_complete+=int(info.get("all_alternatives_safely_completed",False))
            labels_different+=int(info.get("makespan_action_effect",False))
            if info.get("makespan_action_effect",False) and not info.get(
                "all_alternatives_safely_completed",False):
                raise AssertionError("censored makespan falsely labeled completed")
            after56+=e["late_eventful_transitions_after_original56"]
            all_scene.append({
                "id":e["id"],"layout":spec["layout"],"family":spec["map_family"],
                "regime":spec["regime"],"profile":spec["profile"],
                "ticks":len(rows),
                "factual_completed":e["real_full_episode_status"]=="completed",
                "h20_action_effect":info.get("H20_has_state_action_effect",False),
                "candidates":info.get("candidate_count",0),
                "all_action_branches_complete":info.get("all_alternatives_safely_completed",False),
                "completed_makespan_action_effect":info.get("makespan_action_effect",False),
                "observed_events":e["full_event_counts"]
            })
        if set(by_regime)!=set(REGIMES):
            raise AssertionError("missing an entire stress condition")
        if set(map_digest)!=set(p["splits"][split]["maps"]) or any(
            len(v)!=1 for v in map_digest.values()):
            raise AssertionError("map count or public topology identity invalid")
        report={
            "split":split,"genuine_scenarios":len(cases),
            "actual_full_public_t_to_next_transitions":raw_tick,
            "completed":completed,
            "right_censored":len(cases)-completed,
            "station_fault_onsets":fault["station_fault_onset"],
            "station_recoveries":fault["station_recovery"],
            "observed_event_categories":dict(fault),
            "true_H20_alternative_action_effect":h20,
            "all_branches_complete_for_makespan":label_complete,
            "valid_completed_makespan_action_effect":labels_different,
            "real_eventful_transitions_after_old_56":after56,
            "raw_semantic_input_uniqueness_ratio":len(unique_sig)/max(1,raw_tick),
            "different_public_map_count":len(map_digest),
            "map_family":next(iter({x["case"]["map_family"] for x in cases})),
            "by_regime":dict(by_regime),
            "all_source_scenes":all_scene
        }
        results[split]=report
    gate=p["thresholds_pre_registered_before_cpu_acquisition"]
    tr=results["train"]
    val=results["validation"]
    checks={
        "train_24_real_independent_scenes":tr["genuine_scenarios"]==24,
        "validation_12_real_independent_scenes":val["genuine_scenarios"]==12,
        "known_public_train_val_topology_never_overlaps":tr["map_family"]!=val["map_family"] and tr["different_public_map_count"]==6 and val["different_public_map_count"]==2,
        "all_protocol_reserved_BLIND_TEST_untouched":p["splits"]["test"]["generate_in_this_cpu_stage"] is False and not any(c["case"]["split"]=="test" for src in by_split.values() for c in src["scenes"]),
        "true_original_simulator_source_continuous_and_safe":True,
        "train_censor_fraction_below_predeclared":tr["right_censored"]/24<=gate["allow_train_censor_fraction_at_most"],
        "validation_censor_fraction_below_predeclared":val["right_censored"]/12<=gate["allow_validation_censor_fraction_at_most"],
        "train_true_fault_and_recovery_present":tr["station_fault_onsets"]>=gate["at_least_train_station_fault_onsets"] and tr["station_recoveries"]>=gate["at_least_train_station_recoveries"],
        "validation_true_fault_and_recovery_present":val["station_fault_onsets"]>=gate["at_least_validation_station_fault_onsets"] and val["station_recoveries"]>=gate["at_least_validation_station_recoveries"],
        "true_H20_action_relevant_train":tr["true_H20_alternative_action_effect"]>=gate["at_least_train_real_h20_action_effect_scenes"],
        "true_H20_action_relevant_validation":val["true_H20_alternative_action_effect"]>=gate["at_least_validation_real_h20_action_effect_scenes"],
        "true_makespan_complete_action_train":tr["all_branches_complete_for_makespan"]>=gate["at_least_train_complete_legal_branch_groups"],
        "true_makespan_complete_action_validation":val["all_branches_complete_for_makespan"]>=gate["at_least_validation_complete_legal_branch_groups"],
        "train_real_late_dynamic_samples":tr["real_eventful_transitions_after_old_56"]>=gate["at_least_train_rare_late_events_after_tick56"],
        "validation_real_late_dynamic_samples":val["real_eventful_transitions_after_old_56"]>=gate["at_least_validation_rare_late_events_after_tick56"],
        "train_semantic_nonduplicate_ratio":tr["raw_semantic_input_uniqueness_ratio"]>=gate["min_train_distinct_raw_semantic_input_ratio"],
        "all_six_source_real_business_profiles_covered":all(all_profiles[pr]>0 for pr in p["profiles"]),
        "no_future_inputs_or_hidden_TEST_leakage":True,
        "no_GPU_or_new_neural_training":True
    }
    failures=[key for key,val in checks.items() if not val]
    result={
        "schema":"pr59-natural-map-family-physically-unopened-test-data-admission-v1",
        "scientific_verdict":"PASS_QUALITY_TO_HANDOFF_GRAPH_TRAINING" if not failures
                              else "NO_GO_NEURAL_TRAINING_DATA",
        "checks":checks,"failures":failures,
        "source_split_sha256":source_hashes,
        "protocol_sha256":sha,
        "split_statistics":results,
        "BLIND_TEST_source_generated":False,
        "BLIND_TEST_source_read":False,
        "training_run_executed":False,
        "neural_model_OOD_results":None,
        "GPU_hours":0,
        "limits":"Development acquisition only; VAL maps held out from TRAIN and observed for QA, while BLIND horizontal shelf maps have no outcome files. Same 7x5 size. No model trained and no independent OOD score.",
        "next":"If PASS, finish PyG multimap batching/gradient tests; produce GPU agent handoff with a new 60-scene dataset expansion gate and physically untouched original BLIND TEST."
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print("PR59_SOURCE_QUALITY_GATE",json.dumps({
        k:v for k,v in result.items() if k!="split_statistics"
    },ensure_ascii=False),flush=True)
    print("PR59_SOURCE_PER_SPLIT_STAT",json.dumps({
        k:{n:v for n,v in val.items() if n!="all_source_scenes"}
        for k,val in results.items()
    },ensure_ascii=False),flush=True)
    return result


def main():
    pa=argparse.ArgumentParser()
    pa.add_argument("--source",type=Path,required=True)
    pa.add_argument("--out",type=Path,required=True)
    a=pa.parse_args()
    source_quality(sorted(a.source.glob("*/data.json")),a.out)


if __name__=="__main__":
    main()
