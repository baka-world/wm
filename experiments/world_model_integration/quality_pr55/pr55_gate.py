"""PR55 frozen data admission gate, independent of learner/train/test metrics."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import json

from experiments.world_model_integration.quality_pr55.pr55_generate import (
    PROTOCOL,REGIMES,FLAGS,RARE,POSITIVE,validate,flag_transition,
    semantic_fingerprint,sha
)


def verify_scene(e,cfg,protocol_digest):
    scenario=e["case"]
    if scenario["split"]!="pilot_development_only":
        raise AssertionError("pilot may not pretend to be unseen test")
    rows=e["raw_trajectories"]
    if len(rows)!=e["full_public_transitions"] or not rows:
        raise AssertionError("raw true public trajectory truncated or absent")
    if len(rows)>cfg["max_ticks"]:
        raise AssertionError("true simulator exceeded frozen time boundary")
    if rows[-1]["target"]["next_public_tick"]!=e["real_summary"]["ticks"]:
        raise AssertionError("did not preserve true LAST system state")
    selected=e["selected_ticks"]
    if selected!=sorted(set(selected)):
        raise AssertionError("curated ticks duplicate or out of order")
    if len(selected)!=e["curated_examples"] or len(selected)>cfg[
        "curated_training_preview"]["max_per_episode"]:
        raise AssertionError("curated sample count violated fixed budget")
    by_tick={r["input"]["public_tick"]:r for r in rows}
    if len(by_tick)!=len(rows) or list(by_tick)!=list(range(len(rows))):
        raise AssertionError("missing intermediate TRUE public tick")
    if not set(selected).issubset(by_tick):
        raise AssertionError("curated tick not a member of full true world trace")
    for row in rows:
        i=row["input"]
        t=row["target"]
        if (i["public_tick"]+1!=t["next_public_tick"]
            or i["feature_last_observed_tick"]!=i["public_tick"]):
            raise AssertionError("future public target used as input or time mismatch")
        if len(i["graph"]["nodes"])!=len(i["history"])!=len(t["node_target"]):
            raise AssertionError("public graph/time/future labels misaligned")
        if any(not x[-1] for x in i["history_visible_mask"]):
            raise AssertionError("current public nodes incorrectly absent")
        if any("outage_duration" in str(k) or "private_future" in str(k)
               for k in i):
            raise AssertionError("private future environment leaked into learner")
        if t["new_completed_task_stages"]>t["new_visible_task_stages"]:
            raise AssertionError("future public task lifecycle invalid")
        if row["quality"]["flags"]!=flag_transition(row):
            raise AssertionError("labels and frozen event classifier disagree")
        if row["quality"]["semantic_public_input_sha256"]!=semantic_fingerprint(row):
            raise AssertionError("public graph fingerprint mismatch")
        if row["quality"]["training_preview_selected"]!=(i["public_tick"] in selected):
            raise AssertionError("quality curation changed original truth")
    actual=Counter(f for row in rows for f in row["quality"]["flags"])
    if dict(actual)!=e["event_flags_full"]:
        raise AssertionError("claimed event coverage not grounded in actual states")
    return actual


def audit(paths,out):
    cfg=json.loads(PROTOCOL.read_text())
    validate(cfg)
    expected_digest=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    if out.exists():
        raise FileExistsError("data quality audit is immutable")
    episodes=[]
    digests={}
    source_shas=set()
    for path in paths:
        d=json.loads(path.read_text())
        regime=d["regime"]
        if (d["schema"]!="pr55-quality-first-true-public-world-dev-pilot-v1"
            or d["protocol_sha256"]!=expected_digest or regime in digests):
            raise ValueError("data source or protocol differs from frozen pilot")
        if len(d["episodes"])!=6 or d["summary"]["GPU_used"] or d["summary"]["neural_training"]:
            raise ValueError("pilot source incomplete or exceeded resource budget")
        digests[regime]=hashlib.sha256(path.read_bytes()).hexdigest()
        episodes+=d["episodes"]
    if len(episodes)!=18 or set(digests)!=set(REGIMES):
        raise ValueError("not 18 complete unmerged real scene quality sources")
    wanted={x["id"]:x for x in cfg["pilot_cases"]}
    if len({e["case"]["id"] for e in episodes})!=18:
        raise AssertionError("duplicate scene or omitted original profile")
    total=Counter()
    stage=Counter()
    profile=Counter()
    raw_count=0
    curated_count=0
    static_count=0
    all_sig=set()
    unique_curated=set()
    late_events=0
    completed=0
    by_episode=[]
    for e in episodes:
        case=e["case"]
        if case["id"] not in wanted or case!=wanted[case["id"]]:
            raise AssertionError("source scenario modified after protocol precommitment")
        counts=verify_scene(e,cfg,expected_digest)
        total.update(counts)
        stage[case["regime"]]+=1
        profile[case["profile"]]+=1
        late_events+=e["late_eventful_transitions_after_original_56"]
        raw_count+=e["full_public_transitions"]
        curated_count+=e["curated_examples"]
        completed+=int(e["real_summary"]["status"]=="completed")
        for row in e["raw_trajectories"]:
            key=row["quality"]["semantic_public_input_sha256"]
            all_sig.add(key)
            if row["quality"]["training_preview_selected"]:
                unique_curated.add(key)
                static_count+=int(not set(row["quality"]["flags"])&set(RARE+POSITIVE))
        by_episode.append({
            "id":case["id"],"profile":case["profile"],
            "regime":case["regime"],
            "real_completion_status":e["real_summary"]["status"],
            "full_public_ticks":e["full_public_transitions"],
            "event_flags":e["event_flags_full"],
            "eventful_late_after_original56":e["late_eventful_transitions_after_original_56"],
            "semantic_uniqueness_within_episode":e["curated_unique_semantic_input_fraction"],
            "curated_static_ratio":e["curated_static_fraction"]
        })
    gate=cfg["quality_gate_frozen_before_pilot"]
    missing=[name for name in gate["mandatory_global_flags"] if total[name]==0]
    checks={
        "no_source_safety_failure":True,
        "no_future_leakage_found":True,
        "required_event_types_present":len([x for x in FLAGS if total[x]>0])>=
             gate["min_global_event_categories_present"],
        "mandatory_events_present":not missing,
        "all_six_real_profiles_covered":all(profile[p]==3 for p in gate["mandatory_pilot_profiles"]),
        "all_three_stress_regimes_covered":all(stage[r]==6 for r in REGIMES),
        "real_full_episode_late_dynamics_found":late_events>0,
        "curated_static_fraction_within_bound":static_count/max(1,curated_count)<=
             gate["max_curated_static_ratio"],
        "curated_semantic_uniqueness_within_bound":len(unique_curated)/max(1,curated_count)>=
             gate["min_curated_semantic_input_uniqueness_ratio"],
        "all_scenarios_have_factual_full_trajectories":all(
            x["full_public_ticks"]>56 for x in by_episode),
        "no_silent_censoring_of_true_episode":all(
            x["full_public_ticks"]>0 for x in by_episode)
    }
    reason=[k for k,v in checks.items() if not v]
    passed=not reason
    result={
        "schema":"pr55-raw-world-dynamics-data-quality-admission-v1",
        "decision":"PASS_NEXT_PHASE_DESIGN" if passed else "NO_GO_EXPANSION",
        "quality_checks":checks,"failure_reasons":reason,
        "source_data_sha256":digests,"protocol_sha256":expected_digest,
        "independent_dev_scenarios":len(episodes),
        "raw_actual_business_transitions":raw_count,
        "curated_training_preview_transitions":curated_count,
        "raw_event_flag_counts":dict(total),
        "mandatory_missing_event_types":missing,
        "full_episode_late_eventful_transitions_after_original56":late_events,
        "full_true_episode_completed":completed,
        "curated_static_fraction":static_count/max(1,curated_count),
        "curated_semantic_uniqueness_fraction":len(unique_curated)/max(1,curated_count),
        "raw_semantic_uniqueness_fraction":len(all_sig)/max(1,raw_count),
        "by_original_business_scenario":by_episode,
        "training_ready":False,
        "decision_aware_data_ready":False,
        "cross_map_generalization_covered":False,
        "required_next_quality_gate":"Pre-register and measure identical-public-prefix alternate lawful actions and different true next-state outcomes; require genuinely different topology in HELDOUT before OOD/world-model claims.",
        "evaluation_note":"ALL 18 pilot scenarios are development only. This event-enriched subset must never be treated as natural event prevalence or independent blind TEST.",
        "cost":{"GPU":0,"neural_training":0}
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print("PR55_FROZEN_QUALITY_ADMISSION",json.dumps({
        k:v for k,v in result.items() if k not in ("by_original_business_scenario",)
    },ensure_ascii=False),flush=True)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    files=sorted(args.input.glob("*/data.json"))
    audit(files,args.out)


if __name__=="__main__":
    main()
