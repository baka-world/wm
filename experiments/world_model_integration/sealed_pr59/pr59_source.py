"""PR59 true-world TRAIN/VALIDATION data acquisition on distinct 7x5 MAP FAMILIES.

Only real BusinessSimulator + public topology + PR54 causal history.
No GPU, no neural training, no BLIND TEST run. If any case fails or times
out, retain the original source and mark it (do not rewrite completed time).
All existing PR54/PR58 developed scenarios are excluded by new seeds/maps.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import json
import time

from experiments.world_model_integration.topology_pr56.pr56_trial import run_scene
from experiments.world_model_integration.topology_pr56.pr56_public_map import build_topology_business,public_topology_for_graph
from experiments.world_model_integration.topology_pr56.pr56_readonly_stall import articulation_cells
from experiments.world_model_integration.stress_pr48.pr48_scenarios import PROTOCOL as STRESS_PROTOCOL

PROTOCOL=Path(__file__).with_name("protocol.json")
IDENTITY="PR59_SEALED_TOPOLOGY_FAMILY_WORLD_LEARNING_DATA_PRE_GPA_V1_20261112"
REGIMES=("congestion","multi_fault","staggered_arrivals")
APPROVED_SPLITS=("train","validation")


def read_protocol():
    p=json.loads(PROTOCOL.read_text())
    if p.get("protocol")!=IDENTITY:
        raise AssertionError("source split version changed")
    if p["parent_pr58_head"]!="60b1ea6864744a81ec9dad8149ea8396197f0ae1" or p["previous_real_quality_run"]!=43230426878:
        raise AssertionError("previous real high-quality map evidence changed")
    if p["gpu"]["used"] or p["gpu"]["training"]:
        raise AssertionError("CPU true data only")
    expected={name:p["splits"][name]["episode_count"] for name in ("train","validation","test")}
    if expected!={"train":24,"validation":12,"test":12}:
        raise AssertionError("registered train/val/BLIND test sizes changed")
    maps=p["layouts"]
    if len(maps)!=10 or len({m["id"] for m in maps})!=10 or len(p["cohort"])!=48:
        raise AssertionError("missing independent known public map cohorts")
    families={k:{m["family"] for m in maps if m["id"] in p["splits"][k]["maps"]}
              for k in expected}
    if any(families[a]&families[b] for a in families for b in families if a!=b):
        raise AssertionError("map family heldout split violated")
    if len({c["seed"] for c in p["cohort"]})!=48 or min(c["seed"] for c in p["cohort"])<59000000:
        raise AssertionError("reused or duplicated prior study scenario seed")
    if any(c["split"] not in expected for c in p["cohort"]):
        raise AssertionError("unregistered data label")
    for kind,n in expected.items():
        subset=[c for c in p["cohort"] if c["split"]==kind]
        if len(subset)!=n or any(c["layout"] not in p["splits"][kind]["maps"] for c in subset):
            raise AssertionError("map split source leaked or omitted")
        if set(c["regime"] for c in subset)!=set(REGIMES):
            raise AssertionError("no 3 stress mechanisms in a split")
    if p["splits"]["test"]["generate_in_this_cpu_stage"] is not False:
        raise AssertionError("source protocol must never generate blind test")
    if any(x["regime"]=="multi_fault" and x["station_id"]==x["secondary_station_id"]
           for x in p["cohort"]):
        raise AssertionError("invalid double-fault generator")
    return p


def validate_layout_business(case,p,stress):
    business,geo=build_topology_business(case,stress,p)
    points=[s.position for s in business.scenario.stations.values()]
    points += [r.position for r in business.scenario.robots]
    points += [r.home for r in business.scenario.robots]
    if not all(business.scenario.grid.contains(point) for point in points):
        raise AssertionError("station/robot/home in blocked cell")
    if any(business.scenario.grid.shortest_path(points[0],pt) is None for pt in points):
        raise AssertionError("factual business map lacks station/robot connectivity")
    graph={"nodes":[{"id":s.id,"kind":"station","pos":list(s.position)}
                    for s in business.scenario.stations.values()]+[
                  {"id":r.id,"kind":"robot","pos":list(r.position)}
                    for r in business.scenario.robots]}
    topo=public_topology_for_graph(business.scenario.grid,geo,graph)
    if articulation_cells(topo):
        raise AssertionError("new natural source geography has isolated passage cut vertex")
    stations=list(business.scenario.stations.values())
    det=sum(geo.distance(a.position,b.position)>
            abs(a.position[0]-b.position[0])+abs(a.position[1]-b.position[1])
            for i,a in enumerate(stations) for b in stations[i+1:])
    if det<1:
        raise AssertionError("map group has no true geodesic route detour")
    return topo,det


def sample(split,out):
    p=read_protocol()
    if split not in APPROVED_SPLITS:
        raise PermissionError("BLIND test source must not be generated or read by PR59")
    if out.exists():
        raise FileExistsError("original data shards are immutable and cannot be overwritten")
    stress=json.loads(STRESS_PROTOCOL.read_text())
    began=time.process_time()
    cases=[c for c in p["cohort"] if c["split"]==split]
    events=Counter()
    raw_samples=0
    all_sig=set()
    groups=[]
    for case in cases:
        topo,det=validate_layout_business(case,p,stress)
        scene=run_scene(case,stress,p)
        if scene["public_map"]["static_map_digest"]!=topo["static_map_digest"]:
            raise AssertionError("model did not observe true business grid")
        for row in scene["raw_actual_public_world_transitions"]:
            key=(topo["static_map_digest"],row["quality"]["semantic_input_state_sha256"])
            all_sig.add(key)
        events.update(scene["full_event_counts"])
        raw_samples+=scene["full_real_public_transitions"]
        groups.append(scene)
        info=scene["matched_legal_action_information"]
        print("PR59_REAL_NATURAL_WORLD",json.dumps({
            "id":case["id"],"split":split,"regime":case["regime"],
            "family":case["map_family"],"layout":case["layout"],
            "profile":case["profile"],"real_ticks":scene["real_factual_ticks"],
            "completed":scene["real_full_episode_status"]=="completed",
            "map_detours":det,"late_dynamic":scene["late_eventful_transitions_after_original56"],
            "H20_true_action_effect":info.get("H20_has_state_action_effect",False),
            "all_candidate_branches_completed":info.get("all_alternatives_safely_completed",False),
            "censored_final_makespan":not info.get("all_alternatives_safely_completed",False)
        },ensure_ascii=False),flush=True)
    results={
        "split":split,"scenarios":len(groups),"full_raw_factual_transitions":raw_samples,
        "full_real_completion":sum(x["real_full_episode_status"]=="completed" for x in groups),
        "censored_episodes":sum(x["real_full_episode_status"]!="completed" for x in groups),
        "true_station_fault_onsets":events["station_fault_onset"],
        "true_station_recoveries":events["station_recovery"],
        "late_eventful_transitions_after_56":sum(x["late_eventful_transitions_after_original56"] for x in groups),
        "same_state_legal_H20_action_effect":sum(x["matched_legal_action_information"].get("H20_has_state_action_effect",False) for x in groups),
        "complete_action_branch_groups":sum(x["matched_legal_action_information"].get("all_alternatives_safely_completed",False) for x in groups),
        "unique_public_map_semantic_inputs":len(all_sig),
        "raw_map_semantic_uniqueness_ratio":len(all_sig)/max(1,raw_samples),
        "observed_source_events":dict(events),
        "trained_network":False,"gpu_used":0,
        "cpu_process_seconds":time.process_time()-began
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({
        "schema":"pr59-sha-sealed-natural-public-world-map-family-split-v1",
        "split":split,
        "source_protocol_sha256":hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "scenes":groups,"summary":results,
        "BLIND_TEST_RUN":False,
        "label_policy":"All actual public t->t+1 observations preserved; any completed-makespan comparisons require all action branches truly finished. No t+1-target-dependent filtering of this source."
    },ensure_ascii=False)+"\n")
    print("PR59_REAL_PUBLIC_MAP_SPLIT",json.dumps(results,ensure_ascii=False),flush=True)


def main():
    pa=argparse.ArgumentParser()
    pa.add_argument("--split",choices=("train","validation","test"),required=True)
    pa.add_argument("--out",required=True,type=Path)
    a=pa.parse_args()
    sample(a.split,a.out)


if __name__=="__main__":
    main()
