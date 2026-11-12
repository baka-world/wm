"""PR58 preregistered map-only intervention against all original PR57 dev scenes.

Reuse EXACT original PR56 BusinessSimulator, CP-SAT proposal, truth observer,
factual trajectory and matched-prefix counterfactual implementation.
Only the static PUBLIC Grid obstacles differ. All private schedules, jobs,
stations, resource rules, robots, regime, mechanism and the random seed
are checked to be equal to the original corresponding PR56 development case.

This produces DEVELOPMENT DATA only; original PR57 right-censored runs
are preserved separately as mandatory paired controls.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import json
import time

from experiments.world_model_integration.topology_pr56.pr56_public_map import (
    protocol as original_protocol,build_topology_business,
    public_topology_for_graph
)
from experiments.world_model_integration.topology_pr56.pr56_readonly_stall import articulation_cells
from experiments.world_model_integration.topology_pr56.pr56_trial import run_scene
from experiments.world_model_integration.stress_pr48.pr48_scenarios import PROTOCOL as STRESS_PROTOCOL

PROTOCOL=Path(__file__).with_name("protocol.json")
ID="PR58_STATION_FREE_BYPASS_PAIRED_WORLD_DATA_20261112_V1"
REGIMES=("congestion","multi_fault","staggered_arrivals")
ORIGINAL_HEAD="f3dc4605edf0d2afaaab717562c6744f0de64f53"
ORIGINAL_ACTIONS_HEAD="ec22b94a919400ddce7a34e1d0575d6fe9682c6c"


def read_protocol():
    p=json.loads(PROTOCOL.read_text())
    old=original_protocol()
    if (p.get("protocol")!=ID or p["parent_pr57_head"]!=ORIGINAL_HEAD
        or p["original_immutable_run"]!=43219314493
        or p["original_source_head"]!=ORIGINAL_ACTIONS_HEAD):
        raise ValueError("PR58 reproducibility provenance changed")
    if p["status"]["gpu_hours"] or p["status"]["no_new_model_training"] is not True:
        raise ValueError("this is CPU-only true-world data acquisition")
    if p["max_ticks"]!=260 or p["public_history_window"]!=4:
        raise ValueError("original public state history/physical time altered")
    if p["actual_legal_alternatives"]!=old["actual_legal_alternatives"]:
        raise ValueError("original legal action proposal contract changed")
    if len(p["layouts"])!=4 or len(p["cohort"])!=12:
        raise ValueError("need 4 true maps and 12 paired original scenarios")
    original={x["id"]:x for x in old["cohort"]}
    if len(original)!=12 or len({x["seed"] for x in p["cohort"]})!=12:
        raise ValueError("paired original source seed coverage incomplete")
    pairings=set()
    for spec in p["cohort"]:
        original_id=spec["paired_pr56_original_case_id"]
        if original_id not in original or original_id in pairings:
            raise ValueError("original case not uniquely preserved")
        pairings.add(original_id)
        src=original[original_id]
        for k in ("seed","regime","mechanism","station_id",
                  "secondary_station_id","job_count","profile"):
            if spec[k]!=src[k]:
                raise ValueError("not a map-only paired environment; changed "+k)
        if spec["paired_pr56_original_layout"]!=src["layout"]:
            raise ValueError("original obstacle control lost")
        if spec["split"]!="paired_topology_development_only":
            raise ValueError("do not relabel observed development maps as blind test")
    if pairings!=set(original):
        raise ValueError("did not retain all 12 old scenes including five censored")
    for regime in REGIMES:
        group=[x for x in p["cohort"] if x["regime"]==regime]
        if len(group)!=4 or {x["layout"] for x in group}!={x["id"] for x in p["layouts"]}:
            raise ValueError("new map/stress coverage unbalanced")
    if len({json.dumps(x["blocked"]) for x in p["layouts"]})!=4:
        raise ValueError("new layout duplicates")
    return p,old


def compare_map_only(spec,p,old,stress):
    orig=next(x for x in old["cohort"]
              if x["id"]==spec["paired_pr56_original_case_id"])
    oworld,ogeo=build_topology_business(orig,stress,old)
    nworld,ngeo=build_topology_business(spec,stress,p)
    # All factual exogenous source mechanics are frozen; only PUBLIC topology
    # is changed. An ID-based name string difference is not a dynamic change.
    if (oworld.rules!=nworld.rules
        or oworld.jobs!=nworld.jobs
        or oworld.scenario.stations!=nworld.scenario.stations
        or oworld.scenario.robots!=nworld.scenario.robots
        or oworld.scenario.max_ticks!=nworld.scenario.max_ticks):
        raise AssertionError("paired map case changed workflow, hidden faults, jobs, robots or time")
    if (oworld.scenario.grid.width!=nworld.scenario.grid.width
        or oworld.scenario.grid.height!=nworld.scenario.grid.height):
        raise AssertionError("map size changed")
    if oworld.scenario.grid.blocked==nworld.scenario.grid.blocked:
        raise AssertionError("new and old map identical")
    stations=list(nworld.scenario.stations.values())
    graph={"nodes":[
        {"kind":"station","id":s.id,"pos":list(s.position)} for s in stations
    ]+[{"kind":"robot","id":r.id,"pos":list(r.position)}
        for r in nworld.scenario.robots]}
    topo=public_topology_for_graph(nworld.scenario.grid,ngeo,graph)
    cut=articulation_cells(topo)
    if cut:
        raise AssertionError("new layout still has static articulation "+str(cut))
    station_idx=[tuple(s.position) for s in stations]
    if any(not nworld.scenario.grid.contains(pt) for pt in station_idx):
        raise AssertionError("new layout blocked station")
    orig_grid=nworld.scenario.grid
    for r in nworld.scenario.robots:
        for pt in (r.position,r.home):
            if not orig_grid.contains(pt):
                raise AssertionError("new layout blocked robot/home")
    distorted=sum(
        1 for i,a in enumerate(stations) for b in stations[i+1:]
        if ngeo.distance(a.position,b.position)>
           abs(a.position[0]-b.position[0])+abs(a.position[1]-b.position[1]))
    if distorted==0:
        raise AssertionError("new obstacle layout has no consequential station detour")
    if len(topo["walkable_cells"])!=35-len(orig_grid.blocked):
        raise AssertionError("map graph lost a traversable point")
    return {
        "source_pr56_case_id":orig["id"],
        "old_map":orig["layout"],"new_map":spec["layout"],
        "same_true_business_outages_jobs_robots_rules":True,
        "same_random_seed":spec["seed"],
        "map_static_digest":topo["static_map_digest"],
        "station_geodesic_detours":distorted,
        "static_articulation_count":len(cut),
        "changed_grid_obstacles_only":True,
    }


def run(regime,out):
    p,old=read_protocol()
    if regime not in REGIMES:
        raise ValueError("unregistered real stress regime")
    if out.exists():
        raise FileExistsError("write-once original actual world data")
    stress=json.loads(STRESS_PROTOCOL.read_text())
    started=time.process_time()
    scenes=[]
    for spec in p["cohort"]:
        if spec["regime"]!=regime:
            continue
        paired=compare_map_only(spec,p,old,stress)
        # PR56 original actual simulator/trajectory and paired legal action
        # source. No stand-in dynamics or homegrown replacement.
        case=run_scene(spec,stress,p)
        case["original_pr56_map_only_comparison"]=paired
        scenes.append(case)
        info=case["matched_legal_action_information"]
        print("PR58_TRUE_BYPASS_WORLD",json.dumps({
            "id":spec["id"],"old_map":paired["old_map"],
            "new_map":paired["new_map"],
            "true_completed":case["real_full_episode_status"]=="completed",
            "true_ticks":case["real_factual_ticks"],
            "observed_steps":case["full_real_public_transitions"],
            "legal_candidates":info.get("candidate_count",0),
            "all_action_branches_complete":info.get("all_alternatives_safely_completed",False),
            "H20_effect":info.get("H20_has_state_action_effect",False),
            "makespan_effect":info.get("makespan_action_effect",False)
        },ensure_ascii=False),flush=True)
    if len(scenes)!=4:
        raise AssertionError("each stress family must have 4 original paired trials")
    summary={
        "regime":regime,"scenes":len(scenes),
        "original_source_provenance":ORIGINAL_ACTIONS_HEAD,
        "primary_completed":sum(x["real_full_episode_status"]=="completed" for x in scenes),
        "primary_censored":sum(x["real_full_episode_status"]!="completed" for x in scenes),
        "all_candidate_branches_complete":sum(
            x["matched_legal_action_information"].get("all_alternatives_safely_completed",False)
            for x in scenes),
        "H20_actual_action_state_effect":sum(
            x["matched_legal_action_information"].get("H20_has_state_action_effect",False)
            for x in scenes),
        "known_completed_makespan_action_effect":sum(
            x["matched_legal_action_information"].get("makespan_action_effect",False)
            for x in scenes),
        "full_real_t0_to_terminal_transitions":sum(
            x["full_real_public_transitions"] for x in scenes),
        "observed_fault_onset":sum(x["full_event_counts"].get("station_fault_onset",0)
                                   for x in scenes),
        "observed_station_recovery":sum(x["full_event_counts"].get("station_recovery",0)
                                        for x in scenes),
        "late_dynamic_after_old_tick56":sum(
            x["late_eventful_transitions_after_original56"] for x in scenes),
        "gpu":0,"neural_training":0,"cpu_process_seconds":time.process_time()-started,
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({
        "schema":"pr58-real-map-only-paired-public-world-source-v1",
        "protocol_sha256":hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "regime":regime,"scenes":scenes,"summary":summary,
        "source_caveats":"All NEW and OLD scenarios are developer paired worlds, not unseen train/test. Equal fixed seed/schedule does not imply identical future stochastic draw order after altered policy."
    },ensure_ascii=False)+"\n")
    print("PR58_MAP_ONLY_TRUE_WORLD_SUMMARY",json.dumps(summary,ensure_ascii=False),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--regime",required=True,choices=REGIMES)
    ap.add_argument("--out",required=True,type=Path)
    args=ap.parse_args()
    run(args.regime,args.out)


if __name__=="__main__":
    main()
