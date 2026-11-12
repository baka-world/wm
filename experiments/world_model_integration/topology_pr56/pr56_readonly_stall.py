"""PR56 immutable ORIGINAL 12 actual worlds: static chokepoint and timeout audit.

Purely read-only and no simulator, PyTorch, GPU or new trajectories.
The source is the already SHA-verified original Actions source run.
A capped makespan tick=260 is CENSORED, not completion at tick260.

A static articulation node in the map is a structural choke point, not
automatically a proof of the particular dynamic deadlock mechanism.
"""
from __future__ import annotations
from pathlib import Path
from collections import Counter, deque
import argparse
import json


def articulation_cells(topo):
    cells=topo["walkable_cells"]
    adj={i:set() for i in range(len(cells))}
    for a,b in topo["directed_map_edges"]:
        adj[a].add(b)
    def connected_without(ban):
        remains=[i for i in adj if i!=ban]
        if not remains:return True
        start=remains[0]
        seen={start};q=deque([start])
        while q:
            i=q.popleft()
            for j in adj[i]:
                if j!=ban and j not in seen:
                    seen.add(j);q.append(j)
        return len(seen)==len(remains)
    cuts=[cells[i] for i in adj if not connected_without(i)]
    return cuts


def decode_scene(case):
    raw=case["raw_actual_public_world_transitions"]
    if not raw or len(raw)!=case["full_real_public_transitions"]:
        raise AssertionError("incomplete real original public stream")
    last=raw[-1]
    if last["target"]["next_public_tick"]!=case["real_factual_ticks"]:
        raise AssertionError("missing terminal true observation")
    tail=raw[-30:]
    tail_events=Counter(f for row in tail for f in row["quality"]["flags"])
    topo=case["public_map"]
    cuts=articulation_cells(topo)
    station_points={tuple(n["pos"]):n["id"] for n in raw[0]["input"]["graph"]["nodes"]
                    if n["kind"]=="station"}
    station_cuts=[{"cell":c,"station":station_points[tuple(c)]}
                  for c in cuts if tuple(c) in station_points]
    terminal=last["target"]["node_target"]
    entities=last["input"]["graph"]["nodes"]
    status=Counter()
    for n,values in zip(entities,terminal):
        if n["kind"]=="task":
            status[("waiting","assigned","in_transit","completed")[
                max(range(4),key=lambda i:values[i])]]+=1
    branches=case["real_factual_branches"]
    all_complete=all(x["true_completed"] for x in branches)
    primary_completed=case["real_full_episode_status"]=="completed"
    if (case["real_factual_ticks"]==260 and not primary_completed
        and status["waiting"]+status["assigned"]+status["in_transit"]==0):
        raise AssertionError("censored scene has no remaining public known work, investigate")
    return {
        "id":case["id"],"regime":case["case"]["regime"],
        "layout":case["case"]["layout"],
        "original_public_topology_sha":topo["static_map_digest"],
        "real_terminal_tick":case["real_factual_ticks"],
        "original_factual_completed":primary_completed,
        "real_action_branch_complete_status":[x["true_completed"] for x in branches],
        "CENSORED_makespan_if_any_branch_times_out":not all_complete,
        "last_30_tick_events":dict(tail_events),
        "known_visible_stage_counts_by_status":dict(status),
        "original_static_articulation_cells":cuts,
        "station_on_static_articulation":station_cuts,
        "evidence_limit":"articulation is structural static topology only; no proof that it caused every simulator timeout"
    }


def run(source,out):
    if out.exists():raise FileExistsError("immutable diagnosis already exists")
    paths=sorted(source.glob("*/data.json"))
    if len(paths)!=3:raise AssertionError("original three real source artifacts required")
    episodes=[]
    for path in paths:
        raw=json.loads(path.read_text())
        if raw["schema"]!="pr56-original-real-topology-full-episode-and-action-pilot-v1":
            raise AssertionError("not original authentic dataset")
        if len(raw["scenes"])!=4:
            raise AssertionError("missing real 4-map group")
        episodes.extend(decode_scene(x) for x in raw["scenes"])
    if len(episodes)!=12:raise AssertionError("not exact preregistered 12 source scenes")
    by_layout={}
    for name in sorted({x["layout"] for x in episodes}):
        grp=[x for x in episodes if x["layout"]==name]
        by_layout[name]={
            "n":len(grp),"original_complete":sum(x["original_factual_completed"] for x in grp),
            "any_candidate_censored":sum(x["CENSORED_makespan_if_any_branch_times_out"] for x in grp),
            "static_choke_stations":grp[0]["station_on_static_articulation"],
            "static_articulation_cell_count":len(grp[0]["original_static_articulation_cells"]),
            "terminal_tick_by_regime":{x["regime"]:x["real_terminal_tick"] for x in grp}
        }
    out.parent.mkdir(parents=True,exist_ok=True)
    result={
        "schema":"pr56-readonly-original-layout-timeout-topology-diagnosis-v1",
        "original_real_development_scenes":12,
        "real_original_completed":sum(x["original_factual_completed"] for x in episodes),
        "real_original_censored":sum(not x["original_factual_completed"] for x in episodes),
        "CENSORED_action_groups":sum(x["CENSORED_makespan_if_any_branch_times_out"] for x in episodes),
        "by_static_layout":by_layout,
        "source_new_simulations":0,"GPU":0,"neural_training":0,
        "result_label_integrity":"No timed-out episode is assigned a completed-makespan score.",
        "limits":"No causal determination of specific timeout from articulation alone; no independent held-out map family or world-model fit.",
        "original_unfiltered_scenes":episodes
    }
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print("PR56_ORIGINAL_REAL_MAP_STALL_DIAGNOSIS",json.dumps({
        k:v for k,v in result.items() if k!="original_unfiltered_scenes"
    },ensure_ascii=False),flush=True)


def main():
    a=argparse.ArgumentParser()
    a.add_argument("--source",type=Path,required=True)
    a.add_argument("--out",type=Path,required=True)
    o=a.parse_args()
    run(o.source,o.out)


if __name__=="__main__":
    main()
