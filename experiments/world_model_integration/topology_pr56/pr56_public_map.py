"""PR56 PUBLIC map topology encoding with ORIGINAL Grid and Geometry.

Do not fabricate physics. Route lengths and adjacency come from the exact
same grid/Geometry as CP-SAT and original BusinessSimulator. Public map
features contain no task future arrival times, hidden outage end dates or
latent simulator service countdown. Old PR54 models remain frozen/unmodified.
This is a verified dual-graph INPUT BRIDGE for the NEXT model, not a
new trained map encoder.
"""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import hashlib
import json

from lab_transport.domain import Grid
from lab_transport.dynamics_model import Geometry
from experiments.world_model_integration.quality_pr55.pr55_generate import build_quality_business
from experiments.world_model_integration.decision_learning_pr51.pr51_train import candidate_graph

PROTOCOL=Path(__file__).with_name("protocol.json")
ID="PR56_REAL_PUBLIC_TOPOLOGY_ACTION_DATA_GATE_V1_20261112"


def protocol():
    p=json.loads(PROTOCOL.read_text())
    if p.get("protocol")!=ID:
        raise ValueError("topology protocol must be frozen")
    if (p["based_on_pr55_head"]!="e6f6b3600ed1231a7c1467496b1fad9b3f52b5d2"
        or len(p["cohort"])!=12 or len(p["layouts"])!=4):
        raise ValueError("unexpected source baseline or topology count")
    if p["resource"]["gpu"] or p["resource"]["neural_training"]:
        raise ValueError("no GPU and no fitted model in topology data stage")
    layouts={x["id"]:x for x in p["layouts"]}
    if len(layouts)!=4 or len({json.dumps(x["blocked"]) for x in layouts.values()})!=4:
        raise ValueError("duplicate obstacle topology")
    ids={x["id"] for x in p["cohort"]}
    if len(ids)!=12 or len({x["seed"] for x in p["cohort"]})!=12:
        raise ValueError("non-independent original developer scenarios")
    for regime in p["balanced_regimes"]:
        group=[x for x in p["cohort"] if x["regime"]==regime]
        if len(group)!=4 or {x["layout"] for x in group}!=set(layouts):
            raise ValueError("not four layouts in each genuine stress family")
        for item in group:
            if item["split"]!="topology_development_only" or item["profile"]!="baseline":
                raise ValueError("cannot promote developer scene as blind test")
            if item["regime"]=="multi_fault" and item["station_id"]==item["secondary_station_id"]:
                raise ValueError("same station selected for distinct faults")
    return p


def build_topology_business(spec,stress,cfg):
    """True business simulator world modified ONLY in known STATIC grid."""
    business,_=build_quality_business(spec,stress)
    layout={x["id"]:x for x in cfg["layouts"]}[spec["layout"]]
    old=business.scenario.grid
    if not isinstance(old,Grid) or old.width!=7 or old.height!=5 or old.blocked:
        raise AssertionError("cannot silently switch original navigation dynamics")
    blocked=frozenset(tuple(x) for x in layout["blocked"])
    new=Grid(old.width,old.height,blocked)
    if len(blocked)!=len(layout["blocked"]):
        raise AssertionError("duplicated obstacle coordinates")
    points=[s.position for s in business.scenario.stations.values()]
    points += [r.position for r in business.scenario.robots]
    points += [r.home for r in business.scenario.robots]
    for p in points:
        if not new.contains(p):
            raise AssertionError("blocked an original public robot/station/home position")
    if any(new.shortest_path(points[0],p) is None for p in points):
        raise AssertionError("unreachable real business transport location")
    geometry=Geometry(new)
    if len(geometry.adjacency)!=old.width*old.height-len(blocked):
        raise AssertionError("topology public cell count differs from real solver")
    world=replace(business,scenario=replace(business.scenario,grid=new))
    return world,geometry


def public_topology_for_graph(grid,geometry,graph):
    """Attach explicit public navigable cell graph and entity↔cell links.

    All input facts are STATIC map and public entity positions. An encoded
    map exposes route detours explicitly; node coordinates alone do not.
    """
    if not isinstance(grid,Grid):
        raise TypeError("this preregistered experiment supports known Grid only")
    cells=sorted(geometry.adjacency,key=lambda p:(p[1],p[0]))
    index={p:i for i,p in enumerate(cells)}
    if len(cells)==0:
        raise AssertionError("no public walkable cell")
    edges=[]
    for a in cells:
        for b in geometry.adjacency[a]:
            if b not in index:
                raise AssertionError("Geometry has an inconsistent public adjacency")
            edges.append([index[a],index[b]])
    coords=[
        [p[0]/max(1,grid.width-1),p[1]/max(1,grid.height-1),
         len(geometry.adjacency[p])/4.]
        for p in cells
    ]
    entity_map=[]
    for entity_i,node in enumerate(graph["nodes"]):
        if node["kind"] in ("robot","station"):
            point=tuple(node["pos"])
            if point not in index:
                raise AssertionError("visible robot/station outside known public map")
            entity_map.append([entity_i,index[point]])
    stations=[n for n in graph["nodes"] if n["kind"]=="station"]
    dist=[]
    for a in stations:
        dist.append([geometry.distance(tuple(a["pos"]),tuple(b["pos"]))
                     for b in stations])
    maxdist=max(max(row) for row in dist)
    if maxdist>=geometry.unreachable:
        raise AssertionError("public stations disconnected")
    encoded={
        "schema":"pr56-public-grid-cells-v1",
        "width":grid.width,"height":grid.height,
        "blocked_cells":[list(x) for x in sorted(grid.blocked)],
        "walkable_cells":[list(x) for x in cells],
        "map_node_features":coords,
        "directed_map_edges":edges,
        "public_entity_to_map_cell_index":entity_map,
        "station_geodesic_distances_ticks":dist,
        "source":"original Geometry public static route graph and observed robot/station positions"
    }
    encoded["static_map_digest"]=hashlib.sha256(
        json.dumps({"size":[grid.width,grid.height],
                    "blocked":encoded["blocked_cells"],
                    "edges":edges},sort_keys=True).encode()).hexdigest()
    return encoded


def augment_public_transition(row,geometry):
    """No lookahead: time-t graph + static map sidecar, leave labels unchanged."""
    inp=row["input"]
    if inp["public_tick"]!=inp["feature_last_observed_tick"]:
        raise AssertionError("future t+1 in current public input")
    if "public_map" in inp:
        raise AssertionError("cannot silently overwrite public features")
    topo=public_topology_for_graph(geometry.grid,geometry,inp["graph"])
    inp["public_map"]=topo
    return row


def topology_pyg_dual_graph(row):
    """Explicit map-PyG and entity-PyG components; no new model is trained.

    Returns separate graph objects to avoid incorrect PyG auto-increment for
    mixed entity/map index namespaces. A future dual-graph model must test
    its own batching contract before using these inputs for gradient updates.
    """
    import torch
    from torch_geometric.data import Data
    inp=row["input"]
    if "public_map" not in inp:
        raise ValueError("public topology must be explicitly observed")
    topo=inp["public_map"]
    entity=candidate_graph(inp["graph"],0,True)
    map_x=torch.tensor(topo["map_node_features"],dtype=torch.float32)
    raw_edges=topo["directed_map_edges"]
    map_edge=torch.tensor(raw_edges,dtype=torch.long).T.contiguous()
    links=torch.tensor(topo["public_entity_to_map_cell_index"],
                       dtype=torch.long).T.contiguous()
    if (entity.x.ndim!=2 or entity.x.shape[1]!=16
        or map_x.ndim!=2 or map_x.shape[1]!=3
        or map_edge.shape[0]!=2 or links.shape[0]!=2):
        raise AssertionError("dual-graph public tensor boundary malformed")
    return {
        "entity":entity,
        "map":Data(x=map_x,edge_index=map_edge),
        "entity_to_map":links,
        "static_map_digest":topo["static_map_digest"]
    }
