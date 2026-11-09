"""Offline topology diagnostics; never alters the executed directed graph."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import networkx as nx
from experiments.business_constraints.benchmark import make_case,read_config
from lab_transport.dynamics_model import Geometry


def analyze(name,adjacency):
    edges={(a,b) for a,values in adjacency.items() for b in values}
    graph=nx.Graph();graph.add_nodes_from(adjacency);graph.add_edges_from(edges)
    one_way=sum((b,a) not in edges for a,b in edges)
    return {'layout':name,'nodes':len(graph),'directed_edges':len(edges),'one_way_edges':one_way,
      'weakly_connected':nx.is_connected(graph),'undirected_projection_bridge_count':len(list(nx.bridges(graph))),
      'undirected_projection_articulation_count':len(list(nx.articulation_points(graph))),
      'note':'Undirected projection is for offline bridge counting only; execution retains original directed arcs.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--hospital',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('refuse to overwrite topology audit')
    hospital=json.loads(a.hospital.read_text())['cases'];rows=[]
    for case in hospital:
        adjacency={}
        for left,right in case['topology']['edges']:
            left,right=tuple(left),tuple(right);adjacency.setdefault(left,[]).append(right);adjacency.setdefault(right,[])
        rows.append(analyze(case['layout'],adjacency))
    config=read_config()
    for name,seed in (('empty8',67000),('random32',68000)):
        business,_=make_case(config,name,seed,'workflow_serial')
        rows.append(analyze(name,Geometry(business.scenario.grid).adjacency))
    a.output.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(rows,ensure_ascii=False))


if __name__=='__main__':main()
