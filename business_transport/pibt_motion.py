"""One-step bridge to the pinned author's C++ PIBT implementation.

The official solver is invoked afresh for each step; its private priority ages
therefore reset. This is a receding one-step use of PIBT and does not inherit
the original multi-step liveness theorem. No PIBT recursion is reimplemented.
"""
from __future__ import annotations

from collections import OrderedDict
import ctypes
from dataclasses import replace
from pathlib import Path
import threading

from lab_transport.domain import DirectedGraph,Grid
from lab_transport.dynamics_model import Geometry

_CALL_LOCK=threading.Lock()


class PIBTMotion:
    def __init__(self,geometry: Geometry,library,*,planner_seed=20261109,cache_size=64):
        self.geometry=geometry
        path=Path(library)
        if not path.is_file():raise FileNotFoundError('PIBT bridge library is not built: '+str(path))
        self.library=ctypes.CDLL(str(path.resolve()))
        intp=ctypes.POINTER(ctypes.c_int)
        self.call=self.library.wm_pibt_step
        self.call.argtypes=[ctypes.c_int,ctypes.c_int,intp,intp,intp,ctypes.c_int,
                            intp,intp,intp,ctypes.c_uint,intp]
        self.call.restype=ctypes.c_int
        self.planner_seed=planner_seed;self.cache_size=cache_size;self.cache=OrderedDict()
        self.calls=0;self.solver_calls=0;self.unreachable_goals=0;self.locked_goals=0

    def _topology(self,locked):
        key=frozenset(locked)
        if key in self.cache:
            self.cache.move_to_end(key);return self.cache[key]
        source=self.geometry.grid
        if isinstance(source,Grid):
            grid=replace(source,blocked=source.blocked|key)
        elif isinstance(source,DirectedGraph):
            coordinates={p:xy for p,xy in source.coordinates.items() if p not in key}
            edges=tuple((a,b) for a,b in source.edges if a not in key and b not in key)
            grid=DirectedGraph(coordinates,edges)
        else:raise ValueError('PIBT adapter does not support this topology type')
        geometry=Geometry(grid)
        nodes=tuple(sorted(geometry.adjacency));index={p:i for i,p in enumerate(nodes)}
        offsets=[0];edges=[]
        for node in nodes:
            edges.extend(index[p] for p in geometry.adjacency[node]);offsets.append(len(edges))
        entry=(geometry,nodes,index,offsets,edges)
        self.cache[key]=entry
        if len(self.cache)>self.cache_size:self.cache.popitem(last=False)
        return entry

    def step(self,positions,goals,locked,tick):
        """Return one simultaneous legal step from only current public input."""
        self.calls+=1
        if set(positions)!=set(goals) or not set(locked)<=set(positions):
            raise ValueError('positions, goals or locked robot ids do not agree')
        if len(set(positions.values()))!=len(positions):raise ValueError('robots already share a node')
        if any(p not in self.geometry.adjacency for p in positions.values()):
            raise ValueError('robot outside the static topology')
        if any(g not in self.geometry.adjacency for g in goals.values()):
            raise ValueError('target outside the static topology')
        locked=set(locked);blocked={positions[rid] for rid in locked}
        active=sorted(set(positions)-locked)
        if not active:return dict(positions)
        geometry,nodes,index,offsets,edges=self._topology(blocked)
        starts=[index[positions[rid]] for rid in active];effective=[];distances=[]
        for rid in active:
            goal=goals[rid]
            if goal in blocked:
                self.locked_goals+=1;goal=positions[rid]
            elif geometry.distance(positions[rid],goal)>=geometry.unreachable:
                self.unreachable_goals+=1;goal=positions[rid]
            effective.append(index[goal])
            distances.extend(geometry.distance(p,goal) for p in nodes)
        def ints(values):return (ctypes.c_int*len(values))(*values)
        xy=ints([v for p in nodes for v in p]);off=ints(offsets);arc=ints(edges or [0])
        start=ints(starts);goal=ints(effective);dist=ints(distances);out=(ctypes.c_int*len(active))()
        with _CALL_LOCK:
            code=self.call(len(nodes),len(edges),xy,off,arc,len(active),start,goal,dist,
                           ctypes.c_uint((self.planner_seed+1009*tick)&0xffffffff),out)
        self.solver_calls+=1
        if code:raise RuntimeError(f'official PIBT bridge rejected step: {code}')
        result=dict(positions)
        for i,rid in enumerate(active):
            if not 0<=out[i]<len(nodes):raise RuntimeError('PIBT returned an invalid node')
            result[rid]=nodes[out[i]]
        self.validate(positions,result,locked)
        return result

    def validate(self,before,after,locked):
        if set(before)!=set(after) or len(set(after.values()))!=len(after):
            raise ValueError('PIBT produced a vertex collision')
        for rid,pos in before.items():
            nxt=after[rid]
            if (rid in locked and nxt!=pos) or (nxt!=pos and nxt not in self.geometry.adjacency[pos]):
                raise ValueError('PIBT produced a locked or non-edge movement')
        ids=sorted(before)
        if any(before[a]!=before[b] and before[a]==after[b] and before[b]==after[a]
               for i,a in enumerate(ids) for b in ids[i+1:]):
            raise ValueError('PIBT produced an edge swap')
