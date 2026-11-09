"""Call the pinned official LaCAM3 full-path solver on observable MAPF input."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import resource
import subprocess
import time

from lab_transport.dynamics_model import Geometry


@dataclass(frozen=True)
class JointPlanResult:
    status: str
    path: tuple[tuple[tuple[int,int],...],...]
    ids: tuple[str,...]
    primary: str | None
    duplicate_business_goals: bool
    temporary_goals: bool
    wall_ms: float
    peak_rss_kb: int | None
    detail: str


class LaCAM3Planner:
    """One bounded process per complete joint plan; no local joint search."""
    def __init__(self,geometry: Geometry,worker,*,budget_ms=500,seed=20261109):
        self.geometry=geometry;self.worker=Path(worker)
        if not self.worker.is_file():raise FileNotFoundError('LaCAM3 worker is not built')
        if budget_ms!=500:raise ValueError('this experiment freezes the 500 ms budget')
        self.budget_ms=budget_ms;self.seed=seed
        self.directed_edges={(a,b) for a,values in geometry.adjacency.items() for b in values}
        self.bidirectional=all((b,a) in self.directed_edges for a,b in self.directed_edges)

    def plan(self,positions,goals,locked,priority):
        start=time.perf_counter();ids=tuple(sorted(positions));locked=frozenset(locked)
        duplicate=any(v>1 for v in Counter(goals.values()).values())
        def finish(status,path=(),detail='',rss=None,temporary=False):
            return JointPlanResult(status,tuple(path),ids,priority,duplicate,temporary,
                (time.perf_counter()-start)*1000,rss,detail)
        if set(positions)!=set(goals) or not locked<=set(ids):
            raise ValueError('positions, goals and locked identities differ')
        if len(set(positions.values()))!=len(positions):raise ValueError('robots already share a vertex')
        if any(p not in self.geometry.adjacency for p in (*positions.values(),*goals.values())):
            raise ValueError('position or goal outside the current topology')
        if not self.bidirectional:return finish('unsupported_directed',detail='static graph contains one-way arcs')
        active=tuple(r for r in ids if r not in locked)
        if not active:return finish('no_movable_robot')
        if priority not in active:return finish('no_priority_robot')
        blocked={positions[r] for r in locked}
        nodes=tuple(sorted(set(self.geometry.adjacency)-blocked));index={p:i for i,p in enumerate(nodes)}
        effective={r:(goals[r] if not duplicate or r==priority else positions[r]) for r in active}
        if any(g in blocked for g in effective.values()):
            return finish('locked_goal',detail='a required goal is occupied by a locked robot',temporary=duplicate)
        if len(set(effective.values()))!=len(effective):
            return finish('temporary_goal_conflict',detail='fixed temporary goals still overlap',temporary=duplicate)
        arcs=sorted((index[a],index[b]) for a,b in self.directed_edges if a in index and b in index)
        # Geometry.shortest_path supplies a necessary single-agent connectivity
        # check; it is not a replacement for the official joint solver.
        for rid in active:
            if self.geometry.shortest_path(positions[rid],effective[rid],blocked) is None:
                return finish('unreachable_goal',detail='goal unreachable with currently locked vertices',temporary=duplicate)
        remaining=self.budget_ms-(time.perf_counter()-start)*1000
        if remaining<=0:return finish('timeout',detail='mapping exhausted the wall budget',temporary=duplicate)
        lines=[f'{len(nodes)} {len(arcs)} {len(active)} {self.seed} {max(1,int(remaining))}']
        lines.extend(f'{a} {b}' for a,b in arcs)
        lines.extend(f'{index[positions[r]} {index[effective[r]]}' for r in active)
        try:
            completed=subprocess.run([str(self.worker.resolve())],input='\n'.join(lines)+'\n',
                 text=True,capture_output=True,timeout=remaining/1000,check=False)
        except subprocess.TimeoutExpired:
            return finish('timeout',detail='hard 500 ms wall limit',temporary=duplicate)
        if completed.returncode:
            return finish('worker_error',detail=f'exit {completed.returncode}: {completed.stderr[:300]}',temporary=duplicate)
        output=completed.stdout.splitlines()
        if not output:return finish('worker_error',detail='empty worker output',temporary=duplicate)
        header=output[0].split()
        if header[0]=='NO_SOLUTION':
            return finish('no_solution',detail='official solve returned an empty plan',
                          rss=int(header[1]) if len(header)>1 else None,temporary=duplicate)
        if len(header)!=4 or header[0]!='OK':return finish('worker_error',detail=output[0][:200],temporary=duplicate)
        count,n,rss=map(int,header[1:])
        if n!=len(active) or count!=len(output)-1 or count<1:
            return finish('worker_error',detail='invalid complete-path dimensions',rss=rss,temporary=duplicate)
        path=[]
        for line in output[1:]:
            values=[int(v) for v in line.split()]
            if len(values)!=len(active) or any(v<0 or v>=len(nodes) for v in values):
                return finish('worker_error',detail='invalid path vertex',rss=rss,temporary=duplicate)
            state=dict(positions)
            for rid,node_id in zip(active,values):state[rid]=nodes[node_id]
            path.append(tuple(state[r] for r in ids))
        expected=tuple(positions[r] for r in ids)
        if path[0]!=expected:return finish('invalid_path',detail='initial state mismatch',rss=rss,temporary=duplicate)
        for a,b in zip(path,path[1:]):
            self.validate(ids,a,b,locked)
        if any(path[-1][ids.index(r)]!=effective[r] for r in active):
            return finish('invalid_path',detail='final temporary or business goal mismatch',rss=rss,temporary=duplicate)
        return finish('solved',path,rss=rss,temporary=duplicate)

    def validate(self,ids,before,after,locked):
        if len(after)!=len(ids) or len(set(after))!=len(after):raise ValueError('vertex collision')
        for i,rid in enumerate(ids):
            if rid in locked and before[i]!=after[i]:raise ValueError('locked robot moved')
            if before[i]!=after[i] and after[i] not in self.geometry.adjacency[before[i]]:
                raise ValueError('illegal directed edge')
        for i in range(len(ids)):
            for j in range(i+1,len(ids)):
                if before[i]==after[j] and before[j]==after[i] and before[i]!=before[j]:
                    raise ValueError('edge swap')
