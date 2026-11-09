"""Bounded, copyable livelock detection and complete joint-path cache."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RecoverySnapshot:
    completed_count: int | None = None
    no_stage_steps: int = 0
    recent_positions: tuple = ()
    cached_path: tuple = ()
    cached_key: tuple = ()
    attempted_key: tuple = ()


def priority_for(robots):
    """Carrying robot first, then currently visible release time and ID."""
    eligible=[r for r in robots if not r['locked'] and r['task_id'] is not None]
    if not eligible:return None
    return min(eligible,key=lambda r:(r['phase']!='to_dropoff',
               r['release_tick'] if r['release_tick'] is not None else float('inf'),r['id']))['id']


class RecoveryCoordinator:
    """Pure snapshot-in/snapshot-out controller; branches never share a cache."""
    def __init__(self,planner,pibt):
        self.planner=planner;self.pibt=pibt

    def step(self,snapshot,robots,stations,completed_count,normal_service,tick):
        ids=tuple(sorted(r['id'] for r in robots));by_id={r['id']:r for r in robots}
        positions={rid:by_id[rid]['position'] for rid in ids}
        goals={rid:by_id[rid]['goal'] for rid in ids}
        locked={rid for rid in ids if by_id[rid]['locked']}
        current=tuple(positions[rid] for rid in ids)
        task_key=tuple((rid,by_id[rid]['task_id'],by_id[rid]['phase'],goals[rid]) for rid in ids)
        attempt_key=(task_key,tuple(sorted(locked)))
        cache_key=(attempt_key,tuple(sorted(stations)))
        attempted=snapshot.attempted_key
        if attempted and attempted!=attempt_key:attempted=()
        cache=snapshot.cached_path;old_key=snapshot.cached_key
        event={'tick':tick,'triggered':False,'status':'pibt','cache_used':False,
               'cache_invalidated':False,'invalidation_reason':None,'plan_steps':0,
               'plan':None,'duplicate_business_goals':False,'temporary_goals':False,
               'planning_wall_ms':None,'planning_cpu_ms':None,'planning_peak_rss_kb':None}
        if cache:
            if old_key!=cache_key or cache[0]!=current:
                event['cache_invalidated']=True
                event['invalidation_reason']='context_changed' if old_key!=cache_key else 'position_mismatch'
                cache=();old_key=()
        completed_changed=(snapshot.completed_count is not None and completed_count>snapshot.completed_count)
        no_stage=0 if completed_changed else snapshot.no_stage_steps+1
        recent=() if completed_changed or normal_service else snapshot.recent_positions
        repeated=current in recent
        recent=(recent+(current,))[-16:]
        if normal_service:recent=()
        if cache and len(cache)>1:
            try:
                self.planner.validate(ids,cache[0],cache[1],locked)
            except ValueError:
                event['cache_invalidated']=True;event['invalidation_reason']='cached_step_unsafe'
                cache=();old_key=()
            else:
                nxt=cache[1];cache=cache[1:]
                if len(cache)<=1:cache=();old_key=()
                event['cache_used']=True;event['status']='cached_plan'
                return RecoverySnapshot(completed_count,no_stage,recent,cache,old_key,attempted),dict(zip(ids,nxt)),event
        has_task=any(by_id[rid]['task_id'] is not None for rid in ids)
        if no_stage>=16 and repeated and not normal_service and has_task and attempted!=attempt_key:
            attempted=attempt_key;event['triggered']=True
            result=self.planner.plan(positions,goals,locked,priority_for(robots))
            event.update({'status':result.status,'duplicate_business_goals':result.duplicate_business_goals,
                'temporary_goals':result.temporary_goals,'planning_wall_ms':result.wall_ms,
                'planning_cpu_ms':result.cpu_ms,'planning_peak_rss_kb':result.peak_rss_kb,
                'worker_invoked':result.worker_invoked,
                'detail':result.detail,'primary':result.primary,'plan_steps':max(0,len(result.path)-1)})
            if result.status=='solved' and len(result.path)>1:
                cache=result.path;old_key=cache_key;event['plan']=cache
                self.planner.validate(ids,cache[0],cache[1],locked)
                nxt=cache[1];cache=cache[1:]
                if len(cache)<=1:cache=();old_key=()
                event['cache_used']=True
                return RecoverySnapshot(completed_count,no_stage,recent,cache,old_key,attempted),dict(zip(ids,nxt)),event
        updated=self.pibt.step(positions,goals,locked,tick)
        return RecoverySnapshot(completed_count,no_stage,recent,cache,old_key,attempted),updated,event
