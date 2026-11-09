"""CP-SAT adapter for current assignments, configured workflows and capacities.

The model knows only current observable tasks and their public workflow
schemas. Each selected current task is planned through its remaining stages
using the same robot as a finite planning assumption; subsequent real stages
are dispatched again. The simulation remains the path/safety authority.
"""
from __future__ import annotations

import math
import time

from lab_transport.paper_reuse.ortools_dispatch import CPSATDispatcher,cp_model


class BusinessCPSAT(CPSATDispatcher):
    def __init__(self,geometry,rules=None,**kwargs):
        super().__init__(geometry,**kwargs)
        self.rules=rules

    def proposals(self,observation,candidates):
        # Exact previous behavior for callers without business rules.
        if self.rules is None:return super().proposals(observation,candidates)
        start_clock=time.perf_counter();cpu_start=time.process_time();self.calls+=1
        items=self._items(observation,candidates)
        fallback=((items[0][0],),) if items else ((),)
        if not items or cp_model is None:
            reason='no_reachable_candidate' if not items else 'ortools_unavailable'
            self.fallbacks+=1
            self.log.append({'tick':observation.tick,'fallback':reason,'statuses':[],
                'solutions':0,'partial_truncation':False,'wall_seconds':time.perf_counter()-start_clock,
                'cpu_seconds':time.process_time()-cpu_start})
            return fallback
        robots={r.id:r for r in observation.robots}
        samples={s.id:s for s in observation.samples}
        stations={s.id:s for s in observation.stations}
        resources=self.rules.station_resources
        candidates_ready=[]
        for action,_,_ in items:
            sample=samples[action.sample_id]
            wf=getattr(sample,'workflow','')
            idx=getattr(sample,'stage_index',0)
            if wf:
                stages=self.rules.workflows.get(wf)
                if stages is None or not 0<=idx<len(stages) or stages[idx]!=(sample.pickup,sample.dropoff):
                    raise ValueError('observed workflow state differs from public workflow schema')
                route=stages[idx:]
            else:
                route=((sample.pickup,sample.dropoff),)
            if any(self.geometry.distance(stations[a].position,stations[b].position)>=self.geometry.unreachable
                   for a,b in route):
                continue
            candidates_ready.append((action,route))
        if not candidates_ready:
            self.fallbacks+=1
            self.log.append({'tick':observation.tick,'fallback':'no_reachable_workflow','statuses':[],
                'solutions':0,'partial_truncation':False,'wall_seconds':time.perf_counter()-start_clock,
                'cpu_seconds':time.process_time()-cpu_start})
            return ((),)
        # A loose finite bound allows queueing at every resource without
        # pretending that unknown outages have known end times.
        longest=0
        for action,route in candidates_ready:
            distance=self.geometry.distance(robots[action.robot_id].position,stations[route[0][0]].position)
            for pickup,dropoff in route:
                distance+=self.geometry.distance(stations[pickup].position,stations[dropoff].position)
                distance+=math.ceil(stations[pickup].expected_service_ticks)+math.ceil(stations[dropoff].expected_service_ticks)+2
            longest=max(longest,distance)
        horizon=max(20,(longest+10)*(len(robots)+2))
        model=cp_model.CpModel();flags=[];ends=[];by_robot={};by_sample={};robot_intervals={}
        resource_intervals={r.id:[] for r in self.rules.resources}
        resource_demands={r.id:[] for r in self.rules.resources}
        # Reserve a conservative estimate for currently observed services;
        # no private service_remaining or future fault plan is consulted.
        busy=[]
        for robot in observation.robots:
            if robot.task_id not in samples or robot.service_elapsed is None:continue
            sample=samples[robot.task_id]
            sid=sample.pickup if robot.phase=='to_pickup' else sample.dropoff
            station=stations[sid]
            if robot.position!=station.position:continue
            observed=getattr(robot,'service_active_elapsed',None)
            elapsed=observed if observed is not None else 0
            prior=getattr(station,'observed_effective_service_mean',None)
            if prior is None:prior=station.expected_service_ticks
            duration=max(1,math.ceil(prior)-elapsed)+int(not getattr(station,'device_available',station.available))
            rid=resources[sid].id
            interval=model.NewIntervalVar(0,duration,duration,'occupied_'+robot.id)
            resource_intervals[rid].append(interval);resource_demands[rid].append(1)
            busy.append({'robot_id':robot.id,'station_id':sid,'resource_id':rid,'estimated_ticks':duration})
        for index,(action,route) in enumerate(candidates_ready):
            present=model.NewBoolVar(f'assign_{index}');flags.append(present)
            by_robot.setdefault(action.robot_id,[]).append(present)
            by_sample.setdefault(action.sample_id,[]).append(present)
            prev_end=None;previous_position=robots[action.robot_id].position
            records=[]
            for j,(pickup,dropoff) in enumerate(route):
                for kind,sid in [('pickup',pickup),('dropoff',dropoff)]:
                    st=stations[sid];group=resources[sid]
                    travel=self.geometry.distance(previous_position,st.position)
                    if travel>=self.geometry.unreachable:raise ValueError('workflow route became unreachable')
                    dur=max(1,math.ceil(st.expected_service_ticks))
                    sv=model.NewIntVar(0,horizon,f'start_{index}_{j}_{kind}')
                    ev=model.NewIntVar(0,horizon,f'end_{index}_{j}_{kind}')
                    iv=model.NewOptionalIntervalVar(sv,dur,ev,present,f'service_{index}_{j}_{kind}')
                    if prev_end is None:model.Add(sv>=travel).OnlyEnforceIf(present)
                    else:model.Add(sv>=prev_end+travel).OnlyEnforceIf(present)
                    if not getattr(st,'device_available',st.available):model.Add(sv>=1).OnlyEnforceIf(present)
                    resource_intervals[group.id].append(iv);resource_demands[group.id].append(1)
                    records.append((j,kind,sid,group.id,sv,ev))
                    prev_end=ev;previous_position=st.position
            model.Add(prev_end==0).OnlyEnforceIf(present.Not())
            total=model.NewIntVar(0,horizon,f'robot_span_{index}')
            model.Add(total==prev_end).OnlyEnforceIf(present)
            model.Add(total==0).OnlyEnforceIf(present.Not())
            whole=model.NewOptionalIntervalVar(0,total,prev_end,present,f'robot_interval_{index}')
            robot_intervals.setdefault(action.robot_id,[]).append(whole)
            ends.append(prev_end)
            candidates_ready[index]=(action,route,records)
        for values in by_robot.values():model.Add(sum(values)<=1)
        for values in by_sample.values():model.Add(sum(values)<=1)
        for values in robot_intervals.values():model.AddNoOverlap(values)
        for resource in self.rules.resources:
            if resource_intervals[resource.id]:
                model.AddCumulative(resource_intervals[resource.id],resource_demands[resource.id],resource.capacity)
        makespan=model.NewIntVar(0,horizon,'makespan');model.AddMaxEquality(makespan,ends)
        maximum=min(len(by_robot),len(by_sample))
        model.Minimize((horizon+1)*(maximum-sum(flags))+makespan)
        solver=cp_model.CpSolver();solver.parameters.max_time_in_seconds=self.time_limit_seconds
        solver.parameters.max_deterministic_time=self.deterministic_time_limit
        solver.parameters.num_search_workers=1;solver.parameters.random_seed=self.seed
        plans=[];schedules=[];statuses=[];fixed_count=None
        for _ in range(self.candidate_limit):
            status=solver.Solve(model);statuses.append(solver.StatusName(status))
            if status not in (cp_model.OPTIMAL,cp_model.FEASIBLE):break
            chosen=[i for i,b in enumerate(flags) if solver.Value(b)]
            if not chosen:break
            if fixed_count is None:
                fixed_count=len(chosen);model.Add(sum(flags)==fixed_count)
            plans.append(tuple(candidates_ready[i][0] for i in chosen))
            schedules.append([{'robot_id':candidates_ready[i][0].robot_id,
                'sample_id':candidates_ready[i][0].sample_id,'stage_offset':j,'operation':kind,
                'station_id':sid,'resource_id':rid,'start':solver.Value(sv),'end':solver.Value(ev)}
                for i in chosen for j,kind,sid,rid,sv,ev in candidates_ready[i][2]])
            model.Add(sum(flags[i] for i in chosen)<=len(chosen)-1)
        if not plans:
            self.fallbacks+=1;reason='solver_'+(statuses[-1].lower() if statuses else 'unknown');plans=list(fallback)
        else:reason=None
        truncated=any(s in ('UNKNOWN','FEASIBLE') for s in statuses)
        self.truncated+=int(truncated)
        self.log.append({'tick':observation.tick,'fallback':reason,'statuses':statuses,'partial_truncation':truncated,
            'solutions':len(plans),'schedules':schedules,'observed_busy_estimates':busy,
            'wall_seconds':time.perf_counter()-start_clock,'cpu_seconds':time.process_time()-cpu_start})
        return tuple(plans)
