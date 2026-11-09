"""Optional-interval CP-SAT assignment using only current observable inputs.

Each current pickup/delivery has two ordered operations on the same robot.
No station capacity or future job/outage is invented. The simulator remains
responsible for actual path conflict and lifecycle validation.
"""
from __future__ import annotations

import math
import time

from lab_transport.transition_schema import Assignment

try:
    from ortools.sat.python import cp_model
except ImportError:
    cp_model = None


class CPSATDispatcher:
    def __init__(self, geometry, *, time_limit_seconds=0.5, candidate_limit=4, seed=0,
                 deterministic_time_limit=0.05):
        if not 0 < time_limit_seconds <= 1 or not 1 <= candidate_limit <= 20:
            raise ValueError('invalid finite solver budget')
        if not 0 < deterministic_time_limit <= 1:
            raise ValueError('invalid deterministic solver budget')
        self.geometry = geometry
        self.time_limit_seconds = time_limit_seconds
        self.candidate_limit = candidate_limit
        self.seed = seed
        self.deterministic_time_limit = deterministic_time_limit
        self.calls = self.fallbacks = self.truncated = 0
        self.log = []

    def _items(self, observation, candidates):
        robots = {r.id:r for r in observation.robots}
        samples = {s.id:s for s in observation.samples}
        stations = {s.id:s for s in observation.stations}
        items = []
        for action in sorted(set(candidates),key=lambda a:(a.robot_id,a.sample_id)):
            if not isinstance(action, Assignment):
                continue
            robot, sample = robots.get(action.robot_id), samples.get(action.sample_id)
            if (robot is None or sample is None or robot.task_id is not None or
                    sample.status != 'waiting' or sample.release_tick > observation.tick or
                    sample.pickup not in stations or sample.dropoff not in stations):
                continue
            pickup, dropoff = stations[sample.pickup], stations[sample.dropoff]
            try:
                d1 = self.geometry.distance(robot.position, pickup.position)
                d2 = self.geometry.distance(pickup.position, dropoff.position)
            except ValueError:
                continue
            if max(d1,d2) >= self.geometry.unreachable:
                continue
            duration1 = max(1, d1 + math.ceil(max(0.,pickup.expected_service_ticks)))
            duration2 = max(1, d2 + math.ceil(max(0.,dropoff.expected_service_ticks)))
            items.append((action,duration1,duration2))
        return items

    def proposals(self, observation, candidates):
        """Return bounded distinct legal matchings, best CP-SAT objective first."""
        started = time.process_time()
        self.calls += 1
        items = self._items(observation,candidates)
        fallback = ((items[0][0],),) if items else ((),)
        if not items or cp_model is None:
            reason = 'no_reachable_candidate' if not items else 'ortools_unavailable'
            self.fallbacks += 1
            self.log.append({'tick':observation.tick,'fallback':reason,'solutions':0,
                             'cpu_seconds':time.process_time()-started})
            return fallback
        model = cp_model.CpModel()
        # At most one currently unassigned job is started on each idle robot.
        horizon = max(a+b for _,a,b in items)
        horizon = max(horizon,1)
        flags, end_vars = [], []
        by_robot, by_sample, intervals = {}, {}, {}
        for i,(action,pick,drop) in enumerate(items):
            selected = model.NewBoolVar(f'select_{i}')
            start_pick = model.NewIntVar(0,0,f'pick_start_{i}')
            end_pick = model.NewIntVar(0,horizon,f'pick_end_{i}')
            start_drop = model.NewIntVar(0,horizon,f'drop_start_{i}')
            end_drop = model.NewIntVar(0,horizon,f'drop_end_{i}')
            op_pick = model.NewOptionalIntervalVar(start_pick,pick,end_pick,selected,f'pickup_{i}')
            op_drop = model.NewOptionalIntervalVar(start_drop,drop,end_drop,selected,f'delivery_{i}')
            model.Add(start_drop == end_pick).OnlyEnforceIf(selected)
            model.Add(end_drop == 0).OnlyEnforceIf(selected.Not())
            flags.append(selected)
            end_vars.append(end_drop)
            by_robot.setdefault(action.robot_id,[]).append(selected)
            by_sample.setdefault(action.sample_id,[]).append(selected)
            intervals.setdefault(action.robot_id,[]).extend((op_pick,op_drop))
        for group in by_robot.values():
            model.Add(sum(group) <= 1)
        for group in by_sample.values():
            model.Add(sum(group) <= 1)
        for group in intervals.values():
            model.AddNoOverlap(group)
        makespan = model.NewIntVar(0,horizon,'makespan')
        model.AddMaxEquality(makespan,end_vars)
        max_assignments = min(len(by_robot),len(by_sample))
        # One extra assignment always dominates any makespan difference.
        model.Minimize((horizon + 1)*(max_assignments-sum(flags))+makespan)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit_seconds
        solver.parameters.max_deterministic_time = self.deterministic_time_limit
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = self.seed
        plans, statuses = [], []
        fixed_count = None
        for _ in range(self.candidate_limit):
            status = solver.Solve(model)
            statuses.append(solver.StatusName(status))
            if status not in (cp_model.OPTIMAL,cp_model.FEASIBLE):
                break
            selected = [i for i,x in enumerate(flags) if solver.Value(x)]
            if not selected:
                break
            if fixed_count is None:
                fixed_count = len(selected)
                model.Add(sum(flags) == fixed_count)
            plans.append(tuple(items[i][0] for i in selected))
            model.Add(sum(flags[i] for i in selected) <= len(selected)-1)
        if not plans:
            self.fallbacks += 1
            reason = 'solver_' + (statuses[-1].lower() if statuses else 'unknown')
            plans = list(fallback)
        else:
            reason = None
        truncated = any(status in ('UNKNOWN','FEASIBLE') for status in statuses)
        self.truncated += int(truncated)
        self.log.append({'tick':observation.tick,'fallback':reason,'statuses':statuses,
                         'partial_truncation':truncated,'solutions':len(plans),
                         'cpu_seconds':time.process_time()-started})
        return tuple(plans)

    def __call__(self, observation, candidates):
        return self.proposals(observation,candidates)[0]
