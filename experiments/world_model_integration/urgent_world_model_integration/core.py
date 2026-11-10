"""Experimental frozen V4 vs Rule integration on synthetic urgent/retest workflows.

No training, hidden future event, changed production scheduler, or test tuning.
The simulator remains authoritative. Retest branches are unknowable before
detection; a rollout uses the currently public default route until disclosure.
"""
from __future__ import annotations
from dataclasses import asdict
from collections import defaultdict
import math
import time

from business_transport.dispatch import BusinessCPSAT
from business_transport.dynamics import WorkflowDynamics
from business_transport.session_dynamics import SessionRuleDynamics
from business_transport.workflow_transition import check_business_transition
from lab_transport.torch_structured_adapter import ProgressAwareRule
from lab_transport.decision_value import candidates_for, nearest_batch
from lab_transport.multisite_experiment import discrete_safety_violations
from experiments.business_constraints.run import lifecycle_audit
from business_transport.simulation import resource_audit
from experiments.world_model_integration.urgent_workflow_feasibility.core import (
    ConditionalRetestSimulator, PriorityObserver, scene_for, public_spec,
    unique_plans, job_metrics, digest, synthetic_retest,
)


def _route_remaining(state, sample, rules, geometry):
    """A shared observable, optimistic terminal-work estimator.

    Queueing and future retest are not foreknown. The estimate is not a
    calibrated completion-time probability or an oracle.
    """
    stations={s.id:s for s in state.stations}
    robots={r.id:r for r in state.robots}
    route=rules.workflows[sample.workflow]
    index=sample.stage_index
    if not 0<=index<len(route):
        raise ValueError("current workflow stage index outside public route")
    if sample.status=="completed":
        return 0.0 if index==len(route)-1 else _future_work(
            route[index+1:],stations,robots,geometry,None)
    robot=robots.get(sample.robot_id) if sample.robot_id else None
    pos=robot.position if robot else None
    total=0.0
    for offset,(pickup,dropoff) in enumerate(route[index:]):
        first=stations[pickup];last=stations[dropoff]
        if offset==0 and sample.status=="in_transit" and robot is not None:
            total+=geometry.distance(pos,last.position)
            active=(robot.phase=="to_dropoff" and
                    pos==last.position and robot.service_active_elapsed is not None)
            total+=max(0.,float(last.expected_service_ticks)-
                       (robot.service_active_elapsed if active else 0))
            pos=last.position
            continue
        if pos is None:
            total+=min(geometry.distance(r.position,first.position) for r in robots.values())
        else:
            total+=geometry.distance(pos,first.position)
        total+=geometry.distance(first.position,last.position)
        active=(offset==0 and robot is not None and
                robot.phase=="to_pickup" and pos==first.position and
                robot.service_active_elapsed is not None)
        total+=max(0.,float(first.expected_service_ticks)-
                   (robot.service_active_elapsed if active else 0))
        total+=float(last.expected_service_ticks)
        pos=last.position
    return total


def _future_work(route,stations,robots,geometry,pos):
    total=0.
    for pickup,dropoff in route:
        first,last=stations[pickup],stations[dropoff]
        if pos is None:
            total+=min(geometry.distance(r.position,first.position) for r in robots.values())
        else:
            total+=geometry.distance(pos,first.position)
        total+=geometry.distance(first.position,last.position)
        total+=float(first.expected_service_ticks)+float(last.expected_service_ticks)
        pos=last.position
    return total


def _projected_job_cost(initial,trace,rules,geometry):
    """Job-level expected-time proxy shared by both predictors.

    Each particle's sampled world state is followed by the same deterministic
    residual estimator. Sampling noise is paired across candidates by seed.
    """
    final=trace[-1]
    initial_jobs={}
    for sample in initial.samples:
        if sample.job_id:
            initial_jobs.setdefault(sample.job_id,sample)
    final_groups=defaultdict(list)
    for sample in final.samples:
        if sample.job_id:
            final_groups[sample.job_id].append(sample)
    urgent_late=routine_late=urgent_eta=routine_eta=0.
    finished=0
    for job,original in initial_jobs.items():
        group=final_groups[job]
        if not group:
            raise ValueError("known job disappeared from predicted state")
        route=rules.workflows[original.workflow]
        fully_done=any(s.stage_index==len(route)-1 and s.status=="completed" for s in group)
        if fully_done:
            # Find first predicted completion tick, not simply horizon end.
            end_tick=final.tick
            for state in trace:
                if any(s.job_id==job and s.stage_index==len(route)-1 and
                       s.status=="completed" for s in state.samples):
                    end_tick=state.tick
                    break
            finished+=1
        else:
            active=[s for s in group if s.status!="completed"]
            if len(active)>1:
                raise ValueError("more than one active stage in predicted job")
            if active:
                current=active[0]
            else:
                current=max(group,key=lambda s:s.stage_index)
            end_tick=final.tick+_route_remaining(final,current,rules,geometry)
        due=original.deadline_tick
        if due is None:
            raise ValueError("synthetic urgency deadline missing")
        lateness=max(0.,end_tick-due)
        remaining=max(0.,end_tick-initial.tick)
        if original.priority==2:
            urgent_late+=lateness
            urgent_eta+=remaining
        else:
            routine_late+=lateness
            routine_eta+=remaining
    return (urgent_late,urgent_eta,routine_late,routine_eta),finished


class UrgentWorldPolicy:
    def __init__(self,arm,geometry,rules,config,checkpoint,*,seed,steps=5,particles=3):
        if arm not in ("P4","P5"):
            raise ValueError("expected P4 Rule or P5 frozen V4")
        if steps!=5 or particles!=3:
            raise ValueError("first adaptation uses fixed H5/P3 rollout budget")
        self.arm,self.geometry,self.config=arm,geometry,config
        self.seed,self.steps,self.particles=seed,steps,particles
        self.observer=PriorityObserver()
        model=(ProgressAwareRule(geometry) if arm=="P4" else
               SessionRuleDynamics(geometry,rules,checkpoint,experimental=True,
                                   seed=1810,feature_version="v4"))
        self.prediction=WorkflowDynamics(model,rules)
        self.solver=BusinessCPSAT(geometry,rules,
                  candidate_limit=config["max_legal_candidates"],
                  time_limit_seconds=config["cpsat_seconds"],
                  deterministic_time_limit=config["deterministic_cpsat_seconds"],
                  seed=1810)
        self.records=[]
        self.predict_steps=0
        self.prediction_cpu=0.
        self.policy_cpu=0.
        self.failure_count=0
        self.unknown_retest_rollout_count=0

    @property
    def query_calls(self):
        return getattr(self.prediction.model,"query_calls",0)

    @property
    def fallback_reasons(self):
        return getattr(self.prediction.model,"fallback_reasons",{})

    def _synchronize_public_rules(self,rules):
        self.solver.rules=rules
        self.prediction.rules=rules
        if hasattr(self.prediction.model,"rules"):
            self.prediction.model.rules=rules

    def _score(self,observation,plan,rules,index):
        missed=sum(s.status=="waiting" and s.priority==2 and
                   s.id not in {a.sample_id for a in plan}
                   for s in observation.samples)
        outcomes=[]
        starts=time.process_time()
        for particle in range(self.particles):
            self.prediction.reset_seed(1810+observation.tick*1009+particle)
            state=observation
            trace=[]
            for step in range(self.steps):
                # The same fixed continuation is used for P4 and P5.
                actions=plan if step==0 else nearest_batch(
                    state,candidates_for(state),self.geometry)
                nxt=self.prediction.predict_next(state,actions)
                check_business_transition(state,nxt,self.geometry,rules,actions)
                self.predict_steps+=1
                trace.append(nxt)
                state=nxt
            cost,_=_projected_job_cost(observation,trace,rules,self.geometry)
            if not all(math.isfinite(v) for v in cost):
                raise ValueError("non-finite urgent rollout cost")
            outcomes.append(cost)
        self.prediction_cpu+=time.process_time()-starts
        means=tuple(sum(c[i] for c in outcomes)/len(outcomes)
                    for i in range(len(outcomes[0])))
        # Existing primary urgency coverage target is intentionally retained.
        # Later components reflect the same per-job deadline objective in P4/P5.
        return (missed,*means,index),outcomes

    def __call__(self,raw,candidates,sim):
        begin=time.process_time()
        obs=self.observer.observe(raw)
        rules=sim.business.rules
        self._synchronize_public_rules(rules)
        plans=unique_plans(self.solver.proposals(obs,candidates),candidates)
        if not plans:
            return ()
        scores=[]
        for index,plan in enumerate(plans):
            score,particles=self._score(obs,plan,rules,index)
            scores.append({"matching":[asdict(a) for a in plan],
                           "score":score,"particle_costs":particles})
        choice=min(range(len(plans)),key=lambda i:scores[i]["score"])
        selected=plans[choice]
        if any(a not in candidates for a in selected):
            raise AssertionError("world policy chose an illegal assignment")
        self.unknown_retest_rollout_count+=sum(
            s.stage_index==2 and s.status!="completed"
            for s in obs.samples)
        self.policy_cpu+=time.process_time()-begin
        self.records.append({
            "tick":obs.tick,
            "observation_sha256":digest(asdict(obs)),
            "candidate_count":len(plans),
            "plans":scores,
            "selected_candidate_index":choice,
            "selected":[asdict(a) for a in selected],
            "rank":sorted(range(len(plans)),key=lambda i:scores[i]["score"]),
            "predicted_retest_event":"not_modelled_until_disclosed",
            "urgent_waiting":sum(s.status=="waiting" and s.priority==2 for s in obs.samples),
        })
        return selected


def run_case(spec,protocol,arm,checkpoint):
    business,geo=scene_for(spec)
    sim=ConditionalRetestSimulator(
        business,spec["seed"],spec["retest_rate_percent"],collect_metrics=True)
    policy=UrgentWorldPolicy(arm,geo,business.rules,protocol,checkpoint,seed=spec["seed"])
    started=time.process_time()
    result=sim.run(dispatch_policy=lambda o,c:policy(o,c,sim),
                   on_observation=policy.observer.observe)
    cpu=time.process_time()-started
    capacity=resource_audit(result,sim.business.rules)
    motion=discrete_safety_violations(result)
    lifecycle=lifecycle_audit(result["history"])
    if capacity["violations"] or motion or lifecycle:
        raise AssertionError("CAPACITY_MOTION_LIFECYCLE_INTEGRITY_FAILURE")
    metrics=job_metrics(result,sim.business,protocol["scenario_max_ticks"])
    if metrics["job_count"]!=spec["jobs"]:
        raise AssertionError("invalid number of original clinical jobs")
    if len(sim.retest_results)!=sum(e["type"]=="retest_decision" for e in result["events"]):
        raise AssertionError("retest event mismatch")
    for jid,yes in sim.retest_results.items():
        if yes!=synthetic_retest(spec["seed"],jid,spec["retest_rate_percent"]):
            raise AssertionError("retest exogenous mismatch")
    if arm=="P5" and policy.query_calls==0:
        raise AssertionError("V4 loaded but no genuine service-event queries were used")
    return {
        "seed":spec["seed"],"index":spec["index"],"arm":arm,"spec":spec,
        "metrics":metrics,"decision_count":len(policy.records),
        "decisions":policy.records,"query_calls":policy.query_calls,
        "fallback_reasons":policy.fallback_reasons,
        "rollout_steps":policy.predict_steps,
        "unmodelled_detection_sessions":policy.unknown_retest_rollout_count,
        "prediction_cpu_seconds":policy.prediction_cpu,
        "policy_cpu_seconds":policy.policy_cpu,"cpu_seconds":cpu,
        "resource_wait_ticks":capacity["capacity_wait_ticks"],
        "motion_blocked_attempts":result["summary"]["blocked_move_attempts"],
        "resource_violations":len(capacity["violations"]),
        "motion_violations":motion,"lifecycle_violations":len(lifecycle),
        "retest_results":sim.retest_results,
        "history_sha256":digest(result["history"]),"events_sha256":digest(result["events"]),
        "workflow_status":result["summary"]["status"],
        "completed_stages":result["summary"]["completed_count"],
    }
