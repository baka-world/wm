"""Experimental frozen V4 vs Rule integration on synthetic urgent/retest workflows.

No training, hidden future event, changed production scheduler, or test tuning.
The simulator remains authoritative. Retest branches are unknowable before
detection; a rollout uses the currently public default route until disclosure.
"""
from __future__ import annotations
from dataclasses import asdict
from collections import defaultdict
import hashlib
import math
import time

from business_transport.dispatch import BusinessCPSAT
from business_transport.dynamics import WorkflowDynamics
from business_transport.session_dynamics import SessionRuleDynamics
from business_transport.session_events import eligible_sessions
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



def _stable_service_uniform(seed, particle, observation_tick, query):
    """Common uniform for the *same visible service query* across candidates.

    Do not use the simulator's private seed as a future label. This is a
    simulation-only Monte-Carlo coupling seed, not a learned feature.
    The event query identity is action-independent once it is eligible.
    """
    fields=("pr18-keyed-crn-v1",int(seed),int(particle),int(observation_tick),
            query["robot_id"],query["sample_id"],query["phase"],query["station_id"])
    blob="|".join(map(str,fields)).encode("utf8")
    value=int.from_bytes(hashlib.sha256(blob).digest()[:8],"big")
    return value/(1<<64)


class _OneStepUniforms:
    """Supplies exactly one uniform per actual eligible service-session query."""
    def __init__(self,values):
        self.values=tuple(values)
        self.calls=0

    def random(self):
        if self.calls >= len(self.values):
            raise AssertionError("more service samples than current eligible queries")
        value=self.values[self.calls]
        self.calls+=1
        return value


class UrgentWorldPolicy:
    def __init__(self,arm,geometry,rules,config,checkpoint,*,seed,steps=5,particles=3,random_coupling="stream",terminal_mode="optimistic",evaluation_mode="full"):
        if arm not in ("P4","P5"):
            raise ValueError("expected P4 Rule or P5 frozen V4")
        if steps!=5 or particles!=3:
            raise ValueError("first adaptation uses fixed H5/P3 rollout budget")
        if random_coupling not in ("stream","keyed"):
            raise ValueError("unknown service sampling coupling")
        self.random_coupling=random_coupling
        if terminal_mode not in ("optimistic","rule_continuation","budgeted_queue"):
            raise ValueError("unknown terminal value estimator")
        self.terminal_mode=terminal_mode
        if evaluation_mode not in ("full","dominance"):
            raise ValueError("unknown candidate evaluation mode")
        self.evaluation_mode=evaluation_mode
        self.candidates_pruned=0
        self.candidates_evaluated=0
        self.arm,self.geometry,self.config=arm,geometry,config
        self.seed,self.steps,self.particles=seed,steps,particles
        self.observer=PriorityObserver()
        model=(ProgressAwareRule(geometry) if arm=="P4" else
               SessionRuleDynamics(geometry,rules,checkpoint,experimental=True,
                                   seed=1810,feature_version="v4"))
        self.prediction=WorkflowDynamics(model,rules)
        # Optional public-rule-only continuation, *after* the same 5-step
        # frozen Rule/V4 prefix. No hidden simulator events, no private retest.
        self.terminal_prediction=WorkflowDynamics(ProgressAwareRule(geometry),rules)
        self.terminal_steps=0
        self.budgeted_queue_ticks=20
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
        self.terminal_prediction.rules=rules
        if hasattr(self.prediction.model,"rules"):
            self.prediction.model.rules=rules


    def _continue_to_terminal(self,initial,state,trace,rules):
        """Deterministic public-workflow continuation to the synthetic limit.

        This models robot sequencing and resource admission via the existing
        legal transition model rather than summing independent optimistic
        job durations. Future retest outcomes, unobserved job arrivals and
        outage calendars remain unknown. It is a post-hoc diagnostic, not a
        calibrated prediction of a real clinical completion time.
        """
        initial_jobs={s.job_id:s.workflow for s in initial.samples if s.job_id}
        horizon=self.config["scenario_max_ticks"]
        if self.terminal_mode=="budgeted_queue":
            # Fixed and shared budget: simulate twenty additional public
            # transition ticks, never to an oracle future. The remaining
            # cost is still the unchanged optimistic public proxy.
            horizon=min(horizon,state.tick+self.budgeted_queue_ticks)
        while state.tick < horizon:
            # The same public default follow-up is used in both arms.
            actions=nearest_batch(state,candidates_for(state),self.geometry)
            nxt=self.terminal_prediction.predict_next(state,actions)
            check_business_transition(state,nxt,self.geometry,rules,actions)
            if nxt.tick != state.tick+1:
                raise AssertionError("terminal continuation skipped a clock tick")
            self.terminal_steps+=1
            trace.append(nxt)
            state=nxt
            if initial_jobs:
                done={(s.job_id,s.workflow) for s in state.samples
                      if s.job_id and s.status=="completed"
                      and s.stage_index==len(rules.workflows[s.workflow])-1}
                if all((job,flow) in done for job,flow in initial_jobs.items()):
                    break
        return state

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
                if self.arm=="P5" and self.random_coupling=="keyed":
                    # Legacy reset_seed() couples random positions, not
                    # service identities. A candidate can insert a query
                    # and thereby shift draws for all later sessions.
                    # Stable service-keyed draws remove that confound without
                    # changing the V4 network, scoring or legal action set.
                    viable=[q for q,reason in eligible_sessions(state,actions)
                            if reason is None]
                    keyed=_OneStepUniforms([
                        _stable_service_uniform(self.seed,particle,state.tick,q)
                        for q in viable])
                    model=self.prediction.model
                    previous_rng=model.rng
                    model.rng=keyed
                    try:
                        nxt=self.prediction.predict_next(state,actions)
                    finally:
                        model.rng=previous_rng
                    if keyed.calls!=len(viable):
                        raise AssertionError("V4 draw count != eligible query count")
                else:
                    nxt=self.prediction.predict_next(state,actions)
                check_business_transition(state,nxt,self.geometry,rules,actions)
                self.predict_steps+=1
                trace.append(nxt)
                state=nxt
            if self.terminal_mode in ("rule_continuation","budgeted_queue"):
                state=self._continue_to_terminal(observation,state,trace,rules)
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
        # Exact lexicographic dominance: the first score component is
        # uncovered currently-waiting urgent assignments. No downstream
        # predicted loss can overturn a strictly worse first component.
        # Compute this from the current lawful observation, before any
        # stochastic rollouts, without changing the legal candidate set.
        waiting_urgent={s.id for s in obs.samples if s.status=="waiting" and s.priority==2}
        uncovered=[
            len(waiting_urgent-{a.sample_id for a in plan})
            for plan in plans
        ]
        minimum_uncovered=min(uncovered)
        scores=[]
        evaluated=[]
        pruned=[]
        for index,plan in enumerate(plans):
            matching=[asdict(a) for a in plan]
            if self.evaluation_mode=="dominance" and uncovered[index]>minimum_uncovered:
                scores.append({"matching":matching,"score":None,"particle_costs":[],
                               "evaluated":False,"uncovered_urgent":uncovered[index],
                               "pruned_reason":"strictly_worse_first_lexicographic_component"})
                pruned.append(index)
                self.candidates_pruned+=1
                continue
            score,particles=self._score(obs,plan,rules,index)
            if score[0]!=uncovered[index]:
                raise AssertionError("coverage score inconsistency")
            scores.append({"matching":matching,"score":score,"particle_costs":particles,
                           "evaluated":True,"uncovered_urgent":uncovered[index]})
            evaluated.append(index)
            self.candidates_evaluated+=1
        if not evaluated:
            raise AssertionError("no non-dominated candidate")
        choice=min(evaluated,key=lambda i:scores[i]["score"])
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
            # Non-dominated candidates have their exact full-score ranking.
            # Dominated candidates are placed last in first-component order
            # but their downstream scores are intentionally NOT computed.
            "rank":sorted(evaluated,key=lambda i:scores[i]["score"])+
                   sorted(pruned,key=lambda i:(uncovered[i],i)),
            "partial_rank":bool(pruned),
            "candidates_pruned":len(pruned),
            "candidate_evaluation_mode":self.evaluation_mode,
            "predicted_retest_event":"not_modelled_until_disclosed",
            "random_coupling":self.random_coupling,
            "terminal_mode":self.terminal_mode,
            "urgent_waiting":sum(s.status=="waiting" and s.priority==2 for s in obs.samples),
        })
        return selected


def run_case(spec,protocol,arm,checkpoint,*,random_coupling="stream",terminal_mode="optimistic",evaluation_mode="full"):
    business,geo=scene_for(spec)
    sim=ConditionalRetestSimulator(
        business,spec["seed"],spec["retest_rate_percent"],collect_metrics=True)
    policy=UrgentWorldPolicy(arm,geo,business.rules,protocol,checkpoint,seed=spec["seed"],random_coupling=random_coupling,terminal_mode=terminal_mode,evaluation_mode=evaluation_mode)
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
        "random_coupling":random_coupling,
        "terminal_mode":terminal_mode,
        "evaluation_mode":evaluation_mode,
        "candidates_pruned":policy.candidates_pruned,
        "candidates_evaluated":policy.candidates_evaluated,
        "terminal_continuation_steps":policy.terminal_steps,
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
