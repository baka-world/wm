"""Bounded CP-SAT proposal and observable multi-step rollout selection."""
from __future__ import annotations

import time

from experiments.service_value.scoring import workload_components, workload_score
from lab_transport.decision_value import candidates_for, nearest_batch
from lab_transport.paper_reuse.contracts import check_transition
from lab_transport.paper_reuse.ortools_dispatch import CPSATDispatcher


class RolloutDispatch:
    """CP-SAT generates legal matching candidates; model scores their futures."""
    def __init__(self,geometry,dynamics,observer,*,horizon=5,particles=3,seed=123,
                 solver_seconds=0.5,proposals=4):
        if not 1 <= horizon <= 20 or not 1 <= particles <= 10:
            raise ValueError('invalid bounded rollout budget')
        self.geometry,self.dynamics,self.observer = geometry,dynamics,observer
        self.horizon,self.particles,self.seed = horizon,particles,seed
        self.solver = CPSATDispatcher(geometry,time_limit_seconds=solver_seconds,candidate_limit=proposals,seed=seed)
        self.model_calls = self.rollout_steps = self.fallbacks = 0
        self.log = []

    def __call__(self,observation,candidates):
        started = time.process_time()
        observation = self.observer.observe(observation)
        plans = self.solver.proposals(observation,candidates)
        records = []
        calls_before = getattr(getattr(self.dynamics,'distribution',None),'forward_calls',0)
        transition_before = getattr(self.dynamics,'transition_model_calls',0)
        scoring_before = getattr(self.dynamics,'scoring_model_calls',0)
        completions_before = getattr(self.dynamics,'sampled_completion_events',0)
        steps_before = self.rollout_steps
        try:
            for plan in plans:
                particle_scores = []
                for particle in range(self.particles):
                    if hasattr(self.dynamics,'reset_seed'):
                        self.dynamics.reset_seed(self.seed + observation.tick*1009 + particle)
                    state,trace = observation,[]
                    for step in range(self.horizon):
                        actions = plan if step == 0 else nearest_batch(state,candidates_for(state),self.geometry)
                        nxt = self.dynamics.predict_next(state,actions)
                        self.rollout_steps += 1
                        check_transition(state,nxt,self.geometry)
                        trace.append(nxt)
                        state = nxt
                    score = workload_score(observation,trace,self.geometry)
                    if hasattr(self.dynamics,'estimated_service_work'):
                        components = workload_components(observation,trace,self.geometry)
                        learned = self.dynamics.estimated_service_work(observation,trace[-1])
                        score = (score[0]-components['estimated_remaining_service']+learned,
                                 score[1],score[2])
                    particle_scores.append(score)
                scores = tuple(sum(x[i] for x in particle_scores)/len(particle_scores) for i in range(3))
                records.append({'assignments':[{'robot_id':a.robot_id,'sample_id':a.sample_id} for a in plan],
                                'score':scores})
            choice = min(range(len(plans)),key=lambda i:(records[i]['score'],i))
            fallback = None
        except (ValueError,RuntimeError,TypeError,KeyError,StopIteration,OverflowError) as exc:
            self.fallbacks += 1
            choice,fallback = 0,type(exc).__name__
        calls_used = getattr(getattr(self.dynamics,'distribution',None),'forward_calls',0)-calls_before
        self.model_calls += calls_used
        self.log.append({'tick':observation.tick,'candidates':len(candidates),'proposals':records,
            'selected_index':choice,'fallback':fallback,'rollout_steps':self.rollout_steps-steps_before,
            'neural_model_forward_calls':calls_used,
            'neural_transition_calls':getattr(self.dynamics,'transition_model_calls',0)-transition_before,
            'neural_scoring_calls':getattr(self.dynamics,'scoring_model_calls',0)-scoring_before,
            'sampled_completion_events':getattr(self.dynamics,'sampled_completion_events',0)-completions_before,
            'cpu_seconds':time.process_time()-started})
        return plans[choice]


class NearestRuleDispatch:
    """Existing nearest batch, with a matching bounded rule-prediction diagnostic."""
    def __init__(self,geometry,dynamics,observer,horizon=5):
        self.geometry,self.dynamics,self.observer,self.horizon=geometry,dynamics,observer,horizon
        self.model_calls=self.rollout_steps=self.fallbacks=0
        self.log=[]

    def __call__(self,observation,candidates):
        started=time.process_time()
        observation=self.observer.observe(observation)
        plan=nearest_batch(observation,candidates,self.geometry)
        state=observation
        fallback=None
        steps_before=self.rollout_steps
        try:
            for step in range(self.horizon):
                action=plan if step==0 else nearest_batch(state,candidates_for(state),self.geometry)
                nxt=self.dynamics.predict_next(state,action)
                self.rollout_steps+=1
                check_transition(state,nxt,self.geometry)
                state=nxt
        except (ValueError,RuntimeError,TypeError,KeyError,StopIteration,OverflowError) as exc:
            self.fallbacks+=1
            fallback=type(exc).__name__
        self.log.append({'tick':observation.tick,'fallback':fallback,
            'rollout_steps':self.rollout_steps-steps_before,'neural_model_forward_calls':0,
            'neural_transition_calls':0,'neural_scoring_calls':0,'sampled_completion_events':0,
            'cpu_seconds':time.process_time()-started})
        return plan
