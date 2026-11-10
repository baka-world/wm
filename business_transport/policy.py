"""Fair bounded candidate rollouts for five business scheduling baselines."""
from __future__ import annotations

import time
import math
import statistics
from experiments.service_value.scoring import workload_score
from business_transport.dispatch import BusinessCPSAT
from business_transport.dynamics import CapacityDynamics,WorkflowDynamics
from business_transport.workflow_scoring import workflow_components,score_components
from business_transport.service_work import ServiceWorkEstimator
from business_transport.workflow_transition import check_business_transition
from lab_transport.decision_value import batch_candidates,candidates_for,nearest_batch
from lab_transport.paper_reuse.contracts import check_transition


class BusinessPolicy:
    def __init__(self,method,geometry,rules,observer,model=None,*,seed=0,horizon=5,
                 particles=3,budget=4,solver_seconds=.5,workflow_prediction=False,scoring='legacy',
                 max_prediction_seconds=None,risk_weight=0.0):
        if method not in ('nearest','nearest_rollout','cpsat_rule','cpsat_pets','cpsat_structured'):
            raise ValueError('unknown business scheduling baseline')
        self.method,self.geometry,self.rules,self.observer=method,geometry,rules,observer
        self.seed,self.horizon,self.particles,self.budget=seed,horizon,particles,budget
        if scoring not in ('legacy','workflow','workflow_model'):raise ValueError('unknown business score')
        if scoring=='workflow_model' and not workflow_prediction:
            raise ValueError('model-aware scoring needs full workflow transition')
        if horizon < 1 or particles < 1 or budget < 1:raise ValueError('nonpositive rollout budget')
        if scoring=='workflow_model' and horizon not in (5,10,20):
            raise ValueError('model-aware horizon must be 5, 10 or 20')
        if max_prediction_seconds is not None and max_prediction_seconds <= 0:
            raise ValueError('prediction timeout must be positive')
        if not math.isfinite(risk_weight) or risk_weight < 0:
            raise ValueError('invalid risk weight')
        self.max_prediction_seconds=max_prediction_seconds
        self.risk_weight=float(risk_weight)
        self.workflow_prediction,self.scoring=workflow_prediction,scoring
        adapter=WorkflowDynamics if workflow_prediction else CapacityDynamics
        self.model=adapter(model,rules) if model is not None else None
        self.solver=(BusinessCPSAT(geometry,rules,candidate_limit=budget,time_limit_seconds=solver_seconds,seed=seed)
                     if method.startswith('cpsat_') else None)
        self.estimator=ServiceWorkEstimator(self.model) if scoring=='workflow_model' else None
        self.log=[];self.fallbacks=0;self.rollout_steps=0
        self.prediction_calls=0;self.prediction_wall_seconds=0.0

    def candidates(self,observation,candidates):
        if self.method=='nearest':return (nearest_batch(observation,candidates,self.geometry),)
        if self.method=='nearest_rollout':
            # Existing nearest-ranked compatible matchings; empty wait action
            # is not in the CP-SAT action space, so it is excluded here too.
            plans,_=batch_candidates(observation,candidates,self.geometry,self.budget+1)
            plans=tuple(x for x in plans if x)
            return plans or ((),)
        return self.solver.proposals(observation,candidates)

    def evaluate(self,observation,plans):
        """Candidate forecasts are independent; no simulator future is consulted."""
        records=[]
        started_batch=time.perf_counter()
        for index,plan in enumerate(plans):
            outcomes=[];components=[];particle_diagnostics=[]
            for particle in range(self.particles):
                # Reset each candidate to the SAME particle seed (common random
                # numbers), independent of candidate ordering or evaluation count.
                self.model.reset_seed(self.seed+observation.tick*1009+particle)
                state=observation;trace=[]
                reason='prediction_window_end'
                for step in range(self.horizon):
                    if (self.max_prediction_seconds is not None and
                        time.perf_counter()-started_batch > self.max_prediction_seconds):
                        raise TimeoutError('prediction_wall_budget_exceeded')
                    actions=plan if step==0 else nearest_batch(state,candidates_for(state),self.geometry)
                    if not actions and step>0:reason='no_new_legal_assignment'
                    started=time.perf_counter()
                    try:
                        nxt=self.model.predict_next(state,actions)
                    finally:
                        self.prediction_calls+=1;self.prediction_wall_seconds+=time.perf_counter()-started
                    if self.workflow_prediction:check_business_transition(state,nxt,self.geometry,self.rules,actions)
                    else:check_transition(state,nxt,self.geometry)
                    self.rollout_steps+=1;trace.append(nxt);state=nxt
                if self.scoring in ('workflow','workflow_model'):
                    c=workflow_components(observation,trace,self.geometry,self.rules,
                        estimator=self.estimator)
                    outcomes.append(score_components(c));components.append(c)
                else:
                    outcomes.append(workload_score(observation,trace,self.geometry))
                particle_diagnostics.append({'index':particle,'steps':len(trace),
                    'termination':reason,'business_score':outcomes[-1][0]})
            mean_cost=statistics.mean(row[0] for row in outcomes)
            spread=statistics.pstdev(row[0] for row in outcomes) if len(outcomes)>1 else 0.
            ordered=sorted(row[0] for row in outcomes)
            p90=ordered[min(len(ordered)-1,math.ceil(.90*len(ordered))-1)]
            score=list(sum(row[i] for row in outcomes)/len(outcomes) for i in range(3))
            if self.risk_weight:
                # Pre-specified simple p90 penalty, not a calibrated tail CI.
                score[0]+=self.risk_weight*(p90-mean_cost)
            record={'assignments':[{'robot_id':a.robot_id,'sample_id':a.sample_id} for a in plan],
                    'score':tuple(score),'candidate_index':index,
                    'particle_business_costs':[row[0] for row in outcomes],
                    'particle_diagnostics':particle_diagnostics,
                    'cost_distribution':{'mean':mean_cost,'population_std':spread,
                                         'empirical_p90':p90,'particles':len(outcomes),
                                         'calibrated_interval':False},
                    'risk_weight':self.risk_weight}
            if self.scoring in ('workflow','workflow_model'):
                record['components']=components
                record['component_means']={key:statistics.mean(c[key] for c in components) for key in
                    ('unfinished_job_ticks','remaining_distance','estimated_remaining_service',
                     'resource_contention_proxy','terminal_work')}
                if self.scoring=='workflow_model':
                    record['component_means'].update({key:statistics.mean(c[key] for c in components) for key in
                        ('rule_prior_service','model_service','model_minus_prior')})
            records.append(record)
        return records

    def __call__(self,observation,candidates):
        started=time.perf_counter()
        obs=self.observer.observe(observation)
        plans=self.candidates(obs,candidates)
        records=[];reason=None
        before=getattr(getattr(self.model,'distribution',None),'forward_calls',0)
        service_before=getattr(getattr(self.model,'model',None),'service_calls',0)
        try:
            if self.method=='nearest':choice=0
            else:
                records=self.evaluate(obs,plans)
                choice=min(range(len(plans)),key=lambda i:(records[i]['score'],i))
                for i,record in enumerate(records):
                    record['selected']=i==choice
                    record['selection_reason']=('minimum_expected_business_cost' if i==choice else
                        'higher_expected_business_cost_or_lexicographic_tie')
        except (ValueError,RuntimeError,TypeError,KeyError,StopIteration,OverflowError,TimeoutError) as exc:
            self.fallbacks+=1;reason=type(exc).__name__+':'+str(exc)[:120];choice=0
        after=getattr(getattr(self.model,'distribution',None),'forward_calls',0)
        self.log.append({'tick':obs.tick,'candidate_count':len(candidates),'plans':records,
            'selected':[{'robot_id':a.robot_id,'sample_id':a.sample_id} for a in plans[choice]],
            'fallback':reason,'selected_candidate_index':choice,
            'horizon':self.horizon,'particles':self.particles,'scoring':self.scoring,
            'prediction_wall_seconds':self.prediction_wall_seconds,
            'model_forward_calls':after-before,
            'structured_service_calls':getattr(getattr(self.model,'model',None),'service_calls',0)-service_before,
            'wall_seconds':time.perf_counter()-started})
        return plans[choice]


class WorkflowPolicy(BusinessPolicy):
    """Business-facing entry point with full known-workflow rollout enabled.

    BusinessPolicy's original defaults remain available for archived baselines.
    Supply the same rule/PETS/structured movement adapter used by execution.
    """
    def __init__(self,method,geometry,rules,observer,model=None,*,model_aware=False,**kwargs):
        if 'workflow_prediction' in kwargs or 'scoring' in kwargs:
            raise ValueError('WorkflowPolicy always uses workflow prediction and score')
        super().__init__(method,geometry,rules,observer,model,
                         workflow_prediction=True,
                         scoring='workflow_model' if model_aware else 'workflow',**kwargs)
