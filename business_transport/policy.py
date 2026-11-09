"""Fair bounded candidate rollouts for five business scheduling baselines."""
from __future__ import annotations

import time
from experiments.service_value.scoring import workload_score
from business_transport.dispatch import BusinessCPSAT
from business_transport.dynamics import CapacityDynamics
from lab_transport.decision_value import batch_candidates,candidates_for,nearest_batch
from lab_transport.paper_reuse.contracts import check_transition


class BusinessPolicy:
    def __init__(self,method,geometry,rules,observer,model=None,*,seed=0,horizon=5,
                 particles=3,budget=4,solver_seconds=.5):
        if method not in ('nearest','nearest_rollout','cpsat_rule','cpsat_pets','cpsat_structured'):
            raise ValueError('unknown business scheduling baseline')
        self.method,self.geometry,self.rules,self.observer=method,geometry,rules,observer
        self.seed,self.horizon,self.particles,self.budget=seed,horizon,particles,budget
        self.model=CapacityDynamics(model,rules) if model is not None else None
        self.solver=(BusinessCPSAT(geometry,rules,candidate_limit=budget,time_limit_seconds=solver_seconds,seed=seed)
                     if method.startswith('cpsat_') else None)
        self.log=[];self.fallbacks=0;self.rollout_steps=0

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
        records=[]
        for plan in plans:
            outcomes=[]
            for particle in range(self.particles):
                self.model.reset_seed(self.seed+observation.tick*1009+particle)
                state=observation;trace=[]
                for step in range(self.horizon):
                    actions=plan if step==0 else nearest_batch(state,candidates_for(state),self.geometry)
                    nxt=self.model.predict_next(state,actions)
                    check_transition(state,nxt,self.geometry)
                    self.rollout_steps+=1;trace.append(nxt);state=nxt
                outcomes.append(workload_score(observation,trace,self.geometry))
            score=tuple(sum(s[i] for s in outcomes)/len(outcomes) for i in range(3))
            records.append({'assignments':[{'robot_id':a.robot_id,'sample_id':a.sample_id} for a in plan],
                            'score':score})
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
        except (ValueError,RuntimeError,TypeError,KeyError,StopIteration,OverflowError) as exc:
            self.fallbacks+=1;reason=type(exc).__name__;choice=0
        after=getattr(getattr(self.model,'distribution',None),'forward_calls',0)
        self.log.append({'tick':obs.tick,'candidate_count':len(candidates),'plans':records,
            'selected':[{'robot_id':a.robot_id,'sample_id':a.sample_id} for a in plans[choice]],
            'fallback':reason,'model_forward_calls':after-before,
            'structured_service_calls':getattr(getattr(self.model,'model',None),'service_calls',0)-service_before,
            'wall_seconds':time.perf_counter()-started})
        return plans[choice]
