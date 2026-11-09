from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from lab_transport.decision_value import candidates_for
from lab_transport.domain import Grid, RobotSpec, Sample, Scenario, Station, load_scenario
from lab_transport.dynamics_model import Geometry
from lab_transport.paper_reuse.contracts import check_transition
from lab_transport.paper_reuse.ortools_dispatch import CPSATDispatcher
from lab_transport.paper_reuse.pets_service import PETSRuleDynamics, ServiceDistribution
from lab_transport.paper_reuse.rollout_dispatch import RolloutDispatch
from lab_transport.paper_reuse.service_dataset import extract_episode
from lab_transport.simulation import Simulator
from lab_transport.state_v2 import ObservableHistory
from lab_transport.transition_schema import Assignment, Transition

ROOT=Path(__file__).resolve().parents[3]
CHECKPOINT=ROOT/'examples/evidence/paper-reuse-integration-20261109/pets-service-cpu.pt'


def outage_case():
    return Scenario('unit', Grid(5,2,frozenset()),
        {'A':Station('A',(0,0),4,4,((1,5),),0),'B':Station('B',(3,0),1,1,(),0)},
        (RobotSpec('R1',(0,0),(0,0)),),(Sample('S1','A','B',0,None),),0,80)


def rich_rows(case):
    raw=[]
    Simulator(case,'nearest',1).run(on_transition=raw.append)
    observer=ObservableHistory()
    return [Transition(observer.observe(r.observation),r.assignments,
                       observer.observe(r.next_observation),r.next_boundary,r.termination) for r in raw]


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.case=load_scenario(ROOT/'scenarios/laboratory_demo.json')
        self.geometry=Geometry(self.case.grid)
        raw=[]
        Simulator(self.case,'nearest',42).run(on_transition=raw.append)
        self.state=next(r.observation for r in raw if len(candidates_for(r.observation))>=2)

    def test_cpsat_returns_legal_distinct_assignments(self):
        candidates=candidates_for(self.state)
        solver=CPSATDispatcher(self.geometry,time_limit_seconds=.2,candidate_limit=4)
        plans=solver.proposals(self.state,candidates)
        self.assertTrue(plans and plans[0])
        for plan in plans:
            self.assertTrue(all(a in candidates for a in plan))
            self.assertEqual(len({a.robot_id for a in plan}),len(plan))
            self.assertEqual(len({a.sample_id for a in plan}),len(plan))
        self.assertIsNone(solver.log[-1]['fallback'])
        self.assertEqual(plans,solver.proposals(self.state,candidates))

    def test_busy_missing_and_unreachable_candidates_are_not_used(self):
        solver=CPSATDispatcher(self.geometry)
        bad=(Assignment('missing','missing'),)
        self.assertEqual(solver(self.state,bad),())
        self.assertEqual(solver.log[-1]['fallback'],'no_reachable_candidate')
        robot=self.state.robots[0]
        busy=replace(self.state,robots=tuple(replace(r,task_id='busy') if r.id==robot.id else r for r in self.state.robots))
        self.assertTrue(all(a.robot_id!=robot.id for a in solver(busy,candidates_for(self.state))))
        with patch.object(self.geometry,'distance',return_value=self.geometry.unreachable):
            self.assertEqual(solver(self.state,candidates_for(self.state)),())

    def test_solver_unavailable_has_deterministic_legal_fallback(self):
        solver=CPSATDispatcher(self.geometry)
        candidates=candidates_for(self.state)
        with patch('lab_transport.paper_reuse.ortools_dispatch.cp_model',None):
            first=solver(self.state,candidates)
            second=solver(self.state,candidates)
        self.assertEqual(first,second)
        self.assertIn(first[0],candidates)
        self.assertEqual(solver.log[-1]['fallback'],'ortools_unavailable')

    def test_completed_duration_excludes_outage_and_incomplete_is_censored(self):
        rows=rich_rows(outage_case())
        sessions,counts=extract_episode(SimpleNamespace(rows=rows))
        pickup=next(x for x in sessions if x['phase']=='to_pickup')
        self.assertEqual(pickup['duration_active_ticks'],4)
        self.assertEqual(pickup['paused_ticks_for_audit_only'],4)
        partial,censored=extract_episode(SimpleNamespace(rows=rows[:3]))
        self.assertFalse(partial)
        self.assertEqual(censored['excluded_incomplete'],1)

    def test_service_distribution_conditional_probability_and_seed(self):
        d=ServiceDistribution.load(CHECKPOINT)
        params=d.parameters([.4,.4,0.,1.,1.,0.,0.])
        self.assertEqual(len(params),5)
        self.assertGreater(d.mean(params),0)
        for t in range(8):
            self.assertTrue(0 <= d.finish_probability(params,t) <= 1)
        self.assertTrue(d.nll(params,4) < float('inf'))
        rows=rich_rows(outage_case())
        state=rows[0].observation
        a=PETSRuleDynamics(Geometry(outage_case().grid),d,seed=7)
        b=PETSRuleDynamics(Geometry(outage_case().grid),d,seed=7)
        for step in range(6):
            actions=rows[0].assignments if step==0 else ()
            x=a.predict_next(state,actions)
            y=b.predict_next(state,actions)
            self.assertEqual(x,y)
            check_transition(state,x,Geometry(outage_case().grid))
            state=x
        self.assertGreater(d.forward_calls,1)
        self.assertGreater(a.transition_model_calls,0)

    def test_station_outage_pauses_service_with_no_model_call(self):
        rows=rich_rows(outage_case())
        row=next(r for r in rows if r.observation.tick==3)
        d=ServiceDistribution.load(CHECKPOINT)
        dynamics=PETSRuleDynamics(Geometry(outage_case().grid),d,seed=7)
        out=dynamics.predict_next(row.observation,row.assignments)
        self.assertEqual(d.forward_calls,0)
        self.assertEqual(out.samples[0].status,'assigned')
        self.assertEqual(out.robots[0].service_active_elapsed,row.observation.robots[0].service_active_elapsed)

    def test_pets_duration_contributes_before_robot_arrives(self):
        observer=ObservableHistory()
        current=observer.observe(self.state)
        d=ServiceDistribution.load(CHECKPOINT)
        dynamics=PETSRuleDynamics(self.geometry,d,seed=7)
        work=dynamics.estimated_service_work(current,current)
        self.assertGreater(work,0)
        self.assertGreater(d.forward_calls,0)

    def test_same_seed_reproduces_complete_cpsat_pets_assignments(self):
        results=[]
        for _ in range(2):
            observer=ObservableHistory()
            dynamics=PETSRuleDynamics(Geometry(outage_case().grid),ServiceDistribution.load(CHECKPOINT),seed=11)
            policy=RolloutDispatch(Geometry(outage_case().grid),dynamics,observer,seed=11,solver_seconds=.2)
            result=Simulator(outage_case(),'world',11).run(dispatch_policy=policy,on_observation=observer.observe)
            results.append((result['summary'],result['events']))
        self.assertEqual(results[0],results[1])

    def test_forecast_failure_returns_legal_cpsat_plan(self):
        class BadModel:
            def predict_next(self,*args): raise ValueError('forecast failed')
        observer=ObservableHistory()
        # Observer supports a first observation in the middle of a trajectory.
        policy=RolloutDispatch(self.geometry,BadModel(),observer,solver_seconds=.2)
        actions=policy(self.state,candidates_for(self.state))
        self.assertTrue(all(a in candidates_for(self.state) for a in actions))
        self.assertEqual(policy.fallbacks,1)
        self.assertEqual(policy.log[-1]['fallback'],'ValueError')

    def test_failed_service_model_records_call_and_falls_back(self):
        class InvalidDistribution:
            def __init__(self): self.forward_calls=0
            def parameters(self,features):
                self.forward_calls+=1
                raise ValueError('invalid service distribution')
        case=outage_case()
        row=rich_rows(case)[0]
        geo=Geometry(case.grid)
        model=PETSRuleDynamics(geo,InvalidDistribution(),seed=9)
        observer=ObservableHistory()
        policy=RolloutDispatch(geo,model,observer,solver_seconds=.2)
        actions=policy(row.observation,candidates_for(row.observation))
        self.assertTrue(actions)
        self.assertEqual(policy.fallbacks,1)
        self.assertEqual(policy.model_calls,1)
        self.assertEqual(policy.log[-1]['neural_transition_calls'],1)

    def test_injected_future_samples_do_not_affect_dataset_features(self):
        rows=rich_rows(outage_case())
        first,_=extract_episode(SimpleNamespace(rows=rows))
        later=replace(rows[-1],next_observation=replace(rows[-1].next_observation,
            samples=rows[-1].next_observation.samples+(replace(rows[-1].observation.samples[0],id='future',release_tick=999),)))
        altered,_=extract_episode(SimpleNamespace(rows=rows[:-1]+[later]))
        self.assertEqual([(x['features'],x['duration_active_ticks']) for x in first],
                         [(x['features'],x['duration_active_ticks']) for x in altered])

if __name__=='__main__':unittest.main()
