import os
from dataclasses import replace
from pathlib import Path
import unittest

from business_transport.dynamics import CapacityDynamics
from business_transport.models import CountingStructuredService
from business_transport.pibt_integration import CoordinatedBusinessSimulator,MotionObserver,PIBTMovementModel
from business_transport.pibt_motion import PIBTMotion
from business_transport.policy import BusinessPolicy
from business_transport.simulation import resource_audit
from experiments.business_constraints.tests.test_business import fixture
from experiments.service_value.counterfactual import load_pr9_model
from lab_transport.decision_value import candidates_for
from lab_transport.domain import DirectedGraph,Grid
from lab_transport.dynamics_model import Geometry
from lab_transport.paper_reuse.contracts import check_transition
from lab_transport.paper_reuse.pets_service import PETSRuleDynamics,ServiceDistribution
from lab_transport.torch_structured_adapter import ProgressAwareRule

LIBRARY=os.environ.get('WM_PIBT_LIBRARY')
ROOT=Path(__file__).resolve().parents[3]


@unittest.skipUnless(LIBRARY and Path(LIBRARY).is_file(),'set WM_PIBT_LIBRARY to the built official PIBT bridge')
class OfficialPIBTIntegrationTests(unittest.TestCase):
    def test_three_agent_cycle_preserves_one_way_edges(self):
        nodes={(0,0):(0.,0.),(1,0):(1.,0.),(2,0):(2.,0.)}
        edges=(((0,0),(1,0)),((1,0),(2,0)),((2,0),(0,0)))
        backend=PIBTMotion(Geometry(DirectedGraph(nodes,edges)),LIBRARY)
        positions={'A':(0,0),'B':(1,0),'C':(2,0)}
        goals={'A':(1,0),'B':(2,0),'C':(0,0)}
        after=backend.step(positions,goals,set(),0)
        self.assertEqual(after,goals)
        self.assertEqual(backend.solver_calls,1)

    def test_service_and_paused_robot_positions_are_never_entered(self):
        backend=PIBTMotion(Geometry(Grid(3,3,frozenset())),LIBRARY)
        positions={'A':(0,1),'B':(1,1),'C':(2,1)}
        goals={'A':(2,1),'B':(1,1),'C':(0,1)}
        for tick in range(6):
            after=backend.step(positions,goals,{'B'},tick)
            self.assertEqual(after['B'],(1,1))
            self.assertNotEqual(after['A'],(1,1));self.assertNotEqual(after['C'],(1,1))
            backend.validate(positions,after,{'B'});positions=after

    def test_original_backend_default_is_unchanged(self):
        _,_,business=fixture(1);geometry=Geometry(business.scenario.grid)
        def run(default):
            obs=MotionObserver();p=BusinessPolicy('nearest',geometry,business.rules,obs)
            kwargs={} if default else {'motion_backend':'legacy'}
            return CoordinatedBusinessSimulator(business,'world',4,**kwargs).run(
                dispatch_policy=p,on_observation=obs.observe)
        self.assertEqual(run(True),run(False))

    def test_future_pause_schedule_is_not_an_online_observation(self):
        _,_,business=fixture(1)
        first=CoordinatedBusinessSimulator(business,'world',1,motion_backend='pibt',pibt_library=LIBRARY,
                                           motion_pauses={'R1':((10,20),)})
        second=CoordinatedBusinessSimulator(business,'world',1,motion_backend='pibt',pibt_library=LIBRARY,
                                            motion_pauses={'R1':((20,30),)})
        first_now=first._observable_state(0);second_now=second._observable_state(0)
        self.assertEqual(first_now,second_now)
        self.assertTrue(all(r.operational_available for r in first_now.robots))

    def test_world_models_use_same_pibt_steps_as_execution_and_current_pauses(self):
        import torch
        torch.set_num_threads(1)
        _,_,business=fixture(1);geometry=Geometry(business.scenario.grid)
        observer=MotionObserver();policy=BusinessPolicy('nearest',geometry,business.rules,observer)
        sim=CoordinatedBusinessSimulator(business,'world',3,motion_backend='pibt',pibt_library=LIBRARY,
            motion_pauses={'R1':((2,6),)})
        rows=[];result=sim.run(dispatch_policy=policy,on_observation=observer.observe,on_transition=rows.append)
        self.assertEqual(result['summary']['status'],'completed')
        self.assertFalse(resource_audit(result,business.rules)['violations'])
        network,info=load_pr9_model()
        ckpt=ROOT/'examples/evidence/paper-reuse-integration-20261109/pets-service-cpu.pt'
        models=(ProgressAwareRule(geometry),
            PETSRuleDynamics(geometry,ServiceDistribution.load(ckpt),3),
            CountingStructuredService(geometry,network,torch.device('cpu'),info['threshold']))
        for model in models:
            offline=MotionObserver();bridge=PIBTMotion(geometry,LIBRARY)
            predictor=CapacityDynamics(PIBTMovementModel(model,geometry,bridge),business.rules)
            for row in rows:
                current=offline.observe(row.observation)
                predicted=predictor.predict_next(current,row.assignments)
                check_transition(current,predicted,geometry)
                self.assertEqual({r.id:r.position for r in predicted.robots},
                                 {r.id:r.position for r in row.next_observation.robots})
            self.assertGreater(bridge.solver_calls,0)
        paused=[row for row in rows if 2<=row.observation.tick<6]
        self.assertTrue(paused)
        self.assertTrue(all(next(r for r in row.observation.robots if r.id=='R1').operational_available is False
                            for row in paused))

    def test_cp_sat_rollout_uses_the_shared_native_motion_backend(self):
        _,_,business=fixture(1);geometry=Geometry(business.scenario.grid);observer=MotionObserver()
        bridge=PIBTMotion(geometry,LIBRARY)
        model=PIBTMovementModel(ProgressAwareRule(geometry),geometry,bridge)
        policy=BusinessPolicy('cpsat_rule',geometry,business.rules,observer,model,seed=20261109)
        sim=CoordinatedBusinessSimulator(business,'world',1,motion_backend='pibt',pibt_library=LIBRARY,
                                         collect_metrics=True)
        result=sim.run(dispatch_policy=policy,on_observation=observer.observe)
        self.assertEqual(result['summary']['status'],'completed')
        self.assertGreater(sim.pibt.solver_calls,0)
        self.assertGreater(bridge.solver_calls,0)
        self.assertEqual(policy.fallbacks,0)
        self.assertFalse(resource_audit(result,business.rules)['violations'])

    def test_existing_neural_service_models_participate_in_cp_sat_rollout(self):
        import torch
        torch.set_num_threads(1)
        _,_,business=fixture(1);geometry=Geometry(business.scenario.grid)
        network,info=load_pr9_model()
        ckpt=ROOT/'examples/evidence/paper-reuse-integration-20261109/pets-service-cpu.pt'
        for name,model in (
            ('cpsat_pets',PETSRuleDynamics(geometry,ServiceDistribution.load(ckpt),20261109)),
            ('cpsat_structured',CountingStructuredService(geometry,network,torch.device('cpu'),info['threshold']))):
            observer=MotionObserver();bridge=PIBTMotion(geometry,LIBRARY)
            policy=BusinessPolicy(name,geometry,business.rules,observer,PIBTMovementModel(model,geometry,bridge),
                                  seed=20261109)
            sim=CoordinatedBusinessSimulator(business,'world',1,motion_backend='pibt',pibt_library=LIBRARY,
                                             collect_metrics=True)
            result=sim.run(dispatch_policy=policy,on_observation=observer.observe)
            self.assertEqual(result['summary']['status'],'completed')
            self.assertGreater(sim.pibt.solver_calls,0);self.assertGreater(bridge.solver_calls,0)
            self.assertEqual(policy.fallbacks,0)
            self.assertFalse(resource_audit(result,business.rules)['violations'])
            self.assertGreater(model.distribution.forward_calls if name=='cpsat_pets' else model.service_calls,0)


if __name__=='__main__':unittest.main()
