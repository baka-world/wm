"""Real official PIBT/LaCAM3 backend integration for v2 session predictions."""
import os
import unittest
from dataclasses import replace

from business_transport.dynamics import WorkflowDynamics
from business_transport.lacam3_integration import (HybridBusinessSimulator,
    RecoveryMovementModel,RecoveryObserver)
from business_transport.session_dynamics import SessionRuleDynamics
from business_transport.workflow_transition import check_business_transition
from experiments.workflow_prediction.tests.test_workflow import case
from lab_transport.decision_value import candidates_for,nearest_batch
from lab_transport.dynamics_model import Geometry


@unittest.skipUnless(all(os.environ.get(x) for x in
    ("WM_SESSION_CHECKPOINT","WM_PIBT_LIBRARY","WM_LACAM3_WORKER")),
    "requires fixed official motion backends and trained CPU checkpoint")
class SessionMotionIntegrationTests(unittest.TestCase):
    def test_session_model_uses_real_motion_and_isolates_lacam_cache(self):
        business=case(2,1,2)
        # The standard one-tick test services finish immediately. Make this
        # synthetic integration fixture long enough for *observed active*
        # sessions, otherwise a legitimate session predictor has no query.
        scenario=business.scenario
        stations={sid:replace(station,service_ticks=4,prior_ticks=4,jitter_ticks=0)
                  for sid,station in scenario.stations.items()}
        business=replace(business,scenario=replace(scenario,stations=stations))
        geometry=Geometry(business.scenario.grid)
        pibt=os.environ["WM_PIBT_LIBRARY"]
        lacam=os.environ["WM_LACAM3_WORKER"]
        obs=RecoveryObserver()
        rows=[]
        sim=HybridBusinessSimulator(business,"world",1,motion_backend="hybrid",
            pibt_library=pibt,lacam3_worker=lacam,collect_metrics=True)
        sim.run(dispatch_policy=lambda state,c:nearest_batch(state,c,geometry),
                on_observation=obs.observe,on_transition=rows.append)
        observations=RecoveryObserver()
        eligible=[(observations.observe(row.observation),row.assignments) for row in rows[:25]]
        self.assertTrue(eligible)
        underlying=SessionRuleDynamics(geometry,business.rules,os.environ["WM_SESSION_CHECKPOINT"],
                                       experimental=True,seed=1810)
        motion=RecoveryMovementModel(underlying,geometry,business.rules,pibt,lacam)
        model=WorkflowDynamics(motion,business.rules)
        for observed,assignments in eligible:
            snapshot=observed.recovery_snapshot
            model.reset_seed(1810+observed.tick)
            first=model.predict_next(observed,assignments)
            self.assertTrue(check_business_transition(observed,first,geometry,business.rules,assignments))
            self.assertEqual(observed.recovery_snapshot,snapshot)
            model.reset_seed(1810+observed.tick)
            second=model.predict_next(observed,assignments)
            self.assertEqual(first,second)
            self.assertEqual(observed.recovery_snapshot,snapshot)
            self.assertTrue(all(s.resource_occupied<=s.resource_capacity for s in first.stations))
        self.assertGreater(underlying.query_calls,0)
