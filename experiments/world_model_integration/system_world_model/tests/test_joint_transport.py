"""Use real V4 and published pgmpy to test two explicit transport risk modes."""
from __future__ import annotations
import json
import os
from pathlib import Path
import unittest

from business_transport.world_model import HybridSystemWorldModel
from business_transport.world_model.joint_transport import JointServiceTransportWorldModel
from business_transport.world_model.transport_bayes import TransportBayesModel
from experiments.world_model_integration.system_world_model.evaluate_horizons import (
    factual_episode,group_scenario,
)
from lab_transport.decision_value import candidates_for,nearest_batch


class JointTransportTests(unittest.TestCase):
    def test_needs_fitted_bayesian_source(self):
        from experiments.business_constraints.tests.test_business import fixture
        from lab_transport.dynamics_model import Geometry
        _,_,business=fixture()
        geo=Geometry(business.scenario.grid)
        with self.assertRaisesRegex(ValueError,"official pgmpy"):
            JointServiceTransportWorldModel(geo,business.rules,
                                             v4_checkpoint="unused",transport_model=None)

    @unittest.skipUnless(os.environ.get("RHWM_V4_CHECKPOINT") and
                         os.environ.get("RHWM_TRANSPORT_JSON"),
                         "CI restores both immutable sources; local optional")
    def test_joint_shadow_is_exact_v4_and_sampled_is_legal(self):
        from experiments.world_model_integration.transport_events.benchmark import PROTOCOL
        cfg=json.loads(PROTOCOL.read_text())
        spec=cfg["future_horizon_groups"][0]
        business,geo=group_scenario(spec,max_ticks=cfg["max_ticks"])
        states,actions,_=factual_episode(business,geo,spec["seed"])
        tmodel=TransportBayesModel.from_payload(json.loads(
            Path(os.environ["RHWM_TRANSPORT_JSON"]).read_text()))
        frozen=HybridSystemWorldModel(geo,business.rules,kind="v4",
            checkpoint=Path(os.environ["RHWM_V4_CHECKPOINT"]),experimental=True)
        joint=JointServiceTransportWorldModel(
            geo,business.rules,
            v4_checkpoint=Path(os.environ["RHWM_V4_CHECKPOINT"]),
            transport_model=tmodel)
        opts=dict(
            continuation_policy=lambda s:nearest_batch(s,candidates_for(s),geo),
            continuation_policy_name="nearest_public",
            horizon_ticks=10,particles=3,sampling_seed=77)
        baseline=frozen.rollout(states[0],actions[0],**opts)
        shadow=joint.rollout(states[0],actions[0],mode="shadow",**opts)
        self.assertEqual(baseline.distributions,shadow.forecast.distributions)
        self.assertTrue(shadow.transport_risks)
        self.assertEqual(shadow.applied_motion_holds,0)
        sampled=joint.rollout(states[0],actions[0],mode="sampled_hold",**opts)
        self.assertEqual(sampled.forecast.distributions[-1].tick,10)
        for part in sampled.forecast.particles:
            for step in part.steps:
                self.assertEqual(step.next_state.tick,step.before.tick+1)
        self.assertGreaterEqual(sampled.applied_motion_holds,0)
        self.assertTrue(all(0<=r.p_delay<=1 for r in sampled.transport_risks))


if __name__=="__main__":
    unittest.main()
