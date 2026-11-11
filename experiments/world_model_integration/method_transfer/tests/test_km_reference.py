"""Reference lifelines estimator, legal system dynamics, frozen data protocol."""
from __future__ import annotations

import json
import unittest

from business_transport.world_model import HybridSystemWorldModel
from business_transport.world_model.kaplan_meier import (
    KaplanMeierServiceModel, KaplanMeierRuleDynamics,
)
from experiments.world_model_integration.method_transfer.compare_km_v4 import (
    PROTOCOL, extract_sessions, group_scenario, factual_episode, event_brier_on_factual,
)
from lab_transport.decision_value import candidates_for, nearest_batch


class UpstreamReferenceTests(unittest.TestCase):
    def test_actual_lifelines_product_limit_estimator_not_reimplemented(self):
        model = KaplanMeierServiceModel([
            {"station_id":"A0","duration":2,"event_observed":True},
            {"station_id":"A0","duration":3,"event_observed":True},
        ])
        self.assertEqual(model.models["A0"].__class__.__name__, "KaplanMeierFitter")
        self.assertAlmostEqual(model.p_finish_next("A0", 0, 2.5), 0)
        self.assertAlmostEqual(model.p_finish_next("A0", 1, 2.5), 0.5)
        # Survival step: S(2)=0.5 and S(3)=0 -> hazard=1.
        self.assertAlmostEqual(model.p_finish_next("A0", 2, 2.5), 1)
        self.assertAlmostEqual(model.p_finish_next("unseen", 0, 2), 0)
        self.assertAlmostEqual(model.p_finish_next("unseen", 1, 2), 1)
        self.assertEqual(model.group_counts, {"A0":2})

    def test_censored_durations_are_supported_by_upstream_fitter(self):
        model = KaplanMeierServiceModel([
            {"station_id":"A","duration":2,"event_observed":False},
            {"station_id":"A","duration":3,"event_observed":True},
        ])
        # The right-censored row at time 2 is NOT treated as failure.
        self.assertAlmostEqual(model.p_finish_next("A",1,3),0.0)
        self.assertAlmostEqual(model.p_finish_next("A",2,3),0.0)
        self.assertAlmostEqual(model.p_finish_next("A",3-1,3),0.0)

    def test_public_event_log_extractor_and_missing_start_guard(self):
        events = [
            {"type":"service_started","tick":4,"robot_id":"R1","sample_id":"S","phase":"to_pickup","station_id":"A"},
            {"type":"service_finished","tick":7,"robot_id":"R1","sample_id":"S","phase":"to_pickup","station_id":"A","observed_ticks":3},
        ]
        self.assertEqual(extract_sessions(events,end_tick=8),[
            {"station_id":"A","duration":3,"event_observed":True}
        ])
        with self.assertRaisesRegex(AssertionError,"prior public start"):
            extract_sessions(events[1:],end_tick=8)

    def test_factual_state_end_to_end_under_library_probability(self):
        proto = json.loads(PROTOCOL.read_text())
        spec = proto["train_groups"][0]
        business, geo = group_scenario(spec,max_ticks=proto["max_ticks"])
        states, actions, actual = factual_episode(business,geo,spec["seed"])
        rows = extract_sessions(actual["events"],end_tick=actual["summary"]["ticks"])
        self.assertGreater(len(rows),0)
        model = HybridSystemWorldModel(
            geo,business.rules,kind="kaplan_meier",
            service_baseline=KaplanMeierServiceModel(rows))
        first = states[0]
        forecast = model.rollout(
            first,actions[0],
            continuation_policy=lambda s: nearest_batch(s,candidates_for(s),geo),
            continuation_policy_name="public_nearest",
            horizon_ticks=5,particles=3,sampling_seed=41)
        self.assertEqual(forecast.model_version,"rhwm-v1-km-reference")
        self.assertEqual(len(forecast.particles),3)
        self.assertEqual(forecast.particles[0].final_state.tick,5)
        for particle in forecast.particles:
            self.assertTrue(all(step.next_state.tick==step.before.tick+1
                                for step in particle.steps))
        event_score = event_brier_on_factual(model,states,actions)
        self.assertGreater(event_score["eligible_queries"],0)
        self.assertGreaterEqual(event_score["brier"],0)
        self.assertLessEqual(event_score["brier"],1)

    def test_frozen_groups_are_unique_and_outside_previous_holdout(self):
        proto = json.loads(PROTOCOL.read_text())
        old = json.loads((
            PROTOCOL.parent.parent / "system_world_model" /
            "generalization_protocol.json").read_text())
        a = {x["seed"] for x in proto["train_groups"]}
        b = {x["seed"] for x in proto["test_groups"]}
        c = {x["seed"] for x in old["groups"]}
        self.assertTrue(a.isdisjoint(b))
        self.assertTrue(b.isdisjoint(c))
        self.assertTrue(a.isdisjoint(c))
        self.assertEqual(len(b),6)


if __name__ == "__main__":
    unittest.main()
