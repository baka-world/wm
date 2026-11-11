"""Real business-state RHWM-v1 contracts without private simulator future input."""
from __future__ import annotations

import os
import unittest

from business_transport.observation import BusinessObserver
from business_transport.simulation import BusinessSimulator
from business_transport.world_model import HybridSystemWorldModel
from business_transport.workflow_transition import check_business_transition
from experiments.business_constraints.tests.test_business import fixture
from lab_transport.decision_value import candidates_for, nearest_batch
from lab_transport.dynamics_model import Geometry
from lab_transport.transition_schema import Assignment


class RHWMV1Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, _, cls.business = fixture(capacity=1)
        cls.geometry = Geometry(cls.business.scenario.grid)
        factual = []
        BusinessSimulator(cls.business, "world", 11).run(on_transition=factual.append)
        cls.public = BusinessObserver().observe(factual[0].observation)

    @classmethod
    def continuation(cls, state):
        return nearest_batch(state, candidates_for(state), cls.geometry)

    def test_one_step_contains_all_public_business_entities(self):
        wm = HybridSystemWorldModel(self.geometry, self.business.rules, kind="rule")
        action = self.continuation(self.public)
        step = wm.predict_next(self.public, action, sampling_seed=33)
        self.assertEqual(step.next_state.tick, self.public.tick + 1)
        self.assertEqual({r.id for r in step.next_state.robots},
                         {r.id for r in self.public.robots})
        self.assertEqual({s.id for s in step.next_state.stations},
                         {s.id for s in self.public.stations})
        self.assertTrue(set(s.id for s in self.public.samples).issubset(
            {s.id for s in step.next_state.samples}))
        self.assertTrue(check_business_transition(
            self.public, step.next_state, self.geometry, self.business.rules, action))
        self.assertEqual(step.service_events, ())
        self.assertIn("future_unreleased_jobs", step.unknown_exogenous)

    def test_five_step_particles_provide_state_distributions(self):
        wm = HybridSystemWorldModel(self.geometry, self.business.rules, kind="rule")
        first = self.continuation(self.public)
        batch = wm.rollout(self.public, first, continuation_policy=self.continuation,
                           continuation_policy_name="nearest_after_first",
                           horizon_ticks=5, particles=3, sampling_seed=812)
        self.assertEqual(len(batch.particles), 3)
        self.assertEqual(len(batch.distributions), 5)
        for p in batch.particles:
            self.assertEqual(p.steps[0].assignments, first)
            self.assertEqual(p.final_state.tick, self.public.tick + 5)
            for step in p.steps:
                self.assertTrue(check_business_transition(
                    step.before, step.next_state, self.geometry,
                    self.business.rules, step.assignments))
        for summary in batch.distributions:
            for probabilities in summary.sample_status_probabilities.values():
                self.assertAlmostEqual(sum(probabilities.values()), 1.0)
            for probabilities in summary.robot_task_probabilities.values():
                self.assertAlmostEqual(sum(probabilities.values()), 1.0)
            self.assertEqual(set(summary.resource_occupancy_mean),
                             {r.id for r in self.business.rules.resources})
            self.assertLessEqual(summary.expected_waiting_samples,
                                 len(batch.particles[0].final_state.samples))
        second = wm.rollout(self.public, first, continuation_policy=self.continuation,
                            horizon_ticks=5, particles=3, sampling_seed=812)
        self.assertEqual(batch.distributions, second.distributions)

    def test_action_changes_full_system_assignment_not_just_score(self):
        wm = HybridSystemWorldModel(self.geometry, self.business.rules, kind="rule")
        legal = candidates_for(self.public)
        self.assertGreaterEqual(len(legal), 2)
        left = wm.predict_next(self.public, [legal[0]])
        right = wm.predict_next(self.public, [legal[-1]])
        self.assertNotEqual(
            tuple((r.id, r.task_id) for r in left.next_state.robots),
            tuple((r.id, r.task_id) for r in right.next_state.robots))

    def test_future_policy_is_explicit_and_budget_is_bounded(self):
        wm = HybridSystemWorldModel(self.geometry, self.business.rules)
        with self.assertRaisesRegex(ValueError, "continuation"):
            wm.rollout(self.public, (), continuation_policy=None)
        with self.assertRaisesRegex(ValueError, "horizon"):
            wm.rollout(self.public, (), continuation_policy=self.continuation,
                       horizon_ticks=21)
        with self.assertRaisesRegex(ValueError, "particles"):
            wm.rollout(self.public, (), continuation_policy=self.continuation,
                       particles=9)
        with self.assertRaises(ValueError):
            wm.predict_next(self.public, [Assignment("missing", "missing")])
        with self.assertRaises(ValueError):
            HybridSystemWorldModel(self.geometry, self.business.rules, kind="v4",
                                  checkpoint="not_a_model.pt", experimental=False)

    @unittest.skipUnless(os.environ.get("RHWM_V4_CHECKPOINT"),
                         "requires external frozen SHA-verified V4 checkpoint")
    def test_frozen_v4_uses_real_model_and_exposes_event_probabilities(self):
        import torch
        torch.set_num_threads(1)
        wm = HybridSystemWorldModel(
            self.geometry, self.business.rules,
            kind="v4", checkpoint=os.environ["RHWM_V4_CHECKPOINT"],
            experimental=True)
        first = self.continuation(self.public)
        result = wm.rollout(self.public, first,
                            continuation_policy=self.continuation,
                            horizon_ticks=5, particles=2, sampling_seed=17)
        events = [e for p in result.particles for step in p.steps
                  for e in step.service_events]
        self.assertGreater(len(events), 0, "model must actually be queried")
        for e in events:
            self.assertGreaterEqual(e.p_finish_next, 0)
            self.assertLessEqual(e.p_finish_next, 1)
            self.assertIsInstance(e.sampled_finished, bool)
        self.assertEqual(result.model_version, "rhwm-v1-hybrid-v4")


if __name__ == "__main__":
    unittest.main()
