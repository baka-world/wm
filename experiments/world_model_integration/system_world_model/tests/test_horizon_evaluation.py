"""Protocol and factual-vs-forecast tests, independent of V4 checkpoint."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import unittest

from business_transport.world_model import HybridSystemWorldModel
from lab_transport.decision_value import candidates_for, nearest_batch
from experiments.world_model_integration.system_world_model.evaluate_horizons import (
    PROTOCOL, METRICS, aggregate, factual_episode, group_scenario, score,
)


class HorizonEvaluationTests(unittest.TestCase):
    def test_group_protocol_frozen_and_disjoint(self):
        p = json.loads(PROTOCOL.read_text())
        self.assertEqual(p["horizons"], [1, 5, 10])
        self.assertEqual(p["anchors"], [0, 5, 12, 20, 30])
        self.assertEqual(p["particles"], 3)
        self.assertEqual(len(p["groups"]), 6)
        self.assertEqual(len({x["seed"] for x in p["groups"]}), 6)
        self.assertEqual(len({x["id"] for x in p["groups"]}), 6)
        self.assertNotIn(906195, {x["seed"] for x in p["groups"]})

    def test_factual_observations_and_state_metrics(self):
        p = json.loads(PROTOCOL.read_text())
        group = p["groups"][0]
        business, geo = group_scenario(group, max_ticks=p["max_ticks"])
        observations, actions, result = factual_episode(business, geo, group["seed"])
        self.assertEqual(result["summary"]["status"], "completed")
        self.assertIn(0, observations)
        self.assertIn(0, actions)
        self.assertIn(10, observations)
        self.assertEqual({s.job_id for s in observations[0].samples}, {"J1", "J2"})

        wm = HybridSystemWorldModel(geo, business.rules, kind="rule")
        batch = wm.rollout(
            observations[0], actions[0],
            continuation_policy=lambda s: nearest_batch(s, candidates_for(s), geo),
            horizon_ticks=10, particles=3, sampling_seed=1810)
        measured = score(batch, observations[10], 10)
        self.assertEqual(set(measured), set(METRICS))
        self.assertTrue(all(isinstance(value, float) and value >= 0
                            for value in measured.values()))
        self.assertLessEqual(measured["sample_status_error"], 1)
        self.assertLessEqual(measured["station_busy_brier"], 1)
        for part in batch.particles:
            self.assertEqual(len(part.steps), 10)
            self.assertEqual(part.final_state.tick, 10)

    def test_distribution_comparison_detects_factual_state_difference(self):
        p = json.loads(PROTOCOL.read_text())
        spec = p["groups"][0]
        business, geo = group_scenario(spec, max_ticks=p["max_ticks"])
        states, actions, _ = factual_episode(business, geo, spec["seed"])
        wm = HybridSystemWorldModel(geo, business.rules)
        batch = wm.rollout(states[0], actions[0],
                           continuation_policy=lambda s: nearest_batch(
                               s, candidates_for(s), geo),
                           horizon_ticks=1, particles=3)
        truth = states[1]
        wrong = replace(truth, samples=tuple(
            replace(s, status="completed") if s.status != "completed" else s
            for s in truth.samples))
        self.assertGreaterEqual(score(batch, wrong, 1)["sample_status_error"],
                                score(batch, truth, 1)["sample_status_error"])

    def test_group_level_aggregation_is_not_particle_pseudoreplication(self):
        zero = {key:0.0 for key in METRICS}
        one = {key:1.0 for key in METRICS}
        groups = [
            {"arms": {"rule":{"rows":[{"horizon":1,"metrics":zero}]},
                      "v4":{"rows":[{"horizon":1,"metrics":one}]}}},
            {"arms": {"rule":{"rows":[{"horizon":1,"metrics":one}]},
                      "v4":{"rows":[{"horizon":1,"metrics":zero}]}}},
        ]
        a = aggregate(groups, [1])["1"]
        self.assertEqual(a["rule"]["scenario_groups"],2)
        self.assertEqual(a["rule"]["metrics"]["waiting_count_mae"],.5)
        self.assertEqual(a["v4_minus_rule"]["waiting_count_mae"],0.0)


if __name__ == "__main__":
    unittest.main()
