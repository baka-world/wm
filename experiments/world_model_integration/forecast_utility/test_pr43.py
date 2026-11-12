"""Unit and contract checks for PR43 frozen utility comparison."""
from __future__ import annotations
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiments.world_model_integration.forecast_utility.pr43_online import (
    ARMS, PROTOCOL, check_protocol, h20_state_distinct,
    score_batch, select_scores, tied_completed
)


class FrozenUtilityTests(unittest.TestCase):
    def test_original_score_keeps_first_on_tie(self):
        x = [{"completed": 2., "unfinished_late_cost": 4., "terminal_distance": 7.},
             {"completed": 2., "unfinished_late_cost": 3., "terminal_distance": 1.}]
        self.assertEqual(select_scores(x, "v4_completed"), 0)
        self.assertEqual(select_scores(x, "v4_tie_trajectory"), 1)
        self.assertEqual(select_scores(x, "v4_trajectory"), 1)
        self.assertTrue(tied_completed(x))

    def test_tie_break_preserves_primary_complete(self):
        x = [{"completed": 2., "unfinished_late_cost": 30., "terminal_distance": 99.},
             {"completed": 1.66666666667, "unfinished_late_cost": 1., "terminal_distance": 0.}]
        self.assertEqual(select_scores(x, "v4_tie_trajectory"), 0)
        self.assertEqual(select_scores(x, "v4_trajectory"), 1)
        self.assertFalse(tied_completed(x))

    def test_tie_break_and_old_score_same_when_no_tie(self):
        x = [{"completed": 3., "unfinished_late_cost": 10., "terminal_distance": 5.},
             {"completed": 4., "unfinished_late_cost": 20., "terminal_distance": 1.}]
        self.assertEqual(select_scores(x, "v4_completed"), 1)
        self.assertEqual(select_scores(x, "v4_tie_trajectory"), 1)

    def test_original_pr27_cost_is_called_without_changes(self):
        batch = SimpleNamespace(distributions=[
            SimpleNamespace(expected_completed_samples=2.333333333)])
        with patch("experiments.world_model_integration.forecast_utility.pr43_online._trace_cost",
                   return_value=(11., 4.)) as mock:
            row = score_batch("public-state", batch, "public-geometry")
        self.assertEqual(row["completed"], 2.333333333)
        self.assertEqual(row["unfinished_late_cost"], 11.)
        self.assertEqual(row["terminal_distance"], 4.)
        mock.assert_called_once_with("public-state", batch, "public-geometry")

    def test_full_state_distinct_independent_of_completed_count(self):
        d1 = SimpleNamespace(expected_completed_samples=3.,
                             expected_waiting_samples=2., resource_occupancy_mean={"a": 1.})
        d2 = SimpleNamespace(expected_completed_samples=3.,
                             expected_waiting_samples=1., resource_occupancy_mean={"a": 1.})
        self.assertTrue(h20_state_distinct(
            [SimpleNamespace(distributions=[d1]),
             SimpleNamespace(distributions=[d2])]))
        self.assertFalse(h20_state_distinct(
            [SimpleNamespace(distributions=[d1]),
             SimpleNamespace(distributions=[d1])]))

    def test_new_24_scenes_and_four_frozen_arms(self):
        p = json.loads(PROTOCOL.read_text())
        self.assertEqual(len(p["groups"]), 24)
        self.assertEqual(p["arms"], list(ARMS))
        self.assertEqual(len(set(g["seed"] for g in p["groups"])), 24)
        self.assertEqual(
            {k: sum(g["regime"] == k for g in p["groups"])
             for k in ("light", "medium", "heavy", "shift")},
            {"light": 6, "medium": 6, "heavy": 6, "shift": 6})
        self.assertEqual(p["decision_cpu_budget_seconds"], .5)
        self.assertEqual(p["particles"], 3)
        self.assertEqual(p["horizon"], 20)
        self.assertEqual(p["constraints"]["new_training"], 0)
        self.assertEqual(p["constraints"]["gpu"], 0)


if __name__ == "__main__":
    unittest.main()
