"""Tests for 12 independent-scenario stratified PR54 readonly inference."""
import unittest

from experiments.world_model_integration.temporal_pr54.pr54_readonly import (
    scenario_bootstrap,interval
)


class ReadonlyTemporalSceneTests(unittest.TestCase):
    def test_identical_scene_improvement_has_exact_interval(self):
        rows=[{"regime":r,"delta_node_mse":-0.002}
              for r in ("congestion","multi_fault","staggered_arrivals")
              for _ in range(4)]
        x=scenario_bootstrap(rows,"delta_node_mse",replicas=30)
        self.assertAlmostEqual(x["difference_history_minus_snapshot"],-0.002)
        self.assertAlmostEqual(x["scenario_stratified_95pct_CI"][0],-0.002)
        self.assertEqual((x["wins"],x["ties"],x["losses"]),(12,0,0))

    def test_cannot_pseudo_replicate_model_seeds_as_independent_scenes(self):
        rows=[{"regime":"congestion","delta_node_mse":0.}]
        with self.assertRaises(ValueError):
            scenario_bootstrap(rows,"delta_node_mse",replicas=10)

    def test_interpolated_ci(self):
        self.assertEqual(interval([1,2,3,4,5]),[1.1,4.9])


if __name__=="__main__":
    unittest.main()
