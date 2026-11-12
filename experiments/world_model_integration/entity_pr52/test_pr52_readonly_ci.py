"""PR52 true independent scenario-level bootstrap inference contracts."""
from __future__ import annotations
import unittest

from experiments.world_model_integration.entity_pr52.pr52_readonly_ci import (
    paired_bootstrap, interval
)


class ReadonlyRegretAuditTests(unittest.TestCase):
    def test_three_strata_sampling_respects_one_scenario_as_one_unit(self):
        rows=[]
        for r in ("congestion","multi_fault","staggered_arrivals"):
            for i in range(4):
                rows.append({
                    "regime":r,
                    "mean_two_training_seeds_regret":{
                        "rel_entity_pred_only":4.,
                        "rel_entity_joint025":3.,
                        "cp_sat_first":5.
                    }
                })
        result=paired_bootstrap(rows,"rel_entity_joint025","rel_entity_pred_only",
                                reps=101)
        self.assertEqual(result["mean_delta_true_regret_ticks"],-1.)
        self.assertEqual(result["bootstrap_95pct_CI_ticks"],[-1.,-1.])
        self.assertEqual(result["wins"],12)
        self.assertIn("scenario seeds",result["unit"])

    def test_reject_unequal_strata_to_prevent_biased_resampling(self):
        rows=[{"regime":"congestion","mean_two_training_seeds_regret":{
            "rel_entity_pred_only":2.,"rel_entity_joint025":3.}}]
        with self.assertRaises(ValueError):
            paired_bootstrap(rows,"rel_entity_joint025","rel_entity_pred_only")

    def test_fixed_quantile_interpolation(self):
        self.assertEqual(interval([1,2,3,4,5],.25,.75),[2.,4.])


if __name__=="__main__":
    unittest.main()
