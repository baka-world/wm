"""PR53 preregistered independent-scenario bootstrap does not pseudo-replicate candidates."""
from __future__ import annotations
import unittest
from experiments.world_model_integration.action_pr53.pr53_readonly import bootstrap

class PR53ReadonlyBootstrapTest(unittest.TestCase):
    def fake(self):
        rows=[]
        for regime in ("congestion","multi_fault","staggered_arrivals"):
            for idx in range(16):
                rows.append({
                    "regime":regime,
                    "mean_three_seed_regret":{
                        "consistent_pred_only":3.,
                        "consistent_delta_joint":2.,
                        "consistent_delta_value":2.5,
                        "cp_sat_first":4.
                    }})
        return rows
    def test_true_scenario_unit_and_fixed_regime_strata(self):
        rows=self.fake()
        r=bootstrap(rows,"consistent_delta_joint","consistent_pred_only",101)
        self.assertEqual(r["true_regret_delta_ticks"],-1.)
        self.assertEqual(r["stratified_scenario_bootstrap_95pct"],[-1.,-1.])
        self.assertEqual((r["wins"],r["ties"],r["losses"]),(48,0,0))
        self.assertIn("48 independent true-world scenarios",r["inference_unit"])
    def test_near_tie_envelope_is_diagnostic_not_a_new_policy(self):
        from experiments.world_model_integration.action_pr53.pr53_readonly import (
            numerical_ambiguity,ARMS,SEEDS)
        row={"arm_seed":{
            arm:{str(seed):{
                "predicted_score":[0.,1e-6,1.],
                "true_costs":[20.,10.,30.]
            } for seed in SEEDS} for arm in ARMS}}
        outcome=numerical_ambiguity([row])
        for arm in ARMS:
            for seed in SEEDS:
                close=outcome[arm][str(seed)]["1e-05"]
                self.assertEqual(close["ambiguous_groups"],1)
                self.assertEqual(close["ambiguous_groups_with_different_true_makespan"],1)
                self.assertEqual(close["best_possible_regret_within_margin"],0.)
                self.assertEqual(close["worst_possible_regret_within_margin"],10.)

    def test_missing_original_test_case_is_rejected(self):
        with self.assertRaises(ValueError):
            bootstrap(self.fake()[:-1],"consistent_delta_joint","consistent_pred_only",11)
    def test_two_seeds_cannot_be_miscounted_as_scenes(self):
        with self.assertRaises(ValueError):
            bootstrap(self.fake()*3,"consistent_delta_joint","consistent_pred_only",11)

if __name__=="__main__":
    unittest.main()
