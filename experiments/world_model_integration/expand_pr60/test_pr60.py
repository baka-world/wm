"""PR60 frozen incremental 36 truthful TRAIN cases, never TEST or GPU."""
import unittest
from collections import Counter
import json

from experiments.world_model_integration.expand_pr60.pr60_source import protocol
from experiments.world_model_integration.sealed_pr59.pr59_source import validate_layout_business
from experiments.world_model_integration.stress_pr48.pr48_scenarios import PROTOCOL as STRESS_PROTOCOL


class NewTrainingInformationProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p,cls.old=protocol()
        cls.stress=json.loads(STRESS_PROTOCOL.read_text())

    def test_36_new_distinct_train_only_with_no_val_test_changes(self):
        p=self.p
        self.assertEqual(len(p["cohort"]),36)
        self.assertEqual(p["combined_train_count"],60)
        self.assertEqual(p["new_blind_test_count"],0)
        self.assertEqual(p["new_validation_count"],0)
        self.assertEqual(p["GPU_use"],0)
        self.assertEqual({c["split"] for c in p["cohort"]},{"train"})
        old_seeds={x["seed"] for x in self.old["cohort"]}
        new_seeds={x["seed"] for x in p["cohort"]}
        self.assertFalse(old_seeds&new_seeds)
        self.assertEqual(len(new_seeds),36)

    def test_profile_and_map_stratification_before_data_access(self):
        self.assertEqual(set(Counter(c["regime"] for c in self.p["cohort"]).values()),{12})
        self.assertEqual(set(Counter(c["profile"] for c in self.p["cohort"]).values()),{6})
        self.assertEqual(set(Counter(c["layout"] for c in self.p["cohort"]).values()),{6})
        self.assertEqual({c["map_family"] for c in self.p["cohort"]},
                         {"vertical_blocked_route_training"})
        self.assertNotEqual(self.p["validation_family_maps_fixed"],
                            self.p["blind_test_family_maps_frozen_unexecuted"])

    def test_every_new_source_map_is_real_safe_navigable_public_business_grid(self):
        for c in self.p["cohort"]:
            topo,detours=validate_layout_business(c,self.old,self.stress)
            self.assertEqual(topo["schema"],"pr56-public-grid-cells-v1")
            self.assertGreater(detours,0)
            self.assertEqual(len(topo["walkable_cells"])+len(topo["blocked_cells"]),35)

    def test_future_action_quality_gate_is_quantitative_and_not_result_tuned(self):
        gate=self.p["high_quality_additional_train_precommitted_gate"]
        self.assertGreaterEqual(gate["minimum_marginal_semantic_novelty_fraction_vs_original_train"],0.5)
        self.assertGreaterEqual(gate["minimum_36_true_H20_mean_state_L1_at_least_0_01_scenes"],18)
        self.assertLessEqual(gate["maximum_36_censored_fraction"],0.25)
        self.assertTrue(gate["blind_test_never_generated"])


if __name__=="__main__":
    unittest.main()
