"""PR53 no leakage and genuinely fresh predeclared counterfactual labels."""
from __future__ import annotations
import json
import unittest

from experiments.world_model_integration.action_pr53.pr53_labels import (
    PROTOCOL,validate
)


class PR53LabelsTest(unittest.TestCase):
    def test_228_new_seeds_and_independent_splits(self):
        p=json.loads(PROTOCOL.read_text())
        self.assertTrue(validate(p))
        self.assertEqual(p["train_val_test_splits"],
                         {"train":144,"validation":36,"test":48})
        self.assertEqual(len(p["cohort"]),228)
        self.assertEqual(len({x["seed"] for x in p["cohort"]}),228)
        self.assertTrue(all(x["seed"]>=43000000 for x in p["cohort"]))

    def test_fixed_action_contrast_and_original_score(self):
        p=json.loads(PROTOCOL.read_text())
        self.assertEqual(p["counterfactual_max_candidates"],4)
        self.assertEqual(p["prediction_horizons"],[5,20])
        self.assertEqual(p["train"]["seeds"],[1810,1811,1812])
        self.assertTrue(all("decision_weight" in a and "contrast_weight" in a
                            for a in p["train_arms"]))
        self.assertIn("NO trainable ranker",p["score_rule"])

    def test_test_cohort_cannot_be_reassigned_to_train(self):
        p=json.loads(PROTOCOL.read_text())
        p["cohort"][0]["split"]="test"
        with self.assertRaises(ValueError):
            validate(p)


if __name__=="__main__":
    unittest.main()
