"""PR47 frozen-budget, online comparator fairness, protocol and independent-seed tests."""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.world_model_integration.recovery_budget.pr47_online import (
    PROTOCOL, ARMS, MAPPED, validate, run_arm, summarized
)


class EqualBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.proto=json.loads(PROTOCOL.read_text())

    def test_frozen_two_budget_values_and_no_new_weights(self):
        self.assertTrue(validate(self.proto))
        self.assertEqual(self.proto["cpu_budgets_seconds"],[0.5,1.5])
        self.assertEqual(self.proto["online"]["horizon"],20)
        self.assertEqual(self.proto["online"]["particles"],3)
        self.assertEqual(self.proto["online"]["candidate_limit"],4)
        self.assertEqual(self.proto["resource_limits"]["gpu"],0)
        self.assertEqual(self.proto["resource_limits"]["new_training"],0)

    def test_equal_cpu_budget_for_both_world_models(self):
        self.assertEqual(MAPPED["v4_original_05"][1],MAPPED["v4_recovery_05"][1])
        self.assertEqual(MAPPED["v4_original_15"][1],MAPPED["v4_recovery_15"][1])
        self.assertEqual(MAPPED["v4_original_15"][1],1.5)
        self.assertEqual(MAPPED["v4_recovery_15"][1],1.5)
        self.assertEqual(len(ARMS),5)

    def test_new_cohort_stratified_no_previous_seeds(self):
        self.assertEqual(len(self.proto["cohort"]),24)
        self.assertEqual(len({s["seed"] for s in self.proto["cohort"]}),24)
        for mech in ("within_support","extended_outage"):
            self.assertEqual(sum(s["mechanism"]==mech for s in self.proto["cohort"]),12)

    def test_only_budget_is_replaced_at_runtime(self):
        sample={"id":"s","mechanism":"within_support","station_id":"B0",
                "seed":111,"track":"online"}
        captured=[]
        def fake_one(spec,proto,method,checkpoint,km):
            captured.append((method,proto["online"]["decision_cpu_budget_seconds"],
                             proto["online"]["horizon"],proto["online"]["particles"]))
            return {"safety_violations":0}
        with patch("experiments.world_model_integration.recovery_budget.pr47_online.one",
                   side_effect=fake_one):
            one=run_arm(sample,self.proto,"v4_original_15","ckpt","km")
            two=run_arm(sample,self.proto,"v4_recovery_15","ckpt","km")
        self.assertEqual([x[1] for x in captured],[1.5,1.5])
        self.assertEqual([x[2] for x in captured],[20,20])
        self.assertEqual([x[3] for x in captured],[3,3])
        self.assertEqual(one["policy_id"],"v4_original_15")
        self.assertEqual(two["policy_id"],"v4_recovery_15")
        self.assertEqual(self.proto["online"]["decision_cpu_budget_seconds"],.5)

    def test_rejects_modified_budget_or_fake_cohort(self):
        changed=json.loads(json.dumps(self.proto))
        changed["cpu_budgets_seconds"]=[.5,2.]
        with self.assertRaises(ValueError):validate(changed)
        other=json.loads(json.dumps(self.proto))
        other["cohort"][0]["seed"]=other["cohort"][1]["seed"]
        with self.assertRaises(ValueError):validate(other)


if __name__=="__main__":
    unittest.main()
