"""Synthetic records only: no environment, V4 or inference imports."""
import unittest

from experiments.world_model_integration.pr46_evidence_audit.pr46_first_divergence import (
    split_point, audit
)


def choice(t,plan,complete=True,first="P0",cnt=2):
    return {
        "tick":t, "first_proposal":first, "legal_plans":cnt,
        "selected":plan,"selected_index":0 if plan=="P0" else 1,
        "predicted_completed":[1.,2.] if complete else [],
    }


def episode(rows,makespan=100):
    return {"calls":len(rows),"decision_records":rows,"makespan_ticks":makespan,
            "job_metrics":{"late_jobs":0},"fallbacks":0,
            "budget_exceeded":0,"plan_flips":0}


class Pr46ReadonlyTests(unittest.TestCase):
    def test_first_common_prefix_action_disagreement(self):
        a=episode([choice(0,"P0"),choice(7,"P0")])
        b=episode([choice(0,"P0"),choice(7,"P1")])
        d=split_point(a,b)
        self.assertEqual(d["kind"],"same_prefix_different_selected_action")
        self.assertEqual(d["tick"],7)
        self.assertEqual(d["score_or_budget_class"],"both_completed_scoring")

    def test_first_divergence_can_be_budget_related(self):
        a=episode([choice(8,"P1",True)])
        b=episode([choice(8,"P0",False)])
        d=split_point(a,b)
        self.assertEqual(d["score_or_budget_class"],"original_complete_recovery_incomplete")

    def test_candidate_pool_difference_is_not_treated_as_common_state(self):
        a=episode([choice(7,"P0",first="P0")])
        b=episode([choice(7,"P1",first="P2")])
        self.assertEqual(
            split_point(a,b)["kind"],"candidate_pool_diverged_before_selected_action")

    def test_no_action_difference_no_retroactive_causality(self):
        a=episode([choice(0,"P0"),choice(7,"P0")])
        b=episode([choice(0,"P0"),choice(7,"P0")])
        self.assertIsNone(split_point(a,b))

    def test_rejects_nonoriginal_artifact(self):
        with self.assertRaises(ValueError):
            audit({"schema":"fake","groups":[]})


if __name__=="__main__":
    unittest.main()
