"""PR48 read-only audit exercises without real simulator or model inference."""
from __future__ import annotations
import unittest
from experiments.world_model_integration.stress_pr48.pr48_readonly_pr47 import analyze


def row(same=True, budget=0):
    original_plan={"tick":8,"legal_plans":2,"first_proposal":"P0",
                   "selected":"P0","selected_index":0,"predicted_completed":[2.,3.]}
    recovery_plan={**original_plan,
                   "selected":"P0" if same else "P1",
                   "selected_index":0 if same else 1}
    original={"calls":1,"decision_records":[original_plan],
              "budget_exceeded":0,"fallbacks":0,"makespan_ticks":75,
              "job_metrics":{"late_jobs":0},"plan_flips":0}
    repair={"calls":1,"decision_records":[recovery_plan],
            "budget_exceeded":budget,"fallbacks":budget,"makespan_ticks":74,
            "job_metrics":{"late_jobs":0},"plan_flips":int(not same),
            "sampled_repair_events":3}
    return {"id":"case","seed":1,"mechanism":"within_support",
            "station_id":"B0",
            "arms":{"v4_original_15":original,"v4_recovery_15":repair}}


def fake_data():
    x=row(same=False)
    y=row(same=True)
    return {"schema":"pr47-equal-budget-closed-loop-new-cohort-v1",
            "summary":{"scenes":24,"total_episodes":120},
            "gpu":0,"no_new_training":True,
            "scenes":[{**x,"seed":i,"mechanism":"within_support"}
                      for i in range(12)]+
                     [{**y,"seed":i+12,"mechanism":"extended_outage"}
                      for i in range(12)]}


class Pr48ReadOnlyTests(unittest.TestCase):
    def test_detects_completed_scoring_changes_with_common_candidates(self):
        result=analyze(fake_data())
        group=result["summary"]["within_support"]
        self.assertEqual(group["groups"],12)
        self.assertEqual(group["first_difference_types"]["same_prefix_different_selected_action"],12)
        self.assertEqual(group["different_full_scoring_class"]["both_completed_scoring"],12)
        self.assertTrue(result["zero_new_inference"])

    def test_rejects_incomplete_candidate_scoring_budget_history(self):
        data=fake_data()
        data["scenes"][0]["arms"]["v4_recovery_15"]["budget_exceeded"]=1
        with self.assertRaises(AssertionError):
            analyze(data)

    def test_rejects_wrong_or_partial_archive(self):
        data=fake_data()
        data["schema"]="made-up"
        with self.assertRaises(ValueError):
            analyze(data)
        data=fake_data()
        data["scenes"]=data["scenes"][:-1]
        with self.assertRaises(ValueError):
            analyze(data)


if __name__=="__main__":
    unittest.main()
