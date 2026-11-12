"""Read-only source-audit tests; use no real simulator or V4 network."""
import unittest
from experiments.world_model_integration.forecast_utility.pr43_readonly_audit import first_divergence,report

def episode(records, makespan=10):
    return {
        "decision_records":records,"policy_calls":len(records),
        "makespan_ticks":makespan,
        "job_metrics":{"late_jobs":0},
        "decision_budget_exceeded":0,"fallbacks":0,"changed_plan_calls":0,
    }

def rec(t,chosen,pool="P0",done_tie=False,distinct=True):
    return {
        "tick":t,"first_proposal":pool,"legal_plans":2,
        "selected":chosen,"selected_index":0 if chosen=="A" else 1,
        "completed_tie":done_tie,
        "future_state_different":distinct,
    }

class ReadOnlyAuditTests(unittest.TestCase):
    def test_same_prefix_first_change(self):
        ref=episode([rec(0,"A"),rec(1,"A")])
        alt=episode([rec(0,"A"),rec(1,"B",done_tie=True)])
        got=first_divergence(ref,alt)
        self.assertEqual(got["tick"],1)
        self.assertEqual(got["type"],"different_action_on_same_prefix")
        self.assertTrue(got["completion_tie_at_split"])

    def test_distinct_candidate_pool_is_not_same_state(self):
        ref=episode([rec(0,"A")])
        alt=episode([rec(0,"A",pool="P1")])
        self.assertEqual(first_divergence(ref,alt)["type"],
                         "candidate_pool_changed_before_action")

    def test_identical_action_history_zero_divergence(self):
        ref=episode([rec(0,"A"),rec(1,"B")])
        alt=episode([rec(0,"A"),rec(1,"B")])
        self.assertIsNone(first_divergence(ref,alt))

    def test_archive_semantics_reject_nonoriginal_inputs(self):
        with self.assertRaises(ValueError):
            report({"schema":"fabricated","groups":[]})


if __name__ == "__main__":
    unittest.main()
