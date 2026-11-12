"""PR48 read-only stress first divergence contract tests."""
from __future__ import annotations

import unittest
from experiments.world_model_integration.stress_pr48.pr48_readonly_stress import (
    analyze, REGIMES
)


def instance(seed,regime):
    original={"decision_calls":1,"decision_records":[{
        "tick":8,"legal_plans":2,"first_proposal":"P0","selected":"P1",
        "selected_index":1,"predicted_completed":[0.,1.]
    }],"makespan_ticks":80,"job_metrics":{"late_jobs":0},
        "budget_overruns":0,"fallbacks":0,"changed_proposals":1}
    changed={"decision_calls":1,"decision_records":[{
        "tick":8,"legal_plans":2,"first_proposal":"P0","selected":"P0",
        "selected_index":0,"predicted_completed":[2.,1.]
    }],"makespan_ticks":81,"job_metrics":{"late_jobs":0},
        "budget_overruns":1,"fallbacks":1,"changed_proposals":0}
    return {"id":str(seed),"seed":seed,"regime":regime,
            "mechanism":"within_support" if seed%2==0 else "extended_outage",
            "station_id":"B0",
            "arms":{"cp_sat":{},"v4_original_15":original,"v4_recovery_15":changed}}


def originals():
    return {regime:{
        "schema":"pr48-full-online-stress-independent-v1",
        "regime":regime,"new_gpu":0,"new_training":0,
        "protocol_sha256":"test1","v4_checkpoint_sha256":"test2",
        "pr44_train_only_model_sha256":"test3",
        "groups":[instance(1000+i+16*j,regime) for i in range(16)],
        "summary":{"actual_episodes":48}
    } for j,regime in enumerate(REGIMES)}


class Pr48StressReadonlyTests(unittest.TestCase):
    def test_first_difference_budget_vs_complete_scoring_distinguished(self):
        x=analyze(originals())
        self.assertEqual(x["summary"]["overall"]["groups"],48)
        self.assertEqual(x["summary"]["overall"]["first_changed_choice_scoring_class"],
                         {"original_complete_recovery_incomplete":48})
        self.assertTrue(x["zero_new_simulation"])

    def test_missing_stress_regime_is_not_acceptable(self):
        x=originals()
        del x["multi_fault"]
        with self.assertRaises(ValueError):
            analyze(x)

    def test_duplicate_public_seed_or_unrecognized_artifact_rejected(self):
        x=originals()
        x["multi_fault"]["groups"][0]["seed"]=1000
        with self.assertRaises(ValueError):
            analyze(x)
        x=originals()
        x["congestion"]["schema"]="fake"
        with self.assertRaises(ValueError):
            analyze(x)


if __name__=="__main__":
    unittest.main()
