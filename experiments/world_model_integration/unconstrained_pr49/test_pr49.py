"""Frozen high-compute model-quality protocol, no CPU latency optimization."""
from __future__ import annotations
from pathlib import Path
import json
import unittest
from unittest.mock import patch

from experiments.world_model_integration.unconstrained_pr49.pr49_online import (
    PROTOCOL,REGIMES,ARMS,validate,summarize,evaluate_case
)


class Pr49HighComputeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p=json.loads(PROTOCOL.read_text())

    def test_48_new_balanced_seeds_and_same_original_world_mechanics(self):
        self.assertTrue(validate(self.p))
        self.assertEqual(len(self.p["cohort"]),48)
        self.assertEqual(len(set(x["seed"] for x in self.p["cohort"])),48)
        self.assertEqual(set(REGIMES),set(x["regime"] for x in self.p["cohort"]))
        self.assertEqual(self.p["horizon_ticks"],20)
        self.assertEqual(self.p["particles"],3)
        self.assertEqual(self.p["candidate_limit"],4)

    def test_30_second_guard_is_identical_in_both_world_models(self):
        self.assertEqual(self.p["process_cpu_safety_guard_seconds"],30)
        self.assertEqual(self.p["fixed_per_decision_cpu_process_budget_seconds"],30)
        self.assertEqual(self.p["arms"],list(ARMS))
        self.assertEqual(self.p["resources"]["gpu_training"],0)
        self.assertEqual(self.p["resources"]["neural_training"],0)

    def test_online_source_receives_identical_ample_budget_without_changed_score(self):
        calls=[]
        def fake_episode(spec,proto,arm,checkpoint,km):
            calls.append((arm,proto["fixed_per_decision_cpu_process_budget_seconds"],
                          proto["horizon_ticks"],proto["particles"],proto["candidate_limit"]))
            return {"makespan_ticks":80,"model_errors":0}
        with patch("experiments.world_model_integration.unconstrained_pr49.pr49_online.real_episode",
                   side_effect=fake_episode):
            e=evaluate_case(self.p["cohort"][0],self.p,"ckpt","km")
        self.assertEqual(set(e["arms"]),set(ARMS))
        self.assertEqual({x[1] for x in calls},{30})
        self.assertEqual({x[2] for x in calls},{20})
        self.assertEqual({x[3] for x in calls},{3})
        self.assertEqual({x[4] for x in calls},{4})

    def test_no_claim_full_scores_if_any_proposal_fallback(self):
        def arm(m,fall):
            return {"makespan_ticks":m,"budget_overruns":fall,"fallbacks":fall,
                    "job_metrics":{"late_jobs":0,"censored":False},
                    "safety_violations":0,"full_episode_cpu_seconds":1.}
        row={"arms":{"cp_sat":arm(102,0),"v4_original_h20_full":arm(100,0),
                     "v4_recovery_h20_full":arm(99,1)}}
        s=summarize([row])
        self.assertEqual(s["wins"],1)
        self.assertFalse(s["all_candidates_fully_scored_in_both_models"])
        row["arms"]["v4_recovery_h20_full"]["budget_overruns"]=0
        row["arms"]["v4_recovery_h20_full"]["fallbacks"]=0
        self.assertTrue(summarize([row])["all_candidates_fully_scored_in_both_models"])

    def test_rejects_test_seed_reuse_or_mutated_model_horizon(self):
        altered=json.loads(json.dumps(self.p))
        altered["cohort"][0]["seed"]=altered["cohort"][1]["seed"]
        with self.assertRaises(ValueError):
            validate(altered)
        altered=json.loads(json.dumps(self.p))
        altered["particles"]=5
        with self.assertRaises(ValueError):
            validate(altered)


if __name__=="__main__":
    unittest.main()
