"""Check validation selection is sealed and new online groups are unseen."""
from __future__ import annotations
import json
import os
from pathlib import Path
import unittest

from experiments.world_model_integration.posttraining.pr32_online_run import (
    PROTOCOL,ARMS,_no_seen_scenarios,eval_one
)


class PR32ValidationTests(unittest.TestCase):
    def test_scenarios_are_sealed_and_provenance_independent(self):
        cfg=json.loads(PROTOCOL.read_text())
        self.assertEqual(cfg["protocol"],"RHWM_PR32_FROZEN_POSTTRAINED_H20_ONLINE_20261112_V1")
        self.assertEqual(cfg["arms"],list(ARMS))
        self.assertEqual(len(cfg["groups"]),24)
        self.assertEqual({x["regime"] for x in cfg["groups"]},
                         {"light","medium","heavy","shift"})
        _no_seen_scenarios(cfg)
        self.assertEqual(cfg["particles"],3)
        self.assertEqual(cfg["decision_cpu_budget_seconds"],0.5)

    @unittest.skipUnless(all(os.environ.get(x) for x in (
            "RHWM_V4_ORIGINAL","RHWM_V4_SCRATCH","RHWM_V4_WARM")),
            "CI restores all immutable PR31 checkpoints")
    def test_real_five_arm_online_regression_has_no_unsafe_actions(self):
        cfg=json.loads(PROTOCOL.read_text())
        paths={"original_v4_h20":Path(os.environ["RHWM_V4_ORIGINAL"]),
               "warmstart_v4_h20":Path(os.environ["RHWM_V4_WARM"]),
               "scratch_v4_h20":Path(os.environ["RHWM_V4_SCRATCH"])}
        result=eval_one(cfg["groups"][0],cfg,paths)
        self.assertEqual(set(result["arms"]),set(ARMS))
        for r in result["arms"].values():
            self.assertEqual(r["safety_violations"],0)
            self.assertEqual(r["prediction_errors"],0)
            self.assertFalse(r["job_metrics"]["censored"])
            self.assertGreater(r["policy_calls"],0)


if __name__=="__main__":
    unittest.main()
