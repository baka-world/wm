"""Real simulator integration checks. The CI environment supplies original V4 weights."""
from __future__ import annotations
import json
import os
from pathlib import Path
import unittest
from experiments.world_model_integration.urgent_workflow_feasibility.core import public_spec, PROTOCOL
from experiments.world_model_integration.urgent_world_model_integration.core import run_case

class IntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol=json.loads(PROTOCOL.read_text())
        cls.spec=public_spec(0,cls.protocol["smoke_seeds"][0],
                             cls.protocol["scenario_max_ticks"])

    def test_rule_episode(self):
        case=run_case(self.spec,self.protocol,"P4",None)
        self.assertGreater(case["decision_count"],0)
        self.assertGreater(case["rollout_steps"],0)
        for name in ("resource_violations","motion_violations","lifecycle_violations"):
            self.assertEqual(case[name],0)

    def test_v4_episode(self):
        checkpoint=os.environ.get("V4_CHECKPOINT")
        if not checkpoint:self.skipTest("frozen V4 checkpoint not supplied")
        case=run_case(self.spec,self.protocol,"P5",Path(checkpoint))
        self.assertGreater(case["query_calls"],0)
        self.assertGreater(case["rollout_steps"],0)
        for name in ("resource_violations","motion_violations","lifecycle_violations"):
            self.assertEqual(case[name],0)

if __name__=="__main__":
    unittest.main()
