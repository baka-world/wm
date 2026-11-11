"""Real simulator integration checks. The CI environment supplies original V4 weights."""
from __future__ import annotations
import json
import os
from pathlib import Path
import unittest
from experiments.world_model_integration.urgent_workflow_feasibility.core import public_spec, PROTOCOL
from experiments.world_model_integration.urgent_world_model_integration.core import run_case, _stable_service_uniform, _OneStepUniforms

class IntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol=json.loads(PROTOCOL.read_text())
        cls.spec=public_spec(0,cls.protocol["smoke_seeds"][0],
                             cls.protocol["scenario_max_ticks"])

    def test_keyed_draw_identity_and_order(self):
        a={"robot_id":"R2","sample_id":"JOB7::stage3","phase":"to_dropoff","station_id":"C"}
        b={"robot_id":"R1","sample_id":"JOB4::stage1","phase":"to_pickup","station_id":"A"}
        # A's uniform must not depend on the number or order of other sessions.
        ua=_stable_service_uniform(903001,0,32,a)
        stream1=_OneStepUniforms([ua])
        stream2=_OneStepUniforms([_stable_service_uniform(903001,0,32,b),ua])
        self.assertEqual(stream1.random(),stream2.values[1])
        self.assertGreaterEqual(ua,0.)
        self.assertLess(ua,1.)
        self.assertNotEqual(ua,_stable_service_uniform(903001,1,32,a))
        self.assertNotEqual(ua,_stable_service_uniform(903001,0,33,a))
        self.assertNotEqual(ua,_stable_service_uniform(903002,0,32,a))
        with self.assertRaises(AssertionError):
            stream1.random()

    def test_rule_continuation_is_legal_and_executed(self):
        case=run_case(self.spec,self.protocol,"P4",None,terminal_mode="rule_continuation")
        self.assertGreater(case["terminal_continuation_steps"],0)
        self.assertEqual(case["resource_violations"],0)
        self.assertEqual(case["motion_violations"],0)
        self.assertEqual(case["lifecycle_violations"],0)
        self.assertEqual(case["metrics"]["job_count"],self.spec["jobs"])

    def test_fixed_budgeted_public_queueing(self):
        case=run_case(self.spec,self.protocol,"P4",None,terminal_mode="budgeted_queue")
        self.assertGreater(case["terminal_continuation_steps"],0)
        self.assertLessEqual(case["terminal_continuation_steps"],
                             case["decision_count"]*4*3*20)
        self.assertEqual(case["resource_violations"],0)
        self.assertEqual(case["motion_violations"],0)
        self.assertEqual(case["lifecycle_violations"],0)

    def test_dominance_exact_same_action_and_history(self):
        baseline=run_case(self.spec,self.protocol,"P4",None,
                          terminal_mode="budgeted_queue",evaluation_mode="full")
        pruned=run_case(self.spec,self.protocol,"P4",None,
                        terminal_mode="budgeted_queue",evaluation_mode="dominance")
        self.assertEqual(baseline["history_sha256"],pruned["history_sha256"])
        self.assertEqual(baseline["events_sha256"],pruned["events_sha256"])
        self.assertEqual(baseline["metrics"],pruned["metrics"])
        self.assertLess(pruned["candidates_evaluated"],baseline["candidates_evaluated"])
        self.assertGreater(pruned["candidates_pruned"],0)
        for full,fast in zip(baseline["decisions"],pruned["decisions"]):
            self.assertEqual(full["selected"],fast["selected"])
            for original,projected in zip(full["plans"],fast["plans"]):
                if projected["evaluated"]:
                    self.assertEqual(original["score"],projected["score"])
                    self.assertEqual(original["particle_costs"],projected["particle_costs"])
                else:
                    self.assertIsNone(projected["score"])
                    self.assertEqual(projected["particle_costs"],[])

    def test_rule_episode(self):
        case=run_case(self.spec,self.protocol,"P4",None)
        self.assertGreater(case["decision_count"],0)
        self.assertGreater(case["rollout_steps"],0)
        for name in ("resource_violations","motion_violations","lifecycle_violations"):
            self.assertEqual(case[name],0)

    def test_v4_episode(self):
        checkpoint=os.environ.get("V4_CHECKPOINT")
        if not checkpoint:self.skipTest("frozen V4 checkpoint not supplied")
        case=run_case(self.spec,self.protocol,"P5",Path(checkpoint),random_coupling="keyed")
        self.assertGreater(case["query_calls"],0)
        self.assertGreater(case["rollout_steps"],0)
        for name in ("resource_violations","motion_violations","lifecycle_violations"):
            self.assertEqual(case[name],0)

if __name__=="__main__":
    unittest.main()
