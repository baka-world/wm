"""Paired decision gate: online public-only CP-SAT V4 pgmpy policies."""
from __future__ import annotations
import json
import os
from pathlib import Path
import unittest

from experiments.world_model_integration.decision_value.pr27_evaluate import (
    PROTOCOL,SCENES,_scenario_with_deadlines,evaluate_one,ARMS,
)


class PR27DecisionTests(unittest.TestCase):
    def test_frozen_paired_groups_and_deadlines(self):
        config=json.loads(PROTOCOL.read_text())
        scene=json.loads(SCENES.read_text())
        self.assertEqual(config["protocol"],"RHWM_PR27_PAIRED_POLICY_EVALUATION_V1")
        self.assertEqual(config["arms"],list(ARMS))
        chosen={x["seed"] for x in scene["future_decision_groups"]}
        earlier={x["seed"] for x in scene["groups"]+scene["future_horizon_groups"]}
        self.assertEqual(len(chosen),12)
        self.assertTrue(chosen.isdisjoint(earlier))
        business,geo=_scenario_with_deadlines(scene["future_decision_groups"][0],config)
        for job in business.jobs:
            self.assertIsNotNone(job["deadline_tick"])
            self.assertTrue(all(s.deadline_tick==job["deadline_tick"] for s in
                            business.scenario.samples if s.id in job["stages"]))

    @unittest.skipUnless(os.environ.get("RHWM_V4_CHECKPOINT") and
                         os.environ.get("RHWM_TRANSPORT_JSON"),
                         "live official model sources restored only by CI")
    def test_three_online_arms_do_not_violate_motion_or_workflow(self):
        from business_transport.world_model.transport_bayes import TransportBayesModel
        config=json.loads(PROTOCOL.read_text())
        scene=json.loads(SCENES.read_text())
        bayes=TransportBayesModel.from_payload(
            json.loads(Path(os.environ["RHWM_TRANSPORT_JSON"]).read_text()))
        spec=scene["future_decision_groups"][0]
        for arm in ARMS:
            value=evaluate_one(
                spec,config,arm,
                checkpoint=Path(os.environ["RHWM_V4_CHECKPOINT"]),bayes=bayes)
            self.assertEqual(value["arm"],arm)
            self.assertEqual(value["resource_violations"],0)
            self.assertEqual(value["motion_violations"],0)
            self.assertEqual(value["lifecycle_violations"],0)
            self.assertGreater(value["policy_calls"],0)
            self.assertLessEqual(value["changed_from_cpsat_first_plan"],
                                 value["multiple_options"])
            self.assertGreater(value["time_until_end"],0)


if __name__=="__main__":
    unittest.main()
