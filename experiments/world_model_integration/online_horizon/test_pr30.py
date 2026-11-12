"""Online full-simulator H5/H20 evaluation contracts on entirely new groups."""
from __future__ import annotations
import json
import os
from pathlib import Path
import unittest

from experiments.world_model_integration.online_horizon.pr30_run import (
    PROTOCOL,ARMS,OnlineHorizonPolicy,one,_scenario_seed_set
)
from experiments.world_model_integration.dynamic_identifiability.pr28_experiment import build_business


class PR30OnlineTests(unittest.TestCase):
    def test_frozen_24_new_groups_and_protocol(self):
        cfg=json.loads(PROTOCOL.read_text())
        self.assertEqual(cfg["protocol"],"RHWM_PR30_INDEPENDENT_H20_ONLINE_20261112_V1")
        self.assertEqual(len(cfg["groups"]),24)
        self.assertEqual(cfg["arms"],list(ARMS))
        self.assertEqual({x["regime"] for x in cfg["groups"]},
                         {"light","medium","heavy","shift"})
        self.assertEqual(len({x["seed"] for x in cfg["groups"]}),24)
        _scenario_seed_set({"scene_groups":cfg["groups"]})
        self.assertEqual(cfg["decision_cpu_budget_seconds"],0.5)
        self.assertEqual(cfg["particles"],3)

    def test_builds_all_scenarios_from_unmodified_business_rules(self):
        cfg=json.loads(PROTOCOL.read_text())
        for name in ("light","medium","heavy","shift"):
            spec=next(x for x in cfg["groups"] if x["regime"]==name)
            business,geo=build_business(spec,max_ticks=cfg["max_ticks"])
            self.assertEqual(len(business.jobs),spec["job_count"])
            self.assertEqual(len(business.scenario.robots),spec["robot_count"])
            self.assertEqual(business.rules.station_resources["A0"].capacity,
                             spec["capacity"])

    @unittest.skipUnless(os.environ.get("RHWM_V4_CHECKPOINT"),"CI restores V4 checkpoint")
    def test_actual_online_policy_has_legal_factual_execution(self):
        cfg=json.loads(PROTOCOL.read_text())
        spec=next(x for x in cfg["groups"] if x["regime"]=="medium")
        checkpoint=Path(os.environ["RHWM_V4_CHECKPOINT"])
        for arm in ARMS:
            out=one(spec,cfg,arm,checkpoint)
            self.assertEqual(out["safety_violations"],0)
            self.assertFalse(out["job_metrics"]["censored"])
            self.assertEqual(out["job_metrics"]["completed_jobs"],spec["job_count"])
            self.assertGreater(out["policy_calls"],0)
            self.assertGreaterEqual(out["full_episode_cpu_seconds"],0)
            self.assertGreaterEqual(out["multiple_candidates"],out["changed_plan_calls"])


if __name__=="__main__":
    unittest.main()
