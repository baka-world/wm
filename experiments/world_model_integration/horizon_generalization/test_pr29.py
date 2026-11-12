"""PR29 scenario-seed integrity, actual pairs, safe and source-model forecasts."""
from __future__ import annotations

import json
import os
from pathlib import Path
import unittest

from experiments.world_model_integration.horizon_generalization.pr29_evaluate import (
    ARMS, HORIZONS, PROTOCOL, _scenario_seed_set, _select,
    first_action_config, paired_truth, forecast_pair,
)
from experiments.world_model_integration.dynamic_identifiability.pr28_experiment import (
    build_business,
)


class GeneralizationTests(unittest.TestCase):
    def setUp(self):
        self.protocol=json.loads(PROTOCOL.read_text())

    def test_preregistered_groups_are_new_unique_and_four_workloads(self):
        cfg=self.protocol
        self.assertEqual(cfg["schema"],"RHWM_PR29_FROZEN_HORIZON_SCENE_GENERALIZATION_20261112_V1")
        self.assertEqual(list(HORIZONS),[5,10,20])
        self.assertEqual(list(ARMS),["rule","v4"])
        self.assertEqual(len(cfg["scene_groups"]),32)
        self.assertEqual({x["regime"] for x in cfg["scene_groups"]},
            {"light","medium","heavy_new_seed","heavy_service_shift"})
        self.assertEqual({reg:sum(g["regime"]==reg for g in cfg["scene_groups"])
                          for reg in {x["regime"] for x in cfg["scene_groups"]}},
                         {reg:8 for reg in ("light","medium","heavy_new_seed","heavy_service_shift")})
        _scenario_seed_set(cfg)
        self.assertEqual(cfg["prediction"]["ranking"],
            "Higher forecast expected_completed_samples at h; on tie choose CP-SAT first candidate; do not tune weight")

    def test_precommitted_scenarios_reuse_original_workflow(self):
        cfg=self.protocol
        for regime in ("light","medium","heavy_new_seed","heavy_service_shift"):
            spec=next(x for x in cfg["scene_groups"] if x["regime"]==regime)
            business,geo=build_business(spec,max_ticks=cfg["max_ticks"])
            self.assertEqual(len(business.jobs),spec["job_count"])
            self.assertEqual(len(business.scenario.robots),spec["robot_count"])
            self.assertEqual(len(business.scenario.samples),2*spec["job_count"])
            self.assertEqual(business.rules.station_resources["A0"].capacity,
                             spec["capacity"])
            self.assertTrue(all(j["deadline_tick"]==spec["deadline_slack"]
                                for j in business.jobs))

    def test_selection_tie_is_predeclared_first_cpsat_candidate(self):
        self.assertEqual(_select([
            {"expected_completed":1.0},{"expected_completed":1.0}]),0)
        self.assertEqual(_select([
            {"expected_completed":1.0},{"expected_completed":1.2}]),1)
        self.assertEqual(_select([
            {"expected_completed":2.0},{"expected_completed":1.2}]),0)

    def test_actual_pair_has_same_initial_public_state_and_no_safety_violation(self):
        spec=next(x for x in self.protocol["scene_groups"] if x["regime"]=="heavy_new_seed")
        pairs=paired_truth(spec,self.protocol)
        self.assertGreaterEqual(len(pairs),1)
        self.assertLessEqual(len(pairs),2)
        self.assertEqual(pairs[0]["first_state"].tick,0)
        self.assertTrue(pairs[0]["public_observations"])
        self.assertIn(5,pairs[0]["public_observations"])
        self.assertTrue(all(x["outcomes"]["safety_violations"]==0 for x in pairs))
        if len(pairs)>1:
            self.assertEqual(pairs[0]["first_state"],pairs[1]["first_state"])
            self.assertNotEqual(pairs[0]["first_selected"],pairs[1]["first_selected"])

    @unittest.skipUnless(os.environ.get("RHWM_V4_CHECKPOINT"),
                         "CI restores immutable original V4; no local missing checkpoint")
    def test_actual_rule_and_v4_are_callable_with_official_horizon(self):
        spec=next(x for x in self.protocol["scene_groups"] if x["regime"]=="medium")
        pairs=paired_truth(spec,self.protocol)
        checkpoint=Path(os.environ["RHWM_V4_CHECKPOINT"])
        for arm in ARMS:
            forecast=forecast_pair(pairs[0],self.protocol,checkpoint,
                                   arm=arm,horizon=5)
            self.assertEqual(len(forecast),len(pairs))
            for f in forecast:
                self.assertEqual(f["forecast"].distributions[-1].tick,5)
                self.assertEqual(len(f["forecast"].particles),3)
                self.assertGreaterEqual(f["cpu_seconds"],0)
                self.assertEqual(f["safety_ticks"],15)


if __name__=="__main__":
    unittest.main()
