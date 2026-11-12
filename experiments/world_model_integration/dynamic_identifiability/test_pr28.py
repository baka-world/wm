"""Frozen scenario contract and genuine safe first-action interventions."""
from __future__ import annotations
import json
import unittest
from dataclasses import replace

from experiments.world_model_integration.dynamic_identifiability.pr28_experiment import (
    PROTOCOL,build_business,run_factual,
)
from experiments.world_model_integration.transport_events.benchmark import PROTOCOL as PRIOR


class DynamicDecisionIdentifiabilityTests(unittest.TestCase):
    def test_frozen_groups_are_independent_and_scheduled_not_hidden(self):
        proto=json.loads(PROTOCOL.read_text())
        self.assertEqual(proto["schema"],
            "RHWM_PR28_DYNAMIC_ACTION_IDENTIFIABILITY_20261112_V1")
        self.assertEqual([len([g for g in proto["scenarios"] if g["regime"]==r])
            for r in ("simultaneous_load","scheduled_release")],[6,6])
        seeds={s["seed"] for s in proto["scenarios"]}
        self.assertEqual(len(seeds),12)
        old=json.loads(PRIOR.read_text())
        older={s["seed"] for s in old["groups"]+
            old["future_decision_groups"]+old["future_horizon_groups"]}
        self.assertTrue(seeds.isdisjoint(older))
        self.assertEqual(proto["prediction_audit"]["forecast_horizons"],[5,10,20])
        self.assertEqual(proto["bounded_counterfactual"]["candidates"]["limit"],4)

    def test_existing_workflow_semantics_and_explicit_deadlines(self):
        cfg=json.loads(PROTOCOL.read_text())
        for spec in (cfg["scenarios"][0],cfg["scenarios"][6]):
            business,geometry=build_business(spec,max_ticks=cfg["max_ticks"])
            self.assertEqual(len(business.jobs),spec["job_count"])
            self.assertEqual(len(business.scenario.robots),spec["robot_count"])
            self.assertEqual(len(business.scenario.samples),2*spec["job_count"])
            for job in business.jobs:
                self.assertEqual(
                    job["deadline_tick"]-job["release_tick"],
                    spec["deadline_slack"])
            self.assertEqual(
                [j["release_tick"] for j in business.jobs],
                spec["release_ticks"])
            self.assertEqual(
                business.rules.station_resources["A0"].capacity,1)

    def test_first_action_factual_run_has_legal_selections_and_no_violations(self):
        cfg=json.loads(PROTOCOL.read_text())
        scenario=cfg["scenarios"][0]
        first=run_factual(scenario,cfg,0)
        self.assertTrue(first["first_plans"])
        self.assertEqual(first["outcomes"]["safety_violations"],0)
        self.assertGreater(first["outcomes"]["policy_calls"],0)
        self.assertEqual(first["outcomes"]["completed_jobs"],scenario["job_count"])
        self.assertEqual(first["outcomes"]["status"],"completed")
        self.assertGreaterEqual(first["outcomes"]["end_tick"],1)
        self.assertTrue(all(a in first["first_plans"][0] for a in
                            first["first_selected"]))
        if len(first["first_plans"])>1:
            second=run_factual(scenario,cfg,1)
            self.assertEqual(first["first_state"],second["first_state"])
            self.assertEqual(first["first_plans"],second["first_plans"])
            self.assertEqual(second["outcomes"]["safety_violations"],0)


if __name__=="__main__":
    unittest.main()
