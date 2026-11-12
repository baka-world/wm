"""PR46 tests for frozen cohorts, mechanism shift, policy parity and public clock."""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiments.world_model_integration.recovery_shift.pr46_shared import (
    PROTOCOL, MECHANISMS, ONLINE_ARMS, FORECAST_ARMS,
    validate_protocol, _hidden_duration, build_business,
)
from experiments.world_model_integration.recovery_shift.pr46_online import (
    OriginalSignatureRecoveryAdapter, RecoveryOnlinePolicy,
)
from experiments.world_model_integration.recovery_integration.pr45_recovery_world import (
    PublicDownClock,
)
from experiments.world_model_integration.online_horizon.pr30_run import OnlineHorizonPolicy


class Pr46ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.proto=json.loads(PROTOCOL.read_text())

    def test_predeclared_disjoint_splits_and_no_retraining(self):
        self.assertTrue(validate_protocol(self.proto))
        forecast=self.proto["cohort"]["forecast"]
        online=self.proto["cohort"]["online"]
        self.assertEqual(len(forecast),24)
        self.assertEqual(len(online),24)
        self.assertEqual(len({s["seed"] for s in forecast+online}),48)
        self.assertEqual(self.proto["resources"]["gpu"],0)
        self.assertEqual(self.proto["resources"]["new_training"],0)
        self.assertEqual(self.proto["online"]["decision_cpu_budget_seconds"],.5)

    def test_duration_shift_outside_previous_support(self):
        p=self.proto
        for station in ("B0","B1","C"):
            old=p["hidden_environment_only"]["support_by_mechanism"]["within_support"][station]
            new=p["hidden_environment_only"]["support_by_mechanism"]["extended_outage"][station]
            self.assertGreater(max(new),max(old))
        for s in p["cohort"]["forecast"]+p["cohort"]["online"]:
            self.assertIn(_hidden_duration(s,p),
                          p["hidden_environment_only"]["support_by_mechanism"][
                              s["mechanism"]][s["station_id"]])

    def test_hidden_truth_is_only_in_simulator_scenario(self):
        s=self.proto["cohort"]["forecast"][0]
        business,geo=build_business(s,self.proto)
        target=business.scenario.stations[s["station_id"]]
        self.assertEqual(target.outages[0][0],7)
        self.assertEqual(len(business.jobs),10)
        self.assertNotIn("outages",str(business.rules))
        self.assertEqual(business.scenario.max_ticks,170)

    def test_public_health_clock_has_no_private_outage_access(self):
        def obs(t,up):
            return SimpleNamespace(tick=t,stations=[
                SimpleNamespace(id="B0",device_available=up)])
        c=PublicDownClock()
        self.assertIsNone(c.observe(obs(0,True))["B0"])
        self.assertEqual(c.observe(obs(1,False))["B0"],0)
        self.assertEqual(c.observe(obs(2,False))["B0"],1)
        self.assertIsNone(c.observe(obs(3,True))["B0"])
        self.assertIsNone(PublicDownClock().observe(obs(6,False))["B0"])

    def test_adapter_identical_public_inputs_to_frozen_world(self):
        c=PublicDownClock()
        public=SimpleNamespace(tick=1,stations=[
            SimpleNamespace(id="B0",device_available=False)])
        c.observe(SimpleNamespace(tick=0,stations=[
            SimpleNamespace(id="B0",device_available=True)]))
        c.observe(public)
        model=SimpleNamespace(rollout=lambda *a,**k:("forecast",[
            {"sampled_recovery":True},{"sampled_recovery":False}]))
        adapter=OriginalSignatureRecoveryAdapter(model,c)
        value=adapter.rollout(
            public,(),continuation_policy=lambda _:(),
            continuation_policy_name="CP_SAT_predicted_state",
            horizon_ticks=20,particles=3,sampling_seed=1810)
        self.assertEqual(value,"forecast")
        self.assertEqual(adapter.total_recovery_draws,2)
        self.assertEqual(adapter.total_sampled_repairs,1)
        self.assertEqual(adapter.rollout_calls,1)
        with self.assertRaisesRegex(ValueError,"missing contemporaneous"):
            adapter.rollout(
                SimpleNamespace(tick=2),(),continuation_policy=lambda _:(),
                continuation_policy_name="CP_SAT_predicted_state",
                horizon_ticks=20,particles=3,sampling_seed=1810)

    def test_recovery_policy_keeps_parent_scoring_fallback_logic(self):
        self.assertTrue(issubclass(RecoveryOnlinePolicy,OnlineHorizonPolicy))
        self.assertEqual(RecoveryOnlinePolicy.__call__.__code__.co_names[-1],
                         "__call__")
        self.assertEqual(self.proto["online"]["score"].split(",")[0],
                         "max expected completed samples at H20")

    def test_no_untested_extra_arms(self):
        self.assertEqual(tuple(self.proto["forecast"]["arms"]),FORECAST_ARMS)
        self.assertEqual(tuple(self.proto["online"]["arms"]),ONLINE_ARMS)


if __name__=="__main__":
    unittest.main()
