"""PR48 preregistration, true stress generation and fair unchanged models."""
from __future__ import annotations
import json
import unittest
from unittest.mock import patch

from experiments.world_model_integration.stress_pr48.pr48_scenarios import (
    PROTOCOL, ARMS, REGIMES, MECHANISMS, validate, build_business, _private_duration
)
from experiments.world_model_integration.stress_pr48.pr48_online import (
    percentile, real_episode, summarize
)


class Pr48FrozenStressTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p=json.loads(PROTOCOL.read_text())

    def test_48_new_scenarios_balance_and_provenance(self):
        self.assertTrue(validate(self.p))
        self.assertEqual(len(self.p["cohort"]),48)
        self.assertEqual(len({s["seed"] for s in self.p["cohort"]}),48)
        self.assertEqual(list(ARMS),self.p["arms"])
        self.assertEqual(self.p["resources"]["gpu"],0)
        self.assertEqual(self.p["resources"]["new_training"],0)

    def test_both_v4_arms_same_unchanged_cpu_and_score(self):
        self.assertEqual(self.p["fixed_per_decision_cpu_process_budget_seconds"],1.5)
        self.assertEqual(self.p["candidate_limit"],4)
        self.assertEqual(self.p["horizon_ticks"],20)
        self.assertEqual(self.p["particles"],3)
        self.assertIn("expected_completed_samples",self.p["score"])

    def test_congestion_really_has_fourteen_zero_release_jobs(self):
        spec=next(x for x in self.p["cohort"] if x["regime"]=="congestion")
        case,geo=build_business(spec,self.p)
        self.assertEqual(len(case.jobs),14)
        self.assertEqual({j["release_tick"] for j in case.jobs},{0})
        self.assertEqual(len(case.scenario.robots),3)
        outages=sum(bool(st.outages) for st in case.scenario.stations.values())
        self.assertEqual(outages,1)

    def test_multi_fault_has_two_distinct_outage_stations_and_public_provenance(self):
        spec=next(x for x in self.p["cohort"] if x["regime"]=="multi_fault")
        case,geo=build_business(spec,self.p)
        self.assertEqual(len(case.jobs),12)
        self.assertEqual(case.scenario.stations[spec["station_id"]].outages[0][0],7)
        self.assertEqual(case.scenario.stations["C"].outages[0][0],13)
        self.assertEqual(sum(bool(st.outages) for st in case.scenario.stations.values()),2)
        self.assertNotIn("outages",str(case.rules))

    def test_delayed_release_jobs_are_actually_unavailable_initially(self):
        spec=next(x for x in self.p["cohort"] if x["regime"]=="staggered_arrivals")
        case,geo=build_business(spec,self.p)
        self.assertEqual({j["release_tick"] for j in case.jobs},{0,6,12,18})
        self.assertEqual(len(case.jobs),12)
        self.assertEqual(sum(j["release_tick"]>0 for j in case.jobs),9)

    def test_shift_durations_stay_environment_only(self):
        for x in self.p["cohort"]:
            d=_private_duration(x,x["station_id"],0,self.p)
            self.assertIn(d,self.p["predefined_scenario_mechanisms"][x["mechanism"]][x["station_id"]])
            if x["secondary_station_id"]:
                second=_private_duration(x,"C",1,self.p)
                self.assertIn(second,self.p["predefined_scenario_mechanisms"][x["mechanism"]]["C"])

    def test_wall_percentile_is_linear_interpolated_and_nonnegative(self):
        self.assertEqual(percentile([0.,1.,2.],.5),1.)
        self.assertEqual(percentile([0.,1.,2.],.95),1.9)
        self.assertIsNone(percentile([],.95))

    def test_rejects_changed_budget_or_duplicate_testing_seed(self):
        bad=json.loads(json.dumps(self.p))
        bad["fixed_per_decision_cpu_process_budget_seconds"]=2.
        with self.assertRaises(ValueError):validate(bad)
        bad=json.loads(json.dumps(self.p))
        bad["cohort"][1]["seed"]=bad["cohort"][0]["seed"]
        with self.assertRaises(ValueError):validate(bad)


if __name__=="__main__":
    unittest.main()
