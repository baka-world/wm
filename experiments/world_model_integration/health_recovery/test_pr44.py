"""Strict PR44 public station recovery and censoring contract tests."""
from __future__ import annotations
import json
import unittest
from experiments.world_model_integration.health_recovery.pr44_observed_recovery import (
    PROTOCOL, fit_survival, survival_at, recovery_probability,
    spells_from_series, public_station_series, prediction_queries, validate_protocol,
    score_episode, aggregate,
)


class RecoveryPublicContractTests(unittest.TestCase):
    def test_right_censor_is_not_misclassified_as_recovery(self):
        spells=[{"observed_duration":2,"recovered":True},
                {"observed_duration":4,"recovered":False},
                {"observed_duration":6,"recovered":True}]
        km=fit_survival(spells)
        self.assertEqual(km["records"],3)
        self.assertEqual(km["events"],2)
        self.assertEqual(km["censored"],1)
        self.assertAlmostEqual(survival_at(km,2),2./3.)
        self.assertAlmostEqual(survival_at(km,4),2./3.)
        self.assertEqual(survival_at(km,6),0.)
        self.assertAlmostEqual(recovery_probability(km,0,2),1./3.)
        self.assertAlmostEqual(recovery_probability(km,2,2),0.)
        self.assertIsNone(recovery_probability(km,6,1))

    def test_public_down_up_spell_durations_not_future_schedule(self):
        public=[(t, a) for t,a in enumerate((True,True,False,False,False,True,True,
                                            False,False))]
        self.assertEqual(spells_from_series(public),[
            {"observed_duration":3,"recovered":True,"onset_tick_observed":2},
            {"observed_duration":1,"recovered":False,"onset_tick_observed":7},
        ])

    def test_capacity_block_is_not_called_device_outage(self):
        observations=[
            {"tick":0,"stations":{"B0":{"device_available":True,
                    "service_admission_available":False,"capacity_blocked":True}}},
            {"tick":1,"stations":{"B0":{"device_available":False,
                    "service_admission_available":False,"capacity_blocked":False}}},
            {"tick":2,"stations":{"B0":{"device_available":True,
                    "service_admission_available":True,"capacity_blocked":False}}},
        ]
        self.assertEqual(public_station_series(observations,"B0"),
                         [(0,True),(1,False),(2,True)])
        self.assertEqual(spells_from_series(public_station_series(observations,"B0"))[0]["observed_duration"],1)

    def test_missing_future_is_not_treated_as_no_recovery(self):
        observations=[
            {"tick":0,"stations":{"B0":{"device_available":False}}},
            {"tick":1,"stations":{"B0":{"device_available":False}}},
            {"tick":2,"stations":{"B0":{"device_available":True}}},
        ]
        q=prediction_queries(observations,"B0",[1,2,5])
        self.assertEqual(sum(x["truth_available"] is None for x in q),3)
        self.assertTrue(any(x["truth_available"]==1 for x in q))
        self.assertTrue(any(x["truth_available"]==0 for x in q))

    def test_deterministic_no_recovery_baseline_and_survival(self):
        km=fit_survival([{"observed_duration":2,"recovered":True}])
        obs=[
            {"tick":0,"stations":{"B0":{"device_available":False}}},
            {"tick":1,"stations":{"B0":{"device_available":False}}},
            {"tick":2,"stations":{"B0":{"device_available":True}}},
        ]
        models={"pooled":km,"station":{"B0":km}}
        from experiments.world_model_integration.health_recovery.pr44_observed_recovery import score_episode
        row=score_episode({"id":"T","seed":1,"station_id":"B0",
                            "public_observations":obs,"public_spells":[],"safety_violations":0},
                           models,[1,2])
        self.assertAlmostEqual(row["horizons"]["2"]["brier_by_method"]["persist_down"],1.0)
        self.assertAlmostEqual(row["horizons"]["2"]["brier_by_method"]["pooled_km"],0.0)

    def test_prospective_split_is_frozen_and_disjoint(self):
        p=json.loads(PROTOCOL.read_text())
        validate_protocol(p)
        self.assertEqual(len(p["cohort"]["train"]),36)
        self.assertEqual(len(p["cohort"]["test"]),24)
        self.assertEqual(p["execution"]["gpu"],0)
        self.assertEqual(p["execution"]["neural_training"],0)
        self.assertTrue(p["evaluation"]["no_future_labels_in_features"])


if __name__=="__main__":
    unittest.main()
