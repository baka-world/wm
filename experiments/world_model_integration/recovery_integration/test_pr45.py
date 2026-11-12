"""PR45 two-contract recovery event tests, no trained V4 needed."""
from __future__ import annotations
from dataclasses import replace
import json
import unittest

from business_transport.observation import BusinessObserver
from business_transport.scenario import compile_scenario
from business_transport.simulation import BusinessSimulator
from business_transport.world_model import HybridSystemWorldModel
from business_transport.workflow_transition import check_business_transition
from experiments.business_constraints.tests.test_business import fixture
from experiments.world_model_integration.recovery_integration.pr45_recovery_world import (
    PROTOCOL, PublicDownClock, check_recovery_boundary, inject_recovery,
    _sample_recovery,
)
from experiments.world_model_integration.health_recovery.pr44_observed_recovery import (
    PROTOCOL as PR44_PROTOCOL, recovery_probability
)
from lab_transport.dynamics_model import Geometry


def currently_down_fixture():
    scenario, conf, _=fixture(capacity=1)
    stations=dict(scenario.stations)
    stations["B0"]=replace(stations["B0"],outages=((0,10),))
    modified=replace(scenario,stations=stations)
    business=compile_scenario(modified,conf)
    observation=BusinessObserver().observe(
        BusinessSimulator(business,"world",1)._observable_state(0))
    return modified,business,observation


class ExplicitRecoveryContractTests(unittest.TestCase):

    def test_recovery_is_a_separate_validated_exogenous_boundary(self):
        scene,business,raw=currently_down_fixture()
        original=HybridSystemWorldModel(Geometry(scene.grid),business.rules,kind="rule")
        legal=original.predict_next(raw,(),sampling_seed=191)
        baseline=legal.next_state
        self.assertFalse(next(s.device_available for s in baseline.stations if s.id=="B0"))
        after=inject_recovery(baseline,["B0"],business.rules)
        self.assertTrue(next(s.device_available for s in after.stations if s.id=="B0"))
        self.assertTrue(check_recovery_boundary(baseline,after,["B0"],business.rules))
        self.assertEqual(after.robots,baseline.robots)
        self.assertEqual(after.samples,baseline.samples)
        self.assertEqual(after.tick,baseline.tick)
        # The unmodified base contract correctly forbids an exogenous change.
        with self.assertRaises(ValueError):
            check_business_transition(raw,after,Geometry(scene.grid),business.rules,())

    def test_no_unlisted_station_changes_or_healthy_device_recovery(self):
        scene,business,raw=currently_down_fixture()
        with self.assertRaises(ValueError):
            inject_recovery(raw,["A0"],business.rules)
        with self.assertRaises(ValueError):
            inject_recovery(raw,["NO_STATION"],business.rules)
        good=inject_recovery(raw,["B0"],business.rules)
        self.assertIs(good if False else None,None)  # keep constructed state live
        with self.assertRaises(ValueError):
            check_recovery_boundary(raw,replace(good,tick=good.tick+1),
                                    ["B0"],business.rules)
        with self.assertRaises(ValueError):
            check_recovery_boundary(raw,good,[],business.rules)

    def test_public_down_clock_never_invents_an_unseen_onset(self):
        scene,business,down=currently_down_fixture()
        clock=PublicDownClock()
        make=lambda obs,t,up:replace(
            obs,tick=t,stations=tuple(
                replace(s,device_available=up,available=up)
                if s.id=="B0" else s for s in obs.stations))
        healthy=make(down,0,True)
        first=make(down,1,False)
        second=make(down,2,False)
        self.assertIsNone(clock.observe(healthy)["B0"])
        self.assertEqual(clock.observe(first)["B0"],0)
        self.assertEqual(clock.observe(second)["B0"],1)
        self.assertIsNone(PublicDownClock().observe(second)["B0"])
        with self.assertRaises(ValueError):
            clock.observe(second)

    def test_recovery_draws_reproducible_and_not_global_rng(self):
        a=[_sample_recovery(101,2,t,"B0",.4) for t in range(40)]
        b=[_sample_recovery(101,2,t,"B0",.4) for t in range(40)]
        self.assertEqual(a,b)
        self.assertFalse(_sample_recovery(101,2,8,"B0",0.))
        self.assertTrue(_sample_recovery(101,2,8,"B0",1.))
        with self.assertRaises(ValueError):
            _sample_recovery(101,2,8,"B0",1.1)

    def test_survival_after_known_down_age_is_training_only(self):
        sample={"records":3,"events":2,"censored":1,
                "curve":[[0,1.],[2,2./3.],[4,1./3.]]}
        self.assertAlmostEqual(recovery_probability(sample,2,2),.5)

    def test_predeclared_new_cohort_and_frozen_provenance(self):
        p=json.loads(PROTOCOL.read_text())
        old=json.loads(PR44_PROTOCOL.read_text())
        self.assertEqual(p["methods"],[
            "original_v4_h20_no_future_faults",
            "v4_h20_observed_down_km_recovery"])
        self.assertEqual(p["horizons"],[5,20])
        self.assertEqual(p["anchors"],[8,10])
        self.assertEqual(len(p["cohort"]),12)
        old_seeds={s["seed"] for split in old["cohort"].values() for s in split}
        self.assertFalse(old_seeds & {s["seed"] for s in p["cohort"]})
        self.assertTrue(p["resources"]["freeze_v4"])
        self.assertEqual(p["resources"]["gpu"],0)
        self.assertEqual(p["resources"]["new_training"],0)


if __name__=="__main__":
    unittest.main()
