"""Guardrails for offline paired rollout value; not new independent evidence."""
import hashlib
import json
import unittest

from experiments.world_model_integration.rollout_value.evaluate import (
    SEED_OFFSETS,SOURCE_SEEDS,rank,gap_error,restricted_flow,run_branch)
from experiments.world_model_integration.build_dataset import case_for
from experiments.world_model_integration.build_dataset_v4 import CONFIG
from experiments.world_model_integration.v4_dispatch import policy_for
from business_transport.simulation import BusinessSimulator
from experiments.business_constraints.run import rich_rows
from lab_transport.decision_value import candidates_for
from business_transport.dispatch import BusinessCPSAT


class ContractTests(unittest.TestCase):
    def test_frozen_evaluation_design(self):
        self.assertEqual(SOURCE_SEEDS,(43011,43015,43019))
        self.assertEqual(len(SEED_OFFSETS),8)
        self.assertEqual(len(set(SEED_OFFSETS)),8)
        self.assertEqual(SEED_OFFSETS[0],0)

    def test_tie_break_and_centered_differences(self):
        self.assertEqual(rank([4,4,3]),[2,0,1])
        self.assertEqual(gap_error([100,100,102],[40,45,47]),4.0)

    def test_original_simulator_uses_keyed_exogenous_noise(self):
        cfg=json.loads(CONFIG.read_text())
        spec=next(x for x in cfg["groups"]["independent_in"] if x["seed"]==43019)
        business,geo=case_for(spec,cfg)
        a=BusinessSimulator(business,"world",43019)
        b=BusinessSimulator(business,"world",43019)
        c=BusinessSimulator(business,"world",43019+SEED_OFFSETS[1])
        stage=business.jobs[0]["stages"][0]
        pickup=business.scenario.samples[0].pickup
        aa=a._service_duration(stage,"to_pickup",pickup)
        self.assertEqual(aa,b._service_duration(stage,"to_pickup",pickup))
        self.assertGreaterEqual(aa,1)
        # Different future seeds are NOT automatically independent draws:
        # collisions are allowed. Must use the same keyed draw across candidates.
        self.assertGreaterEqual(c._service_duration(stage,"to_pickup",pickup),1)

    def test_restart_at_tick_zero_reuses_original_observable_prefix(self):
        cfg=json.loads(CONFIG.read_text())
        spec=next(x for x in cfg["groups"]["independent_in"] if x["seed"]==43011)
        business,geo=case_for(spec,cfg)
        policy=policy_for("Rule",business,geo,None,None)
        rows=[]
        BusinessSimulator(business,"world",spec["seed"],collect_metrics=True).run(
            dispatch_policy=policy,on_observation=policy.observer.observe,
            on_transition=rows.append)
        self.assertEqual(rows[0].observation.tick,0)
        current=rich_rows(rows)[0].observation
        plans=BusinessCPSAT(geo,business.rules,seed=1810,
                            candidate_limit=4,time_limit_seconds=.5).proposals(current,candidates_for(current))
        first=run_branch(business,geo,spec["seed"],plans[0],"P1_nearest",rows[0].observation)
        second=run_branch(business,geo,spec["seed"],plans[0],"P1_nearest",rows[0].observation)
        self.assertEqual(first["history_sha256"],second["history_sha256"])
        self.assertEqual(first["events_sha256"],second["events_sha256"])
        self.assertEqual(first["restricted_mean_flow"],second["restricted_mean_flow"])


if __name__=="__main__":
    unittest.main()
