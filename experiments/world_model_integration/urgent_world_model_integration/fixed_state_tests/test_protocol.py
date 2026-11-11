"""Tests of frozen sources, seed discipline and strict comparison semantics."""
import json
from pathlib import Path
import unittest
from experiments.world_model_integration.urgent_world_model_integration.fixed_state_experiment import (
    PROBE_PROTOCOL, _assignment, _matches, _stored_matches, sha,
)

class FrozenSameStateProtocolTests(unittest.TestCase):
    def test_five_precommitted_decisions_and_independent_seeds(self):
        p=json.loads(PROBE_PROTOCOL.read_text())
        self.assertEqual(len(p["fixed_states"]),5)
        self.assertEqual([(x["seed"],x["tick"]) for x in p["fixed_states"]],
                         [(905147,0),(905585,0),(905220,41),(905804,0),(906195,138)])
        seeds=p["independent_rng_seeds"]
        self.assertEqual(len(seeds),7)
        self.assertEqual(len(set(seeds)),7)
        self.assertFalse(set(seeds)&{x["seed"] for x in p["fixed_states"]})
        for item in p["fixed_states"]:
            self.assertEqual(len(item["observation_sha256"]),64)
            self.assertLessEqual(item["candidate_count"],4)
        self.assertEqual(p["fixed_budget"]["max_states"],5)

    def test_matching_identity_ignores_assignment_order(self):
        a=[{"robot_id":"R2","sample_id":"U-2"},
           {"robot_id":"R1","sample_id":"U-1"}]
        b=[{"robot_id":"R1","sample_id":"U-1"},
           {"robot_id":"R2","sample_id":"U-2"}]
        self.assertEqual(_stored_matches(a),_stored_matches(b))
        self.assertEqual(_matches(_assignment(a)),_stored_matches(a))

    def test_checkpoint_identity_declared(self):
        p=json.loads(PROBE_PROTOCOL.read_text())
        self.assertEqual(p["checkpoint_sha256"],
            "60c8a6c3d832da3e6519072688ea099f64ae120ba19d3bd2ec705f7f1a11ac0c")
        self.assertEqual(sha(b"abcd"),
            "88d4266fd4e6338d13b845fcf289579d209c897823b9217da3e161936f031589")


if __name__=="__main__":
    unittest.main()
