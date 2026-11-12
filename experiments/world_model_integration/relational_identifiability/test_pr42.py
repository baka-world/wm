"""PR42 contract tests: raw links, neural feature collision, no made-up evidence."""
import json
import unittest
from pathlib import Path

from business_transport.state_features import SCHEMA_VERSION, LABEL_VERSION
from lab_transport.business_state_learning import encode_tokens
from experiments.world_model_integration.relational_identifiability.pr42_audit import (
    PROTOCOL, compare_views, frozen_groups, sorted_token_multiset, summarize
)


def fake_action(robot, sample):
    known = lambda n: {"known": True, "value": n}
    return {
        "robot_id": robot, "sample_id": sample,
        "pickup_resource_capacity": known(1),
        "dropoff_resource_capacity": known(1),
        "pickup_resource_occupied": known(0),
        "dropoff_resource_occupied": known(0),
        "pickup_available": known(True),
        "dropoff_available": known(True),
        "task_stage": known(0),
        "robot_to_pickup_distance": 4,
        "pickup_to_dropoff_distance": 7,
        "replaces_running_task": False,
        "competes_with_running_service": False,
        "robot_phase": "idle",
    }


def fake_payload(robot, sample):
    return {
        "schema_version": SCHEMA_VERSION, "label_version": LABEL_VERSION,
        "observed_facts": {"robots": [], "samples": [], "stations": []},
        "candidate_action": [fake_action(robot, sample)],
    }


class RelationAuditContractTests(unittest.TestCase):

    def test_actual_existing_encoder_drops_candidate_identity(self):
        """One controlled structural collision, NOT a measured real-scene rate."""
        a = sorted_token_multiset(encode_tokens(fake_payload("R1", "S1")))
        b = sorted_token_multiset(encode_tokens(fake_payload("R2", "S2")))
        self.assertEqual(a, b)
        view = {"tokens": a, "context": (0., 0.),
                "eligible_event_probabilities": {}}
        diff = compare_views(view, view, (("R1", "S1", "A", "B"),),
                             (("R2", "S2", "A", "B"),))
        self.assertTrue(diff["raw_relationships_different"])
        self.assertTrue(diff["numeric_token_multisets_equal"])
        self.assertTrue(diff["no_eligible_shared_query"])
        self.assertIsNone(diff["shared_event_probability_linf"])

    def test_distinct_relations_are_required(self):
        view = {"tokens": (), "context": (0.,), 
                "eligible_event_probabilities": {}}
        with self.assertRaises(ValueError):
            compare_views(view, view, (("R1", "S1"),), (("R1", "S1"),))

    def test_shared_query_probabilities_are_measured_only_when_present(self):
        base = {"tokens": (), "context": (0., 1.),
                "eligible_event_probabilities": {"R1|S1|A|to_pickup": 0.1}}
        other = {"tokens": (), "context": (0., 1.2),
                 "eligible_event_probabilities": {"R1|S1|A|to_pickup": 0.3}}
        got = compare_views(base, other, (("R1",),), (("R2",),))
        self.assertEqual(got["same_query_count"], 1)
        self.assertAlmostEqual(got["shared_event_probability_linf"], .2)

    def test_predeclared_reused_cohort_only(self):
        cfg = json.loads(PROTOCOL.read_text())
        ids = frozen_groups(cfg)
        self.assertEqual(len(ids), 8)
        self.assertEqual(len({g["seed"] for g in ids}), 8)
        self.assertTrue(cfg["no_training"] and cfg["no_gpu"])
        self.assertIn("development", cfg["source_used_before"])

    def test_summary_never_claims_training_or_independence(self):
        row = {
            "regime": "heavy", "representation_comparison": {
                "numeric_token_multisets_equal": True,
                "pooled_context_linf": 0., "same_query_count": 0},
            "candidate_predictions_tied": True,
            "h20_full_state_distribution_tied": False,
            "true_makespan_different": True,
        }
        s = summarize([row])
        self.assertEqual(s["sensitive_but_h20_tied"], 1)
        self.assertEqual(s["sensitive_but_h20_full_state_tied"], 0)
        self.assertEqual(s["exact_numeric_token_collisions"], 1)
        self.assertEqual(s["new_training"], 0)
        self.assertIn("development-only", s["promotion"])


if __name__ == "__main__":
    unittest.main()
