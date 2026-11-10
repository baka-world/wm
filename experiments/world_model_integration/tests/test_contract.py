"""CPU integration tests, covering source provenance and actual rollout selection."""
from dataclasses import replace
import math
from pathlib import Path
import tempfile
import unittest

import torch

from business_transport.observation import BusinessObserver
from business_transport.policy import WorkflowPolicy
from business_transport.service_work import ServiceWorkEstimator
from business_transport.state_features import build_business_features, SCHEMA_VERSION
from business_transport.workflow_scoring import workflow_components, score_components
from business_transport.simulation import BusinessSimulator
from experiments.business_constraints.tests.test_business import fixture
from experiments.business_constraints.run import rich_rows
from lab_transport.business_state_learning import (BusinessSetsServiceEvent,
    encode_batch, encode_tokens, load_online, train_cpu)
from lab_transport.decision_value import candidates_for, nearest_batch
from lab_transport.dynamics_model import Geometry
from lab_transport.paper_reuse.pets_service import ServiceDistribution
from lab_transport.torch_structured_adapter import ProgressAwareRule


def trajectory():
    base, config, business = fixture(1)
    geometry = Geometry(base.grid)
    transitions = []
    BusinessSimulator(business, "world", 1, collect_metrics=True).run(
        on_transition=transitions.append,
        dispatch_policy=lambda obs, candidates: nearest_batch(obs, candidates, geometry))
    return business, geometry, rich_rows(transitions)


class FixedDurationPETS:
    """Deterministic test replacement, never presented as pretrained PETS."""
    distribution = ServiceDistribution

    def __init__(self, effective_duration):
        self.duration = effective_duration
        self.calls = 0

    def _parameters(self, features, *, transition):
        self.calls += 1
        if len(features) != 7 or transition:
            raise ValueError("legacy PETS scoring feature contract violated")
        return [(math.log(self.duration), -12.)] * 5


class InterfaceIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_full_state_has_explicit_masks_and_action_distances(self):
        business, geometry, rows = trajectory()
        row = rows[0]
        payload = build_business_features(row.observation, row.assignments, geometry, business.rules)
        self.assertEqual(payload["schema_version"], SCHEMA_VERSION)
        self.assertEqual(payload["unknown"]["future_fault_schedule"], {"known": False, "value": None})
        self.assertEqual(payload["unknown"]["true_remaining_service_countdown"], {"known": False, "value": None})
        self.assertEqual(len(payload["observed_facts"]["stations"]), 5)
        self.assertEqual(len(payload["public_rules"]["resources"]), 3)
        self.assertEqual(len(payload["candidate_action"]), len(row.assignments))
        self.assertTrue(any(s["public_stages_total"]["known"] for s in payload["observed_facts"]["samples"]))
        self.assertFalse(payload["observed_facts"]["robots"][0]["operational_available"]["known"])
        encoded = encode_tokens(payload)
        self.assertEqual(encoded.shape[1], 40)
        x, mask = encode_batch([payload, payload])
        self.assertEqual(tuple(BusinessSetsServiceEvent()(x, mask).shape), (2,))

    def test_future_fields_not_read_from_hidden_truth(self):
        business, geometry, rows = trajectory()
        obs = rows[0].observation
        unchanged = build_business_features(obs, (), geometry, business.rules)
        # The contract is created from public observations, not the business
        # simulator's future faults, unreleased job table or service countdown.
        self.assertEqual(unchanged["unknown"]["unreleased_external_jobs"]["value"], None)
        self.assertEqual(unchanged["unknown"]["counterfactual_outcome"]["value"], None)

    def test_frozen_old_score_and_conditional_pets_replacement(self):
        business, geometry, rows = trajectory()
        # Find factual observed active service; no synthetic future state needed.
        active = [r.observation for r in rows
                  if any(robot.service_elapsed is not None and
                         robot.service_active_elapsed is not None
                         for robot in r.observation.robots)]
        self.assertTrue(active, "fixture must contain at least one observed service session")
        initial, terminal = rows[0].observation, active[0]
        frozen = workflow_components(initial, [terminal], geometry, business.rules)
        self.assertNotIn("rule_prior_service", frozen)
        slow = FixedDurationPETS(20)
        fast = FixedDurationPETS(3)
        fast_score = workflow_components(initial, [terminal], geometry, business.rules,
                                         estimator=ServiceWorkEstimator(fast))
        slow_score = workflow_components(initial, [terminal], geometry, business.rules,
                                         estimator=ServiceWorkEstimator(slow))
        self.assertGreater(fast.calls, 0)
        self.assertGreater(slow.calls, 0)
        self.assertGreater(slow_score["model_service"], fast_score["model_service"])
        self.assertGreater(score_components(slow_score)[0], score_components(fast_score)[0])
        self.assertAlmostEqual(fast_score["rule_prior_service"], frozen["estimated_remaining_service"])
        self.assertTrue(any(row["source"] == "pets_conditional_v1"
                            for row in fast_score["service_estimates"]))
        self.assertTrue(all(row["uncertainty"] is None or
                            row["uncertainty"]["future_device_failure_risk"] == "unknown"
                            for row in fast_score["service_estimates"]))

    def test_unknown_progress_and_future_work_explicit_rule_fallback(self):
        business, geometry, rows = trajectory()
        observation = rows[0].observation
        station = observation.stations[0]
        sample = observation.samples[0]
        estimate = ServiceWorkEstimator(FixedDurationPETS(20)).estimate(
            observation, station, sample=sample, robot=None, phase="to_pickup")
        self.assertFalse(estimate.usable)
        self.assertEqual(estimate.source, "rule_prior")
        self.assertEqual(estimate.fallback_reason, "future_or_unstarted_service")

    def test_opt_in_horizons_execute_and_never_mutate_factual_observation(self):
        business, geometry, rows = trajectory()
        observation = rows[0].observation
        candidates = candidates_for(observation)
        for horizon in (5, 10, 20):
            policy = WorkflowPolicy("cpsat_rule", geometry, business.rules,
                                    BusinessObserver(), ProgressAwareRule(geometry),
                                    model_aware=True, horizon=horizon, particles=2, budget=2)
            before = observation
            choice = policy(observation, candidates)
            self.assertEqual(observation, before)
            self.assertEqual(policy.fallbacks, 0, policy.log[-1]["fallback"])
            self.assertTrue(all(action in candidates for action in choice))
            event = policy.log[-1]
            self.assertEqual(event["horizon"], horizon)
            self.assertEqual(event["scoring"], "workflow_model")
            self.assertEqual(sum(p["selected"] for p in event["plans"]), 1)
            for plan in event["plans"]:
                self.assertEqual(len(plan["particle_business_costs"]), 2)
                self.assertFalse(plan["cost_distribution"]["calibrated_interval"])
                self.assertIn("model_minus_prior", plan["component_means"])
                self.assertEqual(plan["component_means"]["model_minus_prior"], 0.)

    def test_new_model_is_not_silently_deployed(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "unverified.pt"
            torch.save({"metadata": {"model_version": "business-sets-service-event-v1",
                 "schema_version": SCHEMA_VERSION, "label_version": "business-service-event-v1",
                 "online_verified": False}, "model_state": BusinessSetsServiceEvent().state_dict()}, checkpoint)
            with self.assertRaisesRegex(ValueError, "not independently verified"):
                load_online(checkpoint)
            with self.assertRaises((ValueError, FileNotFoundError)):
                train_cpu(Path(folder) / "missing-factual-transitions.json",
                          Path(folder) / "nonexistent.pt")

    def test_controlled_same_state_service_estimate_flips_real_policy_choice(self):
        """Causal wiring only: replace predictions, not simulator future truth."""
        from business_transport.policy import BusinessPolicy
        from business_transport.service_work import ServiceWorkEstimate
        from business_transport.scenario import compile_scenario
        from lab_transport.transition_schema import Assignment
        base, cfg, _ = fixture(1)
        stations = {sid: replace(station, service_ticks=30, prior_ticks=30,
                                 jitter_ticks=0) if sid in ("A0", "A1") else station
                    for sid, station in base.stations.items()}
        business = compile_scenario(replace(base, stations=stations), cfg)
        geometry = Geometry(base.grid)
        traces = []
        BusinessSimulator(business, "world", 1).run(
            on_transition=traces.append,
            dispatch_policy=lambda obs, candidates: nearest_batch(obs, candidates, geometry))
        observation = rich_rows(traces)[0].observation
        candidates = candidates_for(observation)
        plans = ((Assignment("R1", "J1::stage1"),),
                 (Assignment("R2", "J2::stage1"),))
        self.assertTrue(all(a in candidates for p in plans for a in p))

        class ControlledEstimator:
            def __init__(self, favoured):
                self.favoured = favoured

            def estimate(self, observation, station, *, sample=None, robot=None, phase=None):
                prior = float(station.expected_service_ticks)
                if robot is None:
                    return ServiceWorkEstimate(prior, prior, False, "rule_prior",
                                               fallback_reason="unstarted")
                amount = 1. if sample.id.startswith(self.favoured) else 1000.
                return ServiceWorkEstimate(prior, amount, True, "controlled_test_only")

        class FixedPlans(BusinessPolicy):
            def candidates(self, observation, candidates):
                return plans

        choices = []
        records = []
        for favoured in ("J1", "J2"):
            policy = FixedPlans("cpsat_rule", geometry, business.rules,
                                BusinessObserver(), ProgressAwareRule(geometry),
                                scoring="workflow_model", workflow_prediction=True,
                                horizon=5, particles=1, budget=2)
            policy.estimator = ControlledEstimator(favoured)
            selected = policy(observation, candidates)
            self.assertEqual(policy.fallbacks, 0, policy.log[-1]["fallback"])
            self.assertIn(selected, plans)
            choices.append(selected)
            records.append([p["score"] for p in policy.log[-1]["plans"]])
        self.assertNotEqual(choices[0], choices[1])
        self.assertNotEqual(records[0], records[1])


if __name__ == "__main__":
    unittest.main()
