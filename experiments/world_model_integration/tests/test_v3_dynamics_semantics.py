"""Causal service-event contract at the learned->rule boundary.

A decoder probability P(finish at t+1) must control THIS exact service
session rather than an aggregated global event, even with two simultaneous
workstation services. No actual trained checkpoint or hidden time is used:
only deterministic diagnostic logits to test transition semantics.
"""
from dataclasses import replace
import random
import unittest
import torch

from business_transport.dynamics import WorkflowDynamics
from business_transport.session_dynamics import SessionRuleDynamics
from business_transport.workflow_transition import check_business_transition
from experiments.world_model_integration.tests.test_session_v2 import SessionEventV2Tests
from lab_transport.torch_structured_adapter import ProgressAwareRule


class _SessionSpecificLogits(torch.nn.Module):
    def forward(self, tokens, mask, query):
        # station y-coordinate is an OBSERVED target field at index 5.
        # A0 y=1 -> finish, A1 y=3 -> continue.
        return torch.where(query[:,5] < .06,
                           torch.full_like(query[:,5],100.),
                           torch.full_like(query[:,5],-100.))


class SessionTransitionSemanticsTests(unittest.TestCase):
    def test_independent_session_probabilities_advance_matching_workflow_only(self):
        before, geo, business = SessionEventV2Tests()._two_sessions()
        model = SessionRuleDynamics.__new__(SessionRuleDynamics)
        model.network = _SessionSpecificLogits()
        model.geometry,model.rules = geo,business.rules
        model.rule = ProgressAwareRule(geo)
        model.rng = random.Random(1810)
        model.service_calls = model.query_calls = 0
        model.fallback_reasons = {}
        wrapped = WorkflowDynamics(model,business.rules)

        old = before
        after = wrapped.predict_next(before,())
        self.assertEqual(before,old,"forecast must never modify factual state")
        self.assertTrue(check_business_transition(before,after,geo,business.rules,()))
        samples = {s.id:s for s in after.samples}
        self.assertEqual(samples["J1::stage1"].status,"in_transit")
        self.assertEqual(samples["J2::stage1"].status,"assigned")
        self.assertEqual(model.query_calls,2)
        self.assertEqual(model.service_calls,1)
        self.assertEqual([(s.id,s.expected_service_ticks) for s in before.stations],
                         [(s.id,s.expected_service_ticks) for s in after.stations],
                         "temporary per-service sampling thresholds must not leak")

    def test_observed_device_pause_cannot_sample_completion(self):
        before,geo,business=SessionEventV2Tests()._two_sessions()
        stations=tuple(replace(s,device_available=False,available=False)
                       if s.id=="A0" else s for s in before.stations)
        paused=replace(before,stations=stations)
        from business_transport.session_events import eligible_sessions
        rows=eligible_sessions(paused,())
        reasons={q["station_id"]:reason for q,reason in rows}
        self.assertEqual(reasons["A0"],"observed_device_pause")


if __name__=="__main__":
    unittest.main()
