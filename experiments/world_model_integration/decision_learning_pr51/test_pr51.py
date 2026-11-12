"""PR51 test the paired legal action label contract and world-model gradient path."""
from __future__ import annotations
import json
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from experiments.world_model_integration.decision_learning_pr51.pr51_contract import (
    PROTOCOL,validate_protocol
)
from experiments.world_model_integration.decision_learning_pr51.pr51_loss import (
    pairwise_decision_loss,combined_world_model_loss,gradient_path_present
)
from experiments.world_model_integration.decision_learning_pr51.pr51_labels import (
    public_relational_graph
)
from experiments.business_constraints.tests.test_business import fixture


class CounterfactualLabelsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p=json.loads(PROTOCOL.read_text())

    def test_independent_data_sources_and_frozen_counterfactual_policy(self):
        self.assertTrue(validate_protocol(self.p))
        self.assertEqual(len(self.p["cohort"]),72)
        self.assertEqual(len(set(x["seed"] for x in self.p["cohort"])),72)
        self.assertEqual(self.p["counterfactual_max_candidates"],4)
        for regime in ("congestion","multi_fault","staggered_arrivals"):
            r=[s for s in self.p["cohort"] if s["regime"]==regime]
            self.assertEqual(len(r),24)
            self.assertEqual(sum(s["split"]=="train" for s in r),12)
            self.assertEqual(sum(s["split"]=="test" for s in r),8)

    def test_decision_loss_correctly_orders_actual_completion_costs(self):
        labels=torch.tensor([101.,95.,98.])
        good=torch.tensor([101.,95.,98.],requires_grad=True)
        bad=torch.tensor([95.,101.,98.],requires_grad=True)
        self.assertLess(pairwise_decision_loss(good,labels).item(),
                        pairwise_decision_loss(bad,labels).item())
        self.assertGreater(pairwise_decision_loss(bad,labels).item(),0.)

    def test_ties_are_not_assigned_false_winner(self):
        predictions=torch.tensor([4.,7.,1.],requires_grad=True)
        tied=torch.tensor([95.,95.,95.])
        loss=pairwise_decision_loss(predictions,tied)
        self.assertEqual(loss.item(),0.)
        loss.backward()
        self.assertEqual(predictions.grad.abs().sum().item(),0.)

    def test_decision_loss_gradient_must_reach_shared_dynamics(self):
        torch.manual_seed(1810)
        # The head consumes a differentiable *predicted future state*.
        transition=nn.Linear(3,5)
        readout=nn.Linear(5,1)
        x=torch.tensor([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
        predicted_state=transition(x)
        ranking=readout(predicted_state).flatten()
        values=torch.tensor([3.,2.,8.])
        loss=pairwise_decision_loss(ranking,values)
        loss.backward()
        self.assertTrue(gradient_path_present(transition.parameters()))
        self.assertTrue(gradient_path_present(readout.parameters()))
        # An auxiliary ranker that detaches state must fail the same check.
        transition.zero_grad();readout.zero_grad()
        ranking_detached=readout(transition(x).detach()).flatten()
        pairwise_decision_loss(ranking_detached,values).backward()
        self.assertFalse(gradient_path_present(transition.parameters()))

    def test_censored_candidate_excluded_from_ranking(self):
        a=torch.tensor([3.,0.,7.],requires_grad=True)
        y=torch.tensor([4.,100.,8.])
        mask=torch.tensor([True,False,True])
        loss=pairwise_decision_loss(a,y,mask)
        loss.backward()
        self.assertEqual(a.grad[1].item(),0.)

    def test_public_graph_never_serializes_future_workload_schedule(self):
        _,_,business=fixture()
        from business_transport.observation import BusinessObserver
        from business_transport.simulation import BusinessSimulator
        from lab_transport.transition_schema import Assignment
        public=[]
        BusinessSimulator(business,"world",3).run(
            on_observation=lambda raw:public.append(raw) if not public else None)
        observer=BusinessObserver()
        obs=observer.observe(public[0])
        action=(Assignment("R1","J1::stage1"),)
        graph=public_relational_graph(obs,business.rules,[action])
        self.assertTrue(any(x["kind"]=="robot" for x in graph["nodes"]))
        self.assertTrue(any(x["kind"]=="station" for x in graph["nodes"]))
        self.assertTrue(any(x["kind"]=="task" for x in graph["nodes"]))
        self.assertIn("proposed_robot_task_assignment",str(graph["candidate_actions"]))
        for forbidden in ("outages","private_duration","future_job_schedule",
                          "true_future","factual_makespan","seed"):
            self.assertNotIn(forbidden,json.dumps(graph))

    def test_no_decision_training_before_labels_are_verified(self):
        self.assertFalse(self.p["resources"]["initial_training"])
        self.assertTrue(self.p["resources"]["gpu_for_later_training"])


if __name__=="__main__":
    unittest.main()
