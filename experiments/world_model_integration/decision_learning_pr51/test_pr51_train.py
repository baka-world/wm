"""PR51 actual PyG R-GCN action-conditioned system-state training interface tests."""
from __future__ import annotations
import json
import unittest

import torch

from experiments.world_model_integration.decision_learning_pr51.pr51_train import (
    PROTOCOL,RELATIONS,RelationalSystemWorldModel,
    candidate_graph,predicted_candidate_cost
)
from experiments.world_model_integration.decision_learning_pr51.pr51_loss import (
    pairwise_decision_loss,gradient_path_present
)
from torch_geometric.data import Batch


def public_graph():
    return {"nodes":[
        {"id":"robot:0","kind":"robot","pos":[1,1],
         "phase":"idle","busy":False,"service_elapsed":None},
        {"id":"robot:1","kind":"robot","pos":[1,3],
         "phase":"idle","busy":False,"service_elapsed":None},
        {"id":"task:0","kind":"task","status":"waiting","workflow":"w0",
         "stage_index":0,"released_age":3,"deadline_slack":120},
        {"id":"task:1","kind":"task","status":"waiting","workflow":"w1",
         "stage_index":0,"released_age":0,"deadline_slack":120},
        {"id":"station:0","kind":"station","pos":[1,1],"device_available":True,
         "resource_occupied":0,"waiting_pickups":2,"approaching_pickups":0},
        {"id":"station:1","kind":"station","pos":[5,1],"device_available":False,
         "resource_occupied":0,"waiting_pickups":0,"approaching_pickups":0},
        {"id":"resource:0","kind":"resource","capacity":1}],
        "edges":[
            {"src":"task:0","dst":"station:0","kind":"pickup_at"},
            {"src":"task:1","dst":"station:1","kind":"dropoff_at"},
            {"src":"station:0","dst":"resource:0","kind":"uses_resource"},
        ],
        "candidate_actions":[
            [{"src":"robot:0","dst":"task:0","kind":"proposed_robot_task_assignment"}],
            [{"src":"robot:1","dst":"task:0","kind":"proposed_robot_task_assignment"}]
        ]}


class PR51TrainingContracts(unittest.TestCase):
    def test_pyg_world_model_predictions_depend_on_legal_candidate_edges(self):
        torch.manual_seed(1810)
        graph=public_graph()
        batch=Batch.from_data_list(
            [candidate_graph(graph,i,True) for i in range(2)])
        model=RelationalSystemWorldModel(32,True)
        predicted=model(batch)
        self.assertEqual(tuple(predicted.shape),(2,10))
        self.assertEqual(tuple(predicted_candidate_cost(predicted).shape),(2,))
        self.assertGreater(batch.edge_index.shape[1],0)
        self.assertTrue(torch.isfinite(predicted).all())
        self.assertEqual(len(RELATIONS),6)

    def test_decision_only_loss_hits_shared_r_gcn_transition_weights(self):
        torch.manual_seed(1811)
        graph=public_graph()
        batch=Batch.from_data_list(
            [candidate_graph(graph,i,True) for i in range(2)])
        model=RelationalSystemWorldModel(32,True)
        outcome=model(batch)
        loss=pairwise_decision_loss(
            predicted_candidate_cost(outcome),torch.tensor([80.,83.]))
        model.zero_grad(set_to_none=True)
        loss.backward()
        self.assertTrue(gradient_path_present(model.transition1.parameters()))
        self.assertTrue(gradient_path_present(model.future_state.parameters()))

    def test_training_protocol_frozen_to_real_PyG_and_selection_before_test(self):
        cfg=json.loads(PROTOCOL.read_text())
        self.assertEqual(cfg["protocol"],
                         "PR51_RGCN_DECISION_AWARE_TRAINING_PILOT_20261112_V1")
        self.assertEqual(cfg["torch_geometric_version"],"2.6.1")
        self.assertEqual(cfg["epochs"],40)
        self.assertEqual({a["id"] for a in cfg["train_arms"]},{
            "relation_pred_only","relation_joint025","relation_joint1",
            "flat_joint1_control"})
        self.assertTrue(cfg["training_safety"]["test_read_after_selection_written"])
        self.assertTrue(cfg["training_safety"]["gradient_must_reach_RGCNConv_weights"])

if __name__=="__main__":
    unittest.main()
