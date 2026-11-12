"""PR53 strict physical future-state constraints, true paired contrast and gradient."""
from __future__ import annotations
import json
import tempfile
from pathlib import Path
import unittest
import torch
from torch_geometric.data import Batch

from experiments.world_model_integration.entity_pr52.test_pr52_train import example
from experiments.world_model_integration.entity_pr52.pr52_train import _prepare_one
from experiments.world_model_integration.decision_learning_pr51.pr51_loss import gradient_path_present
from experiments.world_model_integration.action_pr53.pr53_train import (
    PROTOCOL,ConsistentEntityWorld,paired_future_state_contrast,loss_components,
    load_physical_split,report
)


class PR53ActionContrastTests(unittest.TestCase):
    def test_task_probabilities_sum_one_and_future_new_completed_never_exceed_arrivals(self):
        g,reason=_prepare_one(example(),True)
        self.assertIsNone(reason)
        torch.manual_seed(1810)
        m=ConsistentEntityWorld(48,True)
        future,event=m(g["batch"])
        tasks=g["batch"].x[:,1].bool()
        self.assertTrue(torch.allclose(
            future[tasks,:,:4].sum(dim=-1),
            torch.ones_like(future[tasks,:,0]),atol=1e-5))
        self.assertTrue(torch.all(event[:,:,0]<=event[:,:,1]+1e-7))
        self.assertTrue(torch.all(event>=0))
        self.assertTrue(torch.all(event<=1))

    def test_contrast_gradient_must_update_actual_RGCN_transitions(self):
        g,_=_prepare_one(example(),True)
        torch.manual_seed(1810)
        model=ConsistentEntityWorld(48,True)
        node,event=model(g["batch"])
        loss=paired_future_state_contrast(node,event,g)
        self.assertTrue(torch.isfinite(loss))
        model.zero_grad(set_to_none=True)
        loss.backward()
        self.assertTrue(gradient_path_present(model.transition1.parameters()))
        self.assertTrue(gradient_path_present(model.node_transition.parameters()))
        self.assertTrue(gradient_path_present(model.new_event_transition.parameters()))

    def test_value_only_sensitivity_gradient_updates_shared_world_model(self):
        # Verify the value-inspired penalty IN ISOLATION, independently from
        # the action-delta loss that is present in the combined objective.
        from experiments.world_model_integration.entity_pr52.pr52_train import fixed_completion_sensitivity_loss
        group,_=_prepare_one(example(),True)
        torch.manual_seed(1812)
        model=ConsistentEntityWorld(48,True)
        nodes,events=model(group["batch"])
        value_only=fixed_completion_sensitivity_loss(nodes,events,group)
        model.zero_grad(set_to_none=True)
        value_only.backward()
        self.assertTrue(gradient_path_present(model.transition1.parameters()))
        self.assertTrue(gradient_path_present(model.new_event_transition.parameters()))

    def test_no_contrast_from_different_public_node_graphs(self):
        g,_=_prepare_one(example(),True)
        torch.manual_seed(1810)
        model=ConsistentEntityWorld(48,True)
        n,e=model(g["batch"])
        g["node_mask"]=g["node_mask"][:-1]
        with self.assertRaises((RuntimeError,ValueError)):
            paired_future_state_contrast(n,e,g)

    def test_all_arms_use_same_fixed_completion_readout(self):
        p=json.loads(PROTOCOL.read_text())
        self.assertEqual(len(p["train_arms"]),5)
        self.assertEqual(p["train"]["seeds"],[1810,1811,1812])
        g,_=_prepare_one(example(),True)
        for arm in p["train_arms"]:
            model=ConsistentEntityWorld(48,arm["consistent"])
            loss,parts=loss_components(model,g,arm)
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(set(parts),{"state","contrast","decision","weighted"})
            self.assertFalse(any("Q_head" in k or "ranker" in k
                                 for k in model.state_dict()))

    def test_train_does_not_read_test_path_for_fit_or_validation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/"train.json").write_text(json.dumps({
                "schema":"pr53-physically-separated-public-entity-world-state-v1",
                "split":"train",
                "cases":[{"seed":j+{"congestion":0,"multi_fault":100,"staggered_arrivals":200}[reg],"split":"train","regime":reg}
                         for reg in ("congestion","multi_fault","staggered_arrivals")
                         for j in range(48)]
            }))
            self.assertEqual(len(load_physical_split(root,"train")),144)
            with self.assertRaises(FileNotFoundError):
                load_physical_split(root,"test")

    def test_invalid_public_label_mask_is_not_counted_as_negative_prediction(self):
        g,_=_prepare_one(example(),True)
        g["node_mask"].zero_()
        torch.manual_seed(1810)
        model=ConsistentEntityWorld(48,True)
        n,e=model(g["batch"])
        with self.assertRaises(ValueError):
            paired_future_state_contrast(n,e,g)


if __name__=="__main__":
    unittest.main()
