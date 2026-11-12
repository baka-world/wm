"""PR52 actual RGCN node-future and no rank-head decision sensitivity tests."""
from __future__ import annotations
import json
import tempfile
from pathlib import Path
import unittest

import torch
from torch_geometric.data import Batch

from experiments.world_model_integration.decision_learning_pr51.test_pr51_train import public_graph
from experiments.world_model_integration.decision_learning_pr51.pr51_train import candidate_graph
from experiments.world_model_integration.decision_learning_pr51.pr51_loss import (
    pairwise_decision_loss,gradient_path_present
)
from experiments.world_model_integration.entity_pr52.pr52_train import (
    PROTOCOL,EntityConditionalRelationalWorld,total_completed_future,
    decision_cost,fixed_completion_sensitivity_loss,entity_losses,
    load_split,_prepare_one
)


def example():
    public=public_graph()
    n=len(public["nodes"])
    base=[[0.]*6 for _ in range(n)]
    masks=[[1.]*6 if node["kind"]!="resource"
           else [1.,1.,1.,1.,0.,0.]
           for node in public["nodes"]]
    def outcome(cost,extra):
        return {
            "factual_makespan_ticks":cost,
            "future_entity_states":{
                horizon:{
                    "node_target":base,"node_target_mask":masks,
                    "global_total_completed_stage_count":int(horizon=="20")+extra,
                    "global_new_stage_completed_count":extra,
                    "global_newly_visible_stage_count":extra+1
                } for horizon in ("5","20")
            }
        }
    return {
        "id":"synthetic","regime":"multi_fault","split":"train",
        "labelable":True,"all_branches_completed":True,
        "full_entity_horizon_labels_available":True,
        "candidate_count":2,"public_graph":public,
        "true_factual_outcomes":[outcome(80,1),outcome(83,2)]
    }


class PR52TrainingTests(unittest.TestCase):
    def test_source_action_graph_to_nodewise_future_shape_and_masks(self):
        group,err=_prepare_one(example(),True)
        self.assertIsNone(err)
        self.assertEqual(group["node_y"].shape[1:],(2,6))
        self.assertEqual(group["node_mask"].shape,group["node_y"].shape)
        self.assertEqual(tuple(group["event_y"].shape),(2,2,2))
        torch.manual_seed(1810)
        model=EntityConditionalRelationalWorld(48,True)
        node,event=model(group["batch"])
        self.assertEqual(node.shape,group["node_y"].shape)
        self.assertEqual(tuple(event.shape),(2,2,2))
        self.assertEqual(tuple(total_completed_future(node,event,group["batch"]).shape),(2,2))
        self.assertEqual(tuple(decision_cost(node,event,group["batch"]).shape),(2,))
        loss=entity_losses(node,event,group)
        self.assertGreater(float(loss["prediction"]),0.)

    def test_decision_only_objective_updates_shared_relation_dynamics(self):
        torch.manual_seed(1810)
        group,_=_prepare_one(example(),True)
        net=EntityConditionalRelationalWorld(48,True)
        node,event=net(group["batch"])
        loss=pairwise_decision_loss(
            decision_cost(node,event,group["batch"]),group["true_cost"])
        net.zero_grad(set_to_none=True)
        loss.backward()
        self.assertTrue(gradient_path_present(net.transition1.parameters()))
        self.assertTrue(gradient_path_present(net.node_transition.parameters()))

    def test_value_sensitivity_is_on_future_completion_state_not_new_score_head(self):
        torch.manual_seed(1811)
        group,_=_prepare_one(example(),True)
        net=EntityConditionalRelationalWorld(48,True)
        node,event=net(group["batch"])
        weighted=fixed_completion_sensitivity_loss(node,event,group)
        self.assertTrue(torch.isfinite(weighted))
        self.assertGreaterEqual(float(weighted),0.)
        net.zero_grad(set_to_none=True)
        weighted.backward()
        self.assertTrue(gradient_path_present(net.transition1.parameters()))
        self.assertFalse(any("ranker" in name or "q_head" in name
                             for name,_ in net.named_parameters()))

    def test_physical_split_reader_does_not_open_unseen_test_file(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)
            doc={"schema":"pr52-physical-isolated-public-entity-split-v1",
                 "split":"train","cases":[]}
            for i,regime in enumerate(("congestion","multi_fault","staggered_arrivals")):
                for j in range(48):
                    doc["cases"].append({"regime":regime,"split":"train",
                                         "seed":i*100+j})
            (path/"train.json").write_text(json.dumps(doc))
            self.assertEqual(len(load_split(path,"train")),144)
            with self.assertRaises(FileNotFoundError):
                load_split(path,"test")

    def test_fixed_protocol_no_new_q_ranker_and_two_seed_ablation(self):
        config=json.loads(PROTOCOL.read_text())
        self.assertEqual(config["training"]["seeds"],[1810,1811])
        self.assertEqual(config["training"]["train_groups"],144)
        self.assertEqual(config["training"]["test_groups"],48)
        self.assertEqual(len(config["train_arms"]),4)
        self.assertTrue(all(x["lambda_decision"]>=0
                            for x in config["train_arms"]))


if __name__=="__main__":
    unittest.main()
