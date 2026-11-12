"""PR54 public one-step temporal RGCN contracts, leakage and gradient tests."""
import json
import tempfile
from pathlib import Path
import unittest

import torch
from torch_geometric.data import Batch

from experiments.world_model_integration.temporal_pr54.pr54_public_sequence import (
    PROTOCOL,validate,run_one,build_transition
)
from experiments.world_model_integration.temporal_pr54.pr54_train import (
    example_to_graph,load_split
)
from experiments.world_model_integration.temporal_pr54.pr54_model import (
    PublicTemporalRelationalWorld,supervised_losses
)
from experiments.world_model_integration.stress_pr48.pr48_scenarios import (
    PROTOCOL as STRESS_PROTOCOL
)


class PR54TemporalWorldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=json.loads(PROTOCOL.read_text())
        cls.scenario=json.loads(STRESS_PROTOCOL.read_text())

    def test_frozen_train_val_new_test_public_timeline(self):
        self.assertTrue(validate(self.cfg))
        self.assertEqual(len(self.cfg["cohort"]),60)
        self.assertEqual(len({x["seed"] for x in self.cfg["cohort"]}),60)
        self.assertEqual(self.cfg["history_window"],4)

    def test_actual_business_simulator_transition_contains_no_future_feature(self):
        spec=self.cfg["cohort"][0]
        result=run_one(spec,self.cfg,self.scenario)
        self.assertGreaterEqual(result["real_observation_transitions"],10)
        seen=False
        for row in result["transitions"]:
            inp=row["input"];truth=row["target"]
            self.assertEqual(truth["next_public_tick"],inp["public_tick"]+1)
            self.assertEqual(inp["feature_last_observed_tick"],inp["public_tick"])
            self.assertEqual(len(inp["history"]),len(inp["graph"]["nodes"]))
            self.assertEqual(len(inp["history_visible_mask"]),len(inp["history"]))
            self.assertTrue(all(x[-1]==1 for x in inp["history_visible_mask"]))
            self.assertEqual(sum(v["kind"]=="robot" for v in inp["graph"]["nodes"]),3)
            self.assertNotIn("future",json.dumps(inp["graph"]["source"]))
            self.assertNotIn("private_outage",json.dumps(inp))
            if any(v[0]==0 for v in inp["history_visible_mask"]):
                seen=True
        self.assertTrue(seen)

    def test_pyg_history_GRU_and_snapshot_arms_have_same_parameter_count(self):
        spec=self.cfg["cohort"][0]
        row=run_one(spec,self.cfg,self.scenario)["transitions"][8]
        g=example_to_graph(row)
        b=Batch.from_data_list([g,g])
        torch.manual_seed(1810)
        history=PublicTemporalRelationalWorld(48,True)
        snapshot=PublicTemporalRelationalWorld(48,False)
        self.assertEqual(sum(x.numel() for x in history.parameters()),
                         sum(x.numel() for x in snapshot.parameters()))
        n,e=history(b)
        self.assertEqual(tuple(n.shape),(b.num_nodes,6))
        self.assertEqual(tuple(e.shape),(2,2))
        task=b.x[:,1].bool()
        self.assertTrue(torch.allclose(
            n[task,:4].sum(dim=-1),torch.ones_like(n[task,0]),atol=1e-5))
        self.assertTrue(torch.all(e[:,0]<=e[:,1]))
        loss=supervised_losses(n,e,b)
        self.assertTrue(torch.isfinite(loss["total"]))
        history.zero_grad(set_to_none=True)
        loss["total"].backward()
        self.assertIsNotNone(history.history_gru.weight_ih_l0.grad)
        self.assertGreater(float(history.history_gru.weight_ih_l0.grad.abs().sum()),0.)
        self.assertGreater(float(history.relation_step1.weight.grad.abs().sum()),0.)

    def test_tampering_history_tick_cannot_access_future(self):
        from types import SimpleNamespace
        before=SimpleNamespace(tick=5)
        after=SimpleNamespace(tick=7)
        with self.assertRaises(ValueError):
            build_transition(before,after,(),[],None,4)
        after=SimpleNamespace(tick=6)
        with self.assertRaises(AssertionError):
            build_transition(before,after,(),
                             [{"tick":6,"keys":[],"features":[]}],None,4)

    def test_independent_test_file_never_needed_for_train(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            rows=[]
            for regime in ("congestion","multi_fault","staggered_arrivals"):
                for idx in range(12):
                    rows.append({"id":regime+str(idx),"seed":len(rows)+1,
                                 "regime":regime,"split":"train"})
            (root/"train.json").write_text(json.dumps({
                "schema":"pr54-physically-isolated-temporal-world-split-v1",
                "split":"train","episodes":rows}))
            self.assertEqual(len(load_split(root,"train")),36)
            with self.assertRaises(FileNotFoundError):
                load_split(root,"test")

if __name__=="__main__":
    unittest.main()
