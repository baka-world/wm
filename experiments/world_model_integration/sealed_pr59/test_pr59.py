"""PR59 actual public map/temporal-world gradient and blind-split contracts.

A true BusinessSimulator sample is obtained only from TRAIN input.
No blind holdout transition is synthesized or read.
"""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

import torch
from torch_geometric.loader import DataLoader

from experiments.world_model_integration.sealed_pr59.pr59_source import (
    read_protocol,validate_layout_business,sample
)
from experiments.world_model_integration.sealed_pr59.pr59_dual_model import (
    actual_public_record_to_graph,PublicMapTemporalRelationalWorld,
    parameter_count,supervised_losses
)
from experiments.world_model_integration.topology_pr56.pr56_public_map import (
    build_topology_business,public_topology_for_graph
)
from experiments.world_model_integration.stress_pr48.pr48_scenarios import (
    PROTOCOL as STRESS_PROTOCOL
)
from experiments.world_model_integration.temporal_pr54.pr54_public_sequence import (
    PublicTransitionCollector
)
from experiments.world_model_integration.decision_learning_pr51.pr51_labels import (
    TrueCPsatBranch
)
from business_transport.simulation import BusinessSimulator


class WorldMapBatchAndHoldoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.p=read_protocol()
        cls.stress=json.loads(STRESS_PROTOCOL.read_text())
        cls.cases=cls.p["cohort"]

    def test_split_registration_not_random_seed_or_map_family_leak(self):
        self.assertEqual({k:sum(c["split"]==k for c in self.cases)
                          for k in ("train","validation","test")},
                         {"train":24,"validation":12,"test":12})
        self.assertEqual(len({c["seed"] for c in self.cases}),48)
        families={k:{c["map_family"] for c in self.cases
                     if c["split"]==k} for k in ("train","validation","test")}
        self.assertEqual(families["train"],{"vertical_blocked_route_training"})
        self.assertEqual(families["validation"],{"diagonal_staggered_validation"})
        self.assertEqual(families["test"],{"horizontal_shelf_blind_topology"})
        self.assertTrue(all(not families[a]&families[b]
                            for a in families for b in families if a!=b))
        self.assertTrue(self.p["splits"]["test"]["physically_sealed_no_actual_data_access"])

    def test_development_source_refuses_to_read_or_generate_blind_test(self):
        with tempfile.TemporaryDirectory() as dir:
            with self.assertRaises(PermissionError):
                sample("test",Path(dir)/"forbidden.json")
        self.assertFalse(self.p["splits"]["test"]["generate_in_this_cpu_stage"])

    def test_all_ten_maps_are_public_connected_without_cut_vertex(self):
        for spec in [next(c for c in self.cases if c["layout"]==m["id"])
                     for m in self.p["layouts"]]:
            topo,dist=validate_layout_business(spec,self.p,self.stress)
            self.assertGreater(dist,0)
            self.assertEqual(topo["schema"],"pr56-public-grid-cells-v1")
            self.assertEqual(len(topo["walkable_cells"])+len(topo["blocked_cells"]),35)

    @classmethod
    def one_real_train_transition(cls):
        case=next(c for c in cls.cases if c["split"]=="train" and c["profile"]=="baseline")
        business,geo=build_topology_business(case,cls.stress,cls.p)
        dispatch=TrueCPsatBranch(geo,business.rules,case["seed"],
                                 {"counterfactual_max_candidates":4})
        actual=PublicTransitionCollector(business.rules,window=4,max_steps=1)
        BusinessSimulator(business,"world",case["seed"],collect_metrics=False).run(
            dispatch_policy=dispatch,on_transition=actual)
        if len(actual.rows)!=1:
            raise AssertionError("source was not a real public first-tick observation")
        row=actual.rows[0]
        row["input"]["public_map"]=public_topology_for_graph(
            business.scenario.grid,geo,row["input"]["graph"])
        return row

    def test_real_PyG_dual_graph_batch_offsets_and_actual_map_gradients(self):
        row=self.one_real_train_transition()
        data=actual_public_record_to_graph(row)
        # Add a different train-only map with SAME public entity state.
        # This is a controlled input topology-sensitivity test, NOT a
        # fabricated future simulator prediction/label.
        alternative=next(c for c in self.cases
                         if c["split"]=="train" and c["layout"]!=row["input"]["public_map"]["static_map_digest"]
                         and c["layout"]=="train_v_east_b")
        business,geo=build_topology_business(alternative,self.stress,self.p)
        modified=copy.deepcopy(row)
        modified["input"]["public_map"]=public_topology_for_graph(
            business.scenario.grid,geo,row["input"]["graph"])
        data2=actual_public_record_to_graph(modified)
        self.assertNotEqual(data.map_x.shape[0],0)
        self.assertNotEqual(row["input"]["public_map"]["static_map_digest"],
                            modified["input"]["public_map"]["static_map_digest"])
        batch=next(iter(DataLoader([data,data2],batch_size=2,shuffle=False)))
        self.assertEqual(batch.num_graphs,2)
        self.assertEqual(batch.map_node_count.sum(),batch.map_x.shape[0])
        off=int(data.x.shape[0])
        mapoff=int(data.map_x.shape[0])
        count1=int(data.entity_to_map_index.shape[1])
        self.assertTrue(torch.all(
            batch.entity_to_map_index[0,count1:]>=off))
        self.assertTrue(torch.all(
            batch.entity_to_map_index[1,count1:]>=mapoff))
        self.assertTrue(torch.all(batch.map_edge_index[:,data.map_edge_index.shape[1]:]>=mapoff))

        torch.manual_seed(5959)
        enabled=PublicMapTemporalRelationalWorld(hidden=48,map_enabled=True)
        masked=PublicMapTemporalRelationalWorld(hidden=48,map_enabled=False)
        masked.load_state_dict(enabled.state_dict())
        self.assertEqual(parameter_count(enabled),parameter_count(masked))
        net_pred,ev_pred=enabled(batch)
        off_pred,off_ev=masked(batch)
        self.assertEqual(tuple(net_pred.shape),(batch.x.shape[0],6))
        self.assertEqual(tuple(ev_pred.shape),(2,2))
        self.assertTrue(torch.all(ev_pred[:,0]<=ev_pred[:,1]))
        self.assertTrue(torch.isfinite(net_pred).all())
        self.assertTrue(torch.isfinite(ev_pred).all())
        base_err=supervised_losses(net_pred,ev_pred,batch)["total"]
        base_err.backward()
        self.assertIsNotNone(enabled.map_conv1.lin_l.weight.grad)
        self.assertGreater(float(enabled.map_conv1.lin_l.weight.grad.abs().sum()),0.)
        self.assertIsNotNone(enabled.entity_world.relation_step1.weight.grad)
        self.assertGreater(float(enabled.entity_world.relation_step1.weight.grad.abs().sum()),0.)
        self.assertGreater(float((net_pred-off_pred).abs().sum()),1e-8)
        # Masked model's outputs must be identical for same factual public
        # state/action, irrespective of different map sidecars.
        n=off_pred.shape[0]//2
        self.assertTrue(torch.allclose(off_pred[:n],off_pred[n:],rtol=0,atol=1e-7))
        self.assertTrue(torch.allclose(off_ev[0],off_ev[1],rtol=0,atol=1e-7))
        self.assertGreater(float((net_pred[:n]-net_pred[n:]).abs().sum()),1e-8)

    def test_gradient_is_only_from_training_label_and_not_BLIND_TEST(self):
        row=self.one_real_train_transition()
        g=actual_public_record_to_graph(row)
        self.assertEqual(g.history.shape[1:],(4,16))
        self.assertEqual(g.node_target.shape[1],6)
        self.assertEqual(int(g.map_node_count[0]),int(g.map_x.shape[0]))
        self.assertEqual(g.event_target.shape,(1,2))
        self.assertEqual(row["target"]["next_public_tick"],row["input"]["public_tick"]+1)


if __name__=="__main__":
    unittest.main()
