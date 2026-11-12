"""PR56 real public geometry bridge: input causality, graph content and safety."""
from __future__ import annotations
import json
import unittest

from experiments.world_model_integration.topology_pr56.pr56_public_map import (
    protocol,build_topology_business,public_topology_for_graph,
    topology_pyg_dual_graph,augment_public_transition
)
from experiments.world_model_integration.temporal_pr54.pr54_public_sequence import graph_public_features
from experiments.world_model_integration.stress_pr48.pr48_scenarios import PROTOCOL as STRESS_PROTOCOL
from business_transport.observation import BusinessObserver


class PR56PublicMapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p=protocol()
        cls.stress=json.loads(STRESS_PROTOCOL.read_text())

    def case(self,regime,layout):
        return next(c for c in self.p["cohort"]
                    if c["regime"]==regime and c["layout"]==layout)

    def test_twelve_original_public_topology_dev_scenes_unique_and_no_gpu(self):
        self.assertEqual(len(self.p["cohort"]),12)
        self.assertTrue(all(x["split"]=="topology_development_only"
                            for x in self.p["cohort"]))
        self.assertEqual(self.p["resource"]["gpu"],0)
        self.assertEqual(self.p["resource"]["neural_training"],0)
        self.assertEqual(len({x["layout"] for x in self.p["cohort"]}),4)

    def test_original_geometry_connected_and_obstacle_free_stations(self):
        for case in self.p["cohort"]:
            business,geo=build_topology_business(case,self.stress,self.p)
            grid=business.scenario.grid
            self.assertTrue(grid.blocked)
            self.assertEqual((grid.width,grid.height),(7,5))
            points=[s.position for s in business.scenario.stations.values()]
            points.extend(r.position for r in business.scenario.robots)
            points.extend(r.home for r in business.scenario.robots)
            self.assertTrue(all(grid.contains(p) for p in points))
            self.assertTrue(all(grid.shortest_path(points[0],p) is not None for p in points))
            self.assertEqual(len(geo.adjacency),35-len(grid.blocked))

    def test_multiple_public_maps_have_distinct_route_lengths_and_obstacle_encodings(self):
        seen=set()
        path={}
        for layout in (x["id"] for x in self.p["layouts"]):
            business,geo=build_topology_business(
                self.case("congestion",layout),self.stress,self.p)
            from lab_transport.dynamics_model import Geometry
            graph={"nodes":[{
                "kind":"station","id":s.id,"pos":list(s.position)
            } for s in sorted(business.scenario.stations.values(),key=lambda s:s.id)]}
            topo=public_topology_for_graph(business.scenario.grid,geo,graph)
            seen.add(topo["static_map_digest"])
            self.assertEqual(len(topo["walkable_cells"]),35-len(business.scenario.grid.blocked))
            path[layout]=topo["station_geodesic_distances_ticks"]
            self.assertEqual(len(topo["directed_map_edges"])%2,0)
            self.assertIn("blocked_cells",topo)
            self.assertNotIn("future",str(topo["source"]).lower())
        self.assertEqual(len(seen),4)
        self.assertNotEqual(path["north_wall"],path["south_wall"])

    def test_static_map_pyG_input_is_nonzero_and_separate_from_entity_index_namespace(self):
        case=self.case("congestion","central_bottleneck")
        business,geo=build_topology_business(case,self.stress,self.p)
        # Generate only public and legal BusinessObserver observation, not a
        # future task schedule or privately simulated next tick.
        from business_transport.simulation import BusinessSimulator
        from experiments.world_model_integration.temporal_pr54.pr54_public_sequence import PublicTransitionCollector
        from experiments.world_model_integration.decision_learning_pr51.pr51_labels import TrueCPsatBranch
        policy=TrueCPsatBranch(geo,business.rules,case["seed"],
                               {"counterfactual_max_candidates":4})
        collector=PublicTransitionCollector(business.rules,4,1)
        result=BusinessSimulator(
            business,"world",case["seed"],collect_metrics=False).run(
            dispatch_policy=policy,on_transition=collector)
        self.assertTrue(collector.rows)
        row=collector.rows[0]
        no_map=dict(row["input"])
        self.assertNotIn("public_map",no_map)
        augment_public_transition(row,geo)
        t=topology_pyg_dual_graph(row)
        self.assertEqual(t["entity"].x.shape[1],16)
        self.assertEqual(t["map"].x.shape[1],3)
        self.assertEqual(len(t["map"].x),31)
        self.assertGreater(int(t["map"].edge_index.numel()),0)
        self.assertEqual(t["entity_to_map"].shape[0],2)
        self.assertTrue(t["entity_to_map"].shape[1]>=8)
        self.assertEqual(row["input"]["public_tick"]+1,row["target"]["next_public_tick"])

    def test_future_input_tamper_is_rejected(self):
        case=self.case("multi_fault","north_wall")
        business,geo=build_topology_business(case,self.stress,self.p)
        row={"input":{"public_tick":7,"feature_last_observed_tick":8,"graph":{"nodes":[]}}}
        with self.assertRaises(AssertionError):
            augment_public_transition(row,geo)


if __name__=="__main__":
    unittest.main()
