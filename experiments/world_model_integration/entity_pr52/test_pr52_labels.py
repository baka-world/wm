"""PR52 node identity alignment, no-future leak, masks and independent labels."""
from __future__ import annotations
import json
from types import SimpleNamespace as NS
import unittest

from experiments.world_model_integration.entity_pr52.pr52_labels import (
    PROTOCOL,validate_protocol,anchor_ids,_future_nodes
)


def observed():
    robots=(NS(id="R2",position=(3,2),phase="to_pickup",task_id="S0",
               service_elapsed=0),)
    tasks=(NS(id="S0",release_tick=0,status="assigned",robot_id="R2",
              stage_index=0), NS(id="NEW",release_tick=10,status="completed",
              robot_id=None,stage_index=0))
    stations=(NS(id="B0",device_available=False,available=False,
                 resource_id="RES",resource_capacity=1,resource_occupied=1,
                 capacity_blocked=True,waiting_pickups=2,approaching_pickups=1),)
    return NS(tick=20,robots=robots,samples=tasks,stations=stations)


class EntityLabelsContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p=json.loads(PROTOCOL.read_text())

    def test_228_new_scenarios_and_all_three_splits(self):
        self.assertTrue(validate_protocol(self.p))
        self.assertEqual(len(self.p["cohort"]),228)
        self.assertEqual(len({x["seed"] for x in self.p["cohort"]}),228)
        for regime in ("congestion","multi_fault","staggered_arrivals"):
            self.assertEqual(sum(x["regime"]==regime and x["split"]=="train"
                                 for x in self.p["cohort"]),48)
            self.assertEqual(sum(x["regime"]==regime and x["split"]=="test"
                                 for x in self.p["cohort"]),16)

    def test_current_node_id_is_only_label_alignment_not_future_input(self):
        public=observed()
        rules=NS(resources=(NS(id="RES",capacity=1),))
        ids=anchor_ids(public,rules)
        self.assertEqual(ids[0],("robot","R2"))
        self.assertEqual(ids[1],("station","B0"))
        self.assertEqual(ids[2],("resource","RES"))
        self.assertEqual(ids[3],("task","S0"))
        # NEW was deliberately not visible at the anchor. The node map uses
        # only the four CURRENT entities from that older public state.
        anchor={"node_keys":ids[:-1],"rules":rules}
        future=_future_nodes(public,anchor)
        self.assertEqual(len(future["node_target"]),4)
        self.assertEqual(future["global_newly_visible_stage_count"],1)
        self.assertEqual(future["global_new_stage_completed_count"],1)
        self.assertEqual(future["global_total_completed_stage_count"],1)
        self.assertFalse(any("NEW" in str(x) for x in future.values()))

    def test_mask_absent_current_task_without_fabricating_completed_status(self):
        public=observed()
        rules=NS(resources=(NS(id="RES",capacity=1),))
        anchor={"node_keys":[("task","UNRELEASED"),("task","S0")],"rules":rules}
        future=_future_nodes(public,anchor)
        self.assertEqual(future["node_target_mask"][0],[0.]*6)
        self.assertEqual(future["node_target_mask"][1],[1.]*6)
        self.assertEqual(future["node_target"][1][1],1.)

    def test_station_and_resource_targets_encode_public_recovery_and_capacity(self):
        public=observed()
        rules=NS(resources=(NS(id="RES",capacity=1),))
        anchor={"node_keys":[("station","B0"),("resource","RES")],"rules":rules}
        future=_future_nodes(public,anchor)
        self.assertEqual(future["node_target"][0][:4],[0.,0.,1.,1.])
        self.assertEqual(future["node_target"][1][:2],[1.,0.])
        self.assertEqual(future["node_target_mask"][1][4:],[0.,0.])

    def test_no_change_to_prior_seen_test_cohort_and_private_data_not_input(self):
        self.assertEqual(self.p["training"]["train_groups"],144)
        self.assertEqual(self.p["training"]["test_groups"],48)
        self.assertIn("future",self.p["hard_requirements"][2].lower())


if __name__=="__main__":
    unittest.main()
