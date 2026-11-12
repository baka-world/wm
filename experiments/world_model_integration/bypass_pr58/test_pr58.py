"""PR58 paired public topology quality and map-only invariance tests.

All 12 simulator worlds are executed by GitHub Actions, not synthetic mocks.
The development-only protocol is frozen independently of their outcomes.
"""
from __future__ import annotations
import json
import unittest
from copy import deepcopy

from experiments.world_model_integration.bypass_pr58.pr58_run import (
    read_protocol,compare_map_only
)
from experiments.world_model_integration.stress_pr48.pr48_scenarios import PROTOCOL as STRESS_PROTOCOL
from experiments.world_model_integration.topology_pr56.pr56_public_map import (
    build_topology_business
)


class StationFreeBypassProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg,cls.old=read_protocol()
        cls.stress=json.loads(STRESS_PROTOCOL.read_text())

    def test_matching_all_original_pr57_cases_including_censored(self):
        new=self.cfg["cohort"]
        original=self.old["cohort"]
        self.assertEqual(len(new),len(original))
        self.assertEqual({x["paired_pr56_original_case_id"] for x in new},
                         {x["id"] for x in original})
        self.assertTrue(all(x["split"]=="paired_topology_development_only"
                            for x in new))
        self.assertEqual(len({x["seed"] for x in new}),12)
        self.assertEqual(self.cfg["status"]["gpu_hours"],0)
        self.assertTrue(self.cfg["status"]["original_5_truncated_scenes_preserved"])

    def test_four_new_layouts_are_2connected_no_station_cuts_and_real_route_detours(self):
        for x in self.cfg["cohort"]:
            result=compare_map_only(x,self.cfg,self.old,self.stress)
            self.assertTrue(result["same_true_business_outages_jobs_robots_rules"])
            self.assertEqual(result["static_articulation_count"],0)
            self.assertGreaterEqual(result["station_geodesic_detours"],1)
            self.assertEqual(x["seed"],result["same_random_seed"])
        self.assertEqual(len({x["layout"] for x in self.cfg["cohort"]}),4)

    def test_original_scenarios_do_not_change_except_map_obstacles(self):
        for spec in self.cfg["cohort"]:
            orig=next(x for x in self.old["cohort"]
                      if x["id"]==spec["paired_pr56_original_case_id"])
            old,old_geo=build_topology_business(orig,self.stress,self.old)
            new,new_geo=build_topology_business(spec,self.stress,self.cfg)
            self.assertEqual(old.rules,new.rules)
            self.assertEqual(old.jobs,new.jobs)
            self.assertEqual(old.scenario.stations,new.scenario.stations)
            self.assertEqual(old.scenario.robots,new.scenario.robots)
            self.assertEqual(old.scenario.max_ticks,new.scenario.max_ticks)
            self.assertNotEqual(old.scenario.grid.blocked,new.scenario.grid.blocked)

    def test_reject_station_blockage_at_preregistered_stage(self):
        modified=deepcopy(self.cfg)
        modified["layouts"][0]["blocked"].append([3,2])
        p=next(x for x in self.cfg["cohort"] if x["layout"]=="west_detour")
        with self.assertRaises(AssertionError):
            compare_map_only(p,modified,self.old,self.stress)

    def test_map_only_development_not_blind_test_or_new_gpu_training(self):
        self.assertTrue(self.cfg["status"]["all_12_development_only"])
        self.assertTrue(self.cfg["status"]["no_independent_heldout_claim"])
        self.assertTrue(self.cfg["status"]["no_new_model_training"])
        self.assertTrue(self.cfg["status"]["no_bulk_data_generation"])
        self.assertEqual(self.cfg["actual_legal_alternatives"]["candidate_limit"],4)
        self.assertEqual(self.cfg["max_ticks"],260)


if __name__=="__main__":
    unittest.main()
