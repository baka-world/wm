"""V4 frozen dataset generator contracts (one tiny factual preflight only)."""
import json
from pathlib import Path
import unittest

from business_transport.session_features_v3 import encode_query_v3
from experiments.world_model_integration.build_dataset_v4 import (CONFIG,build_one,
    coverage_of,record_attributes)
from experiments.world_model_integration.build_dataset import case_for


class V4GeneratorContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=json.loads(CONFIG.read_text())

    def test_complete_32_8_8_8_disjoint_frozen_provenance(self):
        c=self.config
        self.assertEqual(c["protocol"],"pr18-v4-frozen-synthetic-mechanism-20261110")
        self.assertEqual([len(c["groups"][k]) for k in
            ("train","validation","independent_in","independent_ood")],[32,8,8,8])
        groups=[(k,s) for k,v in c["groups"].items() for s in v]
        self.assertEqual(len({s["seed"] for _,s in groups}),len(groups))
        self.assertTrue(all(s["seed"] not in c["old_seeds_forbidden"] for _,s in groups))
        self.assertTrue(all(s["layout"] in (5,6) for s in c["groups"]["independent_in"]))
        self.assertTrue(all(s["layout"] in (7,8) for s in c["groups"]["independent_ood"]))
        self.assertEqual(set(v for s in c["groups"]["independent_ood"]
                             for v in s["station_service_ticks"].values()),{9,10})
        self.assertEqual(set(v for s in c["groups"]["train"]
                             for v in s["station_service_ticks"].values()),
                             set(range(3,9)))
        for s in c["groups"]["train"]:
            self.assertEqual(set(s["station_service_ticks"]),set(c["service_stations"]))
            self.assertIn(s["jitter_ticks"],(0,1,2))
            self.assertIn(s["resource_capacity_A"],(1,2))
            self.assertIn(s["robot_count"],(1,2,3))

    def test_preflight_one_real_group_and_public_priors(self):
        c=self.config
        s=c["groups"]["train"][2]   # multiple robots; device outage during service
        business,geometry=case_for(s,c)
        self.assertEqual({id:station.prior_ticks for id,station in business.scenario.stations.items()},
                         s["station_service_ticks"])
        rows,counts,source=build_one(s,c,"train")
        self.assertGreater(len(rows),5)
        self.assertGreater(sum(r["label"] for r in rows),0)
        self.assertGreater(len(rows)-sum(r["label"] for r in rows),0)
        self.assertGreater(source["observed_paused_session_ticks"],0,
             "forced early outage must actually pause a factual service")
        for row in rows:
            a=record_attributes(row)
            self.assertEqual(a["prior"],s["station_service_ticks"][a["station"]])
            self.assertEqual(row["label"],int(row["event"]=="finished"))
            self.assertEqual(len(encode_query_v3(row["features"],row["query"])),56)
            self.assertEqual(row["features"]["unknown"]["true_remaining_service_countdown"],
                             {"known":False,"value":None})
        report=coverage_of(rows,counts,[source])
        self.assertGreater(report["axes"]["prior"][str(s["station_service_ticks"]["A0"])]["total"],0)
        self.assertTrue("prior_elapsed" in report["axes"])

    def test_novel_ood_layout_compiles_with_three_stage_workflow(self):
        c=self.config
        s=next(x for x in c["groups"]["independent_ood"] if x["workflow_variant"]=="three")
        business,geo=case_for(s,c)
        self.assertGreaterEqual(max(len(route) for route in business.rules.workflows.values()),3)
        self.assertTrue(all(station.prior_ticks in (9,10) for station in business.scenario.stations.values()))


if __name__=="__main__":unittest.main()
