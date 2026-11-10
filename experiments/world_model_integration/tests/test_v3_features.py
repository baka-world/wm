"""CPU contract checks for v3 feature and held-out precommit protocol."""
import json
from pathlib import Path
import unittest
from business_transport.session_features_v3 import (encode_query_v3,INPUT_SCHEMA_V3,QUERY_WIDTH_V3)
from business_transport.session_events import encode_query as encode_query_v2

ROOT=Path(__file__).resolve().parents[1]
class V3FeatureContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # No new synthetic task tuning: existing frozen factual source.
        from experiments.world_model_integration.build_dataset import build_dataset
        cls.data=build_dataset()
    def test_new_target_expected_and_relative_progress_are_numerical(self):
        item=self.data["train"][0]; payload=item["features"]; query=item["query"]
        a=encode_query_v3(payload,query)
        self.assertEqual(len(a),QUERY_WIDTH_V3)
        self.assertEqual(a[:48],encode_query_v2(payload,query))
        self.assertEqual(a[48:50],[0.4,1.0])
        other=json.loads(json.dumps(payload))
        next(s for s in other["observed_facts"]["stations"] if s["id"]==query["station_id"])["expected_service_ticks_public"]["value"]=8
        b=encode_query_v3(other,query)
        self.assertNotEqual(a,b)
        self.assertEqual(a[:48],b[:48],"old v2 had no direct TARGET public expected ticks")
    def test_missing_expected_refuses_and_masked_history_stays_masked(self):
        item=self.data["train"][0];payload=json.loads(json.dumps(item["features"]))
        query=item["query"];station=next(s for s in payload["observed_facts"]["stations"] if s["id"]==query["station_id"])
        self.assertEqual(encode_query_v3(payload,query)[-2:],[0.,0.])
        station["expected_service_ticks_public"]={"known":False,"value":None}
        with self.assertRaisesRegex(ValueError,"required"):encode_query_v3(payload,query)
    def test_grouped_holdout_predeclared_and_new_seeds(self):
        frozen=json.loads((ROOT/"v3_independent_config.json").read_text())
        training=json.loads((ROOT/"dataset_config.json").read_text())
        existing={z["seed"] for group in training["groups"].values() for z in group}
        self.assertFalse(existing.intersection({x["seed"] for x in frozen["groups"]}))
        self.assertTrue(all(x["layout"]>=3 for x in frozen["groups"]))
        self.assertTrue(all(x["seed"] not in (941,947,21071,21073) for x in frozen["groups"]))
if __name__=="__main__":unittest.main()
