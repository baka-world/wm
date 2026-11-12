"""Official pycox method transfer: source interfaces, public features and actual fit."""
from __future__ import annotations

import json
import math
import unittest

from business_transport.world_model import HybridSystemWorldModel
from business_transport.world_model.pycox_survival import (
    FEATURE_NAMES, PycoxLogisticHazardServiceModel, public_service_features,
)
from experiments.world_model_integration.method_transfer.pycox_run import (
    PROTOCOL, _load_group, public_service_rows,
)
from experiments.world_model_integration.system_world_model.evaluate_horizons import (
    file_sha,
)


class PublishedPycoxTransferTests(unittest.TestCase):
    def test_frozen_train_val_and_two_test_splits_disjoint(self):
        proto=json.loads(PROTOCOL.read_text())
        self.assertEqual(proto["horizons"],[1,5,10])
        self.assertEqual(proto["particles"],3)
        names=("train_groups","validation_groups","iid_test_groups","shift_test_groups")
        groups=[[x["seed"] for x in proto[k]] for k in names]
        self.assertEqual([len(x) for x in groups],[48,12,24,12])
        self.assertEqual(len(set(sum(groups,[]))),96)
        self.assertNotIn(941001,set(sum(groups,[])))
        self.assertEqual(proto["method"].split(" + ")[0],"official pycox.models.LogisticHazard")

    def test_service_feature_schema_does_not_include_future_or_duration_label(self):
        proto=json.loads(PROTOCOL.read_text())
        business,geo,states,actions,factual,rows=_load_group(
            proto["train_groups"][0],proto)
        self.assertTrue(rows)
        self.assertEqual(set(FEATURE_NAMES),{
            "station_A0","station_A1","station_B0","station_B1","station_C",
            "pickup_phase","dropoff_phase",
            "expected_service_ticks_div10","resource_capacity_div3"
        })
        self.assertTrue(all(len(x["features"])==len(FEATURE_NAMES) for x in rows))
        self.assertTrue(all(x["duration"]>=1 for x in rows))
        self.assertEqual(sum(r["event_observed"] for r in rows),len(rows))
        self.assertEqual(set(rows[0]),{
            "station_id","phase","duration","event_observed","features"
        })

    def test_official_pycox_training_and_live_world_model_transition(self):
        import torch
        import pycox
        torch.set_num_threads(1)
        self.assertEqual(pycox.__version__,"0.3.0")
        proto=json.loads(PROTOCOL.read_text())
        a=_load_group(proto["train_groups"][0],proto)
        b=_load_group(proto["train_groups"][1],proto)
        train=a[-1]+b[-1]
        valid=a[-1]
        settings={**proto,"epochs":3,"batch_size":16}
        trained=PycoxLogisticHazardServiceModel.train(
            train,valid,settings=settings)
        self.assertEqual(trained.training_summary["library"],
                         "pycox.models.LogisticHazard")
        self.assertLessEqual(trained.training_summary["epochs_executed"],3)
        business,geo,states,actions,factual,rows=a
        model=HybridSystemWorldModel(
            geo,business.rules,kind="pycox_logistic_hazard",
            service_baseline=trained)
        state=states[0]
        first=actions[0]
        step=model.predict_next(state,first)
        self.assertEqual(step.next_state.tick,1)
        forecast=model.rollout(state,first,
            continuation_policy=lambda obs: (),horizon_ticks=5,particles=2)
        self.assertEqual(forecast.particles[0].final_state.tick,5)
        self.assertEqual(len(forecast.distributions),5)
        self.assertTrue(trained.num_queries>=0)
        self.assertEqual(forecast.model_version,
                         "rhwm-v2-pycox-logistic-hazard-reference")

    def test_unknown_station_fails_instead_of_silent_feature_change(self):
        proto=json.loads(PROTOCOL.read_text())
        business,geo,states,actions,factual,rows=_load_group(
            proto["train_groups"][0],proto)
        with self.assertRaisesRegex(ValueError,"unrecognized"):
            public_service_features(states[0],{
                "station_id":"UNKNOWN","phase":"to_dropoff",
            })


if __name__=="__main__":
    unittest.main()
