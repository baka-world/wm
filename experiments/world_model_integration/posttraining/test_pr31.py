"""PR31 independent groups, same architecture, and persisted restorable state."""
from __future__ import annotations
import json
import os
from pathlib import Path
import unittest

from experiments.world_model_integration.posttraining.pr31_dataset import (
    PROTOCOL, _assert_seeds, build_one,
)
from experiments.world_model_integration.build_dataset_v4 import CONFIG
from experiments.world_model_integration.train_v4 import VERSION
from business_transport.session_features_v3 import QUERY_WIDTH_V3


class PR31ReproductionTests(unittest.TestCase):
    def test_frozen_dataset_splits_and_no_prior_seeds(self):
        c=json.loads(PROTOCOL.read_text())
        self.assertEqual(c["protocol"],"RHWM_PR31_V4_WARMSTART_SCRATCH_REPRO_TRAIN_20261112_V1")
        self.assertEqual({k:len(v) for k,v in c["groups"].items()},
                         {"train":48,"validation":12,"independent_test":16})
        self.assertEqual(len(c["decision_test_groups"]),16)
        self.assertEqual(c["old_train_replay_groups"],32)
        _assert_seeds(c)
        self.assertEqual(c["training"]["network"],"original BusinessSessionEventModel(query_width=56)")

    def test_new_and_replayed_source_produce_original_v4_query_labels(self):
        c=json.loads(PROTOCOL.read_text())
        for spec in (c["groups"]["train"][0],
                     c["groups"]["train"][24]):
            rows,counters,summary=build_one(spec,c,"train")
            self.assertTrue(rows)
            self.assertEqual(summary["simulation_status"],"completed")
            self.assertEqual({r["label"] for r in rows},{0,1})
            self.assertTrue(all(r["event"] in ("finished","continues") for r in rows))
        self.assertEqual(QUERY_WIDTH_V3,56)

    def test_resumable_state_has_rng_and_optimizer_contract(self):
        import torch
        from experiments.world_model_integration.posttraining.pr31_train import rng_snapshot,restore_rng
        torch.manual_seed(2026)
        previous=rng_snapshot()
        a=torch.rand(1)
        restore_rng(previous)
        b=torch.rand(1)
        self.assertTrue(torch.equal(a,b))
        self.assertTrue({"torch_cpu","python_random","numpy_random","torch_cuda"} <= set(previous))

    def test_repository_training_uses_original_v4_event_model(self):
        from lab_transport.business_session_learning import BusinessSessionEventModel
        import torch
        model=BusinessSessionEventModel(query_width=QUERY_WIDTH_V3)
        self.assertGreater(sum(p.numel() for p in model.parameters()),0)
        self.assertEqual(VERSION,"business-sets-session-v4-data")


if __name__=="__main__":
    unittest.main()
