"""Freeze PR34 first and prove no previous study seed reuse."""
from __future__ import annotations
import json
from unittest import TestCase,main

from experiments.world_model_integration.posttraining.pr34_online_run import (
    PROTOCOL,_verify_new_scenes,
)


class FrozenNewBudgetStudyTests(TestCase):
    def test_new_independent_groups_and_source_weights(self):
        cfg=json.loads(PROTOCOL.read_text())
        _verify_new_scenes(cfg)
        self.assertEqual(cfg["sample_and_budget"]["groups"],24)
        self.assertEqual(cfg["sample_and_budget"]["gpu"],0)
        self.assertEqual(cfg["sample_and_budget"]["training"],0)
        self.assertEqual(cfg["decision_cpu_budget_seconds"],0.5)
        self.assertEqual(cfg["original_checkpoint_sha256"],
                         "60c8a6c3d832da3e6519072688ea099f64ae120ba19d3bd2ec705f7f1a11ac0c")
        self.assertEqual(len({x["seed"] for x in cfg["groups"]}),24)


if __name__=="__main__":
    main()
