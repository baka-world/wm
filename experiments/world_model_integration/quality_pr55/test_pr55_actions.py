"""PR55 second gate: no false rank labels and exact public future differences."""
from __future__ import annotations
from pathlib import Path
import json
import unittest

from experiments.world_model_integration.quality_pr55.pr55_generate import (
    PROTOCOL as QUALITY_PROTOCOL
)
from experiments.world_model_integration.quality_pr55.pr55_action_quality import (
    PROTOCOL,validate_action,state_difference
)


class PairedActionDataContracts(unittest.TestCase):
    def test_all_18_pilot_scenarios_are_accounted_for_precommitment(self):
        p=json.loads(PROTOCOL.read_text())
        q=json.loads(QUALITY_PROTOCOL.read_text())
        self.assertTrue(validate_action(p,q))
        self.assertEqual(len(p["ids"]),18)
        self.assertEqual(p["minimum_scenes_with_true_h20_entity_effect"],12)
        self.assertEqual(p["minimum_scenes_with_true_final_makespan_effect"],9)
        self.assertEqual(p["resource"]["GPU"],0)

    def test_true_state_effect_is_only_on_jointly_visible_labeled_nodes(self):
        def snap(value,mask):
            return {"5":{
                "node_target":[[value,0,0,0,0,0]],
                "node_target_mask":[[mask,0,0,0,0,0]],
                "global_total_completed_stage_count":0,
                "global_newly_visible_stage_count":0,
                "global_new_stage_completed_count":0
            }}
        a=snap(0.,1);b=snap(1.,1)
        self.assertTrue(state_difference(a,b,"5")["nonzero_true_public_state_effect"])
        self.assertEqual(state_difference(a,b,"5")["changed_public_fields"],1)
        c=snap(1.,0)
        with self.assertRaises(AssertionError):
            state_difference(a,c,"5")
        self.assertIsNone(state_difference(a,b,"20"))

    def test_newly_visible_anonymous_event_is_counted_without_future_job_identity(self):
        a={"20":{"node_target":[[0.]*6],"node_target_mask":[[1.]*6],
                 "global_total_completed_stage_count":0,
                 "global_newly_visible_stage_count":0,
                 "global_new_stage_completed_count":0}}
        b=json.loads(json.dumps(a))
        b["20"]["global_newly_visible_stage_count"]=1
        change=state_difference(a,b,"20")
        self.assertEqual(change["changed_public_fields"],0)
        self.assertEqual(change["anonymous_future_event_l1"],1.)
        self.assertTrue(change["nonzero_true_public_state_effect"])

    def test_unlabeled_future_is_censored_not_assumed_equal(self):
        a={"5":{"node_target":[[0.]*6],"node_target_mask":[[1.]*6],
                "global_total_completed_stage_count":0,
                "global_newly_visible_stage_count":0,
                "global_new_stage_completed_count":0}}
        self.assertIsNone(state_difference(a,{},"5"))


if __name__=="__main__":
    unittest.main()
