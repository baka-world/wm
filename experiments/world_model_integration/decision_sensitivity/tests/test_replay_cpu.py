"""Check executed original V4 CPU replay, never synthesize validation evidence."""
from __future__ import annotations
import json
import os
import random
import unittest
from pathlib import Path
from experiments.world_model_integration.decision_sensitivity.postprocess_replay import summarize


class FrozenCPUTraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=os.environ.get("WM_PR18_REPLAY_JSON")
        if not path:
            raise RuntimeError("WM_PR18_REPLAY_JSON required: no fabricated test input")
        cls.actual=json.loads(Path(path).read_text())
        cls.checked=summarize(cls.actual)

    def test_original_five_step_scores_all_match(self):
        self.assertEqual([x["seed"] for x in self.actual["cases"]], [43011,43015,43019])
        for case in self.actual["cases"]:
            self.assertEqual(case["windows"]["5"]["recorded_archive_5step_parity"],
                             {"Rule":True,"V4":True})

    def test_network_forward_calls_only_when_eligible(self):
        for case in self.checked["cases"]:
            for window in case["windows"].values():
                self.assertTrue(window["instrumentation_checks"]["forward_batches_one_per_nonempty_service_batch"])
                self.assertGreater(window["v4_query_calls"],0)

    def test_sampled_rng_stream_matches_unmodified_seed(self):
        for case in self.actual["cases"]:
            for window in case["windows"].values():
                for candidate in window["runs"]["V4"]["traces"]:
                    for particle,steps in enumerate(candidate):
                        rng=random.Random(1810+particle)
                        for step in steps:
                            for q in step["session"]["queries"]:
                                self.assertAlmostEqual(q["u"],rng.random(),places=15)
                                self.assertEqual(q["sampled_finish"],q["u"]<q["p_v4"])
                                self.assertEqual(q["v4_transition_finished"],int(q["sampled_finish"]))

    def test_trace_snapshots_are_not_retroactively_mutated(self):
        for case in self.actual["cases"]:
            for window in case["windows"].values():
                for arm in ("Rule","V4"):
                    for candidate in window["runs"][arm]["traces"]:
                        for particle_steps in candidate:
                            for left,right in zip(particle_steps,particle_steps[1:]):
                                self.assertEqual(left["after"],right["before"])

    def test_identical_input_events_and_recursive_differences_distinct(self):
        a=next(c for c in self.checked["cases"] if c["seed"]==43011)["windows"]["5"]
        event=a["first_same_input_event_difference"]
        state=a["first_recursive_state_difference"]
        self.assertTrue(event["same_input_as_recursive_rule"])
        self.assertEqual(event["step"],3)
        self.assertEqual(state["step"],3)
        self.assertTrue(state["same_input_before_step"])

    def test_label_is_development_not_new_independent(self):
        self.assertEqual(self.checked["evidence_scope"],
                         "development_deterministic_cpu_replay_not_independent_confirmation")
        self.assertEqual([x["windows"]["5"]["classification"] for x in self.checked["cases"]],
                         ["F3","F2","F2"])
        for case in self.checked["cases"]:
            for horizon in ("10","20"):
                self.assertEqual(case["windows"][horizon]["classification"],"F4")


if __name__=="__main__": unittest.main()
