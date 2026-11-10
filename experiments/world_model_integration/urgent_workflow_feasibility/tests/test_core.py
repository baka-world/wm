"""Actual CPU regression tests. Never train neural checkpoints."""
import unittest
from dataclasses import replace
from pathlib import Path

from experiments.world_model_integration.urgent_workflow_feasibility.core import (
    PROTOCOL,public_spec,scene_for,synthetic_retest,PriorityObserver,
    ConditionalRetestSimulator,job_metrics,run_one,unique_plans,matchkey)
from lab_transport.transition_schema import Assignment


class RetestContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import json
        cls.protocol=json.loads(PROTOCOL.read_text())

    def sample(self,rate=100):
        p=public_spec(0,902001,self.protocol["scenario_max_ticks"])
        p.update(jobs=1,urgent_count=1,retest_rate_percent=rate)
        return p

    def test_keyed_exogenous_does_not_depend_on_tick_policy(self):
        self.assertEqual(synthetic_retest(123,"job-1",35),
                         synthetic_retest(123,"job-1",35))
        self.assertTrue(synthetic_retest(123,"job-1",100))
        self.assertFalse(synthetic_retest(123,"job-1",0))

    def test_priority_observable_after_current_release(self):
        scene,_=scene_for(public_spec(0,902001,165))
        sim=ConditionalRetestSimulator(scene,902001,0)
        observed=[]
        def callback(o):
            observed.append(PriorityObserver().observe(o))
        # Separate observers are used only for this check at tick0.
        sim.run(dispatch_policy=lambda o,c:(),on_observation=lambda o:observed.append(o) if o.tick==0 else None)
        p=PriorityObserver().observe(observed[0])
        self.assertTrue(any(s.priority==2 for s in p.samples))
        self.assertTrue(all(s.deadline_tick is not None for s in p.samples))
        self.assertTrue(all(s.rule_profile in ("urgent","routine") for s in p.samples))
        self.assertTrue(all("retest" not in s.id for s in p.samples))

    def test_no_retest_never_creates_optional_stage(self):
        a=run_one(self.sample(rate=0),self.protocol,"P0")
        self.assertEqual(a["metrics"]["job_count"],1)
        self.assertFalse(a["metrics"]["jobs"][0]["retest"])
        self.assertEqual(len(a["metrics"]["jobs"][0]["stages"]),4)
        self.assertNotIn("retest_stage_created",{e["type"] for e in a["event_rows"]})

    def test_retest_is_created_only_after_detection_completion(self):
        a=run_one(self.sample(rate=100),self.protocol,"P0")
        events=a["event_rows"]
        detections=[e for e in events if e["type"]=="delivered" and
                    e["sample_id"].endswith("::stage3")]
        decisions=[e for e in events if e["type"]=="retest_decision"]
        creations=[e for e in events if e["type"]=="retest_stage_created"]
        self.assertEqual(len(detections),1)
        self.assertEqual(len(decisions),1)
        self.assertEqual(len(creations),1)
        self.assertEqual(decisions[0]["tick"],detections[0]["tick"])
        self.assertTrue(decisions[0]["needs_retest"])
        self.assertEqual(len(a["metrics"]["jobs"][0]["stages"]),5)
        self.assertEqual(a["metrics"]["job_count"],1)
        self.assertLessEqual(creations[0]["tick"],
                             a["metrics"]["jobs"][0]["stages"][3]["completed_tick"]
                             if a["metrics"]["jobs"][0]["stages"][3]["completed_tick"] else
                             self.protocol["scenario_max_ticks"])
        self.assertEqual(a["resource_violations"],0)
        self.assertEqual(a["motion_violations"],0)
        self.assertEqual(a["lifecycle_violations"],0)

    def test_two_policies_see_same_keyed_retest_truth(self):
        s=self.sample(rate=100)
        a=run_one(s,self.protocol,"P0")
        b=run_one(s,self.protocol,"P1")
        self.assertEqual(a["retest_results"],b["retest_results"])

    def test_run_is_bitwise_reproducible(self):
        a=run_one(self.sample(rate=100),self.protocol,"P0")
        b=run_one(self.sample(rate=100),self.protocol,"P0")
        self.assertEqual(a["events_sha256"],b["events_sha256"])
        self.assertEqual(a["history_sha256"],b["history_sha256"])

    def test_metric_censoring_job_not_stage(self):
        spec=self.sample(rate=100)
        a=run_one(spec,self.protocol,"P0")
        self.assertEqual(a["metrics"]["urgent"]["n"],1)
        self.assertEqual(a["metrics"]["job_count"],1)
        self.assertEqual(a["metrics"]["uncompleted_count"],
                         sum(x["censored"] for x in a["metrics"]["jobs"]))
        self.assertEqual(a["metrics"]["all"]["restricted_mean_flow"],
                         a["metrics"]["jobs"][0]["restricted_flow"])

    def test_same_matching_order_not_distinct(self):
        p=(Assignment("R1","A"),Assignment("R2","B"))
        self.assertEqual(matchkey(p),matchkey(tuple(reversed(p))))
        self.assertEqual(len(unique_plans([p,tuple(reversed(p))],p)),1)

    def test_priority_and_deadline_survive_retest(self):
        a=run_one(self.sample(rate=100),self.protocol,"P0")
        jobs=a["metrics"]["jobs"]
        self.assertEqual(len(jobs),1)
        self.assertEqual(jobs[0]["priority_class"],"urgent")
        self.assertEqual(jobs[0]["deadline_tick"],jobs[0]["release_tick"]+
                         self.sample()["urgent_due_offset"])
        self.assertEqual(len(jobs[0]["stages"]),5)

    def test_no_future_retest_in_policy_observations(self):
        a=run_one(self.sample(rate=100),self.protocol,"P0")
        detect=next(e["tick"] for e in a["event_rows"] if e["type"]=="retest_decision")
        before=[d for d in a["decisions"] if d["tick"]<detect]
        self.assertTrue(before)
        self.assertTrue(all(d["retest_visible"]==0 for d in before))
        self.assertFalse(any("retest_results" in d for d in a["decisions"]))


if __name__=="__main__":
    unittest.main()
