"""PR55 immutable quality-first design, controlled simulator changes and signal tests."""
from __future__ import annotations
from copy import deepcopy
import json
import unittest

from experiments.world_model_integration.quality_pr55.pr55_generate import (
    PROTOCOL,validate,build_quality_business,flag_transition,
    semantic_fingerprint,curated_indices,spaced_pick
)
from experiments.world_model_integration.stress_pr48.pr48_scenarios import (
    PROTOCOL as STRESS_PROTOCOL
)


class QualityFirstWorldDataContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=json.loads(PROTOCOL.read_text())
        cls.stress=json.loads(STRESS_PROTOCOL.read_text())

    def case(self,profile,regime="congestion"):
        return next(c for c in self.cfg["pilot_cases"]
                    if c["profile"]==profile and c["regime"]==regime)

    def test_pilot_is_not_claimed_as_blind_test_or_GPU_training(self):
        self.assertTrue(validate(self.cfg))
        self.assertEqual(len(self.cfg["pilot_cases"]),18)
        self.assertEqual(len({c["seed"] for c in self.cfg["pilot_cases"]}),18)
        self.assertTrue(all(c["split"]=="pilot_development_only"
                            for c in self.cfg["pilot_cases"]))
        self.assertEqual(self.cfg["resource_budget"]["gpu"],0)
        self.assertFalse(self.cfg["resource_budget"]["not_authorized_full_scale_scenarios"])
        self.assertFalse(self.cfg["quality_gate_frozen_before_pilot"][
            "action_counterfactual_gate"].startswith("PASS"))

    def test_same_business_simulator_changes_real_capacity_robot_and_service(self):
        base,_=build_quality_business(self.case("baseline"),self.stress)
        cap,_=build_quality_business(self.case("capacity_relief"),self.stress)
        robots,_=build_quality_business(self.case("fewer_robots"),self.stress)
        slow,_=build_quality_business(self.case("slow_service"),self.stress)
        self.assertEqual(len(base.scenario.robots),3)
        self.assertEqual(len(robots.scenario.robots),2)
        original_capacity=next(r.capacity for r in base.rules.resources if r.id=="A")
        relaxed_capacity=next(r.capacity for r in cap.rules.resources if r.id=="A")
        self.assertEqual((original_capacity,relaxed_capacity),(1,2))
        self.assertEqual(slow.scenario.stations["B0"].service_ticks,
                         base.scenario.stations["B0"].service_ticks+2)
        self.assertEqual(slow.scenario.stations["C"].prior_ticks,
                         base.scenario.stations["C"].prior_ticks+2)
        self.assertEqual(base.scenario.grid,robots.scenario.grid)
        self.assertEqual(base.scenario.grid,cap.scenario.grid)

    def test_arrival_shift_changes_factual_release_times_not_network_input(self):
        base,_=build_quality_business(self.case("baseline","staggered_arrivals"),
                                      self.stress)
        changed,_=build_quality_business(self.case("shifted_releases","staggered_arrivals"),
                                         self.stress)
        a=[j["release_tick"] for j in base.jobs]
        b=[j["release_tick"] for j in changed.jobs]
        self.assertNotEqual(a,b)
        self.assertEqual(set(b),{0,3,11,23})
        self.assertEqual(self.stress["predefined_scenario_mechanisms"]
                         ["staggered_arrivals"]["release_ticks"],[0,6,12,18])

    def test_actual_event_classifier_catches_fault_recovery_assignment_completion(self):
        nodes=[{"id":"station:0","kind":"station"},
               {"id":"task:0","kind":"task"},
               {"id":"robot:0","kind":"robot"}]
        old=[[1,1,0,0,0,0],[1,0,0,0,0,0],[0,0,0,0,0,0]]
        after=[[0,0,1,1,0.1,0],[0,0,0,1,0,0],[0,0,1,1,0,0]]
        row={"input":{
            "graph":{"nodes":nodes,"candidate_actions":[[
                {"src":"robot:0","dst":"task:0","kind":"proposed_robot_task_assignment"}
            ]]},"node_persistence":old},
            "target":{"node_target":after,"node_mask":[[1]*6]*3,
                      "new_visible_task_stages":1,"new_completed_task_stages":0}}
        flags=flag_transition(row)
        self.assertTrue(set((
            "station_fault_onset","capacity_transition","queue_transition",
            "new_task_release","task_status_change","task_completed",
            "robot_phase_change","assignment"
        )).issubset(flags))
        row["input"]["node_persistence"][0][0]=0
        row["target"]["node_target"][0][0]=1
        self.assertIn("station_recovery",flag_transition(row))

    def test_clock_drift_is_not_mistaken_for_semantic_state_change(self):
        old={"input":{"graph":{
            "nodes":[{"id":"task:0","kind":"task","status":"waiting",
                      "workflow":"w0","stage_index":0,"released_age":1,"deadline_slack":10}],
            "edges":[],"candidate_actions":[[]]}}}
        new=deepcopy(old)
        new["input"]["graph"]["nodes"][0]["released_age"]=99
        new["input"]["graph"]["nodes"][0]["deadline_slack"]=8
        self.assertEqual(semantic_fingerprint(old),semantic_fingerprint(new))
        new["input"]["graph"]["nodes"][0]["status"]="completed"
        self.assertNotEqual(semantic_fingerprint(old),semantic_fingerprint(new))

    def test_curated_preview_does_not_discard_true_raw_episodes(self):
        def row(i,flag):
            return {"quality":{"flags":[flag] if flag else []},
                    "input":{"public_tick":i}}
        records=[row(i,"station_recovery" if i==97 else
                     "station_fault_onset" if i==7 else
                     "new_task_release" if i==61 else
                     "task_status_change" if i==88 else None)
                 for i in range(110)]
        result=curated_indices(records,self.cfg["curated_training_preview"])
        self.assertEqual(len(records),110)
        self.assertTrue({7,61,88,97}.issubset(result))
        self.assertEqual(result,sorted(set(result)))
        self.assertLessEqual(len(result),72)
        self.assertTrue(any(i>56 for i in result))


if __name__=="__main__":
    unittest.main()
