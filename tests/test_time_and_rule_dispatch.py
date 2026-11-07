from dataclasses import replace
import json
from pathlib import Path
import unittest

from lab_transport.domain import DirectedGraph, Grid, RobotSpec, Sample, Scenario, Station
from lab_transport.rule_dispatch import apply_rule_proxies
from lab_transport.simulation import Simulator
from lab_transport.specimen_rules import load_profiles
from lab_transport.time_model import source_parameters, subdivide_graph

ROOT = Path(__file__).resolve().parents[1]


class TimeModelTests(unittest.TestCase):
    def test_subdivision_preserves_direction_and_uses_shared_reverse_lane(self):
        a, b = (0, 0), (1, 0)
        one_way = DirectedGraph({a: (0.0, 0.0), b: (100.0, 0.0)}, ((a, b),))
        timed = subdivide_graph(one_way, metres_per_pixel=0.1, speed_m_per_s=0.5, seconds_per_tick=10)
        path = timed.shortest_path(a, b)
        self.assertEqual(len(path), 2)  # 10 metres at 0.5 m/s over 10-second steps
        self.assertIsNone(timed.shortest_path(b, a))
        both = subdivide_graph(replace(one_way, edges=((a, b), (b, a))), 0.1, 0.5, 10)
        self.assertEqual(both.shortest_path(a, b), [path[0], b])
        self.assertEqual(both.shortest_path(b, a), [path[0], a])
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            source_parameters(b"wrong map", b"wrong config")

    def test_station_specific_jitter_does_not_change_other_station(self):
        scenario = Scenario("synthetic", Grid(4, 1, frozenset()),
                            {"A": Station("A", (1, 0), 3, 4.5, (), 3),
                             "B": Station("B", (3, 0), 1, 1.0, (), 0)},
                            (RobotSpec("R", (0, 0), (0, 0)),), (Sample("S", "A", "B", 0, None),), 9, 30)
        sim = Simulator(scenario)
        draws = {sim._service_duration(f"S{i}", "to_pickup", "A") for i in range(100)}
        self.assertEqual(draws, {3, 4, 5, 6})
        self.assertEqual({sim._service_duration(f"S{i}", "to_dropoff", "B") for i in range(20)}, {1})


class RuleDispatchTests(unittest.TestCase):
    def setUp(self):
        self.profiles = load_profiles(ROOT / "scenarios/specimen-rule-profiles.json")
        self.doc = json.loads((ROOT / "examples/rule-dispatch-synthetic.json").read_text(encoding="utf-8"))
        self.base = Scenario("test", Grid(4, 2, frozenset()),
                             {"ICU取样点": Station("ICU取样点", (1, 0), 1, 1.0, ()),
                              "实验室交付点": Station("实验室交付点", (3, 0), 1, 1.0, ())},
                             (RobotSpec("R", (0, 0), (0, 0)),),
                             (Sample("old", "ICU取样点", "实验室交付点", 0, None),), 0, 1000,
                             {"time_model": "nominal_source_based", "seconds_per_tick": 10})

    def test_explicit_rule_produces_proxy_and_prioritizes_earlier_deadline(self):
        scenario = apply_rule_proxies(self.base, self.doc, self.profiles)
        self.assertEqual([s.deadline_tick for s in scenario.samples], [180, 120])
        by_policy = {}
        for policy in ("nearest", "rule"):
            result = Simulator(scenario, policy).run()
            by_policy[policy] = next(e["sample_id"] for e in result["events"] if e["type"] == "assigned")
        self.assertEqual(by_policy, {"nearest": "S01", "rule": "S02"})
        self.assertIn("不能证明", scenario.metadata["rule_limit"])

    def test_overdue_task_is_not_deprioritized_due_to_its_own_lateness_penalty(self):
        self.doc["samples"][1]["collected_at"] = "2026-01-01T11:20:00Z"
        scenario = apply_rule_proxies(self.base, self.doc, self.profiles)
        self.assertEqual(scenario.samples[1].deadline_tick, -60)
        decisions = {}
        for policy in ("world", "rule"):
            result = Simulator(scenario, policy).run()
            decisions[policy] = next(e["sample_id"] for e in result["events"] if e["type"] == "assigned")
        self.assertEqual(decisions, {"world": "S01", "rule": "S02"})

    def test_rule_requires_matching_type_and_missing_collection_stays_unknown(self):
        self.doc["samples"][0]["specimen_type"] = "blood_unspecified"
        with self.assertRaisesRegex(ValueError, "does not match"):
            apply_rule_proxies(self.base, self.doc, self.profiles)
        self.doc["samples"][0]["specimen_type"] = "pediatric_blood_gas"
        del self.doc["samples"][0]["collected_at"]
        scenario = apply_rule_proxies(self.base, self.doc, self.profiles)
        self.assertIsNone(scenario.samples[0].deadline_tick)
        self.assertEqual(scenario.samples[1].deadline_tick, 120)
        with self.assertRaisesRegex(ValueError, "nominal time model"):
            apply_rule_proxies(replace(self.base, metadata=None), self.doc, self.profiles)


if __name__ == "__main__":
    unittest.main()
