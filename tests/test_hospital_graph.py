from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from lab_transport.domain import DirectedGraph, RobotSpec, Sample, Scenario, Station, load_scenario
from lab_transport.hospital_case import parse_pinned_building
from lab_transport.report import write_run
from lab_transport.simulation import Simulator


class HospitalGraphTests(unittest.TestCase):
    def test_direction_and_source_integrity(self):
        a, b, c = (0, 0), (1, 0), (2, 0)
        graph = DirectedGraph({a: (1, 1), b: (2, 1), c: (3, 1)}, ((a, b), (b, c)))
        self.assertEqual(graph.shortest_path(a, c), [b, c])
        self.assertIsNone(graph.shortest_path(c, a))
        self.assertIsNone(graph.shortest_path(a, c, frozenset({b})))
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            parse_pinned_building(b"not the published source")

    def test_missing_deadline_is_not_reported_as_a_clinical_success(self):
        root = Path(__file__).resolve().parents[1]
        scenario = load_scenario(root / "scenarios/laboratory_demo.json")
        samples = tuple(replace(s, deadline_tick=None) for s in scenario.samples)
        result = Simulator(replace(scenario, samples=samples)).run()
        self.assertEqual(result["summary"]["deadline_sample_count"], 0)
        self.assertIsNone(result["summary"]["late_count"])
        self.assertIsNone(next(e for e in result["events"] if e["type"] == "assigned")
                          ["selected"]["predicted_lateness"])
        with tempfile.TemporaryDirectory() as temp:
            write_run(result, temp)
            self.assertIn("未设置时限", (Path(temp) / "report.html").read_text())

    def test_two_robot_fallback_can_clear_a_blocked_bottleneck(self):
        a, b, c, pocket = ((i, 0) for i in range(4))
        directed = ((a, b), (b, a), (b, c), (c, b), (b, pocket), (pocket, b))
        graph = DirectedGraph({a: (0, 0), b: (1, 0), c: (2, 0), pocket: (1, 1)}, directed)
        stations = {"pickup": Station("pickup", a, 1, 1, ()),
                    "dropoff": Station("dropoff", c, 1, 1, ())}
        scenario = Scenario("artificial unit-test graph", graph, stations,
                            (RobotSpec("R1", a, a), RobotSpec("R2", b, b)),
                            (Sample("S1", "pickup", "dropoff", 0, None),
                             Sample("S2", "pickup", "dropoff", 0, None)), 0, 50)
        sim = Simulator(scenario, "world")
        sim.robots["R1"].task_id, sim.robots["R1"].phase = "S1", "to_dropoff"
        sim.robots["R2"].task_id, sim.robots["R2"].phase = "S2", "to_pickup"
        for tick in range(8):
            before = {r.id: r.position for r in sim.robots.values()}
            sim._move(tick, set())
            after = {r.id: r.position for r in sim.robots.values()}
            self.assertEqual(len(set(after.values())), 2)
            for rid in before:
                self.assertTrue(before[rid] == after[rid] or (before[rid], after[rid]) in directed)
            if after["R1"] == c:
                break
        self.assertEqual(sim.robots["R1"].position, c)
        self.assertTrue(any(e["type"] == "recovery_planned" for e in sim.events))


if __name__ == "__main__":
    unittest.main()
