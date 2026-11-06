from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from lab_transport.domain import Grid, Sample, load_scenario
from lab_transport.report import write_run
from lab_transport.simulation import Simulator
from lab_transport.world_model import PublicStation, WorldModel


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "scenarios/laboratory_demo.json"


class ScenarioTests(unittest.TestCase):
    def test_invalid_and_disconnected_inputs_are_rejected(self):
        source = json.loads(DEMO.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "scenario.json"
            invalid = dict(source)
            invalid["robots"] = [source["robots"][0], source["robots"][0]]
            path.write_text(json.dumps(invalid), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unique"):
                load_scenario(path)
            invalid = dict(source)
            invalid["grid"] = {"width": 9, "height": 7, "blocked": [[4, y] for y in range(7)]}
            path.write_text(json.dumps(invalid), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "connected"):
                load_scenario(path)


class WorldModelTests(unittest.TestCase):
    def test_forecast_depends_on_action_and_updates_from_observed_service(self):
        model = WorldModel(Grid(5, 3, frozenset()), {
            "A": PublicStation((0, 0), 1), "B": PublicStation((4, 0), 2),
        })
        sample = Sample("S", "A", "B", 0, 20)
        close = model.predict(0, "R1", (0, 1), sample)
        far = model.predict(0, "R2", (3, 2), sample)
        self.assertLess(close["expected_completion_tick"], far["expected_completion_tick"])
        model.observe_service("A", 7)
        revised = model.predict(0, "R1", (0, 1), sample)
        self.assertGreater(revised["expected_completion_tick"], close["expected_completion_tick"])
        observations = model.observations()
        observations["A"].append(99)
        self.assertEqual(model.observations()["A"], [7])

    def test_forecast_does_not_access_future_outage_or_actual_duration(self):
        original = json.loads(DEMO.read_text(encoding="utf-8"))
        altered = json.loads(json.dumps(original))
        altered["stations"][0]["service_ticks"] = 99
        altered["stations"][0]["outages"] = [[0, 200]]
        with tempfile.TemporaryDirectory() as temp:
            first, second = Path(temp) / "a.json", Path(temp) / "b.json"
            first.write_text(json.dumps(original), encoding="utf-8")
            second.write_text(json.dumps(altered), encoding="utf-8")
            a, b = Simulator(load_scenario(first)), Simulator(load_scenario(second))
            sample = next(iter(a.samples.values())).spec
            self.assertEqual(a.model.predict(0, "R1", (0, 0), sample),
                             b.model.predict(0, "R1", (0, 0), sample))

    def test_dispatch_does_not_observe_hidden_service_countdown(self):
        scenario = load_scenario(DEMO)
        decisions = []
        for hidden_remaining in (1, 50):
            simulator = Simulator(scenario, "world")
            robot = simulator.robots["R1"]
            robot.position = scenario.stations["收样处"].position
            robot.task_id, robot.phase = "S01", "to_pickup"
            robot.service_started, robot.service_remaining = 0, hidden_remaining
            simulator.samples["S01"].status = "assigned"
            simulator.samples["S02"].status = "waiting"
            simulator._dispatch(1)
            decisions.append(simulator.events[-1]["candidates"])
        self.assertEqual(decisions[0], decisions[1])


class SimulationTests(unittest.TestCase):
    def setUp(self):
        self.scenario = load_scenario(DEMO)

    def test_six_samples_finish_with_valid_lifecycles_and_grid_safety(self):
        for policy in ("nearest", "world"):
            with self.subTest(policy=policy):
                result = Simulator(self.scenario, policy, seed=42).run()
                self.assertEqual(result["summary"]["status"], "completed")
                self.assertEqual(result["summary"]["completed_count"], 6)
                for sample in result["samples"]:
                    self.assertEqual(sample["status"], "completed")
                    self.assertLessEqual(sample["release_tick"], sample["assigned_tick"])
                    self.assertLess(sample["assigned_tick"], sample["picked_tick"])
                    self.assertLess(sample["picked_tick"], sample["completed_tick"])
                    for kind in ("assigned", "picked_up", "delivered"):
                        events = [e for e in result["events"] if e["type"] == kind and e["sample_id"] == sample["id"]]
                        self.assertEqual(len(events), 1)
                for previous, current in zip(result["history"], result["history"][1:]):
                    before = {r["id"]: tuple(r["position"]) for r in previous["robots"]}
                    after = {r["id"]: tuple(r["position"]) for r in current["robots"]}
                    self.assertEqual(len(set(after.values())), len(after))
                    ids = sorted(before)
                    for i, first in enumerate(ids):
                        self.assertTrue(self.scenario.grid.contains(after[first]))
                        for second in ids[i + 1:]:
                            self.assertFalse(before[first] == after[second] and before[second] == after[first])
                if policy == "world":
                    self.assertIsNotNone(result["summary"]["forecast_mae_ticks"])
                    decisions = [e for e in result["events"] if e["type"] == "assigned"]
                    self.assertTrue(all(e["candidates"] for e in decisions))

    def test_reproducible_and_duration_draws_match_across_policies(self):
        first = Simulator(self.scenario, "world", seed=42).run()
        self.assertEqual(first, Simulator(self.scenario, "world", seed=42).run())
        without_outages = replace(self.scenario, stations={
            key: replace(station, outages=()) for key, station in self.scenario.stations.items()
        })
        by_policy = {}
        for policy in ("world", "nearest"):
            result = Simulator(without_outages, policy, seed=7).run()
            by_policy[policy] = {(e["sample_id"], e["phase"], e["station_id"]): e["observed_ticks"]
                                 for e in result["events"] if e["type"] == "service_finished"}
        self.assertEqual(by_policy["world"], by_policy["nearest"])
        self.assertEqual(len(by_policy["world"]), 12)

    def test_demo_completes_across_service_time_seeds(self):
        for policy in ("nearest", "world"):
            for seed in range(20):
                with self.subTest(policy=policy, seed=seed):
                    result = Simulator(self.scenario, policy, seed).run()
                    self.assertEqual(result["summary"]["status"], "completed")

    def test_outage_delays_service_and_becomes_an_observation(self):
        raw = {"name": "test", "grid": {"width": 4, "height": 2, "blocked": []},
               "stations": [{"id": "A", "position": [1, 0], "service_ticks": 1,
                             "prior_ticks": 1, "outages": [[0, 3]]},
                            {"id": "B", "position": [3, 0], "service_ticks": 1, "prior_ticks": 1}],
               "robots": [{"id": "R", "position": [0, 0], "home": [0, 0]}],
               "samples": [{"id": "S", "pickup": "A", "dropoff": "B", "release_tick": 0,
                            "deadline_tick": 20}], "max_ticks": 30}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "s.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            result = Simulator(load_scenario(path), "world").run()
        self.assertEqual(result["samples"][0]["picked_tick"], 4)
        self.assertEqual(result["station_observations"]["A"], [3])
        self.assertEqual(result["summary"]["status"], "completed")

    def test_robots_can_leave_each_others_occupied_destinations(self):
        raw = {"name": "opposing deliveries", "grid": {"width": 5, "height": 2, "blocked": []},
               "stations": [{"id": "A", "position": [0, 0], "service_ticks": 1},
                            {"id": "B", "position": [4, 0], "service_ticks": 1}],
               "robots": [{"id": "R1", "position": [0, 0], "home": [0, 1]},
                          {"id": "R2", "position": [4, 0], "home": [4, 1]}],
               "samples": [{"id": "S1", "pickup": "A", "dropoff": "B", "release_tick": 0, "deadline_tick": 30},
                           {"id": "S2", "pickup": "B", "dropoff": "A", "release_tick": 0, "deadline_tick": 30}],
               "max_ticks": 50}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "s.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            result = Simulator(load_scenario(path), "nearest").run()
        self.assertEqual(result["summary"]["completed_count"], 2)
        self.assertEqual(result["summary"]["status"], "completed")

    def test_timeout_is_not_reported_as_success(self):
        samples = (replace(self.scenario.samples[0], deadline_tick=0),) + self.scenario.samples[1:]
        result = Simulator(replace(self.scenario, samples=samples, max_ticks=1), "world").run()
        self.assertEqual(result["summary"]["status"], "timed_out")
        self.assertLess(result["summary"]["completed_count"], result["summary"]["sample_count"])
        self.assertEqual(result["summary"]["overdue_unfinished_count"], 1)

    def test_simulator_cannot_be_reused_with_stale_state(self):
        simulator = Simulator(self.scenario, "world")
        simulator.run()
        with self.assertRaisesRegex(RuntimeError, "only be run once"):
            simulator.run()


class ReportTests(unittest.TestCase):
    def test_report_escapes_scenario_text(self):
        malicious = '</script><script>alert("x")</script>'
        scenario = replace(load_scenario(DEMO), name=malicious)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "out"
            write_run(Simulator(scenario).run(), output)
            page = (output / "report.html").read_text(encoding="utf-8")
            self.assertNotIn(malicious, page)
            self.assertIn(r'\u003c/script\u003e', page)

    def test_cli_compare_creates_results_and_will_not_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "run"
            command = [sys.executable, "-m", "lab_transport", "compare", "--scenario", str(DEMO),
                       "--seed", "42", "--output", str(output)]
            first = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
            self.assertEqual(first.returncode, 0, first.stderr)
            comparison = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
            self.assertEqual(comparison["policies"]["world"]["completed_count"], 6)
            self.assertTrue((output / "world/report.html").is_file())
            before = (output / "comparison.json").read_bytes()
            second = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
            self.assertEqual(second.returncode, 2)
            self.assertEqual(before, (output / "comparison.json").read_bytes())


if __name__ == "__main__":
    unittest.main()
