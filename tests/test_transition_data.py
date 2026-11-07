from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from lab_transport.domain import Grid, RobotSpec, Sample, Scenario, Station, load_scenario
from lab_transport.experiment import outcome
from lab_transport.multisite_experiment import discrete_safety_violations
from lab_transport.simulation import Simulator
from lab_transport.transition_schema import SCHEMA_VERSION, Assignment

ROOT = Path(__file__).resolve().parents[1]


class TransitionDataTests(unittest.TestCase):
    def setUp(self):
        self.scenario = load_scenario(ROOT / "scenarios/laboratory_demo.json")

    def test_recording_does_not_change_results_and_actions_match_events(self):
        original = Simulator(self.scenario, "world", 42).run()
        rows = []
        recorded = Simulator(self.scenario, "world", 42).run(on_transition=rows.append)
        self.assertEqual(original, recorded)
        self.assertEqual(len(rows), recorded["summary"]["ticks"])
        self.assertEqual(rows[0].observation.tick, 0)
        for first, second in zip(rows, rows[1:]):
            self.assertEqual(first.next_boundary, "before_dispatch")
            self.assertEqual(first.next_observation, second.observation)
            self.assertEqual(first.next_observation.tick, first.observation.tick + 1)
            self.assertIsNone(first.termination)
        self.assertEqual(rows[-1].termination, "completed")
        self.assertEqual(rows[-1].next_boundary, "end_of_run")
        actions = [(row.observation.tick, a.robot_id, a.sample_id) for row in rows for a in row.assignments]
        events = [(e["tick"], e["robot_id"], e["sample_id"]) for e in recorded["events"] if e["type"] == "assigned"]
        self.assertEqual(actions, events)

    def test_future_arrivals_and_private_service_countdowns_are_not_inputs(self):
        observations, private_countdowns = [], []
        for service, outage, future_id, future_release in ((3, (100, 110), "FUTURE_A", 20),
                                                           (9, (200, 210), "FUTURE_B", 50)):
            scenario = Scenario("hidden state test", Grid(5, 2, frozenset()), {
                "A": Station("A", (0, 0), service, 5.0, (outage,), 0),
                "B": Station("B", (4, 0), 1, 1.0, (), 0),
            }, (RobotSpec("R1", (0, 0), (0, 1)),),
                (Sample("S", "A", "B", 0, None), Sample(future_id, "A", "B", future_release, None)), 0, 4)
            rows = []
            result = Simulator(scenario, "world", 7).run(on_transition=rows.append)
            observations.append(rows[1].observation)
            private_countdowns.append(result["history"][1]["robots"][0]["service_remaining"])
            serialized = json.dumps(asdict(rows[1].observation))
            for forbidden in (future_id, "service_remaining", "outages", "jitter", "seed"):
                self.assertNotIn(forbidden, serialized)
            self.assertEqual([s.id for s in rows[1].observation.samples], ["S"])
        self.assertNotEqual(private_countdowns[0], private_countdowns[1])
        self.assertEqual(observations[0], observations[1])

    def test_release_boundary_and_time_limit_are_explicit(self):
        first, second = self.scenario.samples[:2]
        scenario = replace(self.scenario, samples=(first, replace(second, release_tick=1)), max_ticks=1)
        rows = []
        Simulator(scenario, "world").run(on_transition=rows.append)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].termination, "time_limit")
        self.assertEqual(rows[0].next_boundary, "end_of_run")
        self.assertNotIn(second.id, [s.id for s in rows[0].next_observation.samples])

        scenario = replace(scenario, max_ticks=2)
        rows = []
        Simulator(scenario, "world").run(on_transition=rows.append)
        self.assertEqual(rows[0].next_boundary, "before_dispatch")
        self.assertIn(second.id, [s.id for s in rows[0].next_observation.samples])
        self.assertEqual(rows[0].next_observation, rows[1].observation)

    def test_recorded_metadata_is_declared_only_and_excludes_absolute_collection_time(self):
        sample = replace(self.scenario.samples[0], specimen_type="example_type", packaging="example_bag",
                         temperature_condition=None, rule_profile="example_profile",
                         collected_at="2026-01-01T12:00:00Z")
        rows = []
        Simulator(replace(self.scenario, samples=(sample,), max_ticks=1), "world").run(on_transition=rows.append)
        observed = asdict(rows[0].observation.samples[0])
        self.assertEqual(observed["specimen_type"], "example_type")
        self.assertEqual(observed["packaging"], "example_bag")
        self.assertIsNone(observed["temperature_condition"])
        self.assertEqual(observed["rule_profile"], "example_profile")
        self.assertNotIn("collected_at", observed)

    def test_cli_exports_factual_rows_without_overwriting(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "transitions"
            command = [sys.executable, "-m", "lab_transport.transition_data", "--seed", "42", "--output", str(output)]
            first = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
            self.assertEqual(first.returncode, 0, first.stderr)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            rows = [json.loads(line) for line in (output / "transitions.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(manifest["schema_version"], SCHEMA_VERSION)
            self.assertEqual(len(rows), manifest["bookkeeping_only_not_model_input"]["transition_count"])
            self.assertTrue(rows and rows[-1]["termination"] == "completed")
            self.assertNotIn("service_remaining", json.dumps(rows))
            self.assertNotIn("S03", json.dumps(rows[0]["observation"]))
            before = (output / "transitions.jsonl").read_bytes()
            second = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
            self.assertEqual(second.returncode, 2)
            self.assertEqual(before, (output / "transitions.jsonl").read_bytes())

    def test_injected_policy_receives_current_candidates_and_runs_closed_loop(self):
        received = []

        def policy(observation, candidates):
            received.append((observation, candidates))
            robots, samples, chosen = set(), set(), []
            for candidate in candidates:
                if candidate.robot_id not in robots and candidate.sample_id not in samples:
                    chosen.append(candidate)
                    robots.add(candidate.robot_id)
                    samples.add(candidate.sample_id)
            return chosen

        rows = []
        result = Simulator(self.scenario, "world", 42).run(on_transition=rows.append, dispatch_policy=policy)
        self.assertEqual(result["policy"], "external")
        self.assertEqual(result["summary"]["completed_count"], 6)
        self.assertIsNone(outcome(result, self.scenario.max_ticks, 0)["candidate_evaluations"])
        self.assertEqual(discrete_safety_violations(result), 0)
        self.assertTrue(received)
        initial, candidates = received[0]
        self.assertEqual(initial.tick, 0)
        self.assertEqual({s.id for s in initial.samples}, {"S01", "S02"})
        self.assertEqual({a.sample_id for a in candidates}, {"S01", "S02"})
        self.assertNotIn("service_remaining", json.dumps(asdict(initial)))
        assignments = [a for row in rows for a in row.assignments]
        self.assertEqual(len(assignments), 6)
        self.assertTrue(all(e["policy"] == "external" for e in result["events"] if e["type"] == "assigned"))

    def test_invalid_injected_batch_is_rejected_before_any_assignment(self):
        for chooser in (lambda obs, options: [options[0], options[0]],
                        lambda obs, options: [Assignment("unknown", "S01")]):
            simulator = Simulator(self.scenario, "world", 42)
            with self.assertRaisesRegex(ValueError, "injected"):
                simulator.run(dispatch_policy=chooser)
            self.assertTrue(all(robot.task_id is None for robot in simulator.robots.values()))
            self.assertFalse(any(event["type"] == "assigned" for event in simulator.events))

    def test_injected_policy_may_choose_to_wait_and_timeout_is_recorded(self):
        rows = []
        result = Simulator(replace(self.scenario, max_ticks=2), "world").run(
            on_transition=rows.append, dispatch_policy=lambda observation, candidates: ())
        self.assertEqual(result["summary"]["status"], "timed_out")
        self.assertEqual(result["summary"]["completed_count"], 0)
        self.assertTrue(all(not row.assignments for row in rows))
        self.assertEqual(rows[-1].termination, "time_limit")


if __name__ == "__main__":
    unittest.main()
