from collections import Counter
from dataclasses import replace
from pathlib import Path
import unittest

from lab_transport.clinic_map import make_clinic_scenario
from lab_transport.domain import load_scenario
from lab_transport.experiment import TRAIN_SEEDS
from lab_transport.multisite_experiment import DROPOFFS, PICKUPS, SEEDS, discrete_safety_violations, make_case
from lab_transport.simulation import Simulator

ROOT = Path(__file__).resolve().parents[1]


class MultiSiteRolloutTests(unittest.TestCase):
    def setUp(self):
        self.scenario = load_scenario(ROOT / "scenarios/laboratory_demo.json")

    def test_rejects_unpinned_clinic_sources(self):
        with self.assertRaisesRegex(ValueError, "map SHA-256"):
            make_clinic_scenario(b"unexpected", b"unexpected")

    def test_fixed_case_is_paired_and_training_seeds_are_disjoint(self):
        source = self.scenario.stations["收样处"]
        sink = self.scenario.stations["检验区"]
        proxy = replace(self.scenario, stations={
            name: replace(source, id=name) for name in PICKUPS["clinic"]
        } | {name: replace(sink, id=name) for name in DROPOFFS["clinic"]})
        normal = make_case(proxy, "clinic", 2000, "normal")
        short = make_case(proxy, "clinic", 2000, "short_outage")
        self.assertEqual(normal.samples, short.samples)
        self.assertEqual(Counter(s.pickup for s in normal.samples), {name: 3 for name in PICKUPS["clinic"]})
        self.assertEqual(Counter(s.dropoff for s in normal.samples), {name: 3 for name in DROPOFFS["clinic"]})
        self.assertTrue(all(s.deadline_tick is None for s in normal.samples))
        self.assertEqual(sum(bool(s.outages) for s in short.stations.values()), 1)
        self.assertFalse(set(TRAIN_SEEDS) & {seed for values in SEEDS.values() for seed in values})
        self.assertEqual(short, make_case(proxy, "clinic", 2000, "short_outage"))

    def test_rollout_cannot_observe_future_or_hidden_service_countdown(self):
        results = []
        for hidden, outage, release in ((1, (100, 120), 101), (50, (200, 220), 201)):
            source = self.scenario.stations["收样处"]
            altered = replace(self.scenario,
                stations={**self.scenario.stations, "收样处": replace(source, outages=(outage,), service_ticks=hidden)},
                samples=tuple(replace(s, release_tick=release) if i >= 2 else s
                              for i, s in enumerate(self.scenario.samples)))
            sim = Simulator(altered, "rollout")
            active = sim.robots["R1"]
            active.position = altered.stations["收样处"].position
            active.task_id, active.phase = "S01", "to_pickup"
            active.service_started, active.service_remaining = 0, hidden
            sim.samples["S01"].status, sim.samples["S01"].robot_id = "assigned", "R1"
            sim.samples["S02"].status = "waiting"
            before = (active.position, active.service_remaining, sim.samples["S02"].status)
            results.append(sim._observable_rollout(1, sim.robots["R2"], sim.samples["S02"]))
            self.assertEqual(before, (active.position, active.service_remaining, sim.samples["S02"].status))
        self.assertEqual(results[0], results[1])

    def test_horizon_is_unknown_not_a_completion_or_infeasibility(self):
        sim = Simulator(self.scenario, "rollout")
        sample = sim.samples["S01"]
        sample.status = "waiting"
        forecast = sim._observable_rollout(0, sim.robots["R1"], sample, horizon=1)
        self.assertTrue(forecast["rollout_horizon_reached"])
        self.assertEqual(forecast["predicted_final_sample_status"], "unknown")
        self.assertIsNone(forecast["expected_completion_tick"])
        self.assertEqual(forecast["score"], 1)

    def test_rollout_runs_closed_loop_and_preserves_discrete_safety(self):
        result = Simulator(self.scenario, "rollout", seed=42).run()
        self.assertEqual(result["summary"]["completed_count"], 6)
        self.assertEqual(discrete_safety_violations(result), 0)
        decisions = [e for e in result["events"] if e["type"] == "assigned"]
        self.assertTrue(all(e["selected"]["predictor"] == "observable_rollout" for e in decisions))


if __name__ == "__main__":
    unittest.main()
