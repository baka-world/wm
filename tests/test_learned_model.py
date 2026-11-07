from dataclasses import replace
from pathlib import Path
import unittest

from lab_transport.domain import Sample, Station, load_scenario
from lab_transport.experiment import TEST_SEEDS, TRAIN_SEEDS, SHIFT_SEEDS, make_case, observations
from lab_transport.learned_model import FEATURES, corrected_forecast, train, validate
from lab_transport.simulation import Simulator

ROOT = Path(__file__).resolve().parents[1]


def features(waiting):
    result = {name: 0 for name in FEATURES}
    result.update({"distance_to_pickup": 1, "distance_to_dropoff": 2,
                   "pickup_service_est": 1, "dropoff_service_est": 1, "waiting_count": waiting})
    return result


class LearnedModelTests(unittest.TestCase):
    def test_training_generalizes_simple_synthetic_relation_and_preserves_order(self):
        records = [{"features": features(i), "pickup_residual": float(i), "completion_residual": float(2 * i)}
                   for i in range(8)]
        model = train(records, ridge_lambda=0.01)
        baseline = {"training_features": features(8), "expected_pickup_tick": 10.0,
                    "expected_completion_tick": 20.0}
        forecast = corrected_forecast(baseline, 0, None, model)
        self.assertLess(abs(forecast["expected_pickup_tick"] - 18), 1)
        self.assertLess(abs(forecast["expected_completion_tick"] - 36), 2)
        self.assertGreaterEqual(forecast["expected_completion_tick"], forecast["expected_pickup_tick"] + 3)
        self.assertEqual(train(records, ridge_lambda=0.01), model)
        corrupt = dict(model, scales=[0] * len(FEATURES))
        with self.assertRaisesRegex(ValueError, "scales"):
            validate(corrupt)

    def test_features_ignore_future_outages_and_future_task_arrivals(self):
        original = load_scenario(ROOT / "scenarios/laboratory_demo.json")
        decisions = []
        for outage, release in (((100, 120), 101), ((200, 220), 201)):
            station = original.stations["检验区"]
            altered = replace(original, stations={**original.stations, "检验区": replace(station, outages=(outage,))},
                              samples=(original.samples[0], replace(original.samples[1], release_tick=release)))
            sim = Simulator(altered, "world", seed=9)
            sim.samples[altered.samples[0].id].status = "waiting"
            sim._dispatch(0)
            decisions.append(sim.events[-1]["candidates"])
        self.assertEqual(decisions[0], decisions[1])

    def test_learned_policy_runs_closed_loop_with_valid_forecasts(self):
        records = [{"features": features(i), "pickup_residual": float(i), "completion_residual": float(2 * i)}
                   for i in range(8)]
        model = train(records)
        scenario = load_scenario(ROOT / "scenarios/laboratory_demo.json")
        with self.assertRaisesRegex(ValueError, "requires trained"):
            Simulator(scenario, "learned")
        result = Simulator(scenario, "learned", 42, model).run()
        self.assertEqual(result["summary"]["status"], "completed")
        self.assertEqual(result["scenario"]["prediction_model"]["kind"], "ridge_residual")
        for event in result["events"]:
            if event["type"] == "assigned":
                selected = event["selected"]
                self.assertGreater(selected["expected_pickup_tick"], event["tick"])
                self.assertGreater(selected["expected_completion_tick"], selected["expected_pickup_tick"])
                self.assertEqual(selected["predictor"], "learned_ridge")

    def test_learned_rule_combines_deadline_order_and_learned_prediction(self):
        records = [{"features": features(i), "pickup_residual": float(i), "completion_residual": float(2 * i)}
                   for i in range(8)]
        model = train(records)
        scenario = load_scenario(ROOT / "scenarios/laboratory_demo.json")
        samples = (replace(scenario.samples[0], deadline_tick=80),
                   replace(scenario.samples[1], release_tick=0, deadline_tick=2))
        scenario = replace(scenario, robots=scenario.robots[:1], samples=samples)
        result = Simulator(scenario, "learned_rule", 42, model).run()
        first = next(e for e in result["events"] if e["type"] == "assigned")
        self.assertEqual(first["sample_id"], samples[1].id)
        self.assertEqual(first["selected"]["predictor"], "learned_ridge")
        self.assertEqual(result["summary"]["completed_count"], 2)

    def test_train_and_test_seeds_are_disjoint_and_reproducible(self):
        self.assertFalse(set(TRAIN_SEEDS) & set(TEST_SEEDS))
        self.assertFalse(set(TRAIN_SEEDS) & set(SHIFT_SEEDS))
        base = load_scenario(ROOT / "scenarios/laboratory_demo.json")
        # The experiment's named stations belong to its source-backed map;
        # compare the generated release sequence and outage via a minimal proxy.
        proxy = replace(base, stations={"ICU取样点": replace(base.stations["收样处"], id="ICU取样点"),
                                        "实验室交付点": replace(base.stations["检验区"], id="实验室交付点")})
        first, second = make_case(proxy, 7, "short_outage"), make_case(proxy, 7, "short_outage")
        self.assertEqual(first, second)

    def test_unfinished_assignments_are_not_treated_as_completed_labels(self):
        scenario = load_scenario(ROOT / "scenarios/laboratory_demo.json")
        result = Simulator(replace(scenario, max_ticks=1), "world").run()
        records, excluded = observations(result)
        self.assertEqual(records, [])
        self.assertEqual(excluded, len(scenario.samples))


if __name__ == "__main__":
    unittest.main()
