from dataclasses import replace
from pathlib import Path
import unittest

from lab_transport.domain import Grid, RobotSpec, Sample, Scenario, Station
from lab_transport.dynamics_model import (MOTION_FEATURES, SERVICE_FEATURES, Geometry, HybridDynamicsModel,
                                          train, training_examples, validate)
from lab_transport.dynamics_experiment import SPLITS, audit_task_schedules, check_predicted_step, compare, eligible_windows
from lab_transport.experiment import SHIFT_SEEDS, TEST_SEEDS, TRAIN_SEEDS
from lab_transport.multisite_experiment import SEEDS
from lab_transport.simulation import Simulator
from lab_transport.transition_schema import Assignment


def manual_parameters():
    motion = [0.0] * len(MOTION_FEATURES)
    motion[MOTION_FEATURES.index("stay")] = -1.0
    motion[MOTION_FEATURES.index("route_progress")] = 2.0
    service = [0.0] * len(SERVICE_FEATURES)
    service[SERVICE_FEATURES.index("bias")] = 1.0
    return {"schema_version": 1, "kind": "hybrid_linear_dynamics", "motion_features": list(MOTION_FEATURES),
            "service_features": list(SERVICE_FEATURES), "motion_weights": motion, "service_weights": service}


class DynamicsModelTests(unittest.TestCase):
    def setUp(self):
        self.scenario = Scenario("simple", Grid(5, 2, frozenset()), {
            "A": Station("A", (0, 0), 1, 1.0, (), 0),
            "B": Station("B", (4, 0), 1, 1.0, (), 0),
        }, (RobotSpec("R1", (0, 0), (0, 1)),), (Sample("S", "A", "B", 0, None),), 0, 30)
        self.rows = []
        Simulator(self.scenario, "world", 42).run(on_transition=self.rows.append)

    def test_one_step_respects_assignment_service_order_and_static_map(self):
        model = HybridDynamicsModel(Geometry(self.scenario.grid), manual_parameters())
        first = self.rows[0]
        prediction = model.predict_next(first.observation, first.assignments)
        self.assertEqual(prediction.tick, 1)
        self.assertEqual(prediction.robots[0].position, (0, 0))
        self.assertEqual(prediction.robots[0].phase, "to_dropoff")
        self.assertEqual(prediction.samples[0].status, "in_transit")
        second = model.predict_next(prediction, ())
        self.assertEqual(second.robots[0].position, (1, 0))
        self.assertEqual(second.samples[0].status, "in_transit")
        self.assertEqual(second.stations, prediction.stations)

    def test_invalid_or_unknown_assignments_are_rejected(self):
        model = HybridDynamicsModel(Geometry(self.scenario.grid), manual_parameters())
        with self.assertRaisesRegex(ValueError, "assignment"):
            model.predict_next(self.rows[0].observation, (Assignment("R1", "unreleased"),))
        with self.assertRaisesRegex(ValueError, "assignment"):
            model.predict_next(self.rows[0].observation, (Assignment("R1", "S"), Assignment("R1", "S")))

    def test_prediction_preserves_discrete_occupancy_and_does_not_invent_tasks(self):
        scenario = replace(self.scenario, robots=(RobotSpec("R1", (0, 1), (0, 1)),
                                                 RobotSpec("R2", (2, 1), (2, 1))),
                           samples=(Sample("S1", "A", "B", 0, None),
                                    Sample("S2", "A", "B", 0, None),
                                    Sample("future", "A", "B", 8, None)))
        rows = []
        Simulator(scenario, "world", 5).run(on_transition=rows.append)
        observation = rows[0].observation
        model = HybridDynamicsModel(Geometry(scenario.grid), manual_parameters())
        predicted = model.predict_next(observation, rows[0].assignments)
        self.assertEqual({s.id for s in predicted.samples}, {s.id for s in observation.samples})
        self.assertNotIn("future", {s.id for s in predicted.samples})
        before = {r.id: r.position for r in observation.robots}
        after = {r.id: r.position for r in predicted.robots}
        self.assertEqual(len(set(after.values())), 2)
        self.assertFalse(before["R1"] == after["R2"] and before["R2"] == after["R1"])
        for rid in before:
            self.assertIn(after[rid], (before[rid], *model.geometry.adjacency[before[rid]]))
        check_predicted_step(observation, predicted, model.geometry)
        invalid = replace(predicted, robots=(predicted.robots[0],
                          replace(predicted.robots[1], position=predicted.robots[0].position)))
        with self.assertRaisesRegex(ValueError, "discrete robot"):
            check_predicted_step(observation, invalid, model.geometry)

    def test_training_is_deterministic_and_learns_simple_separable_examples(self):
        move_a = [0.0] * len(MOTION_FEATURES)
        move_b = [0.0] * len(MOTION_FEATURES)
        move_a[0], move_b[1] = 1.0, 1.0
        service_a = [0.0] * len(SERVICE_FEATURES)
        service_b = [0.0] * len(SERVICE_FEATURES)
        service_a[0], service_b[0], service_b[1] = 1.0, 1.0, 2.0
        moves = [([move_a, move_b], 1)] * 12
        services = [(service_a, 0), (service_b, 1)] * 12
        parameters = train(moves, services)
        self.assertEqual(parameters, train(moves, services))
        self.assertGreater(parameters["motion_weights"][1], parameters["motion_weights"][0])
        self.assertGreater(parameters["service_weights"][1], 0)
        validate(parameters)
        with self.assertRaisesRegex(ValueError, "schema"):
            validate(dict(parameters, motion_features=["wrong"]))
        with self.assertRaisesRegex(ValueError, "motion training"):
            train([([[0.0]], 0)], services)
        with self.assertRaisesRegex(ValueError, "service training"):
            train(moves, [(service_a, 3)])

    def test_extracted_factual_examples_have_valid_labels(self):
        geometry = Geometry(self.scenario.grid)
        moves, services, excluded = [], [], 0
        for row in self.rows:
            m, s, skipped = training_examples(geometry, row.observation, row.assignments, row.next_observation)
            moves.extend(m)
            services.extend(s)
            excluded += skipped
        self.assertTrue(moves and services)
        self.assertEqual(excluded, 0)
        self.assertTrue(all(0 <= label < len(options) for options, label in moves))
        self.assertEqual({label for _, label in services}, {1})

    def test_sample_identifiers_only_match_entities_not_numeric_features(self):
        model = HybridDynamicsModel(Geometry(self.scenario.grid), manual_parameters())
        original = self.rows[0].observation
        renamed_sample = replace(original.samples[0], id="another_name")
        renamed = replace(original, samples=(renamed_sample,))
        first = model.predict_next(original, (Assignment("R1", "S"),))
        second = model.predict_next(renamed, (Assignment("R1", "another_name"),))
        self.assertEqual(first.robots[0].position, second.robots[0].position)
        self.assertEqual(first.robots[0].phase, second.robots[0].phase)
        self.assertEqual(first.samples[0].status, second.samples[0].status)

    def test_group_splits_are_disjoint_from_prior_experiments(self):
        groups = [set(seeds) for _, seeds in SPLITS.values()]
        previous = set(TRAIN_SEEDS) | set(TEST_SEEDS) | set(SHIFT_SEEDS) | {s for seeds in SEEDS.values() for s in seeds}
        for index, group in enumerate(groups):
            self.assertFalse(group & previous)
            self.assertTrue(all(not group & other for other in groups[index + 1:]))

    def test_different_seeds_can_still_repeat_a_task_schedule(self):
        source, sink = self.scenario.stations["A"], self.scenario.stations["B"]
        proxy = replace(self.scenario, stations={
            "ICU取样点": replace(source, id="ICU取样点"),
            "ICU2_假设取样": replace(source, id="ICU2_假设取样"),
            "实验室交付点": replace(sink, id="实验室交付点"),
        })
        audit = audit_task_schedules({"tartu": proxy}, {"train": ("tartu", (3001,)),
                                                        "test_tartu": ("tartu", (5001,))})
        self.assertEqual(audit["cross_split_duplicate_groups"], 1)
        self.assertEqual(audit["cross_split_task_duplicate_groups"], 1)
        duplicate = audit["cross_split_duplicates"][0]
        self.assertEqual({(x["split"], x["seed"], x["regime"]) for x in duplicate},
                         {("train", 3001, "normal"), ("test_tartu", 5001, "normal")})

    def test_multi_step_window_filter_reports_exclusions_and_no_future_actions(self):
        rows = [r for r in self.rows if r.next_boundary == "before_dispatch"]
        windows, coverage = eligible_windows(rows, 3)
        self.assertEqual(coverage["candidate_windows"], sum(v for k, v in coverage.items() if k != "candidate_windows"))
        self.assertTrue(windows)
        for window in windows:
            self.assertTrue(all(not row.assignments for row in window[1:]))
            original_ids = {s.id for s in window[0].observation.samples}
            self.assertTrue(all({s.id for s in row.next_observation.samples} == original_ids for row in window))

    def test_score_does_not_count_new_arrivals_as_model_errors(self):
        first = self.rows[0]
        extra = replace(first.next_observation.samples[0], id="new_arrival")
        actual = replace(first.next_observation, samples=first.next_observation.samples + (extra,))
        prediction = HybridDynamicsModel(Geometry(self.scenario.grid), manual_parameters()).predict_next(
            first.observation, first.assignments)
        scored = compare(first.observation, actual, prediction)
        self.assertEqual(scored["sample_total"], len(first.observation.samples))
        self.assertEqual(scored["new_arrivals"], 1)


if __name__ == "__main__":
    unittest.main()
