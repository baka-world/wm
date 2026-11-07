from dataclasses import replace
from contextlib import redirect_stdout
import io
import unittest

from lab_transport.domain import Grid, RobotSpec, Sample, Scenario, Station
from lab_transport.dynamics_experiment import Episode, SPLITS
from lab_transport.dynamics_model import Geometry, MOTION_FEATURES, SERVICE_FEATURES
from lab_transport.dynamics_revision import (CHECKPOINTS, FEATURES_V2, SPLITS_V2, RevisedDynamicsModel,
                                             motion_features_v2, train_checkpoints, validate_v2, validation_score)
from lab_transport.simulation import Simulator


class RevisedDynamicsTests(unittest.TestCase):
    def setUp(self):
        self.scenario = Scenario("revision test", Grid(5, 2, frozenset()), {
            "A": Station("A", (0, 0), 1, 1.0, (), 0), "B": Station("B", (4, 0), 1, 1.0, (), 0),
        }, (RobotSpec("R1", (0, 0), (0, 1)),), (Sample("S", "A", "B", 0, None),), 0, 30)
        self.geometry = Geometry(self.scenario.grid)
        self.rows = []
        Simulator(self.scenario, "world", 1).run(on_transition=self.rows.append)
        a, b = [0.0] * len(FEATURES_V2), [0.0] * len(FEATURES_V2)
        a[0], b[1] = 1.0, 1.0
        x, y = [0.0] * len(SERVICE_FEATURES), [0.0] * len(SERVICE_FEATURES)
        x[0], y[0], y[1] = 1.0, 1.0, 2.0
        self.moves, self.services = [([a, b], 1)] * 6, [(x, 0), (y, 1)] * 6
        with redirect_stdout(io.StringIO()):
            self.models = train_checkpoints(self.moves, self.services)

    def test_checkpoint_training_is_deterministic_and_versioned(self):
        self.assertEqual([m["epoch"] for m in self.models], list(CHECKPOINTS))
        with redirect_stdout(io.StringIO()):
            self.assertEqual(self.models, train_checkpoints(self.moves, self.services))
        for model in self.models:
            validate_v2(model)
            self.assertEqual(len(model["motion_weights"]), len(MOTION_FEATURES) + 1)
        with self.assertRaisesRegex(ValueError, "schema"):
            validate_v2(dict(self.models[0], motion_features=["wrong"]))

    def test_feature_marks_only_staying_when_already_at_goal(self):
        robot = self.rows[0].observation.robots[0]
        goal = robot.position
        stay = motion_features_v2(self.geometry, robot, "idle", goal, goal, ())
        move = motion_features_v2(self.geometry, robot, "idle", goal, (1, 0), ())
        self.assertEqual((stay[-1], move[-1]), (1.0, 0.0))

    def test_validation_score_ignores_all_test_episodes(self):
        validation = Episode("validation", "tartu", 4000, "normal", "world", "completed", 1, self.rows[:-1])
        test = replace(validation, split="test_tartu", seed=9000)
        a = validation_score([validation], {"tartu": self.geometry}, self.models[0])
        b = validation_score([validation, test], {"tartu": self.geometry}, self.models[0])
        self.assertEqual(a, b)
        prediction = RevisedDynamicsModel(self.geometry, self.models[0]).predict_next(
            validation.rows[0].observation, validation.rows[0].assignments)
        self.assertEqual(prediction.tick, 1)

    def test_new_test_seeds_were_not_in_pilot(self):
        pilot = {seed for _, seeds in SPLITS.values() for seed in seeds}
        self.assertEqual(SPLITS_V2["train"], SPLITS["train"])
        self.assertEqual(SPLITS_V2["validation"], SPLITS["validation"])
        for split in ("test_tartu", "test_clinic"):
            self.assertFalse(set(SPLITS_V2[split][1]) & pilot)


if __name__ == "__main__":
    unittest.main()
