"""Behavioral checks for the optional neural training/integration stack."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from lab_transport.multisite_experiment import make_case
    from lab_transport.torch_rollout import RolloutPolicy, run as run_rollout
    from lab_transport.torch_evaluate import evaluate
    from lab_transport.torch_smoke_adapter import collate, encode_transition, build_model
    from lab_transport.torch_training import (geometry_from_map, load_checkpoint, observation_from_dict,
                                              resolve_device, train)
    from lab_transport.training_data import export_dataset
    from lab_transport.transition_schema import Assignment
    from test_training_data import grid_fixture


@unittest.skipIf(torch is None, "optional PyTorch dependency is not installed")
class TorchTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.dataset = cls.root / "dataset"
        cls.base = grid_fixture()
        cls.manifest = export_dataset({"tartu": cls.base}, {"fixture": "synthetic"}, cls.dataset,
                                      {"train": ("tartu", (3000,)), "validation": ("tartu", (4000,)),
                                       "test_tartu": ("tartu", (9000,))})
        common = {"dataset": cls.dataset, "device_name": "cpu", "train_limit": 120, "validation_limit": 90,
                  "batch_size": 32}
        cls.full = cls.root / "full"
        cls.resumed = cls.root / "resumed"
        train(output=cls.full, epochs=2, **common)
        train(output=cls.resumed, epochs=1, **common)
        train(output=cls.resumed, epochs=2, resume=cls.resumed / "epoch-0001.pt", **common)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_resume_restores_model_optimizer_and_shuffled_batches(self):
        straight = load_checkpoint(self.full / "epoch-0002.pt")
        resumed = load_checkpoint(self.resumed / "epoch-0002.pt")
        for key in straight["model_state"]:
            self.assertTrue(torch.equal(straight["model_state"][key], resumed["model_state"][key]), key)
        self.assertTrue(torch.equal(straight["loader_rng"], resumed["loader_rng"]))
        self.assertEqual(straight["epoch_report"]["validation_loss"], resumed["epoch_report"]["validation_loss"])
        for key, value in straight["optimizer_state"]["state"].items():
            for part, tensor in value.items():
                self.assertTrue(torch.equal(tensor, resumed["optimizer_state"]["state"][key][part]))
        with self.assertRaisesRegex(ValueError, "does not match"):
            train(self.dataset, self.resumed, epochs=3, batch_size=64, train_limit=120, validation_limit=90,
                  resume=self.resumed / "epoch-0002.pt")

    def test_future_robot_moves_new_arrivals_and_station_changes_are_not_inputs(self):
        entry = self.manifest["episodes"][0]
        row = json.loads((self.dataset / entry["file"]).read_text(encoding="utf-8").splitlines()[0])
        current, future = observation_from_dict(row["observation"]), observation_from_dict(row["next_observation"])
        actions = tuple(Assignment(**a) for a in row["assignments"])
        topology = json.loads((self.dataset / self.manifest["maps"]["tartu"]["file"]).read_text())
        geometry = geometry_from_map(topology)
        original = encode_transition(geometry, current, actions, future)
        altered = replace(future,
            robots=tuple(replace(r, position=next(x.position for x in current.robots if x.id == r.id))
                         for r in future.robots),
            samples=future.samples + (replace(current.samples[0], id="new-future-sample", release_tick=future.tick),),
            stations=tuple(replace(s, available=not s.available) for s in future.stations))
        changed = encode_transition(geometry, current, actions, altered)
        self.assertTrue(original)
        self.assertEqual([(item["kind"], item["input"]) for item in original],
                         [(item["kind"], item["input"]) for item in changed])

    def test_variable_candidate_options_are_masked(self):
        from lab_transport.dynamics_revision import FEATURES_V2
        vector = [0.0] * len(FEATURES_V2)
        inputs, labels = collate([{"kind": "motion", "input": [vector, vector], "target": 1},
                                  {"kind": "motion", "input": [vector], "target": 0}])
        scores = build_model()(inputs)["motion"]
        self.assertEqual(tuple(scores.shape), (2, 2))
        self.assertTrue(torch.isfinite(scores[0]).all())
        self.assertTrue(torch.isneginf(scores[1, 1]))
        self.assertEqual(labels["motion"].tolist(), [1, 0])

    def test_saved_model_drives_bounded_two_robot_simulation(self):
        scenario = make_case(self.base, "tartu", 11000, "short_outage")
        result = run_rollout(self.full / "epoch-0002.pt", self.root / "closed-loop", scenario,
                             device_name="cpu", horizon=3, budget=2, seed=11000)
        self.assertEqual(result["summary"]["completed_count"], 6)
        self.assertEqual(result["discrete_safety_violations"], 0)
        self.assertGreater(result["candidate_evaluations"], 0)
        self.assertEqual(result["fallbacks"], 0)
        self.assertEqual(result["model_steps"], 3 * result["candidate_evaluations"])
        records = json.loads((self.root / "closed-loop" / "decisions.json").read_text())
        self.assertTrue(all(len(row["candidates_evaluated"]) <= 2 for row in records))

    def test_invalid_model_prediction_uses_recorded_available_fallback(self):
        entry = self.manifest["episodes"][0]
        row = json.loads((self.dataset / entry["file"]).read_text(encoding="utf-8").splitlines()[0])
        current = observation_from_dict(row["observation"])
        topology = json.loads((self.dataset / self.manifest["maps"]["tartu"]["file"]).read_text())
        geometry = geometry_from_map(topology)
        candidates = tuple(Assignment(r.id, current.samples[0].id) for r in current.robots)

        class Broken:
            def predict_next(self, observation, actions):
                raise ValueError("invalid prediction")

        policy = RolloutPolicy(Broken(), geometry)
        self.assertEqual(policy(current, candidates), [candidates[0]])
        self.assertEqual(policy.fallbacks, 1)
        self.assertEqual(policy.log[0]["fallback"], "ValueError")

    def test_requested_cuda_fails_instead_of_silent_cpu_fallback(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "CUDA requested"):
                resolve_device("cuda")

    def test_fixed_checkpoint_evaluates_only_requested_held_out_prefix(self):
        result = evaluate(self.dataset, self.full / "epoch-0002.pt", self.root / "evaluation",
                          splits=("test_tartu",), limit_per_split=25)
        self.assertEqual(result["transitions_per_split"], {"test_tartu": 25})
        summary = result["summaries"]["test_tartu"]
        self.assertEqual(summary["one_step"]["learned"]["counts"]["joint_total"], 25)
        self.assertLessEqual(summary["multi_step"]["3"]["coverage"]["eligible"], 23)
        self.assertNotIn("completed_tasks", summary)


if __name__ == "__main__":
    unittest.main()
