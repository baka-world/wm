from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lab_transport.domain import Grid, RobotSpec, Sample, Scenario, Station
from lab_transport.training_data import download_sources, export_dataset, validate_dataset


def grid_fixture():
    stations = {"ICU取样点": Station("ICU取样点", (2, 0), 2, 2.0, (), 0),
                "ICU2_假设取样": Station("ICU2_假设取样", (2, 3), 2, 2.0, (), 0),
                "实验室交付点": Station("实验室交付点", (5, 1), 1, 1.0, (), 0)}
    return Scenario("synthetic fixture", Grid(6, 4, frozenset()), stations,
                    (RobotSpec("R1", (0, 0), (0, 0)), RobotSpec("R2", (0, 3), (0, 3))),
                    (Sample("placeholder", "ICU取样点", "实验室交付点", 0, None),), 0, 350,
                    {"time_model": "test", "future_secret": "must-not-be-exported"})


def write_manifest(path, manifest):
    path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


class TrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "dataset"
        self.splits = {"train": ("tartu", (3000,)), "validation": ("tartu", (4000,))}
        self.manifest = export_dataset({"tartu": grid_fixture()}, {"fixture": "synthetic"}, self.root, self.splits)

    def _tamper_first_row(self, change):
        episode = self.manifest["episodes"][0]
        path = self.root / episode["file"]
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        change(rows[0])
        raw = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows).encode()
        path.write_bytes(raw)
        episode["sha256"] = hashlib.sha256(raw).hexdigest()
        write_manifest(self.root / "manifest.json", self.manifest)

    def test_export_has_grouped_factual_transitions_and_no_hidden_metadata(self):
        summary = validate_dataset(self.root)
        self.assertEqual(summary["episodes"], 8)
        self.assertGreater(summary["transitions"], 0)
        self.assertEqual(summary["maps"], 1)
        self.assertEqual(summary["schedules_not_verifiable_from_trajectory"], 0)
        topology = json.loads((self.root / self.manifest["maps"]["tartu"]["file"]).read_text())
        self.assertNotIn("future_secret", topology["metadata"])
        saw_arrival = False
        for entry in self.manifest["episodes"]:
            rows = [json.loads(line) for line in (self.root / entry["file"]).read_text().splitlines()]
            row = rows[0]
            self.assertNotIn("seed", row["observation"])
            self.assertNotIn("service_remaining", json.dumps(row["observation"]))
            self.assertEqual(row["next_observation"]["tick"], row["observation"]["tick"] + 1)
            saw_arrival |= any({s["id"] for s in r["next_observation"]["samples"]} -
                               {s["id"] for s in r["observation"]["samples"]} for r in rows)
        self.assertTrue(saw_arrival, "new arrivals must remain outcomes rather than current inputs")

    def test_rejects_hidden_service_even_if_checksum_is_rewritten(self):
        self._tamper_first_row(lambda row: row["observation"]["robots"][0].update({"service_remaining": 99}))
        with self.assertRaisesRegex(ValueError, "observation robots"):
            validate_dataset(self.root)

    def test_rejects_future_release_even_if_checksum_is_rewritten(self):
        def change(row):
            row["observation"]["samples"][0]["release_tick"] = row["observation"]["tick"] + 1
        self._tamper_first_row(change)
        with self.assertRaisesRegex(ValueError, "future task"):
            validate_dataset(self.root)

    def test_rejects_cross_split_task_hash_and_duplicate_file(self):
        first = self.manifest["episodes"][0]
        other = next(e for e in self.manifest["episodes"] if e["split"] != first["split"])
        original_hash = other["task_schedule_sha256"]
        other["task_schedule_sha256"] = first["task_schedule_sha256"]
        write_manifest(self.root / "manifest.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "task schedule"):
            validate_dataset(self.root)
        other["task_schedule_sha256"] = original_hash
        other["file"] = first["file"]
        write_manifest(self.root / "manifest.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "listed more than once"):
            validate_dataset(self.root)

    def test_checksums_and_existing_output_are_protected(self):
        first = self.manifest["episodes"][0]
        path = self.root / first["file"]
        original = path.read_bytes()
        with self.assertRaises(FileExistsError):
            export_dataset({"tartu": grid_fixture()}, {"fixture": "synthetic"}, self.root, self.splits)
        self.assertEqual(path.read_bytes(), original)
        path.write_bytes(original + b" ")
        with self.assertRaisesRegex(ValueError, "checksum"):
            validate_dataset(self.root)

    def test_rejects_a_forged_task_schedule_hash(self):
        for entry in self.manifest["episodes"]:
            if entry["split"] == "train" and entry["seed"] == 3000:
                entry["task_schedule_sha256"] = "0" * 64
        write_manifest(self.root / "manifest.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "task schedule checksum"):
            validate_dataset(self.root)

    def test_rejects_invalid_robot_position_even_with_updated_checksum(self):
        self._tamper_first_row(lambda row: row["observation"]["robots"][0].update({"position": [-1, -1]}))
        with self.assertRaisesRegex(ValueError, "robot occupancy"):
            validate_dataset(self.root)


class PinnedSourceTests(unittest.TestCase):
    def test_download_accepts_only_matching_pinned_content(self):
        data = b"known test source"
        sources = {"example": ("https://example.invalid/pinned", hashlib.sha256(data).hexdigest())}
        with patch("lab_transport.training_data.SOURCES", sources), \
             patch("lab_transport.training_data.urlopen", side_effect=lambda request, timeout: io.BytesIO(data)):
            self.assertEqual(download_sources(), {"example": data})
        with patch("lab_transport.training_data.SOURCES", sources), \
             patch("lab_transport.training_data.urlopen", side_effect=lambda request, timeout: io.BytesIO(b"changed")):
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                download_sources()


if __name__ == "__main__":
    unittest.main()
