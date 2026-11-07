import json
from pathlib import Path
import tempfile
import unittest

from lab_transport.specimen_rules import evaluate, load_profiles

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "scenarios/specimen-rule-profiles.json"


class SpecimenRuleTests(unittest.TestCase):
    def setUp(self):
        self.profiles = load_profiles(PROFILES)
        self.sample = {"id": "synthetic", "rule_profile": "michigan_pediatric_blood_gas",
                       "specimen_type": "pediatric_blood_gas",
                       "collected_at": "2026-01-01T12:00:00+00:00",
                       "packaging": "sealed_biohazard_bag", "temperature_condition": "room_temperature"}

    def test_receipt_and_analysis_are_distinct_and_missing_is_unknown(self):
        sample = dict(self.sample, received_at="2026-01-01T12:30:00+00:00")
        result = evaluate(sample, self.profiles)
        self.assertEqual(result["checks"]["max_collection_to_receipt_seconds"]["status"], "within_recorded_limit")
        self.assertEqual(result["checks"]["max_collection_to_analysis_seconds"]["status"], "unknown")
        sample["analyzed_at"] = "2026-01-01T12:30:01+00:00"
        self.assertEqual(evaluate(sample, self.profiles)["checks"]["max_collection_to_analysis_seconds"]["status"],
                         "beyond_recorded_limit")
        sample.pop("received_at")
        sample["delivered_at"] = "2026-01-01T12:20:00+00:00"
        self.assertEqual(evaluate(sample, self.profiles)["checks"]["max_collection_to_receipt_seconds"]["status"], "unknown")

    def test_specific_type_required_and_other_profiles_do_not_apply(self):
        for specimen_type, expected in (("blood_unspecified", "not_applicable"), (None, "unknown")):
            with self.subTest(specimen_type=specimen_type):
                result = evaluate(dict(self.sample, specimen_type=specimen_type), self.profiles)
                self.assertEqual(result["applicability"], expected)
                self.assertEqual(result["checks"], {})

    def test_attested_packaging_and_temperature_are_not_assumed(self):
        sample = dict(self.sample)
        sample.pop("packaging")
        sample["temperature_condition"] = "refrigerated"
        checks = evaluate(sample, self.profiles)["checks"]
        self.assertEqual(checks["packaging"]["status"], "unknown")
        self.assertEqual(checks["temperature_condition"]["status"], "differs_from_record")

    def test_invalid_times_and_missing_sources_are_rejected(self):
        for time in ("2026-01-01T11:59:59Z", "2026-01-01T12:01:00"):
            with self.subTest(time=time), self.assertRaises(ValueError):
                evaluate(dict(self.sample, received_at=time), self.profiles)
        data = json.loads(PROFILES.read_text(encoding="utf-8"))
        data["profiles"][0]["source_url"] = ""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source_url"):
                load_profiles(path)


if __name__ == "__main__":
    unittest.main()
