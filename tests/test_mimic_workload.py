import json
import unittest
from unittest.mock import patch

from lab_transport.domain import DirectedGraph
from lab_transport.hospital_case import make_scenario
from lab_transport.mimic_workload import CHECKSUMS, DATASET_DOI, select_timeline


class MimicWorkloadTests(unittest.TestCase):
    def test_deduplicates_measurements_and_never_combines_patients(self):
        items = [{"itemid": "10", "fluid": "Blood"}, {"itemid": "11", "fluid": "Urine"}]
        stays = [
            {"subject_id": "10", "hadm_id": "100", "stay_id": "1", "first_careunit": "ICU",
             "intime": "2100-01-01 00:00:00", "outtime": "2100-01-03 00:00:00"},
            {"subject_id": "11", "hadm_id": "200", "stay_id": "2", "first_careunit": "Medical ICU",
             "intime": "2100-01-01 00:00:00", "outtime": "2100-01-03 00:00:00"},
        ]
        rows = [
            {"subject_id": "10", "hadm_id": "100", "specimen_id": "a", "itemid": "10", "charttime": "2100-01-01 01:00:00"},
            {"subject_id": "10", "hadm_id": "100", "specimen_id": "a", "itemid": "10", "charttime": "2100-01-01 01:00:00"},
            {"subject_id": "10", "hadm_id": "100", "specimen_id": "b", "itemid": "11", "charttime": "2100-01-01 02:00:00"},
            {"subject_id": "11", "hadm_id": "200", "specimen_id": "c", "itemid": "10", "charttime": "2100-01-01 01:00:00"},
            {"subject_id": "11", "hadm_id": "200", "specimen_id": "d", "itemid": "10", "charttime": "2100-01-01 01:05:00"},
            {"subject_id": "11", "hadm_id": "200", "specimen_id": "e", "itemid": "10", "charttime": "2100-01-01 01:15:00"},
            {"subject_id": "11", "hadm_id": "200", "specimen_id": "f", "itemid": "10", "charttime": "2100-01-01 02:00:00"},
        ]
        result = select_timeline(items, rows, stays, 4)
        self.assertEqual([x["relative_charttime_seconds"] for x in result["samples"]], [0, 300, 900, 3600])
        self.assertEqual(len(result["samples"]), 4)
        self.assertTrue(all(x["specimen_type"] == "blood_unspecified" for x in result["samples"]))
        serialized = json.dumps(result)
        for private_field in ("subject_id", "hadm_id", "2100-01-01"):
            self.assertNotIn(private_field, serialized)
        self.assertTrue(all("specimen_id" not in sample for sample in result["samples"]))

    def test_rejects_insufficient_single_patient_data(self):
        items = [{"itemid": "10", "fluid": "Blood"}]
        stays = [{"subject_id": "1", "hadm_id": "1", "stay_id": "1", "first_careunit": "ICU",
                  "intime": "2100-01-01 00:00:00", "outtime": "2100-01-03 00:00:00"}]
        rows = [{"subject_id": "1", "hadm_id": "1", "specimen_id": "a", "itemid": "10",
                 "charttime": "2100-01-01 01:00:00"}]
        with self.assertRaisesRegex(ValueError, "enough"):
            select_timeline(items, rows, stays, 2)

    def test_replay_requires_explicit_time_assumption_and_preserves_order(self):
        nodes = [(190, 0), (193, 0), (204, 0), (208, 0)]
        graph = DirectedGraph({p: (p[0], 0) for p in nodes},
                              tuple((a, b) for a in nodes for b in nodes if a != b))
        workload = {"schema_version": 1, "source": {"doi": DATASET_DOI, "checksums": CHECKSUMS},
                    "samples": [{"id": f"TRACE-{i + 1:02}", "specimen_type": "blood_unspecified",
                                 "relative_charttime_seconds": n} for i, n in enumerate([0, 61, 120, 121])]}
        with patch("lab_transport.hospital_case.parse_pinned_building", return_value=graph):
            with self.assertRaisesRegex(ValueError, "explicit positive"):
                make_scenario(b"test", workload)
            scenario = make_scenario(b"test", workload, 60)
            self.assertEqual([s.release_tick for s in scenario.samples], [0, 2, 2, 3])
            self.assertTrue(all(s.deadline_tick is None for s in scenario.samples))
            self.assertTrue(all(s.specimen_type == "blood_unspecified" for s in scenario.samples))
            self.assertIn("不是实际", scenario.metadata["timing_assumption"])


if __name__ == "__main__":
    unittest.main()
