"""Read-only time-boundary audit of the actual factual per-session training rows.

The target labels at t+1 must match progress-at-t, not leaked completion
duration. This runs against freshly recreated factual episodes, never an
inferred hidden event calendar.
"""
import unittest
from collections import defaultdict

from experiments.world_model_integration.build_dataset import build_dataset


def _progress(row):
    query=row["query"]
    return next(robot["service_active_elapsed"]["value"]
                for robot in row["features"]["observed_facts"]["robots"]
                if robot["id"]==query["robot_id"])


class FactServiceTimeAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset=build_dataset()

    def test_valid_event_labels_use_only_started_active_service_at_t(self):
        for split in ("train","validation","test"):
            for row in self.dataset[split]:
                self.assertEqual(row["features"]["tick"],row["tick"])
                self.assertIsInstance(_progress(row),int)
                self.assertGreaterEqual(_progress(row),0)
                self.assertEqual(row["event"],"finished" if row["label"] else "continues")
                query=row["query"]
                sample=next(s for s in row["features"]["observed_facts"]["samples"]
                            if s["id"]==query["sample_id"])
                robot=next(r for r in row["features"]["observed_facts"]["robots"]
                           if r["id"]==query["robot_id"])
                self.assertEqual(sample["pickup" if query["phase"]=="to_pickup"
                                             else "dropoff"],query["station_id"])
                self.assertEqual(robot["current_task"]["value"],query["sample_id"])
                self.assertEqual(robot["phase"],query["phase"])

    def test_contiguous_surviving_service_progress_increases_by_one(self):
        total=0
        for split in ("train","validation","test"):
            sessions=defaultdict(list)
            for row in self.dataset[split]:
                q=row["query"]
                key=(row["trajectory_group"],q["robot_id"],q["sample_id"],
                     q["station_id"],q["phase"])
                sessions[key].append(row)
            for records in sessions.values():
                records.sort(key=lambda r:r["tick"])
                for before,after in zip(records,records[1:]):
                    if after["tick"]!=before["tick"]+1:
                        continue  # paused/unobserved gap must not be made a negative
                    self.assertEqual(before["label"],0,
                         "completed service must not yield a surviving next query")
                    self.assertEqual(_progress(after),_progress(before)+1)
                    total+=1
        self.assertGreater(total,100,"fixture must actually exercise service continuation")


if __name__=="__main__":
    unittest.main()
