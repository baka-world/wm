"""CPU-only regression tests for PR17 retrospective-only fault labels."""
from __future__ import annotations

import unittest

from experiments.oracle_value.observable_audit import (
    fault_timeline, group_for, inspect_state, station_status,
)


def obs(tick, down=(), blocked=(), service_elapsed=None):
    down, blocked = set(down), set(blocked)
    return {
        "tick": tick,
        "stations": [
            {"id": "A", "device_available": "A" not in down,
             "available": "A" not in down and "A" not in blocked,
             "capacity_blocked": "A" in blocked, "resource_occupied": 1 if "A" in blocked else 0,
             "resource_capacity": 1, "resource_id": "pool"},
            {"id": "B", "device_available": "B" not in down, "available": "B" not in down,
             "capacity_blocked": False, "resource_occupied": 0,
             "resource_capacity": 1, "resource_id": "poolB"},
        ],
        "robots": [{"id": "R1", "service_elapsed": service_elapsed,
                    "phase": "to_dropoff", "task_id": "S1"}],
        "samples": [{"id": "S1", "status": "waiting"}],
    }


class TimelineTests(unittest.TestCase):
    def test_current_unavailable_is_not_new_fault(self):
        first = obs(10, down=("A",))
        timeline = [obs(t, down=("A",)) for t in range(11, 31)]
        result = fault_timeline(first, timeline)
        self.assertEqual(result["initial_down"], ["A"])
        self.assertEqual(result["onsets"], [])
        self.assertEqual(group_for(5, result), "1_currently_unavailable")
        self.assertEqual(group_for(20, result), "1_currently_unavailable")

    def test_onset_is_actual_tick_not_regime(self):
        first = obs(7)
        timeline = [obs(t, down=("A",) if 14 <= t <= 16 else ())
                    for t in range(8, 28)]
        result = fault_timeline(first, timeline)
        self.assertEqual(result["onsets"], [{"station_id": "A", "absolute_tick": 14, "offset": 7}])
        self.assertEqual(result["recoveries"], [{"station_id": "A", "absolute_tick": 17, "offset": 10}])
        self.assertEqual(group_for(5, result),
                         "3_currently_normal_no_fault_within_horizon")
        self.assertEqual(group_for(20, result),
                         "2_future_fault_from_currently_normal")

    def test_blocked_capacity_is_not_device_fault(self):
        first = obs(0, blocked=("A",))
        result = fault_timeline(first, [obs(t, blocked=("A",)) for t in range(1, 21)])
        self.assertEqual(result["initial_down"], [])
        self.assertFalse(result["onsets"])
        self.assertEqual(group_for(20, result),
                         "3_currently_normal_no_fault_within_horizon")

    def test_two_faults_count_separately(self):
        first = obs(0)
        states = [obs(t, down=("A",) if 2 <= t <= 3 or 6 <= t <= 8 else ())
                  for t in range(1, 21)]
        self.assertEqual([x["offset"] for x in fault_timeline(first, states)["onsets"]],
                         [2, 6])

    def test_no_negative_labels_for_short_trace(self):
        with self.assertRaisesRegex(ValueError, "truncated"):
            fault_timeline(obs(0), [obs(t) for t in range(1, 5)])

    def test_no_capacity_fallback_for_missing_device_status(self):
        example = obs(0)
        del example["stations"][0]["device_available"]
        with self.assertRaisesRegex(ValueError, "explicit"):
            station_status(example)

    def test_nonconsecutive_ticks_rejected(self):
        with self.assertRaisesRegex(ValueError, "nonconsecutive"):
            fault_timeline(obs(0), [obs(t) for t in range(2, 22)])

    def test_censored_previous_regret_not_silently_converted(self):
        case = {"layout": "a", "seed": 1, "regime": "workflow_outage"}
        initial = obs(0)
        decision = {"observable_state": initial, "tick": 0, "plans": [[], []],
                    "branches": [{"trace_first20": [obs(i) for i in range(1, 21)]}] * 2}
        old = {"valid": True, "tick": 0, "layout": "a", "seed": 1,
               "regime": "workflow_outage", "flow_spread": None,
               "outcome_tied": False, "all_candidates_complete": False,
               "methods": {name: {"choice": 0, "regret_steps": None,
                                   "completion_deficit": 1}
                           for name in ("Rule-5", "Perfect-State-5", "Perfect-State-20",
                                        "Full-outcome Oracle")}}
        with self.assertRaisesRegex(ValueError, "uncensored"):
            inspect_state(case, decision, old, set())

    def test_per_candidate_fault_schedule_divergence_rejected(self):
        case = {"layout": "a", "seed": 1, "regime": "workflow_outage"}
        initial = obs(0)
        normal = [obs(i) for i in range(1, 21)]
        fault = [obs(i, down=("A",) if i > 3 else ()) for i in range(1, 21)]
        decision = {"observable_state": initial, "tick": 0, "plans": [[], []],
                    "branches": [{"trace_first20": normal}, {"trace_first20": fault}]}
        old = {"valid": True, "tick": 0, "layout": "a", "seed": 1,
               "regime": "workflow_outage", "flow_spread": 1.0,
               "outcome_tied": False, "all_candidates_complete": True,
               "methods": {name: {"choice": 0, "regret_steps": 0,
                                   "completion_deficit": 0}
                           for name in ("Rule-5", "Perfect-State-5", "Perfect-State-20",
                                        "Full-outcome Oracle")}}
        with self.assertRaisesRegex(ValueError, "candidate-dependent outage timeline"):
            inspect_state(case, decision, old, set())


if __name__ == "__main__":
    unittest.main()
