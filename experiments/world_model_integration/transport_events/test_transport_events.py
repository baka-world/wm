"""Transport labels are genuine arrival events, not service admission times."""
from __future__ import annotations
import json
import unittest

from business_transport.transport_observability import ObservableTransportSimulator
from experiments.business_constraints.tests.test_business import fixture
from experiments.world_model_integration.transport_events.benchmark import (
    PROTOCOL, observed_episode, transport_rows, scenario_metrics,
)
from lab_transport.dynamics_model import Geometry


class TransportFactsTests(unittest.TestCase):
    def test_arrival_separate_from_service_and_original_simulator_unchanged(self):
        _,_,business=fixture(capacity=1)
        original=__import__("business_transport.simulation",fromlist=["BusinessSimulator"]).BusinessSimulator(
            business,"world",123).run()
        observed=ObservableTransportSimulator(business,"world",123).run()
        old_events=[e for e in observed["events"] if e["type"]!="transport_arrived"]
        self.assertEqual(old_events,original["events"])
        self.assertTrue(any(e["type"]=="transport_arrived" for e in observed["events"]))
        self.assertEqual(observed["summary"],original["summary"])

    def test_public_leg_timestamps_and_known_graph_baseline(self):
        p=json.loads(PROTOCOL.read_text())
        result,states,geo=observed_episode(p["groups"][0],max_ticks=p["max_ticks"])
        legs=transport_rows(result["events"],states,geo,episode_end=result["summary"]["ticks"])
        self.assertTrue(legs)
        self.assertEqual(sum(r["right_censored"] for r in legs),0)
        self.assertTrue(any(r["phase"]=="to_pickup" for r in legs))
        self.assertTrue(any(r["phase"]=="to_dropoff" for r in legs))
        for row in legs:
            self.assertLessEqual(row["start_tick"],row["arrived_tick"])
            self.assertLessEqual(row["arrived_tick"],row["service_admitted_tick"])
            self.assertEqual(row["duration_ticks"],row["arrived_tick"]-row["start_tick"])
            self.assertGreaterEqual(row["distance"],0)
            self.assertGreaterEqual(row["delay_ticks"],0)
            self.assertEqual(row["delay_label"],int(
                row["duration_ticks"]>row["path_only_baseline_ticks"]))
        stats=scenario_metrics(legs)
        self.assertGreater(stats["legs"],0)
        self.assertGreaterEqual(stats["baseline_mae_ticks"],0)

    def test_disjoint_seed_splits_and_future_holdouts(self):
        p=json.loads(PROTOCOL.read_text())
        self.assertEqual(p["protocol"],"TRANSPORT_PUBLIC_EVENTS_V1")
        sets=[]
        for split in ("train","validation","holdout"):
            rows=[r["seed"] for r in p["groups"] if r["split"]==split]
            self.assertEqual(len(rows),{"train":24,"validation":8,"holdout":12}[split])
            sets.append(set(rows))
        sets.extend([{x["seed"] for x in p[k]} for k in
                     ("future_horizon_groups","future_decision_groups")])
        self.assertEqual(sum(len(x) for x in sets),len(set.union(*sets)))


if __name__=="__main__":
    unittest.main()
