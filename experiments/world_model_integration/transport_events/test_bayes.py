"""Actual pgmpy estimator and payload integrity, no home-grown Bayes inference."""
from __future__ import annotations

import json
import unittest

from business_transport.world_model.transport_bayes import (
    TransportBayesModel,EDGES,public_transport_features,
)
from experiments.world_model_integration.transport_events.benchmark import PROTOCOL
from experiments.world_model_integration.transport_events.evaluate_bayes import (
    _split_rows,score_group,
)


class TransportBayesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.proto=json.loads(PROTOCOL.read_text())
        cls.groups=_split_rows(cls.proto,[
            x for x in cls.proto["groups"] if x["split"]=="train"])
        cls.rows=[r for g in cls.groups for r in g["rows"]]
        cls.model=TransportBayesModel.fit(cls.rows)

    def test_upstream_pgmpy_bayesian_estimator_fitted_actual_facts(self):
        import pgmpy
        self.assertTrue(pgmpy.__version__.startswith("1.0"))
        self.assertEqual(set(self.model.model.nodes()),
                         {"distance_bin","other_near","phase","delayed"})
        self.assertTrue(self.model.model.check_model())
        self.assertGreater(self.model.positive_legs,0)
        self.assertLess(self.model.positive_legs,self.model.total_legs)
        self.assertEqual(self.model.total_legs,240)

    def test_fitted_posterior_stable_and_serializable(self):
        sample=self.rows[0]
        a=self.model.predict_delay_probability(sample)
        restored=TransportBayesModel.from_payload(self.model.to_payload())
        b=restored.predict_delay_probability(sample)
        self.assertAlmostEqual(a,b,places=12)
        self.assertGreaterEqual(a,0)
        self.assertLessEqual(a,1)
        self.assertEqual(set(public_transport_features(sample)),
                         {"distance_bin","other_near","phase"})
        self.assertNotIn("duration_ticks",public_transport_features(sample))

    def test_train_group_metrics_finite(self):
        r=score_group(self.groups[0],self.model)
        self.assertEqual(r["legs"],len(self.groups[0]["rows"]))
        self.assertGreaterEqual(r["brier"]["pgmpy"],0)
        self.assertGreaterEqual(r["duration_mae"]["pgmpy"],0)
        self.assertLessEqual(r["brier"]["pgmpy"],1)


if __name__=="__main__":
    unittest.main()
