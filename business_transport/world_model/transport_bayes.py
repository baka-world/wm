"""Published-model transport-delay reference using upstream pgmpy inference.

Primary methodological precedent: Tang & Wang (2025), Scientific Reports,
DOI 10.1038/s41598-024-82870-1. Their approach uses Bayesian probability
for AGV obstacles/transport uncertainty. Here the observed label is *elapsed
transport exceeding a public shortest-path time*, NOT an observed obstacle.
This is an analogous engineering adaptation, NOT reproduction of their graph
structure or their reported performance.

All Bayesian CPD fit/inference routines are upstream pgmpy 1.0.0 (MIT).
"""
from __future__ import annotations

from dataclasses import dataclass
import math

EDGES=(
    ("distance_bin","delayed"),
    ("other_near","delayed"),
    ("phase","delayed"),
)
STATES={
    "distance_bin":[0,1,2],
    "other_near":[0,1],
    "phase":[0,1],
    "delayed":[0,1],
}


def public_transport_features(row):
    """Only initial public geometry/occupancy, never eventual arrival/result."""
    distance=row["distance"]
    nearby=row["other_near_target"]
    phase=row["phase"]
    if type(distance) is not int or distance<0 or type(nearby) is not int or nearby<0:
        raise ValueError("transport feature missing or invalid")
    if phase not in ("to_pickup","to_dropoff"):
        raise ValueError("invalid transport phase")
    return {
        "distance_bin":min(2,distance//4),
        "other_near":int(nearby>0),
        "phase":int(phase=="to_dropoff"),
    }


class TransportBayesModel:
    """Low-dimensional pgmpy BayesianEstimator + VariableElimination adapter."""

    def __init__(self, model, *,positive_delay_ticks,total_legs,positive_legs):
        from pgmpy.inference import VariableElimination
        if total_legs<1 or positive_legs<1 or positive_legs>=total_legs:
            raise ValueError("transport data must contain positive and negative observed legs")
        self.model=model
        self.inference=VariableElimination(model)
        self.positive_delay_ticks=float(positive_delay_ticks)
        self.total_legs=int(total_legs)
        self.positive_legs=int(positive_legs)
        self.reference="Tang and Wang 2025; upstream pgmpy BayesianEstimator and VariableElimination"

    @classmethod
    def fit(cls, rows):
        import pandas as pd
        from pgmpy.models import DiscreteBayesianNetwork
        from pgmpy.estimators import BayesianEstimator
        clean=[r for r in rows if not r["right_censored"]]
        if not clean:
            raise ValueError("no completed factual AGV transport legs")
        positive=[r for r in clean if r["delay_label"]==1]
        if not positive or len(positive)==len(clean):
            raise ValueError("cannot fit binary delayed event without two observed classes")
        data=pd.DataFrame([{**public_transport_features(r),
                            "delayed":r["delay_label"]} for r in clean])
        network=DiscreteBayesianNetwork(EDGES)
        network.fit(data,estimator=BayesianEstimator,
                    prior_type="BDeu",equivalent_sample_size=5,
                    state_names=STATES,n_jobs=1)
        network.check_model()
        positive_delay=sum(r["delay_ticks"] for r in positive)/len(positive)
        return cls(network,positive_delay_ticks=positive_delay,
                   total_legs=len(clean),positive_legs=len(positive))

    def predict_delay_probability(self, public_row):
        """Posterior P(delay=1 | current public covariates) via pgmpy."""
        features=public_transport_features(public_row)
        dist=self.inference.query(variables=["delayed"],
                                  evidence=features,show_progress=False)
        p=float(dist.get_value(delayed=1))
        if not math.isfinite(p) or not 0<=p<=1:
            raise ValueError("pgmpy returned invalid delayed posterior")
        return p

    def expected_transport_ticks(self,public_row):
        return (float(public_row["path_only_baseline_ticks"]) +
                self.predict_delay_probability(public_row)*self.positive_delay_ticks)

    def to_payload(self):
        """Plain JSON, source-identified CPDs, no pickle or unknown execution."""
        cpds=[]
        for cpd in sorted(self.model.get_cpds(),key=lambda c:c.variable):
            cpds.append({
                "variable":cpd.variable,
                "variable_card":int(cpd.variable_card),
                "evidence":list(cpd.variables[1:]),
                "evidence_card":[int(v) for v in cpd.cardinality[1:]],
                "state_names":{k:list(v) for k,v in cpd.state_names.items()},
                "values":cpd.get_values().tolist(),
            })
        return {
            "schema":"rhwm-transport-pgmpy-reference-v1",
            "method":"pgmpy 1.0.0 BayesianEstimator BDeu + VariableElimination",
            "source":"https://github.com/pgmpy/pgmpy",
            "paper":"Tang and Wang 2025 DOI 10.1038/s41598-024-82870-1 (method inspiration, not replication)",
            "edges":[list(e) for e in EDGES],
            "cpds":cpds,
            "positive_delay_ticks":self.positive_delay_ticks,
            "train_legs":self.total_legs,
            "train_positive":self.positive_legs,
        }

    @classmethod
    def from_payload(cls,payload):
        from pgmpy.models import DiscreteBayesianNetwork
        from pgmpy.factors.discrete import TabularCPD
        if (payload.get("schema")!="rhwm-transport-pgmpy-reference-v1"
                or tuple(map(tuple,payload.get("edges",())))!=EDGES):
            raise ValueError("wrong transport model/graph schema")
        net=DiscreteBayesianNetwork(EDGES)
        for c in payload["cpds"]:
            net.add_cpds(TabularCPD(
                c["variable"],c["variable_card"],c["values"],
                evidence=c["evidence"] or None,
                evidence_card=c["evidence_card"] or None,
                state_names=c["state_names"]))
        net.check_model()
        return cls(net,positive_delay_ticks=payload["positive_delay_ticks"],
                   total_legs=payload["train_legs"],
                   positive_legs=payload["train_positive"])
