"""Exact upstream pycox LogisticHazard and torchtuples MLP training adapter.

Reference: Kvamme & Borgan (2019), arXiv:1910.06724.
Pycox 0.3.0: https://github.com/havakv/pycox/blob/master/pycox/models/logistic_hazard.py
BSD-2-Clause. Pycox owns log-likelihood, discretization, optimization and
survival probabilities. We only map public, service-start features and elapsed
ticks to the standard survival data convention.

This is a *baseline covariate* hazard model. No time-varying neural features,
graph network, new loss or privileged future features are introduced.
"""
from __future__ import annotations

import math
import random

# Schema fixed by the frozen data protocol. Stable station identifiers come
# from the public business scenario, not from the simulator's private state.
STATION_ORDER = ("A0", "A1", "B0", "B1", "C")
FEATURE_NAMES = (
    *(f"station_{sid}" for sid in STATION_ORDER),
    "pickup_phase", "dropoff_phase",
    "expected_service_ticks_div10", "resource_capacity_div3",
)


def public_service_features(observation, query):
    """Service-start features that can also be reconstructed in later queries."""
    import numpy as np

    station_id = query["station_id"]
    phase = query["phase"]
    if station_id not in STATION_ORDER or phase not in ("to_pickup", "to_dropoff"):
        raise ValueError("unrecognized public station or service phase")
    station = next(s for s in observation.stations if s.id == station_id)
    expected = float(station.expected_service_ticks)
    capacity = int(station.resource_capacity)
    if not math.isfinite(expected) or expected <= 0 or capacity < 1:
        raise ValueError("invalid public station service prior or capacity")
    return np.asarray(
        [float(station_id == sid) for sid in STATION_ORDER] +
        [float(phase == "to_pickup"), float(phase == "to_dropoff"),
         expected / 10., capacity / 3.], dtype=np.float32)


class PycoxLogisticHazardServiceModel:
    """Library-trained conditional service-time hazard with explicit age lookup."""

    def __init__(self, model, cuts, *, training_summary):
        import numpy as np

        self.model = model
        self.cuts = np.asarray(cuts)
        if self.cuts.ndim != 1 or self.cuts.size < 3 or self.cuts[0] != 0 or not np.all(
                np.diff(self.cuts) == 1):
            raise ValueError("event-duration cuts must be training-derived integer grid")
        self.model.net.eval()
        self.training_summary = dict(training_summary)
        self.num_queries = 0
        self.rule_fallbacks = 0

    @classmethod
    def train(cls, train_rows, validation_rows, *, settings):
        import numpy as np
        import torch
        import torchtuples as tt
        from pycox.models import LogisticHazard

        if not train_rows or not validation_rows:
            raise ValueError("pycox training/validation must each contain factual service sessions")
        if not all(row["duration"] >= 1 and type(row["event_observed"]) is bool and
                   len(row["features"]) == len(FEATURE_NAMES)
                   for row in (*train_rows, *validation_rows)):
            raise ValueError("invalid factual service-session contract")
        seed = int(settings["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(1)

        # Survival label transform and negative log-likelihood are both
        # upstream pycox implementations. Time cuts never inspect val/test.
        max_train_duration = max(row["duration"] for row in train_rows)
        cuts = np.arange(0, max_train_duration + 3, dtype=np.float64)
        labtrans = LogisticHazard.label_transform(cuts)
        def pack(rows):
            x = np.asarray([row["features"] for row in rows],dtype=np.float32)
            t = np.asarray([row["duration"] for row in rows],dtype=np.float64)
            e = np.asarray([float(row["event_observed"]) for row in rows],dtype=np.float32)
            return x,t,e
        x_tr,t_tr,e_tr = pack(train_rows)
        x_va,t_va,e_va = pack(validation_rows)
        y_tr = labtrans.fit_transform(t_tr,e_tr)
        y_va = labtrans.transform(t_va,e_va)
        net = tt.practical.MLPVanilla(
            len(FEATURE_NAMES),
            list(settings["hidden_layers"]),
            labtrans.out_features,
            batch_norm=bool(settings["batch_norm"]),
            dropout=float(settings["dropout"]))
        model = LogisticHazard(
            net,tt.optim.Adam(lr=float(settings["learning_rate"])),
            duration_index=labtrans.cuts,device="cpu")
        # Official torchtuples callbacks select the best validation loss
        # and restore its weights, not last-epoch weights.
        callbacks=[tt.callbacks.BestWeights(metric="loss",dataset="val",
                                            load_best=True,rm_file=True),
                   tt.callbacks.EarlyStopping(metric="loss",dataset="val",patience=6)]
        log=model.fit(
            x_tr,y_tr,batch_size=int(settings["batch_size"]),
            epochs=int(settings["epochs"]),callbacks=callbacks,
            verbose=False,val_data=(x_va,y_va),shuffle=True)
        history=log.to_pandas()
        if "val_loss" not in history.columns:
            raise AssertionError("upstream validation loss absent from training log")
        best=float(history["val_loss"].min())
        if not math.isfinite(best):
            raise AssertionError("nonfinite validation likelihood")
        info={
            "library":"pycox.models.LogisticHazard",
            "loss":"pycox.models.loss.NLLLogistiHazardLoss",
            "net":"torchtuples.practical.MLPVanilla",
            "selection":"torchtuples.callbacks.BestWeights(val_loss)",
            "train_sessions":len(train_rows),
            "validation_sessions":len(validation_rows),
            "observed_train_events":sum(r["event_observed"] for r in train_rows),
            "right_censored_train":sum(not r["event_observed"] for r in train_rows),
            "max_train_duration":int(max_train_duration),
            "max_grid_duration":int(cuts[-1]),
            "epochs_executed":len(history),
            "min_val_nll":best,
            "feature_names":list(FEATURE_NAMES),
            "time_cuts_from_train_only":True,
        }
        return cls(model,labtrans.cuts,training_summary=info)

    def query_probability(self, observation, query, active_elapsed, fallback_mean):
        """Next-tick service event hazard from the fitted pycox distribution.

        Since cuts enumerate integer elapsed ticks, the next completion is
        represented by the bin index (active_elapsed+1). Input covariates
        are fixed service-start fields; present state supplies only elapsed.
        """
        import numpy as np

        if type(active_elapsed) is not int or active_elapsed < 0:
            raise ValueError("invalid observed effective service age")
        if not math.isfinite(float(fallback_mean)) or fallback_mean <= 0:
            raise ValueError("invalid public fallback service expectation")
        if active_elapsed + 1 >= len(self.cuts):
            self.rule_fallbacks += 1
            return float(active_elapsed + 1 >= math.ceil(fallback_mean))
        x=public_service_features(observation,query)[None,:]
        # Crucially call the official model predictor rather than re-create
        # its learned hazard/logistic transform in project code.
        predicted=self.model.predict_hazard(x,numpy=True)
        value=float(np.asarray(predicted)[0,active_elapsed+1])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("pycox produced invalid conditional service hazard")
        self.num_queries += 1
        return value
