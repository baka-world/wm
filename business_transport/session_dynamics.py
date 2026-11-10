"""Explicitly experimental v2 per-session completion model + existing legal rules.

The network only samples service completion for observed and eligible service
sessions. Motion, capacity and workflow successor semantics remain unchanged.
Online loads are rejected unless separately certified; experiment=True is
visible opt-in for offline research, never production admission.
"""
from __future__ import annotations

from dataclasses import replace
import random
import torch

from business_transport.session_events import eligible_sessions
from business_transport.state_features import build_business_features
from lab_transport.business_session_learning import encode_session_batch, load_session_model
from lab_transport.torch_structured_adapter import ProgressAwareRule
from lab_transport.state_v2 import ObservationV2
from lab_transport.dynamics_model import _intent


class SessionRuleDynamics:
    def __init__(self,geometry,rules,checkpoint,*,experimental=False,seed=1810,
                 feature_version='v2'):
        if feature_version=='v2':
            self.network,self.metadata=load_session_model(checkpoint,experimental=experimental)
            self.query_input_schema='business-service-query-input-v2'
        elif feature_version=='v3':
            from lab_transport.business_session_v3 import load_v3
            self.network,self.metadata=load_v3(checkpoint,experimental=experimental)
            self.query_input_schema=self.metadata['input_schema']
        else:
            raise ValueError('unknown frozen service-query feature version')
        self.network.eval()
        self.geometry,self.rules=geometry,rules
        self.rule=ProgressAwareRule(geometry)
        self.rng=random.Random(seed)
        self.service_calls=0
        self.query_calls=0
        self.fallback_reasons={}
        self.experimental=experimental

    def reset_seed(self,seed):
        self.rng.seed(seed)

    def predict_next(self,observation,assignments):
        if not isinstance(observation,ObservationV2):
            raise ValueError("v2 per-session inference requires observable history V2")
        assignments=tuple(assignments)
        query_rows=eligible_sessions(observation,assignments)
        viable=[q for q,reason in query_rows if reason is None]
        for _,reason in query_rows:
            if reason is not None:
                self.fallback_reasons[reason]=self.fallback_reasons.get(reason,0)+1
        overrides={}
        if viable:
            features=build_business_features(observation,assignments,self.geometry,self.rules)
            rows=[{"features":features,"query":q} for q in viable]
            with torch.inference_mode():
                if self.query_input_schema=='business-service-query-input-v2':
                    tokens,masks,queries=encode_session_batch(rows)
                else:
                    from lab_transport.business_session_v3 import encode_rows
                    tokens,masks,queries=encode_rows(rows,self.query_input_schema)
                probabilities=torch.sigmoid(self.network(tokens,masks,queries)).tolist()
            self.service_calls+=1
            self.query_calls+=len(viable)
            by_robot={r.id:r for r in observation.robots}
            for query,prob in zip(viable,probabilities):
                if not 0<=prob<=1:
                    raise ValueError("invalid learned service completion probability")
                robot=by_robot[query["robot_id"]]
                progress=robot.service_active_elapsed
                # Set the rule model's observed duration threshold to realize
                # the sampled completion event. No job or robot state fabricated.
                advance=self.rng.random()<prob
                overrides[query["station_id"]]=progress+(1 if advance else 2)
        if overrides:
            modified=replace(observation,stations=tuple(
                replace(station,expected_service_ticks=overrides[station.id],
                        observed_effective_service_mean=float(overrides[station.id]))
                if station.id in overrides else station for station in observation.stations))
        else:
            modified=observation
        result=self.rule.predict_next(modified,assignments)
        previous={s.id:s for s in observation.stations}
        return replace(result,stations=tuple(
            replace(station,
                    expected_service_ticks=previous[station.id].expected_service_ticks,
                    observed_effective_service_mean=previous[station.id].observed_effective_service_mean)
            for station in result.stations))
