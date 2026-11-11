"""Service query correspondence, factual labeling and v2 rollout safeguards."""
from dataclasses import replace
import json
import os
import tempfile
from pathlib import Path
import unittest
import torch

from business_transport.session_events import eligible_sessions, encode_query, QUERY_WIDTH
from business_transport.session_dynamics import SessionRuleDynamics
from business_transport.state_features import build_business_features
from experiments.business_constraints.tests.test_business import fixture
from experiments.world_model_integration.build_dataset import label_for, build_dataset
from lab_transport.business_session_learning import (BusinessSessionEventModel, encode_session_batch,
                                                       verify_dataset, probabilities)


class SessionEventV2Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def _two_sessions(self):
        base,cfg,business=fixture(2)
        from business_transport.observation import BusinessObserver
        from lab_transport.dynamics_model import Geometry
        from business_transport.simulation import BusinessSimulator
        from lab_transport.decision_value import nearest_batch
        raw=[]
        geo=Geometry(base.grid)
        BusinessSimulator(business,"world",1).run(
            on_transition=raw.append,
            dispatch_policy=lambda o,c:nearest_batch(o,c,geo))
        before=BusinessObserver().observe(raw[0].observation)
        ids={"R1":("J1::stage1","A0"),"R2":("J2::stage1","A1")}
        robots=tuple(replace(r,position=next(st.position for st in before.stations if st.id==ids[r.id][1]),
            task_id=ids[r.id][0],phase="to_pickup",service_elapsed=1,
            service_active_elapsed=1,service_paused_elapsed=0) if r.id in ids else r
            for r in before.robots)
        samples=tuple(replace(s,status="assigned",robot_id="R1" if s.id.startswith("J1") else "R2")
                      for s in before.samples)
        stations=tuple(replace(s,resource_capacity=2,resource_occupied=2,
             service_session_active=True,service_in_progress=True,available=True,
             device_available=True,capacity_blocked=False) if s.id in ("A0","A1") else s
             for s in before.stations)
        return replace(before,robots=robots,samples=samples,stations=stations),geo,business

    def test_two_station_queries_and_swapped_labels(self):
        before,geo,business=self._two_sessions()
        session_rows=eligible_sessions(before,(),require_active=True)
        self.assertEqual(len(session_rows),2)
        q1,q2=(x[0] for x in session_rows)
        self.assertEqual({q1["station_id"],q2["station_id"]},{"A0","A1"})
        observed=build_business_features(before,(),geo,business.rules)
        vectors=encode_session_batch([{"features":observed,"query":q1},
                                      {"features":observed,"query":q2}])
        self.assertEqual(tuple(vectors[2].shape),(2,QUERY_WIDTH))
        self.assertFalse(torch.equal(vectors[2][0],vectors[2][1]))
        model=BusinessSessionEventModel()
        self.assertEqual(tuple(model(*vectors).shape),(2,))
        finished=replace(before,tick=before.tick+1,samples=tuple(replace(s,status="in_transit") if s.id==q1["sample_id"]
                                          else s for s in before.samples))
        self.assertEqual(label_for(before,finished,q1),(1,"finished"))
        self.assertEqual(label_for(before,finished,q2),(0,"continues"))
        self.assertEqual(label_for(before,finished,q2)[0],0)
        self.assertEqual(label_for(before,finished,q1)[0],1)
        with self.assertRaises(ValueError):
            encode_query(observed,{**q1,"station_id":q2["station_id"]})

    def test_permutation_invariance_and_observed_feature_sensitivity(self):
        obs,geo,business=self._two_sessions()
        query=eligible_sessions(obs,(),require_active=True)[0][0]
        base=build_business_features(obs,(),geo,business.rules)
        changed=replace(obs,stations=tuple(replace(s,resource_capacity=1) if
                                           s.id==query["station_id"] else s for s in obs.stations))
        reordered=replace(obs,robots=tuple(reversed(obs.robots)),
                          samples=tuple(reversed(obs.samples)),
                          stations=tuple(reversed(obs.stations)))
        self.assertEqual(encode_query(base,query),
                         encode_query(build_business_features(reordered,(),geo,business.rules),query))
        self.assertNotEqual(encode_query(base,query),
                            encode_query(build_business_features(changed,(),geo,business.rules),query))
        masked=replace(obs,robots=tuple(replace(r,service_active_elapsed=None) if
                         r.id==query["robot_id"] else r for r in obs.robots))
        vector=encode_query(build_business_features(masked,(),geo,business.rules),query)
        self.assertNotEqual(vector,encode_query(base,query))

    def test_data_contract_invariants_and_group_separation(self):
        dataset=build_dataset()
        counts=verify_dataset(dataset)
        self.assertGreater(counts["train"]["rows"],100)
        self.assertGreater(counts["validation"]["positive"],0)
        self.assertGreater(counts["test"]["positive"],0)
        ids={split:{r["trajectory_group"] for r in dataset[split]}
             for split in ("train","validation","test")}
        self.assertFalse(ids["train"]&ids["test"])
        self.assertFalse(ids["train"]&ids["validation"])
        self.assertFalse(ids["test"]&ids["validation"])
        self.assertGreater(dataset["counts"]["train"].get("excluded_not_started",0),0)
        self.assertGreater(dataset["counts"]["train"].get("excluded_observed_device_pause",0),0)
        self.assertTrue(all(row["event"] in ("continues","finished") for row in dataset["train"]))
        bad=json.loads(json.dumps(dataset))
        bad["test"][0]["trajectory_group"]=bad["train"][0]["trajectory_group"]
        with self.assertRaisesRegex(ValueError,"leaked"):
            verify_dataset(bad)

    @unittest.skipUnless(os.environ.get("WM_SESSION_CHECKPOINT"),
                         "CI supplies a freshly trained, offline-only checkpoint")
    def test_trained_session_events_drive_legal_workflow_rollout(self):
        from experiments.business_constraints.run import rich_rows
        from business_transport.dynamics import WorkflowDynamics
        from business_transport.workflow_transition import check_business_transition
        from business_transport.policy import WorkflowPolicy
        from business_transport.observation import BusinessObserver
        from lab_transport.decision_value import candidates_for, nearest_batch
        from lab_transport.dynamics_model import Geometry
        from business_transport.simulation import BusinessSimulator
        base,cfg,business=fixture(1)
        geo=Geometry(base.grid)
        raw=[]
        BusinessSimulator(business,"world",1).run(
            on_transition=raw.append,
            dispatch_policy=lambda o,c:nearest_batch(o,c,geo))
        obs=rich_rows(raw)[0].observation
        model=WorkflowDynamics(SessionRuleDynamics(geo,business.rules,
            os.environ["WM_SESSION_CHECKPOINT"],experimental=True,seed=1810),business.rules)
        state=obs
        for tick in range(20):
            actions=nearest_batch(state,candidates_for(state),geo)
            predicted=model.predict_next(state,actions)
            self.assertTrue(check_business_transition(state,predicted,geo,business.rules,actions))
            state=predicted
        self.assertGreater(model.model.query_calls,0)
        policy=WorkflowPolicy("cpsat_session",geo,business.rules,
                              BusinessObserver(),SessionRuleDynamics(geo,business.rules,
                              os.environ["WM_SESSION_CHECKPOINT"],experimental=True),
                              seed=1810,model_aware=True,horizon=10,particles=2,budget=2)
        selected=policy(obs,candidates_for(obs))
        self.assertEqual(policy.fallbacks,0,policy.log[-1]["fallback"])
        self.assertTrue(all(a in candidates_for(obs) for a in selected))

    def test_uncertified_checkpoint_cannot_be_deployed(self):
        with tempfile.TemporaryDirectory() as folder:
            ckpt=Path(folder)/"unverified.pt"
            torch.save({"metadata":{"model_version":"business-sets-session-v2",
                  "schema_version":"business-observable-v1","label_version":"business-service-query-v2",
                  "online_verified":False},"model_state":BusinessSessionEventModel().state_dict()},ckpt)
            from lab_transport.business_session_learning import load_session_model
            with self.assertRaisesRegex(ValueError,"not certified"):
                load_session_model(ckpt)
            model,metadata=load_session_model(ckpt,experimental=True)
            self.assertFalse(metadata["online_verified"])


if __name__=="__main__":
    unittest.main()
