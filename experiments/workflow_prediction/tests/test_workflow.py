from dataclasses import replace
import os
from pathlib import Path
import unittest

import torch

from business_transport.dynamics import CapacityDynamics,WorkflowDynamics
from business_transport.observation import BusinessObserver
from business_transport.scenario import compile_scenario
from business_transport.simulation import BusinessSimulator,resource_audit
from business_transport.workflow_scoring import workflow_components
from business_transport.workflow_transition import check_business_transition
from experiments.business_constraints.run import make_model,rich_rows
from experiments.business_constraints.tests.test_business import fixture
from experiments.service_value.counterfactual import load_pr9_model
from experiments.service_value.scoring import workload_components
from lab_transport.decision_value import nearest_batch
from lab_transport.dynamics_model import Geometry
from lab_transport.paper_reuse.contracts import check_transition
from lab_transport.torch_structured_adapter import ProgressAwareRule


def case(stages=2,capacity=1,jobs=1,outage=False):
    base,cfg,_=fixture(capacity)
    stations={key:replace(value,service_ticks=1,prior_ticks=1,jitter_ticks=0,
                          outages=((0,30),) if outage and key=='B0' else ())
              for key,value in base.stations.items()}
    base=replace(base,stations=stations,robots=base.robots[:max(1,jobs)])
    route=[{'pickup':'A0','dropoff':'B0'},{'pickup':'B0','dropoff':'C'},
           {'pickup':'C','dropoff':'B1'}][:stages]
    workflows={**cfg['workflows'],'w0':route}
    selected=cfg['jobs'][:jobs]
    return compile_scenario(base,{**cfg,'workflows':workflows,'jobs':selected,'max_ticks':80})


def factual(business):
    rows=[];geometry=Geometry(business.scenario.grid)
    result=BusinessSimulator(business,'world',1,collect_metrics=True).run(
        on_transition=rows.append,dispatch_policy=lambda obs,candidates:nearest_batch(obs,candidates,geometry))
    return result,rich_rows(rows),geometry


class WorkflowTransitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_single_two_three_stage_releases_and_next_tick_dispatch(self):
        for stages in (1,2,3):
            with self.subTest(stages=stages):
                business=case(stages);result,rows,geo=factual(business)
                self.assertEqual(result['summary']['status'],'completed')
                model=WorkflowDynamics(ProgressAwareRule(geo),business.rules)
                releases=[]
                for row in rows:
                    predicted=model.predict_next(row.observation,row.assignments)
                    check_business_transition(row.observation,predicted,geo,business.rules,row.assignments)
                    prior={s.id for s in row.observation.samples}
                    created=[s for s in predicted.samples if s.id not in prior]
                    for sample in created:
                        releases.append(sample)
                        self.assertEqual(sample.release_tick,row.observation.tick+1)
                        self.assertEqual(sample.status,'waiting')
                        self.assertIsNone(sample.robot_id)
                        self.assertTrue(all(r.task_id!=sample.id for r in predicted.robots))
                        actual={s.id:s for s in row.next_observation.samples}
                        self.assertEqual((sample.id,sample.release_tick,sample.workflow,sample.stage_index),
                          (actual[sample.id].id,actual[sample.id].release_tick,actual[sample.id].workflow,actual[sample.id].stage_index))
                self.assertEqual(len(releases),stages-1)
                self.assertEqual([s.id for s in releases],[f'J1::stage{i}' for i in range(2,stages+1)])

    def test_capacity_one_and_two_and_multiple_jobs(self):
        for cap in (1,2):
            business=case(2,cap,2);result,rows,geo=factual(business)
            self.assertEqual(result['summary']['status'],'completed')
            self.assertFalse(resource_audit(result,business.rules)['violations'])
            model=WorkflowDynamics(ProgressAwareRule(geo),business.rules)
            for row in rows:
                predicted=model.predict_next(row.observation,row.assignments)
                self.assertTrue(check_business_transition(row.observation,predicted,geo,business.rules,row.assignments))
                for station in predicted.stations:
                    self.assertLessEqual(station.resource_occupied,station.resource_capacity)

    def test_observed_outage_pauses_service_and_does_not_release(self):
        business=case(2,outage=True);_,rows,geo=factual(business)
        model=WorkflowDynamics(ProgressAwareRule(geo),business.rules)
        checked=0
        for row in rows:
            station=next(s for s in row.observation.stations if s.id=='B0')
            robot=next((r for r in row.observation.robots if r.task_id=='J1::stage1' and
                        r.phase=='to_dropoff' and r.position==station.position),None)
            if robot is None or station.device_available:continue
            predicted=model.predict_next(row.observation,row.assignments)
            check_business_transition(row.observation,predicted,geo,business.rules,row.assignments)
            self.assertEqual(next(s for s in predicted.samples if s.id=='J1::stage1').status,'in_transit')
            self.assertFalse(any(s.id=='J1::stage2' for s in predicted.samples))
            checked+=1
        self.assertGreater(checked,0)

    def test_dynamic_validator_rejects_unknown_early_duplicate_and_mutated_stages(self):
        business=case(2);_,rows,geo=factual(business)
        model=WorkflowDynamics(ProgressAwareRule(geo),business.rules)
        pairs=[(row,model.predict_next(row.observation,row.assignments)) for row in rows]
        row,predicted=next((r,p) for r,p in pairs if len(p.samples)>len(r.observation.samples))
        old=row.observation;created=next(s for s in predicted.samples if s.id not in {x.id for x in old.samples})
        bad=[replace(predicted,samples=predicted.samples+(created,)),
             replace(predicted,samples=tuple(s for s in predicted.samples if s.id!=created.id)),
             replace(predicted,samples=tuple(replace(s,job_id='stranger') if s.id==created.id else s for s in predicted.samples)),
             replace(predicted,samples=tuple(replace(s,status='assigned',robot_id='R1') if s.id==created.id else s for s in predicted.samples)),
             replace(predicted,samples=tuple(replace(s,deadline_tick=999) if s.id!=created.id else s for s in predicted.samples)),
             replace(predicted,samples=tuple(s for s in predicted.samples if s.id!=old.samples[0].id))]
        for value in bad:
            with self.assertRaises(ValueError):check_business_transition(old,value,geo,business.rules,row.assignments)
        with self.assertRaises(ValueError):check_transition(old,predicted,geo)
        early=rows[0].observation
        premature=replace(created,release_tick=early.tick+1)
        false=replace(model.predict_next(early,rows[0].assignments),
                      samples=model.predict_next(early,rows[0].assignments).samples+(premature,))
        with self.assertRaises(ValueError):check_business_transition(early,false,geo,business.rules,rows[0].assignments)

    def test_new_score_counts_one_unfinished_job_after_stage_one(self):
        business=case(2);_,rows,geo=factual(business)
        model=WorkflowDynamics(ProgressAwareRule(geo),business.rules)
        for row in rows:
            predicted=model.predict_next(row.observation,row.assignments)
            if any(s.id=='J1::stage2' for s in predicted.samples) and not any(
                    s.id=='J1::stage2' for s in row.observation.samples):break
        new=workflow_components(row.observation,[predicted],geo,business.rules)
        old=workload_components(row.observation,[predicted],geo)
        self.assertEqual(new['known_jobs'],1)
        self.assertEqual(new['unfinished_job_ticks'],1)
        self.assertEqual(old['known_unfinished_ticks'],0)
        self.assertGreater(new['terminal_work'],0)

    def test_three_models_use_same_legal_release_mechanism(self):
        business=case(2);_,rows,geo=factual(business);network,info=load_pr9_model()
        for name in ('rule','pets','structured'):
            model=WorkflowDynamics(make_model(name,geo,network,info['threshold'],37),business.rules)
            for row in rows:
                state=model.predict_next(row.observation,row.assignments)
                self.assertTrue(check_business_transition(row.observation,state,geo,business.rules,row.assignments))

    def test_unknown_future_jobs_and_outages_do_not_change_current_forecast(self):
        base,cfg,business=fixture(1)
        jobs=cfg['jobs']+[{'id':'future','workflow':'w0','release_tick':50}]
        stations={**base.stations,'C':replace(base.stations['C'],outages=((40,60),))}
        alternate=compile_scenario(replace(base,stations=stations),{**cfg,'jobs':jobs})
        forecasts=[]
        for item in (business,alternate):
            _,rows,geo=factual(item);row=rows[0]
            self.assertFalse(any(s.id.startswith('future') for s in row.observation.samples))
            model=WorkflowDynamics(ProgressAwareRule(geo),item.rules)
            forecasts.append(model.predict_next(row.observation,row.assignments))
        self.assertEqual(forecasts[0],forecasts[1])

    def test_stage_assignment_invalidates_cached_path_without_mutating_branch(self):
        from business_transport.lacam3_coordinator import RecoveryCoordinator,RecoverySnapshot
        from experiments.lacam3_recovery.tests.test_recovery import _FakePlanner,_Stay,inputs
        planner=_FakePlanner();coordinator=RecoveryCoordinator(planner,_Stay());snapshot=RecoverySnapshot()
        for tick in range(16):
            snapshot,_,_=coordinator.step(snapshot,inputs(),(('station',True),),0,False,tick)
        self.assertTrue(snapshot.cached_path)
        before=snapshot
        changed=inputs();changed[0]['position']=(1,0);changed[0]['task_id']='J1::stage2'
        branch,_,event=coordinator.step(snapshot,changed,(('station',True),),0,False,16)
        self.assertTrue(event['cache_invalidated'])
        self.assertFalse(event['triggered'])  # unchanged goal/lock must not be retried
        self.assertEqual(snapshot,before)
        self.assertNotEqual(branch,before)


@unittest.skipUnless(os.environ.get('WM_LACAM3_WORKER') and os.environ.get('WM_PIBT_LIBRARY') and
                     os.environ.get('WM_HOSPITAL_SOURCES'),'set the existing motion backends and sources')
class WorkflowRecoveryIntegrationTests(unittest.TestCase):
    def test_all_three_models_preserve_isolated_recovery_snapshot_across_stage_release(self):
        from business_transport.lacam3_integration import HybridBusinessSimulator,RecoveryMovementModel,RecoveryObserver
        from business_transport.policy import BusinessPolicy
        from experiments.pibt_motion.hospital_reproduction import load_bases,pilot_case
        from lab_transport.decision_value import candidates_for
        torch.set_num_threads(1)
        pibt=os.environ['WM_PIBT_LIBRARY'];lacam=os.environ['WM_LACAM3_WORKER']
        bases,_=load_bases(os.environ['WM_HOSPITAL_SOURCES']);business,_=pilot_case('clinic',bases['clinic'])
        geo=Geometry(business.scenario.grid);observer=RecoveryObserver()
        policy=BusinessPolicy('nearest',geo,business.rules,observer)
        sim=HybridBusinessSimulator(business,'world',61000,motion_backend='hybrid',pibt_library=pibt,
                                    lacam3_worker=lacam,collect_metrics=True)
        rows=[];result=sim.run(dispatch_policy=policy,on_observation=observer.observe,on_transition=rows.append)
        self.assertEqual(result['summary']['status'],'completed')
        history=RecoveryObserver();states=[history.observe(row.observation) for row in rows]
        network,info=load_pr9_model()
        for name in ('rule','pets','structured'):
            base=make_model(name,geo,network,info['threshold'],1)
            adapter=WorkflowDynamics(RecoveryMovementModel(base,geo,business.rules,pibt,lacam),business.rules)
            releases=0
            for row,old in zip(rows,states):
                snapshot=old.recovery_snapshot
                predicted=adapter.predict_next(old,row.assignments)
                check_business_transition(old,predicted,geo,business.rules,row.assignments)
                self.assertEqual(old.recovery_snapshot,snapshot)
                if len(predicted.samples)>len(old.samples):
                    releases+=1
                    self.assertTrue(hasattr(predicted,'recovery_snapshot'))
                    actions=nearest_batch(predicted,candidates_for(predicted),geo)
                    next_state=adapter.predict_next(predicted,actions)
                    check_business_transition(predicted,next_state,geo,business.rules,actions)
                    self.assertEqual(old.recovery_snapshot,snapshot)
            self.assertGreater(releases,0,name)


if __name__=='__main__':unittest.main()
