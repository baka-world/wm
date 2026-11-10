from dataclasses import asdict,replace
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
from experiments.workflow_prediction.verify import contains_hidden
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

    def test_evidence_auditor_flags_hidden_fields(self):
        self.assertTrue(contains_hidden({'robots':[{'service_remaining':3}]}))
        self.assertFalse(contains_hidden({'robots':[{'service_elapsed':3}]}))

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
                        previous=next(s for s in row.observation.samples if s.job_id==sample.job_id and
                                      s.stage_index==sample.stage_index-1)
                        released_robot=next(r for r in predicted.robots if r.id==previous.robot_id)
                        self.assertEqual((released_robot.task_id,released_robot.phase),(None,'idle'))
                        actual={s.id:s for s in row.next_observation.samples}
                        self.assertEqual((sample.id,sample.release_tick,sample.workflow,sample.stage_index),
                          (actual[sample.id].id,actual[sample.id].release_tick,actual[sample.id].workflow,actual[sample.id].stage_index))
                self.assertEqual(len(releases),stages-1)
                self.assertEqual([s.id for s in releases],[f'J1::stage{i}' for i in range(2,stages+1)])

    def test_recursive_prediction_completes_all_three_stages_without_factual_future(self):
        from lab_transport.decision_value import candidates_for
        business=case(3);_,rows,geo=factual(business)
        state=rows[0].observation;model=WorkflowDynamics(ProgressAwareRule(geo),business.rules)
        released=[]
        for tick in range(80):
            actions=nearest_batch(state,candidates_for(state),geo)
            predicted=model.predict_next(state,actions)
            check_business_transition(state,predicted,geo,business.rules,actions)
            self.assertEqual(actions,rows[tick].assignments)
            actual=rows[tick].next_observation
            self.assertEqual([(s.id,s.status,s.release_tick) for s in predicted.samples],
                             [(s.id,s.status,s.release_tick) for s in actual.samples])
            self.assertEqual([(r.position,r.task_id,r.phase) for r in predicted.robots],
                             [(r.position,r.task_id,r.phase) for r in actual.robots])
            released.extend(s.id for s in predicted.samples if s.id not in {x.id for x in state.samples})
            state=predicted
            if any(s.id=='J1::stage3' and s.status=='completed' for s in state.samples):break
        self.assertEqual(released,['J1::stage2','J1::stage3'])
        self.assertEqual({s.id:s.status for s in state.samples},
                         {'J1::stage1':'completed','J1::stage2':'completed','J1::stage3':'completed'})

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

    def test_capacity_denial_clears_station_service_relationship(self):
        business=case(2,1,2);_,rows,geo=factual(business)
        model=WorkflowDynamics(ProgressAwareRule(geo),business.rules)
        old_model=CapacityDynamics(ProgressAwareRule(geo),business.rules);corrected=0
        for row in rows:
            old=old_model.predict_next(row.observation,row.assignments)
            predicted=model.predict_next(row.observation,row.assignments)
            check_business_transition(row.observation,predicted,geo,business.rules,row.assignments)
            for previous,station in zip(old.stations,predicted.stations):
                if previous.service_session_active and not station.service_session_active:
                    robot=next(r for r in predicted.robots if r.id==station.occupant_robot_id)
                    self.assertIsNone(robot.service_elapsed)
                    self.assertIsNone(station.service_task_id)
                    corrected+=1
        self.assertGreater(corrected,0)

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
             replace(predicted,samples=tuple(replace(s,priority=9) if s.id!=created.id else s for s in predicted.samples)),
             replace(predicted,samples=tuple(s for s in predicted.samples if s.id!=old.samples[0].id)),
             replace(predicted,stations=(replace(predicted.stations[0],resource_occupied=99),)+predicted.stations[1:]),
             replace(predicted,robots=(replace(predicted.robots[0],position=(999,999)),)+predicted.robots[1:])]
        for value in bad:
            with self.assertRaises(ValueError):check_business_transition(old,value,geo,business.rules,row.assignments)
        with self.assertRaises(ValueError):check_transition(old,predicted,geo)
        early=rows[0].observation
        premature=replace(created,release_tick=early.tick+1)
        false=replace(model.predict_next(early,rows[0].assignments),
                      samples=model.predict_next(early,rows[0].assignments).samples+(premature,))
        with self.assertRaises(ValueError):check_business_transition(early,false,geo,business.rules,rows[0].assignments)

    def test_successor_inherits_public_sample_properties(self):
        business=case(2);_,rows,geo=factual(business)
        model=WorkflowDynamics(ProgressAwareRule(geo),business.rules)
        row=next(row for row in rows if any(s.id=='J1::stage2' for s in row.next_observation.samples) and
                 not any(s.id=='J1::stage2' for s in row.observation.samples))
        updated=tuple(replace(s,deadline_tick=50,specimen_type='public-type',packaging='sealed',
                              temperature_condition='cool',rule_profile='visible',priority=2)
                      for s in row.observation.samples)
        before=replace(row.observation,samples=updated)
        after=model.predict_next(before,row.assignments)
        check_business_transition(before,after,geo,business.rules,row.assignments)
        successor=next(s for s in after.samples if s.id=='J1::stage2')
        self.assertEqual((successor.deadline_tick,successor.specimen_type,successor.packaging,
                          successor.temperature_condition,successor.rule_profile,successor.priority),
                         (50,'public-type','sealed','cool','visible',2))

    def test_configured_properties_match_real_successor_observation(self):
        base,cfg,_=fixture(1)
        job={**cfg['jobs'][0],'specimen_type':'example','packaging':'sealed',
             'temperature_condition':'cool','rule_profile':'visible','deadline_tick':80}
        business=compile_scenario(base,{**cfg,'jobs':[job]})
        _,rows,geo=factual(business);model=WorkflowDynamics(ProgressAwareRule(geo),business.rules)
        row=next(row for row in rows if any(s.id=='J1::stage2' for s in row.next_observation.samples) and
                 not any(s.id=='J1::stage2' for s in row.observation.samples))
        predicted=model.predict_next(row.observation,row.assignments)
        check_business_transition(row.observation,predicted,geo,business.rules,row.assignments)
        actual=next(s for s in row.next_observation.samples if s.id=='J1::stage2')
        estimate=next(s for s in predicted.samples if s.id=='J1::stage2')
        fields=('deadline_tick','specimen_type','packaging','temperature_condition','rule_profile')
        self.assertEqual(tuple(getattr(estimate,k) for k in fields),tuple(getattr(actual,k) for k in fields))

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

    def test_workflow_score_repeats_across_python_hash_seeds(self):
        import subprocess
        import sys
        script='''
from dataclasses import replace
import json
from experiments.business_constraints.tests.test_business import fixture
from business_transport.scenario import compile_scenario
from business_transport.observation import BusinessObserver
from business_transport.simulation import BusinessSimulator
from business_transport.workflow_scoring import workflow_components
from lab_transport.decision_value import nearest_batch
from lab_transport.dynamics_model import Geometry
base,cfg,_=fixture(1)
jobs=[{'id':f'J{i}', 'workflow':('w0' if i%2 else 'w1'), 'release_tick':0} for i in range(1,7)]
business=compile_scenario(base,{**cfg,'jobs':jobs});geo=Geometry(base.grid);rows=[]
BusinessSimulator(business,'world',1).run(on_transition=rows.append,dispatch_policy=lambda o,c:nearest_batch(o,c,geo))
obs=BusinessObserver().observe(rows[0].observation)
obs=replace(obs,stations=tuple(replace(s,observed_effective_service_mean=(i+1)*1.1) for i,s in enumerate(obs.stations)))
print(json.dumps(workflow_components(obs,[obs],geo,business.rules),sort_keys=True))
'''
        outputs=[]
        for seed in ('123','456'):
            env={**os.environ,'PYTHONHASHSEED':seed}
            outputs.append(subprocess.check_output([sys.executable,'-c',script],env=env,text=True))
        self.assertEqual(outputs[0],outputs[1])

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

    def test_prediction_failure_records_fallback_to_legal_candidate(self):
        from business_transport.policy import BusinessPolicy
        from lab_transport.decision_value import candidates_for
        class BrokenModel:
            def reset_seed(self,seed):pass
            def predict_next(self,observation,assignments):raise RuntimeError('injected test failure')
        business=case(2);_,rows,geo=factual(business)
        observer=BusinessObserver()
        policy=BusinessPolicy('cpsat_rule',geo,business.rules,observer,BrokenModel(),
                              workflow_prediction=True,scoring='workflow')
        observation=rows[0].observation;legal=candidates_for(observation)
        chosen=policy(observation,legal)
        self.assertEqual(policy.fallbacks,1)
        self.assertEqual(policy.log[-1]['fallback'],'RuntimeError')
        self.assertTrue(all(action in legal for action in chosen))

    def test_public_workflow_policy_enables_business_prediction(self):
        from business_transport.policy import WorkflowPolicy
        business=case(2);_,rows,geo=factual(business)
        policy=WorkflowPolicy('cpsat_rule',geo,business.rules,BusinessObserver(),ProgressAwareRule(geo))
        self.assertTrue(policy.workflow_prediction)
        self.assertEqual(policy.scoring,'workflow')
        from lab_transport.decision_value import candidates_for
        chosen=policy(rows[0].observation,candidates_for(rows[0].observation))
        self.assertEqual(policy.fallbacks,0)
        self.assertTrue(chosen)

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
    def test_fact_window_metrics_run_and_conditional_logic_remains_consistent(self):
        from business_transport.lacam3_integration import HybridBusinessSimulator,RecoveryObserver
        from business_transport.policy import BusinessPolicy
        from experiments.workflow_prediction.evaluate import evaluate_state_windows,rich_rows
        torch.set_num_threads(1)
        business=case(2,1,2);geo=Geometry(business.scenario.grid);observer=RecoveryObserver()
        policy=BusinessPolicy('nearest',geo,business.rules,observer)
        sim=HybridBusinessSimulator(business,'world',1,motion_backend='hybrid',
              pibt_library=os.environ['WM_PIBT_LIBRARY'],lacam3_worker=os.environ['WM_LACAM3_WORKER'])
        rows=[];sim.run(dispatch_policy=policy,on_observation=observer.observe,on_transition=rows.append)
        network,info=load_pr9_model()
        config={'prediction_window_stride':5,'prediction_horizons':[1,3,5],'planner_seed':20261109}
        observed=rich_rows(rows)
        result=evaluate_state_windows(observed,business,geo,network,info['threshold'],
                        Path(os.environ['WM_PIBT_LIBRARY']),Path(os.environ['WM_LACAM3_WORKER']),config)
        self.assertGreater(result['old_fixed_entity_model_missing_successors'],0)
        for horizon in ('1','3','5'):
            for model in result['horizons'][horizon]['dynamic'].values():
                self.assertEqual(model.get('conditional_release_logic_errors',0),0)
                self.assertEqual(model.get('conditional_resource_errors',0),0)
        from experiments.workflow_prediction.coverage_report import report_case
        case_data={'layout':'unit','seed':1,'assumptions':{'workflows':business.rules.workflows},
                   'state_prediction':result,'factual_visible_transitions':[
                       {'observation':asdict(r.observation),'assignments':[asdict(a) for a in r.assignments],
                        'next_observation':asdict(r.next_observation)} for r in observed]}
        coverage=report_case(case_data,1)
        self.assertGreater(coverage['eligible_active_multistage'],0)

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
