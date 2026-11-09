from dataclasses import replace
from pathlib import Path
import unittest

from business_transport.dispatch import BusinessCPSAT
from business_transport.dynamics import CapacityDynamics
from business_transport.observation import BusinessObserver
from business_transport.policy import BusinessPolicy
from business_transport.scenario import compile_scenario
from business_transport.simulation import BusinessSimulator,resource_audit
from lab_transport.decision_value import candidates_for,nearest_batch
from lab_transport.domain import Grid,RobotSpec,Sample,Scenario,Station,load_scenario
from lab_transport.dynamics_model import Geometry
from lab_transport.paper_reuse.contracts import check_transition
from lab_transport.paper_reuse.ortools_dispatch import CPSATDispatcher
from lab_transport.paper_reuse.pets_service import PETSRuleDynamics,ServiceDistribution
from lab_transport.service_dynamics import RuleMotionNeuralService
from experiments.service_value.counterfactual import load_pr9_model
from experiments.business_constraints.run import lifecycle_audit,run_policy
from lab_transport.simulation import Simulator
from lab_transport.torch_structured_adapter import ProgressAwareRule
from lab_transport.transition_schema import Assignment

ROOT=Path(__file__).resolve().parents[3]


def fixture(capacity=1):
    stations={s.id:s for s in [Station('A0',(1,1),3,3,(),0),Station('A1',(1,3),3,3,(),0),
        Station('B0',(5,1),2,2,(),0),Station('B1',(5,3),2,2,(),0),Station('C',(3,2),1,1,(),0)]}
    robots=(RobotSpec('R1',(1,1),(0,0)),RobotSpec('R2',(1,3),(0,4)),RobotSpec('R3',(0,2),(0,2)))
    base=Scenario('unit',Grid(7,5,frozenset()),stations,robots,(),0,150)
    config={'resources':[{'id':'A','stations':['A0','A1'],'capacity':capacity},
                         {'id':'B','stations':['B0','B1'],'capacity':2},
                         {'id':'station:C','stations':['C'],'capacity':1}],
      'workflows':{'w0':[{'pickup':'A0','dropoff':'B0'},{'pickup':'B0','dropoff':'C'}],
                   'w1':[{'pickup':'A1','dropoff':'B1'},{'pickup':'B1','dropoff':'C'}]},
      'jobs':[{'id':'J1','workflow':'w0','release_tick':0},{'id':'J2','workflow':'w1','release_tick':0}],
      'max_ticks':150}
    return base,config,compile_scenario(base,config)


def run_nearest(business):
    geo=Geometry(business.scenario.grid)
    observer=BusinessObserver()
    policy=BusinessPolicy('nearest',geo,business.rules,observer)
    result=BusinessSimulator(business,'world',1,collect_metrics=True).run(
        dispatch_policy=policy,on_observation=observer.observe)
    return result,policy


class BusinessConstraintsTests(unittest.TestCase):
    def test_existing_single_stage_simulator_is_unchanged(self):
        case=load_scenario(ROOT/'scenarios/laboratory_demo.json')
        old=Simulator(case,'nearest',7,collect_metrics=True).run()
        new=BusinessSimulator(None,'nearest',7,scenario=case,collect_metrics=True).run()
        self.assertEqual(old,new)

    def test_existing_cpsat_without_business_rules_is_unchanged(self):
        case=load_scenario(ROOT/'scenarios/laboratory_demo.json')
        rows=[];Simulator(case,'nearest',7).run(on_transition=rows.append)
        current=rows[0].observation;geo=Geometry(case.grid);candidates=candidates_for(current)
        old=CPSATDispatcher(geo,seed=7).proposals(current,candidates)
        new=BusinessCPSAT(geo,None,seed=7).proposals(current,candidates)
        self.assertEqual(old,new)

    def test_configuration_rejects_overcapacity_and_disconnected_stages(self):
        base,cfg,_=fixture()
        with self.assertRaisesRegex(ValueError,'capacity'):
            compile_scenario(base,{**cfg,'resources':[{'id':'bad','stations':['A0'],'capacity':2}]})
        with self.assertRaisesRegex(ValueError,'explicitly configured capacity'):
            compile_scenario(base,{**cfg,'resources':cfg['resources'][:-1]})
        with self.assertRaisesRegex(ValueError,'teleporting'):
            compile_scenario(base,{**cfg,'workflows':{'w0':[{'pickup':'A0','dropoff':'B0'},
                                               {'pickup':'A1','dropoff':'C'}],'w1':cfg['workflows']['w1']}})

    def test_capacity_and_sequential_stage_execution(self):
        _,_,business=fixture(1)
        result,_=run_nearest(business)
        audit=resource_audit(result,business.rules)
        self.assertEqual(result['summary']['status'],'completed')
        self.assertGreater(audit['capacity_wait_ticks'],0)
        self.assertEqual(audit['peaks']['A'],1)
        self.assertEqual(audit['peaks']['B'],2)
        self.assertEqual(audit['violations'],[])
        samples={s['id']:s for s in result['samples']}
        for job in business.jobs:
            first,second=(samples[x] for x in job['stages'])
            self.assertEqual(first['completed_tick'],second['release_tick'])
            self.assertGreaterEqual(second['assigned_tick'],first['completed_tick'])

    def test_capacity_two_allows_both_ports(self):
        _,_,business=fixture(2)
        result,_=run_nearest(business)
        audit=resource_audit(result,business.rules)
        self.assertEqual(audit['peaks']['A'],2)
        self.assertEqual(audit['violations'],[])
        self.assertEqual(result['summary']['status'],'completed')

    def test_three_configured_stages_release_only_after_predecessor(self):
        base,cfg,_=fixture(1)
        route=cfg['workflows']['w0']+[{'pickup':'C','dropoff':'B1'}]
        data={**cfg,'workflows':{**cfg['workflows'],'w0':route},
              'jobs':[{'id':'J1','workflow':'w0','release_tick':0}]}
        business=compile_scenario(base,data)
        result,_=run_nearest(business)
        stages={s['id']:s for s in result['samples']}
        self.assertEqual(result['summary']['status'],'completed')
        for left,right in zip(business.jobs[0]['stages'],business.jobs[0]['stages'][1:]):
            self.assertEqual(stages[left]['completed_tick'],stages[right]['release_tick'])
            self.assertGreaterEqual(stages[right]['assigned_tick'],stages[left]['completed_tick'])

    def test_cpsat_enforces_capacity_and_precedence(self):
        _,_,business=fixture(1);states=[]
        sim=BusinessSimulator(business,'world',1,collect_metrics=True)
        geo=Geometry(business.scenario.grid);observer=BusinessObserver()
        solver=BusinessCPSAT(geo,business.rules,time_limit_seconds=.5,candidate_limit=4,seed=1)
        def policy(obs,candidates):
            rich=observer.observe(obs)
            plans=solver.proposals(rich,candidates)
            if not states:states.append((rich,candidates,plans,solver.log[-1]))
            return plans[0]
        result=sim.run(dispatch_policy=policy,on_observation=observer.observe)
        self.assertEqual(result['summary']['status'],'completed')
        self.assertFalse(resource_audit(result,business.rules)['violations'])
        _,candidates,plans,entry=states[0]
        self.assertTrue(all(a in candidates for plan in plans for a in plan))
        for schedule in entry['schedules']:
            self.assertTrue(any(op['stage_offset']==1 for op in schedule))
            for tick in range(max(op['end'] for op in schedule)):
                active=[o for o in schedule if o['resource_id']=='A' and o['start']<=tick<o['end']]
                self.assertLessEqual(len(active),1)
            groups={}
            for op in schedule:groups.setdefault(op['sample_id'],[]).append(op)
            for ops in groups.values():
                for left,right in zip(ops,ops[1:]):self.assertLessEqual(left['end'],right['start'])

    def test_future_tasks_and_faults_do_not_enter_current_decision(self):
        base,cfg,business=fixture(1)
        modified={**cfg,'jobs':cfg['jobs']+[{'id':'future','workflow':'w0','release_tick':50}]}
        altered_base=replace(base,stations={**base.stations,'C':replace(base.stations['C'],outages=((40,60),))})
        alt=compile_scenario(altered_base,modified)
        values=[]
        for case in (business,alt):
            rows=[];sim=BusinessSimulator(case,'world',1)
            sim.run(on_transition=rows.append)
            current=BusinessObserver().observe(rows[0].observation)
            self.assertFalse(any(s.id.startswith('future') for s in current.samples))
            solver=BusinessCPSAT(Geometry(case.scenario.grid),case.rules,seed=1)
            values.append((current,solver.proposals(current,candidates_for(current))))
        self.assertEqual(values[0],values[1])

    def test_policy_randomness_is_independent_of_simulator_seed(self):
        import torch
        torch.set_num_threads(1)
        base,cfg,_=fixture(1)
        noisy=replace(base,stations={key:replace(station,jitter_ticks=2)
                                     for key,station in base.stations.items()})
        business=compile_scenario(noisy,cfg)
        outputs=[]
        for environment_seed in (11,17):
            _,rows,metrics=run_policy(business,environment_seed,'cpsat_pets',None,None,
                                      capture=True,policy_seed=20261109)
            outputs.append((rows[0].observation,metrics['decisions'][0]['plans'],
                            metrics['decisions'][0]['selected']))
        self.assertEqual(outputs[0],outputs[1])

    def test_blocked_service_does_not_gain_fake_progress_in_prediction(self):
        _,_,business=fixture(1);rows=[]
        geo=Geometry(business.scenario.grid)
        policy=lambda obs,candidates:nearest_batch(obs,candidates,geo)
        BusinessSimulator(business,'world',1).run(on_transition=rows.append,dispatch_policy=policy)
        observer=BusinessObserver();obs=observer.observe(rows[0].observation)
        dynamic=CapacityDynamics(ProgressAwareRule(geo),business.rules)
        predicted=dynamic.predict_next(obs,rows[0].assignments)
        blocked=next(r for r in predicted.robots if r.id=='R2')
        self.assertEqual(blocked.task_id,'J2::stage1')
        self.assertIsNone(blocked.service_elapsed)
        self.assertIsNone(blocked.service_active_elapsed)

    def test_all_three_existing_models_obey_resource_admission(self):
        import torch
        _,_,business=fixture(1);rows=[];geo=Geometry(business.scenario.grid)
        BusinessSimulator(business,'world',1).run(on_transition=rows.append,
            dispatch_policy=lambda o,c:nearest_batch(o,c,geo))
        obs=BusinessObserver().observe(rows[0].observation)
        network,info=load_pr9_model()
        checkpoint=ROOT/'examples/evidence/paper-reuse-integration-20261109/pets-service-cpu.pt'
        for model in (ProgressAwareRule(geo),
                      PETSRuleDynamics(geo,ServiceDistribution.load(checkpoint),1),
                      RuleMotionNeuralService(geo,network,torch.device('cpu'),info['threshold'])):
            predicted=CapacityDynamics(model,business.rules).predict_next(obs,rows[0].assignments)
            check_transition(obs,predicted,geo)
            blocked=next(r for r in predicted.robots if r.id=='R2')
            self.assertIsNone(blocked.service_elapsed)
            self.assertEqual(blocked.task_id,'J2::stage1')

    def test_capacity_auditor_detects_overlapping_services(self):
        _,_,business=fixture(1)
        events=[{'type':'service_started','tick':0,'robot_id':'R1','sample_id':'s1','phase':'to_pickup','station_id':'A0'},
                {'type':'service_started','tick':0,'robot_id':'R2','sample_id':'s2','phase':'to_pickup','station_id':'A1'},
                {'type':'service_finished','tick':1,'robot_id':'R1','sample_id':'s1','phase':'to_pickup','station_id':'A0'},
                {'type':'service_finished','tick':1,'robot_id':'R2','sample_id':'s2','phase':'to_pickup','station_id':'A1'}]
        result={'events':events,'summary':{'ticks':1}}
        self.assertTrue(resource_audit(result,business.rules)['violations'])

    def test_lifecycle_auditor_detects_invalid_reverse_transition(self):
        snapshots=[{'tick':0,'robots':[],'samples':[{'id':'S','status':'completed','robot_id':None}]},
                   {'tick':1,'robots':[],'samples':[{'id':'S','status':'waiting','robot_id':None}]}]
        self.assertEqual(lifecycle_audit(snapshots)[0]['type'],'invalid_status_transition')

    def test_nearest_rollout_and_cpsat_have_bounded_legal_candidates(self):
        _,_,business=fixture(1)
        geo=Geometry(business.scenario.grid)
        for method in ('nearest_rollout','cpsat_rule'):
            obs=BusinessObserver();p=BusinessPolicy(method,geo,business.rules,obs,ProgressAwareRule(geo),seed=1)
            result=BusinessSimulator(business,'world',1,collect_metrics=True).run(dispatch_policy=p,on_observation=obs.observe)
            self.assertEqual(result['summary']['status'],'completed')
            self.assertFalse(resource_audit(result,business.rules)['violations'])
            self.assertTrue(all(len(x['plans'])<=4 for x in p.log))
            self.assertEqual(p.fallbacks,0)

if __name__=='__main__':unittest.main()
