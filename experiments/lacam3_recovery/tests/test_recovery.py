import os
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from business_transport.lacam3_coordinator import RecoveryCoordinator,RecoverySnapshot
from business_transport.lacam3_integration import HybridBusinessSimulator,RecoveryMovementModel,RecoveryObserver
from business_transport.lacam3_motion import JointPlanResult,LaCAM3Planner
from business_transport.dynamics import CapacityDynamics
from business_transport.policy import BusinessPolicy
from experiments.lacam3_recovery.clinic_states import frozen_states
from experiments.lacam3_recovery.benchmark import plan_sessions
from experiments.pibt_motion.hospital_reproduction import load_bases,pilot_case
from lab_transport.domain import DirectedGraph,Grid
from lab_transport.dynamics_model import Geometry
from lab_transport.torch_structured_adapter import ProgressAwareRule

LACAM=os.environ.get('WM_LACAM3_WORKER');PIBT=os.environ.get('WM_PIBT_LIBRARY')
SOURCES=os.environ.get('WM_HOSPITAL_SOURCES')


class _Stay:
    def step(self,positions,goals,locked,tick):return dict(positions)


class _FakePlanner:
    def __init__(self,status='solved'):self.calls=0;self.status=status
    def plan(self,positions,goals,locked,priority):
        self.calls+=1;ids=tuple(sorted(positions));current=tuple(positions[x] for x in ids)
        path=(current,((1,0),(3,0)),((2,0),(3,0))) if self.status=='solved' else ()
        return JointPlanResult(self.status,path,ids,priority,False,False,1.0,2000,0.5,True,'')
    def validate(self,ids,before,after,locked):
        if any(before[i]!=after[i] for i,rid in enumerate(ids) if rid in locked):
            raise ValueError('locked')


def inputs(goal=(2,0),locked=False):
    return [dict(id='A',position=(0,0),goal=goal,task_id='S',phase='to_dropoff',release_tick=0,locked=locked),
            dict(id='B',position=(3,0),goal=(3,0),task_id=None,phase='idle',release_tick=None,locked=False)]


class RecoveryLogicTests(unittest.TestCase):
    def test_plan_execution_accounting_handles_invalidation(self):
        def event(triggered=False, status='idle', steps=0, used=False, invalidated=False):
            return {'tick': 0, 'triggered': triggered, 'status': status,
                    'plan_steps': steps, 'cache_used': used, 'cache_invalidated': invalidated}
        events=[event(True,'solved',4,True),event(used=True),event(invalidated=True),
                event(True,'solved',2,True),event(used=True),event()]
        sessions=plan_sessions(events)
        self.assertEqual([(s['planned_steps'],s['executed_steps']) for s in sessions],[(4,2),(2,2)])
        self.assertEqual([s['executed_over_planned'] for s in sessions],[.5,1.0])

    def test_normal_service_never_triggers_on_repeated_positions(self):
        fake=_FakePlanner();coordinator=RecoveryCoordinator(fake,_Stay());state=RecoverySnapshot()
        for tick in range(40):state,_,event=coordinator.step(state,inputs(),(('station',True),),0,True,tick)
        self.assertEqual(fake.calls,0);self.assertFalse(event['triggered'])

    def test_cache_is_consumed_and_invalidated_on_context_or_position_change(self):
        fake=_FakePlanner();c=RecoveryCoordinator(fake,_Stay());state=RecoverySnapshot()
        for tick in range(16):state,_,event=c.step(state,inputs(),(('station',True),),0,False,tick)
        self.assertTrue(event['triggered']);self.assertTrue(event['cache_used']);self.assertEqual(fake.calls,1)
        current=inputs();current[0]['position']=(1,0)
        following,after,event=c.step(state,current,(('station',True),),0,False,16)
        self.assertTrue(event['cache_used']);self.assertEqual(after['A'],(2,0));self.assertEqual(fake.calls,1)
        self.assertTrue(state.cached_path);self.assertFalse(following.cached_path)
        for change in ('target','lock','station','position'):
            altered=inputs();altered[0]['position']=(1,0);station=(('station',True),)
            if change=='target':altered[0]['goal']=(1,0)
            if change=='lock':altered[0]['locked']=True
            if change=='station':station=(('station',False),)
            if change=='position':altered[0]['position']=(0,0)
            _,_,event=c.step(state,altered,station,0,False,16)
            self.assertTrue(event['cache_invalidated'],change)

    def test_failed_same_goal_and_locks_not_retried_until_changed(self):
        fake=_FakePlanner('timeout');c=RecoveryCoordinator(fake,_Stay());state=RecoverySnapshot()
        for tick in range(50):state,_,_=c.step(state,inputs(),(('station',True),),0,False,tick)
        self.assertEqual(fake.calls,1)
        new_task=inputs();new_task[0]['task_id']='S2'
        state,_,event=c.step(state,new_task,(('station',False),),0,False,50)
        self.assertFalse(event['triggered']);self.assertEqual(fake.calls,1)
        state,_,event=c.step(state,inputs(goal=(1,0)),(('station',True),),0,False,51)
        self.assertTrue(event['triggered']);self.assertEqual(fake.calls,2)


@unittest.skipUnless(LACAM and Path(LACAM).is_file(),'set WM_LACAM3_WORKER')
class OfficialLaCAMTests(unittest.TestCase):
    def test_frozen_clinic_states_all_have_valid_full_paths(self):
        data=frozen_states();t=data['topology']
        graph=DirectedGraph({tuple(x['position']):tuple(x['source_coordinates']) for x in t['coordinates']},
                            tuple((tuple(a),tuple(b)) for a,b in t['edges']))
        planner=LaCAM3Planner(Geometry(graph),LACAM)
        from business_transport.lacam3_coordinator import priority_for
        for state in data['states']:
            rs=state['robots'];positions={r['id']:tuple(r['position']) for r in rs}
            goals={r['id']:tuple(r['target']) for r in rs};locked={r['id'] for r in rs if r['locked']}
            result=planner.plan(positions,goals,locked,priority_for(rs))
            self.assertEqual(result.status,'solved');self.assertGreater(len(result.path),1)
            self.assertTrue(result.duplicate_business_goals);self.assertTrue(result.temporary_goals)

    def test_directed_graph_is_explicitly_unsupported(self):
        graph=DirectedGraph({(0,0):(0.,0.),(1,0):(1.,0.)},(((0,0),(1,0)),))
        result=LaCAM3Planner(Geometry(graph),LACAM).plan({'A':(0,0)},{'A':(1,0)},set(),'A')
        self.assertEqual(result.status,'unsupported_directed')

    def test_hard_500ms_timeout_and_locked_goal(self):
        graph=Geometry(Grid(3,2,frozenset()))
        with tempfile.TemporaryDirectory() as tmp:
            worker=Path(tmp)/'slow';worker.write_text('#!/bin/sh\nsleep 2\n');worker.chmod(0o755)
            planner=LaCAM3Planner(graph,worker)
            result=planner.plan({'A':(0,0)},{'A':(2,0)},set(),'A')
            self.assertEqual(result.status,'timeout');self.assertLess(result.wall_ms,1200)
        result=LaCAM3Planner(graph,LACAM).plan({'A':(0,0),'B':(2,0)},
                                                {'A':(2,0),'B':(2,0)},{'B'},'A')
        self.assertEqual(result.status,'locked_goal')

    def test_malformed_worker_and_fixed_temporary_goal_conflict_are_safe(self):
        graph=Geometry(Grid(3,2,frozenset()))
        with tempfile.TemporaryDirectory() as tmp:
            worker=Path(tmp)/'bad';worker.write_text('#!/bin/sh\necho "OK bad 1 2 3"\n');worker.chmod(0o755)
            result=LaCAM3Planner(graph,worker).plan({'A':(0,0)},{'A':(2,0)},set(),'A')
            self.assertEqual(result.status,'worker_error')
        result=LaCAM3Planner(graph,LACAM).plan({'A':(0,0),'B':(1,0)},
                                               {'A':(1,0),'B':(1,0)},set(),'A')
        self.assertEqual(result.status,'temporary_goal_conflict')
        self.assertFalse(result.worker_invoked)


@unittest.skipUnless(LACAM and PIBT and SOURCES and Path(LACAM).is_file() and Path(PIBT).is_file(),
                     'set WM_LACAM3_WORKER, WM_PIBT_LIBRARY and WM_HOSPITAL_SOURCES')
class ClinicIntegrationTests(unittest.TestCase):
    def test_clinic_caches_full_plan_and_prediction_branches_are_independent(self):
        bases,_=load_bases(SOURCES);business,_=pilot_case('clinic',bases['clinic']);geometry=Geometry(business.scenario.grid)
        observer=RecoveryObserver();policy=BusinessPolicy('nearest',geometry,business.rules,observer)
        sim=HybridBusinessSimulator(business,'world',61000,motion_backend='hybrid',pibt_library=PIBT,
                                    lacam3_worker=LACAM,collect_metrics=True)
        rows=[];result=sim.run(dispatch_policy=policy,on_observation=observer.observe,on_transition=rows.append)
        self.assertEqual(result['summary']['status'],'completed')
        event=next(e for e in sim.recovery_log if e['triggered'] and e['status']=='solved')
        self.assertGreater(event['plan_steps'],1)
        tick=event['tick']+1
        offline=RecoveryObserver();observed={}
        for row in rows:observed[row.observation.tick]=offline.observe(row.observation)
        initial=observed[tick];snapshot=initial.recovery_snapshot
        self.assertGreater(len(snapshot.cached_path),2)
        model=CapacityDynamics(RecoveryMovementModel(ProgressAwareRule(geometry),geometry,business.rules,PIBT,LACAM),business.rules)
        for _ in range(2):
            state=initial
            for step in range(tick,tick+3):
                state=model.predict_next(state,rows[step].assignments)
                self.assertEqual({r.id:r.position for r in state.robots},
                                 {r.id:r.position for r in rows[step].next_observation.robots})
            self.assertEqual(initial.recovery_snapshot,snapshot)

    def test_all_existing_models_preserve_cache_with_real_capacity_contention(self):
        from experiments.lacam3_recovery.model_alignment import run
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'alignment.json'
            run(Path(SOURCES),Path(PIBT),Path(LACAM),output)
            data=json.loads(output.read_text())
        self.assertEqual(data['factual_status'],'completed')
        self.assertTrue(data['actual_cache_invalidations'])
        for metrics in data['models'].values():
            self.assertEqual(metrics['one_step_positions_match'],metrics['factual_transitions'])
            self.assertTrue(all(metrics['cached_multistep_positions_match']))
            self.assertTrue(metrics['branch_isolation'])


if __name__=='__main__':unittest.main()
