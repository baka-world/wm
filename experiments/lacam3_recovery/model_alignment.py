"""Validate isolated LaCAM3 cache snapshots through existing world models."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import torch

from business_transport.dynamics import CapacityDynamics
from business_transport.lacam3_integration import HybridBusinessSimulator,RecoveryMovementModel,RecoveryObserver
from business_transport.models import CountingStructuredService
from business_transport.policy import BusinessPolicy
from experiments.pibt_motion.hospital_reproduction import load_bases,pilot_case
from experiments.pibt_motion.model_alignment import small_case
from experiments.service_value.counterfactual import load_pr9_model
from lab_transport.dynamics_model import Geometry
from lab_transport.paper_reuse.contracts import check_transition
from lab_transport.paper_reuse.pets_service import PETSRuleDynamics,ServiceDistribution
from lab_transport.torch_structured_adapter import ProgressAwareRule

CKPT=Path('examples/evidence/paper-reuse-integration-20261109/pets-service-cpu.pt')


def run(source_dir,pibt_library,lacam_worker,output):
    if output.exists():raise FileExistsError('refuse to overwrite model-alignment evidence')
    torch.set_num_threads(1);bases,_=load_bases(source_dir);business,_=pilot_case('clinic',bases['clinic'])
    geometry=Geometry(business.scenario.grid);observer=RecoveryObserver()
    policy=BusinessPolicy('nearest',geometry,business.rules,observer)
    sim=HybridBusinessSimulator(business,'world',61000,motion_backend='hybrid',pibt_library=pibt_library,
                                lacam3_worker=lacam_worker,collect_metrics=True)
    rows=[];result=sim.run(dispatch_policy=policy,on_observation=observer.observe,on_transition=rows.append)
    offline=RecoveryObserver();observations={row.observation.tick:offline.observe(row.observation) for row in rows}
    first=next(e for e in sim.recovery_log if e['triggered'] and e['status']=='solved')
    first_cache_tick=first['tick']+1
    network,info=load_pr9_model();metrics={};closed_loop={}
    for name,base in (
        ('rule',ProgressAwareRule(geometry)),
        ('pets',PETSRuleDynamics(geometry,ServiceDistribution.load(CKPT),20261109)),
        ('structured',CountingStructuredService(geometry,network,torch.device('cpu'),info['threshold']))):
        wrapper=RecoveryMovementModel(base,geometry,business.rules,pibt_library,lacam_worker)
        model=CapacityDynamics(wrapper,business.rules)
        exact=conditional=conditional_total=snapshot_equal=conditional_snapshot=0
        for row in rows:
            current=observations[row.observation.tick]
            model.reset_seed(20261109+current.tick)
            predicted=model.predict_next(current,row.assignments)
            check_transition(current,predicted,geometry)
            same_positions={r.id:r.position for r in predicted.robots}=={r.id:r.position for r in row.next_observation.robots}
            exact+=same_positions
            known={s.id for s in current.samples}
            actual={s.id:s.status for s in row.next_observation.samples if s.id in known}
            pred={s.id:s.status for s in predicted.samples if s.id in known}
            same_snapshot=predicted.recovery_snapshot==row.next_observation.recovery_snapshot
            if actual==pred:
                conditional_total+=1;conditional+=same_positions;conditional_snapshot+=same_snapshot
            snapshot_equal+=same_snapshot
        state=observations[first_cache_tick];original=state.recovery_snapshot
        multi=[]
        for tick in range(first_cache_tick,first_cache_tick+4):
            state=model.predict_next(state,rows[tick].assignments)
            multi.append({r.id:r.position for r in state.robots}==
                         {r.id:r.position for r in rows[tick].next_observation.robots})
        # A second branch starts from the immutable original snapshot, not
        # from the first branch's advanced plan or the live simulator.
        branch=model.predict_next(observations[first_cache_tick],rows[first_cache_tick].assignments)
        isolated=(observations[first_cache_tick].recovery_snapshot==original and
                  branch.recovery_snapshot!=original and
                  sim.recovery_snapshot!=branch.recovery_snapshot)
        metrics[name]={'factual_transitions':len(rows),'one_step_positions_match':exact,
            'matching_service_state_transitions':conditional_total,'conditional_motion_match':conditional,
            'snapshot_match':snapshot_equal,'conditional_snapshot_match':conditional_snapshot,
            'cached_multistep_start_tick':first_cache_tick,
            'cached_multistep_positions_match':multi,'branch_isolation':isolated,
            'pibt_prediction_calls':wrapper.pibt.solver_calls,
            'pets_forward_calls':base.distribution.forward_calls if name=='pets' else 0,
            'structured_service_calls':base.service_calls if name=='structured' else 0}
        mini=small_case();mini_geo=Geometry(mini.scenario.grid);mini_observer=RecoveryObserver()
        if name=='rule':mini_base=ProgressAwareRule(mini_geo)
        elif name=='pets':mini_base=PETSRuleDynamics(mini_geo,ServiceDistribution.load(CKPT),20261109)
        else:mini_base=CountingStructuredService(mini_geo,network,torch.device('cpu'),info['threshold'])
        mini_wrapper=RecoveryMovementModel(mini_base,mini_geo,mini.rules,pibt_library,lacam_worker)
        mini_policy=BusinessPolicy('cpsat_'+name,mini_geo,mini.rules,mini_observer,mini_wrapper,seed=20261109)
        mini_sim=HybridBusinessSimulator(mini,'world',1,motion_backend='hybrid',pibt_library=pibt_library,
                                        lacam3_worker=lacam_worker,collect_metrics=True)
        mini_result=mini_sim.run(dispatch_policy=mini_policy,on_observation=mini_observer.observe)
        closed_loop[name]={'status':mini_result['summary']['status'],'stages_completed':mini_result['summary']['completed_count'],
            'stages':mini_result['summary']['sample_count'],'prediction_pibt_calls':mini_wrapper.pibt.solver_calls,
            'execution_pibt_calls':mini_sim.pibt.solver_calls,'model_fallbacks':mini_policy.fallbacks,
            'pets_forward_calls':mini_base.distribution.forward_calls if name=='pets' else 0,
            'structured_service_calls':mini_base.service_calls if name=='structured' else 0}
    invalidations=[{'tick':e['tick'],'reason':e['invalidation_reason']} for e in sim.recovery_log if e['cache_invalidated']]
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({'scope':'factual clinic snapshots; cached-path multi-step check has no unseen future events',
        'factual_status':result['summary']['status'],'first_recovery_tick':first['tick'],
        'actual_cache_invalidations':invalidations,'models':metrics,'cp_sat_closed_loop':closed_loop},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'factual_status':result['summary']['status'],'first_recovery_tick':first['tick'],
                      'actual_cache_invalidations':invalidations,'models':metrics,'cp_sat_closed_loop':closed_loop},ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-dir',type=Path,required=True)
    p.add_argument('--pibt-library',type=Path,required=True);p.add_argument('--lacam-worker',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    run(a.source_dir,a.pibt_library,a.lacam_worker,a.output)


if __name__=='__main__':main()
