"""Check shared PIBT motion against execution and existing-model rollouts."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import torch

from business_transport.dynamics import CapacityDynamics
from business_transport.models import CountingStructuredService
from business_transport.pibt_integration import CoordinatedBusinessSimulator,MotionObserver,PIBTMovementModel
from business_transport.pibt_motion import PIBTMotion
from business_transport.policy import BusinessPolicy
from business_transport.scenario import compile_scenario
from business_transport.simulation import resource_audit
from experiments.business_constraints.benchmark import make_case,read_config
from experiments.service_value.counterfactual import load_pr9_model
from lab_transport.dynamics_model import Geometry
from lab_transport.multisite_experiment import discrete_safety_violations
from lab_transport.paper_reuse.contracts import check_transition
from lab_transport.paper_reuse.pets_service import PETSRuleDynamics,ServiceDistribution
from lab_transport.torch_structured_adapter import ProgressAwareRule

CKPT=Path('examples/evidence/paper-reuse-integration-20261109/pets-service-cpu.pt')
SEED=20261109


def small_case():
    prior,config=make_case(read_config(),'empty8',67000,'workflow_serial')
    config={**config,'jobs':config['jobs'][:2]}
    return compile_scenario(replace(prior.scenario,samples=()),config)


def make_model(name,geometry,network,threshold):
    if name=='rule':return ProgressAwareRule(geometry)
    if name=='pets':return PETSRuleDynamics(geometry,ServiceDistribution.load(CKPT),SEED)
    if name=='structured':return CountingStructuredService(geometry,network,torch.device('cpu'),threshold)
    raise ValueError('unknown model')


def run(library,output):
    if output.exists():raise FileExistsError('refuse to overwrite alignment result')
    torch.set_num_threads(1);business=small_case();geometry=Geometry(business.scenario.grid)
    network,info=load_pr9_model();threshold=info['threshold']
    observer=MotionObserver();policy=BusinessPolicy('nearest',geometry,business.rules,observer)
    simulation=CoordinatedBusinessSimulator(business,'world',1,motion_backend='pibt',pibt_library=library,
        motion_pauses={'R1':((2,6),)},collect_metrics=True)
    rows=[];factual=simulation.run(dispatch_policy=policy,on_observation=observer.observe,on_transition=rows.append)
    parity={};closed={}
    for name in ('rule','pets','structured'):
        model=make_model(name,geometry,network,threshold);bridge=PIBTMotion(geometry,library)
        predictor=CapacityDynamics(PIBTMovementModel(model,geometry,bridge),business.rules)
        offline=MotionObserver();matches=0;observed_pause=0
        for row in rows:
            current=offline.observe(row.observation)
            predicted=predictor.predict_next(current,row.assignments)
            check_transition(current,predicted,geometry)
            matches+=({r.id:r.position for r in predicted.robots}==
                      {r.id:r.position for r in row.next_observation.robots})
            observed_pause+=sum(r.operational_available is False for r in current.robots)
        parity[name]={'matching_one_step_robot_configurations':matches,'factual_transitions':len(rows),
                      'current_paused_robot_observations':observed_pause,'native_prediction_calls':bridge.solver_calls}
        model=make_model(name,geometry,network,threshold);bridge=PIBTMotion(geometry,library)
        observer=MotionObserver()
        chooser=BusinessPolicy('cpsat_'+name,geometry,business.rules,observer,
                                PIBTMovementModel(model,geometry,bridge),seed=SEED)
        sim=CoordinatedBusinessSimulator(business,'world',1,motion_backend='pibt',pibt_library=library,
                                         planner_seed=SEED,collect_metrics=True)
        result=sim.run(dispatch_policy=chooser,on_observation=observer.observe)
        closed[name]={'status':result['summary']['status'],'stages_completed':result['summary']['completed_count'],
            'stages':result['summary']['sample_count'],'execution_native_calls':sim.pibt.solver_calls,
            'prediction_native_calls':bridge.solver_calls,'model_fallbacks':chooser.fallbacks,
            'pets_forward_calls':model.distribution.forward_calls if name=='pets' else 0,
            'structured_service_calls':model.service_calls if name=='structured' else 0,
            'capacity_violations':len(resource_audit(result,business.rules)['violations']),
            'discrete_safety_violations':discrete_safety_violations(result)}
    data={'scope':'small synthetic functionality check, no training; current pause visible, future changes unknown',
          'factual_status':factual['summary']['status'],'factual_execution_native_calls':simulation.pibt.solver_calls,
          'parity':parity,'closed_loop':closed}
    output.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(data,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--library',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.library,a.output)


if __name__=='__main__':main()
