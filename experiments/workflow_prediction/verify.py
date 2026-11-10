"""Check experiment invariants without requiring a performance improvement."""
import json
from pathlib import Path
import sys

from experiments.workflow_prediction.config_audit import audit


def contains_hidden(value):
    if isinstance(value,dict):
        return any(k in ('service_remaining','future_fault_schedule','future_job_table') or contains_hidden(v)
                   for k,v in value.items())
    if isinstance(value,list):return any(contains_hidden(x) for x in value)
    return False


def verify(root,profile):
    proof=audit();root=Path(root)
    data=json.loads((root/'raw.json').read_text());summary=data['summary']
    assert data['profile']==profile and not data['limited_pilot']
    assert data['config_sha256']==proof['config_sha256']
    assert len(data['cases'])=={'development':6,'confirmation':8}[profile]
    assert summary['actual_successors']>0 and summary['old_missing_successors']==summary['actual_successors']
    for model in ('rule','pets','structured'):
        assert summary['old_model_successor_audit'][model]['missed']==summary['actual_successors']
    for method,metrics in summary['methods'].items():
        assert metrics['jobs']>0 and 0<=metrics['jobs_completed']<=metrics['jobs']
        for field in ('resource_violations','safety_violations','invalid_edge_violations',
                      'locked_move_violations','lifecycle_violations','duplicate_assignments','model_fallbacks'):
            assert metrics[field]==0,(method,field,metrics[field])
        if method!='nearest':assert metrics['model_prediction_calls']>0
    assert summary['methods']['workflow_pets']['pets_forward_calls']>0
    assert summary['methods']['workflow_structured']['structured_service_calls']>0
    for h in ('1','3','5'):
        values=summary['state_prediction'][h]
        assert values['coverage'].get('eligible',0)>=values['coverage'].get('selected',0)>0
        for name,metrics in values['dynamic'].items():
            assert metrics['windows']>0,(h,name)
            assert metrics.get('conditional_release_logic_errors',0)==0,(h,name,'release')
            assert metrics.get('conditional_resource_errors',0)==0,(h,name,'resource')
            assert metrics.get('conditional_service_matching_steps',0)>0
            for kind in ('legacy','workflow'):
                assert values['fixed_entity'][name][kind]['total']>=0
    assert summary['same_state_decisions']>0
    for case in data['cases']:
        assert not contains_hidden(case['factual_visible_transitions'])
        for decision in case['same_state']:
            assert not contains_hidden(decision['observable_state'])
            assert 1<=len(decision['plans'])<=4
            legal={(a['robot_id'],a['sample_id']) for a in decision['candidate_pairs']}
            for plan in decision['plans']:
                assert len({a['robot_id'] for a in plan})==len(plan)
                assert len({a['sample_id'] for a in plan})==len(plan)
                assert all((a['robot_id'],a['sample_id']) in legal for a in plan)
            assert len(decision['branches'])==len(decision['plans'])
            for label,forecast in decision['forecasts'].items():
                assert sorted(forecast['ranking'])==list(range(len(decision['plans'])),label
                assert forecast['choice']==forecast['ranking'][0]
    print(f'{profile}: validated {len(data["cases"])} complete scenarios, dynamic workflow invariants, '
          'observability, fixed candidate replay and actual model calls')


if __name__=='__main__':verify(sys.argv[1],sys.argv[2])
