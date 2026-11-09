"""Check independent replay and policy contracts without requiring favorable results."""
import json
from pathlib import Path
import sys


def check_counterfactual(root,profile):
    raw = json.loads((root/'raw.json').read_text(encoding='utf-8'))
    summary = json.loads((root/'summary.json').read_text(encoding='utf-8'))
    assert raw['profile'] == profile and raw['task_schedule_audit']['cross_group_duplicates'] == 0
    assert raw['candidate_budget'] == 8 and list(raw['horizons']) == [3,5,10,20]
    assert len(raw['cases']) == 16 and summary['comparable_decisions'] > 0
    assert {c['regime'] for c in raw['cases']} == {'normal','burst','outage_processing','outage_arrival'}
    count = 0
    for case in raw['cases']:
        for d in case['decisions']:
            assert len(d['candidates']) == min(d['candidate_total'],8)
            assert len({(c['action']['robot_id'],c['action']['sample_id']) for c in d['candidates']}) == len(d['candidates'])
            if not d['comparable']:
                continue
            count += 1
            flows = [c['actual_mean_flow'] for c in d['candidates']]
            assert all(c['actual_discrete_safety_violations'] == 0 for c in d['candidates'])
            for h in ('3','5','10','20'):
                for name in ('rule','hybrid','service_oracle','actual_short_state'):
                    record = d['horizons'][h]
                    for kind in (record,record['workload']):
                        i = kind['choices'][name]
                        assert 0 <= i < len(flows)
                        assert abs(kind['regret'][name] - (flows[i]-min(flows))) < 1e-9
    assert count == summary['comparable_decisions']
    audit_path = root/'oracle-audit.json'
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding='utf-8'))
        assert audit['profile'] == profile
        keys = [(c['layout'],c['seed'],c['regime'],d['tick'],
                 item['action']['robot_id'],item['action']['sample_id'])
                for c in raw['cases'] for d in c['decisions'] for item in d['candidates']]
        observed = [(c['layout'],c['seed'],c['regime'],c['decision_tick'],
                     c['candidate']['robot_id'],c['candidate']['sample_id']) for c in audit['candidates']]
        assert keys == observed
        assert all(v['active_ticks'] >= 1 for c in audit['candidates'] for v in c['privileged_active_service_durations'])
    return raw


def check_online(root,profile):
    raw = json.loads((root/'raw.json').read_text(encoding='utf-8'))
    summary = json.loads((root/'summary.json').read_text(encoding='utf-8'))
    assert raw['profile'] == profile and raw['horizon'] == 20 and raw['budget'] == 8
    assert len(raw['cases']) == 16
    names = {'rule_original','rule_workload','hybrid_original','hybrid_workload'}
    assert set(summary['all']) == names
    for case in raw['cases']:
        assert set(case['methods']) == names
        for method in case['methods'].values():
            assert method['discrete_safety_violations'] == 0 and method['fallbacks'] == 0
            assert len(method['decision_cpu_samples_ms']) == method['decision_count']
            assert all(len(d['evaluated']) <= 8 for d in method['decisions'])
    for method in summary['all'].values():
        assert 0 <= method['task_completion_rate'] <= 1
        assert method['pooled_decision_cpu_p95_ms'] is not None
    paired = json.loads((root/'paired.json').read_text(encoding='utf-8'))
    assert paired['profile'] == profile
    assert set(paired['paired']['all']) == {'rule_new_minus_rule_old','hybrid_new_minus_hybrid_old',
                                            'hybrid_old_minus_rule_old','hybrid_new_minus_rule_new'}
    return raw


def main(root):
    config = json.loads((root/'scenario-config.json').read_text(encoding='utf-8'))
    assert len(config['cases']) == 32 and config['schedule_audit']['cross_group_duplicates'] == 0
    for profile in ('development','confirmation'):
        ordinary = check_counterfactual(root/('counterfactual-'+profile),profile)
        assert (root/('counterfactual-'+profile)/'oracle-audit.json').is_file()
        aligned = check_counterfactual(root/('aligned-'+profile),profile)
        check_online(root/('online-'+profile),profile)
        old_cases = ordinary['cases']
        new_cases = aligned['cases']
        assert len(old_cases) == len(new_cases)
        for a,b in zip(old_cases,new_cases):
            assert (a['layout'],a['seed'],a['regime']) == (b['layout'],b['seed'],b['regime'])
            for x,y in zip(a['decisions'],b['decisions']):
                assert x['tick'] == y['tick']
                assert [(c['action'],c['actual_mean_flow']) for c in x['candidates']] == [
                    (c['action'],c['actual_mean_flow']) for c in y['candidates']]
    print('Validated split audit, branch-specific replay, 4 horizons, scoring, continuation sensitivity, and online boundaries')


if __name__ == '__main__':
    main(Path(sys.argv[1]))
