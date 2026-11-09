"""Validity checks; no model is required to beat a baseline."""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab_transport.service_evaluation import MODELS
from lab_transport.service_learning import contract_hashes
from lab_transport.training_data import validate_dataset


def main(root):
    dataset, training, evaluation = root/'dataset', root/'training', root/'evaluation'
    validated = validate_dataset(dataset)
    assert validated['episodes'] == 80
    manifest = json.loads((dataset/'manifest.json').read_text())
    coverage = json.loads((dataset/'coverage.json').read_text())
    for split in ('train','validation','test_tartu','test_clinic'):
        faults = [c for c in coverage if c['split'] == split and c['regime'] == 'outage_processing']
        assert faults and all(c['service_wait_ticks'] > 0 and c['pending_at_recovery'] > 0 for c in faults)
        assert all(c['service_wait_ticks'] == 0 for c in coverage if c['split'] == split and c['regime'] == 'normal')
    assert manifest['schedule_audit']['cross_split_task_duplicate_groups'] == 0
    report = json.loads((training/'training.json').read_text())
    assert report['contract_code_sha256'] == contract_hashes()
    assert report['training_service_positive'] > 0 and report['training_service_negative'] > 0
    assert report['train_sequence_windows']['positive'] > 0 and report['train_sequence_windows']['negative'] > 0
    for name in ('original','weighted','sequence'):
        assert hashlib.sha256((training/f'{name}.pt').read_bytes()).hexdigest() == report['variants'][name]['checkpoint_sha256']
    summary = json.loads((evaluation/'summary.json').read_text())
    assert summary['horizon'] == 3 and summary['candidate_budget'] == 8
    assert summary['same_state_decisions'] > 0
    for split in ('test_tartu/all', 'test_clinic/all'):
        for model in MODELS:
            metrics = summary['factual_prediction'][split]['models'][model]
            assert metrics['service_event_counts']['actual'] > 0
            assert all(metrics['multi_step'][h]['coverage']['evaluated'] > 0 for h in ('3','5'))
    for model in MODELS:
        item = summary['closed_loop']['all'][model]
        assert item['episodes'] == 16 and item['safety_violations'] == 0 and item['fallbacks'] == 0
    same_state = json.loads((evaluation/'same-state.json').read_text())
    assert len(same_state) == summary['same_state_decisions']
    for decision in same_state:
        expected = [(a['robot_id'], a['sample_id']) for a in decision['candidates'][:8]]
        for model in MODELS:
            chosen = decision['selected'][model]
            records = chosen['decision']['candidates_evaluated']
            assert [(a['robot_id'], a['sample_id']) for a in records] == expected
            assert chosen['fallbacks'] == 0 and chosen['model_steps'] == 3*len(expected)
    print('Validated 80 factual episodes, protected checkpoints, paired prediction, 16 closed loops per model, and common candidates')


if __name__ == '__main__':
    main(Path(sys.argv[1]))
