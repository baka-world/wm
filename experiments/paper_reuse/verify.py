"""Check acceptance invariants without requiring any method to win."""
from __future__ import annotations
import hashlib
import gzip
import json
from pathlib import Path
import sys
import torch


def main(root):
    root=Path(root)
    config=json.loads((root/'scenarios.json').read_text(encoding='utf-8'))
    assert config['schedule_audit']['cross_group_duplicates']==0 and len(config['cases'])==32
    for profile in ('development','confirmation'):
        data=json.loads((root/profile/'raw.json').read_text(encoding='utf-8'))
        assert data['profile']==profile and len(data['cases'])==16
        assert data['schedule_audit']['cross_group_duplicates']==0
        assert {(c['layout'],c['seed'],c['regime']) for c in data['cases']} == {
            (c['layout'],c['seed'],c['regime']) for c in config['cases'] if c['profile']==profile}
        assert set(data['summary'])=={'nearest_rule','cpsat_rule','cpsat_pets'}
        for case in data['cases']:
            for name,m in case['methods'].items():
                assert m['samples']>=m['completed']>=0
                assert m['duplicate_assignments']==0 and m['discrete_safety_violations']==0
                assert m['prediction_fallbacks']>=0 and m['solver_fallbacks']>=0
                assert m['solver_partial_truncations']==0
                assert sum(x['neural_model_forward_calls'] for x in m['decisions'])==m['neural_model_forward_calls']
                assert sum(x['rollout_steps'] for x in m['decisions'])==m['rollout_steps']
                assert m['neural_transition_calls']+m['neural_scoring_calls']==m['neural_model_forward_calls']
                if name=='cpsat_pets':assert m['neural_model_forward_calls']>0
            for h in ('1','3','5'):
                models=case['state_prediction'][h]['models']
                assert all(m['correct']<=m['windows'] for m in models.values())
        assert data['summary']['cpsat_pets']['neural_transition_calls']>0
        assert (root/profile/'analysis.json').is_file()
    assert torch.version.cuda is None
    original=Path('examples/evidence/paper-reuse-integration-20261109/pets-service-cpu.pt')
    model=torch.load(original,map_location='cpu',weights_only=True)
    dataset=Path('examples/evidence/paper-reuse-integration-20261109/existing-service-labels.json.gz')
    assert model['metadata']['dataset_sha256']==hashlib.sha256(dataset.read_bytes()).hexdigest()
    source=Path('examples/evidence/service-dynamics-20261109/dataset-manifest.json')
    assert model['metadata']['source_manifest_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
    with gzip.open(dataset,'rt',encoding='utf-8') as handle:
        labels=json.load(handle)
    assert {r['split'] for r in labels['examples']} == {'train','validation'}
    assert all(r['duration_active_ticks']>0 for r in labels['examples'])
    retrained=torch.load(root/'retrained-cpu.pt',map_location='cpu',weights_only=True)
    assert retrained['metadata']['dataset_sha256']==model['metadata']['dataset_sha256']
    print('Verified 32 frozen scenarios, legal assignments/states, actual PETS calls, '
          'recorded fallbacks, checkpoint provenance and CPU-only execution')


if __name__=='__main__':main(sys.argv[1])
