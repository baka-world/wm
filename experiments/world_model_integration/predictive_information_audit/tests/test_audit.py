"""Regression tests using ONLY existing frozen JSON/JSON.gz, no simulation import."""
import ast
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
try:
    from experiments.world_model_integration.predictive_information_audit import audit
except ModuleNotFoundError:
    import audit

SOURCE=Path(audit.__file__).resolve().parent.parent/'urgent_workflow_feasibility'/'evidence'
if not SOURCE.exists(): SOURCE=Path(audit.__file__).resolve().parent/'results'


class OfflineAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=audit.read_archive(SOURCE)

    def test_sha_reproduction_of_frozen_evidence(self):
        for f,sha in audit.SOURCE_SHA.items():
            self.assertEqual(audit.sha_file(SOURCE/f),sha)
        self.assertEqual(len(self.data['cases']),96)
        self.assertEqual(len(self.data['seeds']),24)

    def test_strict_identical_p2_p3_state_and_candidates(self):
        score=audit.audit_scores(self.data)
        self.assertEqual(score['counts']['paired_scenarios'],24)
        self.assertEqual(score['counts']['all_strict_paired_calls'],389)
        self.assertEqual(score['counts']['multi_candidate_calls'],330)
        self.assertEqual(score['mask_metrics'].get('top_choice_change',0),0)

    def test_score_classes_partition_and_tiebreak(self):
        result=audit.audit_scores(self.data)
        self.assertEqual({k:result['counts'].get(k,0) for k in ['C0','C1','C2','C3','C4','C5']},
                         {'C0':37,'C1':168,'C2':29,'C3':94,'C4':2,'C5':0})
        self.assertEqual(result['mask_metrics']['numeric_score_changed'],198)
        self.assertEqual(result['mask_metrics']['numeric_changed_but_first_layer_unique'],102)
        self.assertEqual({(d['seed'],d['tick']) for d in result['state_details'] if d['class']=='C4'},
                         {(903220,78),(903439,89)})

    def test_mismatch_cannot_be_paired_as_same_state(self):
        data={'seeds':list(self.data['seeds']),'cases':dict(self.data['cases'])}
        key=(903220,'P3');changed=copy.deepcopy(data['cases'][key]);changed['history_rows'][0]['robots'][0]['position']=[99,99]
        data['cases'][key]=changed
        with self.assertRaises(AssertionError):audit.audit_scores(data)

    def test_urgent_transitions_are_derived_per_job(self):
        jobs=audit.audit_jobs(self.data)
        p0=jobs['by_arm']['P0']['counts'];p1=jobs['by_arm']['P1']['counts']
        self.assertEqual((p0['on_time'],p0['late_complete'],p0['censored_overdue']),(22,12,14))
        self.assertEqual((p1['on_time'],p1['late_complete'],p1['censored_overdue']),(22,23,3))
        m=jobs['P0_to_other_job_level_transitions']['P1']['matrix']
        self.assertEqual(m['censored_overdue => late_complete'],11)
        self.assertFalse(any('=> on_time' in k and not k.startswith('on_time') for k in m))

    def test_retest_cannot_be_observed_before_detection(self):
        x=audit.audit_retest(self.data)
        self.assertTrue(x['same_job_same_seed_truth_on_observed_decisions'])
        self.assertEqual(x['per_arm']['P0']['counts']['decision_before_detection'],0)
        self.assertEqual(x['pre_retest_signal_verdict'],
                         'PRE_RETEST_PREDICTION_NOT_IDENTIFIABLE_FROM_CURRENT_OBSERVATION')

    def test_frozen_oracle_selection_and_public_tick_release(self):
        r=audit.audit_oracle(self.data)
        self.assertEqual([x['id'] for x in r],['903074:10','903293:10','903512:13'])
        for row in r:
            self.assertEqual(row['current']['urgent_count']+row['current']['routine_count'],
                             row['current']['waiting_count'])
            self.assertEqual(row['factual_index'],0)
            self.assertNotEqual(row['best_index'],row['factual_index'])
            self.assertEqual(row['factual_vs_best']['capacity_wait_delta'],0)
            self.assertTrue(all(not c['event_history_available'] for c in row['candidates']))

    def test_local_runner_cannot_import_or_execute_simulator(self):
        source=Path(audit.__file__).read_text(encoding='utf8')
        parsed=ast.parse(source)
        roots=set()
        for node in ast.walk(parsed):
            if isinstance(node,ast.Import):roots.update(alias.name.split('.')[0] for alias in node.names)
            if isinstance(node,ast.ImportFrom) and node.module:roots.add(node.module.split('.')[0])
        self.assertFalse(roots & {'lab_transport','business_transport','torch','ortools','numpy','scipy'})

    def test_output_contains_integrity_manifest(self):
        with tempfile.TemporaryDirectory() as t:
            out=Path(t)/'results';summary=audit.build(SOURCE,out)
            self.assertEqual(summary['new_trajectories'],0)
            m=json.loads((out/'sha256.json').read_text())
            for filename,sha in m.items():self.assertEqual(audit.sha_file(out/filename),sha)

if __name__=='__main__': unittest.main()
