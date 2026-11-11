import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from archive_analysis import analyze


class ArchiveAuditTests(unittest.TestCase):
    def test_rejects_unfrozen_archive(self):
        with self.assertRaises(ValueError):
            analyze({'gate_pass': False})

    def test_real_artifact_if_present(self):
        directory=Path(__file__).resolve().parents[2]/'v4_evidence'
        if not (directory/'v4-dispatch.json').exists():
            self.skipTest('frozen GitHub Actions ZIP not extracted adjacent to package')
        dispatch=json.loads((directory/'v4-dispatch.json').read_text())
        data=json.loads((directory/'v4-independent-in-data.json').read_text())
        out=analyze(dispatch,data)
        self.assertEqual(out['summary']['fixed_states'],3)
        self.assertEqual(out['summary']['candidates_total'],9)
        self.assertEqual(out['summary']['v4_vs_rule_score_changed_candidates'],0)
        self.assertEqual(out['summary']['v3c_vs_rule_score_changed_candidates'],1)
        self.assertEqual(out['summary']['rule_and_v4_ex_post_regret_by_state'],[5,0,6.75])


if __name__=='__main__':unittest.main()
