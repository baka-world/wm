"""Unit tests for outcome ordering, censored regret and cluster-safe summaries."""
import unittest

from experiments.oracle_value.analysis import (
    analyze, comparable_regret, pairwise_alignment, select_oracle
)


def outcome(flow, completed=6, jobs=6):
    return {"restricted_mean_flow":flow,"jobs_completed":completed,"jobs":jobs}


def state(flows, rule=0, perfect=1):
    branches=[outcome(x) for x in flows]
    methods={}
    scores=[(x,0,0) for x in flows]
    for label,choice in (("Rule-5",rule),("PETS-5",rule),
                         ("Structured-5",rule),("Perfect-State-5",perfect),
                         ("Perfect-State-20",perfect),
                         ("Full-outcome Oracle",select_oracle(branches))):
        methods[label]={"choice":choice,"scores":scores if label!="Full-outcome Oracle" else None}
    return {"tick":0,"plans":[[{"robot_id":"R1","sample_id":str(i)}] for i in range(len(flows))],
            "branches":branches,"methods":methods}


class TestOracleValue(unittest.TestCase):
    def test_lexicographic_complete_first(self):
        rows=[outcome(50,5),outcome(100,6),outcome(110,6)]
        self.assertEqual(select_oracle(rows),1)
        self.assertIsNone(comparable_regret(rows,2,1))

    def test_completed_regret(self):
        rows=[outcome(9),outcome(7),outcome(12)]
        self.assertEqual(select_oracle(rows),1)
        self.assertEqual(comparable_regret(rows,2,1),5)

    def test_tie_and_fixed_index(self):
        self.assertEqual(select_oracle([outcome(5),outcome(5)]),0)
        pairs=pairwise_alignment([(1,0,0),(1,0,0)],[outcome(6),outcome(4)])
        self.assertEqual(pairs["score_ties"],1)

    def test_cluster_weighting_not_pseudo_replication(self):
        cases=[
            {"layout":"empty8","seed":1,"regime":"a",
             "decisions":[state([10,0],rule=0,perfect=1),state([10,0],rule=0,perfect=1)]},
            {"layout":"empty8","seed":1,"regime":"b",
             "decisions":[state([10,0],rule=0,perfect=1)]},
            {"layout":"random32","seed":2,"regime":"a",
             "decisions":[state([2,0],rule=0,perfect=1)]},
        ]
        summary=analyze(cases)
        self.assertEqual(summary["clusters"],2)
        self.assertEqual(summary["actual_decisions"],4)
        self.assertEqual(summary["methods"]["Rule-5"]["regret"]["mean_of_cluster_means"],6)
        self.assertEqual(summary["methods"]["Perfect-State-5"]["regret"]["mean_of_cluster_means"],0)
        self.assertEqual(summary["candidate_gaps"]["gap_ge_3"],3)

    def test_failed_replay_kept_but_not_scored(self):
        x=state([4,2])
        x["branches"][1]={"error":"invalid replay"}
        result=analyze([{"layout":"empty8","seed":1,"regime":"a","decisions":[x]}])
        self.assertEqual(result["failed_decisions"],1)
        self.assertEqual(result["actual_decisions"],1)
        self.assertIsNone(result["finite_candidate_rule_opportunity"]["mean_of_cluster_means"])


if __name__=="__main__":
    unittest.main()
