"""Contract-only three independent cohort comparison, zero simulator calls."""
from __future__ import annotations

from unittest import TestCase,main

from .test_replication import ReplicationSourceContractTests
from .compare_three import run,SCHEMA38,NAMES38


def sample38():
    cases=[]
    for regime in ("light","medium","heavy","shift"):
        for i in range(6):
            arms={}
            for key,name in NAMES38.items():
                m={"cp":11,"rule":10,"v4":9,"scratch":10}[key]
                arms[name]={
                    "makespan_ticks":m,
                    "job_metrics":{"late_jobs":0,"censored":False},
                    "resource_wait_ticks":0,"safety_violations":0,
                    "full_episode_cpu_seconds":.2,
                    "strict_budget_audit":{"model_changes_after_cpu_cap":0},
                    "decision_records":[{
                        "tick":0,"selected_index":1,"process_cpu_seconds":.3}],
                }
            cases.append({"group":f"38-{regime}-{i}","regime":regime,
                         "seed":38000000+len(cases),"arms":arms})
    return {"schema":SCHEMA38,"groups":cases,
            "summary":{"independent_groups":24}}


class ThreeCohortSchemaTests(TestCase):
    def test_source38_joins_two_sources_without_replaying(self):
        fixture=ReplicationSourceContractTests()
        result=run(fixture.synthetic(36),fixture.synthetic(37),sample38())
        self.assertEqual(result["summary"]["groups"],72)
        self.assertEqual(result["summary"]["late_model_reranking_accepted"],0)
        for v in result["summary"]["by_regime"].values():
            self.assertEqual(v["groups"],18)
            self.assertEqual(v["paired"]["v4_vs_cp"]["wins"],18)
            self.assertEqual(v["paired"]["v4_vs_cp"]["mean_paired_delta_tick"],-2)

    def test_third_source_late_applied_action_is_rejected(self):
        fixture=ReplicationSourceContractTests()
        bad=sample38()
        decision=bad["groups"][0]["arms"]["original_v4_h20"]
        decision["decision_records"][0]["process_cpu_seconds"]=.7
        with self.assertRaisesRegex(AssertionError,"accepted late"):
            run(fixture.synthetic(36),fixture.synthetic(37),bad)


if __name__=="__main__":
    main()
