"""No further simulator runs needed for cross-cohort schema tests."""
from __future__ import annotations
from unittest import TestCase,main

from experiments.world_model_integration.h20_replication.compare import (
    run,ARMS36,ARMS37,SOURCE36,SOURCE37,
)


class ReplicationSourceContractTests(TestCase):
    def synthetic(self,source):
        is36=source==36
        names=ARMS36 if is36 else ARMS37
        groups=[]
        for regime in ("light","medium","heavy","shift"):
            for i in range(6):
                arms={}
                for name in names.values():
                    score=11 if name==names["cp"] else (9 if name==names["v4"] else 10)
                    facts={
                        "makespan_ticks":score,
                        "job_metrics":{"censored":False,"late_jobs":0},
                        "resource_wait_ticks":1,
                        "safety_violations":0,
                    }
                    if is36:
                        facts["whole_episode_cpu_seconds"]=0.2
                        facts["decision_trace"]=[{
                            "tick":0,"selected_index":1,"cpu_seconds":0.1}]
                    else:
                        facts["full_episode_cpu_seconds"]=0.2
                        facts["decision_records"]=[{
                            "tick":0,"selected_index":1,"process_cpu_seconds":0.1}]
                    arms[name]=facts
                groups.append({
                    "group":f"{source}_{regime}_{i}",
                    "regime":regime,
                    "seed":source*1000000+len(groups),
                    "arms":arms,
                })
        return {
            "protocol":SOURCE36 if is36 else SOURCE37,
            "groups":groups,
            "summary":{"independent_groups":24}
            if is36 else {"total_independent_groups":24},
        }

    def test_48_groups_one_regime_has_12_pairs(self):
        result=run(self.synthetic(36),self.synthetic(37))
        self.assertEqual(result["summary"]["groups"],48)
        self.assertFalse(result["summary"]["late_applied_nonfirst"])
        for cohort in result["summary"]["per_regime"].values():
            self.assertEqual(cohort["groups"],12)
            diff=cohort["comparisons"]["v4_vs_cp"]
            self.assertEqual(diff["mean_paired_delta_tick"],-2)
            self.assertEqual(diff["wins"],12)
            self.assertEqual(diff["losses"],0)

    def test_repeated_scene_seed_is_rejected(self):
        a=self.synthetic(36)
        b=self.synthetic(37)
        b["groups"][0]["seed"]=a["groups"][0]["seed"]
        with self.assertRaisesRegex(AssertionError,"collision"):
            run(a,b)


if __name__=="__main__":
    main()
