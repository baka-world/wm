"""Deterministic hard evidence for after-last-candidate fallback.

No simulator cost, weights or probability training. The fake CPU clock
reproduces exactly the bug found in PR33 before a new research run.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest import TestCase,main
from unittest.mock import patch

from experiments.world_model_integration.online_horizon import pr30_run


class FixedClockForecast:
    def __init__(self,clock,cost):
        self.clock=clock
        self.cost=cost

    def rollout(self,state,action,**kwargs):
        self.clock[0]+=self.cost[action]
        # Second plan has higher forecast value, so the old bug selects it
        # even when it consumed more than 0.5 seconds before return.
        score=1 if action==("first",) else 9
        return SimpleNamespace(distributions=[
            SimpleNamespace(expected_completed_samples=score)])


def fake_policy(clock,cost,budget=0.5):
    p=object.__new__(pr30_run.OnlineHorizonPolicy)
    p.arm="v4_h20"
    p.geometry=None
    p.rules=None
    p.seed=431
    p.protocol={"decision_cpu_budget_seconds":budget,"particles":3}
    p.horizon=20
    p.observer=SimpleNamespace(observe=lambda raw:SimpleNamespace(tick=0))
    p.solver=object()
    p.forecast_solver=object()
    p.model=FixedClockForecast(clock,cost)
    p.n_calls=p.n_multi=p.n_flip=p.n_model_queries=p.n_model_steps=0
    p.n_budget_exceeded=p.n_fallback=p.n_errors=0
    p.decisions=[]
    p.cpu_decision_seconds=0.
    return p


class PostcandidateBudgetTests(TestCase):
    def test_last_candidate_overshoot_must_fallback_to_first(self):
        clock=[0.]
        policy=fake_policy(clock,{("first",):0.18,("second",):0.48})
        with patch.object(pr30_run,"_unique_plans",
                          return_value=[("first",),("second",)]),\
             patch.object(pr30_run.time,"process_time",side_effect=lambda:clock[0]):
            selected=policy(None,())
        self.assertEqual(selected,("first",))
        self.assertEqual(policy.n_budget_exceeded,1)
        self.assertEqual(policy.n_fallback,1)
        self.assertEqual(policy.n_flip,0)
        self.assertEqual(policy.n_model_queries,2)
        self.assertGreater(policy.decisions[-1]["process_cpu_seconds"],0.5)

    def test_within_budget_second_candidate_is_still_chosen(self):
        clock=[0.]
        policy=fake_policy(clock,{("first",):0.10,("second",):0.12})
        with patch.object(pr30_run,"_unique_plans",
                          return_value=[("first",),("second",)]),\
             patch.object(pr30_run.time,"process_time",side_effect=lambda:clock[0]):
            selected=policy(None,())
        self.assertEqual(selected,("second",))
        self.assertEqual(policy.n_budget_exceeded,0)
        self.assertEqual(policy.n_fallback,0)
        self.assertEqual(policy.n_flip,1)

    def test_first_candidate_crossing_budget_never_ranks_second(self):
        clock=[0.]
        policy=fake_policy(clock,{("first",):0.61,("second",):0.01})
        with patch.object(pr30_run,"_unique_plans",
                          return_value=[("first",),("second",)]),\
             patch.object(pr30_run.time,"process_time",side_effect=lambda:clock[0]):
            selected=policy(None,())
        self.assertEqual(selected,("first",))
        self.assertEqual(policy.n_budget_exceeded,1)
        self.assertEqual(policy.n_fallback,1)
        self.assertEqual(policy.n_model_queries,1)


if __name__=="__main__":
    main()
