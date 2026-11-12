"""PR54 real chronological PUBLIC one-step business world-state transitions.

Model features use only observations at t-3..t and the ACTUAL legal
assignment at t. Transition after at t+1 is a LABEL ONLY, never input.
Private schedules exist solely inside the true simulator.

Reuse PR51 public action graph / node features and PR52 node truth labels.
"""
from __future__ import annotations
from collections import deque
from pathlib import Path
import argparse
import hashlib
import json
import time

from business_transport.observation import BusinessObserver
from business_transport.simulation import BusinessSimulator,resource_audit
from experiments.business_constraints.run import lifecycle_audit
from experiments.world_model_integration.stress_pr48.pr48_scenarios import (
    PROTOCOL as STRESS_PROTOCOL,build_business
)
from experiments.world_model_integration.decision_learning_pr51.pr51_labels import (
    TrueCPsatBranch,public_relational_graph
)
from experiments.world_model_integration.decision_learning_pr51.pr51_train import (
    node_features
)
from experiments.world_model_integration.entity_pr52.pr52_labels import (
    _future_nodes,anchor_ids
)
from lab_transport.multisite_experiment import discrete_safety_violations

PROTOCOL=Path(__file__).with_name("protocol.json")
REGIMES=("congestion","multi_fault","staggered_arrivals")
SPLITS=("train","validation","test")
D=16


def validate(cfg):
    if cfg["protocol"]!="PR54_TEMPORAL_ACTION_RELATIONAL_NEXT_STATE_CPU_PREFLIGHT_20261112_V1":
        raise ValueError("PR54 protocol mismatch")
    if cfg["parent_head"]!="7bfefc35840ab09677e88b8db375bc8a06b86e50":
        raise ValueError("source model SHA changed")
    if cfg["history_window"]!=4 or cfg["horizon_ticks"]!=1:
        raise ValueError("changed causal time window")
    if cfg["candidate_limit"]!=4 or cfg["train_seeds"]!=[1810,1811]:
        raise ValueError("pre-registered controls changed")
    cases=cfg["cohort"]
    if len(cases)!=60 or len({c["seed"] for c in cases})!=60:
        raise ValueError("expected 60 independent cases")
    if any(c["seed"]<54000000 for c in cases):
        raise ValueError("reused prior study seed namespace")
    for regime in REGIMES:
        for split,count in (("train",12),("validation",4),("test",4)):
            subset=[c for c in cases if c["split"]==split and c["regime"]==regime]
            if len(subset)!=count or sum(c["mechanism"]=="within_support" for c in subset)!=count//2:
                raise ValueError("invalid scenario stress and split balance")
    return True


def graph_public_features(obs,rules,assignments=()):
    graph=public_relational_graph(obs,rules,[tuple(assignments)])
    keys=anchor_ids(obs,rules)
    if len(keys)!=len(graph["nodes"]):
        raise AssertionError("public graph and entity keys diverged")
    expected=[f"{kind}:{i}" for kind in ("robot","station","resource","task")
              for i in range(sum(name==kind for name,_ in keys))]
    if [n["id"] for n in graph["nodes"]]!=expected:
        raise AssertionError("PR51 graph order changed; temporal alignment unsafe")
    selected={e["src"] for e in graph["candidate_actions"][0]}
    selected.update(e["dst"] for e in graph["candidate_actions"][0])
    return graph,keys,[node_features(n,selected) for n in graph["nodes"]]


def build_transition(before,after,assignments,previous,rules,window=4):
    """Seal public input before accessing future AFTER as the training label."""
    if after.tick!=before.tick+1:
        raise ValueError("consecutive one-step public observations required")
    if len(previous)>window-1:
        raise ValueError("past history window exceeded")
    if any(item["tick"]>=before.tick for item in previous):
        raise AssertionError("future observation leaked into historical input")
    graph,keys,current=graph_public_features(before,rules,assignments)
    n=len(keys)
    history=[[[0.]*D for _ in range(window)] for _ in range(n)]
    visible=[[0]*window for _ in range(n)]
    for j,prior in enumerate(previous):
        at=window-1-len(previous)+j
        lookup=dict(zip(prior["keys"],prior["features"]))
        for i,key in enumerate(keys):
            val=lookup.get(key)
            if val is not None:
                history[i][at]=val
                visible[i][at]=1
    for i,features in enumerate(current):
        history[i][-1]=features
        visible[i][-1]=1
    anchor={"node_keys":keys,"rules":rules}
    persistent=_future_nodes(before,anchor)
    future=_future_nodes(after,anchor)
    if len(future["node_target"])!=n:
        raise AssertionError("future node labels misaligned")
    if persistent["global_newly_visible_stage_count"]:
        raise AssertionError("new future task was added at current observation")
    if future["global_new_stage_completed_count"]>future["global_newly_visible_stage_count"]:
        raise AssertionError("new task completed count exceeds new task visibility")
    x={
        "public_tick":before.tick,"graph":graph,"history":history,
        "history_visible_mask":visible,
        "node_persistence":persistent["node_target"],
        "feature_last_observed_tick":before.tick,
        "input_source":"public t-3..t only, plus current actually selected dispatch"
    }
    label={
        "next_public_tick":after.tick,
        "node_target":future["node_target"],
        "node_mask":future["node_target_mask"],
        "new_visible_task_stages":future["global_newly_visible_stage_count"],
        "new_completed_task_stages":future["global_new_stage_completed_count"],
        "target_source":"next actual public BusinessObserver observation only"
    }
    return x,label,{"tick":before.tick,"keys":keys,"features":current}


class PublicTransitionCollector:
    def __init__(self,rules,window,max_steps):
        self.rules=rules
        self.window=window
        self.max_steps=max_steps
        self.observer=BusinessObserver()
        self.prior=deque(maxlen=window-1)
        self.rows=[]
        self.tick=-1

    def __call__(self,tr):
        before=self.observer.observe(tr.before)
        after=self.observer.observe(tr.after)
        if before.tick!=self.tick+1:
            raise AssertionError("true transition callback not monotonic")
        self.tick=before.tick
        # Future AFTER only appears in the LABEL, never in the past deque.
        if len(self.rows)<self.max_steps:
            inp,truth,memo=build_transition(before,after,tr.assignments,
                                            list(self.prior),self.rules,self.window)
            self.rows.append({"input":inp,"target":truth})
            self.prior.append(memo)
        if self.rows and self.rows[-1]["input"]["public_tick"]>=self.max_steps:
            raise AssertionError("max public time sampling window violated")


def run_one(spec,cfg,scenario):
    business,geometry=build_business(spec,scenario)
    policy=TrueCPsatBranch(
        geometry,business.rules,spec["seed"],
        {"counterfactual_max_candidates":cfg["candidate_limit"]})
    collector=PublicTransitionCollector(
        business.rules,cfg["history_window"],cfg["max_transitions_per_episode"])
    result=BusinessSimulator(
        business,"world",spec["seed"],collect_metrics=True).run(
        dispatch_policy=policy,on_transition=collector)
    if resource_audit(result,business.rules)["violations"]:
        raise AssertionError("unsafe true business resource state")
    if discrete_safety_violations(result) or lifecycle_audit(result["history"]):
        raise AssertionError("unsafe true motion or workflow")
    if len(collector.rows)<10:
        raise AssertionError("insufficient public time history")
    for row in collector.rows:
        inp=row["input"]
        if inp["feature_last_observed_tick"]+1!=row["target"]["next_public_tick"]:
            raise AssertionError("future training label leaked into features")
        if len(inp["history"])!=len(inp["graph"]["nodes"]):
            raise AssertionError("node / historical memory dimension mismatch")
    return {
        "id":spec["id"],"seed":spec["seed"],"regime":spec["regime"],
        "split":spec["split"],"mechanism":spec["mechanism"],
        "real_simulator_status":result["summary"]["status"],
        "real_observation_transitions":len(collector.rows),
        "transitions":collector.rows
    }


def run(regime,out):
    cfg=json.loads(PROTOCOL.read_text())
    validate(cfg)
    if regime not in REGIMES:
        raise ValueError("unregistered regime")
    if out.exists():
        raise FileExistsError("immutable public training source exists")
    scenario=json.loads(STRESS_PROTOCOL.read_text())
    start=time.process_time()
    episodes=[]
    for spec in cfg["cohort"]:
        if spec["regime"]!=regime:
            continue
        item=run_one(spec,cfg,scenario)
        episodes.append(item)
        print("PR54_PUBLIC_EPISODE",spec["id"],
              "one_step_samples",item["real_observation_transitions"],flush=True)
    if len(episodes)!=20:
        raise AssertionError("20 original episodes per stress class required")
    summary={
        "episodes":len(episodes),
        "by_split":{s:sum(x["split"]==s for x in episodes) for s in SPLITS},
        "true_public_one_step_samples":sum(x["real_observation_transitions"] for x in episodes),
        "public_only_features":True,"lookahead_feature_leakage":False,
        "training":0,"gpu":0,"cpu_process_seconds":time.process_time()-start
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({
        "schema":"pr54-true-public-temporal-world-transitions-v1",
        "regime":regime,
        "protocol_sha256":hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "episodes":episodes,"summary":summary
    },ensure_ascii=False)+"\n")
    print("PR54_PUBLIC_SOURCE_RESULT",json.dumps(summary),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--regime",required=True,choices=REGIMES)
    p.add_argument("--out",required=True,type=Path)
    x=p.parse_args()
    run(x.regime,x.out)


if __name__=="__main__":
    main()
