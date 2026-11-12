"""Generate grouped factual V4-format service-event records via EXISTING generator.

No new labels, simulator mechanics, oracle future or private countdown.
PR31 train = original V4 Train replay + new preregistered groups.
The original V4 validation, all old test groups and all PR28-30 scenes
are excluded from model development. Independent Test is generated only
AFTER checkpoint and validation selection have been frozen.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import argparse

from experiments.world_model_integration.build_dataset_v4 import (
    build_one,CONFIG as OLD_CONFIG,
)
from business_transport.session_events import QUERY_LABEL_VERSION
from business_transport.session_features_v3 import INPUT_SCHEMA_V3
from business_transport.state_features import SCHEMA_VERSION

PROTOCOL=Path(__file__).with_name("pr31_protocol.json")


def _assert_seeds(protocol):
    original=json.loads(OLD_CONFIG.read_text())
    new=[x["seed"] for section in (
        "train","validation","independent_test") for x in protocol["groups"][section]]
    new.extend(x["seed"] for x in protocol["decision_test_groups"])
    if len(new)!=len(set(new)):raise AssertionError("PR31 scene split seed overlap")
    old={x["seed"] for grp in original["groups"].values() for x in grp}
    if old&set(new):raise AssertionError("original V4/PR31 seed overlap")
    from experiments.world_model_integration.horizon_generalization.pr29_evaluate import _scenario_seed_set
    _scenario_seed_set({"scene_groups":[
        x for section in ("train","validation","independent_test")
        for x in protocol["groups"][section]]+protocol["decision_test_groups"]})
    pr29=json.loads((PROTOCOL.parents[1]/"horizon_generalization"/"pr29_protocol.json").read_text())
    pr30=json.loads((PROTOCOL.parents[1]/"online_horizon"/"pr30_protocol.json").read_text())
    previous={x["seed"] for x in pr29["scene_groups"]+pr30["groups"]}
    if set(new)&previous:
        raise AssertionError("known PR29/30 research groups leaked into PR31")


def make(part,config_path=PROTOCOL):
    p=Path(config_path)
    cfg=json.loads(p.read_text())
    if cfg["protocol"]!="RHWM_PR31_V4_WARMSTART_SCRATCH_REPRO_TRAIN_20261112_V1":
        raise ValueError("wrong precommitted training split")
    _assert_seeds(cfg)
    if part not in ("development","independent_test"):
        raise ValueError("must generate train/validation before independent test")
    original=json.loads(OLD_CONFIG.read_text())
    sub=("train","validation") if part=="development" else ("independent_test",)
    all_rows={}
    all_groups={}
    coverage={}
    for split in sub:
        sources=[]
        if split=="train":
            sources.extend(("replay_v4_train",spec,original)
                           for spec in original["groups"]["train"])
        sources.extend(("new_"+split,spec,cfg) for spec in cfg["groups"][split])
        rows=[]
        used=set()
        counter=Counter()
        facts=[]
        for source,spec,build_cfg in sources:
            selected,counts,summary=build_one(spec,build_cfg,split)
            group=summary["trajectory_group"]
            if group in used:raise AssertionError("duplicated factual source group")
            used.add(group)
            if not selected:raise AssertionError("empty factual group "+group)
            if summary["simulation_status"]!="completed":
                raise AssertionError("synthetic factual source incomplete")
            rows.extend(selected)
            counter.update(counts)
            facts.append({"source":source,"id":spec.get("id",group),
                          "seed":spec["seed"],"jobs":len(spec["job_workflows"]),
                          "jitter":spec["jitter_ticks"],"records":len(selected),
                          "positive":sum(x["label"] for x in selected)})
        if not 0<sum(x["label"] for x in rows)<len(rows):
            raise AssertionError("training/validation contains only one label")
        all_rows[split]=rows
        all_groups[split]=facts
        coverage[split]={
            "records":len(rows),"positives":sum(x["label"] for x in rows),
            "groups":len(facts),"counts":dict(counter),
            "source_groups":facts}
    return {
        "schema":"rhwm-pr31-factual-service-query-v3",
        "part":part,
        "protocol_sha256":hashlib.sha256(p.read_bytes()).hexdigest(),
        "old_v4_config_sha256":hashlib.sha256(OLD_CONFIG.read_bytes()).hexdigest(),
        "input_schema":INPUT_SCHEMA_V3,
        "observation_schema":SCHEMA_VERSION,
        "label_schema":QUERY_LABEL_VERSION,
        "source":"build_dataset_v4.build_one -> BusinessSimulator original factual transitions",
        "source_groups":all_groups,
        "coverage":coverage,**all_rows,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--part",choices=("development","independent_test"),required=True)
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError("frozen data is write-once")
    result=make(args.part)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,sort_keys=True,ensure_ascii=False)+"\n")
    print("PR31_DATA_BUILD",json.dumps({k:{"groups":v["groups"],"rows":v["records"],"positives":v["positives"]}
         for k,v in result["coverage"].items()}))


if __name__=="__main__":
    main()
