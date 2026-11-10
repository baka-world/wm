"""One comparison for a single frozen V4 split, with factual labels only.

V4 and frozen V3-C use identical input schema/architecture and output P(finish
next step | public state and executed candidate); B0/B1/B2 fit only V4 Train.
The independent subsets are run once with separately pre-committed configs.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import torch
from business_transport.dynamics import WorkflowDynamics
from business_transport.session_events import eligible_sessions
from experiments.world_model_integration.build_dataset import case_for,rows_from_factual,label_for
from experiments.world_model_integration.build_dataset_v4 import (CONFIG,record_attributes)
from experiments.world_model_integration.empirical_baselines import EmpiricalHazards
from lab_transport.business_session_v3 import (load_v3,probs as v3_probs,
                                                stats as base_stats,grouped as base_grouped)
from experiments.world_model_integration.train_v4 import load_v4,sha
from lab_transport.torch_structured_adapter import ProgressAwareRule


def _conditional(records,probs):
    groups=defaultdict(list)
    for row,p in zip(records,probs):
        a=record_attributes(row)
        groups[(a["prior"],a["elapsed"])].append((row,p))
    return [{"public_ticks":d,"active_elapsed":e,"n":len(values),
             "finished":sum(r["label"] for r,p in values),
             "observed_rate":sum(r["label"] for r,p in values)/len(values),
             "mean_predicted":statistics.mean(p for r,p in values)}
            for (d,e),values in sorted(groups.items())]


def _group(rows,probs,dimension):
    result={}
    for row,p in zip(rows,probs):
        if dimension=="trajectory":key=row["trajectory_group"]
        else:key=str(record_attributes(row)[dimension])
        result.setdefault(key,[]).append((row,p))
    return {key:base_stats([r for r,p in values],[p for r,p in values])
            for key,values in sorted(result.items())}


def _factual_rule(records,part,config):
    keys={(r["seed"],r["tick"],r["query"]["robot_id"],r["query"]["sample_id"]):i
          for i,r in enumerate(records)}
    if len(keys)!=len(records):raise ValueError("duplicate factual supervised event")
    results=[None]*len(records);matched=set()
    for spec in config["groups"][part]:
        business,geo=case_for(spec,config)
        events,_=rows_from_factual(business,geo,spec["seed"])
        rule=WorkflowDynamics(ProgressAwareRule(geo),business.rules)
        for factual in events:
            before,actions,after=factual.observation,factual.assignments,factual.next_observation
            accepted=[]
            for query,reason in eligible_sessions(before,actions):
                if reason is not None:continue
                label,event=label_for(before,after,query)
                if label is None:continue
                key=(spec["seed"],before.tick,query["robot_id"],query["sample_id"])
                if key not in keys:raise ValueError("factual rule pairing failed")
                accepted.append((key,query))
            if accepted:
                predicted=rule.predict_next(before,actions)
                result={s.id:s.status for s in predicted.samples}
                for key,query in accepted:
                    target="in_transit" if query["phase"]=="to_pickup" else "completed"
                    i=keys[key]
                    if key in matched:raise ValueError("multiple predictions for same factual event")
                    matched.add(key)
                    results[i]=float(result[query["sample_id"]]==target)
    if len(matched)!=len(keys):
        raise ValueError("factual rule predictions lack paired events: "+str(len(keys)-len(matched)))
    return results


def evaluate(dev_path,split_path,v3_path,v4_path,part,output):
    path=Path(output)
    if path.exists():raise FileExistsError("V4 independent metrics are immutable")
    configbytes=CONFIG.read_bytes()
    config=json.loads(configbytes)
    if part not in ("validation","independent_in","independent_ood"):
        raise ValueError("evaluation scope must be predeclared")
    dev=json.loads(Path(dev_path).read_text())
    data=dev if part=="validation" else json.loads(Path(split_path).read_text())
    split=dev["validation"] if part=="validation" else data[part]
    if data["source_config_sha256"]!=sha(CONFIG) or dev["source_config_sha256"]!=sha(CONFIG):
        raise ValueError("frozen source config differs")
    if part.startswith("independent") and data.get("part")!=part:
        raise ValueError("one-shot independent split wrong or already contaminated")
    if len({r["trajectory_group"] for r in split})!=len(config["groups"][part]):
        raise ValueError("entire factual trajectory groups not found")
    old,oldmeta=load_v3(v3_path,experimental=True)
    new,newmeta=load_v4(v4_path,experimental=True)
    if oldmeta.get("variant")!="V3-C" or oldmeta.get("online_verified") is True or newmeta["online_verified"]:
        raise ValueError("nonfrozen/production model used in research evaluation")
    if oldmeta["input_schema"]!=newmeta["input_schema"]:
        raise ValueError("different target input schema, cannot compare model-only data intervention")
    if newmeta.get("dataset_sha256")!=sha(dev_path):
        raise ValueError("V4 checkpoint training data hash differs from verified V4 development split")
    torch.set_num_threads(1)
    hazards=EmpiricalHazards(dev["train"])
    probs={
        "B0":hazards.predict(split,"B0"),
        "B1":hazards.predict(split,"B1"),
        "B2":hazards.predict(split,"B2"),
        "Rule":_factual_rule(split,part,config),
        "V3-C frozen":v3_probs(old,split,oldmeta["input_schema"]),
        "V4":v3_probs(new,split,newmeta["input_schema"])}
    result={"protocol":"pr18-v4-frozen-synthetic-mechanism-20261110",
      "evaluation_scope":part,
      "one_shot":part!="validation",
      "source_config_sha256":sha(CONFIG),
      "source_data_sha256":sha(split_path) if part!="validation" else sha(dev_path),
      "training_data_sha256":sha(dev_path),
      "v3_frozen_checkpoint_sha256":sha(v3_path),
      "v4_checkpoint_sha256":sha(v4_path),
      "n_events":len(split),"n_positive":sum(r["label"] for r in split),
      "trajectory_groups":len(data["groups"][part]),
      "coverage":data["coverage"][part],
      "models":{}}
    for name,pp in probs.items():
        if len(pp)!=len(split) or any(not 0<=float(p)<=1 for p in pp):
            raise ValueError("invalid conditional probability "+name)
        result["models"][name]={
          "overall":base_stats(split,pp),
          "per_trajectory":_group(split,pp,"trajectory"),
          "per_public_prior":_group(split,pp,"prior"),
          "per_elapsed":_group(split,pp,"elapsed"),
          "per_station":_group(split,pp,"station"),
          "per_phase":_group(split,pp,"phase"),
          "per_capacity":_group(split,pp,"capacity"),
          "conditional_curve":_conditional(split,pp)}
    models=result["models"]
    b0=models["B0"]["overall"]["brier"]
    v4=models["V4"]["overall"]["brier"]
    old=models["V3-C frozen"]["overall"]["brier"]
    rule=models["Rule"]["overall"]["brier"]
    improved=sum(models["V4"]["per_trajectory"][g]["brier"]<
                 models["B0"]["per_trajectory"][g]["brier"]
                 for g in models["B0"]["per_trajectory"])
    result["gate"]={
      "v4_brier_below_B0":v4<b0,
      "v4_brier_below_Rule":v4<rule,
      "v4_brier_below_frozen_V3_C":v4<old,
      "v4_brier_below_B1":v4<models["B1"]["overall"]["brier"],
      "v4_brier_below_B2":v4<models["B2"]["overall"]["brier"],
      "groups_improved_vs_B0":improved,
      "requires_groups_improved":config["decision_gate"]["minimum_groups_improved_vs_B0"],
      "eligible_for_limited_dispatch":bool(part=="independent_in" and v4<b0 and improved>=
            config["decision_gate"]["minimum_groups_improved_vs_B0"])}
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"part":part,"n":len(split),"positive":result["n_positive"],
         "brier":{k:v["overall"]["brier"] for k,v in models.items()},
         "log_loss":{k:v["overall"]["log_loss"] for k,v in models.items()},
         "gate":result["gate"]},ensure_ascii=False))
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--development",type=Path,required=True)
    p.add_argument("--split",type=Path)
    p.add_argument("--v3-frozen",type=Path,required=True)
    p.add_argument("--v4",type=Path,required=True)
    p.add_argument("--part",choices=("validation","independent_in","independent_ood"),required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    evaluate(a.development,a.split,a.v3_frozen,a.v4,a.part,a.output)

if __name__=="__main__":
    main()
