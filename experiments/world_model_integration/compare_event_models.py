"""Same factual service-event probability comparison on held-out grouped test.

Run once after model selection. Factual t+1 is used solely as label; models
receive only t's business observation and executed current assignment.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch

from business_transport.dynamics import WorkflowDynamics
from business_transport.session_events import eligible_sessions
from business_transport.state_features import build_business_features
from business_transport.session_dynamics import SessionRuleDynamics
from experiments.business_constraints.run import make_model
from experiments.business_constraints.tests.test_business import fixture
from experiments.service_value.counterfactual import load_pr9_model
from experiments.world_model_integration.build_dataset import case_for, rows_from_factual, CONFIG
from lab_transport.business_session_learning import metrics, probabilities, load_session_model
from lab_transport.dynamics_model import _service_features
from lab_transport.paper_reuse.service_dataset import service_features
from lab_transport.torch_structured_adapter import ProgressAwareRule,entity_tokens


def structured_probability(network,observation,assignments,query):
    robots={r.id:r for r in observation.robots}
    samples={s.id:s for s in observation.samples}
    stations={s.id:s for s in observation.stations}
    robot=robots[query["robot_id"]];sample=samples[query["sample_id"]]
    station=stations[query["station_id"]]
    expected=(station.observed_effective_service_mean if
              station.observed_effective_service_mean is not None else station.expected_service_ticks)
    values=entity_tokens(observation,assignments)
    with torch.inference_mode():
        tokens=torch.tensor([values],dtype=torch.float32)
        mask=torch.ones(tokens.shape[:2],dtype=torch.bool)
        context=network.context(tokens,mask)[0]
        features=_service_features(robot,query["phase"],expected,
                                   sample.id in {a.sample_id for a in assignments})
        logit=network.service(torch.cat((torch.tensor(features,dtype=torch.float32),context))).item()
        return float(torch.sigmoid(torch.tensor(logit)))


def evaluate(checkpoint,dataset_path,output):
    dataset=json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    cfg=json.loads(CONFIG.read_text(encoding="utf-8"))
    if dataset["source_config_sha256"]!=__import__("hashlib").sha256(CONFIG.read_bytes()).hexdigest():
        raise ValueError("dataset source hash changed")
    model,meta=load_session_model(checkpoint,experimental=True)
    network,info=load_pr9_model();network.eval()
    records={"rule":[],"pets":[],"structured":[],"session_v2":[]}
    row_groups={r["id"]:r for r in dataset["test"]}
    added=set()
    for spec in cfg["groups"]["test"]:
        business,geo=case_for(spec,cfg)
        source,_=rows_from_factual(business,geo,spec["seed"])
        pets=make_model("pets",geo,network,info["threshold"],spec["seed"]).distribution
        rule=WorkflowDynamics(ProgressAwareRule(geo),business.rules)
        for row in source:
            obs,actions,actual=row.observation,row.assignments,row.next_observation
            matches=eligible_sessions(obs,actions,require_active=True)
            if not matches:
                continue
            prediction=rule.predict_next(obs,actions)
            real_samples={s.id:s for s in actual.samples}
            pred_samples={s.id:s for s in prediction.samples}
            robots={r.id:r for r in obs.robots}
            samples={s.id:s for s in obs.samples}
            stations={s.id:s for s in obs.stations}
            for query,_ in matches:
                lookup=next((r for r in row_groups.values() if r["seed"]==spec["seed"] and
                             r["tick"]==obs.tick and r["query"]==query),None)
                if lookup is None:continue
                id_=lookup["id"]
                if id_ in added:raise ValueError("duplicate held-out service event")
                added.add(id_)
                sample=samples[query["sample_id"]]
                target="in_transit" if query["phase"]=="to_pickup" else "completed"
                rule_prob=float(pred_samples[sample.id].status==target)
                robot=robots[query["robot_id"]]
                station=stations[query["station_id"]]
                parameters=pets.parameters(service_features(obs,robot,sample,query["phase"],station))
                pets_prob=pets.finish_probability(parameters,robot.service_active_elapsed)
                structured_prob=structured_probability(network,obs,actions,query)
                learned_prob=probabilities(model,[lookup])[0]
                for name,p in (("rule",rule_prob),("pets",pets_prob),("structured",structured_prob),
                               ("session_v2",learned_prob)):
                    records[name].append((lookup,p))
    if added!=set(row_groups):
        raise ValueError("not all eligible held-out service events were paired")
    out={"status":"fixed_group_test_one_time","rows":len(added),
         "checkpoint_best_epoch":meta["best_validation_epoch"],
         "checkpoint_validation_threshold":meta["selected_validation_threshold"],
         "models":{}}
    for name,values in records.items():
        rows=[r for r,p in values]
        probs=[p for r,p in values]
        threshold=({"structured":info["threshold"],"session_v2":meta["selected_validation_threshold"]}
                   .get(name,0.5))
        groups={}
        for r,p in values:
            key=f"layout{r['map_layout']}:seed{r['seed']}"
            groups.setdefault(key,[]).append((r,p))
        out["models"][name]={"overall":metrics(rows,probs,threshold),
            "grouped":{k:metrics([r for r,p in group],[p for r,p in group],threshold)
                        for k,group in sorted(groups.items())}}
    if Path(output).exists():raise FileExistsError("results are write-once")
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(out,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({"rows":out["rows"],"models":{k:v["overall"] for k,v in out["models"].items()}},
                     ensure_ascii=False))
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    evaluate(args.checkpoint,args.dataset,args.output)


if __name__=="__main__":
    main()
