"""PR51 controlled pilot: OFF-THE-SHELF PyG RGCNConv, predictive vs decision training.

Shared action-conditioned relational dynamics predicts public H5/H20 system
state summaries. The dispatch score in EVERY arm is the original PR30
readout: *negative predicted H20 completed-stage count*. There is NO
standalone trainable action-ranking head. The pairwise decision loss can
only improve ranking by changing shared predicted future system state.

Input = PR51 verified public graphs only; true counterfactual consequences
are labels. Train/validation files are read BEFORE checkpoint selection;
independent test path is opened ONLY AFTER selection.json is written.
This is a small-data training pilot, not the complete deployable RHWM.
"""
from __future__ import annotations

from pathlib import Path
from itertools import combinations
import argparse
import json
import math
import random
import time

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data,Batch
from torch_geometric.nn import RGCNConv,global_mean_pool

from experiments.world_model_integration.decision_learning_pr51.pr51_loss import (
    pairwise_decision_loss,gradient_path_present
)

PROTOCOL=Path(__file__).with_name("pr51_train_protocol.json")
RELATIONS=(
    "currently_assigned","uses_resource","pickup_at","dropoff_at",
    "same_workflow_successor","proposed_robot_task_assignment"
)
EDGE_ID={r:i for i,r in enumerate(RELATIONS)}
TARGETS=("completed_stage_count","waiting_stage_count","in_transit_count",
         "busy_resource_port_count","station_device_down_count")


def node_features(node, selected):
    x=[0.]*16
    t=node["kind"]
    x[("robot","task","station","resource").index(t)]=1.
    if "pos" in node:
        x[4],x[5]=node["pos"][0]/7.,node["pos"][1]/5.
    if t=="robot":
        x[6]=float(node["busy"])
        x[7]=float(node["phase"]=="to_pickup")
        x[8]=float(node["phase"]=="to_dropoff")
        x[9]=min(1.,max(0.,(node["service_elapsed"] or 0)/20.))
    elif t=="task":
        x[6]=float(node["status"]=="waiting")
        x[7]=float(node["status"]=="assigned")
        x[8]=float(node["status"]=="in_transit")
        x[9]=float(node["status"]=="completed")
        x[10]=min(1.,node["released_age"]/40.)
        x[11]=min(1.,max(0.,(node["deadline_slack"] or 0)/200.))
        x[12]=float(node["workflow"]=="w1")
        x[13]=min(1.,node["stage_index"]/4.)
    elif t=="station":
        x[6]=float(node["device_available"])
        x[7]=min(1.,node["resource_occupied"]/3.)
        x[8]=min(1.,node["waiting_pickups"]/10.)
        x[9]=min(1.,node["approaching_pickups"]/10.)
    elif t=="resource":
        x[6]=min(1.,node["capacity"]/4.)
    x[15]=float(node["id"] in selected)
    return x


def candidate_graph(graph, candidate_idx, relational):
    nodes=graph["nodes"]
    idx={n["id"]:i for i,n in enumerate(nodes)}
    candidate=graph["candidate_actions"][candidate_idx]
    chosen={edge["src"] for edge in candidate}|{edge["dst"] for edge in candidate}
    x=torch.tensor([node_features(node,chosen) for node in nodes],dtype=torch.float32)
    edges=[]
    kinds=[]
    if relational:
        for row in list(graph["edges"])+candidate:
            if row["kind"] not in EDGE_ID:
                raise ValueError("unregistered public relation")
            a,b=idx[row["src"]],idx[row["dst"]]
            code=EDGE_ID[row["kind"]]
            edges.extend(((a,b),(b,a)))
            kinds.extend((code,code+len(RELATIONS)))
    if edges:
        edge_index=torch.tensor(edges,dtype=torch.long).T.contiguous()
        edge_type=torch.tensor(kinds,dtype=torch.long)
    else:
        edge_index=torch.empty(2,0,dtype=torch.long)
        edge_type=torch.empty(0,dtype=torch.long)
    return Data(x=x,edge_index=edge_index,edge_type=edge_type)


def future_target(outcome,scales):
    future=outcome["true_future_public_snapshots"]
    if "5" not in future or "20" not in future:
        return None
    row=[]
    for h in ("5","20"):
        state=future[h]
        row.extend(state[t]/scales[i] for i,t in enumerate(TARGETS))
    return torch.tensor(row,dtype=torch.float32)


def load_grouped(file):
    docs=json.loads(file.read_text())
    if not isinstance(docs,list):
        raise ValueError("unexpected split file")
    return docs


def prepare(groups,scales,relations,split):
    ready=[]
    rejected=[]
    for case in groups:
        if case["split"]!=split:
            raise ValueError("different cohort split in same file")
        if not case["labelable"] or not case.get("all_branches_completed"):
            rejected.append({"id":case["id"],"reason":"not eligible for complete-label pilot"})
            continue
        k=case["candidate_count"]
        graph=case["public_graph"]
        if len(graph["candidate_actions"])!=k:
            raise AssertionError("candidate action edges do not align with true branches")
        targets=[]
        for outcome in case["true_factual_outcomes"]:
            t=future_target(outcome,scales)
            if t is None:
                break
            targets.append(t)
        if len(targets)!=k:
            rejected.append({"id":case["id"],"reason":"public H5/H20 truth censored"})
            continue
        data=[candidate_graph(graph,j,relations) for j in range(k)]
        prices=torch.tensor(
            [float(x["makespan_ticks"]) for x in case["true_factual_outcomes"]])
        ready.append({
            "id":case["id"],"regime":case["regime"],"batch":Batch.from_data_list(data),
            "target":torch.stack(targets),"factual_cost":prices
        })
    return ready,rejected


class RelationalSystemWorldModel(nn.Module):
    def __init__(self,hidden=32,relational=True):
        super().__init__()
        self.relational=relational
        self.embedding=nn.Linear(16,hidden)
        self.transition1=RGCNConv(hidden,hidden,num_relations=2*len(RELATIONS),
                                  num_bases=4)
        self.transition2=RGCNConv(hidden,hidden,num_relations=2*len(RELATIONS),
                                  num_bases=4)
        self.future_state=nn.Sequential(
            nn.Linear(hidden,hidden),nn.ReLU(),
            nn.Linear(hidden,10),nn.Sigmoid()
        )

    def forward(self,batch):
        h=F.relu(self.embedding(batch.x))
        if self.relational and batch.edge_index.shape[1]:
            h=F.relu(self.transition1(h,batch.edge_index,batch.edge_type))
            h=F.relu(self.transition2(h,batch.edge_index,batch.edge_type))
        pooled=global_mean_pool(h,batch.batch)
        # Observed task graph+selected relation edges -> predicted state.
        return self.future_state(pooled)


def predicted_candidate_cost(pred):
    # frozen decision projection: original PR30 looks for maximum H20
    # expected completed stages. We never train a standalone ranker.
    return -pred[:,5]


def score_groups(model,groups,device):
    model.eval()
    all_mse=[];regrets=[];pair_hits=0;pair_total=0;first_regrets=[]
    breakdown={}
    with torch.no_grad():
        for g in groups:
            forecast=model(g["batch"].to(device))
            target=g["target"].to(device)
            mse=F.mse_loss(forecast,target).item()
            predicted=predicted_candidate_cost(forecast).cpu().tolist()
            truth=g["factual_cost"].tolist()
            choice=min(range(len(truth)),key=lambda i:(predicted[i],i))
            oracle=min(truth)
            reg=truth[choice]-oracle
            all_mse.append(mse)
            regrets.append(reg)
            first_regrets.append(truth[0]-oracle)
            bucket=breakdown.setdefault(g["regime"],[])
            bucket.append(reg)
            for i,j in combinations(range(len(truth)),2):
                if truth[i]==truth[j]:
                    continue
                pair_total+=1
                if (predicted[i]-predicted[j])*(truth[i]-truth[j])>0:
                    pair_hits+=1
    return {
        "groups":len(groups),
        "normalized_future_state_mse":sum(all_mse)/len(all_mse),
        "mean_true_candidate_regret_ticks":sum(regrets)/len(regrets),
        "mean_first_cpsat_candidate_regret_ticks":sum(first_regrets)/len(first_regrets),
        "pairwise_correct":pair_hits,
        "pairwise_comparable":pair_total,
        "pairwise_accuracy":pair_hits/pair_total if pair_total else None,
        "by_regime_regret_ticks":{r:sum(xs)/len(xs) for r,xs in breakdown.items()},
    }


def train_one(arm,training,valid,config,output,device):
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    torch.set_num_threads(1)
    model=RelationalSystemWorldModel(
        config["hidden_size"],relational=arm["relations"]).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=config["learning_rate"])
    history=[]
    grad_verified=None
    for epoch in range(config["epochs"]):
        model.train()
        order=list(range(len(training)))
        random.Random(config["seed"]+epoch).shuffle(order)
        loss_total=0.
        for index in order:
            g=training[index]
            predicted=model(g["batch"].to(device))
            target=g["target"].to(device)
            observed=g["factual_cost"].to(device)
            pred_loss=F.mse_loss(predicted,target)
            decision_loss=pairwise_decision_loss(
                predicted_candidate_cost(predicted),observed)
            loss=pred_loss+arm["lambda"]*decision_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if arm["lambda"]>0 and grad_verified is None and (
                torch.unique(observed).numel()>1):
                # Decision-only gradient must reach actual relation transition
                # weights. Only enough to verify connectivity.
                optimizer.zero_grad(set_to_none=True)
                temp=model(g["batch"].to(device))
                pairwise_decision_loss(predicted_candidate_cost(temp),observed).backward()
                grad_verified=gradient_path_present(model.transition1.parameters())
                if not grad_verified and arm["relations"]:
                    raise AssertionError("decision loss bypasses shared transition model")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
            optimizer.step()
            loss_total+=loss.item()
        if epoch in (0,config["epochs"]-1):
            val_metrics=score_groups(model,valid,device)
            history.append({"epoch":epoch+1,
                            "mean_train_loss":loss_total/len(training),
                            "validation":val_metrics})
    if arm["relations"] and arm["lambda"]>0 and not grad_verified:
        raise AssertionError("no nonzero gradient into relational dynamics from decision loss")
    path=output/(arm["id"]+".pt")
    state={
        "model":model.state_dict(),
        "optimizer":optimizer.state_dict(),
        "torch_rng_state":torch.get_rng_state(),
        "arm":arm,"seed":config["seed"],"epochs":config["epochs"],
        "model_class":"PyG_RGCNConv_relational_future_world_summary",
        "fixed_dispatch_score":"-predicted_H20_completed_stage_count",
        "checkpoint_complete":True,
    }
    torch.save(state,path)
    return model,{
        "model_id":arm["id"],
        "arm":arm,
        "validation":score_groups(model,valid,device),
        "train":score_groups(model,training,device),
        "only_decision_gradient_to_relation_transition_verified":grad_verified,
        "train_history":history,
        "checkpoint":path.name,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train",type=Path,required=True)
    ap.add_argument("--validation",type=Path,required=True)
    ap.add_argument("--test",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():
        raise FileExistsError("PR51 training artifacts must be immutable")
    cfg=json.loads(PROTOCOL.read_text())
    if cfg["protocol"]!="PR51_RGCN_DECISION_AWARE_TRAINING_PILOT_20261112_V1":
        raise ValueError("changed model protocol")
    args.out.mkdir(parents=True)
    # STRICT: train/validation alone, neither test file nor test outcomes read.
    train_cases=load_grouped(args.train)
    validation_cases=load_grouped(args.validation)
    if {c["seed"] for c in train_cases}&{c["seed"] for c in validation_cases}:
        raise AssertionError("same seed across train & validation")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arms={}
    for arm in cfg["train_arms"]:
        training,train_rejected=prepare(
            train_cases,cfg["normalized_state_target_scales"],arm["relations"],"train")
        validation,val_rejected=prepare(
            validation_cases,cfg["normalized_state_target_scales"],arm["relations"],"validation")
        if not training or not validation:
            raise AssertionError("missing multi-step public-state ground truth")
        model,report=train_one(arm,training,validation,cfg,args.out,device)
        report["train_rejected"]=train_rejected
        report["validation_rejected"]=val_rejected
        arms[arm["id"]]=report
        print("PR51_TRAINED_RELATIONAL_WORLD_MODEL",json.dumps({
            "arm":arm["id"],"val":report["validation"],
            "grad_verified":report["only_decision_gradient_to_relation_transition_verified"]
        },ensure_ascii=False),flush=True)
    # Fix the selected arm from VALIDATION only. Ties prefer lower state MSE.
    chosen=min(arms,key=lambda a:(
        arms[a]["validation"]["mean_true_candidate_regret_ticks"],
        arms[a]["validation"]["normalized_future_state_mse"],
        list(arms).index(a)))
    freeze={
        "schema":"pr51-validation-only-relational-world-model-selection-v1",
        "selection_model_id":chosen,
        "criterion":cfg["checkpoint_selection"],
        "all_validation":{a:v["validation"] for a,v in arms.items()},
        "all_training":{a:v["train"] for a,v in arms.items()},
        "independent_test_was_not_opened":True,
        "model_training_completed":True,
        "gpu_training_used":device.type=="cuda",
        "fixed_decision_score":cfg["use_fixed_original_pr30_score"],
    }
    (args.out/"selection.json").write_text(json.dumps(freeze,ensure_ascii=False,indent=2)+"\n")
    # Only now load independent test source. No test-derived tuning or retrain.
    test_cases=load_grouped(args.test)
    seen={c["seed"] for c in train_cases+validation_cases}
    if seen & {c["seed"] for c in test_cases}:
        raise AssertionError("independent test seed leakage")
    results={}
    for arm in cfg["train_arms"]:
        a=arm["id"]
        prepared,rejected=prepare(test_cases,cfg["normalized_state_target_scales"],
                                  arm["relations"],"test")
        checkpoint=torch.load(args.out/(a+".pt"),map_location=device,weights_only=False)
        model=RelationalSystemWorldModel(cfg["hidden_size"],arm["relations"]).to(device)
        model.load_state_dict(checkpoint["model"])
        results[a]={"metrics":score_groups(model,prepared,device),"rejected":rejected}
    final={
        "schema":"pr51-pilot-relational-decision-aware-trained-world-model-v1",
        "models_trained":len(cfg["train_arms"]),
        "model_selection_before_independent_test":True,
        "validation_chosen":chosen,
        "selection":freeze,
        "independent_test":results,
        "source_unit":"72 independent true-world scenarios, 216 paired alternate candidates",
        "gpu_training_used":device.type=="cuda",
        "pilot_limit":"Small synthetic same-map data, predicted 10 aggregate public state metrics not full per-entity distribution, no new full-online closed-loop evaluation"
    }
    (args.out/"result.json").write_text(json.dumps(final,ensure_ascii=False,indent=2)+"\n")
    print("PR51_TRAINING_PILOT_RESULT",json.dumps({
        "selected":chosen,
        "validation_regret":{a:r["validation"]["mean_true_candidate_regret_ticks"]
                             for a,r in arms.items()},
        "test":{a:z["metrics"] for a,z in results.items()},
        "decision_gradient_verified":{a:r["only_decision_gradient_to_relation_transition_verified"]
                                      for a,r in arms.items()},
        "gpu_training":device.type=="cuda"
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
