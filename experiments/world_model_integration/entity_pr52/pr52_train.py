"""PR52: action-conditioned nodewise relational world state learning.

This is an ENTITY-level H5/H20 probabilistic state predictor, not a
fully autoregressive simulator. Uses OFFICIAL PyTorch Geometric RGCNConv
with public entities/edges + candidate assignment edges. The decision
readout is FIXED to expected H20 completed stages, from node completion
probabilities plus new-stage completion events. NO Q-head or trainable
candidate scorer. Same score and network dimensions for all arms.

Compare prediction only, prediction+same-prefix true ranking loss, and
fixed H20-completion sensitivity weighting inspired by VaGraM.
Not an exact implementation of the official VaGraM algorithm.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
import argparse
import hashlib
import json
import math
import random

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import RGCNConv,global_mean_pool

from experiments.world_model_integration.decision_learning_pr51.pr51_train import (
    candidate_graph,RELATIONS
)
from experiments.world_model_integration.decision_learning_pr51.pr51_loss import (
    pairwise_decision_loss,gradient_path_present
)

PROTOCOL=Path(__file__).with_name("pr52_protocol.json")
REGIMES=("congestion","multi_fault","staggered_arrivals")
SPLITS=("train","validation","test")
ARMS=("rel_entity_pred_only","rel_entity_joint025","rel_entity_gradweighted",
      "flat_entity_joint025_control")
TYPES=("robot","task","station","resource")
SCALE=32.


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def load_split(root,split):
    if split not in SPLITS:
        raise ValueError("unregistered split")
    groups=[]
    prov=set()
    for regime in REGIMES:
        path=root/regime/"data.json"
        doc=json.loads(path.read_text())
        if (doc["schema"]!="pr52-true-entity-world-state-paired-actions-v1"
            or doc["regime"]!=regime):
            raise ValueError("not original PR52 sources")
        prov.add(doc["protocol_sha256"])
        wanted=[g for g in doc["cases"] if g["split"]==split]
        expected={"train":48,"validation":12,"test":16}[split]
        if len(wanted)!=expected:
            raise AssertionError("source splits differ from predeclared cohort")
        groups.extend(wanted)
    if len(prov)!=1 or len({g["seed"] for g in groups})!=len(groups):
        raise AssertionError("source label fingerprint or split nonindependent")
    return groups


def _prepare_one(case,relational):
    if not case["labelable"] or not case["all_branches_completed"]:
        return None,"not full true-world branch group"
    if not case.get("full_entity_horizon_labels_available",False):
        return None,"future public H5/H20 entity state censored"
    public=case["public_graph"]
    k=case["candidate_count"]
    if len(public["candidate_actions"])!=k or len(case["true_factual_outcomes"])!=k:
        raise ValueError("candidate proposals and labels misaligned")
    graphs=[]
    ys=[];masks=[];events=[];totals=[]
    for idx in range(k):
        graphs.append(candidate_graph(public,idx,relational))
        future=case["true_factual_outcomes"][idx]["future_entity_states"]
        if set(future)!={"5","20"}:
            return None,"future H5 or H20 missing"
        sizes={len(future[h]["node_target"]) for h in ("5","20")}
        if sizes!={len(public["nodes"])}:
            raise AssertionError("task/robot/station node count differs from anchor")
        targets=torch.stack([
            torch.tensor(future[h]["node_target"],dtype=torch.float32)
            for h in ("5","20")
        ],dim=1)
        weights=torch.stack([
            torch.tensor(future[h]["node_target_mask"],dtype=torch.float32)
            for h in ("5","20")
        ],dim=1)
        if targets.shape!=(len(public["nodes"]),2,6) or targets.shape!=weights.shape:
            raise AssertionError("expected two horizons, six features per node")
        if weights.sum()<=0:
            return None,"all nodes masked"
        ys.append(targets);masks.append(weights)
        events.append(torch.tensor([
            [future[h]["global_new_stage_completed_count"]/SCALE,
             future[h]["global_newly_visible_stage_count"]/SCALE]
            for h in ("5","20")],dtype=torch.float32))
        totals.append(torch.tensor([
            future[h]["global_total_completed_stage_count"]/SCALE
            for h in ("5","20")],dtype=torch.float32))
    batch=Batch.from_data_list(graphs)
    if batch.x.shape[0]!=sum(len(x["node_target"]) for x in
                             [case["true_factual_outcomes"][j]["future_entity_states"]["5"]
                              for j in range(k)]):
        raise AssertionError("PyG batch concatenation inconsistent with labels")
    factual=torch.tensor([x["factual_makespan_ticks"]
                          for x in case["true_factual_outcomes"]],dtype=torch.float32)
    return {
        "id":case["id"],"regime":case["regime"],
        "batch":batch,"node_y":torch.cat(ys,dim=0),
        "node_mask":torch.cat(masks,dim=0),
        "event_y":torch.stack(events),
        "total_y":torch.stack(totals),
        "true_cost":factual,
    },None


def prepare(groups,relational,split):
    valid=[];excluded=[]
    for c in groups:
        if c["split"]!=split:
            raise ValueError("train / validation / test mixing")
        sample,reason=_prepare_one(c,relational)
        if sample is None:
            excluded.append({"id":c["id"],"reason":reason})
        else:
            valid.append(sample)
    return valid,excluded


class EntityConditionalRelationalWorld(nn.Module):
    def __init__(self,hidden=48,relational=True):
        super().__init__()
        self.relational=bool(relational)
        self.embedding=nn.Linear(16,hidden)
        self.transition1=RGCNConv(hidden,hidden,num_relations=2*len(RELATIONS),
                                  num_bases=4)
        self.transition2=RGCNConv(hidden,hidden,num_relations=2*len(RELATIONS),
                                  num_bases=4)
        self.node_transition=nn.Sequential(
            nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,12))
        self.new_event_transition=nn.Sequential(
            nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,4))

    def forward(self,batch):
        h=F.relu(self.embedding(batch.x))
        if self.relational and batch.edge_index.numel():
            h=F.relu(self.transition1(h,batch.edge_index,batch.edge_type))
            h=F.relu(self.transition2(h,batch.edge_index,batch.edge_type))
        node_state=torch.sigmoid(self.node_transition(h)).view(-1,2,6)
        future_new_stage_events=torch.sigmoid(self.new_event_transition(
            global_mean_pool(h,batch.batch))).view(-1,2,2)
        return node_state,future_new_stage_events


def total_completed_future(node_state,event_future,batch):
    """Differentiable expected number of completed stages at H5 and H20."""
    k=event_future.shape[0]
    completed=node_state[:,:,3]*batch.x[:,1,None]  # public current task nodes
    state_count=completed.new_zeros(k,2)
    state_count.index_add_(0,batch.batch,completed)
    return state_count+SCALE*event_future[:,:,0]


def decision_cost(node_state,event_future,batch):
    # The ORIGINAL PR30 criterion: higher expected H20 completed stages wins.
    return -total_completed_future(node_state,event_future,batch)[:,1]


def entity_losses(node_state,event_future,g):
    node_y=g["node_y"].to(node_state.device)
    mask=g["node_mask"].to(node_state.device)
    event_y=g["event_y"].to(node_state.device)
    total_y=g["total_y"].to(node_state.device)
    node_mse=((node_state-node_y).square()*mask).sum()/mask.sum().clamp(min=1.)
    event_mse=F.mse_loss(event_future,event_y)
    expected_total=total_completed_future(node_state,event_future,g["batch"])
    total_mse=F.mse_loss(expected_total/SCALE,total_y)
    return {
        "node_mse":node_mse,"new_stage_event_mse":event_mse,
        "total_completed_mse":total_mse,
        "prediction":node_mse+0.3*event_mse+0.5*total_mse
    }


def fixed_completion_sensitivity_loss(node_state,event_future,g):
    """Value-gradient-inspired, not the exact VaGraM paper implementation.

    Use the FROZEN H20 completion-value derivative to weight model errors
    for action-relevant state channels. Sensitivity does not come from a
    trained Q/ranker and does not see true future scores.
    """
    y=g["node_y"].to(node_state.device)
    m=g["node_mask"].to(node_state.device)
    ey=g["event_y"].to(node_state.device)
    cost=decision_cost(node_state,event_future,g["batch"])
    gn,ge=torch.autograd.grad(
        cost.sum(),(node_state,event_future),retain_graph=True)
    weighted_nodes=(gn.detach().square()*m*(node_state-y).square()).sum()
    weighted_events=((ge.detach()/SCALE).square()*(event_future-ey).square()).sum()
    count=(gn.detach().abs()*m).gt(0).sum()+((ge.detach()/SCALE).abs()>0).sum()
    return (weighted_nodes+weighted_events)/count.clamp(min=1)


def loss_arm(model,g,arm,device):
    batch=g["batch"].to(device)
    node,event=model(batch)
    base=entity_losses(node,event,g)
    loss=base["prediction"]
    dec=pairwise_decision_loss(decision_cost(node,event,batch),
                               g["true_cost"].to(device))
    if arm["lambda_decision"]:
        loss=loss+arm["lambda_decision"]*dec
    gradient_sensitive=node.new_tensor(0.)
    if arm["sensitivity_weight"]:
        gradient_sensitive=fixed_completion_sensitivity_loss(node,event,g)
        loss=loss+arm["sensitivity_weight"]*gradient_sensitive
    return loss,dec,gradient_sensitive,base


def evaluate(model,data,device):
    if not data:
        raise ValueError("empty set")
    model.eval()
    totals={"groups":len(data),"node_mse":0.,"station_device_brier":0.,
            "station_device_count":0,"mean_true_candidate_regret_ticks":0.,
            "first_cpsat_regret_ticks":0.,"total_completion_mae":0.,
            "pairwise_correct":0,"pairwise_count":0,
            "chosen_first":0}
    regime={}
    with torch.no_grad():
        for g in data:
            b=g["batch"].to(device)
            n,e=model(b)
            y=g["node_y"].to(device)
            mask=g["node_mask"].to(device)
            losses=entity_losses(n,e,g)
            totals["node_mse"]+=float(losses["node_mse"])
            predicted_total=total_completed_future(n,e,b)
            totals["total_completion_mae"]+=float(
                (predicted_total/SCALE-g["total_y"].to(device)).abs().mean()*SCALE)
            station_mask=(b.x[:,2,None]*mask[:,:,0])
            device_error=(n[:,:,0]-y[:,:,0]).square()*station_mask
            totals["station_device_brier"]+=float(device_error.sum())
            totals["station_device_count"]+=int(station_mask.sum())
            costs=decision_cost(n,e,b).detach().cpu().tolist()
            factual=g["true_cost"].tolist()
            choice=min(range(len(costs)),key=lambda i:(costs[i],i))
            baseline=min(factual)
            regret=factual[choice]-baseline
            totals["mean_true_candidate_regret_ticks"]+=regret
            totals["first_cpsat_regret_ticks"]+=factual[0]-baseline
            totals["chosen_first"]+=int(choice==0)
            regime.setdefault(g["regime"],[]).append(regret)
            for i,j in combinations(range(len(costs)),2):
                if factual[i]==factual[j]:
                    continue
                totals["pairwise_count"]+=1
                if (costs[i]-costs[j])*(factual[i]-factual[j])>0:
                    totals["pairwise_correct"]+=1
    for key in ("node_mse","mean_true_candidate_regret_ticks",
                "first_cpsat_regret_ticks","total_completion_mae"):
        totals[key]/=len(data)
    totals["station_device_brier"]/=max(1,totals["station_device_count"])
    totals["pairwise_accuracy"]=(totals["pairwise_correct"]/totals["pairwise_count"]
                                 if totals["pairwise_count"] else None)
    totals["by_regime_regret_ticks"]={k:sum(v)/len(v) for k,v in regime.items()}
    return totals


def train_one(arm,seed,train,valid,cfg,out,device):
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)
    model=EntityConditionalRelationalWorld(
        cfg["training"]["hidden_size"],arm["relations"]).to(device)
    optim=torch.optim.Adam(model.parameters(),lr=cfg["training"]["learning_rate"])
    verified=False
    history=[]
    for epoch in range(cfg["training"]["epochs"]):
        order=list(range(len(train)))
        random.Random(seed+epoch).shuffle(order)
        model.train()
        total=0.
        for index in order:
            g=train[index]
            optim.zero_grad(set_to_none=True)
            loss,_,_,_=loss_arm(model,g,arm,device)
            loss.backward()
            if (not verified and arm["relations"]
                and (arm["lambda_decision"]>0 or arm["sensitivity_weight"]>0)
                and len(set(g["true_cost"].tolist()))>1):
                optim.zero_grad(set_to_none=True)
                b=g["batch"].to(device)
                n,e=model(b)
                active=(pairwise_decision_loss(decision_cost(n,e,b),
                        g["true_cost"].to(device)) if arm["lambda_decision"]>0
                        else fixed_completion_sensitivity_loss(n,e,g))
                active.backward()
                verified=gradient_path_present(model.transition1.parameters())
                if not verified:
                    raise AssertionError("value/decision gradient does not update shared relational dynamics")
                optim.zero_grad(set_to_none=True)
                loss,_,_,_=loss_arm(model,g,arm,device)
                loss.backward()
            optim.step()
            total+=float(loss)
        if epoch in (0,cfg["training"]["epochs"]-1):
            history.append({"epoch":epoch+1,"mean_total_loss":total/len(train),
                            "validation":evaluate(model,valid,device)})
    checkpoint=out/(arm["id"]+"_seed"+str(seed)+".pt")
    torch.save({
        "model":model.state_dict(),"optimizer":optim.state_dict(),
        "torch_rng":torch.get_rng_state(),"arm":arm,"seed":seed,
        "trained_epochs":cfg["training"]["epochs"],
        "shared_dynamics_gradient_verified":verified,
        "model_type":"official_PyG_RGCNConv_entity_future_state_H5_H20",
        "fixed_decision_rule":"-H20 expected completed stages derived from predicted public future"
    },checkpoint)
    return model,{
        "arm":arm["id"],"seed":seed,
        "trained_epochs":cfg["training"]["epochs"],
        "gradient_reaches_shared_relation_dynamics":verified,
        "train":evaluate(model,train,device),
        "validation":evaluate(model,valid,device),
        "history":history,
        "checkpoint":checkpoint.name
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args()
    if args.out.exists():
        raise FileExistsError("never overwrite actual research checkpoint")
    cfg=json.loads(PROTOCOL.read_text())
    if cfg["protocol"]!="PR52_ENTITY_CONDITIONAL_DECISION_TRAIN_20261112_V1":
        raise ValueError("not PR52 model protocol")
    args.out.mkdir(parents=True)
    train=load_split(args.input,"train")
    valid=load_split(args.input,"validation")
    if {g["seed"] for g in train}&{g["seed"] for g in valid}:
        raise AssertionError("train vs validation not isolated")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report={}
    for arm in cfg["train_arms"]:
        a=arm["id"]
        training,train_rejected=prepare(train,arm["relations"],"train")
        validation,valid_rejected=prepare(valid,arm["relations"],"validation")
        if not training or not validation:
            raise AssertionError("node supervision missing")
        report[a]={}
        for seed in cfg["training"]["seeds"]:
            net,summary=train_one(arm,seed,training,validation,cfg,args.out,device)
            summary["rejected_train"]=train_rejected
            summary["rejected_validation"]=valid_rejected
            report[a][str(seed)]=summary
            print("PR52_TRAIN_FINISHED",a,seed,json.dumps(
                summary["validation"],ensure_ascii=False),flush=True)
    # Explicit model choice uses ONLY validation, averaging same predeclared seeds.
    best=min(ARMS,key=lambda a:(
        sum(report[a][str(s)]["validation"]["mean_true_candidate_regret_ticks"]
            for s in cfg["training"]["seeds"])/len(cfg["training"]["seeds"]),
        sum(report[a][str(s)]["validation"]["node_mse"]
            for s in cfg["training"]["seeds"])/len(cfg["training"]["seeds"]),
        ARMS.index(a)
    ))
    selection={
        "schema":"pr52-validation-only-multi-seed-world-model-selection-v1",
        "selected_arm":best,
        "selection_unit":"36 independent scenario seeds, 2 training seeds per arm",
        "all_validation":{a:{str(s):report[a][str(s)]["validation"]
                             for s in cfg["training"]["seeds"]} for a in ARMS},
        "all_training":{a:{str(s):report[a][str(s)]["train"]
                           for s in cfg["training"]["seeds"]} for a in ARMS},
        "independent_test_not_read_during_training_or_selection":True,
        "gpu_training_used":device.type=="cuda",
        "predeclared_decision_rule":cfg["ranking_rule"]
    }
    (args.out/"selection.json").write_text(json.dumps(selection,indent=2,ensure_ascii=False)+"\n")
    # Strict: open TEST source only after selection artifacts exist.
    test=load_split(args.input,"test")
    all_seen={g["seed"] for g in train+valid}
    if all_seen & {g["seed"] for g in test}:
        raise AssertionError("TEST duplicates train/validation")
    results={}
    for arm in cfg["train_arms"]:
        a=arm["id"]
        data,rejected=prepare(test,arm["relations"],"test")
        if not data:
            raise AssertionError("test entity labels missing")
        for seed in cfg["training"]["seeds"]:
            ckpt=torch.load(args.out/(a+"_seed"+str(seed)+".pt"),
                            map_location=device,weights_only=False)
            net=EntityConditionalRelationalWorld(cfg["training"]["hidden_size"],
                                                arm["relations"]).to(device)
            net.load_state_dict(ckpt["model"])
            results.setdefault(a,{})[str(seed)]={
                "metrics":evaluate(net,data,device),"excluded":rejected
            }
    summary={
        "schema":"pr52-trained-entity-conditional-world-model-independent-test-v1",
        "models_trained":len(cfg["train_arms"])*len(cfg["training"]["seeds"]),
        "selected_arm_validation_only":best,
        "selection":selection,"independent_test":results,
        "splits":{"train":len(train),"validation":len(valid),"test":len(test)},
        "gpu_training_used":device.type=="cuda",
        "new_full_online_closed_loop_evaluation":False,
        "scope":"True entity H5/H20 forecast and first-action true candidate regret; fixed current-map synthetic world, not yet recurrent transition or business deployment"
    }
    (args.out/"result.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
    print("PR52_INDEPENDENT_ENTITY_MODEL_TEST",json.dumps({
        "selection":best,
        "test":{a:{s:r["metrics"] for s,r in d.items()} for a,d in results.items()},
        "grad":{a:{s:report[a][s]["gradient_reaches_shared_relation_dynamics"]
                   for s in report[a]} for a in report},
        "gpu_training":device.type=="cuda"
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
