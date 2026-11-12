"""PR54 train/validation/fresh-test of REAL public temporal business world model.

Samples come from actual simulator consecutive public transition callbacks.
Independence unit is a SCENARIO, not one of its correlated timestep pairs.
True future public observations are labels only. Read the physically sealed
TEST file only AFTER all validation decisions/checkpoints are frozen.

Two arms have identical GRU/RGCN parameters: for snapshot-only, the
historical inputs are zeroed except current t. No optimizer on TEST.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import hashlib
import json
import random
import time

import torch
from torch_geometric.loader import DataLoader
from experiments.world_model_integration.decision_learning_pr51.pr51_train import candidate_graph
from experiments.world_model_integration.temporal_pr54.pr54_model import (
    PublicTemporalRelationalWorld,supervised_losses
)

PROTOCOL=Path(__file__).with_name("protocol.json")
ARMS=("snapshot_only","public_history_gru")
SPLITS=("train","validation","test")
SCALE=32.


def load_split(source,split):
    """Only ONE physical split file is opened; missing TEST fails explicitly."""
    if split not in SPLITS:
        raise ValueError("unregistered cohort")
    p=source/(split+".json")
    obj=json.loads(p.read_text())
    if (obj.get("schema")!="pr54-physically-isolated-temporal-world-split-v1"
        or obj.get("split")!=split):
        raise ValueError("not PR54 physically isolated simulator episodes")
    episodes=obj["episodes"]
    n={"train":36,"validation":12,"test":12}[split]
    if len(episodes)!=n or len({e["seed"] for e in episodes})!=n:
        raise ValueError("source episodes incomplete or have duplicate seeds")
    if any(e["split"]!=split for e in episodes):
        raise ValueError("cross-split source leakage")
    for regime in ("congestion","multi_fault","staggered_arrivals"):
        if sum(e["regime"]==regime for e in episodes)!=n//3:
            raise ValueError("unexpected regime balance")
    return episodes


def example_to_graph(record):
    inp=record["input"]
    label=record["target"]
    if label["next_public_tick"]!=inp["public_tick"]+1:
        raise ValueError("future-one-step label is not one tick after input")
    if label["next_public_tick"]<=inp["feature_last_observed_tick"]:
        raise ValueError("future label alignment is corrupt")
    graph=inp["graph"]
    data=candidate_graph(graph,0,True)
    n=data.num_nodes
    h=torch.tensor(inp["history"],dtype=torch.float32)
    hm=torch.tensor(inp["history_visible_mask"],dtype=torch.float32)
    yt=torch.tensor(label["node_target"],dtype=torch.float32)
    mask=torch.tensor(label["node_mask"],dtype=torch.float32)
    persistence=torch.tensor(inp["node_persistence"],dtype=torch.float32)
    if (h.shape!=(n,4,16) or hm.shape!=(n,4)
        or yt.shape!=(n,6) or mask.shape!=(n,6)
        or persistence.shape!=(n,6)):
        raise AssertionError("node temporal data dimensions invalid")
    if not torch.all(hm[:,-1]==1):
        raise AssertionError("current graph disappeared from time window")
    if not torch.all(h[:,-1,:]==data.x):
        raise AssertionError("latest history snapshot mismatches true current candidate graph")
    if not torch.all((hm==0)|(hm==1)):
        raise AssertionError("invalid public historical visibility mask")
    if torch.any(h[:,:-1][hm[:,:-1]==0]!=0):
        raise AssertionError("nonexistent historical node has future signal")
    data.history=h
    data.history_mask=hm
    data.node_target=yt
    data.target_mask=mask
    data.node_persistence=persistence
    data.event_target=torch.tensor([[
        label["new_completed_task_stages"]/SCALE,
        label["new_visible_task_stages"]/SCALE
    ]],dtype=torch.float32)
    if data.event_target[0,0]>data.event_target[0,1]:
        raise AssertionError("source future counts violate task lifecycle")
    return data


def prepare(episodes):
    result=[]
    for e in episodes:
        graphs=[example_to_graph(v) for v in e["transitions"]]
        if not graphs:
            raise AssertionError("empty REAL public transition episode")
        result.append({
            "id":e["id"],"seed":e["seed"],"regime":e["regime"],"graphs":graphs
        })
    return result


def evaluate(net,episodes,device):
    net.eval()
    groups=[]
    with torch.no_grad():
        for e in episodes:
            node_sum=0.;node_fields=0
            persistence_sum=0.
            station_sum=0.;station_fields=0
            persist_station_sum=0.
            event_abs_sum=0.;n_graph=0
            for batch in DataLoader(e["graphs"],batch_size=32,shuffle=False):
                b=batch.to(device)
                n,event=net(b)
                err=(n-b.node_target).square()*b.target_mask
                pers=(b.node_persistence-b.node_target).square()*b.target_mask
                node_sum+=float(err.sum())
                persistence_sum+=float(pers.sum())
                node_fields+=int(b.target_mask.sum())
                station_mask=b.x[:,2,None]*b.target_mask[:,0,None]
                station_sum+=float(((n[:,0,None]-b.node_target[:,0,None]).square()
                                   *station_mask).sum())
                persist_station_sum+=float((
                    (b.node_persistence[:,0,None]-b.node_target[:,0,None]).square()
                    *station_mask).sum())
                station_fields+=int(station_mask.sum())
                event_abs_sum+=float((event[:,1]-b.event_target[:,1]).abs().sum()*SCALE)
                n_graph+=event.shape[0]
            if node_fields==0 or station_fields==0 or n_graph==0:
                raise AssertionError("missing public node or station truth")
            groups.append({
                "id":e["id"],"regime":e["regime"],"n_transition_samples":n_graph,
                "node_masked_mse":node_sum/node_fields,
                "persistence_node_mse":persistence_sum/node_fields,
                "device_brier":station_sum/station_fields,
                "persistence_device_brier":persist_station_sum/station_fields,
                "new_visible_task_count_mae":event_abs_sum/n_graph
            })
    report={
        "independent_scenarios":len(groups),
        "real_public_transitions":sum(x["n_transition_samples"] for x in groups),
        "scenario_macro_entity_mse":sum(x["node_masked_mse"] for x in groups)/len(groups),
        "scenario_macro_persistence_mse":sum(x["persistence_node_mse"] for x in groups)/len(groups),
        "scenario_macro_device_brier":sum(x["device_brier"] for x in groups)/len(groups),
        "scenario_macro_persistence_device_brier":sum(x["persistence_device_brier"] for x in groups)/len(groups),
        "scenario_macro_new_visible_count_mae":sum(x["new_visible_task_count_mae"] for x in groups)/len(groups),
        "by_regime":{},
        "per_scenario":groups
    }
    for r in ("congestion","multi_fault","staggered_arrivals"):
        subgroup=[x for x in groups if x["regime"]==r]
        report["by_regime"][r]={
            "scenes":len(subgroup),
            "node_mse":sum(x["node_masked_mse"] for x in subgroup)/len(subgroup),
            "persistence_mse":sum(x["persistence_node_mse"] for x in subgroup)/len(subgroup),
            "device_brier":sum(x["device_brier"] for x in subgroup)/len(subgroup)}
    return report


def train_arm(cfg,arm,seed,train_episodes,val_episodes,out,device):
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)
    net=PublicTemporalRelationalWorld(
        hidden=cfg["hidden_size"],history_enabled=(arm=="public_history_gru")).to(device)
    optim=torch.optim.Adam(net.parameters(),lr=cfg["learning_rate"])
    samples=[g for episode in train_episodes for g in episode["graphs"]]
    if len(samples)<50:
        raise ValueError("insufficient true one-step public trajectories")
    best_loss=float("inf")
    best_epoch=None
    history=[]
    path=out/(arm+"_seed"+str(seed)+"_best.pt")
    for epoch in range(cfg["epochs"]):
        net.train()
        generator=torch.Generator().manual_seed(seed*1000+epoch)
        loader=DataLoader(samples,batch_size=32,shuffle=True,generator=generator)
        epoch_loss=0.;steps=0
        for batch in loader:
            b=batch.to(device)
            node,event=net(b)
            losses=supervised_losses(node,event,b)
            optim.zero_grad(set_to_none=True)
            losses["total"].backward()
            optim.step()
            epoch_loss+=float(losses["total"].detach())
            steps+=1
        # Validation, never training samples, picks the best epoch.
        check=evaluate(net,val_episodes,device)
        ranking=(check["scenario_macro_entity_mse"]+
                 0.25*check["scenario_macro_device_brier"])
        history.append({"epoch":epoch+1,"training_batch_loss":epoch_loss/steps,
                        "val_macro_node_mse":check["scenario_macro_entity_mse"],
                        "val_macro_device_brier":check["scenario_macro_device_brier"]})
        if ranking<best_loss:
            best_loss=ranking
            best_epoch=epoch+1
            torch.save({
                "model":net.state_dict(),
                "optimizer":optim.state_dict(),
                "torch_rng":torch.get_rng_state(),
                "seed":seed,"arm":arm,"epoch":best_epoch,
                "history_window":cfg["history_window"],
                "source":"pr54 real public t-3..t -> next-world-state",
                "parameter_count":sum(x.numel() for x in net.parameters())
            },path)
    torch.save({
        "model":net.state_dict(),"optimizer":optim.state_dict(),
        "torch_rng":torch.get_rng_state(),"arm":arm,"seed":seed,
        "epoch":cfg["epochs"],"last_not_best":True
    },out/(arm+"_seed"+str(seed)+"_last.pt"))
    best=torch.load(path,map_location=device,weights_only=False)
    net.load_state_dict(best["model"])
    return {
        "arm":arm,"seed":seed,"best_epoch":best_epoch,
        "best_val_macro_weighted_loss":best_loss,
        "parameter_count":best["parameter_count"],
        "validation_best":evaluate(net,val_episodes,device),
        "history":history,"best_checkpoint":path.name,
        "last_checkpoint":arm+"_seed"+str(seed)+"_last.pt"
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    opt=parser.parse_args()
    if opt.out.exists():
        raise FileExistsError("no overwrite of original trained checkpoint")
    cfg=json.loads(PROTOCOL.read_text())
    if cfg["protocol"]!="PR54_TEMPORAL_ACTION_RELATIONAL_NEXT_STATE_CPU_PREFLIGHT_20261112_V1":
        raise ValueError("changed training experiment protocol")
    if opt.device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("real CUDA device required; never pretend CPU is GPU")
    dev=torch.device(opt.device)
    opt.out.mkdir(parents=True)
    # Neither hidden TEST labels nor the test file is opened until selection.
    train=prepare(load_split(opt.input,"train"))
    valid=prepare(load_split(opt.input,"validation"))
    if {g["seed"] for g in train}&{g["seed"] for g in valid}:
        raise AssertionError("independent episodes repeated in train & validation")
    fit={}
    for arm in ARMS:
        fit[arm]={}
        for seed in cfg["train_seeds"]:
            item=train_arm(cfg,arm,seed,train,valid,opt.out,dev)
            fit[arm][str(seed)]=item
            v=item["validation_best"]
            print("PR54_REAL_ONE_STEP_TRAIN_FINISHED",arm,seed,
                  "node_mse",v["scenario_macro_entity_mse"],
                  "device_brier",v["scenario_macro_device_brier"],flush=True)
    selected=min(ARMS,key=lambda name:(
        sum(fit[name][str(s)]["validation_best"]["scenario_macro_entity_mse"]
            for s in cfg["train_seeds"])/len(cfg["train_seeds"]),
        sum(fit[name][str(s)]["validation_best"]["scenario_macro_device_brier"]
            for s in cfg["train_seeds"])/len(cfg["train_seeds"]),
        ARMS.index(name)))
    selection={
        "schema":"pr54-locked-validation-selected-temporal-world-model-v1",
        "selected_arm":selected,
        "selection_unit":"12 independent business simulator episodes and 2 training seeds",
        "all_train_val":fit,
        "test_was_not_read_before_selection":True,
        "device":dev.type,
        "model_limit":"causal teacher-forced one-step, not autoregressive H20"
    }
    (opt.out/"selection.json").write_text(
        json.dumps(selection,ensure_ascii=False,indent=2)+"\n")
    print("PR54_VALIDATION_SELECTION_FROZEN",selected,flush=True)
    unseen=prepare(load_split(opt.input,"test"))
    if {e["seed"] for e in unseen}&{e["seed"] for e in train+valid}:
        raise AssertionError("test episodes overlap data used to fit the model")
    tests={}
    for arm in ARMS:
        tests[arm]={}
        for seed in cfg["train_seeds"]:
            f=opt.out/(arm+"_seed"+str(seed)+"_best.pt")
            ckpt=torch.load(f,map_location=dev,weights_only=False)
            net=PublicTemporalRelationalWorld(
                cfg["hidden_size"],history_enabled=(arm=="public_history_gru")).to(dev)
            net.load_state_dict(ckpt["model"])
            tests[arm][str(seed)]=evaluate(net,unseen,dev)
    final={
        "schema":"pr54-new-one-step-public-history-independent-test-v1",
        "selected_before_test":selected,"selection":selection,
        "splits":{"train":36,"validation":12,"test":12},
        "test":tests,"trained_models":len(ARMS)*len(cfg["train_seeds"]),
        "real_GPU_training":dev.type=="cuda",
        "new_online_rollout_evaluation":False,
        "limits":"One-step public entity forecast with chronological observed histories only, not executable stochastic transition or decision-aware learned model"
    }
    (opt.out/"result.json").write_text(json.dumps(final,ensure_ascii=False,indent=2)+"\n")
    print("PR54_INDEPENDENT_TEMPORAL_WORLD_RESULT",json.dumps({
        "validation_selected":selected,
        "test":{k:{s:{
            "node_mse":v["scenario_macro_entity_mse"],
            "device_brier":v["scenario_macro_device_brier"],
            "persistence_mse":v["scenario_macro_persistence_mse"],
            "transitions":v["real_public_transitions"]
        } for s,v in seeds.items()} for k,seeds in tests.items()},
        "device":dev.type
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
