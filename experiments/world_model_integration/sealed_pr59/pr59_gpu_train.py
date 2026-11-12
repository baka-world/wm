"""PR59 CUDA-ONLY reproducible GPU-agent training entrypoint (NOT executed here).

It trains learned t->t+1 PUBLIC map + relational world dynamics; never a
standalone Q or task ranking network. Every true sample originates in a
SHA-verified complete BusinessSimulator trajectory and PR56 public map.

Strictly NO test source parameter exists in this script. Unopened OOD
topology-family TEST must only be generated/used after model selection is
committed, in an entirely separate stage by the user's GPU Agent.

This file is part of the GPU handoff. CPU GitHub Actions execute only
unit/compilation checks, never training.
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

from experiments.world_model_integration.sealed_pr59.pr59_dual_model import (
    PublicMapTemporalRelationalWorld,actual_public_record_to_graph,
    supervised_losses,parameter_count
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_actual_source(paths,expected):
    if expected not in ("train","validation") or not paths:
        raise ValueError("No BLIND/TEST mode exists in GPU trainer")
    episodes=[]
    ids=set()
    sources={}
    for path in paths:
        src=json.loads(path.read_text())
        if src.get("split")!=expected or src.get("BLIND_TEST_RUN") is not False:
            raise AssertionError("wrong or BLIND source supplied to training")
        if src.get("schema") not in (
            "pr59-sha-sealed-natural-public-world-map-family-split-v1",
            "pr60-additional-real-public-map-train-source-v1"):
            raise ValueError("unsupported true BusinessSimulator source")
        sources[str(path.resolve())]=sha256(path)
        for scene in src["scenes"]:
            case=scene["case"]
            if (case["split"]!=expected or scene["id"] in ids
                or not scene["raw_actual_public_world_transitions"]):
                raise ValueError("duplicate/no source or leaked wrong split")
            ids.add(scene["id"])
            maps=[row["input"].get("public_map",{}).get("static_map_digest")
                  for row in scene["raw_actual_public_world_transitions"]]
            if len(set(maps))!=1 or not all(maps):
                raise AssertionError("no constant public map observation in real episode")
            graphs=[actual_public_record_to_graph(row)
                    for row in scene["raw_actual_public_world_transitions"]]
            episodes.append({"id":scene["id"],
                             "family":case["map_family"],"regime":case["regime"],
                             "seed":case["seed"],"graphs":graphs,
                             "censored":scene["real_full_episode_status"]!="completed"})
    if not episodes:
        raise ValueError("no authentic world-model examples")
    return episodes,sources


def evaluate(model,episodes,device,batch_size):
    model.eval()
    rows=[]
    with torch.no_grad():
        for e in episodes:
            err=0.;valid=0.
            brier=0.;n_station=0
            new_visible=0.;n_trans=0
            for batch in DataLoader(e["graphs"],batch_size=batch_size,shuffle=False):
                b=batch.to(device)
                n,pred_events=model(b)
                masked=(n-b.node_target).square()*b.target_mask
                err+=float(masked.sum().item())
                valid+=float(b.target_mask.sum().item())
                stations=b.x[:,2,None]*b.target_mask[:,0,None]
                brier+=float((((n[:,0,None]-b.node_target[:,0,None]).square())*
                              stations).sum().item())
                n_station+=float(stations.sum().item())
                new_visible+=float((pred_events[:,1]-b.event_target[:,1]).abs().sum().item())
                n_trans+=int(pred_events.shape[0])
            if not valid or not n_station or not n_trans:
                raise AssertionError("source has no real device/public targets")
            rows.append({
                "id":e["id"],"family":e["family"],"regime":e["regime"],
                "true_time_right_censored":e["censored"],
                "n_real_steps":n_trans,
                "node_mse":err/valid,
                "device_brier":brier/n_station,
                "next_new_public_stage_mae":new_visible/n_trans*32.
            })
    return {
        "macro_node_mse":sum(x["node_mse"] for x in rows)/len(rows),
        "macro_device_brier":sum(x["device_brier"] for x in rows)/len(rows),
        "macro_new_task_mae":sum(x["next_new_public_stage_mae"] for x in rows)/len(rows),
        "n_independent_scenarios":len(rows),
        "all_raw_public_transitions":sum(x["n_real_steps"] for x in rows),
        "per_real_independent_scenario":rows
    }


def save_checkpoint(path,model,optimizer,epoch,seed,arm,width,source_sha,val,source_head):
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({
        "schema":"pr59-public-map-world-actual-gpu-checkpoint-v1",
        "architecture":"real PyG SAGEConv public map / PR54 GRU-RGCN relational state transition",
        "epoch":epoch,"seed":seed,"arm":arm,"hidden":width,
        "model_state_dict":model.state_dict(),
        "optimizer_state_dict":optimizer.state_dict(),
        "torch_rng_state":torch.get_rng_state(),
        "cuda_rng_state_all":torch.cuda.get_rng_state_all(),
        "source_sha256":source_sha,
        "git_head_source":source_head,
        "frozen_validation_metrics":val,
        "train_map_vs_masked_same_parameter_count":True
    },path)


def train_one(args,train,validation,sources,seed,arm,out):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.set_num_threads(args.cpu_threads)
    enabled=arm=="map_on"
    history=arm!="snapshot_map_off"
    model=PublicMapTemporalRelationalWorld(
        hidden=args.hidden,history_enabled=history,map_enabled=enabled).to("cuda")
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    raw=[g for ep in train for g in ep["graphs"]]
    if len(raw)<50:
        raise AssertionError("actual training source too small")
    seed_root=out/arm/str(seed)
    seed_root.mkdir(parents=True,exist_ok=False)
    best=float("inf")
    per_epoch=[]
    for epoch in range(args.epochs):
        model.train()
        running=0.;steps=0
        generator=torch.Generator().manual_seed(seed*100000+epoch)
        loader=DataLoader(raw,batch_size=args.batch_size,shuffle=True,generator=generator)
        for batch in loader:
            b=batch.to("cuda")
            optimizer.zero_grad(set_to_none=True)
            nodes,events=model(b)
            losses=supervised_losses(nodes,events,b)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),args.max_grad_norm)
            optimizer.step()
            running+=float(losses["total"].detach().item())
            steps+=1
        val=evaluate(model,validation,"cuda",args.batch_size)
        # Validity gate: decide by NEW validation ONLY. Keep explicit device
        # probability calibration alongside entity MSE, not just aggregate.
        score=val["macro_node_mse"]+args.device_brier_selection_weight*val["macro_device_brier"]
        item={"epoch":epoch,"training_loss_batch_avg":running/max(1,steps),
              "validation":{k:v for k,v in val.items() if k!="per_real_independent_scenario"},
              "selection_score":score}
        per_epoch.append(item)
        source_head=args.git_head
        if score<best:
            best=score
            save_checkpoint(seed_root/"best.pt",model,optimizer,epoch,
                            seed,arm,args.hidden,sources,val,source_head)
        save_checkpoint(seed_root/"last.pt",model,optimizer,epoch,
                        seed,arm,args.hidden,sources,val,source_head)
        print("PR59_GPU_WORLD_FIT_EPOCH",json.dumps({
            "arm":arm,"seed":seed,"param_count":parameter_count(model),**item
        }),flush=True)
    final={
        "arm":arm,"seed":seed,"hidden":args.hidden,
        "model_parameters":parameter_count(model),
        "best_selection_score":best,
        "per_epoch_validation_only":per_epoch,
        "peak_memory_allocated_bytes":torch.cuda.max_memory_allocated(),
        "true_source_sha256":sources
    }
    (seed_root/"train_report.json").write_text(json.dumps(final,indent=2)+"\n")
    return final


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train",required=True,type=Path,nargs="+")
    ap.add_argument("--validation",required=True,type=Path)
    ap.add_argument("--output",required=True,type=Path)
    ap.add_argument("--git-head",required=True)
    ap.add_argument("--hidden",type=int,default=48,choices=(48,96,128))
    ap.add_argument("--epochs",type=int,default=20)
    ap.add_argument("--batch-size",type=int,default=24)
    ap.add_argument("--lr",type=float,default=0.0003)
    ap.add_argument("--weight-decay",type=float,default=0.0001)
    ap.add_argument("--device-brier-selection-weight",type=float,default=0.25)
    ap.add_argument("--max-grad-norm",type=float,default=1.)
    ap.add_argument("--cpu-threads",type=int,default=2)
    ap.add_argument("--seeds",default="1810,1811,1812")
    ap.add_argument("--arms",default="map_off,map_on")
    args=ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This is the user's GPU-Agent training entrypoint: real CUDA GPU required; do NOT silently fall back to CPU")
    if (args.epochs<1 or args.batch_size<1 or args.lr<=0
        or args.device_brier_selection_weight<0):
        raise ValueError("invalid GPU training budget/config")
    seeds=[int(x) for x in args.seeds.split(",")]
    arms=args.arms.split(",")
    if (len(seeds)!=len(set(seeds)) or any(
        x not in ("map_off","map_on","snapshot_map_off") for x in arms)):
        raise ValueError("invalid controlled random seeds or model arm")
    if args.output.exists():
        raise FileExistsError("never overwrite source/last checkpoint archive")
    args.output.mkdir(parents=True)
    train,s1=load_actual_source(args.train,"train")
    val,s2=load_actual_source([args.validation],"validation")
    if {e["seed"] for e in train}&{e["seed"] for e in val}:
        raise AssertionError("TRAIN and validation source scenarios overlap")
    train_maps={e["family"] for e in train}
    val_maps={e["family"] for e in val}
    if train_maps&val_maps or train_maps!={"vertical_blocked_route_training"} or val_maps!={"diagonal_staggered_validation"}:
        raise AssertionError("train/validation MAP FAMILY contamination")
    source_sha={"train":s1,"validation":s2}
    results=[]
    started=time.time()
    for arm in arms:
        for seed in seeds:
            results.append(train_one(args,train,val,source_sha,seed,arm,args.output))
    means={arm:{
        "validation_score_avg_seeds":sum(x["best_selection_score"] for x in results if x["arm"]==arm)/len(seeds),
        "parameters":next(x["model_parameters"] for x in results if x["arm"]==arm)
    } for arm in arms}
    if "map_on" in means and "map_off" in means:
        if means["map_on"]["parameters"]!=means["map_off"]["parameters"]:
            raise AssertionError("capacity confound between map controls")
    # Weights are frozen after VALIDATION selection; GPU Agent must review
    # device Brier and state error before promotion.
    eligible=sorted(means,key=lambda k:means[k]["validation_score_avg_seeds"])
    select={
        "schema":"pr59-locked-validation-public-map-world-model-selection-v1",
        "training_scenes":len(train),"validation_scenes":len(val),
        "BLIND_TEST_FILES_READ":0,
        "BLIND_TEST_EPISODES_GENERATED":0,
        "device":"CUDA","duration_wall_seconds":time.time()-started,
        "frozen_eval_criterion":"validation macro node masked MSE + fixed device-brier selection weight",
        "dev_brier_selection_weight":args.device_brier_selection_weight,
        "pre_selected_by_validation_only":eligible[0],
        "controlled_arms":means,
        "all_training_seed_checkpoints":results,
        "source_sha256":source_sha,
        "resume_requires_same_git_head":args.git_head,
        "limitation":"teacher-forced true t->t+1 state training; no autoregressive H20 or online CP-SAT benefit proven"
    }
    (args.output/"selection.json").write_text(json.dumps(select,indent=2)+"\n")
    print("PR59_GPU_AGENT_VALIDATION_SELECTION_FROZEN",json.dumps({
        k:v for k,v in select.items() if k!="all_training_seed_checkpoints"
    }),flush=True)


if __name__=="__main__":
    main()
