"""PR31 V4 warm-start vs scratch training using upstream PyTorch Adam/BCE.

Architecture/optimizer/loss all reuse original train_v4.py. Every training
checkpoint archives model, Adam state, Torch/Python/NumPy RNG and epoch counters.
A weight warm-start from PR18 V4 is NOT claimed to resume the original Adam.
Selection only by validation Brier; test is created/opened afterward.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from business_transport.session_events import QUERY_LABEL_VERSION
from business_transport.session_features_v3 import INPUT_SCHEMA_V3,QUERY_WIDTH_V3
from business_transport.state_features import SCHEMA_VERSION
from lab_transport.business_session_learning import BusinessSessionEventModel
from lab_transport.business_session_v3 import encode_rows,probs,stats
from experiments.world_model_integration.train_v4 import load_v4,VERSION
from experiments.world_model_integration.posttraining.pr31_dataset import PROTOCOL
from experiments.world_model_integration.posttraining.repro_environment import sha


def rng_snapshot():
    n=np.random.get_state()
    return {
        "torch_cpu":torch.get_rng_state().clone(),
        "python_random":random.getstate(),
        "numpy_random":{
            "bit_generator":n[0],"state":n[1].tolist(),
            "position":int(n[2]),"has_gauss":int(n[3]),"cached_gauss":float(n[4]),
        },
        "torch_cuda":[x.clone() for x in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available() else [],
    }


def restore_rng(s):
    torch.set_rng_state(s["torch_cpu"])
    random.setstate(s["python_random"])
    n=s["numpy_random"]
    np.random.set_state((n["bit_generator"],np.asarray(n["state"],dtype=np.uint32),
                         n["position"],n["has_gauss"],n["cached_gauss"]))
    if s["torch_cuda"]:
        torch.cuda.set_rng_state_all(s["torch_cuda"])


def load_archive(path):
    return torch.load(path,map_location="cpu",weights_only=True)


def validate_development(path,protocol):
    content=json.loads(Path(path).read_text())
    if (content["schema"]!="rhwm-pr31-factual-service-query-v3" or
        content["part"]!="development" or
        content["protocol_sha256"]!=sha(PROTOCOL) or
        content["input_schema"]!=INPUT_SCHEMA_V3 or
        content["label_schema"]!=QUERY_LABEL_VERSION or
        content["observation_schema"]!=SCHEMA_VERSION):
        raise ValueError("training dataset source or contract mismatch")
    if (content["coverage"]["train"]["groups"]!=80 or
        content["coverage"]["validation"]["groups"]!=12):
        raise AssertionError("unexpected factual groups")
    train,valid=content["train"],content["validation"]
    if set(x["trajectory_group"] for x in train)&set(
            x["trajectory_group"] for x in valid):
        raise AssertionError("train-validation group leakage")
    for records in (train,valid):
        if set(x["label"] for x in records)!={0,1}:
            raise AssertionError("positive and negative factual events required")
    return train,valid


def _make_model(arm,old_v4):
    m=BusinessSessionEventModel(query_width=QUERY_WIDTH_V3).cpu()
    if arm=="warmstart_v4":
        previous,_=load_v4(old_v4,experimental=True)
        m.load_state_dict(previous.state_dict())
    elif arm!="scratch_v4":
        raise ValueError("unknown train arm")
    return m


def _train_arm(arm,train,valid,*,config,old_v4,dev_path,outdir,environment):
    options=config["training"]
    seed=int(options["seed"])+int(arm=="scratch_v4")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(options["threads"]))
    model=_make_model(arm,old_v4)
    optimizer=torch.optim.Adam(model.parameters(),lr=float(options["lr"]))
    x,m,q=encode_rows(train,INPUT_SCHEMA_V3)
    vx,vm,vq=encode_rows(valid,INPUT_SCHEMA_V3)
    y=torch.tensor([int(r["label"]) for r in train],dtype=torch.float32)
    vy=torch.tensor([int(r["label"]) for r in valid],dtype=torch.float32)
    if min(len(y),len(vy))<100:raise AssertionError("too few labeled service events")
    history=[]
    best=None
    bad=0
    updates=0
    begun=time.process_time()
    for epoch in range(1,int(options["max_epochs"])+1):
        if time.process_time()-begun>options["max_cpu_seconds_per_arm"]:
            raise TimeoutError("training CPU cap exceeded")
        model.train()
        order=torch.randperm(len(y),generator=torch.Generator().manual_seed(seed+epoch))
        batches=[]
        for indices in order.split(options["batch_size"]):
            if time.process_time()-begun>options["max_cpu_seconds_per_arm"]:
                raise TimeoutError("training CPU cap exceeded mid-epoch")
            optimizer.zero_grad(set_to_none=True)
            loss=F.binary_cross_entropy_with_logits(model(x[indices],m[indices],q[indices]),y[indices])
            if not torch.isfinite(loss):raise AssertionError("nonfinite BCE")
            loss.backward()
            optimizer.step()
            updates+=1
            batches.append((float(loss.detach()),len(indices)))
        model.eval()
        with torch.inference_mode():
            logits=model(vx,vm,vq)
            pred=torch.sigmoid(logits)
            val_brier=float(torch.mean((pred-vy).square()))
            val_bce=float(F.binary_cross_entropy_with_logits(logits,vy))
        if not math.isfinite(val_brier):raise AssertionError("nonfinite validation")
        avg=sum(u*n for u,n in batches)/len(train)
        history.append({"epoch":epoch,"train_bce":avg,"validation_bce":val_bce,
                        "validation_brier":val_brier,"gradient_updates_total":updates})
        if best is None or val_brier<best["brier"]-1e-8:
            best={"epoch":epoch,"brier":val_brier,"model_state":copy.deepcopy(model.state_dict()),
                  "optimizer_state":copy.deepcopy(optimizer.state_dict()),
                  "rng":rng_snapshot(),"updates":updates}
            bad=0
        else:bad+=1
        if bad>=options["early_stop_patience"]:
            break
    elapsed=time.process_time()-begun
    if elapsed>options["max_cpu_seconds_per_arm"]:raise TimeoutError("training CPU cap exceeded")
    last_epoch=history[-1]["epoch"]
    meta={
        "model_version":VERSION,
        "input_schema":INPUT_SCHEMA_V3,"label_version":QUERY_LABEL_VERSION,
        "observation_schema":SCHEMA_VERSION,
        # Stable inference-interface protocol, with explicit PR31 training provenance.
        "protocol":"pr18-v4-frozen-synthetic-mechanism-20261110",
        "training_run_protocol":config["protocol"],
        "training_arm":arm,
        "source_model_sha256":sha(old_v4) if arm=="warmstart_v4" else None,
        "original_adam_was_not_restored":True if arm=="warmstart_v4" else None,
        "source_data_sha256":sha(dev_path),
        "environment_json_sha256":sha(Path(environment)/"environment.json"),
        "source_manifest_sha256":sha(Path(environment)/"source_manifest.json"),
        "online_verified":False,"seed":seed,
        "train_records":len(train),"validation_records":len(valid),
        "train_groups":80,"validation_groups":12,
        "best_epoch":best["epoch"],"epochs_executed":len(history),
        "gradient_updates_best":best["updates"],
        "gradient_updates_total":updates,
        "validation_brier":best["brier"],"cpu_process_seconds":elapsed,
    }
    best_source={
        "model_state":best["model_state"],
        "metadata":meta,
    }
    resume_base={"schema":"pr31-resumable-v4-training-state-v1",
                 "metadata":meta,"training_config":options,
                 "loss":"PyTorch binary_cross_entropy_with_logits",
                 "optimizer":"PyTorch Adam",
                 "all_train_history":history,"best_epoch":best["epoch"],
                 "best_brier":best["brier"],"best_model_state":best["model_state"],
                 "best_optimizer_state":best["optimizer_state"],
                 "optimizer_reinitialized_if_warmstart":arm=="warmstart_v4"}
    latest={
        **resume_base,
        "resume_from_epoch":last_epoch+1,"last_completed_epoch":last_epoch,
        "bad_epochs_since_best":bad,
        "model_state":copy.deepcopy(model.state_dict()),
        "optimizer_state":copy.deepcopy(optimizer.state_dict()),
        "rng":rng_snapshot(),
    }
    best_state={
        **resume_base,
        "resume_from_epoch":best["epoch"]+1,"last_completed_epoch":best["epoch"],
        "bad_epochs_since_best":0,
        "model_state":best["model_state"],"optimizer_state":best["optimizer_state"],
        "rng":best["rng"],
    }
    outdir.mkdir(parents=True,exist_ok=True)
    for path,obj in (
        (outdir/(arm+"-best.pt"),best_source),
        (outdir/(arm+"-best-resume.pt"),best_state),
        (outdir/(arm+"-latest-resume.pt"),latest)):
        if path.exists():raise FileExistsError("training outputs must be immutable")
        torch.save(obj,path)
    model.load_state_dict(best["model_state"])
    measured=stats(valid,probs(model,valid,INPUT_SCHEMA_V3))
    return {"arm":arm,"metadata":meta,
            "validation_metrics":{k:measured[k] for k in ("count","positive","brier","log_loss","ece5")},
            "history":history,
            "files":{name:{"sha256":sha(outdir/(arm+"-"+name+".pt")),
                             "bytes":(outdir/(arm+"-"+name+".pt")).stat().st_size}
                     for name in ("best","best-resume","latest-resume")}}


def _one_step_resumed(checkpoint,train):
    """Prove a saved optimizer+RNG state can actually restart and update."""
    state=load_archive(checkpoint)
    a=state["metadata"]
    if state["schema"]!="pr31-resumable-v4-training-state-v1":
        raise AssertionError("wrong saved resume checkpoint")
    x,m,q=encode_rows(train[:32],INPUT_SCHEMA_V3)
    y=torch.tensor([int(r["label"]) for r in train[:32]],dtype=torch.float32)
    digests=[]
    for _ in range(2):
        model=BusinessSessionEventModel(query_width=QUERY_WIDTH_V3)
        model.load_state_dict(state["model_state"])
        opt=torch.optim.Adam(model.parameters(),lr=state["training_config"]["lr"])
        opt.load_state_dict(state["optimizer_state"])
        restore_rng(state["rng"])
        model.train()
        opt.zero_grad(set_to_none=True)
        loss=F.binary_cross_entropy_with_logits(model(x,m,q),y)
        loss.backward()
        opt.step()
        digests.append(hashlib.sha256(
            b"".join(t.detach().cpu().numpy().tobytes()
                     for t in model.state_dict().values())).hexdigest())
    if digests[0]!=digests[1]:
        raise AssertionError("resuming full environment/model/optimizer/RNG is not deterministic")
    return {"checkpoint":Path(checkpoint).name,
            "resumed_from_epoch":state["resume_from_epoch"],
            "identical_one_step_update":True,"post_update_sha256":digests[0],
            "has_optimizer_state":bool(state["optimizer_state"].get("state")),
            "saved_rng":"Python NumPy torch.cpu (and CUDA if used)"}


def train(dev_path,original,outdir,envdir):
    cfg=json.loads(PROTOCOL.read_text())
    if cfg["protocol"]!="RHWM_PR31_V4_WARMSTART_SCRATCH_REPRO_TRAIN_20261112_V1":
        raise ValueError("wrong training protocol")
    if sha(original)!=cfg["old_checkpoint_sha256"]:
        raise ValueError("historical V4 checkpoint hash mismatch")
    train,valid=validate_development(dev_path,cfg)
    if outdir.exists():raise FileExistsError("research output directory exists")
    outdir.mkdir(parents=True)
    base,_=load_v4(original,experimental=True)
    frozen=stats(valid,probs(base,valid,INPUT_SCHEMA_V3))
    results={}
    for arm in ("warmstart_v4","scratch_v4"):
        results[arm]=_train_arm(
            arm,train,valid,config=cfg,old_v4=original,
            dev_path=dev_path,outdir=outdir,environment=envdir)
    selection={
        "protocol":cfg["protocol"],"selection_only":"new independent development validation",
        "frozen_v4_validation":{k:frozen[k] for k in ("count","positive","brier","log_loss","ece5")},
        "arms":results,
    }
    scores={"frozen_v4":frozen["brier"],
            **{name:item["validation_metrics"]["brier"] for name,item in results.items()}}
    selection["best_by_validation_brier"]=min(
        scores,key=lambda k:(scores[k],list(scores).index(k)))
    selection["validation_scores"]=scores
    selection["data_sha256"]=sha(dev_path)
    selection["old_v4_sha256"]=sha(original)
    selection["environment_sha256"]=sha(Path(envdir)/"environment.json")
    report=outdir/"validation-selection.json"
    if report.exists():raise FileExistsError("selection previously generated")
    report.write_text(json.dumps(selection,indent=2,ensure_ascii=False)+"\n")
    resumption=[]
    for arm in ("warmstart_v4","scratch_v4"):
        for checkpoint in ("best-resume","latest-resume"):
            resumption.append(_one_step_resumed(
                outdir/(arm+"-"+checkpoint+".pt"),train))
    (outdir/"resume-verified.json").write_text(json.dumps(
        resumption,indent=2,ensure_ascii=False)+"\n")
    print("PR31_TRAINING_COMPLETED",json.dumps({
        "train_rows":len(train),"val_rows":len(valid),
        "val_brier":scores,"selected":selection["best_by_validation_brier"],
        "resume_verified":len(resumption),
        "training_cpu_seconds":{a:results[a]["metadata"]["cpu_process_seconds"]
                                for a in results}},ensure_ascii=False))
    return selection


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--development",required=True,type=Path)
    p.add_argument("--original",required=True,type=Path)
    p.add_argument("--outdir",required=True,type=Path)
    p.add_argument("--environment",required=True,type=Path)
    args=p.parse_args()
    train(args.development,args.original,args.outdir,args.environment)


if __name__=="__main__":
    main()
