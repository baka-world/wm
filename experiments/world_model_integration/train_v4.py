"""Only one frozen V4-data training run, identical V3 architecture and optimizer.

Train-only gradient updates, Validation-only Brier selection. No historical
independent holdout opened by this module. Checkpoint is research-only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import torch
from torch import nn

from business_transport.session_events import QUERY_LABEL_VERSION
from business_transport.session_features_v3 import INPUT_SCHEMA_V3,QUERY_WIDTH_V3
from business_transport.state_features import SCHEMA_VERSION
from lab_transport.business_session_learning import BusinessSessionEventModel
from lab_transport.business_session_v3 import encode_rows,stats,grouped

CONFIG=Path(__file__).with_name("v4_frozen_config.json")
VERSION="business-sets-session-v4-data"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_development(data,config):
    if data.get("part")!="development" or data.get("protocol")!=config["protocol"]:
        raise ValueError("V4 training source not the frozen development split")
    if data.get("source_config_sha256")!=sha(CONFIG):
        raise ValueError("source data does not match committed config")
    if data.get("schema_version")!=SCHEMA_VERSION or data.get("label_version")!=QUERY_LABEL_VERSION:
        raise ValueError("changed factual observation/label contract")
    train,validation=data["train"],data["validation"]
    if not train or not validation:raise ValueError("empty required split")
    if {r["trajectory_group"] for r in train}&{r["trajectory_group"] for r in validation}:
        raise ValueError("train-validation trajectory leakage")
    for split,rows in (("train",train),("validation",validation)):
        if len({r["trajectory_group"] for r in rows})!=config["train_groups" if split=="train" else "validation_groups"]:
            raise ValueError("wrong number of unique groups in "+split)
        if set(row["label"] for row in rows)!={0,1}:
            raise ValueError("missing real positive or negative event")
        for row in rows:
            if row["event"] not in ("finished","continues") or row["label"]!=int(row["event"]=="finished"):
                raise ValueError("invalid supervised event semantics")
    c=data["coverage"]["train"]
    for prior in range(3,9):
        k=c["axes"]["nominal"].get(str(prior),{})
        if k.get("positive",0)<config["data_gate"]["each_prior_train_min_complete"] or k.get("negative",0)<config["data_gate"]["each_prior_train_min_continue"]:
            raise ValueError("insufficient coverage for public prior "+str(prior))
    if c["observed_paused_session_ticks"]<1:raise ValueError("no genuine service pause recorded")
    return train,validation


def load_v4(path,*,experimental=False):
    state=torch.load(path,map_location="cpu",weights_only=True)
    meta=state["metadata"]
    if (meta.get("model_version")!=VERSION or meta.get("input_schema")!=INPUT_SCHEMA_V3 or
        meta.get("label_version")!=QUERY_LABEL_VERSION or meta.get("observation_schema")!=SCHEMA_VERSION or
        meta.get("protocol")!="pr18-v4-frozen-synthetic-mechanism-20261110"):
        raise ValueError("V4 checkpoint does not match frozen v3-input service-event contract")
    if not meta.get("online_verified") and not experimental:
        raise ValueError("V4 research checkpoint not certified for online use")
    model=BusinessSessionEventModel(query_width=QUERY_WIDTH_V3)
    model.load_state_dict(state["model_state"])
    return model.eval(),meta


def train_v4(data_path,checkpoint,report):
    checkpoint,report=Path(checkpoint),Path(report)
    if checkpoint.exists() or report.exists():raise FileExistsError("V4 outputs must be write-once")
    config=json.loads(CONFIG.read_text())
    data=json.loads(Path(data_path).read_text())
    train,validation=validate_development(data,config)
    cpu=config["cpu"]
    if not (cpu["optimizer"]=="Adam" and cpu["batch_size"]==32 and cpu["max_epochs"]==100
            and cpu["training_seconds_max"]==180 and cpu["threads"]==1
            and cpu["seed"]==1810 and cpu["lr"]==0.001 and cpu["patience"]==10):
        raise ValueError("frozen V4 training setup was modified")
    torch.set_num_threads(1);torch.manual_seed(cpu["seed"])
    model=BusinessSessionEventModel(query_width=QUERY_WIDTH_V3).cpu()
    optimizer=torch.optim.Adam(model.parameters(),lr=cpu["lr"])
    x,m,q=encode_rows(train,INPUT_SCHEMA_V3)
    vx,vm,vq=encode_rows(validation,INPUT_SCHEMA_V3)
    y=torch.tensor([r["label"] for r in train],dtype=torch.float32)
    vy=torch.tensor([r["label"] for r in validation],dtype=torch.float32)

    # Discarded CPU-only preflight; no selection from held-out data.
    trial=BusinessSessionEventModel(query_width=QUERY_WIDTH_V3).cpu()
    t_opt=torch.optim.Adam(trial.parameters(),lr=cpu["lr"])
    check_start=time.monotonic()
    for _ in range(3):
        t_opt.zero_grad(set_to_none=True)
        l=nn.functional.binary_cross_entropy_with_logits(trial(x[:32],m[:32],q[:32]),y[:32])
        l.backward();t_opt.step()
    preflight_seconds=time.monotonic()-check_start
    projected_steps=math.ceil(len(train)/32)*cpu["max_epochs"]
    # Preflight estimate errs conservatively: 3x observed batch speed + 3s eval budget.
    projected_seconds=projected_steps*(preflight_seconds/3)*3+3
    if projected_seconds>cpu["training_seconds_max"]:
        raise TimeoutError("V4 frozen CPU preflight estimate exceeds the budget; no training attempted")
    del trial,t_opt

    started=time.monotonic();history=[];best=None;bad=0;total_steps=0
    for epoch in range(1,cpu["max_epochs"]+1):
        order=torch.randperm(len(y),generator=torch.Generator().manual_seed(cpu["seed"]+epoch))
        train_losses=[];epoch_steps=0
        model.train()
        for ix in order.split(32):
            if time.monotonic()-started>cpu["training_seconds_max"]:
                raise TimeoutError("V4 hard 180s per-training CPU budget exceeded")
            optimizer.zero_grad(set_to_none=True)
            loss=nn.functional.binary_cross_entropy_with_logits(model(x[ix],m[ix],q[ix]),y[ix])
            if not torch.isfinite(loss):raise ValueError("non-finite V4 train loss")
            loss.backward();optimizer.step()
            epoch_steps+=1;total_steps+=1
            train_losses.append((float(loss.detach()),len(ix)))
        model.eval()
        with torch.inference_mode():
            logits=model(vx,vm,vq)
            distribution=torch.sigmoid(logits).cpu().tolist()
            val_loss=float(nn.functional.binary_cross_entropy_with_logits(logits,vy))
        result=stats(validation,distribution)
        group=grouped(validation,distribution,"elapsed")
        history.append({"epoch":epoch,"optimizer_steps_epoch":epoch_steps,
            "optimizer_steps_total":total_steps,
            "train_bce":sum(v*n for v,n in train_losses)/len(train),
            "validation_bce":val_loss,"validation_brier":result["brier"],
            "validation_distribution":result["distribution"],
            "effective_elapsed":{k:{"n":v["count"],"brier":v["brier"],"positive":v["positive"]}
                                 for k,v in group.items()}})
        if best is None or result["brier"]<best["brier"]-1e-8:
            best={"epoch":epoch,"steps":total_steps,"brier":result["brier"],
                  "weights":{k:v.detach().clone() for k,v in model.state_dict().items()}}
            bad=0
        else:bad+=1
        if bad>=cpu["patience"]:break
    model.load_state_dict(best["weights"])
    from lab_transport.business_session_v3 import probs
    final=probs(model,validation,INPUT_SCHEMA_V3)
    summary={"model_version":VERSION,"input_schema":INPUT_SCHEMA_V3,
      "label_version":QUERY_LABEL_VERSION,"observation_schema":SCHEMA_VERSION,
      "protocol":config["protocol"],"config_sha256":sha(CONFIG),
      "dataset_sha256":sha(data_path),"seed":cpu["seed"],"online_verified":False,
      "gradient_steps_total":total_steps,"gradient_steps_best":best["steps"],
      "best_epoch":best["epoch"],"epochs_executed":len(history),
      "train_records":len(train),"validation_records":len(validation),
      "train_groups":len(data["groups"]["train"]),"validation_groups":len(data["groups"]["validation"]),
      "validation":stats(validation,final),
      "validation_by_public_nominal":evaluate_group(validation,final,"nominal"),
      "validation_by_observed_expectation":evaluate_group(validation,final,"prior"),
      "validation_by_elapsed":grouped(validation,final,"elapsed"),
      "preflight_seconds":preflight_seconds,"preflight_projected_seconds":projected_seconds,
      "train_elapsed_seconds":time.monotonic()-started,
      "history":history,"note":"OFFLINE experimental weights, not production authorized"}
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"metadata":summary,"model_state":model.state_dict()},checkpoint)
    result={"checkpoint_sha256":sha(checkpoint),
            "checkpoint":checkpoint.name,"config_sha256":sha(CONFIG),
            "dataset_sha256":sha(data_path),"metadata":summary}
    report.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"checkpoint_sha256":result["checkpoint_sha256"],
      "train_rows":len(train),"validation_rows":len(validation),
      "best_epoch":best["epoch"],"steps_total":total_steps,"steps_best":best["steps"],
      "validation":{k:summary["validation"][k] for k in ("brier","log_loss","ece5","precision","recall")},
      "cpu_seconds":summary["train_elapsed_seconds"]},ensure_ascii=False))
    return result


def evaluate_group(rows,probs,dimension):
    from lab_transport.business_session_v3 import stats
    from experiments.world_model_integration.build_dataset_v4 import record_attributes
    values={}
    for r,p in zip(rows,probs):
        k=str(record_attributes(r)[dimension])
        values.setdefault(k,[]).append((r,p))
    return {k:stats([r for r,p in v],[p for r,p in v]) for k,v in sorted(values.items())}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True)
    a=p.parse_args()
    train_v4(a.dataset,a.checkpoint,a.report)


if __name__=="__main__":
    main()
