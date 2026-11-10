"""Business-session Deep Sets v2, not compatible with old PR9/PETS weights.

The global entity set provides context; each queried active service gets a
separate target representation and completion logit. Only factual completed
sessions can supervise the model. New checkpoints are not online-certified by
training alone.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time
import torch
from torch import nn
from business_transport.session_events import QUERY_LABEL_VERSION, QUERY_WIDTH, encode_query
from business_transport.state_features import SCHEMA_VERSION
from lab_transport.business_state_learning import TOKEN_WIDTH, encode_tokens

MODEL_VERSION = "business-sets-session-v2"
THRESHOLDS = (0.3,0.4,0.5,0.6,0.7)
CONFIG = {"seed":1810,"epochs":10,"lr":0.001,"budget_seconds":180,
          "checkpoint_metric":"validation_brier","threshold_metric":"validation_f1",
          "thresholds": list(THRESHOLDS), "ece_bins":5}


class BusinessSessionEventModel(nn.Module):
    def __init__(self,query_width=QUERY_WIDTH):
        super().__init__()
        self.entity = nn.Sequential(nn.Linear(TOKEN_WIDTH,32),nn.ReLU(),
                                    nn.Linear(32,32),nn.ReLU())
        self.target = nn.Sequential(nn.Linear(query_width,32),nn.ReLU())
        self.service_event = nn.Sequential(nn.Linear(96,32),nn.ReLU(),nn.Linear(32,1))

    def forward(self,tokens,mask,query):
        if tokens.ndim!=3 or tokens.shape[-1]!=TOKEN_WIDTH or mask.shape!=tokens.shape[:2]:
            raise ValueError("invalid entity tensor/mask")
        if query.ndim!=2 or query.shape[-1]!=self.target[0].in_features or query.shape[0]!=tokens.shape[0]:
            raise ValueError("invalid per-session query tensor")
        context=(self.entity(tokens)*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
        target=self.target(query)
        return self.service_event(torch.cat((context,target,context*target),dim=-1)).squeeze(-1)


def encode_session_batch(rows):
    entities=[encode_tokens(item["features"]) for item in rows]
    maxlen=max(t.shape[0] for t in entities)
    x=torch.zeros(len(rows),maxlen,TOKEN_WIDTH)
    mask=torch.zeros(len(rows),maxlen)
    q=torch.tensor([encode_query(item["features"],item["query"]) for item in rows],dtype=torch.float32)
    for i,t in enumerate(entities):
        x[i,:len(t)]=t
        mask[i,:len(t)]=1.
    return x,mask,q


def probabilities(model, rows):
    model.eval()
    with torch.inference_mode():
        x,m,q=encode_session_batch(rows)
        return torch.sigmoid(model(x,m,q)).cpu().tolist()


def metrics(rows,probs,threshold=0.5):
    if len(rows)!=len(probs) or not rows:raise ValueError("empty or mismatched metrics")
    y=[int(row["label"]) for row in rows]
    tp=sum(p>=threshold and a==1 for p,a in zip(probs,y))
    fp=sum(p>=threshold and a==0 for p,a in zip(probs,y))
    fn=sum(p<threshold and a==1 for p,a in zip(probs,y))
    tn=len(y)-tp-fp-fn
    brier=sum((p-a)**2 for p,a in zip(probs,y))/len(y)
    ece=0.
    bins=[]
    for i in range(5):
        idx=[j for j,p in enumerate(probs) if i/5 <= p < (i+1)/5 or (i==4 and p==1)]
        if idx:
            confidence=sum(probs[j] for j in idx)/len(idx)
            frequency=sum(y[j] for j in idx)/len(idx)
            ece+=len(idx)/len(y)*abs(confidence-frequency)
            bins.append({"count":len(idx),"predicted":confidence,"observed":frequency})
    precision=tp/(tp+fp) if tp+fp else None
    recall=tp/(tp+fn) if tp+fn else None
    return {"count":len(y),"positive":sum(y),"negative":len(y)-sum(y),
            "brier":brier,"ece5":ece,"calibration_bins":bins,
            "precision":precision,"recall":recall,"tp":tp,"fp":fp,"tn":tn,"fn":fn,
            "threshold":threshold,"accuracy":(tp+tn)/len(y)}


def _f1(stats):
    p=stats["precision"] or 0.
    r=stats["recall"] or 0.
    return 2*p*r/(p+r) if p+r else 0.


def verify_dataset(dataset):
    if (dataset.get("schema_version") != SCHEMA_VERSION or
        dataset.get("label_version") != QUERY_LABEL_VERSION or
        dataset.get("source_type") != "factual_observed_transition"):
        raise ValueError("wrong dataset provenance or version")
    all_groups=set()
    for split in ("train","validation","test"):
        rows=dataset.get(split)
        if not rows:
            raise ValueError("empty required split: "+split)
        groups={r["trajectory_group"] for r in rows}
        if groups & all_groups:
            raise ValueError("trajectory group leaked across split")
        all_groups.update(groups)
        for r in rows:
            if r["label"] not in (0,1) or not isinstance(r["id"],str):
                raise ValueError("invalid binary factual label")
            if r.get("event") not in ("finished","continues"):
                raise ValueError("censored/paused service incorrectly became a label")
            encode_query(r["features"],r["query"])
        if split in ("train","validation") and set(r["label"] for r in rows)!={0,1}:
            raise ValueError("train/validation needs genuine positive and negative events")
    return {name: {"rows":len(dataset[name]),"positive":sum(r["label"] for r in dataset[name]),
                   "groups":len({r["trajectory_group"] for r in dataset[name]})}
            for name in ("train","validation","test")}


def train_session_cpu(dataset_path,checkpoint_path,*,seed=1810,epochs=10,seconds=180):
    if (seed,epochs)!=(CONFIG["seed"],CONFIG["epochs"]) or seconds!=CONFIG["budget_seconds"]:
        raise ValueError("training config is frozen at seed1810 epochs10 CPU180s")
    dataset_path=Path(dataset_path)
    checkpoint_path=Path(checkpoint_path)
    if checkpoint_path.exists():raise FileExistsError("no checkpoint overwrite")
    dataset=json.loads(dataset_path.read_text(encoding="utf-8"))
    splits=verify_dataset(dataset)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model=BusinessSessionEventModel().cpu()
    optimizer=torch.optim.Adam(model.parameters(),lr=CONFIG["lr"])
    start=time.monotonic()
    x,m,q=encode_session_batch(dataset["train"])
    y=torch.tensor([r["label"] for r in dataset["train"]],dtype=torch.float32)
    history=[]
    best=None
    for epoch in range(1,epochs+1):
        if time.monotonic()-start>=seconds:raise TimeoutError("CPU training budget reached before checkpoint")
        model.train();optimizer.zero_grad()
        loss=nn.functional.binary_cross_entropy_with_logits(model(x,m,q),y)
        if not torch.isfinite(loss):raise ValueError("nonfinite training loss")
        loss.backward();optimizer.step()
        val=metrics(dataset["validation"],probabilities(model,dataset["validation"]))
        history.append({"epoch":epoch,"train_bce":float(loss),"validation_brier":val["brier"]})
        if best is None or val["brier"]<best["brier"]:
            best={"epoch":epoch,"brier":val["brier"],
                  "state":{k:v.detach().clone() for k,v in model.state_dict().items()}}
    model.load_state_dict(best["state"])
    validation_probs=probabilities(model,dataset["validation"])
    scored=[(_f1(metrics(dataset["validation"],validation_probs,t)), -abs(t-0.5),t)
            for t in THRESHOLDS]
    threshold=max(scored)[2]
    val=metrics(dataset["validation"],validation_probs,threshold)
    test=metrics(dataset["test"],probabilities(model,dataset["test"]),threshold)
    def grouped(rows):
        buckets={}
        for row in rows:
            buckets.setdefault(row["trajectory_group"],[]).append(row)
        return {key:metrics(items,probabilities(model,items),threshold)
                for key,items in sorted(buckets.items())}
    meta={"model_version":MODEL_VERSION,"schema_version":SCHEMA_VERSION,
          "label_version":QUERY_LABEL_VERSION,
          "source_type":"factual_observed_transition",
          "dataset_sha256":hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
          "config":CONFIG,"split_counts":splits,
          "training_log":history,"best_validation_epoch":best["epoch"],
          "selected_validation_threshold":threshold,
          "validation":val,"test_once":test,
          "validation_groups":grouped(dataset["validation"]),
          "test_groups":grouped(dataset["test"]),
          "online_verified":False,
          "note":"TRAINED OFFLINE, NOT CERTIFIED FOR ONLINE OPERATIONS"}
    checkpoint_path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"metadata":meta,"model_state":model.state_dict()},checkpoint_path)
    return meta


def load_session_model(path,*,experimental=False):
    saved=torch.load(path,map_location="cpu",weights_only=True)
    m=saved["metadata"]
    if (m.get("model_version")!=MODEL_VERSION or m.get("schema_version")!=SCHEMA_VERSION or
        m.get("label_version")!=QUERY_LABEL_VERSION):
        raise ValueError("v2 session model checkpoint contract mismatch")
    if not m.get("online_verified") and not experimental:
        raise ValueError("v2 checkpoint not certified for online use; experimental only")
    model=BusinessSessionEventModel()
    model.load_state_dict(saved["model_state"])
    return model.eval(),m
