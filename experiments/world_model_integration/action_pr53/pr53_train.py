"""PR53: calibrated-structure and action-difference learned relational dynamics.

The network PREDICTS per-entity public future states and new-arrival
event counts. Its candidate score is always fixed to H20 expected
completed stages. There is no Q function or trainable ranking head.
This is a registered controlled adaptation of PyG RGCNConv / object-
factored dynamics, not a copy of an external architecture/loss paper.

All trained models share true outcomes from NEW PR53 factual episodes.
The original PR52 test and its near-tie diagnostics do NOT participate
in fitting, validation, or choice. Train/validation/test are physical
separate files and independent test is read only after selection.json.
"""
from __future__ import annotations
from itertools import combinations
from pathlib import Path
import argparse
import json
import random

import torch
from torch import nn
import torch.nn.functional as F

from experiments.world_model_integration.entity_pr52.pr52_train import (
    EntityConditionalRelationalWorld,prepare,decision_cost,entity_losses,
    fixed_completion_sensitivity_loss,total_completed_future,SCALE
)
from experiments.world_model_integration.decision_learning_pr51.pr51_loss import (
    pairwise_decision_loss,gradient_path_present
)

PROTOCOL=Path(__file__).with_name("pr53_protocol.json")
REGIMES=("congestion","multi_fault","staggered_arrivals")
ARMS=("unconstrained_pred_control","consistent_pred_only",
      "consistent_action_delta","consistent_delta_joint",
      "consistent_delta_value")
SEEDS=(1810,1811,1812)


def load_physical_split(root,split):
    """Never open raw combined source archive or future TEST during training."""
    if split not in ("train","validation","test"):
        raise ValueError("split name unregistered")
    data=json.loads((root/(split+".json")).read_text())
    if (data.get("schema")!="pr53-physically-separated-public-entity-world-state-v1"
        or data.get("split")!=split):
        raise ValueError("not independent PR53 public entity split")
    groups=data["cases"]
    length={"train":144,"validation":36,"test":48}[split]
    if len(groups)!=length or len({x["seed"] for x in groups})!=length:
        raise AssertionError("cohort size or seed uniqueness changed")
    if any(x["split"]!=split for x in groups):
        raise ValueError("mixed cohorts")
    if any(sum(x["regime"]==reg for x in groups)!=length//3 for reg in REGIMES):
        raise ValueError("regime imbalance")
    return groups


class ConsistentEntityWorld(EntityConditionalRelationalWorld):
    """Same pretrained-class architecture and parameters; constrained outputs."""

    def __init__(self,hidden=48,consistent=True):
        super().__init__(hidden,relational=True)
        self.consistent=bool(consistent)

    def forward(self,batch):
        node,event=super().forward(batch)
        if not self.consistent:
            return node,event
        # All existing PUBLIC task stage types have exactly one of:
        # waiting, assigned, in_transit, completed. Model this as a
        # valid categorical probability, not four independent Bernoullis.
        task=batch.x[:,1].view(-1,1,1)>0.5
        categorical=torch.softmax(
            torch.logit(node[:,:,:4].clamp(1e-5,1.-1e-5)),
            dim=-1)
        node=torch.cat((
            torch.where(task,categorical,node[:,:,:4]),
            node[:,:,4:]),dim=-1)
        # Physical event relation: a newly visible task stage cannot be
        # completed unless it has appeared. This is stochastic conditional
        # probability factorization, not access to future job identities.
        newly_visible=event[:,:,1]
        newly_completed_given_visible=event[:,:,0]
        event=torch.stack(
            (newly_visible*newly_completed_given_visible,
             newly_visible),dim=-1)
        return node,event


def paired_future_state_contrast(node,event,g):
    """Match observed paired FUTURE-STATE differences across legal actions.

    This deliberately updates the same state-transition parameters.
    It is NOT a stand-alone ranker and does not infer rank from factual
    episode cost. No fabricated winners for tied real-world outcomes.
    """
    k=len(g["true_cost"])
    if k<2:
        return node.sum()*0.
    if node.shape[0]%k:
        raise ValueError("unequal public node counts between candidates")
    n=node.shape[0]//k
    pn=node.view(k,n,2,6)
    yn=g["node_y"].to(node.device).view(k,n,2,6)
    mask=g["node_mask"].to(node.device).view(k,n,2,6)
    joined=mask[1:]*mask[:1]
    if not joined.any():
        raise ValueError("no same known nodes across paired actions")
    d_pred=pn[1:]-pn[:1]
    d_true=yn[1:]-yn[:1]
    node_delta=((d_pred-d_true).square()*joined).sum()/joined.sum().clamp(min=1.)
    # Unknown FUTURE new task IDs never appear in any input: only the
    # anonymized observed event count is paired.
    e=g["event_y"].to(event.device)
    event_delta=F.mse_loss(event[1:]-event[:1],e[1:]-e[:1])
    return node_delta+0.3*event_delta


def loss_components(net,g,arm):
    b=g["batch"]
    n,e=net(b)
    predicted=entity_losses(n,e,g)
    state=predicted["prediction"]
    contrast=paired_future_state_contrast(n,e,g)
    rank=pairwise_decision_loss(decision_cost(n,e,b),g["true_cost"])
    weighted=n.new_tensor(0.)
    if arm["value_sensitivity"]>0:
        weighted=fixed_completion_sensitivity_loss(n,e,g)
    loss=(state+
          arm["contrast_weight"]*contrast+
          arm["decision_weight"]*rank+
          arm["value_sensitivity"]*weighted)
    return loss,{"state":state,"contrast":contrast,
                 "decision":rank,"weighted":weighted}


def report(net,groups):
    if not groups:
        raise ValueError("no public cases in evaluation split")
    net.eval()
    sums={"groups":len(groups),"entity_mse":0.,"station_device_brier_total":0.,
          "station_device_count":0,"completed_count_mae":0.,
          "true_regret_ticks":0.,"cp_sat_first_regret_ticks":0.,
          "near_top2_1e_5":0,"near_top2_1e_3":0,
          "truly_nontrivial_cases":0,
          "pairwise_correct":0,"pairwise_comparable":0}
    by_regime={}
    with torch.no_grad():
        for g in groups:
            n,e=net(g["batch"])
            ground=g["node_y"]
            mask=g["node_mask"]
            summary=entity_losses(n,e,g)
            sums["entity_mse"]+=float(summary["node_mse"])
            station=(g["batch"].x[:,2,None]*mask[:,:,0])
            sums["station_device_brier_total"]+=float(
                ((n[:,:,0]-ground[:,:,0]).square()*station).sum())
            sums["station_device_count"]+=int(station.sum())
            predicted_total=total_completed_future(n,e,g["batch"])
            sums["completed_count_mae"]+=float(
                (predicted_total/SCALE-g["total_y"]).abs().mean()*SCALE)
            pred=decision_cost(n,e,g["batch"]).tolist()
            actual=g["true_cost"].tolist()
            idx=min(range(len(pred)),key=lambda i:(pred[i],i))
            regret=actual[idx]-min(actual)
            sums["true_regret_ticks"]+=regret
            sums["cp_sat_first_regret_ticks"]+=actual[0]-min(actual)
            by_regime.setdefault(g["regime"],[]).append(regret)
            sorted_pred=sorted(pred)
            gap=sorted_pred[1]-sorted_pred[0]
            sums["near_top2_1e_5"]+=int(gap<1e-5)
            sums["near_top2_1e_3"]+=int(gap<1e-3)
            sums["truly_nontrivial_cases"]+=int(len(set(actual))>1)
            for i,j in combinations(range(len(actual)),2):
                if actual[i]==actual[j]:
                    continue
                sums["pairwise_comparable"]+=1
                sums["pairwise_correct"]+=int(
                    (pred[i]-pred[j])*(actual[i]-actual[j])>0)
    for col in ("entity_mse","completed_count_mae",
                "true_regret_ticks","cp_sat_first_regret_ticks"):
        sums[col]/=len(groups)
    sums["station_device_brier"]=(
        sums.pop("station_device_brier_total")/
        max(1,sums["station_device_count"]))
    sums["pairwise_accuracy"]=(
        sums["pairwise_correct"]/sums["pairwise_comparable"]
        if sums["pairwise_comparable"] else None)
    sums["regret_by_regime"]={r:sum(v)/len(v) for r,v in by_regime.items()}
    return sums


def train_arm(arm,seed,train,valid,cfg,out):
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)
    model=ConsistentEntityWorld(
        hidden=cfg["train"]["hidden_size"],consistent=arm["consistent"])
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg["train"]["learning_rate"])
    grad_verified=None
    timeline=[]
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        rng=random.Random(seed+epoch)
        order=list(range(len(train)))
        rng.shuffle(order)
        summed=0.
        for index in order:
            g=train[index]
            optimizer.zero_grad(set_to_none=True)
            loss,terms=loss_components(model,g,arm)
            loss.backward()
            if (grad_verified is None and arm["consistent"] and
                (arm["contrast_weight"]>0 or arm["decision_weight"]>0
                 or arm["value_sensitivity"]>0)):
                # Verify ONLY the target term; no credit from state MSE.
                optimizer.zero_grad(set_to_none=True)
                n,e=model(g["batch"])
                if arm["decision_weight"] and len(set(g["true_cost"].tolist()))>1:
                    term=pairwise_decision_loss(decision_cost(n,e,g["batch"]),
                                                g["true_cost"])
                elif arm["contrast_weight"]:
                    term=paired_future_state_contrast(n,e,g)
                elif arm["value_sensitivity"]:
                    term=fixed_completion_sensitivity_loss(n,e,g)
                else:
                    term=None
                if term is not None:
                    term.backward()
                    grad_verified=gradient_path_present(model.transition1.parameters())
                    if not grad_verified:
                        raise AssertionError("registered auxiliary loss bypasses shared world dynamics")
                optimizer.zero_grad(set_to_none=True)
                loss,_=loss_components(model,g,arm)
                loss.backward()
            optimizer.step()
            summed+=float(loss.detach())
        if epoch in (0,cfg["train"]["epochs"]-1):
            timeline.append({"epoch":epoch+1,"train_loss":summed/len(train),
                             "validation":report(model,valid)})
    if arm["consistent"] and (
        arm["contrast_weight"]>0 or arm["decision_weight"]>0
        or arm["value_sensitivity"]>0) and not grad_verified:
        raise AssertionError("no auxiliary loss gradient to shared world model")
    key=arm["id"]+"_seed"+str(seed)
    file=out/(key+".pt")
    torch.save({
        "model":model.state_dict(),"optimizer":optimizer.state_dict(),
        "torch_rng":torch.get_rng_state(),"arm":arm,"seed":seed,
        "epochs":cfg["train"]["epochs"],
        "auxiliary_gradient_into_relational_dynamics":grad_verified,
        "frozen_readout":"- expected H20 completed stages; no trainable ranker",
        "trained_true_world_split":"PR53 new separate train only"
    },file)
    return {"checkpoint":file.name,"seed":seed,"arm":arm["id"],
            "gradient_to_relational_dynamics":grad_verified,
            "training":report(model,train),
            "validation":report(model,valid),"trace":timeline}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():
        raise FileExistsError("PR53 trained models are write-once")
    cfg=json.loads(PROTOCOL.read_text())
    if cfg["protocol"]!="PR53_ACTION_CONTRAST_CONSISTENT_WORLD_DYNAMICS_20261112_V1":
        raise ValueError("invalid frozen PR53 protocol")
    args.out.mkdir(parents=True)
    train=load_physical_split(args.input,"train")
    valid=load_physical_split(args.input,"validation")
    if {x["seed"] for x in train}&{x["seed"] for x in valid}:
        raise AssertionError("train-validation scenario overlap")
    outputs={}
    for arm in cfg["train_arms"]:
        a=arm["id"]
        train_p,train_exclusions=prepare(train,True,"train")
        valid_p,val_exclusions=prepare(valid,True,"validation")
        if train_exclusions or val_exclusions:
            raise AssertionError("cannot silently drop training/validation groups")
        outputs[a]={}
        for seed in cfg["train"]["seeds"]:
            z=train_arm(arm,seed,train_p,valid_p,cfg,args.out)
            outputs[a][str(seed)]=z
            print("PR53_TRAIN_FINISHED",a,seed,json.dumps(
                z["validation"],ensure_ascii=False),flush=True)
    seeds=cfg["train"]["seeds"]
    def average(name,col):
        return sum(outputs[name][str(s)]["validation"][col]
                   for s in seeds)/len(seeds)
    control="consistent_pred_only"
    base_mse=average(control,"entity_mse")
    base_brier=average(control,"station_device_brier")
    eligible=[a for a in ARMS if
              average(a,"entity_mse")<=1.15*base_mse and
              average(a,"station_device_brier")<=1.15*base_brier]
    if control not in eligible:
        raise AssertionError("same-class baseline must be eligible")
    chosen=min(eligible,key=lambda a:(
        average(a,"true_regret_ticks"),average(a,"entity_mse"),ARMS.index(a)))
    freeze={
        "schema":"pr53-immutable-validation-only-calibrated-model-selection-v1",
        "selected":chosen,
        "eligible_under_prediction_and_brier_guard":eligible,
        "prediction_guard_ratio":1.15,
        "validation_mean_over_three_training_seeds":{
            arm:{key:average(arm,key) for key in (
                "true_regret_ticks","entity_mse","station_device_brier",
                "completed_count_mae","near_top2_1e_5","near_top2_1e_3")}
            for arm in ARMS},
        "all_models":outputs,
        "independent_test_not_loaded":True,
        "new_training_device":"CPU hosted reproducible proxy",
        "fixed_score":cfg["score_rule"]
    }
    (args.out/"selection.json").write_text(
        json.dumps(freeze,ensure_ascii=False,indent=2)+"\n")
    print("PR53_VALIDATION_LOCKED",json.dumps({
        "selected":chosen,"eligible":eligible,
        "validation_metrics":freeze["validation_mean_over_three_training_seeds"]
    },ensure_ascii=False),flush=True)
    # Physically isolated unseen split opened only AFTER freeze file exists.
    test=load_physical_split(args.input,"test")
    if {x["seed"] for x in test}&{x["seed"] for x in train+valid}:
        raise AssertionError("new blind test overlaps earlier fitted data")
    tests={}
    for arm in cfg["train_arms"]:
        a=arm["id"]
        prepared,exclusions=prepare(test,True,"test")
        if exclusions:
            raise AssertionError("no test drop permitted")
        tests[a]={}
        for seed in seeds:
            key=a+"_seed"+str(seed)
            checkpoint=torch.load(args.out/(key+".pt"),map_location="cpu",weights_only=False)
            net=ConsistentEntityWorld(
                cfg["train"]["hidden_size"],arm["consistent"])
            net.load_state_dict(checkpoint["model"])
            tests[a][str(seed)]=report(net,prepared)
    result={
        "schema":"pr53-consistent-relational-action-contrast-independent-test-v1",
        "selected_by_validation_before_test":chosen,
        "number_of_fitted_neural_models":len(ARMS)*len(seeds),
        "distinct_scenes":{"train":144,"validation":36,"new_blind_test":48},
        "validation_selected":freeze,
        "all_arms_new_blind_test":tests,
        "cpu_training":True,"gpu_training":False,
        "new_true_world_data_only":True,
        "new_full_online_system_episodes":0,
        "limits":"H5/H20 entity conditional forecasts, not full recurrent stochastic world simulator or GPU deployment"
    }
    (args.out/"result.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print("PR53_NEW_BLIND_TEST_RESULT",json.dumps({
        "selected":chosen,
        "per_arm_and_seed":tests,
        "source_groups":result["distinct_scenes"]
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
