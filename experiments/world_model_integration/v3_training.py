"""One frozen v3 ablation, baseline and selection pass; NO held-out Test access."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import torch
from experiments.world_model_integration.empirical_baselines import EmpiricalHazards
from lab_transport.business_session_v3 import fit_variant,load_v3,stats,probs,grouped,SCHEMA_V2
from lab_transport.business_session_learning import load_session_model,probabilities as v2_probs

ROOT=Path(__file__).resolve().parent
PROTOCOL=ROOT/"v3_training_protocol.json"

def run(dataset_path,v2_path,output_dir):
    dpath=Path(dataset_path);oldpath=Path(v2_path);folder=Path(output_dir)
    if folder.exists():raise FileExistsError("V3 train output dir must be new")
    cfg_bytes=PROTOCOL.read_bytes()
    config=json.loads(cfg_bytes)
    if config['protocol']!='pr18-v3-frozen-training-20261110':
        raise ValueError("unexpected frozen training protocol")
    data=json.loads(dpath.read_text())
    if data['label_version']!='business-service-query-v2':
        raise ValueError("source label schema changed")
    from lab_transport.business_session_learning import verify_dataset
    verify_dataset(data)
    if hashlib.sha256(oldpath.read_bytes()).hexdigest()!=config["ablation"]["V2"]["source"].split("sha256 ")[1]:
        raise ValueError("original V2 checkpoint checksum mismatch")
    if data['source_type']!="factual_observed_transition":raise ValueError("not factual")
    folder.mkdir(parents=True)
    torch.set_num_threads(1)
    baselines=EmpiricalHazards(data['train'])
    val=data['validation']
    reports={"protocol_sha256":hashlib.sha256(cfg_bytes).hexdigest(),
        "source_dataset_sha256":hashlib.sha256(dpath.read_bytes()).hexdigest(),
        "source_old_v2_sha256":hashlib.sha256(oldpath.read_bytes()).hexdigest(),
        "source":"train+validation only, never original seen test",
        "training_baselines":baselines.provenance(),
        "validation":{},"ablation":{}}
    for name in ("B0","B1","B2"):
        p=baselines.predict(val,name)
        reports["validation"][name]={"overall":stats(val,p),
            "grouped":{"trajectory":grouped(val,p,"trajectory"),
                       "elapsed":grouped(val,p,"elapsed")}}
    original,metadata=load_session_model(oldpath,experimental=True)
    reports["validation"]["V2"]={"overall":stats(val,v2_probs(original,val)),
        "checkpoint_is_original_frozen":True}
    total_start=time.monotonic()
    # preflight uses only TRAIN, a throwaway model and does not affect versions.
    from lab_transport.business_session_v3 import encode_rows
    from lab_transport.business_session_learning import BusinessSessionEventModel
    tx,tm,tq=encode_rows(data["train"][:32],SCHEMA_V2)
    torch.manual_seed(config['cpu']['seed'])
    trial=BusinessSessionEventModel()
    optimizer=torch.optim.Adam(trial.parameters(),lr=config['cpu']['lr'])
    start=time.monotonic()
    for i in range(config['cpu']['preflight_train_steps']):
        optimizer.zero_grad(set_to_none=True)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(trial(tx,tm,tq),
                   torch.tensor([r["label"] for r in data["train"][:32]],dtype=torch.float32))
        loss.backward();optimizer.step()
    reports["preflight"]={"steps":config['cpu']['preflight_train_steps'],
          "wall_seconds":time.monotonic()-start,"device":"cpu","threads":torch.get_num_threads()}
    for name in ("V3-A","V3-B","V3-C"):
        if time.monotonic()-total_start>=config["cpu"]["seconds_max_total"]:
            raise TimeoutError("frozen TOTAL v3 CPU budget exhausted")
        ckpt=folder/(name.lower()+".pt")
        metadata=fit_variant(data,name,config,ckpt)
        network,meta=load_v3(ckpt,experimental=True)
        p=probs(network,val,meta["input_schema"])
        reports["ablation"][name]={"training":metadata,"overall":stats(val,p),
          "grouped":{"trajectory":grouped(val,p,"trajectory"),
                     "elapsed":grouped(val,p,"elapsed"),
                     "station":grouped(val,p,"station"),
                     "phase":grouped(val,p,"phase")}}
    # pre-registered minimum Validation Brier, tie preference A -> B -> C.
    selected=min(("V3-A","V3-B","V3-C"),key=lambda n:reports["ablation"][n]["overall"]["brier"])
    reports["selection"]={"winner":selected,"candidate_file":selected.lower()+".pt",
        "rule":"validation Brier only; fixed threshold0.5",
        "gate_vs_B0":reports["ablation"][selected]["overall"]["brier"]<
                   reports["validation"]["B0"]["overall"]["brier"],
        "baseline_brier":reports["validation"]["B0"]["overall"]["brier"],
        "selected_brier":reports["ablation"][selected]["overall"]["brier"]}
    reports["time_seconds"]=time.monotonic()-total_start
    (folder/"v3_selection.json").write_text(json.dumps(reports,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"baseline_validation":{k:reports["validation"][k]["overall"]["brier"]
                       for k in ("B0","B1","B2","V2")},
         "variants":{k:{"brier":v["overall"]["brier"],
                        "optimizer_steps":v["training"]["optimization_steps_total"],
                        "best_steps":v["training"]["optimization_steps_best"],
                        "epochs":v["training"]["epochs_completed"],
                        "sha256":v["training"]["sha256"]}
                      for k,v in reports["ablation"].items()},
         "selection":reports["selection"],"preflight":reports["preflight"]},ensure_ascii=False))
    return reports

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--frozen-v2",type=Path,required=True)
    p.add_argument("--outdir",type=Path,required=True)
    a=p.parse_args();run(a.dataset,a.frozen_v2,a.outdir)
if __name__=="__main__":main()
