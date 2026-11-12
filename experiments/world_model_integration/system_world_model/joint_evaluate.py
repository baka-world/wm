"""Frozen 12-group independent future-state test: legal transport-risk coupling.

V4 and Rule use the same factual trajectory; joint_shadow must remain
bit-exact to V4 on all future states. sampled_hold is a separate experimental
event-time adapter with no promised gain. No new training or new scene tuning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from business_transport.world_model import HybridSystemWorldModel
from business_transport.world_model.joint_transport import JointServiceTransportWorldModel
from business_transport.world_model.transport_bayes import TransportBayesModel
from experiments.world_model_integration.system_world_model.evaluate_horizons import (
    METRICS, factual_episode, group_scenario, score, file_sha,
)
from lab_transport.decision_value import candidates_for,nearest_batch


ROOT=Path(__file__).resolve().parent
PROTOCOL=ROOT.parent/"transport_events"/"transport_protocol.json"
ARMS=("rule","v4","joint_shadow","joint_sampled_hold")
HORIZONS=(1,5,10)
ANCHORS=(0,5,12,20,30)


def evaluate(*,v4_checkpoint,pgmpy_json,cpu_budget=240):
    cfg=json.loads(PROTOCOL.read_text())
    if cfg["protocol"]!="TRANSPORT_PUBLIC_EVENTS_V1":
        raise AssertionError("transport seeded protocol changed")
    if cfg["max_ticks"]!=240 or len(cfg["future_horizon_groups"])!=12:
        raise ValueError("frozen 12 independent groups missing")
    from business_transport.world_model.transport_bayes import TransportBayesModel
    model_data=json.loads(Path(pgmpy_json).read_text())
    bayes=TransportBayesModel.from_payload(model_data)
    if bayes.total_legs!=240:
        raise AssertionError("transport model is not PR25 fit on frozen 24 training scenes")
    from torch import set_num_threads
    set_num_threads(1)
    start=time.process_time()
    reports=[]
    for spec in cfg["future_horizon_groups"]:
        if time.process_time()-start>cpu_budget:
            raise TimeoutError("exceeded fixed CPU evaluation budget")
        business,geo=group_scenario(spec,max_ticks=cfg["max_ticks"])
        states,actions,factual=factual_episode(business,geo,spec["seed"])
        models={
            "rule":HybridSystemWorldModel(geo,business.rules,kind="rule"),
            "v4":HybridSystemWorldModel(geo,business.rules,kind="v4",
                                        checkpoint=v4_checkpoint,experimental=True),
            "joint":JointServiceTransportWorldModel(
                geo,business.rules,v4_checkpoint=v4_checkpoint,
                transport_model=bayes),
        }
        arm_rows={key:[] for key in ARMS}
        risks={key:0 for key in ARMS}
        holds=0
        for anchor in ANCHORS:
            if anchor not in actions:continue
            available=[h for h in HORIZONS if anchor+h in states]
            if not available:continue
            first=actions[anchor]
            obs=states[anchor]
            seed=1810+spec["seed"]*101+anchor
            opts=dict(
                continuation_policy=lambda st:nearest_batch(
                    st,candidates_for(st),geo),
                continuation_policy_name="nearest_public",
                horizon_ticks=max(available),particles=3,sampling_seed=seed)
            rule=models["rule"].rollout(obs,first,**opts)
            v4=models["v4"].rollout(obs,first,**opts)
            shadow=models["joint"].rollout(obs,first,mode="shadow",**opts)
            sampled=models["joint"].rollout(obs,first,mode="sampled_hold",**opts)
            # Shadow is a strict augmentation; adding risk metadata must not
            # secretly change V4 motion, services or business state.
            if shadow.forecast.distributions!=v4.distributions:
                raise AssertionError("shadow risk changed frozen V4 state forecast")
            if shadow.forecast.particles!=v4.particles:
                # The joint wrapper has its own model_version name, so compare
                # state sequences instead of decorated ForecastStep equality.
                for a,b in zip(shadow.particles,v4.particles):
                    if [s.next_state for s in a.steps]!=[s.next_state for s in b.steps]:
                        raise AssertionError("shadow altered V4 factual trajectory")
            risks["joint_shadow"]+=len(shadow.transport_risks)
            risks["joint_sampled_hold"]+=len(sampled.transport_risks)
            holds+=sampled.applied_motion_holds
            for h in available:
                truth=states[anchor+h]
                for name,forecast in (
                    ("rule",rule),("v4",v4),
                    ("joint_shadow",shadow.forecast),
                    ("joint_sampled_hold",sampled.forecast)):
                    arm_rows[name].append({
                        "anchor":anchor,"horizon":h,"metrics":score(forecast,truth,h),
                    })
        reports.append({
            "group":spec["id"],"seed":spec["seed"],
            "truth_hash":hashlib.sha256(
                json.dumps(factual["events"],sort_keys=True).encode()).hexdigest(),
            "arms":arm_rows,"risk_queries":risks,
            "sampled_motion_holds":holds,
        })
    summary={}
    for h in HORIZONS:
        arms={}
        for name in ARMS:
            group_metrics=[]
            for group in reports:
                rows=[r["metrics"] for r in group["arms"][name] if r["horizon"]==h]
                if not rows:continue
                group_metrics.append({k:sum(row[k] for row in rows)/len(rows)
                                     for k in METRICS})
            if not group_metrics:
                raise AssertionError("incomplete independent horizon test")
            arms[name]={"group_count":len(group_metrics),
                        "mean":{k:sum(g[k] for g in group_metrics)/len(group_metrics)
                                for k in METRICS}}
        summary[str(h)]=arms
    elapsed=time.process_time()-start
    if elapsed>cpu_budget:raise TimeoutError("fixed CPU budget exceeded")
    return {
        "schema":"rhwm-joint-transport-service-horizon-v1",
        "frozen_transport_protocol_sha256":file_sha(PROTOCOL),
        "pgmpy_model_sha256":file_sha(pgmpy_json),
        "frozen_v4_checkpoint_sha256":file_sha(v4_checkpoint),
        "groups":reports,
        "summary":{
            "groups":len(reports),"by_horizon":summary,
            "transport_risk_queries":sum(x["risk_queries"]["joint_shadow"] for x in reports),
            "sampled_motion_holds":sum(x["sampled_motion_holds"] for x in reports),
            "gpu":0,"new_training":0,"cpu_seconds":elapsed,
            "mode_explanation":"shadow=V4 state+transport risk; sampled_hold=experimental one-event-per-leg dwell delay",
            "caveat":"Synthetic-only, preselected seed groups; sampled_hold is engineering event-time mapping, not original Tang/Wang result",
        }
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--transport",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():raise FileExistsError("heldout results immutable")
    v=evaluate(v4_checkpoint=args.checkpoint,pgmpy_json=args.transport)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(v,indent=2,ensure_ascii=False)+"\n")
    text={h:{arm:round(x["mean"]["sample_status_error"],6)
             for arm,x in results.items()}
          for h,results in v["summary"]["by_horizon"].items()}
    print("PR26_JOINT_WORLD_MODEL_VALIDATED",json.dumps({
        "groups":v["summary"]["groups"],
        "sample_state_error_by_horizon":text,
        "transport_risk_queries":v["summary"]["transport_risk_queries"],
        "sampled_motion_holds":v["summary"]["sampled_motion_holds"],
        "cpu_seconds":v["summary"]["cpu_seconds"]}))


if __name__=="__main__":
    main()
