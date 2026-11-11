"""Reference-method migration: lifelines KM versus V4 and Rule on actual episodes.

Not a new learning algorithm. Uses KaplanMeierFitter from lifelines==0.30.0.
The only project-specific work is observable service-event extraction, method
adaptation to RHWM's lawful transition interface, and grouped evaluations.

The research protocol and group IDs are sealed in km_reference_protocol.json
before any evaluation runs; no prior PR21 holdout is reused.
"""
from __future__ import annotations

from collections import defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import time

from business_transport.session_events import eligible_sessions
from business_transport.world_model import HybridSystemWorldModel
from business_transport.world_model.kaplan_meier import KaplanMeierServiceModel
from experiments.world_model_integration.system_world_model.evaluate_horizons import (
    METRICS, factual_episode, group_scenario, score, file_sha,
)
from lab_transport.decision_value import candidates_for, nearest_batch

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "km_reference_protocol.json"


def extract_sessions(factual_events, *, end_tick):
    """Match public service_started/finished, treating unfinished as censored."""
    started, records = {}, []
    for e in factual_events:
        if e["type"] not in ("service_started", "service_finished"):
            continue
        key = (e["robot_id"], e["sample_id"], e["phase"], e["station_id"])
        if e["type"] == "service_started":
            if key in started:
                raise AssertionError("duplicate observed service start")
            started[key] = e["tick"]
            continue
        if key not in started:
            raise AssertionError("service finished without a prior public start")
        duration = e["tick"] - started.pop(key)
        if duration < 1 or duration != e["observed_ticks"]:
            raise AssertionError("service-time event log inconsistent")
        records.append({
            "station_id":e["station_id"], "duration":duration, "event_observed":True
        })
    for (_, _, _, sid), tick in sorted(started.items()):
        elapsed = end_tick - tick
        if elapsed > 0:
            records.append({
                "station_id":sid,"duration":elapsed,"event_observed":False
            })
    return records


def _query_key(query):
    return (query["robot_id"],query["sample_id"],query["phase"],query["station_id"])


def event_brier_on_factual(model, states, actions):
    """Existing current-session V4/KM probability vs next factual event."""
    losses = []
    for t in sorted(actions):
        if t + 1 not in states:
            continue
        before, after, action = states[t], states[t+1], actions[t]
        viable = [q for q,reason in eligible_sessions(before, action) if reason is None]
        if not viable:
            continue
        prediction = model.predict_next(before, action, sampling_seed=5117+t)
        events = {_query_key(e.__dict__):e for e in prediction.service_events}
        # EventForecast stores attributes rather than the 'query' dictionary.
        if len(events) != len(prediction.service_events):
            raise AssertionError("repeated service query")
        if set(events) != {_query_key(q) for q in viable}:
            raise AssertionError("event queried by predictor differs from public eligibility")
        following = {s.id:s.status for s in after.samples}
        for q in viable:
            p = events[_query_key(q)].p_finish_next
            success_status = "in_transit" if q["phase"] == "to_pickup" else "completed"
            y = int(following[q["sample_id"]] == success_status)
            losses.append((p-y)**2)
    return {
        "eligible_queries":len(losses),
        "brier":sum(losses)/len(losses) if losses else None,
    }


def state_rollout_on_factual(model, states, actions, geo, protocol):
    rows = []
    for anchor in protocol["fixed_anchors"]:
        if anchor not in actions:
            continue
        legal = [h for h in protocol["fixed_horizons"] if anchor+h in states]
        if not legal:
            continue
        next_action = actions[anchor]
        if any(a not in candidates_for(states[anchor]) for a in next_action):
            raise AssertionError("actual first action was not public and legal")
        forecast = model.rollout(
            states[anchor],next_action,
            continuation_policy=lambda obs: nearest_batch(obs,candidates_for(obs),geo),
            continuation_policy_name="nearest_batch_public",
            horizon_ticks=max(legal),
            particles=protocol["particles"],
            sampling_seed=protocol["seed_for_model"]+anchor)
        for h in legal:
            rows.append({"anchor":anchor,"horizon":h,
                         "metrics":score(forecast,states[anchor+h],h)})
    if not rows:
        raise AssertionError("zero eligible predeclared anchors")
    return rows


def grouped_metric(groups, arms, horizons):
    results = {}
    for split in sorted({g["split"] for g in groups}):
        subset = [g for g in groups if g["split"] == split]
        by_horizon = {}
        for h in horizons:
            by_arm = {}
            for arm in arms:
                per_group = []
                for g in subset:
                    chosen = [r["metrics"] for r in g["arms"][arm]["state_rows"]
                              if r["horizon"] == h]
                    if not chosen:
                        continue
                    per_group.append({key:sum(row[key] for row in chosen)/len(chosen)
                                      for key in METRICS})
                if not per_group:
                    raise AssertionError("missing scenario-level results")
                by_arm[arm] = {
                    "groups":len(per_group),
                    "mean":{key:sum(x[key] for x in per_group)/len(per_group)
                             for key in METRICS},
                }
            by_horizon[str(h)] = by_arm
        by_brier = {}
        for arm in ("v4","kaplan_meier"):
            vals = [g["arms"][arm]["service_event"]["brier"]
                    for g in subset if g["arms"][arm]["service_event"]["brier"] is not None]
            n = sum(g["arms"][arm]["service_event"]["eligible_queries"] for g in subset)
            by_brier[arm] = {"groups":len(vals),"eligible_queries":n,
                             "scenario_mean":sum(vals)/len(vals) if vals else None}
        results[split] = {
            "groups":len(subset), "by_horizon":by_horizon,
            "service_event_brier":by_brier,
        }
    return results


def evaluate(*, checkpoint, protocol=PROTOCOL, cpu_budget=300):
    conf = json.loads(Path(protocol).read_text())
    if (conf["protocol"]!="RHWM_REFERENCE_KM_TRANSFER_20261111_V1"
            or conf["fixed_horizons"]!=[1,5,10]
            or conf["particles"]!=3
            or conf["method"]!="Kaplan-Meier product-limit survival estimator as implemented in lifelines 0.30.0"):
        raise ValueError("reference method and protocol must stay frozen")
    if file_sha(checkpoint)!=conf["frozen_v4_sha256"]:
        raise ValueError("wrong frozen V4 checkpoint")
    if not 1 <= cpu_budget <= conf["resource_budget"]["cpu_process_seconds"]:
        raise ValueError("invalid CPU budget")
    import lifelines
    from torch import set_num_threads
    assert lifelines.__version__ == "0.30.0"
    set_num_threads(1)
    started = time.process_time()
    sessions = []
    training = []
    train_seed_set = {x["seed"] for x in conf["train_groups"]}
    test_seed_set = {x["seed"] for x in conf["test_groups"]}
    if train_seed_set & test_seed_set:
        raise AssertionError("leakage: overlapping train/test scene seed")
    if len(test_seed_set)!=len(conf["test_groups"]):
        raise AssertionError("repeated test group")
    for spec in conf["train_groups"]:
        if time.process_time()-started > cpu_budget:
            raise TimeoutError("CPU budget exhausted in training data")
        business, geo = group_scenario(spec,max_ticks=conf["max_ticks"])
        _, _, factual = factual_episode(business, geo, spec["seed"])
        if any(e["type"] in ("device_unavailable", "outage") for e in factual["events"]):
            raise AssertionError("outage violates service-active time calibration protocol")
        items = extract_sessions(factual["events"], end_tick=factual["summary"]["ticks"])
        sessions.extend(items)
        training.append({
            "id":spec["id"],"seed":spec["seed"],
            "completed_sessions":sum(row["event_observed"] for row in items),
            "censored_sessions":sum(not row["event_observed"] for row in items),
        })
    service = KaplanMeierServiceModel(sessions)
    summary = {
        "method":"Kaplan-Meier (1958), lifelines 0.30.0",
        "license":"MIT",
        "implementation":"lifelines.KaplanMeierFitter",
        "number_of_train_sessions":len(sessions),
        "station_event_counts":service.group_counts,
        "training":training,
        "station_survival_curves":{
            station:[{"t":float(t),"s":float(row.iloc[0])}
                     for t,row in fitter.survival_function_.iterrows()]
            for station,fitter in service.models.items()
        }
    }
    test_rows = []
    for spec in conf["test_groups"]:
        if time.process_time()-started > cpu_budget:
            raise TimeoutError("CPU budget exhausted before next test group")
        business, geo = group_scenario(spec,max_ticks=conf["max_ticks"])
        states, actions, factual = factual_episode(business,geo,spec["seed"])
        result = {
            "id":spec["id"],"seed":spec["seed"],"split":spec["split"],
            "event_log_hash":hashlib.sha256(
                json.dumps(factual["events"],sort_keys=True).encode()).hexdigest(),
            "test_factual_sessions":len(extract_sessions(
                factual["events"], end_tick=factual["summary"]["ticks"])),
            "arms":{},
        }
        for arm in ("rule","kaplan_meier","v4"):
            model = HybridSystemWorldModel(
                geo,business.rules,kind=arm,
                checkpoint=checkpoint if arm=="v4" else None,
                experimental=(arm=="v4"),
                service_baseline=service if arm=="kaplan_meier" else None)
            # Primary output for an external reference model is next-step
            # event probability. State forecasts remain paired, secondary.
            events = (
                event_brier_on_factual(model, states, actions)
                if arm != "rule" else {"eligible_queries":0,"brier":None})
            rows = state_rollout_on_factual(model,states,actions,geo,{
                **conf, "seed_for_model":500100+spec["seed"]*7})
            result["arms"][arm] = {
                "service_event":events,
                "state_rows":rows,
            }
        test_rows.append(result)
    return {
        "protocol":conf["protocol"],"protocol_sha256":file_sha(protocol),
        "checkpoint_sha256":file_sha(checkpoint),
        "source_library":"lifelines==0.30.0",
        "source_method":"Kaplan-Meier product-limit estimator",
        "fitted_reference":summary,
        "group_results":test_rows,
        "summary":{
            "train_groups":len(training),
            "test_groups":len(test_rows),
            "group_summaries":grouped_metric(
                test_rows,("rule","kaplan_meier","v4"),conf["fixed_horizons"]),
            "cpu_process_seconds":time.process_time()-started,
            "gpu":0,"new_neural_training":0,
            "actual_observation_source":"BusinessObserver and public event log",
            "truth_used_only_for_labels":True,
            "limitations":"Synthetic station duration-jitter; not AGV motion delay or real enterprise log",
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--cpu-budget",type=int,default=300)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError("reference method results must not be overwritten")
    result = evaluate(checkpoint=args.checkpoint,cpu_budget=args.cpu_budget)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print("REFERENCE_TRANSFER_RESULT",json.dumps(result["summary"],ensure_ascii=False))


if __name__ == "__main__":
    main()
