"""PR44: real simulator PUBLIC-observation recovery survival study, CPU only.

Only after a workstation is observed DOWN is a future recovery distribution
estimated, using prior observed outages. No unobserved onset forecasting.
Private simulator outages are a scenario GENERATOR only, never model features.
Survival curves use conventional Kaplan-Meier event/risk-set computation with
right-censored historical outages and conditional survival S(a+h)/S(a).
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import argparse
import hashlib
import json
import math
import time

from business_transport.observation import BusinessObserver
from business_transport.policy import BusinessPolicy
from business_transport.scenario import compile_scenario
from business_transport.simulation import BusinessSimulator, resource_audit
from experiments.business_constraints.tests.test_business import fixture
from lab_transport.dynamics_model import Geometry
from lab_transport.multisite_experiment import discrete_safety_violations

PROTOCOL = Path(__file__).with_name("pr44_protocol.json")
METHODS = ("persist_down", "pooled_km", "station_km")
STATIONS = ("B0", "B1", "C")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def private_duration(spec, proto):
    """Used ONLY inside the sim constructor. Predictor never sees its output."""
    support = proto["synthetic_outage_design"]["group_duration_support"][spec["station_id"]]
    h = hashlib.sha256(("pr44|hidden-simulator|" + str(spec["seed"])).encode()).digest()
    return support[int.from_bytes(h[:8], "big") % len(support)]


def public_station_series(observations, station_id):
    """Only timestamped device_available; NOT resource-confounded 'available'."""
    series = []
    previous = -1
    for obs in observations:
        tick = obs["tick"]
        if tick != previous + 1:
            raise ValueError("missing or nonchronological public observation")
        station = obs["stations"][station_id]
        if "device_available" not in station:
            raise ValueError("device_available missing; cannot infer from capacity")
        series.append((tick, bool(station["device_available"])))
        previous = tick
    if not series:
        raise ValueError("no public observations")
    return series


def spells_from_series(series):
    """Train only on publicly observed down/up transitions.

    A down spell without recovery by the end of observation is RIGHT CENSORED.
    No end tick is imputed from known private simulator outage schedules.
    """
    spells = []
    start = None
    last = None
    for tick, up in series:
        if last is not None and tick != last + 1:
            raise ValueError("nonconsecutive station status")
        if start is None and not up:
            start = tick
        elif start is not None and up:
            spells.append({"observed_duration": tick - start,
                           "recovered": True, "onset_tick_observed": start})
            start = None
        last = tick
    if start is not None:
        spells.append({"observed_duration": last-start,
                       "recovered": False, "onset_tick_observed": start})
    return spells


def build_episode(spec, proto):
    """Run unchanged BusinessSimulator and keep ONLY its real public stream."""
    opts = proto["synthetic_outage_design"]
    base, raw_cfg, _ = fixture(capacity=opts["capacity"])
    stations = dict(base.stations)
    station = stations[spec["station_id"]]
    duration = private_duration(spec, proto)
    begin = opts["start_tick"]
    stations[spec["station_id"]] = replace(
        station, outages=((begin, begin + duration),))
    scene = replace(base, name="pr44-" + spec["id"],
                    stations=stations, max_ticks=opts["max_ticks"])
    jobs = [{"id":"J"+str(i+1),
             "workflow":"w0" if i%2==0 else "w1",
             "release_tick":0,"deadline_tick":opts["max_ticks"]-15}
            for i in range(opts["num_jobs"])]
    business = compile_scenario(
        scene, {**raw_cfg, "name":"pr44-"+spec["id"],"jobs":jobs,
                "max_ticks":opts["max_ticks"]})
    geometry = Geometry(scene.grid)
    observer = BusinessObserver()
    policy = BusinessPolicy("nearest", geometry, business.rules, observer)
    observed = []
    def on_observation(raw):
        state = observer.observe(raw)
        observed.append({
            "tick": state.tick,
            "stations": {
                s.id: {"device_available": bool(s.device_available),
                       "service_admission_available": bool(s.available),
                       "capacity_blocked": bool(s.capacity_blocked)}
                for s in state.stations}
        })
    factual = BusinessSimulator(
        business, "world", spec["seed"], collect_metrics=True).run(
            on_observation=on_observation, dispatch_policy=policy)
    audit = resource_audit(factual, business.rules)
    if audit["violations"] or discrete_safety_violations(factual):
        raise AssertionError("business real simulator violated motion/capacity safety")
    series = public_station_series(observed, spec["station_id"])
    if not any(not available for tick, available in series):
        raise AssertionError("declared outage never publicly observed")
    # Store observable status sequence, but NEVER private outage schedule.
    return {
        "id":spec["id"], "split":spec["id"].split("_",1)[0],
        "seed":spec["seed"], "station_id":spec["station_id"],
        "public_observations":observed,
        "simulator_status":factual["summary"]["status"],
        "simulator_end_tick":factual["summary"]["ticks"],
        "safety_violations":0,
        "public_spells":spells_from_series(series),
    }


def fit_survival(observed_spells):
    """Conventional discrete time Kaplan-Meier for known-age DOWN spells."""
    records = [(int(x["observed_duration"]), bool(x["recovered"]))
               for x in observed_spells]
    if not records:
        return {"records":0,"events":0,"curve":[[0,1.0]]}
    if any(d < 0 for d,_ in records):
        raise ValueError("negative observed downtime")
    times=sorted({d for d,event in records if event})
    survival=1.
    curve=[[0,1.]]
    for t in times:
        at_risk=sum(d>=t for d,_ in records)
        events=sum(d==t and event for d,event in records)
        if at_risk <= 0 or events > at_risk:
            raise AssertionError("invalid Kaplan-Meier risk set")
        survival *= 1.-events/at_risk
        if t==0:
            curve[0][1]=survival
        else:
            curve.append([t,survival])
    return {"records":len(records),"events":sum(e for _,e in records),
            "censored":sum(not e for _,e in records),"curve":curve}


def survival_at(km, t):
    if t < 0:raise ValueError("negative downtime age")
    p=1.
    for event_time, surv in km["curve"]:
        if event_time > t:break
        p=surv
    return p


def recovery_probability(km, age, horizon):
    """P(device recovers by t+h | device is observed down at t, age known)."""
    if age < 0 or horizon < 1:
        raise ValueError("invalid public age or forecast horizon")
    past=survival_at(km,age)
    if past <= 0:
        # No available risk support for such an old observed outage.
        return None
    eventual=survival_at(km,age+horizon)
    return max(0.,min(1.,1.-eventual/past))


def prediction_queries(observations,station,horizons):
    """Queries at down ticks, labels only when t+h is ACTUALLY observed."""
    series=public_station_series(observations, station)
    lookup=dict(series)
    down_start=None
    result=[]
    for t,up in series:
        if up:
            down_start=None
            continue
        if down_start is None:
            down_start=t
        for h in horizons:
            if t+h not in lookup:
                result.append({"tick":t,"age":t-down_start,"horizon":h,
                               "truth_available":None})
            else:
                result.append({"tick":t,"age":t-down_start,"horizon":h,
                               "truth_available":int(lookup[t+h])})
    return result


def score_episode(episode,models,horizons):
    """Never supply private repair date or future status to a predictor."""
    sid=episode["station_id"]
    queries=prediction_queries(episode["public_observations"],sid,horizons)
    by_h={}
    query_evidence=[]
    for h in horizons:
        own=[q for q in queries if q["horizon"]==h]
        scored=[q for q in own if q["truth_available"] is not None]
        scores={a:[] for a in METHODS}
        unsupported={a:0 for a in METHODS}
        for q in scored:
            age=q["age"]
            guesses={
                "persist_down":0.,
                "pooled_km":recovery_probability(models["pooled"],age,h),
                "station_km":recovery_probability(models["station"][sid],age,h)}
            query_evidence.append({
                "tick":q["tick"],"public_observed_down_age":age,
                "horizon":h,"offline_truth_available_by_h":q["truth_available"],
                "predicted_recovery_probabilities":guesses,
            })
            for name,p in guesses.items():
                if p is None:
                    unsupported[name]+=1
                    continue
                scores[name].append((p-q["truth_available"])**2)
        by_h[str(h)]={
            "known_label_queries":len(scored),
            "future_unobserved_queries":len(own)-len(scored),
            "brier_by_method":{a:sum(v)/len(v) if v else None
                               for a,v in scores.items()},
            "scored_by_method":{a:len(v) for a,v in scores.items()},
            "unsupported_age_by_method":unsupported,
        }
    return {"id":episode["id"],"seed":episode["seed"],"station_id":sid,
            "public_spells":episode["public_spells"],"horizons":by_h,
            "prediction_label_pairs":query_evidence,
            "safety_violations":episode["safety_violations"]}


def aggregate(rows,horizons):
    """Episode-level macro. Multiple queries from one outage are not IID."""
    output={}
    for h in horizons:
        block={}
        for group in ("overall",)+STATIONS:
            selected=[r for r in rows if group=="overall" or r["station_id"]==group]
            means={}
            coverage={}
            for m in METHODS:
                xs=[r["horizons"][str(h)]["brier_by_method"][m]
                    for r in selected]
                observed=[x for x in xs if x is not None]
                means[m]=sum(observed)/len(observed) if observed else None
                coverage[m]=len(observed)
            block[group]={
                "test_episodes":len(selected),
                "episodes_with_scored_forecast":coverage,
                "episode_macro_brier":means,
                "scored_public_queries":sum(
                    r["horizons"][str(h)]["known_label_queries"] for r in selected),
                "future_not_observed_queries":sum(
                    r["horizons"][str(h)]["future_unobserved_queries"] for r in selected),
            }
        output[str(h)]=block
    return output


def validate_protocol(cfg):
    if cfg["protocol"]!="PR44_PUBLIC_CONDITIONED_RECOVERY_SURVIVAL_20261112_V1":
        raise ValueError("changed preregistration protocol")
    if cfg["source_commit"]!="e6988a322d29f6224ead69ff26590236a9927919":
        raise ValueError("incorrect source commit")
    if cfg["station_ids"]!=list(STATIONS):
        raise ValueError("changed station cohort")
    if cfg["evaluation"]["horizons"] != [1,5,10]:
        raise ValueError("changed horizons")
    if cfg["evaluation"]["methods"] != list(METHODS):
        raise ValueError("changed comparator methods")
    train=cfg["cohort"]["train"]; test=cfg["cohort"]["test"]
    if len(train)!=36 or len(test)!=24:
        raise ValueError("changed frozen scenario counts")
    if len({x["seed"] for x in train+test})!=len(train+test):
        raise ValueError("seed reuse train versus held-out test")
    for sp in (train,test):
        each=len(sp)//3
        if any(sum(x["station_id"]==sid for x in sp)!=each for sid in STATIONS):
            raise ValueError("station strata changed")


def run(output):
    cfg=json.loads(PROTOCOL.read_text())
    validate_protocol(cfg)
    if output.exists():raise FileExistsError("write-once health evidence")
    began=time.process_time()
    train=[build_episode(s,cfg) for s in cfg["cohort"]["train"]]
    if time.process_time()-began>cfg["execution"]["cpu_seconds_limit"]:
        raise TimeoutError("PR44 training observation collection over CPU limit")
    spells_by_station={sid:[] for sid in STATIONS}
    for episode in train:
        spells_by_station[episode["station_id"]].extend(episode["public_spells"])
    models={
        "pooled":fit_survival(sum(spells_by_station.values(), [])),
        "station":{sid:fit_survival(spells_by_station[sid]) for sid in STATIONS}}
    # The frozen estimator is fitted only now, BEFORE held-out observations exist.
    heldout=[build_episode(s,cfg) for s in cfg["cohort"]["test"]]
    raw=[score_episode(e,models,cfg["evaluation"]["horizons"]) for e in heldout]
    summary=aggregate(raw,cfg["evaluation"]["horizons"])
    result={
        "schema":"pr44-observed-downstation-recovery-cpu-v1",
        "scope":"predeclared_synthetic_existing_simulator_observed_recovery_only",
        "protocol_sha256":sha(PROTOCOL),
        "source_commit":cfg["source_commit"],
        "gpu":0,"new_neural_training":0,"network_inference":0,
        "original_world_model_unchanged":True,
        "no_fault_oracle_model_features":True,
        "train_episodes":len(train),"heldout_episodes":len(heldout),
        "train_repair_models":models,
        "train_observed_spells":sum(len(v) for v in spells_by_station.values()),
        "train_censored_spells":sum(not s["recovered"]
                                    for v in spells_by_station.values() for s in v),
        "heldout_scored":raw,
        "public_status_streams":{
            split:[{
                "id":e["id"],"station_id":e["station_id"],
                "public_device_status":[{
                    "tick":obs["tick"],
                    "device_available":obs["stations"][e["station_id"]]["device_available"],
                    "service_admission_available":obs["stations"][e["station_id"]]["service_admission_available"],
                    "capacity_blocked":obs["stations"][e["station_id"]]["capacity_blocked"],
                } for obs in e["public_observations"]]
            } for e in collection]
            for split,collection in (("train",train),("heldout",heldout))
        },
        "summary":summary,
        "execution_cpu_seconds":time.process_time()-began,
        "limits":"Same-map synthetic availability schedule; station identity may proxy synthetic duration regime; cannot predict onset, no real telemetry or integrated H20 scheduling gains."
    }
    if result["execution_cpu_seconds"]>cfg["execution"]["cpu_seconds_limit"]:
        raise TimeoutError("PR44 CPU cap exceeded")
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print("PR44_OBSERVABLE_RECOVERY",json.dumps({
        "train":result["train_episodes"],
        "heldout":result["heldout_episodes"],
        "spells":result["train_observed_spells"],
        "censored":result["train_censored_spells"],
        "cpu":result["execution_cpu_seconds"],
        "brier_by_horizon":summary},ensure_ascii=False))
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args()
    run(args.out)


if __name__=="__main__":
    main()
