"""PR48 freeze: actual 7x5 business simulator with three nontrivial stress regimes.

Private simulator outage durations are sampled only here. Predictor sees
PublicBusinessRules and current observation; never this scenario's outage
schedule or future release table.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import hashlib
import json

from business_transport.scenario import compile_scenario
from experiments.business_constraints.tests.test_business import fixture
from experiments.world_model_integration.health_recovery.pr44_observed_recovery import PROTOCOL as PR44_PROTOCOL
from experiments.world_model_integration.recovery_shift.pr46_shared import PROTOCOL as PR46_PROTOCOL
from experiments.world_model_integration.recovery_budget.pr47_online import PROTOCOL as PR47_PROTOCOL
from lab_transport.dynamics_model import Geometry

PROTOCOL=Path(__file__).with_name("pr48_protocol.json")
REGIMES=("congestion","multi_fault","staggered_arrivals")
MECHANISMS=("within_support","extended_outage")
ARMS=("cp_sat","v4_original_15","v4_recovery_15")


def validate(config):
    if config.get("protocol")!="PR48_INDEPENDENT_THREE_STRESS_REGIMES_20261112_V1":
        raise ValueError("PR48 protocol identity mismatch")
    if (config.get("arms")!=list(ARMS) or
        config.get("fixed_per_decision_cpu_process_budget_seconds")!=1.5 or
        config.get("particles")!=3 or config.get("horizon_ticks")!=20 or
        config.get("candidate_limit")!=4):
        raise ValueError("fixed candidate/scoring/budget changed")
    if config["resources"]["gpu"] or config["resources"]["new_training"]:
        raise ValueError("new GPU or model training forbidden")
    if len(config["cohort"])!=48 or len({x["seed"] for x in config["cohort"]})!=48:
        raise ValueError("pre-registered 48-seed cohort corrupted")
    if set(x["regime"] for x in config["cohort"])!=set(REGIMES):
        raise ValueError("regimes changed")
    original=json.loads(PR47_PROTOCOL.read_text())
    if (config["source_pr47_head"]!="a2c01e66733ef27d36f9e59d22eb8292d4cb2103"
        or config["v4_sha256"]!=original["frozen_v4_sha256"]):
        raise ValueError("frozen V4 or baseline provenance mismatch")
    if original["online"]["particles"]!=config["particles"] or original["online"]["horizon"]!=config["horizon_ticks"] or original["online"]["candidate_limit"]!=config["candidate_limit"]:
        raise ValueError("original decision method changed")
    all_prior={x["seed"] for x in original["cohort"]}
    pr46=json.loads(PR46_PROTOCOL.read_text())
    all_prior.update(x["seed"] for v in pr46["cohort"].values() for x in v)
    pr44=json.loads(PR44_PROTOCOL.read_text())
    all_prior.update(x["seed"] for v in pr44["cohort"].values() for x in v)
    all_prior.update(7100009+s*10000+i*271 for s in range(3) for i in range(4))
    if all_prior & {x["seed"] for x in config["cohort"]}:
        raise ValueError("old tested scenarios reused")
    if (config["predefined_scenario_mechanisms"]["within_support"] !=
        pr46["hidden_environment_only"]["support_by_mechanism"]["within_support"]
        or config["predefined_scenario_mechanisms"]["extended_outage"] !=
        pr46["hidden_environment_only"]["support_by_mechanism"]["extended_outage"]):
        raise ValueError("outage duration support altered")
    for regime in REGIMES:
        selected=[x for x in config["cohort"] if x["regime"]==regime]
        if len(selected)!=16:
            raise ValueError("regime not balanced")
        for mech in MECHANISMS:
            if len([x for x in selected if x["mechanism"]==mech])!=8:
                raise ValueError("duration distribution not balanced")
    for x in config["cohort"]:
        if x["secondary_station_id"]!=("C" if x["regime"]=="multi_fault" else None):
            raise ValueError("unregistered secondary fault")
        if x["job_count"]!=(14 if x["regime"]=="congestion" else 12):
            raise ValueError("unregistered workload")
    return True


def _private_duration(spec,station,ordinal,protocol):
    support=protocol["predefined_scenario_mechanisms"][spec["mechanism"]][station]
    raw=f"pr48|hidden_environment|{spec['seed']}|{spec['regime']}|{spec['mechanism']}|{station}|{ordinal}".encode()
    index=int.from_bytes(hashlib.sha256(raw).digest()[:8],"big")%len(support)
    return support[index]


def build_business(spec,protocol):
    """Only the factual world constructor uses the external fault schedule."""
    base,cfg,_=fixture(capacity=1)
    meta=protocol["predefined_scenario_mechanisms"]
    stations=dict(base.stations)
    primary=spec["station_id"]
    events=[(primary,meta["outage_start_tick"])]
    if spec["regime"]=="multi_fault":
        events.append((spec["secondary_station_id"],meta["secondary_outage_start_tick"]))
    if len({x[0] for x in events})!=len(events):
        raise ValueError("multi-fault scenario requires two distinct stations")
    for ordinal,(station,start) in enumerate(events):
        duration=_private_duration(spec,station,ordinal,protocol)
        original=stations[station]
        stations[station]=replace(original,outages=((start,start+duration),))
    family=meta[spec["regime"]]
    releases=family["release_ticks"]
    jobs=[{
        "id":f"J{i+1}",
        "workflow":"w0" if i%2==0 else "w1",
        "release_tick":releases[i%len(releases)],
        "deadline_tick":family["deadline"]
    } for i in range(spec["job_count"])]
    case=replace(base,
                 name="pr48-"+spec["id"],
                 stations=stations,max_ticks=family["max_ticks"])
    result=compile_scenario(case,{
        **cfg,"name":"pr48-"+spec["id"],
        "jobs":jobs,"max_ticks":family["max_ticks"]})
    return result,Geometry(case.grid)
