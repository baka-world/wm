"""PR46 shared frozen synthetic environment and public recovery contracts.

Hidden outage configuration belongs ONLY to real simulator construction.
Predictors get public BusinessObserver state, geometry, declared business rules
and the archived train-only KM estimates; no private duration or schedule.
"""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import hashlib
import json

from business_transport.scenario import compile_scenario
from experiments.business_constraints.tests.test_business import fixture
from experiments.world_model_integration.recovery_integration.pr45_recovery_world import (
    PublicDownClock, ObservedRecoveryWorld, load_archived_km,
)
from experiments.world_model_integration.health_recovery.pr44_observed_recovery import PROTOCOL as PR44_PROTOCOL
from experiments.world_model_integration.system_world_model.evaluate_horizons import file_sha
from lab_transport.dynamics_model import Geometry

PROTOCOL = Path(__file__).with_name("pr46_protocol.json")
FORECAST_ARMS = ("v4_original", "v4_public_km_recovery")
ONLINE_ARMS = ("cp_sat", "v4_original", "v4_public_km_recovery")
STATIONS = ("B0", "B1", "C")
MECHANISMS = ("within_support", "extended_outage")


def _hidden_duration(spec, proto):
    """TRUE simulator builder only. Callers cannot pass this into predictors."""
    durations = proto["hidden_environment_only"]["support_by_mechanism"][
        spec["mechanism"]][spec["station_id"]]
    raw = f"pr46-true-world|{spec['seed']}|{spec['mechanism']}|{spec['station_id']}".encode()
    idx = int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % len(durations)
    return durations[idx]


def build_business(spec, proto):
    """Original simulator scenario with NEW external outage duration mechanism."""
    setting = proto["hidden_environment_only"]
    base, raw_cfg, _ = fixture(capacity=setting["capacity"])
    stations = dict(base.stations)
    sid = spec["station_id"]
    original = stations[sid]
    initial = setting["outage_start_tick"]
    duration = _hidden_duration(spec, proto)
    stations[sid] = replace(original, outages=((initial, initial + duration),))
    scene = replace(
        base, name="pr46-" + spec["id"], stations=stations,
        max_ticks=setting["max_ticks"])
    jobs = [
        {"id":f"J{i+1}", "workflow":"w0" if i%2==0 else "w1",
         "release_tick":0, "deadline_tick":setting["deadline_tick"]}
        for i in range(setting["job_count"])
    ]
    business = compile_scenario(
        scene, {**raw_cfg, "name":"pr46-" + spec["id"],
                "jobs":jobs, "max_ticks":setting["max_ticks"]})
    return business, Geometry(scene.grid)


def frozen_setup(checkpoint, pr44_result):
    p = json.loads(PROTOCOL.read_text())
    validate_protocol(p)
    if file_sha(checkpoint) != p["frozen_original_v4_sha256"]:
        raise ValueError("wrong original V4 checkpoint")
    km = load_archived_km(pr44_result)
    return p, km


def validate_protocol(p):
    if p.get("protocol") != "PR46_OBSERVED_RECOVERY_SHIFT_VS_ONLINE_20261112_V1":
        raise ValueError("PR46 protocol identity changed")
    if p["source_pr45_head"] != "805bfb2e35c3257f69fa6d96aaa854b61b611cf5":
        raise ValueError("PR45 original source mismatch")
    if p["forecast"]["arms"] != list(FORECAST_ARMS) or p["online"]["arms"] != list(ONLINE_ARMS):
        raise ValueError("unregistered policy arm")
    if (p["forecast"]["anchors"] != [8,10]
        or p["forecast"]["horizons"] != [5,20]
        or p["forecast"]["particles"] != 3
        or p["online"]["horizon"] != 20 or p["online"]["particles"] != 3
        or p["online"]["candidate_limit"] != 4
        or p["online"]["decision_cpu_budget_seconds"] != .5):
        raise ValueError("frozen forecast/online configurations changed")
    if p["resources"]["gpu"] != 0 or p["resources"]["new_training"] != 0:
        raise ValueError("training or GPU prohibited")
    if len(p["cohort"]["forecast"]) != 24 or len(p["cohort"]["online"]) != 24:
        raise ValueError("unexpected cohort size")
    all_cases=p["cohort"]["forecast"]+p["cohort"]["online"]
    if len({x["seed"] for x in all_cases})!=48:
        raise ValueError("duplicate/overlapping cohort seeds")
    for track in ("forecast","online"):
        for mech in MECHANISMS:
            for sid in STATIONS:
                cases=[x for x in p["cohort"][track]
                       if x["mechanism"]==mech and x["station_id"]==sid]
                if len(cases)!=4 or any(x["track"]!=track for x in cases):
                    raise ValueError("pre-registered stratum changed")
    older=json.loads(PR44_PROTOCOL.read_text())
    old_seeds={x["seed"] for k in ("train","test") for x in older["cohort"][k]}
    # PR45 12 independent forecast seeds.
    pr45_seeds={7100009 + s*10000 + i*271 for s in range(3) for i in range(4)}
    if old_seeds & {x["seed"] for x in all_cases} or pr45_seeds & {x["seed"] for x in all_cases}:
        raise ValueError("independent scenes overlap PR44 or PR45")
    original_support=older["synthetic_outage_design"]["group_duration_support"]
    if p["hidden_environment_only"]["support_by_mechanism"]["within_support"]!=original_support:
        raise ValueError("original synthetic support changed")
    shifted=p["hidden_environment_only"]["support_by_mechanism"]["extended_outage"]
    if any(max(shifted[sid])<=max(original_support[sid]) for sid in STATIONS):
        raise ValueError("shifted recovery mechanism must exceed original maximum")
    return True
