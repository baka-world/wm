"""Targeted, observable service-session queries (business-service-query-v2).

One query identifies one currently *started* station service. No private
remaining-time, future fault calendar, unreleased job or counterfactual labels.
"""
from __future__ import annotations

from lab_transport.dynamics_model import _intent

QUERY_LABEL_VERSION = "business-service-query-v2"
QUERY_WIDTH = 48


def eligible_sessions(observation, assignments=(), *, require_active=False):
    """Return (query, reason) rows; non-started service is never a negative label."""
    intents = _intent(observation, tuple(assignments))
    samples = {s.id: s for s in observation.samples}
    stations = {s.id: s for s in observation.stations}
    results = []
    for robot in sorted(observation.robots, key=lambda r:r.id):
        phase, sample_id = intents[robot.id]
        if sample_id is None:
            continue
        sample = samples[sample_id]
        station_id = sample.pickup if phase == "to_pickup" else sample.dropoff
        station = stations[station_id]
        if robot.position != station.position or robot.task_id != sample_id or robot.phase != phase:
            continue
        query = {"robot_id": robot.id, "sample_id": sample_id,
                 "phase": phase, "station_id": station_id}
        if robot.service_elapsed is None:
            reason = "not_started"
        elif getattr(robot, "service_active_elapsed", None) is None or getattr(robot, "service_paused_elapsed", None) is None:
            reason = "missing_observed_progress"
        elif not getattr(station, "device_available", station.available):
            reason = "observed_device_pause"
        elif getattr(station,"capacity_blocked",False) or not station.available:
            reason = "resource_or_admission_pause"
        else:
            reason = None
        if reason is None or not require_active:
            results.append((query, reason))
    return results


def _scalar(field, scale=1.):
    if not isinstance(field, dict) or type(field.get("known")) is not bool:
        raise ValueError("missing explicit field mask")
    if not field["known"]:
        if field.get("value") is not None:
            raise ValueError("unknown scalar carries a value")
        return [0.,0.]
    v = field.get("value")
    if isinstance(v,bool):v=float(v)
    if not isinstance(v,(int,float)) or not -1e12 < float(v) < 1e12:
        raise ValueError("invalid known numeric scalar")
    return [max(-4.,min(4.,float(v)/scale)),1.]


def encode_query(payload, query):
    """Numerical target joining robot, sample, station and candidate resources."""
    from business_transport.state_features import SCHEMA_VERSION
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("business observation feature schema mismatch")
    obs = payload["observed_facts"]
    robots = {row["id"]:row for row in obs["robots"]}
    samples = {row["id"]:row for row in obs["samples"]}
    stations = {row["id"]:row for row in obs["stations"]}
    try:
        robot = robots[query["robot_id"]]
        sample = samples[query["sample_id"]]
        station = stations[query["station_id"]]
        phase = query["phase"]
    except (KeyError,TypeError) as e:
        raise ValueError("query refers to absent observed entity") from e
    if phase not in ("to_pickup","to_dropoff"):
        raise ValueError("unknown service phase")
    if sample["pickup" if phase=="to_pickup" else "dropoff"] != station["id"]:
        raise ValueError("service query target station mismatch")
    if robot["phase"] != phase or not robot["current_task"]["known"] or robot["current_task"]["value"] != sample["id"]:
        raise ValueError("query robot/sample relationship mismatch")
    if tuple(robot["position"]) != tuple(station["position"]):
        raise ValueError("query robot not at observed service station")

    v = [float(phase=="to_pickup"), float(phase=="to_dropoff"),
         robot["position"][0]/30.,robot["position"][1]/30.,
         station["position"][0]/30.,station["position"][1]/30.]
    for row,key,scale in (
        (robot,"service_elapsed",30),
        (robot,"service_active_elapsed",30),
        (robot,"service_paused_elapsed",30),
        (robot,"operational_available",1),
        (sample,"deadline_tick",60),
        (sample,"priority",10),
        (sample,"public_stages_total",5),
        (station,"device_available",1),
        (station,"service_admission_available",1),
        (station,"capacity_blocked",1),
        (station,"resource_capacity",10),
        (station,"resource_occupied",10),
        (station,"waiting_pickups",10),
        (station,"approaching_pickups",10),
        (station,"approaching_dropoffs",10),
        (station,"observed_effective_service_mean",10),
        (station,"observed_effective_service_count",20),
        (station,"history_complete_since_start",1),
    ):
        v.extend(_scalar(row[key],scale))
    v.extend((sample["stage_index"]/5.,
              len(sample["completed_stage_indices"])/5.,
              len(sample["public_remaining_route"])/5.))
    resource_id = station["resource_id"]["value"]
    competition = sum(
        (a["pickup_resource"]==resource_id) + (a["dropoff_resource"]==resource_id)
        for a in payload["candidate_action"])
    v.extend((min(competition,10)/10.,
              len(payload["candidate_action"])/10.,
              len(obs["robots"])/10.))
    if len(v)>QUERY_WIDTH:
        raise ValueError("query projection exceeds reserved width")
    return v+[0.]*(QUERY_WIDTH-len(v))
