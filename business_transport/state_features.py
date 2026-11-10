"""Business-state observation contract v1, independent of all legacy checkpoints.

Only current derived observations and configured public rules are accepted.
Unknown data is represented as {known: false, value: null}, never silently
converted to a factual zero. This contract is NOT the PETS seven-feature or
Structured 26-wide token contract.
"""
from __future__ import annotations

from lab_transport.dynamics_model import _intent

SCHEMA_VERSION = "business-observable-v1"
LABEL_VERSION = "business-service-event-v1"

FIELD_SOURCES = {
    "robot_position_phase_task": "current ObservationV2.robots",
    "service_wall_active_paused": "ObservableHistory-derived RobotV2",
    "robot_operational_available": "RobotV2 if exposed; otherwise unknown",
    "sample_job_stage_status": "current BusinessSampleV2; known released stages only",
    "sample_release_deadline": "current sample public fields",
    "workflow_future_stages": "PublicBusinessRules.workflows for an already released job",
    "station_device_capacity_occupation": "current BusinessStationV2 and PublicBusinessRules.resources",
    "station_session_wait_approach": "current StationV2 relationship counts",
    "service_effective_history": "ObservableHistory completed service statistics",
    "candidate_assignment": "current legal candidate assignments",
    "candidate_distance_resource": "public geometry and PublicBusinessRules.station_resources",
    "unknown_future_fault": "not available; explicitly masked",
    "unknown_future_external_jobs": "not available; explicitly masked",
    "hidden_service_countdown": "not available; explicitly masked",
}


def _value(value):
    return {"known": value is not None, "value": value}


def _station(station):
    return {
        "id": station.id,
        "position": station.position,
        "service_admission_available": _value(station.available),
        "device_available": _value(getattr(station, "device_available", None)),
        "resource_id": _value(getattr(station, "resource_id", None)),
        "resource_capacity": _value(getattr(station, "resource_capacity", None)),
        "resource_occupied": _value(getattr(station, "resource_occupied", None)),
        "capacity_blocked": _value(getattr(station, "capacity_blocked", None)),
        "occupant_robot_id": _value(getattr(station, "occupant_robot_id", None)),
        "service_task_id": _value(getattr(station, "service_task_id", None)),
        "service_session_active": _value(getattr(station, "service_session_active", None)),
        "service_in_progress": _value(getattr(station, "service_in_progress", None)),
        "waiting_pickups": _value(getattr(station, "waiting_pickups", None)),
        "approaching_pickups": _value(getattr(station, "approaching_pickups", None)),
        "approaching_dropoffs": _value(getattr(station, "approaching_dropoffs", None)),
        "expected_service_ticks_public": _value(station.expected_service_ticks),
        "observed_effective_service_count": _value(getattr(station, "observed_effective_service_count", None)),
        "observed_effective_service_mean": _value(getattr(station, "observed_effective_service_mean", None)),
        "history_complete_since_start": _value(getattr(station, "history_complete_since_start", None)),
    }


def build_business_features(observation, assignments, geometry, rules):
    """Build public, action-conditioned features; neither oracle nor simulator accepted."""
    assignments = tuple(assignments)
    intents = _intent(observation, assignments)  # validates current action semantics
    stations = {s.id: s for s in observation.stations}
    samples = {s.id: s for s in observation.samples}
    robots = {r.id: r for r in observation.robots}
    if len(stations) != len(observation.stations) or len(samples) != len(observation.samples):
        raise ValueError("duplicate observed station or sample")
    if len(robots) != len(observation.robots):
        raise ValueError("duplicate observed robot")
    for assignment in assignments:
        if assignment.robot_id not in robots or assignment.sample_id not in samples:
            raise ValueError("candidate references unobserved robot or sample")
    robot_features = []
    for robot in sorted(observation.robots, key=lambda x: x.id):
        robot_features.append({
            "id": robot.id, "position": robot.position, "phase": robot.phase,
            "current_task": _value(robot.task_id),
            "service_elapsed": _value(robot.service_elapsed),
            "service_active_elapsed": _value(getattr(robot, "service_active_elapsed", None)),
            "service_paused_elapsed": _value(getattr(robot, "service_paused_elapsed", None)),
            "operational_available": _value(getattr(robot, "operational_available", None)),
            "candidate_distances": [{
                "sample_id": item.sample_id,
                "to_pickup": geometry.distance(robot.position, stations[samples[item.sample_id].pickup].position),
                "to_dropoff": geometry.distance(robot.position, stations[samples[item.sample_id].dropoff].position)}
                for item in assignments if item.robot_id == robot.id],
        })
    groups = {}
    for sample in observation.samples:
        job = getattr(sample, "job_id", "") or sample.id
        groups.setdefault(job, []).append(sample)
    sample_features = []
    for sample in sorted(observation.samples, key=lambda x: x.id):
        workflow = getattr(sample, "workflow", "")
        stage = getattr(sample, "stage_index", 0)
        route = rules.workflows.get(workflow) if workflow else None
        if route is not None and (not isinstance(stage, int) or stage < 0 or stage >= len(route)):
            raise ValueError("current sample workflow stage invalid")
        job = getattr(sample, "job_id", "") or sample.id
        visible = groups[job]
        sample_features.append({
            "id": sample.id, "job_id": job, "workflow": _value(workflow or None),
            "stage_index": stage, "status": sample.status,
            "pickup": sample.pickup, "dropoff": sample.dropoff,
            "robot_id": _value(sample.robot_id), "release_tick": sample.release_tick,
            "deadline_tick": _value(sample.deadline_tick),
            "priority": _value(getattr(sample, "priority", None)),
            "completed_stage_indices": sorted(getattr(s, "stage_index", 0) for s in visible if s.status == "completed"),
            "public_remaining_route": list(route[stage+1:]) if route is not None else [],
            "public_stages_total": _value(len(route) if route is not None else None),
        })
    action_features = []
    for item in assignments:
        sample = samples[item.sample_id];robot = robots[item.robot_id]
        first, last = stations[sample.pickup], stations[sample.dropoff]
        resource_first = rules.station_resources[sample.pickup]
        resource_last = rules.station_resources[sample.dropoff]
        action_features.append({
            "robot_id": item.robot_id, "sample_id": item.sample_id,
            "previous_task": _value(robot.task_id),
            "replaces_running_task": robot.task_id is not None and robot.task_id != sample.id,
            "pickup_resource": resource_first.id, "dropoff_resource": resource_last.id,
            "pickup_station": first.id, "dropoff_station": last.id,
            "robot_to_pickup_distance": geometry.distance(robot.position, first.position),
            "pickup_to_dropoff_distance": geometry.distance(first.position, last.position),
            "competes_with_running_service": any(
                other.id != robot.id and other.service_elapsed is not None and
                other.position in (first.position, last.position) for other in observation.robots),
        })
    return {
        "schema_version": SCHEMA_VERSION, "label_version": LABEL_VERSION,
        "tick": observation.tick,
        "observed_facts": {"robots": robot_features, "samples": sample_features,
                           "stations": [_station(s) for s in sorted(observation.stations, key=lambda x: x.id)]},
        "public_rules": {"workflow_routes_for_visible_jobs": {
            key: list(rules.workflows[key]) for key in sorted({
                getattr(s, "workflow", "") for s in observation.samples
            } - {""}) if key in rules.workflows},
            "resources": [{"id": r.id, "capacity": r.capacity, "stations": list(r.stations)}
                          for r in sorted(rules.resources, key=lambda x: x.id)]},
        "candidate_action": action_features,
        "unknown": {"future_fault_schedule": _value(None),
                    "true_remaining_service_countdown": _value(None),
                    "unreleased_external_jobs": _value(None),
                    "counterfactual_outcome": _value(None)},
        "learnable_next_events": ["service_completion_next_step"],
        "deterministic_transition_owner": "business_transport.workflow_transition",
        "feature_sources": FIELD_SOURCES,
    }
