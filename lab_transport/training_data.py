"""Export and validate model-independent synthetic training trajectories.

This module uses CPU simulation only. It does not train a model or select a
GPU. Exported map derivatives and task records stay in the requested local
directory; users must check source licences before sharing them.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any
from urllib.request import Request, urlopen

from .clinic_map import COMMIT as CLINIC_COMMIT, CONFIG_SHA256 as CLINIC_CONFIG_SHA256, MAP_SHA256 as CLINIC_MAP_SHA256, make_clinic_scenario
from .domain import DirectedGraph, Grid
from .dynamics_experiment import audit_task_schedules, collect_episodes
from .dynamics_revision import SPLITS_V2
from .hospital_case import SOURCE_COMMIT, SOURCE_SHA256, make_scenario
from .multisite_experiment import make_case, prepare_tartu
from .time_model import TIAGO_CONFIG_SHA256
from .transition_schema import SCHEMA_VERSION

DATASET_VERSION = 1
OBSERVATION_KEYS = {"schema_version", "tick", "robots", "samples", "stations"}
ROBOT_KEYS = {"id", "position", "home", "task_id", "phase", "service_elapsed"}
SAMPLE_KEYS = {"id", "pickup", "dropoff", "status", "robot_id", "release_tick", "deadline_tick",
               "specimen_type", "packaging", "temperature_condition", "rule_profile"}
STATION_KEYS = {"id", "position", "available", "expected_service_ticks"}
STATIC_METADATA_KEYS = {"time_model", "seconds_per_tick", "metres_per_pixel", "nominal_speed_m_per_s",
                        "reference_metres", "source_commit", "source_map_sha256", "source_adapter_sha256",
                        "timing_assumption"}
SOURCES = {
    "hospital_map": (f"https://raw.githubusercontent.com/scafld/covsg24_fleet_backend/{SOURCE_COMMIT}/covsg24_environments/environments/hospital_2nd_floor/hospital_2nd_floor.building.yaml", SOURCE_SHA256),
    "tiago_config": (f"https://raw.githubusercontent.com/scafld/covsg24_fleet_backend/{SOURCE_COMMIT}/covsg24_fleet_server/launch/fleet_adapters/tiago_fleet_adapter.launch.xml", TIAGO_CONFIG_SHA256),
    "clinic_map": (f"https://raw.githubusercontent.com/open-rmf/rmf_demos/{CLINIC_COMMIT}/rmf_demos_maps/maps/clinic/clinic.building.yaml", CLINIC_MAP_SHA256),
    "clinic_config": (f"https://raw.githubusercontent.com/open-rmf/rmf_demos/{CLINIC_COMMIT}/rmf_demos/config/clinic/deliveryRobot_config.yaml", CLINIC_CONFIG_SHA256),
}
MAX_SOURCE_BYTES = 2 * 1024 * 1024


def _bytes(data: Any) -> bytes:
    return (json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download_sources() -> dict[str, bytes]:
    """Fetch only fixed public sources, verify size and hash, keep in memory."""
    result = {}
    for name, (url, expected) in SOURCES.items():
        request = Request(url, headers={"User-Agent": "sample-transport-data-preparation"})
        with urlopen(request, timeout=30) as response:
            data = response.read(MAX_SOURCE_BYTES + 1)
        if len(data) > MAX_SOURCE_BYTES:
            raise ValueError(f"source too large: {name}")
        if _digest(data) != expected:
            raise ValueError(f"source SHA-256 mismatch: {name}")
        result[name] = data
    return result


def _topology(scenario) -> dict[str, Any]:
    grid = scenario.grid
    stations = [{"id": sid, "position": list(s.position)} for sid, s in sorted(scenario.stations.items())]
    metadata = {key: value for key, value in (scenario.metadata or {}).items() if key in STATIC_METADATA_KEYS}
    if isinstance(grid, DirectedGraph):
        return {"kind": "directed_graph", "nodes": [{"position": list(p), "coordinates": list(xy)}
                for p, xy in sorted(grid.coordinates.items())],
                "edges": [[list(a), list(b)] for a, b in grid.edges], "stations": stations,
                "metadata": metadata}
    if isinstance(grid, Grid):
        return {"kind": "grid", "width": grid.width, "height": grid.height,
                "blocked": [list(p) for p in sorted(grid.blocked)], "stations": stations,
                "metadata": metadata}
    raise ValueError("unsupported topology")


def _task_hash(scenario, layout: str) -> str:
    # This hash is for split auditing only: it summarizes future tasks and
    # must never be treated as a model input.
    tasks = [[s.pickup, s.dropoff, s.release_tick] for s in scenario.samples]
    return _digest(_bytes({"layout": layout, "tasks": tasks}))


def export_dataset(bases: dict[str, Any], source_hashes: dict[str, str], output: Path,
                   splits=SPLITS_V2) -> dict[str, Any]:
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    audit = audit_task_schedules(bases, splits)
    if audit["cross_split_duplicate_groups"] or audit["cross_split_task_duplicate_groups"]:
        raise ValueError("task schedules overlap across dataset splits")
    episodes = collect_episodes(bases, splits)
    output.mkdir(parents=True, exist_ok=True)
    maps = {}
    for layout in sorted({layout for layout, _ in splits.values()}):
        relative = f"maps/{layout}.json"
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        data = _bytes(_topology(bases[layout]))
        path.write_bytes(data)
        maps[layout] = {"file": relative, "sha256": _digest(data)}
    entries = []
    for episode in episodes:
        relative = f"episodes/{episode.split}/{episode.layout}-{episode.seed}-{episode.regime}-{episode.policy}.jsonl"
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        with path.open("xb") as handle:
            for row in episode.rows:
                line = _bytes(asdict(row))
                digest.update(line)
                handle.write(line)
        scenario = make_case(bases[episode.layout], episode.layout, episode.seed, episode.regime)
        entries.append({"split": episode.split, "layout": episode.layout, "seed": episode.seed,
                        "regime": episode.regime, "behavior_policy": episode.policy, "file": relative,
                        "sha256": digest.hexdigest(), "task_schedule_sha256": _task_hash(scenario, episode.layout),
                        "transitions": len(episode.rows), "episode_status": episode.status,
                        "completed_tasks": episode.completed, "scenario_sample_count": len(scenario.samples)})
    manifest = {"dataset_version": DATASET_VERSION, "observation_schema_version": SCHEMA_VERSION,
        "scope": "Synthetic simulation on cited public maps; not real clinical transport records.",
        "maps": maps, "source_sha256": source_hashes, "schedule_audit": audit,
        "input_boundary": "Only observation, assignments and the known static map are model inputs; metadata, seed and next_observation are not.",
        "limitations": ["Only factual executed assignments are labeled; no counterfactual ground truth.",
                        "New arrivals and station changes in next_observation are outcomes, not input features.",
                        "Final end_of_run transition is excluded, including for time-limited episodes.",
                        "Training batches, model architecture and GPU trainer are not provided by this exporter.",
                        "Generated map derivatives and task identifiers may require licence and privacy review before sharing."],
        "episodes": entries}
    (output / "manifest.json").write_bytes(_bytes(manifest))
    return manifest


def _path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("dataset file path must be relative")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("dataset file path escapes the dataset directory")
    return resolved


def _position(value: Any) -> tuple[int, int]:
    if (not isinstance(value, list) or len(value) != 2 or any(
            not isinstance(x, int) or isinstance(x, bool) for x in value)):
        raise ValueError("invalid discrete position")
    return value[0], value[1]


def _check_map(topology: Any) -> tuple[set[tuple[int, int]], set[tuple[tuple[int, int], tuple[int, int]]], dict[str, tuple[int, int]]]:
    if not isinstance(topology, dict) or not isinstance(topology.get("stations"), list):
        raise ValueError("invalid static map")
    if set(topology.get("metadata", {})) - STATIC_METADATA_KEYS:
        raise ValueError("unexpected static map metadata")
    kind = topology.get("kind")
    if kind == "directed_graph":
        nodes = topology.get("nodes")
        raw_edges = topology.get("edges")
        if not isinstance(nodes, list) or not isinstance(raw_edges, list) or not nodes:
            raise ValueError("invalid graph topology")
        positions = {_position(node["position"]) for node in nodes}
        if len(positions) != len(nodes):
            raise ValueError("duplicate graph nodes")
        for node in nodes:
            xy = node.get("coordinates")
            if (not isinstance(xy, list) or len(xy) != 2 or any(
                    not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x) for x in xy)):
                raise ValueError("invalid graph coordinates")
        edges = set()
        for edge in raw_edges:
            if not isinstance(edge, list) or len(edge) != 2:
                raise ValueError("invalid graph edge")
            a, b = _position(edge[0]), _position(edge[1])
            if a not in positions or b not in positions or a == b:
                raise ValueError("invalid graph edge endpoints")
            edges.add((a, b))
    elif kind == "grid":
        width, height, raw_blocked = topology.get("width"), topology.get("height"), topology.get("blocked")
        if (not isinstance(width, int) or isinstance(width, bool) or width < 1 or
                not isinstance(height, int) or isinstance(height, bool) or height < 1 or
                not isinstance(raw_blocked, list)):
            raise ValueError("invalid grid topology")
        blocked = {_position(p) for p in raw_blocked}
        if any(not (0 <= x < width and 0 <= y < height) for x, y in blocked):
            raise ValueError("blocked cell is outside grid")
        positions = {(x, y) for x in range(width) for y in range(height)} - blocked
        edges = {(p, q) for p in positions for q in ((p[0] + 1, p[1]), (p[0], p[1] + 1),
                                                     (p[0] - 1, p[1]), (p[0], p[1] - 1)) if q in positions}
    else:
        raise ValueError("unsupported map topology")
    station_map = {}
    for station in topology["stations"]:
        sid, position = station["id"], _position(station["position"])
        if not isinstance(sid, str) or not sid or sid in station_map or position not in positions:
            raise ValueError("invalid static station")
        station_map[sid] = position
    return positions, edges, station_map


def _check_observation(obs: Any) -> None:
    if not isinstance(obs, dict) or set(obs) != OBSERVATION_KEYS or obs["schema_version"] != SCHEMA_VERSION:
        raise ValueError("invalid observation schema")
    tick = obs["tick"]
    if not isinstance(tick, int) or isinstance(tick, bool) or tick < 0:
        raise ValueError("invalid observation tick")
    for field, keys in (("robots", ROBOT_KEYS), ("samples", SAMPLE_KEYS), ("stations", STATION_KEYS)):
        values = obs[field]
        if not isinstance(values, list) or any(not isinstance(v, dict) or set(v) != keys for v in values):
            raise ValueError(f"invalid observation {field}")
        ids = [v["id"] for v in values]
        if any(not isinstance(x, str) or not x for x in ids) or len(ids) != len(set(ids)):
            raise ValueError(f"invalid or duplicate {field} identifiers")
    for sample in obs["samples"]:
        if (sample["status"] not in {"waiting", "assigned", "in_transit", "completed"}
                or not isinstance(sample["release_tick"], int) or isinstance(sample["release_tick"], bool)
                or not 0 <= sample["release_tick"] <= tick):
            raise ValueError("future task leaked into current observation")
    for robot in obs["robots"]:
        elapsed = robot["service_elapsed"]
        if (robot["phase"] not in {"idle", "to_pickup", "to_dropoff"}
                or (elapsed is not None and (not isinstance(elapsed, int) or isinstance(elapsed, bool) or elapsed < 0))):
            raise ValueError("invalid observable robot state")
    for station in obs["stations"]:
        expected = station["expected_service_ticks"]
        if (not isinstance(station["available"], bool) or not isinstance(expected, (int, float))
                or isinstance(expected, bool) or not math.isfinite(expected) or expected <= 0):
            raise ValueError("invalid observable station state")


def validate_dataset(root: Path) -> dict[str, int]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("dataset_version") != DATASET_VERSION or manifest.get("observation_schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported dataset schema")
    audit = manifest.get("schedule_audit", {})
    if audit.get("cross_split_duplicate_groups") != 0 or audit.get("cross_split_task_duplicate_groups") != 0:
        raise ValueError("dataset has cross-split schedule overlap")
    maps, paths = manifest.get("maps"), set()
    if not isinstance(maps, dict) or not maps:
        raise ValueError("dataset has no static maps")
    topologies = {}
    for layout, entry in maps.items():
        path = _path(root, entry["file"])
        raw = path.read_bytes()
        if _digest(raw) != entry["sha256"]:
            raise ValueError(f"map checksum mismatch: {layout}")
        topologies[layout] = _check_map(json.loads(raw))
    seen_seed_groups: dict[tuple[str, int], str] = {}
    seen_task_groups: dict[str, str] = {}
    count, unverified_schedules = 0, 0
    for episode in manifest.get("episodes", []):
        split, layout, seed = episode["split"], episode["layout"], episode["seed"]
        if layout not in maps or not isinstance(split, str) or not isinstance(seed, int):
            raise ValueError("invalid episode grouping")
        group, task_hash = (layout, seed), episode["task_schedule_sha256"]
        if group in seen_seed_groups and seen_seed_groups[group] != split:
            raise ValueError("seed group appears across splits")
        if task_hash in seen_task_groups and seen_task_groups[task_hash] != split:
            raise ValueError("task schedule appears across splits")
        seen_seed_groups[group], seen_task_groups[task_hash] = split, split
        path = _path(root, episode["file"])
        if path in paths:
            raise ValueError("episode file is listed more than once")
        paths.add(path)
        raw = path.read_bytes()
        if _digest(raw) != episode["sha256"]:
            raise ValueError("episode checksum mismatch")
        previous, rows = None, 0
        for line in raw.splitlines():
            row = json.loads(line)
            if set(row) != {"observation", "assignments", "next_observation", "next_boundary", "termination"}:
                raise ValueError("invalid transition schema")
            current, nxt = row["observation"], row["next_observation"]
            _check_observation(current)
            _check_observation(nxt)
            if row["next_boundary"] != "before_dispatch" or row["termination"] is not None:
                raise ValueError("dataset contains an unsupported final boundary")
            if nxt["tick"] != current["tick"] + 1 or (previous is not None and previous != current):
                raise ValueError("episode transition boundaries are not contiguous")
            if not {x["id"] for x in current["samples"]}.issubset({x["id"] for x in nxt["samples"]}):
                raise ValueError("known task disappeared from the next observation")
            positions, edges, static_stations = topologies[layout]
            for observation in (current, nxt):
                seen_positions = [_position(robot["position"]) for robot in observation["robots"]]
                if len(set(seen_positions)) != len(seen_positions) or any(p not in positions for p in seen_positions):
                    raise ValueError("invalid discrete robot occupancy")
                if any(_position(robot["home"]) not in positions for robot in observation["robots"]):
                    raise ValueError("robot home is not on the map")
                stations = {s["id"]: _position(s["position"]) for s in observation["stations"]}
                if stations != static_stations:
                    raise ValueError("observable station map changed")
                if any(s["pickup"] not in stations or s["dropoff"] not in stations or s["pickup"] == s["dropoff"]
                       for s in observation["samples"]):
                    raise ValueError("sample references invalid stations")
            before = {r["id"]: _position(r["position"]) for r in current["robots"]}
            after = {r["id"]: _position(r["position"]) for r in nxt["robots"]}
            if before.keys() != after.keys():
                raise ValueError("robot set changed during a transition")
            if any(before[rid] != after[rid] and (before[rid], after[rid]) not in edges for rid in before):
                raise ValueError("robot moved outside the discrete map edges")
            robot_order = sorted(before)
            if any(before[a] == after[b] and before[b] == after[a]
                   for index, a in enumerate(robot_order) for b in robot_order[index + 1:]):
                raise ValueError("robots swapped an edge")
            robots = {x["id"]: x for x in current["robots"]}
            samples = {x["id"]: x for x in current["samples"]}
            robot_ids, sample_ids = set(), set()
            for action in row["assignments"]:
                if not isinstance(action, dict) or set(action) != {"robot_id", "sample_id"}:
                    raise ValueError("invalid assignment schema")
                rid, sid = action["robot_id"], action["sample_id"]
                if (rid not in robots or sid not in samples or robots[rid]["task_id"] is not None
                        or samples[sid]["status"] != "waiting" or rid in robot_ids or sid in sample_ids):
                    raise ValueError("assignment does not match the current observation")
                robot_ids.add(rid)
                sample_ids.add(sid)
            previous, rows = nxt, rows + 1
        if rows != episode["transitions"] or rows == 0:
            raise ValueError("episode transition count mismatch")
        expected_samples = episode["scenario_sample_count"]
        if not isinstance(expected_samples, int) or isinstance(expected_samples, bool) or expected_samples < 1:
            raise ValueError("invalid scenario sample count")
        known = sorted(previous["samples"], key=lambda sample: sample["id"])
        if len(known) == expected_samples:
            observed_tasks = [[s["pickup"], s["dropoff"], s["release_tick"]] for s in known]
            if _digest(_bytes({"layout": layout, "tasks": observed_tasks})) != task_hash:
                raise ValueError("task schedule checksum does not match observed tasks")
        elif len(known) < expected_samples and episode["episode_status"] == "timed_out":
            # Future tasks were legitimately never released; the source-side
            # audit still applies, but the trajectory alone cannot verify it.
            unverified_schedules += 1
        else:
            raise ValueError("episode sample count is inconsistent with its outcome")
        count += rows
    if not paths:
        raise ValueError("dataset has no episodes")
    return {"episodes": len(paths), "transitions": count, "maps": len(maps),
            "schedules_not_verifiable_from_trajectory": unverified_schedules}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    for argument in ("hospital-map", "tiago-config", "clinic-map", "clinic-config"):
        export.add_argument("--" + argument)
    export.add_argument("--output", required=True)
    export.add_argument("--download-sources", action="store_true", help="download and verify the four pinned public source files")
    export.add_argument("--profile", choices=("smoke", "full"), default="full")
    check = commands.add_parser("validate")
    check.add_argument("--dataset", required=True)
    args = parser.parse_args()
    try:
        if args.command == "validate":
            print(json.dumps(validate_dataset(Path(args.dataset))))
            return 0
        paths = {"hospital_map": args.hospital_map, "tiago_config": args.tiago_config,
                 "clinic_map": args.clinic_map, "clinic_config": args.clinic_config}
        if args.download_sources:
            if any(paths.values()):
                raise ValueError("use --download-sources or local source paths, not both")
            source = download_sources()
        else:
            if not all(paths.values()):
                raise ValueError("provide all four source paths or use --download-sources")
            source = {key: Path(path).read_bytes() for key, path in paths.items()}
        bases = {"tartu": prepare_tartu(make_scenario(source["hospital_map"], seconds_per_tick=10,
                                                       tiago_config=source["tiago_config"])),
                 "clinic": make_clinic_scenario(source["clinic_map"], source["clinic_config"])}
        splits = ({name: (layout, seeds[:1]) for name, (layout, seeds) in SPLITS_V2.items()}
                  if args.profile == "smoke" else SPLITS_V2)
        output = Path(args.output)
        manifest = export_dataset(bases, {name: _digest(data) for name, data in source.items()}, output, splits)
        checked = validate_dataset(output)
        print(json.dumps({"profile": args.profile, **checked, "source_files": len(manifest["source_sha256"])}))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"训练数据准备失败：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
