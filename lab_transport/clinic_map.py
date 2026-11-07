"""Pinned Open-RMF clinic demo L1/graph-0 adaptation for simulation only."""
from __future__ import annotations

import hashlib
import math
import re

from .domain import DirectedGraph, RobotSpec, Sample, Scenario, Station
from .time_model import subdivide_graph

COMMIT = "672a67a1643ac12d89fc21565fa92f33f5c2e0fc"
MAP_SHA256 = "780bfabff950c4fb3132dd1933994255afe3b93cbc25a8265e0214adb62ff7f0"
CONFIG_SHA256 = "e63fd0f2e74eac14c4651ad6ee04d20d9e97b8a0a13cc415791ce013b6d9c316"


def make_clinic_scenario(map_data: bytes, config_data: bytes, seconds_per_tick: int = 10) -> Scenario:
    """Read the fixed source, preserving graph-0 directions and labelled nodes.

    Treatment/nurse nodes are *hypothetical* sample pickup/handover points,
    not a claim about the open-source demo's actual intended workflow.
    """
    if hashlib.sha256(map_data).hexdigest() != MAP_SHA256:
        raise ValueError("clinic map SHA-256 differs from the pinned source")
    if hashlib.sha256(config_data).hexdigest() != CONFIG_SHA256:
        raise ValueError("clinic fleet config SHA-256 differs from the pinned source")
    lines = map_data.decode("utf-8").splitlines()
    if "coordinate_system: reference_image" not in lines:
        raise ValueError("unexpected clinic coordinate system")
    level = lines[lines.index("  L1:") + 1:lines.index("  L2:")]

    def block(start: str, end: str) -> list[str]:
        a, b = level.index(start), level.index(end)
        if a >= b:
            raise ValueError("unexpected clinic map section order")
        return level[a + 1:b]

    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
    vertex_pattern = re.compile(r"^\s+- \[(" + number + r"),\s*(" + number + r"),")
    vertices = []
    for line in block("    vertices:", "    walls:"):
        match = vertex_pattern.match(line)
        if not match:
            raise ValueError("unexpected clinic vertex")
        vertices.append((float(match.group(1)), float(match.group(2))))
    if len(vertices) != 449:
        raise ValueError("unexpected clinic vertex count")
    edges = []
    for line in block("    lanes:", "    layers:"):
        endpoint = re.match(r"^\s+- \[(\d+),\s*(\d+),\s*\{", line)
        direction = re.search(r"bidirectional: \[4, (true|false)\]", line)
        index = re.search(r"graph_idx: \[2, (\d+)\]", line)
        if not endpoint or not direction or not index:
            raise ValueError("unexpected clinic lane")
        if index.group(1) != "0":
            continue
        a, b = int(endpoint.group(1)), int(endpoint.group(2))
        if a == b or a >= len(vertices) or b >= len(vertices):
            raise ValueError("invalid clinic lane endpoint")
        edges.append(((a, 0), (b, 0)))
        if direction.group(1) == "true":
            edges.append(((b, 0), (a, 0)))
    ids = {p for edge in edges for p in edge}
    if len(ids) != 54 or len(edges) != 110:
        raise ValueError("unexpected clinic graph 0 size")
    graph = DirectedGraph({p: vertices[p[0]] for p in sorted(ids)}, tuple(edges))
    measurements = block("    measurements:", "    models:")
    if len(measurements) != 1:
        raise ValueError("unexpected clinic reference scale count")
    measurement = re.fullmatch(r"\s+- \[(\d+),\s*(\d+),\s*\{distance: \[3,\s*(" + number + r")\]\}\]", measurements[0])
    if not measurement:
        raise ValueError("unexpected clinic reference scale")
    a, b, metres = int(measurement.group(1)), int(measurement.group(2)), float(measurement.group(3))
    pixels = math.dist(vertices[a], vertices[b])
    if pixels <= 0 or metres <= 0:
        raise ValueError("invalid clinic scale")
    linear = re.findall(r"^\s*linear:\s*\[(" + number + r"),\s*" + number + r"\]", config_data.decode("utf-8"), re.MULTILINE)
    if len(linear) != 1:
        raise ValueError("unexpected clinic nominal speed")
    speed = float(linear[0])
    graph = subdivide_graph(graph, metres / pixels, speed, seconds_per_tick)
    # The reference paper's 30–60s loading distribution is borrowed from a
    # different hospital. Delivery 10s remains an explicit test assumption.
    low = max(1, (30 + seconds_per_tick - 1) // seconds_per_tick)
    high = max(low, (60 + seconds_per_tick - 1) // seconds_per_tick)
    stations = {
        "诊疗左_假设取样": Station("诊疗左_假设取样", (384, 0), low, (low + high) / 2, (), high - low),
        "诊疗右_假设取样": Station("诊疗右_假设取样", (409, 0), low, (low + high) / 2, (), high - low),
        "护理左_假设交接": Station("护理左_假设交接", (398, 0), 1, 1.0, (), 0),
        "护理右_假设交接": Station("护理右_假设交接", (403, 0), 1, 1.0, (), 0),
    }
    robots = (RobotSpec("R1", (390, 0), (390, 0)), RobotSpec("R2", (431, 0), (431, 0)))
    important = [r.position for r in robots] + [s.position for s in stations.values()]
    if any(graph.shortest_path(a, b) is None for a in important for b in important):
        raise ValueError("clinic scenario waypoints are not mutually reachable")
    return Scenario("Open-RMF 诊所演示图（任务站点用途为仿真假设）", graph, stations, robots,
                    (Sample("placeholder", "诊疗左_假设取样", "护理左_假设交接", 0, None),), 0, 350,
                    {"time_model": "nominal_source_based", "seconds_per_tick": seconds_per_tick,
                     "source_commit": COMMIT, "source_map_sha256": MAP_SHA256,
                     "metres_per_pixel": round(metres / pixels, 9), "nominal_speed_m_per_s": speed,
                     "timing_assumption": "公开诊所演示地图和标称速度；样本取送点是仿真假设，不是实测运输。"})
