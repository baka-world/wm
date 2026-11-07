"""Build a nominal, source-based travel-time approximation for an RMF graph.

The source's single scale measurement is in metres and its waypoint
coordinates are image pixels. This model omits acceleration, doors and people.
"""
from __future__ import annotations

import hashlib
import math
import re
import xml.etree.ElementTree as ET

from .domain import DirectedGraph, Position

MAP_SHA256 = "8c2ebd01b49818f1d40f32e849b2c41ac800861f0b4ce82498d4298c6f2ad058"
TIAGO_CONFIG_SHA256 = "5f2a4d5f7d75e2b5865d2bdc98842c37b2ccb97c822cb00189da75f922569530"


def source_parameters(map_data: bytes, config_data: bytes) -> dict:
    if hashlib.sha256(map_data).hexdigest() != MAP_SHA256:
        raise ValueError("hospital map SHA-256 differs from the pinned source")
    if hashlib.sha256(config_data).hexdigest() != TIAGO_CONFIG_SHA256:
        raise ValueError("TIAGo adapter SHA-256 differs from the pinned source")
    lines = map_data.decode("utf-8").splitlines()
    if "coordinate_system: reference_image" not in lines:
        raise ValueError("unsupported map coordinate system")
    vertices_lines = lines[lines.index("    vertices:") + 1:lines.index("    walls:")]
    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
    pattern = re.compile(r"^\s+- \[(" + number + r"),\s*(" + number + r"),")
    vertices = []
    for line in vertices_lines:
        match = pattern.match(line)
        if not match:
            raise ValueError("unexpected map vertex")
        vertices.append((float(match.group(1)), float(match.group(2))))
    measurements = lines[lines.index("    measurements:") + 1:lines.index("    models:")]
    if len(measurements) != 1:
        raise ValueError("expected exactly one reference measurement")
    match = re.fullmatch(r"\s+- \[(\d+),\s*(\d+),\s*\{distance: \[3,\s*(" + number + r")\]\}\]", measurements[0])
    if not match:
        raise ValueError("unexpected map scale measurement")
    first, second, metres = int(match.group(1)), int(match.group(2)), float(match.group(3))
    if first >= len(vertices) or second >= len(vertices):
        raise ValueError("invalid reference measurement endpoints")
    pixels = math.dist(vertices[first], vertices[second])
    if pixels <= 0 or not math.isfinite(metres) or metres <= 0:
        raise ValueError("invalid reference measurement")
    root = ET.fromstring(config_data)
    velocities = [arg.get("value") for arg in root.iter("arg") if arg.get("name") == "linear_velocity" and arg.get("value")]
    if len(velocities) != 1:
        raise ValueError("expected one nominal linear velocity")
    speed = float(velocities[0])
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError("invalid nominal linear velocity")
    return {"reference_metres": metres, "reference_pixels": pixels, "metres_per_pixel": metres / pixels,
            "nominal_speed_m_per_s": speed, "map_sha256": MAP_SHA256,
            "tiago_adapter_sha256": TIAGO_CONFIG_SHA256}


def subdivide_graph(graph: DirectedGraph, metres_per_pixel: float, speed_m_per_s: float,
                    seconds_per_tick: int) -> DirectedGraph:
    """Split original lanes into shared directed chains of equal-duration steps.

    Every original lane takes ceil(distance / speed / tick) steps. Intermediate
    vertices are shared by both directions of a bidirectional lane so that
    existing node/edge-swap checks remain meaningful at the graph resolution.
    """
    if (not isinstance(seconds_per_tick, int) or isinstance(seconds_per_tick, bool)
            or not 1 <= seconds_per_tick <= 300):
        raise ValueError("seconds_per_tick must be an integer from 1 to 300")
    if not math.isfinite(metres_per_pixel) or not math.isfinite(speed_m_per_s) or metres_per_pixel <= 0 or speed_m_per_s <= 0:
        raise ValueError("scale and speed must be finite positive numbers")
    coords = dict(graph.coordinates)
    existing = set(graph.edges)
    pairs = sorted({tuple(sorted((a, b))) for a, b in graph.edges})
    next_index = max(p[0] for p in coords) + 1
    edges: list[tuple[Position, Position]] = []
    for a, b in pairs:
        x1, y1 = coords[a]
        x2, y2 = coords[b]
        distance = math.hypot(x2 - x1, y2 - y1) * metres_per_pixel
        parts = max(1, math.ceil(distance / speed_m_per_s / seconds_per_tick))
        if len(coords) + parts - 1 > 500:
            raise ValueError("time step creates more than 500 graph nodes; choose a coarser step")
        chain = [a]
        for i in range(1, parts):
            position = (next_index, 0)
            next_index += 1
            coords[position] = (x1 + (x2 - x1) * i / parts, y1 + (y2 - y1) * i / parts)
            chain.append(position)
        chain.append(b)
        if (a, b) in existing:
            edges.extend(zip(chain, chain[1:]))
        if (b, a) in existing:
            edges.extend(zip(reversed(chain), list(reversed(chain))[1:]))
    return DirectedGraph(coords, tuple(edges))
