"""Adapt a pinned published hospital navigation graph for a CPU-only smoke run.

The original file is not redistributed here. Its lane direction and waypoint
coordinates are read at run time; synthetic arrivals are not hospital records.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

from .domain import DirectedGraph, RobotSpec, Sample, Scenario, Station
from .report import check_output_directory, write_run
from .simulation import Simulator

SOURCE_SHA256 = "8c2ebd01b49818f1d40f32e849b2c41ac800861f0b4ce82498d4298c6f2ad058"
SOURCE_COMMIT = "b9e32eeebb3a568af0e41c6e34500076cd2f6cad"


def parse_pinned_building(data: bytes) -> DirectedGraph:
    """Extract the pinned RMF file's L2 graph; fail closed on changed source.

    This is deliberately a narrow parser for the pinned file, not a general
    YAML parser. It checks every lane and waypoint line in the relevant blocks.
    """
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError("hospital map SHA-256 differs from the pinned upstream file")
    lines = data.decode("utf-8").splitlines()

    def block(start: str, end: str) -> list[str]:
        first, last = lines.index(start), lines.index(end)
        if first >= last:
            raise ValueError("unexpected hospital map section order")
        return lines[first + 1:last]

    waypoint_lines = block("    vertices:", "    walls:")
    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
    vertex_pattern = re.compile(r"^\s+- \[(" + number + r"),\s*(" + number + r"),\s*" + number + r",\s*(?:\"[^\"]*\"|[^,\]\s]+)")
    vertices = []
    for line in waypoint_lines:
        match = vertex_pattern.match(line)
        if not match:
            raise ValueError("unexpected waypoint line in pinned hospital map")
        vertices.append((float(match.group(1)), float(match.group(2))))

    edges = []
    for line in block("    lanes:", "    layers:"):
        endpoints = re.match(r"^\s+- \[(\d+),\s*(\d+),\s*\{", line)
        direction = re.search(r"bidirectional: \[4, (true|false)\]", line)
        graph_index = re.search(r"graph_idx: \[2, (\d+)\]", line)
        if not endpoints or not direction or not graph_index or graph_index.group(1) != "0":
            raise ValueError("unexpected lane line in pinned hospital map")
        a, b = int(endpoints.group(1)), int(endpoints.group(2))
        if a >= len(vertices) or b >= len(vertices) or a == b:
            raise ValueError("invalid lane endpoint")
        edges.append(((a, 0), (b, 0)))
        if direction.group(1) == "true":
            edges.append(((b, 0), (a, 0)))
    ids = {p for edge in edges for p in edge}
    if len(vertices) != 209 or len(edges) != 63 or len(ids) != 32:
        raise ValueError("unexpected pinned hospital graph size")
    return DirectedGraph({p: vertices[p[0]] for p in sorted(ids)}, tuple(edges))


def make_scenario(data: bytes) -> Scenario:
    graph = parse_pinned_building(data)
    # Source indices 208 and 204 are named icu1_pickup_location and
    # lab_dropoff_location respectively. Durations are abstract test steps.
    stations = {
        "ICU取样点": Station("ICU取样点", (208, 0), 1, 1.0, ()),
        "实验室交付点": Station("实验室交付点", (204, 0), 1, 1.0, ()),
    }
    # Two robots, their start points, and this deterministic demand sequence
    # are our explicit functional-test settings, not measured hospital data.
    robots = (RobotSpec("R1", (190, 0), (190, 0)), RobotSpec("R2", (193, 0), (193, 0)))
    samples = tuple(Sample(f"S{i + 1:02}", "ICU取样点", "实验室交付点", release, None)
                    for i, release in enumerate((0, 12, 24, 36)))
    important = [r.position for r in robots] + [s.position for s in stations.values()]
    if any(graph.shortest_path(a, b) is None for a in important for b in important):
        raise ValueError("required hospital waypoints are not mutually reachable")
    return Scenario("Tartu 医院公开路网适配（功能测试任务）", graph, stations, robots,
                    samples, duration_jitter=0, max_ticks=150)


def main() -> int:
    parser = argparse.ArgumentParser(description="公开医院路网适配；任务序列为功能测试设置")
    parser.add_argument("--map-file", required=True, help="固定版本的上游 hospital_2nd_floor.building.yaml")
    parser.add_argument("--output", required=True, help="输出目录；不覆盖已有结果")
    args = parser.parse_args()
    try:
        scenario = make_scenario(Path(args.map_file).read_bytes())
        output = Path(args.output)
        if (output / "comparison.json").exists():
            raise FileExistsError(f"refusing to overwrite existing output: {output / 'comparison.json'}")
        for policy in ("nearest", "world"):
            check_output_directory(output / policy)
        results = {policy: Simulator(scenario, policy, 0).run() for policy in ("nearest", "world")}
        for policy, result in results.items():
            write_run(result, output / policy)
        comparison = {
            "scenario": scenario.name,
            "source_commit": SOURCE_COMMIT,
            "source_sha256": SOURCE_SHA256,
            "note": "公开路网与人工设置的功能测试任务；图边为抽象步，不是实际时间或临床证据。",
            "policies": {policy: result["summary"] for policy, result in results.items()},
        }
        (output / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(comparison, ensure_ascii=False))
        return 0 if all(r["summary"]["status"] == "completed" for r in results.values()) else 2
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"运行失败：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
