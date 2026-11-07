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
from .mimic_workload import CHECKSUMS as MIMIC_CHECKSUMS, DATASET_DOI
from .time_model import source_parameters, subdivide_graph

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


def make_scenario(data: bytes, workload: dict | None = None, seconds_per_tick: int | None = None,
                  tiago_config: bytes | None = None) -> Scenario:
    graph = parse_pinned_building(data)
    timing = None
    if tiago_config is not None:
        if seconds_per_tick is None:
            raise ValueError("TIAGo time model requires explicit --seconds-per-tick")
        timing = source_parameters(data, tiago_config)
        graph = subdivide_graph(graph, timing["metres_per_pixel"], timing["nominal_speed_m_per_s"], seconds_per_tick)
    # Source indices 208 and 204 are named icu1_pickup_location and
    # lab_dropoff_location respectively. Durations are abstract test steps.
    stations = {
        "ICU取样点": Station("ICU取样点", (208, 0), 1, 1.0, ()),
        "实验室交付点": Station("实验室交付点", (204, 0), 1, 1.0, ()),
    }
    if timing is not None:
        assert seconds_per_tick is not None
        # Rossetti et al. model loading at 30–60 s based on observations in a
        # different hospital. Rounding to ticks is an explicit adaptation.
        pickup_low = max(1, (30 + seconds_per_tick - 1) // seconds_per_tick)
        pickup_high = max(pickup_low, (60 + seconds_per_tick - 1) // seconds_per_tick)
        stations["ICU取样点"] = Station("ICU取样点", (208, 0), pickup_low,
                                      (pickup_low + pickup_high) / 2, (), pickup_high - pickup_low)
        # No Tartu handover-time measurement was found; one tick is a test assumption.
        stations["实验室交付点"] = Station("实验室交付点", (204, 0), 1, 1.0, (), 0)
    # Two robots, their start points, and this deterministic demand sequence
    # are our explicit functional-test settings, not measured hospital data.
    robots = (RobotSpec("R1", (190, 0), (190, 0)), RobotSpec("R2", (193, 0), (193, 0)))
    metadata = None
    if workload is None:
        if seconds_per_tick is not None and timing is None:
            raise ValueError("seconds_per_tick requires a MIMIC workload or TIAGo time model")
        samples = tuple(Sample(f"S{i + 1:02}", "ICU取样点", "实验室交付点", release, None)
                        for i, release in enumerate((0, 12, 24, 36)))
        name, max_ticks = "Tartu 医院公开路网适配（功能测试任务）", 1000 if timing else 150
    else:
        if not isinstance(seconds_per_tick, int) or isinstance(seconds_per_tick, bool) or seconds_per_tick <= 0:
            raise ValueError("MIMIC workload requires explicit positive --seconds-per-tick")
        if not isinstance(workload, dict):
            raise ValueError("MIMIC workload must be an object")
        source = workload.get("source", {})
        if not isinstance(source, dict):
            raise ValueError("invalid MIMIC workload source")
        if (workload.get("schema_version") != 1 or source.get("doi") != DATASET_DOI
                or source.get("checksums") != MIMIC_CHECKSUMS):
            raise ValueError("invalid MIMIC workload provenance")
        records = workload.get("samples")
        if not isinstance(records, list) or not 1 <= len(records) <= 20:
            raise ValueError("MIMIC workload must contain 1 to 20 samples")
        samples_list = []
        last = -1
        for i, item in enumerate(records):
            if not isinstance(item, dict) or item.get("id") != f"TRACE-{i + 1:02}" or item.get("specimen_type") != "blood_unspecified":
                raise ValueError("invalid MIMIC sample record")
            seconds = item.get("relative_charttime_seconds")
            if not isinstance(seconds, int) or isinstance(seconds, bool) or seconds < 0 or seconds < last or seconds >= 86400:
                raise ValueError("invalid or unsorted MIMIC relative time")
            last = seconds
            # Rounding up ensures an event is never released before charttime.
            tick = (seconds + seconds_per_tick - 1) // seconds_per_tick
            samples_list.append(Sample(item["id"], "ICU取样点", "实验室交付点", tick, None,
                                       specimen_type=item["specimen_type"]))
        if records[0]["relative_charttime_seconds"] != 0:
            raise ValueError("MIMIC relative timeline must begin at zero")
        samples = tuple(samples_list)
        name = "Tartu 路网与 MIMIC 单次 ICU 采样时序（跨机构仿真）"
        max_ticks = max(s.release_tick for s in samples) + (600 if timing else 200)
        if max_ticks > 10000:
            raise ValueError("workload exceeds 10000 ticks; choose a coarser test step")
        metadata = {"workload_kind": "mimic_single_stay", "dataset_doi": DATASET_DOI,
                    "timing_assumption": f"每步对应 {seconds_per_tick} 秒仅为测试设定，不是实际路段耗时。",
                    "mapping_assumption": "采样记录假设为 ICU 到实验室的任务；原始数据未记录机器人运输路线。"}
    if timing is not None:
        metadata = dict(metadata or {})
        metadata.update({"time_model": "nominal_source_based", "seconds_per_tick": seconds_per_tick,
                         "reference_metres": round(timing["reference_metres"], 3),
                         "metres_per_pixel": round(timing["metres_per_pixel"], 9),
                         "nominal_speed_m_per_s": timing["nominal_speed_m_per_s"],
                         "source_adapter_sha256": timing["tiago_adapter_sha256"],
                         "timing_assumption": "按公开比例尺与标称速度估算，不是实际路段耗时；取样借鉴另一医院模型，交付耗时为假设，未包含人群与门等待。"})
        name += " · 标称时间估计"
    important = [r.position for r in robots] + [s.position for s in stations.values()]
    if any(graph.shortest_path(a, b) is None for a in important for b in important):
        raise ValueError("required hospital waypoints are not mutually reachable")
    return Scenario(name, graph, stations, robots, samples, duration_jitter=0,
                    max_ticks=max_ticks, metadata=metadata)


def main() -> int:
    parser = argparse.ArgumentParser(description="公开医院路网适配；支持明确标注的跨机构时序测试")
    parser.add_argument("--map-file", required=True, help="固定版本的上游 hospital_2nd_floor.building.yaml")
    parser.add_argument("--output", required=True, help="输出目录；不覆盖已有结果")
    parser.add_argument("--workload", help="mimic_workload 生成的相对时序 JSON")
    parser.add_argument("--seconds-per-tick", type=int, help="相对时序到仿真步的显式假设；不是实测值")
    parser.add_argument("--tiago-config", help="固定版本 TIAGo 适配器，启用基于公开参数的时间估计")
    parser.add_argument("--model", help="可选的仿真训练预测器 JSON；额外运行 learned 策略")
    args = parser.parse_args()
    try:
        workload = json.loads(Path(args.workload).read_text(encoding="utf-8")) if args.workload else None
        config = Path(args.tiago_config).read_bytes() if args.tiago_config else None
        parameters = json.loads(Path(args.model).read_text(encoding="utf-8")) if args.model else None
        scenario = make_scenario(Path(args.map_file).read_bytes(), workload, args.seconds_per_tick, config)
        output = Path(args.output)
        if (output / "comparison.json").exists():
            raise FileExistsError(f"refusing to overwrite existing output: {output / 'comparison.json'}")
        policies = ("nearest", "world", "learned") if parameters is not None else ("nearest", "world")
        for policy in policies:
            check_output_directory(output / policy)
        results = {policy: Simulator(scenario, policy, 0, parameters if policy == "learned" else None).run()
                   for policy in policies}
        for policy, result in results.items():
            write_run(result, output / policy)
        comparison = {
            "scenario": scenario.name,
            "source_commit": SOURCE_COMMIT,
            "source_sha256": SOURCE_SHA256,
            "note": ("跨机构采样时序及运输映射为测试输入，时间为模型估计，不是实测运输或临床证据。"
                     if workload and config else "跨机构采样时序仅作测试输入；路线映射及步长为假设，不是实测运输或临床证据。"
                     if workload else "根据公开比例尺与标称速度估计，未包含完整实际延迟。"
                     if config else "公开路网与人工设置的功能测试任务；图边为抽象步，不是实际时间或临床证据。"),
            "metadata": scenario.metadata,
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
