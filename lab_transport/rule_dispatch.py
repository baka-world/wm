"""Research-only rule-aware scheduling with an explicit proxy deadline.

Projected delivery is not laboratory receipt. A delivery no later than the
proxy deadline cannot establish compliance with a clinical SOP.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import timedelta
import json
import math
from pathlib import Path
import sys
from typing import Any

from .domain import Sample, Scenario
from .hospital_case import make_scenario
from .report import check_output_directory, write_run
from .simulation import Simulator
from .specimen_rules import _timestamp, load_profiles


def apply_rule_proxies(base: Scenario, document: dict[str, Any],
                       profiles: dict[str, dict[str, Any]]) -> Scenario:
    """Convert explicitly supplied collection times into delivery proxy ticks."""
    if not isinstance(document, dict) or document.get("purpose") != "cross_institution_synthetic_test":
        raise ValueError("rule example must explicitly declare cross_institution_synthetic_test")
    meta = dict(base.metadata or {})
    if meta.get("time_model") != "nominal_source_based":
        raise ValueError("rule proxy scheduling requires the explicit nominal time model")
    tick_seconds = meta.get("seconds_per_tick")
    if not isinstance(tick_seconds, int) or isinstance(tick_seconds, bool) or tick_seconds <= 0:
        raise ValueError("invalid nominal tick duration")
    origin = _timestamp(document.get("clock_origin"), "clock_origin")
    if origin is None:
        raise ValueError("clock_origin is required")
    robot_count = document.get("robot_count")
    if not isinstance(robot_count, int) or isinstance(robot_count, bool) or not 1 <= robot_count <= len(base.robots):
        raise ValueError("robot_count must select available robots")
    records = document.get("samples")
    if not isinstance(records, list) or not records:
        raise ValueError("samples must be a nonempty list")
    samples = []
    rule_sources = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("each sample must be an object")
        sid, release = record.get("id"), record.get("release_tick")
        if not isinstance(sid, str) or not sid.strip():
            raise ValueError("sample id is required")
        if not isinstance(release, int) or isinstance(release, bool) or not 0 <= release < base.max_ticks:
            raise ValueError("invalid sample release_tick")
        stype, rule_id = record.get("specimen_type"), record.get("rule_profile")
        for key in ("specimen_type", "rule_profile", "collected_at", "packaging", "temperature_condition"):
            value = record.get(key)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{key} must be a nonempty string")
        collected = _timestamp(record.get("collected_at"), "collected_at")
        if collected and collected > origin + timedelta(seconds=release * tick_seconds):
            raise ValueError("collection cannot occur after task release")
        due_tick = None
        if rule_id is not None:
            if rule_id not in profiles:
                raise ValueError("unknown rule profile")
            profile = profiles[rule_id]
            if stype != profile["specimen_type"]:
                raise ValueError("explicit rule profile does not match the specimen type")
            limit = profile["requirements"].get("max_collection_to_receipt_seconds")
            if collected is not None and limit is not None:
                # Floor gives a simulated delivery proxy using the source
                # reception limit. Actual receipt may have a different time.
                due_tick = math.floor(((collected + timedelta(seconds=limit)) - origin).total_seconds() / tick_seconds)
            rule_sources[rule_id] = {"institution": profile["institution"], "scope": profile["scope"],
                                     "source_url": profile["source_url"]}
        samples.append(Sample(sid, "ICU取样点", "实验室交付点", release, due_tick,
                              specimen_type=stype, collected_at=record.get("collected_at"),
                              packaging=record.get("packaging"), temperature_condition=record.get("temperature_condition"),
                              rule_profile=rule_id))
    if len({s.id for s in samples}) != len(samples):
        raise ValueError("sample ids must be unique")
    meta.update({"rule_mode": "receipt_due_as_delivery_proxy", "rule_sources": rule_sources,
                 "rule_limit": "以仿真交付完成近似接收时间，仅作为调度代理；模拟按时交付不能证明实际接收或临床合规。",
                 "rule_scope": "不同机构的公开规则与地图仅用于人工构造的研究测试。"})
    return replace(base, name="公开路网与跨机构样本规则（人工测试）", robots=base.robots[:robot_count],
                   samples=tuple(samples), metadata=meta)


def main() -> int:
    parser = argparse.ArgumentParser(description="有来源的规则调度代理示例；不是临床判断")
    parser.add_argument("--map-file", required=True)
    parser.add_argument("--tiago-config", required=True)
    parser.add_argument("--seconds-per-tick", required=True, type=int)
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--model", help="可选的仿真训练预测器 JSON；额外运行 learned_rule")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        profiles = load_profiles(args.profiles)
        parameters = json.loads(Path(args.model).read_text(encoding="utf-8")) if args.model else None
        doc = json.loads(Path(args.samples).read_text(encoding="utf-8"))
        base = make_scenario(Path(args.map_file).read_bytes(), seconds_per_tick=args.seconds_per_tick,
                             tiago_config=Path(args.tiago_config).read_bytes())
        scenario = apply_rule_proxies(base, doc, profiles)
        output = Path(args.output)
        if (output / "comparison.json").exists():
            raise FileExistsError(f"refusing to overwrite existing output: {output / 'comparison.json'}")
        policies = ("nearest", "world", "rule", "learned_rule") if parameters is not None else ("nearest", "world", "rule")
        for policy in policies:
            check_output_directory(output / policy)
        results = {policy: Simulator(scenario, policy, 0, parameters if policy == "learned_rule" else None).run()
                   for policy in policies}
        for policy, result in results.items():
            write_run(result, output / policy)
        comparison = {"scenario": scenario.name,
                      "note": "跨机构人工样本，仅对比模拟调度；代理上限不是实验室实际接收或临床合规。",
                      "policies": {policy: {**result["summary"],
                                            "first_assigned_sample": next((e["sample_id"] for e in result["events"]
                                                                            if e["type"] == "assigned"), None)}
                                   for policy, result in results.items()}}
        (output / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(comparison, ensure_ascii=False))
        return 0 if all(r["summary"]["status"] == "completed" for r in results.values()) else 2
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"规则调度失败：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
