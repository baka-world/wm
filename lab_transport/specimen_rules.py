"""Check explicitly recorded specimen metadata against a cited rule profile.

This is a simulation/research data check, not a clinical decision system.
It never converts uncalibrated simulator ticks to wall-clock time.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any
from urllib.parse import urlparse


def load_profiles(path: str | Path) -> dict[str, dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1 or not isinstance(data.get("profiles"), list):
        raise ValueError("invalid rule profile schema")
    profiles = {}
    for profile in data["profiles"]:
        if not isinstance(profile, dict):
            raise ValueError("each rule profile must be an object")
        for key in ("id", "label", "specimen_type", "institution", "scope", "source_url", "source_accessed"):
            if not isinstance(profile.get(key), str) or not profile[key].strip():
                raise ValueError(f"profile {key} must be a nonempty string")
        if urlparse(profile["source_url"]).scheme != "https" or not urlparse(profile["source_url"]).netloc:
            raise ValueError("profile source_url must be an HTTPS URL")
        try:
            datetime.strptime(profile["source_accessed"], "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError("profile source_accessed must be YYYY-MM-DD") from exc
        requirements = profile.get("requirements")
        if not isinstance(requirements, dict) or not requirements:
            raise ValueError("profile requirements must be a nonempty object")
        allowed = {"packaging", "temperature_condition", "max_collection_to_receipt_seconds",
                   "max_collection_to_analysis_seconds"}
        if set(requirements) - allowed:
            raise ValueError("unknown profile requirement")
        for name, value in requirements.items():
            if name.startswith("max_"):
                if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                    raise ValueError(f"{name} must be a positive integer")
            elif not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        if profile["id"] in profiles:
            raise ValueError("duplicate rule profile id")
        profiles[profile["id"]] = profile
    if not profiles:
        raise ValueError("at least one rule profile is required")
    return profiles


def _timestamp(value: Any, field: str) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO 8601 timestamp with timezone")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid {field} timestamp") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")
    return result


def evaluate(sample: dict[str, Any], profiles: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(sample, dict) or not isinstance(sample.get("id"), str) or not sample["id"].strip():
        raise ValueError("sample must have an id")
    profile_id = sample.get("rule_profile")
    if not isinstance(profile_id, str) or profile_id not in profiles:
        raise ValueError("sample must explicitly select a known rule profile")
    profile = profiles[profile_id]
    result = {"sample_id": sample["id"], "rule_profile": profile_id,
              "institution": profile["institution"], "scope": profile["scope"],
              "source_url": profile["source_url"], "checks": {},
              "note": "仅比较录入数据；不是临床判断、自动接收判定或真实运输条件证明。"}
    if sample.get("specimen_type") != profile["specimen_type"]:
        result["applicability"] = "not_applicable" if sample.get("specimen_type") is not None else "unknown"
        return result
    result["applicability"] = "applicable_to_recorded_type"
    req = profile["requirements"]
    for field in ("packaging", "temperature_condition"):
        if field not in req:
            continue
        value = sample.get(field)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{field} must be a nonempty string when provided")
        result["checks"][field] = {"status": "unknown" if value is None else ("matches_record" if value == req[field] else "differs_from_record"),
                                   "expected": req[field], "recorded": value}
    collected = _timestamp(sample.get("collected_at"), "collected_at")
    received = _timestamp(sample.get("received_at"), "received_at")
    analyzed = _timestamp(sample.get("analyzed_at"), "analyzed_at")
    if collected and received and received < collected or collected and analyzed and analyzed < collected:
        raise ValueError("receipt or analysis cannot precede collection")
    for field, end in (("max_collection_to_receipt_seconds", received),
                       ("max_collection_to_analysis_seconds", analyzed)):
        if field not in req:
            continue
        elapsed = (end - collected).total_seconds() if collected and end else None
        result["checks"][field] = {
            "status": "unknown" if elapsed is None else ("within_recorded_limit" if elapsed <= req[field] else "beyond_recorded_limit"),
            "limit_seconds": req[field], "elapsed_seconds": elapsed,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="按来源和适用范围检查样本记录（非临床判断）")
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--samples", required=True)
    args = parser.parse_args()
    try:
        profiles = load_profiles(args.profiles)
        data = json.loads(Path(args.samples).read_text(encoding="utf-8"))
        samples = data.get("samples") if isinstance(data, dict) else None
        if not isinstance(samples, list) or not samples:
            raise ValueError("samples must be a nonempty list")
        print(json.dumps({"results": [evaluate(sample, profiles) for sample in samples]}, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"规则检查失败：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
