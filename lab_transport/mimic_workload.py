"""Extract a bounded single-stay sample timeline from the public MIMIC-IV demo.

MIMIC charttime usually approximates acquisition; these are NOT transport
orders. Patient-level dates are shifted independently and must not be merged.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable

CHECKSUMS = {
    "labitems": "72ea5f020469fd24543bd98ab3bd0a4f645a5a4b7802d9d52af20d736f9d76db",
    "labevents": "d51a4cf1ea63245abe7d544b0d50a928e9ee2ec5eb9a5ce85500aed0bd6c6dfc",
    "icustays": "e05e81aa52a3022e522b6832a898101a69b84e64c17bb344d819e458d5bc21b3",
}
DATASET_DOI = "https://doi.org/10.13026/dp1f-ex47"


def _when(value: str) -> datetime:
    return datetime.fromisoformat(value)


def select_timeline(labitems: Iterable[dict[str, str]], labevents: Iterable[dict[str, str]],
                    icustays: Iterable[dict[str, str]], count: int = 4) -> dict:
    """Select the first eligible single ICU stay in stable pseudonym order.

    Deduplicate measurements by specimen_id and use earliest charttime for
    each group. Do not export patient IDs, test results, absolute dates or
    inferred sample-specific clinical requirements.
    """
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 20:
        raise ValueError("count must be between 1 and 20")
    blood_items = {r["itemid"] for r in labitems if r.get("fluid", "").strip().lower() == "blood"}
    if not blood_items:
        raise ValueError("no blood item definitions found")
    stays = []
    for r in icustays:
        if "icu" not in r.get("first_careunit", "").lower():
            continue
        if not all(r.get(k) for k in ("subject_id", "hadm_id", "stay_id", "intime", "outtime")):
            continue
        start, end = _when(r["intime"]), _when(r["outtime"])
        if end <= start:
            raise ValueError("ICU stay end must follow start")
        stays.append((int(r["subject_id"]), int(r["hadm_id"]), start, int(r["stay_id"]), end))
    if not stays:
        raise ValueError("no ICU stays found")

    # A specimen can have multiple test-result rows. Do not treat them as
    # separate transport requests. Keep only the earliest charttime.
    specimens: dict[tuple[int, int, str], datetime] = {}
    for r in labevents:
        if r.get("itemid") not in blood_items or not all(r.get(k) for k in ("subject_id", "hadm_id", "specimen_id", "charttime")):
            continue
        key = (int(r["subject_id"]), int(r["hadm_id"]), r["specimen_id"])
        time = _when(r["charttime"])
        if key not in specimens or time < specimens[key]:
            specimens[key] = time
    for subject, admission, start, _stay, end in sorted(stays):
        eligible = sorted((when, key[2]) for key, when in specimens.items()
                          if key[0] == subject and key[1] == admission and start <= when < end)
        if not eligible:
            continue
        first = eligible[0][0]
        selected = [(when, sid) for when, sid in eligible if when < first + timedelta(hours=24)][:count]
        if len(selected) < count:
            continue
        return {
            "schema_version": 1,
            "source": {"dataset": "MIMIC-IV Clinical Database Demo", "version": "2.2",
                       "doi": DATASET_DOI, "license": "ODbL-1.0", "checksums": CHECKSUMS},
            "selection": "第一个按匿名标识排序、24小时内至少包含所需数量血液样本的单次 ICU stay；按 specimen_id 合并检验行，取最早 charttime。",
            "limitations": ["charttime通常近似采样时间，不是确定的转运请求或交付时间。",
                            "每人日期独立平移，不可跨患者按绝对日期拼接。",
                            "血液检验不一定需要机器人转运；取点与目的地是跨机构仿真假设。",
                            "本示例不含实际患者标识、检验结果、绝对日期或临床时限。"],
            "samples": [{"id": f"TRACE-{i + 1:02}", "specimen_type": "blood_unspecified",
                         "relative_charttime_seconds": int((when - first).total_seconds())}
                        for i, (when, _) in enumerate(selected)],
        }
    raise ValueError("no single ICU stay contains enough eligible specimens within 24 hours")


def load_verified_gzip(path: str | Path, kind: str) -> list[dict[str, str]]:
    file = Path(path)
    if hashlib.sha256(file.read_bytes()).hexdigest() != CHECKSUMS[kind]:
        raise ValueError(f"{kind} SHA-256 differs from MIMIC-IV demo 2.2")
    with gzip.open(file, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description="提取公开 MIMIC 示例的单个 ICU stay 相对采样时间")
    parser.add_argument("--labitems", required=True)
    parser.add_argument("--labevents", required=True)
    parser.add_argument("--icustays", required=True)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        path = Path(args.output)
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {path}")
        timeline = select_timeline(load_verified_gzip(args.labitems, "labitems"),
                                   load_verified_gzip(args.labevents, "labevents"),
                                   load_verified_gzip(args.icustays, "icustays"), args.count)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(timeline, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"source": DATASET_DOI, "sample_count": len(timeline["samples"]),
                          "relative_timeline": "single ICU stay", "transport_requests": "not observed"}, ensure_ascii=False))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"数据提取失败：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
