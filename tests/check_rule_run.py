"""Verify the deliberately synthetic rule-priority diagnostic scenario."""
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
results = {p: json.loads((root / p / "result.json").read_text(encoding="utf-8"))
           for p in ("nearest", "world", "rule")}
for policy, result in results.items():
    assert result["summary"]["completed_count"] == 2
    assert result["scenario"]["metadata"]["rule_mode"] == "receipt_due_as_delivery_proxy"
    assert result["scenario"]["metadata"]["seconds_per_tick"] == 10
    assert result["scenario"]["metadata"]["nominal_speed_m_per_s"] == 0.5
    assert result["scenario"]["metadata"]["reference_metres"] == 67.2
    samples = {s["id"]: s for s in result["samples"]}
    assert (samples["S01"]["deadline_tick"], samples["S02"]["deadline_tick"]) == (180, 120)
    assert all(s["collected_at"] is not None for s in samples.values())
    assert all("received_at" not in s for s in samples.values())
    edges = {(tuple(a), tuple(b)) for a, b in result["scenario"]["graph"]["edges"]}
    for a, b in zip(result["history"], result["history"][1:]):
        before, after = tuple(a["robots"][0]["position"]), tuple(b["robots"][0]["position"])
        assert before == after or (before, after) in edges
    first = next(e["sample_id"] for e in result["events"] if e["type"] == "assigned")
    assert first == ("S02" if policy == "rule" else "S01")
    print(f"{policy}: 2/2 synthetic tasks; first={first}; delivery-proxy tardiness={result['summary']['late_count']}")
assert results["rule"]["summary"]["late_count"] == 0
assert results["nearest"]["summary"]["late_count"] == 1
assert results["world"]["summary"]["late_count"] == 1
