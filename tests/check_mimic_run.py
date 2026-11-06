"""Check a private runner-local MIMIC-derived replay without printing records."""
import json
from pathlib import Path
import sys

trace = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
root = Path(sys.argv[2])
seconds_per_tick = int(sys.argv[3])
assert len(trace["samples"]) == 4
offsets = [s["relative_charttime_seconds"] for s in trace["samples"]]
assert offsets == sorted(offsets) and offsets[0] == 0 and offsets[-1] < 86400
for policy in ("nearest", "world"):
    result = json.loads((root / policy / "result.json").read_text(encoding="utf-8"))
    assert result["summary"]["status"] == "completed", result["summary"]
    assert result["summary"]["completed_count"] == 4
    assert result["summary"]["late_count"] is None
    assert result["scenario"]["metadata"]["workload_kind"] == "mimic_single_stay"
    samples = sorted(result["samples"], key=lambda s: s["id"])
    assert [s["release_tick"] for s in samples] == [(x + seconds_per_tick - 1) // seconds_per_tick for x in offsets]
    assert all(s["specimen_type"] == "blood_unspecified" and s["collected_at"] is None
               and s["rule_profile"] is None and s["deadline_tick"] is None for s in samples)
    edges = {(tuple(a), tuple(b)) for a, b in result["scenario"]["graph"]["edges"]}
    for previous, current in zip(result["history"], result["history"][1:]):
        before = {r["id"]: tuple(r["position"]) for r in previous["robots"]}
        after = {r["id"]: tuple(r["position"]) for r in current["robots"]}
        assert len(set(after.values())) == len(after)
        for rid in before:
            assert before[rid] == after[rid] or (before[rid], after[rid]) in edges
        assert not (before["R1"] == after["R2"] and before["R2"] == after["R1"])
    print(f"{policy}: 4/4 replay tasks; source-directed moves; no deadline assumed; cross-institution mapping is simulated")
