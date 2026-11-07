"""Check a generated run against the pinned, locally supplied source graph."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab_transport.hospital_case import make_scenario

source, output = Path(sys.argv[1]), Path(sys.argv[2])
scenario = make_scenario(source.read_bytes())
allowed = set(scenario.grid.edges)
assert len(scenario.grid.coordinates) == 32 and len(allowed) == 63
for policy in ("nearest", "world"):
    result = json.loads((output / policy / "result.json").read_text(encoding="utf-8"))
    assert result["summary"]["status"] == "completed"
    assert result["summary"]["completed_count"] == 4
    assert result["summary"]["late_count"] is None
    assert any(event["type"] == "recovery_planned" for event in result["events"])
    assert all(s["release_tick"] <= s["assigned_tick"] < s["picked_tick"] < s["completed_tick"]
               for s in result["samples"])
    moves = 0
    for previous, current in zip(result["history"], result["history"][1:]):
        before = {r["id"]: tuple(r["position"]) for r in previous["robots"]}
        after = {r["id"]: tuple(r["position"]) for r in current["robots"]}
        assert len(set(after.values())) == len(after)
        assert not (before["R1"] == after["R2"] and before["R2"] == after["R1"])
        for rid in before:
            if before[rid] != after[rid]:
                assert (before[rid], after[rid]) in allowed
                moves += 1
    print(f"{policy}: 4/4 test samples; {moves} source-directed movements; no node overlap or edge swap; deadline unspecified")
