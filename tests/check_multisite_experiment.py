"""Validate a full experiment and reproduce saved non-timing results."""
from collections import Counter
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab_transport.multisite_experiment import DROPOFFS, LAYOUTS, PICKUPS, POLICIES, REGIMES, SEEDS

result = json.loads((Path(sys.argv[1]) / "experiment-results.json").read_text(encoding="utf-8"))
reference = json.loads((Path(sys.argv[2]) / "experiment-results.json").read_text(encoding="utf-8"))
assert result["test_episodes"] == len(result["cases"]) == 20
assert result["rollout_horizon_ticks"] == 120
cases = {(r["layout"], r["regime"], r["seed"]): r for r in result["cases"]}
expected_cases = {(layout, regime, seed) for layout in LAYOUTS for regime in REGIMES for seed in SEEDS[layout]}
assert set(cases) == expected_cases
assert set(result["model_training_seeds"]).isdisjoint(seed for _, _, seed in cases)
for layout, regime, seed in sorted(cases):
    row = cases[layout, regime, seed]
    paired = cases[layout, "normal", seed]
    assert row["samples"] == paired["samples"]
    assert Counter(s["pickup"] for s in row["samples"]) == {name: 3 for name in PICKUPS[layout]}
    assert Counter(s["dropoff"] for s in row["samples"]) == {
        name: 6 // len(DROPOFFS[layout]) for name in DROPOFFS[layout]}
    releases = [s["release_tick"] for s in row["samples"]]
    assert releases[0] == 0 and all(b - a in (4, 8, 12, 16) for a, b in zip(releases, releases[1:]))
    assert len(row["outages"]) == (0 if regime == "normal" else 1)
    for intervals in row["outages"].values():
        assert len(intervals) == 1
        start, end = intervals[0]
        assert 10 <= start <= 55 and 6 <= end - start <= 14
    assert set(row["policies"]) == set(POLICIES)
    for policy, metrics in row["policies"].items():
        assert 0 <= metrics["completed"] <= metrics["total"] == 6
        assert math.isfinite(metrics["restricted_mean_flow_ticks"])
        assert metrics["cpu_seconds"] >= 0 and metrics["candidate_evaluations"] >= len(metrics["assignments"])
        assert metrics["discrete_safety_violations"] == 0
        if policy == "rollout":
            assert 0 <= metrics["censored_candidate_evaluations"] <= metrics["candidate_evaluations"]
for layout in LAYOUTS:
    for regime in REGIMES:
        summary = result["aggregates"][layout][regime]
        assert summary["episodes"] == 5
        assert summary["offline_same_world_decisions"]["completed_records"] + summary["offline_same_world_decisions"]["censored_excluded"] == 30
        for metrics in summary["policies"].values():
            assert 0 <= metrics["completed"] <= metrics["total"] == 30


def without_timing(value):
    if isinstance(value, dict):
        return {k: without_timing(v) for k, v in value.items() if k not in {"cpu_seconds", "mean_episode_cpu_seconds"}}
    if isinstance(value, list):
        return [without_timing(x) for x in value]
    return value


assert without_timing(result) == without_timing(reference), "saved non-timing results differ"
print("Verified 20 paired scenarios, no rollout future leak via unit tests, discrete safety and saved non-timing results")
