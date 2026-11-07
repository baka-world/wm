"""Check episode grouping, metric accounting and saved dynamics experiment."""
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab_transport.dynamics_experiment import BEHAVIOR_POLICIES, HORIZONS, PREDICTORS, REGIMES, SPLITS, aggregate
from lab_transport.dynamics_model import validate

root = Path(sys.argv[1])
parameters = json.loads((root / "dynamics-parameters.json").read_text(encoding="utf-8"))
results = json.loads((root / "dynamics-results.json").read_text(encoding="utf-8"))
validate(parameters)
seed_sets = [set(seeds) for _, seeds in SPLITS.values()]
assert all(not left & right for index, left in enumerate(seed_sets) for right in seed_sets[index + 1:])
assert results["training"]["episodes"] == 48
assert results["training"]["completed_tasks"] <= 48 * 6
assert results["training"]["motion_examples"] == parameters["training_motion_examples"] > 0
assert results["training"]["service_examples"] == parameters["training_service_examples"] > 0
assert 0 < results["training"]["service_finishes"] < results["training"]["service_examples"]
assert parameters["provenance"]["seeds"] == list(SPLITS["train"][1])
expected = {(split, layout, seed, regime, policy)
            for split, (layout, seeds) in SPLITS.items() if split != "train"
            for seed in seeds for regime in REGIMES for policy in BEHAVIOR_POLICIES}
cases = results["cases"]
actual = {(c["split"], c["layout"], c["seed"], c["regime"], c["behavior_policy"]) for c in cases}
assert len(cases) == len(expected) and actual == expected
for case in cases:
    assert 0 <= case["completed"] <= 6
    for predictor in PREDICTORS:
        count = case["one_step"][predictor]
        assert count["robot_total"] == 2 * count["joint_total"]
        assert 0 <= count["robot_position_correct"] <= count["robot_total"]
        assert 0 <= count["robot_actual_moves_correct"] <= count["robot_actual_moves"] <= count["robot_total"]
        assert 0 <= count["sample_status_correct"] <= count["sample_total"]
        assert 0 <= count["service_finish_exact_true_positive"] <= count["service_actual_finishes"]
    for horizon in HORIZONS:
        block = case["multi_step"][str(horizon)]
        coverage = block["coverage"]
        assert coverage["candidate_windows"] == sum(value for key, value in coverage.items() if key != "candidate_windows")
        assert all(block["counts"][name]["joint_total"] == coverage["eligible"] for name in PREDICTORS)
for split in ("validation", "test_tartu", "test_clinic"):
    assert results["aggregates"][split] == aggregate(cases, split)
    assert results["aggregates"][split]["episodes"] == 16

if len(sys.argv) > 2:
    reference = Path(sys.argv[2])
    saved_model = json.loads((reference / "dynamics-parameters.json").read_text(encoding="utf-8"))
    saved_result = json.loads((reference / "dynamics-results.json").read_text(encoding="utf-8"))
    validate(saved_model)
    for key in ("motion_weights", "service_weights"):
        assert all(math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-8) for a, b in zip(parameters[key], saved_model[key]))
    def without_timing(value):
        if isinstance(value, dict):
            return {key: without_timing(item) for key, item in value.items() if key != "cpu_seconds"}
        if isinstance(value, list):
            return [without_timing(item) for item in value]
        return value
    assert without_timing(results) == without_timing(saved_result), "saved non-timing results differ"
print('Verified dynamics: grouped seeds, accounting, conditional rollout coverage and saved results')
