"""Check validation-only checkpoint selection and fresh held-out cases."""
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab_transport.dynamics_experiment import BEHAVIOR_POLICIES, HORIZONS, PREDICTORS, REGIMES, SPLITS, aggregate
from lab_transport.dynamics_revision import CHECKPOINTS, SPLITS_V2, validate_v2

root = Path(sys.argv[1])
parameters = json.loads((root / "dynamics-parameters.json").read_text(encoding="utf-8"))
results = json.loads((root / "dynamics-results.json").read_text(encoding="utf-8"))
validate_v2(parameters)
prior_seeds = {s for _, values in SPLITS.values() for s in values}
assert all(set(SPLITS_V2[name][1]).isdisjoint(prior_seeds) for name in ("test_tartu", "test_clinic"))
assert results["training"]["episodes"] == 48
assert results["training"]["motion_examples"] == parameters["training_motion_examples"] > 0
assert results["training"]["service_examples"] == parameters["training_service_examples"] > 0
assert results["schedule_audit"]["cross_split_duplicate_groups"] == 0
assert results["schedule_audit"]["cross_split_duplicates"] == []
assert results["schedule_audit"]["cross_split_task_duplicate_groups"] == 0
assert results["schedule_audit"]["cross_split_task_duplicates"] == []
assert results["schedule_audit"]["unique_task_patterns"] == 24
assert results["schedule_audit"]["scenario_patterns"] == results["schedule_audit"]["unique_patterns"] == 48
assert parameters["provenance"]["train_seeds"] == list(SPLITS_V2["train"][1])
assert parameters["provenance"]["validation_seeds"] == list(SPLITS_V2["validation"][1])
candidates = results["selection"]["validation_candidates"]
assert [item["epoch"] for item in candidates] == list(CHECKPOINTS)
for item in candidates:
    c = item["counts"]
    joint = c["joint_correct"] / c["joint_total"]
    denominator = c["service_actual_finishes"] + c["service_predicted_finishes"]
    f1 = 2 * c["service_finish_exact_true_positive"] / denominator if denominator else 0.0
    assert math.isclose(item["selection_score"], 0.5 * (joint + f1), abs_tol=1e-8)
def exact_score(item):
    c = item["counts"]
    denominator = c["service_actual_finishes"] + c["service_predicted_finishes"]
    return 0.5 * (c["joint_correct"] / c["joint_total"] +
                  (2 * c["service_finish_exact_true_positive"] / denominator if denominator else 0.0))
selected = max(candidates, key=lambda x: (exact_score(x), -x["epoch"]))["epoch"]
assert selected == parameters["epoch"] == results["selection"]["selected_epoch"]
expected = {(split, layout, seed, regime, policy)
            for split, (layout, seeds) in SPLITS_V2.items() if split != "train"
            for seed in seeds for regime in REGIMES for policy in BEHAVIOR_POLICIES}
cases = results["cases"]
actual = {(c["split"], c["layout"], c["seed"], c["regime"], c["behavior_policy"]) for c in cases}
assert len(cases) == len(expected) and actual == expected
for case in cases:
    for name in PREDICTORS:
        c = case["one_step"][name]
        assert c["robot_total"] == 2 * c["joint_total"]
        assert 0 <= c["robot_position_correct"] <= c["robot_total"]
        assert 0 <= c["service_finish_exact_true_positive"] <= c["service_actual_finishes"]
    for horizon in HORIZONS:
        b = case["multi_step"][str(horizon)]
        coverage = b["coverage"]
        assert coverage["candidate_windows"] == sum(v for k, v in coverage.items() if k != "candidate_windows")
        assert all(b["counts"][name]["joint_total"] == coverage["eligible"] for name in PREDICTORS)
for split in ("validation", "test_tartu", "test_clinic"):
    assert results["aggregates"][split] == aggregate(cases, split)
    assert results["aggregates"][split]["episodes"] == 16
if len(sys.argv) > 2:
    reference = Path(sys.argv[2])
    saved_model = json.loads((reference / "dynamics-parameters.json").read_text(encoding="utf-8"))
    saved_results = json.loads((reference / "dynamics-results.json").read_text(encoding="utf-8"))
    validate_v2(saved_model)
    for key in ("motion_weights", "service_weights"):
        assert all(math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-8) for a, b in zip(parameters[key], saved_model[key]))
    def without_timing(value):
        if isinstance(value, dict):
            return {k: without_timing(v) for k, v in value.items() if k != "cpu_seconds"}
        if isinstance(value, list):
            return [without_timing(v) for v in value]
        return value
    assert without_timing(results) == without_timing(saved_results), "saved non-timing results differ"
print('Verified revised dynamics: validation-only selection, fresh seed groups and saved results')
