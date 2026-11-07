"""Check a generated experiment's split, accounting and model integrity."""
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab_transport.experiment import SHIFT_SEEDS, TEST_SEEDS, TRAIN_SEEDS, testing_cases
from lab_transport.learned_model import validate

root = Path(sys.argv[1])
model = json.loads((root / "learned-parameters.json").read_text(encoding="utf-8"))
result = json.loads((root / "experiment-results.json").read_text(encoding="utf-8"))
validate(model)
assert set(TRAIN_SEEDS).isdisjoint(TEST_SEEDS) and set(TRAIN_SEEDS).isdisjoint(SHIFT_SEEDS)
assert model["provenance"]["training_seeds"] == list(TRAIN_SEEDS)
assert result["training_records"] == model["training_records"] > 0
assert result["training_records"] + result["training_censored_excluded"] == 40 * 6
assert result["test_episodes"] == len(result["cases"]) == 30
assert {(row["regime"], row["seed"]) for row in result["cases"]} == set(testing_cases())
for regime, summary in result["aggregates"].items():
    assert summary["episodes"] == 10
    assert 0 <= summary["same_assignment_trace_world_and_learned"] <= 10
    assert summary["offline_same_decisions"]["count"] + result["offline_censored_excluded"][regime] == 60
    for policy, values in summary["policies"].items():
        assert policy in {"nearest", "world", "learned"}
        assert 0 <= values["completed"] <= values["total"] == 60
        assert math.isfinite(values["mean_restricted_flow_ticks"])
        assert values["mean_episode_cpu_seconds"] >= 0
    for values in summary["offline_same_decisions"]["errors"].values():
        assert all(value is None or math.isfinite(value) and value >= 0 for value in values.values())
print('Experiment verified: disjoint seeds, simulation-only model, completion accounting and held-out prediction metrics')

if len(sys.argv) > 2:
    reference_root = Path(sys.argv[2])
    reference_model = json.loads((reference_root / "learned-parameters.json").read_text(encoding="utf-8"))
    reference = json.loads((reference_root / "experiment-results.json").read_text(encoding="utf-8"))
    validate(reference_model)
    for field in ("means", "scales", "pickup_weights", "completion_weights"):
        assert all(math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-8)
                   for a, b in zip(model[field], reference_model[field])), field
    current_cases = {(x["regime"], x["seed"]): x for x in result["cases"]}
    reference_cases = {(x["regime"], x["seed"]): x for x in reference["cases"]}
    assert current_cases.keys() == reference_cases.keys()
    for key in current_cases:
        for policy in ("nearest", "world", "learned"):
            current = current_cases[key]["policies"][policy]
            expected = reference_cases[key]["policies"][policy]
            for field in ("completed", "total", "timed_out", "candidate_evaluations", "assignments"):
                assert current[field] == expected[field], (key, policy, field)
            assert math.isclose(current["restricted_mean_flow_ticks"], expected["restricted_mean_flow_ticks"], abs_tol=1e-8)
    for regime in ("normal", "short_outage", "shifted"):
        current = result["aggregates"][regime]["offline_same_decisions"]["errors"]
        expected = reference["aggregates"][regime]["offline_same_decisions"]["errors"]
        for policy in ("world", "learned"):
            for metric in ("pickup_mae_ticks", "completion_mae_ticks"):
                assert math.isclose(current[policy][metric], expected[policy][metric], abs_tol=1e-3)
    print('Saved results reproduced: parameters, assignments and metrics match; CPU timings intentionally excluded')
