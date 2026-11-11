"""Offline diagnostic of archived PR18 P4/P5 three-particle decisions.

Reads two fixed GitHub Actions ZIP artifacts. No simulation, model loading,
training, GPU or new candidate predictions. The 27 bootstrap outcomes are
resamplings of three ALREADY archived particles, not independent trials.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
from itertools import product
import gzip
import json
from pathlib import Path
from zipfile import ZipFile

ARCHIVES = {
    "confirmation12": ("confirmation", "a5ec5d7a1f886f2d84ba10f3f76decf68d984a26d852f9fd050c7035f8157233"),
    "later8": ("new-full", "0aae5c4e365fab8694b3c239bebe1c3e2c08e2ff617a983f9f672af80afb3dc5"),
}
DRAWS = tuple(product(range(3), repeat=3))


def load(archive_path: Path, name: str):
    folder, expected_sha = ARCHIVES[name]
    actual_sha = sha256(archive_path.read_bytes()).hexdigest()
    if actual_sha != expected_sha:
        raise AssertionError(f"input ZIP hash mismatch: {name}: {actual_sha}")
    with ZipFile(archive_path) as z:
        cases = json.loads(gzip.decompress(z.read(folder + "/cases.json.gz")))
        references = json.loads(gzip.decompress(z.read(folder + "/reference_cases.json.gz")))
        summary = json.loads(z.read(folder + "/summary.json"))
    assert summary["scenarios"] == len(cases) // 2
    return cases, references, summary, actual_sha


def selected(plans, picks=(0, 1, 2)):
    scores = []
    for i, plan in enumerate(plans):
        particles = plan["particle_costs"]
        assert len(particles) == 3 and all(len(v) == 4 for v in particles)
        means = tuple(sum(particles[k][j] for k in picks) / len(picks) for j in range(4))
        scores.append((plan["score"][0], *means, i))
    return min(scores)[-1]


def votes(decision):
    counts = Counter(selected(decision["plans"], picks) for picks in DRAWS)
    return counts[decision["selected_candidate_index"]], dict(sorted(counts.items()))


def leading_margin(plans):
    a, b = sorted((p["score"] for p in plans))[:2]
    for i, (u, v) in enumerate(zip(a, b)):
        if u != v:
            return i, v - u
    raise AssertionError("unique candidate-index tie break not present")


def audit(name: str, cases, references, summary, input_sha):
    by = {(c["seed"], c["arm"]): c for c in cases}
    baseline = {(c["seed"], c["arm"]): c for c in references}
    seeds = sorted({c["seed"] for c in cases})
    assert len(by) == 2 * len(seeds)
    assert len(baseline) == 4 * len(seeds)
    arms = {}
    for arm in ("P4", "P5"):
        states = [d for seed in seeds for d in by[seed, arm]["decisions"]
                  if len(d["plans"]) > 1]
        sensitive = 0
        equal_particles = 0
        for d in states:
            assert selected(d["plans"]) == d["selected_candidate_index"]
            for plan in d["plans"]:
                for j in range(4):
                    original_mean = sum(v[j] for v in plan["particle_costs"]) / 3
                    assert abs(original_mean - plan["score"][j + 1]) < 1e-8
            n, _ = votes(d)
            sensitive += int(n != 27)
            equal_particles += int(all(p["particle_costs"][0] ==
                                       p["particle_costs"][1] ==
                                       p["particle_costs"][2] for p in d["plans"]))
        arms[arm] = {"multi_candidate_states": len(states),
                     "resampling_sensitive": sensitive,
                     "all_candidates_three_particles_identical": equal_particles}

    matched = 0
    mismatched = 0
    flips = []
    scenes = []
    for seed in seeds:
        p4, p5, p1 = by[seed, "P4"], by[seed, "P5"], baseline[seed, "P1"]
        rule = {(d["tick"], d["observation_sha256"]): d for d in p4["decisions"]}
        model = {(d["tick"], d["observation_sha256"]): d for d in p5["decisions"]}
        for key in sorted(rule.keys() & model.keys()):
            a, b = rule[key], model[key]
            if a["candidate_count"] < 2:
                continue
            if [p["matching"] for p in a["plans"]] != [p["matching"] for p in b["plans"]]:
                mismatched += 1
                continue
            matched += 1
            if a["selected"] != b["selected"]:
                n, counts = votes(b)
                dimension, margin = leading_margin(b["plans"])
                flips.append({
                    "seed": seed, "tick": key[0],
                    "observation_sha256": key[1],
                    "rule_selected_index": a["selected_candidate_index"],
                    "v4_selected_index": b["selected_candidate_index"],
                    "v4_selected_support_out_of_27": n,
                    "v4_resampling_sensitive": n != 27,
                    "v4_candidate_vote_counts": counts,
                    "v4_winner_deciding_score_component": dimension,
                    "v4_winner_score_margin": margin,
                    "full_scene_outcome_is_not_single_action_causal_effect": True,
                })
        urgent = lambda case: {
            j["job_id"]: j for j in case["metrics"]["jobs"]
            if j["priority_class"] == "urgent"
        }
        jobs4, jobs5, jobs1 = urgent(p4), urgent(p5), urgent(p1)
        assert jobs4.keys() == jobs5.keys() == jobs1.keys()
        flips_urgent = [{
            "job_id": job,
            "P1_ontime": jobs1[job]["on_time"],
            "P4_ontime": jobs4[job]["on_time"],
            "P5_ontime": jobs5[job]["on_time"],
            "P4_final_tick": jobs4[job]["final_tick"],
            "P5_final_tick": jobs5[job]["final_tick"],
            "deadline_tick": jobs5[job]["deadline_tick"],
        } for job in jobs4 if jobs4[job]["on_time"] != jobs5[job]["on_time"]]
        scenes.append({
            "seed": seed,
            "P1_urgent_ontime_jobs": sum(j["on_time"] for j in jobs1.values()),
            "P4_urgent_ontime_jobs": sum(j["on_time"] for j in jobs4.values()),
            "P5_urgent_ontime_jobs": sum(j["on_time"] for j in jobs5.values()),
            "changed_urgent_job_outcomes": flips_urgent,
            "P4_routine_restricted_flow": p4["metrics"]["routine"]["restricted_mean_flow"],
            "P5_routine_restricted_flow": p5["metrics"]["routine"]["restricted_mean_flow"],
        })

    return {
        "dataset": name,
        "original_ZIP_SHA256": input_sha,
        "scenarios": len(seeds),
        "strict_same_observation_multi_candidate_states": matched,
        "candidate_set_mismatches": mismatched,
        "arms": arms,
        "different_P4_P5_actions": len(flips),
        "action_flips_sensitive_to_resampling": sum(x["v4_resampling_sensitive"] for x in flips),
        "action_flips": flips,
        "scene_business_outcomes": scenes,
    }


def test_known_three_particle_example():
    # 905147: one distinct particle favors candidate 1; other two tie.
    a = {"score": [0, 0, 109.66666666666667, 0, 0, 0],
         "particle_costs": [[0, 110, 0, 0], [0, 109, 0, 0], [0, 110, 0, 0]]}
    b = {"score": [0, 0, 109.33333333333333, 0, 0, 1],
         "particle_costs": [[0, 110, 0, 0], [0, 109, 0, 0], [0, 109, 0, 0]]}
    assert selected([a, b]) == 1
    assert votes({"plans": [a, b], "selected_candidate_index": 1})[0] == 19


def main():
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument("--confirmation-zip", type=Path, required=True)
    pa.add_argument("--later-zip", type=Path, required=True)
    pa.add_argument("--out", type=Path, required=True)
    a = pa.parse_args()
    test_known_three_particle_example()
    result = {"method": "exhaustive 27 shared-index with-replacement bootstrap over THREE archived particle costs",
              "not_27_independent_trials": True,
              "no_new_predictions_or_trajectories": True,
              "not_single_action_causal_attribution": True,
              "groups": {}}
    for label, path in (("confirmation12", a.confirmation_zip),
                        ("later8", a.later_zip)):
        result["groups"][label] = audit(label, *load(path, label))
    result["combined"] = {
        "scenarios": 20,
        "v4_multi_candidate_states": sum(g["arms"]["P5"]["multi_candidate_states"] for g in result["groups"].values()),
        "v4_resampling_sensitive_states": sum(g["arms"]["P5"]["resampling_sensitive"] for g in result["groups"].values()),
        "rule_resampling_sensitive_states": sum(g["arms"]["P4"]["resampling_sensitive"] for g in result["groups"].values()),
        "actual_strict_same_state_action_flips": sum(g["different_P4_P5_actions"] for g in result["groups"].values()),
        "action_flips_sensitive_to_resampling": sum(g["action_flips_sensitive_to_resampling"] for g in result["groups"].values()),
    }
    if a.out.exists():
        raise FileExistsError("Never overwrite previous evidence")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    print(json.dumps(result["combined"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
