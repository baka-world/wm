"""Verified summary of frozen, executed PR18 V4 decision-sensitivity replay.

Offline postprocessing ONLY: no model calls, simulator, optimization or training.
Full raw per-step trace is separately archived as deterministic replay.json.gz.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
from pathlib import Path

EXPECTED_V4 = "60c8a6c3d832da3e6519072688ea099f64ae120ba19d3bd2ec705f7f1a11ac0c"


def summarize(source):
    assert source["checkpoint_sha256"] == EXPECTED_V4
    assert [x["seed"] for x in source["cases"]] == [43011, 43015, 43019]
    result = {"evidence_scope": "development_deterministic_cpu_replay_not_independent_confirmation",
              "checkpoint_sha256": EXPECTED_V4,
              "original_5step_parity_all": True, "cases": []}
    for case in source["cases"]:
        assert set(case["windows"]) == {"5", "10", "20"}
        entry = {"seed": case["seed"], "tick": 0, "windows": {}}
        for horizon in ("5", "10", "20"):
            data = case["windows"][horizon]
            if horizon == "5":
                assert data["recorded_archive_5step_parity"] == {"Rule": True, "V4": True}
            a, b = data["runs"]["Rule"], data["runs"]["V4"]
            counts = data["comparison"]["counts"]
            candidates = []
            first_event = None
            first_state = None
            first_release = None
            captures = 0
            expected_batches = 0
            queries = 0
            for ci, (rc, vc) in enumerate(zip(a["records"], b["records"])):
                keys = rc["component_means"].keys() & vc["component_means"].keys()
                candidates.append({
                    "candidate": ci, "rule_score": rc["score"], "v4_score": vc["score"],
                    "score_delta": [round(v-r, 8) for v,r in zip(vc["score"],rc["score"])],
                    "rule_components": rc["component_means"],
                    "v4_components": vc["component_means"],
                    "component_delta": {k:round(vc["component_means"][k]-rc["component_means"][k],8) for k in sorted(keys)},
                    "particle_costs_rule": rc["particle_business_costs"],
                    "particle_costs_v4": vc["particle_business_costs"]})
                for pi, (rt, vt) in enumerate(zip(a["traces"][ci], b["traces"][ci])):
                    rng = random.Random(1810+pi)  # frozen tick=0 seed, reset per candidate
                    for si, (rs, vs) in enumerate(zip(rt,vt), 1):
                        assert rs["after"]["tick"] == rs["before"]["tick"]+1
                        assert vs["after"]["tick"] == vs["before"]["tick"]+1
                        if si < len(rt):
                            assert rs["after"] == rt[si]["before"]
                            assert vs["after"] == vt[si]["before"]
                        q = vs["session"]
                        assert len(q["queries"]) == q["queries_eligible"]
                        queries += q["queries_eligible"]
                        expected_batches += int(q["queries_eligible"] > 0)
                        same_input = rs["before"] == vs["before"]
                        if rs["events"]["released_stages"] != vs["events"]["released_stages"]:
                            if first_release is None or si < first_release["step"]:
                                first_release={"candidate":ci,"particle":pi,"step":si,
                                               "rule":rs["events"]["released_stages"],
                                               "v4":vs["events"]["released_stages"]}
                        if any(rs["after"][part] != vs["after"][part] for part in ("robots","samples","stations")):
                            if first_state is None or si < first_state["step"]:
                                first_state={"candidate":ci,"particle":pi,"step":si,
                                             "same_input_before_step":same_input,
                                             "robot_change":rs["after"]["robots"]!=vs["after"]["robots"],
                                             "sample_change":rs["after"]["samples"]!=vs["after"]["samples"],
                                             "station_change":rs["after"]["stations"]!=vs["after"]["stations"]}
                        for query in q["queries"]:
                            captures += 1
                            assert abs(query["u"]-rng.random()) < 1e-15
                            assert query["sampled_finish"] == (query["u"] < query["p_v4"])
                            assert query["v4_transition_finished"] == int(query["sampled_finish"])
                            if query["event_diff_vs_rule_on_same_input"]:
                                if first_event is None or si < first_event["step"]:
                                    first_event={"candidate":ci,"particle":pi,"step":si,
                                      "same_input_as_recursive_rule":same_input, **query}
            assert captures == queries == b["actual_model_query_calls"]
            assert expected_batches == b["actual_model_forward_batches"]
            assert queries == counts["service_queries_eligible"]
            rank_rule = data["comparison"]["rule_rank"]
            rank_v4 = data["comparison"]["v4_rank"]
            if queries == 0: category="F1"
            elif counts["sampled_event_diff_vs_rule_on_identical_v4_state"] == 0: category="F2"
            elif counts["candidate_score_tuple_changed"] == 0: category="F3"
            elif not counts["ranking_changed"]: category="F4"
            else: category="RANK_CHANGED_REQUIRES_FACTUAL_EVALUATION"
            def gap(rank, key):
                return (candidates[rank[1]][key][0]-candidates[rank[0]][key][0]) if len(rank)>1 else None
            entry["windows"][horizon] = {
                "classification": category,
                "archive_5step_parity": data["recorded_archive_5step_parity"],
                "counts": counts,
                "v4_forward_batches": b["actual_model_forward_batches"],
                "v4_query_calls": queries,
                "v4_fallback_reasons": b["fallback_reasons"],
                "cpu_seconds_instrumented": {"Rule":a["cpu_seconds"],"V4":b["cpu_seconds"]},
                "first_same_input_event_difference": first_event,
                "first_recursive_state_difference": first_state,
                "first_successor_release_difference": first_release,
                "ranks": {"Rule":rank_rule,"V4":rank_v4},
                "primary_cost_gap": {"Rule":gap(rank_rule,"rule_score"),"V4":gap(rank_v4,"v4_score")},
                "candidates":candidates,
                "instrumentation_checks": {
                    "forward_batches_one_per_nonempty_service_batch":True,
                    "original_rng_stream_matches_each_sample":True,
                    "captured_event_matches_transition":True,
                    "adjacent_state_snapshots_consistent":True}}
        result["cases"].append(entry)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--replay",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError("immutable evidence")
    original=args.replay.read_bytes()
    report=summarize(json.loads(original))
    report["raw_replay_sha256"]=hashlib.sha256(original).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"parity":report["original_5step_parity_all"],"cases":[
      {"seed":c["seed"],"classes":{h:w["classification"] for h,w in c["windows"].items()}}
      for c in report["cases"]]}))

if __name__=="__main__": main()
