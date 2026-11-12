"""Read-only paired utility, horizon coverage, and CPU-risk audit.

Only parses the immutable, checksum-verified PR29 JSON. No simulator/model
runs, no fitting, no re-ranking, no modifying seeds, scores or thresholds.
Includes descriptive percentile paired bootstrap intervals (with 8 scenario
groups, uncertainty is necessarily substantial).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

ARMS=("rule","v4")
HORIZONS=("5","10","20")


def _mean(xs):
    return sum(xs)/len(xs) if xs else None


def _bootstrap_ci(vals,seed=1112,repeats=4000):
    """Plain paired scenario-level percentile bootstrap, for description only."""
    if not vals:return None
    rng=random.Random(seed)
    n=len(vals)
    means=[sum(vals[rng.randrange(n)] for _ in range(n))/n
           for _ in range(repeats)]
    means.sort()
    return [means[int(0.025*repeats)],means[int(0.975*repeats)]]


def audit(data):
    if data.get("schema")!="RHWM_PR29_FROZEN_HORIZON_SCENE_GENERALIZATION_20261112_V1":
        raise ValueError("wrong PR29 evidence schema")
    groups=data["groups"]
    if len(groups)!=32 or data["summary"]["gpu"]!=0 or data["summary"]["new_training"]!=0:
        raise AssertionError("unexpected input groups or training state")
    out={}
    for regime in sorted({g["regime"] for g in groups}):
        subset=[g for g in groups if g["regime"]==regime]
        if len(subset)!=8:raise AssertionError("frozen group size changed")
        arm_info={}
        for arm in ARMS:
            per_h={}
            for h in HORIZONS:
                deltas=[]
                regret=[]
                coverage=0
                no_truth=0
                budget_overrun=0
                accurate_ranking=0
                ranking_n=0
                rank_wrong=[]
                cpu=[]
                changed=[]
                for group in subset:
                    row=group["predictions"][arm][h]
                    facts=group["factual"]
                    c=row["selected_candidate"]
                    if len(facts)!=2:raise AssertionError("expected two legal alternatives")
                    if not (facts[0]["censored"] or facts[1]["censored"]):
                        delta=facts[c]["makespan_tick"]-facts[0]["makespan_tick"]
                        deltas.append(delta)
                        regret.append(row["makespan_regret"])
                        if facts[0]["makespan_tick"]!=facts[1]["makespan_tick"]:
                            ranking_n+=1
                            good=(facts[c]["makespan_tick"]==
                                  min(x["makespan_tick"] for x in facts))
                            accurate_ranking+=int(good)
                            if not good:rank_wrong.append(group["group"])
                    if c!=0:changed.append(group["group"])
                    budget_overrun+=row["over_fixed_cpu_budget"]
                    cpu.append(row["cpu_seconds_pair"])
                    coverage+=sum(x["truth_at_h_available"] for x in row["candidate_rows"])
                    no_truth+=sum(not x["truth_at_h_available"] for x in row["candidate_rows"])
                per_h[h]={
                    "group_count":len(subset),
                    "selected_non_base_groups":changed,
                    "selected_non_base_count":len(changed),
                    "regret_mean":_mean(regret),
                    "candidate_rank_correct_on_sensitive_groups":accurate_ranking,
                    "candidate_rank_sensitive_groups":ranking_n,
                    "wrong_rank_group_ids":rank_wrong,
                    "delta_makespan_vs_cpsat_mean":_mean(deltas),
                    "delta_makespan_vs_cpsat_paired_percentile_CI95":_bootstrap_ci(
                        deltas,seed=20261112+int(h)+int(arm=="v4")),
                    "wins_vs_cpsat":sum(x<0 for x in deltas),
                    "ties_vs_cpsat":sum(x==0 for x in deltas),
                    "losses_vs_cpsat":sum(x>0 for x in deltas),
                    "mean_pair_forecast_cpu_seconds":_mean(cpu),
                    "max_pair_forecast_cpu_seconds":max(cpu),
                    "cpu_budget_exceeding_groups":budget_overrun,
                    "truth_at_h_candidate_coverage":coverage,
                    "unscored_no_factual_observation_at_h":no_truth,
                }
            arm_info[arm]=per_h
        out[regime]=arm_info
    return {
        "schema":"RHWM_PR29_READONLY_PAIRED_DECISION_AUDIT_V1",
        "source_protocol_sha256":data["protocol_sha256"],
        "source_checkpoint_sha256":data["checkpoint_sha256"],
        "groups":len(groups),
        "by_regime":out,
        "gpu":0,"new_simulations":0,"new_forecasts":0,"training":0,
        "limits":"Percentile CI is descriptive with only eight scenarios per regime. Forecast CPU excludes model construction and GitHub startup; one first action only, no online whole-episode policy trial."
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():raise FileExistsError("write-once read-only audit artifact")
    v=audit(json.loads(args.source.read_text()))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(v,ensure_ascii=False,indent=2)+"\n")
    print("PR29_READONLY_DECISION_AUDIT",json.dumps({
        regime:{
            h:{key:val
               for key,val in arms["v4"][h].items()
               if key in ("selected_non_base_count","wins_vs_cpsat","ties_vs_cpsat",
                          "losses_vs_cpsat","delta_makespan_vs_cpsat_mean",
                          "delta_makespan_vs_cpsat_paired_percentile_CI95",
                          "candidate_rank_correct_on_sensitive_groups",
                          "candidate_rank_sensitive_groups",
                          "truth_at_h_candidate_coverage","cpu_budget_exceeding_groups")}
            for h in HORIZONS}
        for regime,arms in v["by_regime"].items()},ensure_ascii=False))


if __name__=="__main__":
    main()
