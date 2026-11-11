"""Read-only explanation of all actual strict same-observation V4 vs Rule action flips.

Uses the original full-prediction frozen confirmation artifact. Never
replays, simulates or assumes that entire-scene outcomes are caused by one
particular flipped decision.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def flips(summary):
    rows=[]
    for group in summary["paired"]:
        for item in group["same_state"]["decision_rows"]:
            if item["rule_selected"]==item["v4_selected"]:
                continue
            s4=item["rule_scores"];s5=item["v4_scores"]
            a=min(range(len(s4)),key=lambda i:s4[i])
            b=min(range(len(s5)),key=lambda i:s5[i])
            if item["rule_selected"]!=item["v4_selected"]:
                rows.append({
                    "seed":group["seed"],"tick":item["tick"],
                    "state_sha256":item["observation_sha256"],
                    "rule_candidate_index":a,
                    "v4_candidate_index":b,
                    "rule_action":item["rule_selected"],
                    "v4_action":item["v4_selected"],
                    "rule_scores_full":s4,
                    "v4_scores_full":s5,
                    "rule_chosen_score":s4[a],
                    "v4_chosen_score":s5[b],
                    "shared_first_component":s4[a][0]==s5[b][0],
                    "full_episode_paired_delta_V4_minus_Rule":group["deltas"],
                    "causal_scope":"full_trajectory_association_only_not_single_action_effect"
                })
    rows.sort(key=lambda x:(x["seed"],x["tick"]))
    return rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--summary",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    data=json.loads(args.summary.read_text())
    x=flips(data)
    if len(x)!=6:
        raise AssertionError(f"FROZEN_BASELINE_ACTION_FLIPS_CHANGED_{len(x)}")
    if args.out.exists():raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps({
        "source_run_id":42986822400,
        "source_artifact_id":13840027102,
        "scope":"6 strict same-state P4/P5 action changes in previously examined synthetic confirmation",
        "count":len(x),"changes":x
    },indent=2,ensure_ascii=False)+"\n")
    for a in x:
        print("FLIP",a["seed"],a["tick"],
              "Rule",a["rule_candidate_index"],
              "V4",a["v4_candidate_index"],
              "delta",a["full_episode_paired_delta_V4_minus_Rule"],flush=True)


if __name__=="__main__":
    main()
