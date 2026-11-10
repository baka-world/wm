"""Frozen finite-candidate opportunity summaries. A decision is not an independent sample."""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
import json
import statistics

METHODS = ("Rule-5", "PETS-5", "Structured-5", "Perfect-State-5",
           "Perfect-State-20", "Full-outcome Oracle")
EPS = 1e-9


def outcome_key(branch, index):
    """Completion count first, then restricted flow; deterministic original-index tie."""
    return (-branch["jobs_completed"], branch["restricted_mean_flow"], index)


def select_oracle(branches):
    if not branches or any("error" in b for b in branches):
        raise ValueError("all candidate outcomes must be replayed successfully")
    return min(range(len(branches)), key=lambda i: outcome_key(branches[i], i))


def comparable_regret(branches, chosen, best):
    """Never collapse different completion counts into an arbitrary scalar."""
    complete = all(b["jobs_completed"] == b["jobs"] for b in branches)
    return (branches[chosen]["restricted_mean_flow"] -
            branches[best]["restricted_mean_flow"]) if complete else None


def pairwise_alignment(scores, branches):
    """Non-tied ordinal comparisons only; ties and censored pairs are explicit."""
    total = Counter()
    for i, j in combinations(range(len(branches)), 2):
        if branches[i]["jobs_completed"] != branches[j]["jobs_completed"]:
            actual = (outcome_key(branches[i], i) > outcome_key(branches[j], j)) - (
                outcome_key(branches[i], i) < outcome_key(branches[j], j))
        else:
            x = branches[i]["restricted_mean_flow"] - branches[j]["restricted_mean_flow"]
            actual = (x > EPS) - (x < -EPS)
        if actual == 0:
            total["outcome_ties"] += 1
            continue
        total["informative_pairs"] += 1
        predicted = (scores[i] > scores[j]) - (scores[i] < scores[j])
        if predicted == 0:
            total["score_ties"] += 1
        elif predicted == actual:
            total["concordant"] += 1
        else:
            total["discordant"] += 1
    return dict(total)


def analyze(cases):
    states = []
    for case in cases:
        for state in case["decisions"]:
            row = {"layout": case["layout"], "seed": case["seed"], "regime": case["regime"],
                   "tick": state["tick"], "candidate_count": len(state["plans"])}
            branches = state["branches"]
            if (len(branches) < 2 or any("error" in b for b in branches) or
                any(name not in state.get("methods", {}) or
                    state["methods"][name].get("choice") is None for name in METHODS)):
                row.update(valid=False, reason="insufficient candidates or failed replay",
                           failure_count=sum("error" in b for b in branches))
                states.append(row)
                continue
            best = select_oracle(branches)
            all_done = all(b["jobs_completed"] == b["jobs"] for b in branches)
            values = [b["restricted_mean_flow"] for b in branches]
            gap = max(values) - min(values) if all_done else None
            row.update(valid=True, all_candidates_complete=all_done,
                       oracle_index=best, flow_spread=gap,
                       completion_spread=max(b["jobs_completed"] for b in branches) -
                           min(b["jobs_completed"] for b in branches),
                       gap_1=gap is not None and gap >= 1-EPS,
                       gap_3=gap is not None and gap >= 3-EPS,
                       gap_lt_1=gap is not None and gap < 1-EPS,
                       outcome_tied=all_done and gap <= EPS)
            row["methods"] = {}
            baseline = state["methods"]["Rule-5"]["choice"]
            for name in METHODS:
                m = state["methods"][name]
                selected = m["choice"]
                score = m.get("scores")
                delta = branches[selected]["restricted_mean_flow"] - branches[baseline]["restricted_mean_flow"]
                compare = outcome_key(branches[selected], selected)[:2]
                reference = outcome_key(branches[baseline], baseline)[:2]
                # A value difference within floating tolerance is a substantive tie.
                if branches[selected]["jobs_completed"] != branches[baseline]["jobs_completed"]:
                    relative = "better" if branches[selected]["jobs_completed"] > branches[baseline]["jobs_completed"] else "worse"
                else:
                    relative = "better" if delta < -EPS else ("worse" if delta > EPS else "equal")
                row["methods"][name] = {
                    "choice": selected,
                    "oracle_hit": compare == outcome_key(branches[best], best)[:2] or
                        (branches[selected]["jobs_completed"] == branches[best]["jobs_completed"] and
                         abs(branches[selected]["restricted_mean_flow"] -
                             branches[best]["restricted_mean_flow"]) <= EPS),
                    "regret_steps": comparable_regret(branches, selected, best),
                    "completion_deficit": branches[best]["jobs_completed"]-branches[selected]["jobs_completed"],
                    "restricted_flow_delta_vs_oracle": branches[selected]["restricted_mean_flow"] -
                        branches[best]["restricted_mean_flow"],
                    "same_choice_as_rule": selected == baseline,
                    "relative_to_rule": relative,
                    "ranking": pairwise_alignment(score, branches) if score else None,
                }
            states.append(row)
    valid = [s for s in states if s.get("valid")]
    clusters = defaultdict(list)
    for state in valid:
        clusters[(state["layout"], state["seed"])].append(state)

    def cluster_mean(accessor):
        per = []
        for key, rows in sorted(clusters.items()):
            values = [accessor(row) for row in rows]
            values = [x for x in values if x is not None]
            if values:
                per.append({"layout":key[0], "seed":key[1], "n":len(values),
                            "mean":statistics.mean(values)})
        return {"cluster_means":per, "mean_of_cluster_means":
                statistics.mean(x["mean"] for x in per) if per else None,
                "n_clusters":len(per)}

    method_summary = {}
    for name in METHODS:
        rows = [r["methods"][name] for r in valid]
        counts = Counter(m["relative_to_rule"] for m in rows if not m["same_choice_as_rule"])
        method_summary[name] = {
            "regret":cluster_mean(lambda row: row["methods"][name]["regret_steps"]),
            "restricted_flow_delta_vs_oracle":cluster_mean(
                lambda row: row["methods"][name]["restricted_flow_delta_vs_oracle"]),
            "completion_deficit":cluster_mean(lambda row: row["methods"][name]["completion_deficit"]),
            "oracle_best_hits":sum(m["oracle_hit"] for m in rows),
            "same_choice_as_rule":sum(m["same_choice_as_rule"] for m in rows),
            "different_choice":sum(not m["same_choice_as_rule"] for m in rows),
            "different_better":counts["better"],
            "different_worse":counts["worse"],
            "different_equal":counts["equal"],
            "regret_undefined_unfinished":sum(m["regret_steps"] is None for m in rows),
            "ranking_pairs":dict(sum((Counter(m["ranking"]) for m in rows if m["ranking"]), Counter()))
        }
    gap_summary = {
        "identical_complete_outcomes":sum(s["outcome_tied"] for s in valid),
        "gap_lt_1":sum(s["gap_lt_1"] for s in valid),
        "gap_ge_1":sum(s["gap_1"] for s in valid),
        "gap_ge_3":sum(s["gap_3"] for s in valid),
        "mixed_completion_candidates":sum(not s["all_candidates_complete"] for s in valid),
        "cluster_spread":cluster_mean(lambda s:s["flow_spread"]),
        "largest_complete_gap":max((s["flow_spread"] for s in valid
                                    if s["flow_spread"] is not None), default=None)
    }
    rule_opportunity = method_summary["Rule-5"]["regret"]["mean_of_cluster_means"]
    perfect5 = method_summary["Perfect-State-5"]["regret"]["mean_of_cluster_means"]
    perfect20 = method_summary["Perfect-State-20"]["regret"]["mean_of_cluster_means"]
    # Diagnostic labels are hypotheses, not automatically promoted to causal claims.
    if not valid:
        decision = "E: insufficient successful replay evidence"
    elif rule_opportunity is None:
        decision = "E: mixed completion outcomes preclude a universal step-regret"
    elif gap_summary["gap_lt_1"] > len(valid)/2:
        decision = "A-candidate: majority of fixed-candidate gaps are below one step"
    elif (perfect5 is not None and perfect20 is not None and
          perfect20 < perfect5-EPS and perfect5 < rule_opportunity-EPS):
        decision = ("E: multiple bottlenecks; 20-step true states outperform "
                    "5-step true states, and 5-step truth also beats the predictors")
    elif (perfect5 is not None and perfect20 is not None and
          perfect5 >= rule_opportunity-EPS and perfect20 < rule_opportunity-EPS):
        decision = ("C: short-window information or the workflow score limits decisions; "
                    "20-step truth is substantially more useful")
    elif (perfect5 is not None and perfect20 is not None and
          perfect5 >= rule_opportunity-EPS and perfect20 >= rule_opportunity-EPS):
        decision = "C-candidate: even perfect 20-step states do not reduce regret; inspect scoring"
    elif perfect5 is not None and perfect5 < rule_opportunity-EPS:
        decision = "B-candidate: 5-step truth reduces regret; inspect prediction disagreements"
    else:
        decision = "E: overlapping, unidentifiable or insufficient bottlenecks"
    return {
        "sampling_unit": "map/seed cluster; decisions nested within scenario",
        "scenarios":len(cases),
        "clusters":len({(c["layout"],c["seed"]) for c in cases}),
        "actual_decisions":len(states),
        "valid_decisions":len(valid),
        "candidate_total":sum(r["candidate_count"] for r in states),
        "failed_decisions":len(states)-len(valid),
        "candidate_gaps":gap_summary,
        "finite_candidate_rule_opportunity":method_summary["Rule-5"]["regret"],
        "methods":method_summary,
        "diagnostic_not_final_causal_judgement":decision,
        "states":states
    }


def write_analysis(raw_path, target):
    from pathlib import Path
    import gzip
    p = Path(raw_path)
    op = gzip.open if p.suffix == ".gz" else open
    with op(p, "rt", encoding="utf-8") as stream:
        raw = json.load(stream)
    result=analyze(raw["cases"])
    Path(target).write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n",
                            encoding="utf-8")
    return result


if __name__ == "__main__":
    import sys
    write_analysis(sys.argv[1],sys.argv[2])
