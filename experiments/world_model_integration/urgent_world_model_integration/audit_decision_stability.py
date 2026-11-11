"""Read-only frozen V4 decision sensitivity audit for PR #18.

Usage:
 python -m experiments.world_model_integration.urgent_world_model_integration.audit_decision_stability \
   --confirmation /path/github-actions-artifact-13840027102.zip \
   --new-regression /path/github-actions-artifact-13842933799.zip \
   --out /tmp/pr18-decision-stability.json

Never runs the simulator, a model, or a training loop.
"""
from __future__ import annotations
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
from zipfile import ZipFile

SOURCES = {
    "confirmation": ("a5ec5d7a1f886f2d84ba10f3f76decf68d984a26d852f9fd050c7035f8157233", "confirmation/", 12),
    "new_regression": ("0aae5c4e365fab8694b3c239bebe1c3e2c08e2ff617a983f9f672af80afb3dc5", "new-full/", 8),
}


def load(path, cohort):
    expected,prefix,n=SOURCES[cohort]
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        raise AssertionError("artifact sha256 mismatch: "+cohort)
    with ZipFile(path) as z:
        manifest=json.loads(z.read(prefix+"sha256.json"))
        for name, sha in manifest.items():
            if hashlib.sha256(z.read(prefix+name)).hexdigest()!=sha:
                raise AssertionError("inner evidence sha256 mismatch: "+name)
        cases=json.loads(gzip.decompress(z.read(prefix+"cases.json.gz")))
        summary=json.loads(z.read(prefix+"summary.json"))
    if len(cases)!=2*n or len(summary["paired"])!=n:
        raise AssertionError("scenario count differs from frozen protocol")
    return cases,summary


def winning_index(decision, drop=None):
    scores=[]
    for i,p in enumerate(decision["plans"]):
        samples=p["particle_costs"]
        assert len(samples)==3 and all(len(x)==4 for x in samples)
        selected=[x for k,x in enumerate(samples) if k!=drop]
        means=[sum(x[j] for x in selected)/len(selected) for j in range(4)]
        score=(p["score"][0],*means,i)
        if drop is None:
            if any(not math.isclose(x,y,rel_tol=1e-11,abs_tol=1e-10)
                   for x,y in zip(score,p["score"])):
                raise AssertionError("stored particle mean score changed")
        scores.append(score)
    return min(range(len(scores)),key=lambda i:scores[i])


def audit(cohort,cases,summary):
    indexed={(c["seed"],c["arm"]):c for c in cases}
    assert len(indexed)==len(cases)
    counters={a:Counter() for a in ("P4","P5")}
    decisions={}
    for c in cases:
        arm=c["arm"]
        for d in c["decisions"]:
            if len(d["plans"])<2:continue
            chosen=winning_index(d)
            if chosen!=d["selected_candidate_index"]:
                raise AssertionError("full score choice mismatch")
            omit=[winning_index(d,k) for k in range(3)]
            unstable=any(x!=chosen for x in omit)
            counters[arm]["multi_candidate"]+=1
            counters[arm]["LOO_sensitive"]+=unstable
            decisions[(arm,c["seed"],d["tick"],d["observation_sha256"])]={
                "chosen":chosen,"omitted_particle_choices":omit,"unstable":unstable}
    flips=[]
    transitions=Counter()
    for pair in summary["paired"]:
        seed=pair["seed"]
        left=indexed[(seed,"P4")]["metrics"]["jobs"]
        right=indexed[(seed,"P5")]["metrics"]["jobs"]
        ur={j["job_id"]:j for j in left if j["priority_class"]=="urgent"}
        vr={j["job_id"]:j for j in right if j["priority_class"]=="urgent"}
        if ur.keys()!=vr.keys():raise AssertionError("paired urgent job IDs differ")
        for job,j in ur.items():
            transitions[f"{int(j['on_time'])}->{int(vr[job]['on_time'])}"]+=1
        for d in pair["same_state"]["decision_rows"]:
            if d["rule_selected"]==d["v4_selected"]:continue
            key=(seed,d["tick"],d["observation_sha256"])
            r=decisions[("P4",*key)]
            v=decisions[("P5",*key)]
            rule_index=d["rule_rank"][0]
            v4_index=d["v4_rank"][0]
            assert r["chosen"]==rule_index and v["chosen"]==v4_index
            rule_tie=d["rule_scores"][rule_index][:-1]==d["rule_scores"][v4_index][:-1]
            rule_score_v4=d["v4_scores"][rule_index]
            chosen_score_v4=d["v4_scores"][v4_index]
            first=next((i for i in range(5) if chosen_score_v4[i]!=rule_score_v4[i]),None)
            gap=None if first is None else rule_score_v4[first]-chosen_score_v4[first]
            if gap is not None and gap<=0:raise AssertionError("v4 score preference mismatch")
            flips.append({
                "seed":seed,"tick":d["tick"],
                "observation_sha256":d["observation_sha256"],
                "rule_index":rule_index,"v4_index":v4_index,
                "rule_business_tie":rule_tie,
                "v4_omit_one_particle_choices":v["omitted_particle_choices"],
                "v4_omit_one_particle_unstable":v["unstable"],
                "v4_first_differing_component":first,
                "v4_advantage_in_that_component":gap,
                "full_episode_metric_difference":pair["deltas"],
                "causal_warning":"full trajectory association only; no single-action counterfactual"
            })
    return {
        "cohort":cohort,"scenarios":len(summary["paired"]),
        "candidate_decision_diagnostics":{arm:dict(counters[arm]) for arm in counters},
        "actual_matched_state_action_changes":len(flips),
        "Rule_tied_for_both_actions":sum(f["rule_business_tie"] for f in flips),
        "V4_LOO_sensitive_action_changes":sum(f["v4_omit_one_particle_unstable"] for f in flips),
        "urgent_on_time_task_transitions":dict(sorted(transitions.items())),
        "action_changes":flips,
        "source_artifact_sha256":SOURCES[cohort][0]
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--confirmation",type=Path,required=True)
    p.add_argument("--new-regression",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError("derived evidence is write-once")
    results={cohort:audit(cohort,*load(path,cohort)) for cohort,path in [
        ("confirmation",a.confirmation),("new_regression",a.new_regression)]}
    x,y=results["confirmation"],results["new_regression"]
    if (x["actual_matched_state_action_changes"],y["actual_matched_state_action_changes"])!=(6,4):
        raise AssertionError("frozen actual action changes do not match")
    if x["Rule_tied_for_both_actions"]+y["Rule_tied_for_both_actions"]!=9:
        raise AssertionError("rule-tie audit mismatch")
    if x["V4_LOO_sensitive_action_changes"]+y["V4_LOO_sensitive_action_changes"]!=8:
        raise AssertionError("LOO audit mismatch")
    if x["candidate_decision_diagnostics"]["P5"].get("LOO_sensitive")!=17 or y["candidate_decision_diagnostics"]["P5"].get("LOO_sensitive")!=8:
        raise AssertionError("all-state stability audit mismatch")
    for result in (x,y):
        if result["candidate_decision_diagnostics"]["P4"].get("LOO_sensitive",0):
            raise AssertionError("unexpected stochastic rule preference")
    results["summary"]={
        "scenes":20,"real_rule_vs_v4_action_flips":10,
        "flips_with_rule_business_tie":9,
        "flips_sensitive_to_removing_one_of_three_V4_particles":8,
        "V4_multicandidate_decisions":298,"V4_LOO_sensitive_decisions":25,
        "Rule_multicandidate_decisions":295,"Rule_LOO_sensitive_decisions":0,
        "method":"existing particle costs, leave-one-out sensitivity, not independent sampling",
        "limitations":["no query-level logits paired with true events in these archives",
                       "no same-state actual counterfactual for every rejected action",
                       "full-episode delta cannot isolate one action",
                       "synthetic and already-examined scenes"]
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(results,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
    print(json.dumps(results["summary"],ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
