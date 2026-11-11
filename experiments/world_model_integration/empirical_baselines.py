"""Pre-registered at-risk conditional completion hazard baselines.

Fit strictly on TRAIN factual active (not paused / not started) service
transitions. Each row represents a session at risk at t and observed outcome
in t+1. The event hazard P(finish at t+1 | survived e active steps) is NOT a
private countdown or a retrospective eventual-duration estimate.
"""
from __future__ import annotations
from collections import defaultdict
import math

ELAPSED_ENDS=(0,1,2,3,4,5,6,8)
RATIO_ENDS=(0.25,0.5,0.75,1.,1.25)
B1_PRIOR_STRENGTH=4
B2_PRIOR_STRENGTH=6

def _target(row):
    q=row["query"];f=row["features"]["observed_facts"]
    robot=next((r for r in f["robots"] if r["id"]==q["robot_id"]),None)
    station=next((s for s in f["stations"] if s["id"]==q["station_id"]),None)
    if robot is None or station is None:
        raise ValueError("empirical hazard target mismatch")
    elapsed=robot["service_active_elapsed"]
    expected=station["expected_service_ticks_public"]
    if not elapsed["known"] or not expected["known"] or expected["value"] is None or expected["value"]<=0:
        raise ValueError("unknown effective progress or public service prior")
    e=elapsed["value"]; prior=expected["value"]
    if not isinstance(e,int) or e<0:raise ValueError("invalid effective elapsed")
    return e,float(prior)

def elapsed_group(e):
    for upper in ELAPSED_ENDS:
        if e<=upper:return str(upper)
    return "9+"

def ratio_group(e,prior):
    ratio=e/max(prior,1e-8)
    for upper in RATIO_ENDS:
        if ratio<=upper:return str(upper)
    return "gt1.25"

def prior_group(prior):
    return "short<=3" if prior<=3 else "medium<=5" if prior<=5 else "long>5"

class EmpiricalHazards:
    def __init__(self,train):
        if not train:raise ValueError("no factual risk set")
        self.train_positive=sum(row["label"] for row in train)
        self.train_count=len(train)
        self.p0=self.train_positive/self.train_count
        self.e=defaultdict(lambda:[0,0])
        self.er=defaultdict(lambda:[0,0])
        for row in train:
            y=row["label"]
            if y not in (0,1) or row["event"] not in ("continues","finished"):
                raise ValueError("nonfactual or censored row in hazard fit")
            elapsed,expected=_target(row)
            egroup=elapsed_group(elapsed)
            rgroup=ratio_group(elapsed,expected)
            key=(egroup,prior_group(expected),rgroup)
            self.e[egroup][0]+=y;self.e[egroup][1]+=1
            self.er[key][0]+=y;self.er[key][1]+=1
    def predict_one(self,row,method):
        if method=="B0":return self.p0
        e,prior=_target(row);group=elapsed_group(e)
        positive,total=self.e[group]
        p1=(positive+B1_PRIOR_STRENGTH*self.p0)/(total+B1_PRIOR_STRENGTH)
        if method=="B1":return p1
        if method=="B2":
            positive,total=self.er[(group,prior_group(prior),ratio_group(e,prior))]
            return (positive+B2_PRIOR_STRENGTH*p1)/(total+B2_PRIOR_STRENGTH)
        raise ValueError("unknown frozen hazard baseline")
    def predict(self,rows,method):
        return [self.predict_one(row,method) for row in rows]
    def provenance(self):
        return {"fit_split":"train","train_records":self.train_count,
                "train_completions":self.train_positive,"B0_probability":self.p0,
                "B1_prior_strength":B1_PRIOR_STRENGTH,
                "B2_prior_strength":B2_PRIOR_STRENGTH,
                "B1_elapsed_bins":ELAPSED_ENDS,"B2_normalized_bins":RATIO_ENDS,
                "B1_at_risk_bin_count":len(self.e),
                "B2_at_risk_cell_count":len(self.er)}
