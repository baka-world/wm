"""Predeclared CPU prediction diagnostics; ground truth is used ONLY as labels.

One-step forecasts receive the factual current assignment. Multi-step
forecasts use their own predicted state to select later actions. The
simulator's private future-fault schedule is never passed to any model.
These development metrics are not independent confirmation.
"""
from __future__ import annotations

from collections import Counter
import math
import statistics
import time

from business_transport.dynamics import WorkflowDynamics
from business_transport.service_work import ServiceWorkEstimator
from business_transport.workflow_transition import check_business_transition
from experiments.business_constraints.run import make_model
from experiments.service_value.scoring import _prior
from lab_transport.decision_value import nearest_batch, candidates_for


PREDICTION_HORIZONS = (1, 3, 5)
VALIDITY_HORIZONS = (5, 10, 20)
MODELS = ("rule", "pets", "structured")


def _device_event(before, after):
    stations = {s.id: s for s in after.stations}
    return any(getattr(s, "device_available", s.available) !=
               getattr(stations[s.id], "device_available", stations[s.id].available)
               for s in before.stations)


def _external_release(before, after):
    # Public successors of previously visible jobs are not external arrivals.
    jobs = {getattr(s, "job_id", "") for s in before.samples}
    old = {s.id for s in before.samples}
    return any(getattr(s, "job_id", "") not in jobs for s in after.samples if s.id not in old)


def _joint(state, actual):
    robots = {r.id: r for r in state.robots}
    facts = {r.id: r for r in actual.robots}
    samples = {s.id: s for s in state.samples}
    truth = {s.id: s for s in actual.samples}
    stations = {s.id: s for s in state.stations}
    ports = {s.id: s for s in actual.stations}
    robot_errors = sum((robots[k].position, robots[k].phase, robots[k].task_id) !=
                       (r.position, r.phase, r.task_id) for k, r in facts.items()
                       if k in robots) + len(set(facts) ^ set(robots))
    position_errors = sum(robots[k].position != r.position for k, r in facts.items()
                          if k in robots) + len(set(facts) ^ set(robots))
    task_errors = sum((samples[k].status, samples[k].robot_id) !=
                      (s.status, s.robot_id) for k,s in truth.items()
                      if k in samples) + len(set(samples) ^ set(truth))
    resource_errors = sum(
        (stations[k].available, getattr(stations[k], "resource_occupied", None),
         getattr(stations[k], "capacity_blocked", None)) !=
        (station.available, getattr(station, "resource_occupied", None),
         getattr(station, "capacity_blocked", None))
        for k,station in ports.items() if k in stations
    ) + len(set(stations) ^ set(ports))
    return {"robot_relation_errors": robot_errors, "robot_position_errors": position_errors,
            "sample_relation_errors": task_errors, "resource_errors": resource_errors,
            "joint_correct": int(not (robot_errors or task_errors or resource_errors))}


def _events(current, next_state):
    old = {s.id:s for s in current.samples}
    new = {s.id:s for s in next_state.samples}
    return {sid for sid,s in old.items() if sid in new and
            ((s.status == "assigned" and new[sid].status == "in_transit") or
             (s.status == "in_transit" and new[sid].status == "completed"))}


def _factual_remaining_active(rows, index, robot, sample):
    station_id = sample.pickup if robot.phase == "to_pickup" else sample.dropoff
    after_status = "in_transit" if robot.phase == "to_pickup" else "completed"
    ticks = 0
    for row in rows[index:]:
        old = {s.id:s for s in row.observation.samples}
        new = {s.id:s for s in row.next_observation.samples}
        if sample.id not in old or sample.id not in new:
            return None
        status = old[sample.id].status
        if status == after_status:
            return ticks
        if status != sample.status:
            return None  # a different workflow phase or censored session
        station = next(s for s in row.observation.stations if s.id == station_id)
        ticks += int(getattr(station,"device_available",station.available))
        if new[sample.id].status == after_status:
            return ticks
    return None  # right-censored; not silently zero


def evaluate_predictions(rows,business,geometry,network,threshold,seed=1810):
    results = {}
    for name in MODELS:
        per_horizon={}
        for horizon in PREDICTION_HORIZONS:
            counters=Counter();events=Counter()
            sampled=range(0,max(0,len(rows)-horizon+1),5)
            count=0
            for index in sampled:
                if count >= 12:break
                segment=rows[index:index+horizon]
                if any(segment[k+1].observation.tick != segment[k].next_observation.tick
                       for k in range(len(segment)-1)):
                    continue
                first=segment[0].observation
                actual=segment[-1].next_observation
                has_external=any(_device_event(row.observation,row.next_observation) or
                                 _external_release(row.observation,row.next_observation)
                                 for row in segment)
                label="external_windows" if has_external else "normal_windows"
                counters[label]+=1
                count+=1
                model=WorkflowDynamics(make_model(name,geometry,network,threshold,seed+index*37+horizon),
                                       business.rules)
                forecast=first
                try:
                    for step in range(horizon):
                        action=(segment[0].assignments if step==0 else
                                nearest_batch(forecast,candidates_for(forecast),geometry))
                        nxt=model.predict_next(forecast,action)
                        check_business_transition(forecast,nxt,geometry,business.rules,action)
                        forecast=nxt
                    diff=_joint(forecast,actual)
                    counters["legal"]+=1
                    counters.update(diff)
                    old={s.id for s in first.samples}
                    release_truth={s.id for s in actual.samples if s.id not in old}
                    release_pred={s.id for s in forecast.samples if s.id not in old}
                    counters["stage_release_errors"]+=len(release_truth ^ release_pred)
                    if horizon==1:
                        actual_events=_events(first,actual)
                        forecast_events=_events(first,forecast)
                        events["tp"]+=len(actual_events & forecast_events)
                        events["fp"]+=len(forecast_events - actual_events)
                        events["fn"]+=len(actual_events - forecast_events)
                except (ValueError,RuntimeError,TypeError,KeyError,OverflowError) as exc:
                    counters["invalid"]+=1
            precision=(events["tp"]/(events["tp"]+events["fp"])
                       if events["tp"]+events["fp"] else None)
            recall=(events["tp"]/(events["tp"]+events["fn"])
                    if events["tp"]+events["fn"] else None)
            per_horizon[str(horizon)]={
                "attempts": count, "valid": counters["legal"], "invalid": counters["invalid"],
                "normal_windows": counters["normal_windows"],
                "external_windows": counters["external_windows"],
                "joint_exact_windows": counters["joint_correct"],
                "joint_exact_accuracy": counters["joint_correct"]/count if count else None,
                "stage_release_errors": counters["stage_release_errors"],
                "resource_relation_errors": counters["resource_errors"],
                "robot_position_errors": counters["robot_position_errors"],
                "robot_relation_errors": counters["robot_relation_errors"],
                "sample_relation_errors": counters["sample_relation_errors"],
                "service_precision": precision if horizon==1 else None,
                "service_recall": recall if horizon==1 else None,
                "service_tp_fp_fn": dict(events) if horizon==1 else None,
            }
        remaining_errors=[];censored=0;eligible=0
        for index,row in enumerate(rows):
            state=row.observation
            for robot in state.robots:
                if robot.service_elapsed is None or robot.service_active_elapsed is None:
                    continue
                sample=next((s for s in state.samples if s.id==robot.task_id),None)
                if sample is None:continue
                target=sample.pickup if robot.phase=="to_pickup" else sample.dropoff
                station=next((s for s in state.stations if s.id==target),None)
                if station is None or station.position!=robot.position:continue
                eligible+=1
                truth=_factual_remaining_active(rows,index,robot,sample)
                if truth is None:
                    censored+=1;continue
                adapter=ServiceWorkEstimator(make_model(name,geometry,network,threshold,seed+index))
                estimate=adapter.estimate(state,station,robot=robot,sample=sample,phase=robot.phase)
                amount=(estimate.estimated_steps if estimate.usable else
                        max(0.,_prior(station)-robot.service_active_elapsed))
                remaining_errors.append(abs(amount-truth))
        valid={}
        for horizon in VALIDITY_HORIZONS:
            attempted=successful=steps=0;wall=0.;model_calls=0
            for index in range(0,min(len(rows),36),7):
                attempted+=1
                model=WorkflowDynamics(make_model(name,geometry,network,threshold,seed+index+horizon),
                                       business.rules)
                state=rows[index].observation
                start=time.perf_counter()
                try:
                    for step in range(horizon):
                        assignments=(rows[index].assignments if step==0 else
                                     nearest_batch(state,candidates_for(state),geometry))
                        nxt=model.predict_next(state,assignments)
                        check_business_transition(state,nxt,geometry,business.rules,assignments)
                        state=nxt;steps+=1
                    successful+=1
                except (ValueError,RuntimeError,TypeError,KeyError,OverflowError):
                    pass
                wall+=time.perf_counter()-start
                distribution=getattr(getattr(model,"model",None),"distribution",None)
                model_calls+=getattr(distribution,"forward_calls",0)
            valid[str(horizon)]={"attempted":attempted,"valid_full_windows":successful,
                                  "coverage":successful/attempted if attempted else None,
                                  "legal_steps":steps,"wall_seconds":wall,
                                  "pets_distribution_forwards":model_calls}
        results[name]={"state_windows":per_horizon,
                       "remaining_service_absolute_error_mean":
                       statistics.mean(remaining_errors) if remaining_errors else None,
                       "remaining_service_uncensored_evaluated":len(remaining_errors),
                       "remaining_service_right_censored":censored,
                       "remaining_service_known_sessions":eligible,
                       "conditional_horizon_legality":valid}
    return results
