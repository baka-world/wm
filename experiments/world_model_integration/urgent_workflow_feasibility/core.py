"""Isolated synthetic urgent + conditional retest business feasibility.

Only the simulator owns exogenous retest outcomes. The online dispatch receives
only released samples via BusinessObserver; no private result or future outage
may be passed to the policy. All shared production modules remain unchanged.
"""
from __future__ import annotations
from dataclasses import asdict,replace
from collections import Counter
import hashlib
import json
import math
import time
from pathlib import Path

from lab_transport.domain import Grid,Station,RobotSpec,Scenario
from lab_transport.dynamics_model import Geometry
from lab_transport.simulation import SampleState
from lab_transport.transition_schema import Assignment
from lab_transport.paper_reuse.ortools_dispatch import cp_model
from business_transport.scenario import compile_scenario,stage_id
from business_transport.simulation import BusinessSimulator,resource_audit
from business_transport.observation import BusinessObserver
from business_transport.dispatch import BusinessCPSAT
from lab_transport.decision_value import candidates_for
from lab_transport.multisite_experiment import discrete_safety_violations
from experiments.business_constraints.run import lifecycle_audit

ROOT=Path(__file__).resolve().parent
PROTOCOL=ROOT/"protocol.json"
STATIONS=("A","B","C","D")
BASE_ROUTE=(("A","B"),("B","C"),("C","D"),("D","A"))
BRANCH_ROUTE=(("A","B"),("B","C"),("C","D"),("D","C"),("C","A"))


def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,
                                      separators=(",",":")).encode("utf-8")).hexdigest()


def public_spec(index,seed,max_ticks):
    """Factorized, deterministic and result-blind synthetic design."""
    jobs=4+index%3
    urgent_count=(1,2,3)[index%3]
    rate=(0,35,65)[(index//3)%3]
    robots=1+(index//2)%2
    cap=1+(index//9)%2
    outage=(index%4 in (2,3))
    return {"seed":seed,"index":index,"jobs":jobs,"urgent_count":min(jobs,urgent_count),
            "retest_rate_percent":rate,"robot_count":robots,"capacity":cap,
            "outage":outage,"max_ticks":max_ticks,
            "jitter":1+(index%2),"urgent_due_offset":65+(index%3)*5,
            "routine_due_offset":145,
            "source":"SYNTHETIC_ASSUMPTION_NOT_COMPANY"}


def scene_for(spec):
    if cp_model is None:
        raise RuntimeError("OR_TOOLS_REQUIRED_FOR_FAIR_BUSINESS_CPSAT")
    seed=spec["seed"];i=spec["index"]
    station_positions={"A":(1,1),"B":(4,1),"C":(7,1),"D":(7,5)}
    services={"A":2+(i%2),"B":3+(i%2),"C":3+(i%3),"D":2+(i%2)}
    stations={}
    for k,pos in station_positions.items():
        windows=((19,28),) if k=="C" and spec["outage"] else ()
        stations[k]=Station(k,pos,services[k],float(services[k]),windows)
    robots=tuple(RobotSpec("R"+str(j+1),(1+j,1),(1+j,1))
                 for j in range(spec["robot_count"]))
    base=Scenario("synthetic-urgent-"+str(seed),Grid(11,9,frozenset()),
                  stations,robots,(),spec["jitter"],spec["max_ticks"],
                  metadata={"business":"synthetic mechanism, NOT measured hospital TAT",
                            "source_seed":seed})
    jobs=[];workflows={}
    release_sequence=(0,0,2,5,8,11)
    for j in range(spec["jobs"]):
        jid="U"+str(seed)+"-"+str(j)
        urgent=j<spec["urgent_count"]
        release=release_sequence[j]
        due=release+(spec["urgent_due_offset"] if urgent else spec["routine_due_offset"])
        name="wf_"+jid
        workflows[name]=[{"pickup":x,"dropoff":y} for x,y in BASE_ROUTE]
        jobs.append({"id":jid,"workflow":name,"release_tick":release,
                     "deadline_tick":due,"rule_profile":"urgent" if urgent else "routine",
                     "specimen_type":"synthetic-tube"})
    rules={"resources":[{"id":"POOL_CORE","stations":["B","C"],
                          "capacity":spec["capacity"]},
                         {"id":"POOL_OUTER","stations":["A","D"],"capacity":spec["capacity"]}],
           "workflows":workflows,"jobs":jobs,"max_ticks":spec["max_ticks"],
           "name":"synthetic priority/retest "+str(seed)}
    business=compile_scenario(base,rules)
    return business,Geometry(base.grid)


def synthetic_retest(seed,job_id,rate):
    """Private keyed external event; never draw by policy/tick/invocation order."""
    if not 0<=rate<=100:raise ValueError("invalid retest rate")
    data=("urgent-retest-v1|"+str(seed)+"|"+job_id).encode()
    value=int.from_bytes(hashlib.sha256(data).digest()[:8],"big")%100
    return value<rate


class ConditionalRetestSimulator(BusinessSimulator):
    """Insert a *new* active stage only after detecting result at stage3 finish.

    Stages 1-3 and normal archive stage4 are public. Until a result is
    generated stage4 is unreleased and an optional retest stage does NOT exist.
    On retest, the private unreleased archive is replaced with a newly built
    stage4 retest and new stage5 archive. There is no predicted retest in
    observation, nor a new clinical job counted for a sample.
    """
    def __init__(self,business,seed,retest_rate_percent,**kwargs):
        self.retest_rate_percent=retest_rate_percent
        self.retest_results={}
        super().__init__(business,"world",seed,**kwargs)

    def _release_after(self,prev,successor,new_tick):
        target=self.samples[successor]
        if target.status!="unreleased":
            raise AssertionError("successor already active/released")
        target.spec=replace(target.spec,release_tick=new_tick)
        self._event(new_tick,"stage_ready",sample_id=successor,predecessor=prev)

    def _branch(self,job_id,finished_id,new_tick):
        if job_id in self.retest_results:raise AssertionError("duplicate retest decision")
        yes=synthetic_retest(self.seed,job_id,self.retest_rate_percent)
        self.retest_results[job_id]=yes
        self._event(new_tick,"retest_decision",job_id=job_id,needs_retest=yes,
                    released_after_detection=True)
        old=self.business
        archive=old.stage_next[finished_id]
        if not yes:
            self._release_after(finished_id,archive,new_tick)
            return
        if archive!=stage_id(job_id,3) or self.samples[archive].status!="unreleased":
            raise AssertionError("retest cannot replace an active archive stage")
        old_archive=self.samples.pop(archive).spec
        retest_spec=replace(old_archive,id=archive,pickup="D",dropoff="C",
                            release_tick=new_tick)
        final=stage_id(job_id,4)
        if final in self.samples:raise AssertionError("retest duplicate final stage")
        archive_spec=replace(old_archive,id=final,pickup="C",dropoff="A",
                             release_tick=self.scenario.max_ticks+1)
        self.samples[archive]=SampleState(retest_spec)
        self.samples[final]=SampleState(archive_spec)
        flow=old.stage_metadata[finished_id][0]
        if old.rules.workflows[flow]!=BASE_ROUTE:
            raise AssertionError("branch expected original public workflow")
        workflows=dict(old.rules.workflows);workflows[flow]=BRANCH_ROUTE
        metadata=dict(old.stage_metadata)
        metadata[archive]=(flow,3,job_id)
        metadata[final]=(flow,4,job_id)
        stage_next=dict(old.stage_next)
        stage_next[finished_id]=archive
        stage_next[archive]=final
        jobs=tuple({**j,"stages":(*j["stages"],final)} if j["id"]==job_id else j
                   for j in old.jobs)
        self.business=replace(old,rules=replace(old.rules,workflows=workflows),
                              stage_metadata=metadata,stage_next=stage_next,jobs=jobs)
        self.scenario=replace(self.scenario,samples=tuple(s.spec for s in self.samples.values()))
        self._event(new_tick,"retest_stage_created",job_id=job_id,
                    sample_id=archive,final_stage_id=final)
        self._release_after(finished_id,archive,new_tick)

    def _handle_station(self,tick,robot):
        sid=robot.task_id
        if sid is None or robot.phase!="to_dropoff":
            return super()._handle_station(tick,robot)
        meta=self.business.stage_metadata.get(sid)
        if meta is None or meta[1]!=2:
            return super()._handle_station(tick,robot)
        if sid not in self.business.stage_next:
            raise AssertionError("no archive after detection")
        previous=self.business
        # Stop base class from automatically releasing its preconfigured
        # archive before the detection result has been generated.
        self.business=replace(previous,stage_next={k:v for k,v in previous.stage_next.items()
                                                   if k!=sid})
        before=self.samples[sid].status
        try:
            done=super()._handle_station(tick,robot)
        finally:
            self.business=previous
        if before!="completed" and self.samples[sid].status=="completed":
            self._branch(meta[2],sid,tick+1)
        return done


class PriorityObserver:
    """Public priority derives exclusively from current released sample."""
    def __init__(self):
        self.base=BusinessObserver()

    def observe(self,raw):
        rich=self.base.observe(raw)
        return replace(rich,samples=tuple(
            replace(s,priority=2 if s.rule_profile=="urgent" else 1)
            for s in rich.samples))


def unique_plans(plans,legal):
    allowed=set(legal);seen=set();out=[]
    for plan in plans:
        ids=tuple(sorted((a.robot_id,a.sample_id) for a in plan))
        if not ids or ids in seen:continue
        if not set(plan)<=allowed:raise AssertionError("illegal candidate plan")
        if len({x.robot_id for x in plan})!=len(plan) or len({x.sample_id for x in plan})!=len(plan):
            raise AssertionError("assignment duplicates")
        seen.add(ids);out.append(tuple(plan))
    return out


def matchkey(plan):
    return tuple(sorted((a.robot_id,a.sample_id) for a in plan))


def route_time(observation,matching,rules,geometry,jitter_mean=0):
    """Observable nominal ETA, no hidden future retest or outage schedule."""
    robots={r.id:r for r in observation.robots}
    samples={s.id:s for s in observation.samples}
    stations={x.id:x for x in observation.stations}
    end={}
    for action in matching:
        r=robots[action.robot_id];sample=samples[action.sample_id]
        route=rules.workflows[sample.workflow][sample.stage_index:]
        pos=r.position;total=observation.tick
        for pickup,dropoff in route:
            dst=stations[pickup];other=stations[dropoff]
            total+=geometry.distance(pos,dst.position)+geometry.distance(dst.position,other.position)
            total+=max(1,math.ceil(dst.expected_service_ticks))+jitter_mean
            total+=max(1,math.ceil(other.expected_service_ticks))+jitter_mean
            pos=other.position
        end[sample.job_id]=total
    return end


def cdf_lateness_for(eta_nominal,deadline,service_count,jitter):
    """Exact convolution of independent public uniform discrete service jitters.

    Only planning proxy: ignores queues and unknown outages; NOT a calibrated
    full business completion CDF and not the simulator's hidden SHA result.
    """
    dist={0:1.}
    for _ in range(service_count):
        nxt={}
        for value,p in dist.items():
            for noise in range(jitter+1):
                nxt[value+noise]=nxt.get(value+noise,0.)+p/(jitter+1)
        dist=nxt
    expected_late=sum(prob*max(0,eta_nominal+noise-deadline)
                      for noise,prob in dist.items())
    return expected_late


class IsolatedPolicy:
    def __init__(self,name,geometry,config,public_jitter):
        if name not in ("P0","P1","P2","P3"):raise ValueError("bad policy")
        self.name=name;self.geometry=geometry;self.config=config
        self.public_jitter=public_jitter
        self.observer=PriorityObserver()
        self.rules=None
        self.solver=None
        self.records=[]
        self.cpu_seconds=0.
        self.forced={}
        self.factual_prefix=None
        self.target=None

    def score(self,obs,plan,index):
        selected={a.sample_id for a in plan}
        waiting=[s for s in obs.samples if s.status=="waiting"]
        urgent=[s for s in waiting if s.priority==2]
        missed=sum(s.id not in selected for s in urgent)
        robots={r.id:r for r in obs.robots}
        samples={s.id:s for s in obs.samples}
        stations={s.id:s for s in obs.stations}
        distance=sum(self.geometry.distance(robots[a.robot_id].position,
                                            stations[samples[a.sample_id].pickup].position)
                     for a in plan)
        if self.name=="P1":
            urgency_due=sum(samples[a.sample_id].deadline_tick-obs.tick
                            for a in plan if samples[a.sample_id].priority==2)
            return (missed,urgency_due,distance,index)
        eta=route_time(obs,plan,self.rules,self.geometry,
                       self.public_jitter/2 if self.name=="P3" else 0)
        urgent_cost=0.;routine_cost=0.
        for a in plan:
            s=samples[a.sample_id]
            e=eta[s.job_id]
            if self.name=="P3":
                route=self.rules.workflows[s.workflow][s.stage_index:]
                nominal=e-len(route)*self.public_jitter  # 2 operations per stage * mean jitter
                # Prior ETA includes jitter mean per operation; remove it here.
                nominal=e-len(route)*self.public_jitter
                risk=cdf_lateness_for(nominal,s.deadline_tick,2*len(route),self.public_jitter)
            else:
                risk=max(0,e-s.deadline_tick)
            if s.priority==2:urgent_cost+=risk
            else:routine_cost+=risk
        return (missed,urgent_cost,routine_cost,distance,index)

    def __call__(self,raw,candidates,sim):
        t=time.process_time()
        obs=self.observer.observe(raw)
        self.rules=sim.business.rules
        if self.solver is None:
            self.solver=BusinessCPSAT(self.geometry,self.rules,
                candidate_limit=self.config["max_legal_candidates"],
                time_limit_seconds=self.config["cpsat_seconds"],
                deterministic_time_limit=self.config["deterministic_cpsat_seconds"],seed=1810)
        self.solver.rules=self.rules
        proposals=self.solver.proposals(obs,candidates)
        plans=unique_plans(proposals,candidates)
        records=[{"matching":[{"robot_id":a.robot_id,"sample_id":a.sample_id} for a in p],
                  "score":self.score(obs,p,i) if self.name!="P0" else None}
                 for i,p in enumerate(plans)]
        if self.target is not None and obs.tick<self.target:
            selected=self.factual_prefix.get(obs.tick,())
            choice=None
        elif self.target is not None and obs.tick==self.target:
            selected=tuple(self.forced[self.target]);choice=None
        else:
            choice=(0 if self.name=="P0" else
                    min(range(len(plans)),key=lambda i:self.score(obs,plans[i],i))) if plans else None
            selected=plans[choice] if choice is not None else ()
        if any(a not in candidates for a in selected):
            raise AssertionError("injected action outside observed legal candidates")
        self.records.append({
            "tick":obs.tick,
            "urgent_waiting":sum(s.status=="waiting" and s.priority==2 for s in obs.samples),
            "routine_waiting":sum(s.status=="waiting" and s.priority==1 for s in obs.samples),
            "active_service":sum(r.service_elapsed is not None for r in obs.robots),
            "idle_robots":sum(r.task_id is None for r in obs.robots),
            "outage_visible":sum(not st.device_available for st in obs.stations),
            "capacity_blocked":sum(st.capacity_blocked for st in obs.stations),
            "priorities":{s.id:s.priority for s in obs.samples},
            "plans":records,"candidate_count":len(plans),
            "selected":[asdict(a) for a in selected],
            "retest_visible":sum(s.stage_index==3 and s.status=="waiting" and
                        self.rules.workflows[s.workflow]==BRANCH_ROUTE for s in obs.samples),
            "solver":self.solver.log[-1],
        })
        self.cpu_seconds+=time.process_time()-t
        return tuple(selected)


def job_metrics(result,business,max_ticks):
    by_id={x["id"]:x for x in result["samples"]}
    jobs=[];groups={"urgent":[],"routine":[]}
    for job in business.jobs:
        rows=[by_id[sid] for sid in job["stages"]]
        final=rows[-1]["completed_tick"]
        due=job["deadline_tick"];release=job["release_tick"]
        priority=rows[0]["rule_profile"]
        if priority not in groups:raise AssertionError("priority dropped")
        truncated=max_ticks-release if final is None else final-release
        late=max(0,(max_ticks if final is None else final)-due)
        record={"job_id":job["id"],"priority_class":priority,
           "release_tick":release,"deadline_tick":due,
           "final_tick":final,"completed":final is not None,
           "censored":final is None,
           "restricted_flow":truncated,
           "late_ticks_or_censored_lower_bound":late,
           "on_time":final is not None and final<=due,
           "stages":[{"id":sid,"status":by_id[sid]["status"],
                      "completed_tick":by_id[sid]["completed_tick"]} for sid in job["stages"]],
           "retest":len(job["stages"])==5}
        jobs.append(record);groups[priority].append(record)
    def aggregate(rs):
        if not rs:return {"n":0,"on_time_rate":None,"late_mean_lower_bound":None,
                          "incomplete_rate":None,"restricted_mean_flow":None}
        return {"n":len(rs),"on_time_rate":sum(x["on_time"] for x in rs)/len(rs),
            "late_mean_lower_bound":sum(x["late_ticks_or_censored_lower_bound"] for x in rs)/len(rs),
            "incomplete_rate":sum(x["censored"] for x in rs)/len(rs),
            "restricted_mean_flow":sum(x["restricted_flow"] for x in rs)/len(rs),
            "overdue_unfinished":sum(x["censored"] and x["deadline_tick"]<max_ticks for x in rs)}
    return {"jobs":jobs,"urgent":aggregate(groups["urgent"]),
            "routine":aggregate(groups["routine"]),"all":aggregate(jobs),
            "uncompleted_count":sum(x["censored"] for x in jobs),
            "job_count":len(jobs),
            "retest_triggered":sum(x["retest"] for x in jobs)}


def judge_score(metrics):
    urg=metrics["urgent"];reg=metrics["routine"]
    # Lexicographic finite-time development oracle objective, group job sizes fixed.
    return (urg["incomplete_rate"],urg["late_mean_lower_bound"],
            reg["restricted_mean_flow"],metrics["all"]["restricted_mean_flow"])


def run_one(spec,protocol,arm,forced=None,prefix=None,target=None,collect_obs=False):
    business,geo=scene_for(spec)
    sim=ConditionalRetestSimulator(business,spec["seed"],spec["retest_rate_percent"],
                                   collect_metrics=True)
    strategy=IsolatedPolicy(arm,geo,protocol,spec["jitter"])
    strategy.target=target
    strategy.factual_prefix=prefix
    if forced is not None and target is not None:
        strategy.forced={target:tuple(forced)}
    captured=[]
    def on_observation(obs):
        # Continuous public history is needed for valid active service progress.
        strategy.observer.observe(obs)
        if collect_obs:
            captured.append({"tick":obs.tick,"sha":digest(asdict(obs))})
    def dispatch(o,c):
        return strategy(o,c,sim)
    start=time.process_time()
    result=sim.run(dispatch_policy=dispatch,on_observation=on_observation)
    cpu=time.process_time()-start
    capacity=resource_audit(result,sim.business.rules)
    motion=discrete_safety_violations(result)
    lifecycle=lifecycle_audit(result["history"])
    if capacity["violations"] or motion or lifecycle:
        raise AssertionError("CAPACITY_MOTION_LIFECYCLE_INTEGRITY_FAILURE")
    if len(sim.retest_results)!=sum(e["type"]=="retest_decision" for e in result["events"]):
        raise AssertionError("RETEST_RESULT_EVENT_MISMATCH")
    metrics=job_metrics(result,sim.business,protocol["scenario_max_ticks"])
    if metrics["job_count"]!=spec["jobs"]:
        raise AssertionError("RETEST_INCORRECTLY_CREATED_CLINICAL_JOB")
    for jid,yes in sim.retest_results.items():
        if yes!=synthetic_retest(spec["seed"],jid,spec["retest_rate_percent"]):
            raise AssertionError("RETEST_NOT_KEYED_EXOGENOUS")
    return {"spec":spec,"arm":arm,"metrics":metrics,
       "objective":judge_score(metrics),"policy_calls":len(strategy.records),
       "cpu_seconds":cpu,"strategy_cpu_seconds":strategy.cpu_seconds,
       "resource_wait_ticks":capacity["capacity_wait_ticks"],
       "motion_blocked_attempts":result["summary"]["blocked_move_attempts"],
       "resource_violations":len(capacity["violations"]),
       "motion_violations":motion,"lifecycle_violations":len(lifecycle),
       "retest_results":sim.retest_results,
       "history_sha256":digest(result["history"]),
       "events_sha256":digest(result["events"]),
       "event_rows":result["events"],"history_rows":result["history"],
       "observed_rows":captured,
       "decisions":strategy.records,"workflow_status":result["summary"]["status"],
       "completed_stages":result["summary"]["completed_count"]}


def valid_category(d):
    if d["urgent_waiting"]>=2:return "urgent_vs_urgent"
    if d["urgent_waiting"] and d["routine_waiting"]:return "urgent_vs_routine"
    if d["routine_waiting"]>=2:return "routine_vs_routine"
    return "other"


def classify_opportunities(case):
    out=Counter()
    for d in case["decisions"]:
        cat=valid_category(d)
        if d["candidate_count"]<2:out["no_multiple_legal_matchings"]+=1
        else:
            out["multi_candidate"]+=1
            out[cat]+=1
            if d["active_service"]:out["service_active_with_choice"]+=1
            if d["capacity_blocked"]:out["capacity_blocked_with_choice"]+=1
            if d["outage_visible"]:out["outage_with_choice"]+=1
            if d["retest_visible"]:out["retest_waiting_with_choice"]+=1
    out["all_policy_calls"]=len(case["decisions"])
    out["retest_created"]=sum(e["type"]=="retest_stage_created" for e in case["event_rows"])
    out["retest_decision_no"]=sum(e["type"]=="retest_decision" and
                               not e["needs_retest"] for e in case["event_rows"])
    out["device_pause_events"]=sum(e["type"]=="resource_wait" for e in case["event_rows"])
    return dict(out)
