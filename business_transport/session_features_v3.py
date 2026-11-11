"""Versioned v3 observable service-query extension, v2 unaffected.

The next-step label semantics remain business-service-query-v2; only the
numerical INPUT schema changes. No actual completed duration, hidden countdown
or future outage is imported.
"""
from __future__ import annotations
from business_transport.session_events import encode_query as encode_query_v2

INPUT_SCHEMA_V3="business-service-query-input-v3"
QUERY_WIDTH_V3=56

def _scalar(value,scale):
    if not isinstance(value,dict) or type(value.get("known")) is not bool:
        raise ValueError("v3 feature has no explicit known mask")
    if not value["known"]:
        if value.get("value") is not None:raise ValueError("fabricated unknown v3 scalar")
        return [0.,0.]
    x=value.get("value")
    if not isinstance(x,(int,float)) or isinstance(x,bool) or not 0<=float(x)<1e9:
        raise ValueError("invalid or missing nonnegative v3 scalar")
    return [min(4.,float(x)/scale),1.]

def encode_query_v3(payload,query):
    """Add TARGET station expected duration and observed relative progress.

    Old target fields service_elapsed, service_active_elapsed, paused elapsed,
    station availability/resource and historical mean/count already live in v2.
    Public expected service ticks are required for all v3 queries.
    """
    base=encode_query_v2(payload,query) # validates exact robot/sample/phase/station join
    stations={s["id"]:s for s in payload["observed_facts"]["stations"]}
    robots={r["id"]:r for r in payload["observed_facts"]["robots"]}
    station=stations[query["station_id"]]; robot=robots[query["robot_id"]]
    expected_field=station["expected_service_ticks_public"]
    expected=_scalar(expected_field,10)
    if not expected[1] or float(expected_field["value"])<=0:
        raise ValueError("v3 target public expected service duration required")
    prior=float(expected_field["value"])
    active=robot["service_active_elapsed"]; wall=robot["service_elapsed"]
    historical=station["observed_effective_service_mean"]
    def ratio(numerator, denominator):
        if (not numerator["known"] or not denominator["known"] or
            denominator.get("value") is None or float(denominator["value"])<=0):
            return [0.,0.]
        return [min(4.,float(numerator["value"])/float(denominator["value"])),1.]
    extra=(expected+
           ratio(active,expected_field)+
           ratio(wall,expected_field)+
           ratio(active,historical))
    result=base+extra
    if len(result)!=QUERY_WIDTH_V3:
        raise ValueError("v3 target feature width mismatch")
    return result
