"""Compare independent fixed-seed runs, excluding only measured timing."""
import json
from pathlib import Path
import sys

TIMING={'wall_seconds','cpu_seconds','decision_wall_ms','decision_wall_mean_ms','decision_wall_p95_ms'}

def stable(value):
    if isinstance(value,dict):return {k:stable(v) for k,v in value.items() if k not in TIMING}
    if isinstance(value,list):return [stable(x) for x in value]
    return value

a=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
b=json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))
if stable(a)!=stable(b):raise ValueError('same-seed output differs beyond wall/CPU timing')
print(a['profile']+': same candidate scores, actions, capacity, forecasts and outcomes; timing excluded')
