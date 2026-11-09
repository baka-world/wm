"""Compare independently repeated experiments, excluding actual timing only."""
import json
from pathlib import Path
import sys

TIMING={'wall_seconds','cpu_seconds','path_wall_ms','path_wall_mean_ms','path_wall_p95_ms'}

def stable(value):
    if isinstance(value,dict):return {k:stable(v) for k,v in value.items() if k not in TIMING}
    if isinstance(value,list):return [stable(v) for v in value]
    return value

a=json.loads(Path(sys.argv[1]).read_text());b=json.loads(Path(sys.argv[2]).read_text())
if stable(a)!=stable(b):raise ValueError('repeated experiment differs beyond timing')
print(Path(sys.argv[1]).name+': repeat identical except measured timing')
