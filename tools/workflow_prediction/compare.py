"""Require repeated fixed experiments to match, excluding measured timing."""
import json
from pathlib import Path
import sys

TIMING={'wall_seconds','cpu_seconds','decision_wall_ms','decision_wall_mean_ms',
        'decision_wall_p95_ms','model_prediction_total_wall_ms','model_prediction_mean_wall_ms'}


def stable(value):
    if isinstance(value,dict):return {k:stable(v) for k,v in value.items() if k not in TIMING}
    if isinstance(value,list):return [stable(v) for v in value]
    return value


if __name__=='__main__':
    first,second=(json.loads(Path(p).read_text()) for p in sys.argv[1:3])
    if stable(first)!=stable(second):raise ValueError('repeat differs beyond measured timing')
    print(Path(sys.argv[1]).name+': repeated candidates, predictions, transitions and outcomes match')
