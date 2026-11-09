"""Require same-seed outcomes, actions and forecasts to match except CPU time."""
import json
from pathlib import Path
import sys


def strip_timing(value):
    if isinstance(value,dict):
        return {k:strip_timing(v) for k,v in value.items() if 'cpu' not in k and 'seconds' not in k}
    if isinstance(value,list):
        return [strip_timing(v) for v in value]
    return value


def main(a,b):
    first=json.loads(Path(a).read_text(encoding='utf-8'))
    second=json.loads(Path(b).read_text(encoding='utf-8'))
    if first['profile']!=second['profile']:
        raise ValueError('profile differs')
    if strip_timing(first)!=strip_timing(second):
        raise ValueError('same-seed candidate, score, action or metric differs beyond timing')
    for case in second['cases']:
        for method in case['methods'].values():
            if method['solver_partial_truncations']:
                raise ValueError('solver truncated candidate enumeration')
    print(first['profile']+': candidates, scores, actions and outcomes reproduced; CPU time excluded')


if __name__=='__main__':main(sys.argv[1],sys.argv[2])
