"""Archive offline evidence without duplicating aligned branch observations."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil


def write_gzip(path, value):
    raw = (json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode()
    with path.open('xb') as handle:
        with gzip.GzipFile(filename='',mode='wb',fileobj=handle,mtime=0,compresslevel=9) as compressed:
            compressed.write(raw)


def aligned_delta(data):
    cases = []
    for case in data['cases']:
        decisions = []
        for d in case['decisions']:
            decisions.append({'tick':d['tick'],'comparable':d['comparable'],
                              'horizons':d.get('horizons'),
                              'candidates':[{'action':c['action'],'predictions':c.get('predictions')}
                                            for c in d['candidates']]})
        cases.append({'layout':case['layout'],'seed':case['seed'],'regime':case['regime'],'decisions':decisions})
    return {'profile':data['profile'],'scope':data['scope'],
            'reference':'Match layout, seed, regime, decision tick and action against counterfactual raw.',
            'cases':cases}


def online_compact(data):
    cases = []
    for case in data['cases']:
        cases.append({key:value for key,value in case.items() if key not in ('config','design')})
    return {**{key:value for key,value in data.items() if key != 'cases'},'cases':cases}


def archive(root,output):
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'refusing to overwrite archive: {output}')
    output.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(root/'scenario-config.json',output/'scenario-config.json')
    for profile in ('development','confirmation'):
        for kind in ('counterfactual','aligned','online'):
            directory = root/f'{kind}-{profile}'
            for name in ('summary.json','analysis.json','paired.json'):
                source = directory/name
                if source.exists():
                    shutil.copyfile(source,output/f'{kind}-{profile}-{name}')
            data = json.loads((directory/'raw.json').read_text(encoding='utf-8'))
            value = aligned_delta(data) if kind == 'aligned' else online_compact(data) if kind == 'online' else data
            suffix = 'delta' if kind == 'aligned' else 'raw'
            write_gzip(output/f'{kind}-{profile}-{suffix}.json.gz',value)
            audit = directory/'oracle-audit.json'
            if audit.exists():
                write_gzip(output/f'{kind}-{profile}-oracle-audit.json.gz',
                           json.loads(audit.read_text(encoding='utf-8')))
    with (output/'SHA256SUMS').open('x',encoding='utf-8') as handle:
        for path in sorted(output.iterdir()):
            if path.name != 'SHA256SUMS':
                handle.write(hashlib.sha256(path.read_bytes()).hexdigest()+'  '+path.name+'\n')
    return len(list(output.iterdir()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    print(f'Archived {archive(args.input,args.output)} files; aligned records reference the primary replay')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
