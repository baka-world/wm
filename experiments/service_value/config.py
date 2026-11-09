"""Export fixed synthetic scenario configuration for offline reproducibility."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from lab_transport.decision_value import case_config
from lab_transport.decision_value_round2 import make_round2_cases
from .counterfactual import REGIMES, SEEDS, audit_schedules, load_pr9_model
from lab_transport.service_fault_data import source_bases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise FileExistsError(f'refusing to overwrite: {args.output}')
        bases,sources = source_bases()
        audit = audit_schedules(bases)
        if audit['cross_group_duplicates']:
            raise ValueError('scenario task schedule overlaps')
        _,checkpoint = load_pr9_model()
        cases = []
        for profile,layouts in SEEDS.items():
            for layout,seeds in layouts.items():
                for seed in seeds:
                    variants,design = make_round2_cases(bases[layout],layout,seed)
                    for regime in REGIMES:
                        case,pauses,incident = variants[regime]
                        cases.append({'profile':profile,'layout':layout,'seed':seed,'regime':regime,
                                      'incident':incident,'probe_design':design,'config':case_config(case,pauses)})
        result = {'scope':'Offline simulation configuration; future schedule and simulator settings never model inputs.',
                  'source_sha256':sources,'schedule_audit':audit,'checkpoint':checkpoint,'cases':cases}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(f'Wrote {len(cases)} predeclared synthetic scenario configurations; no schedule overlap')
        return 0
    except (OSError,ValueError,KeyError,RuntimeError) as exc:
        print(f'configuration export failed: {exc}',file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
