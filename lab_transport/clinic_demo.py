"""Generate a self-contained two-robot playback on the pinned clinic demo map."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .clinic_map import make_clinic_scenario
from .multisite_experiment import POLICIES, REGIMES, make_case
from .report import write_run
from .simulation import Simulator


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinic-map", required=True)
    parser.add_argument("--clinic-config", required=True)
    parser.add_argument("--policy", choices=POLICIES, default="world")
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--regime", choices=REGIMES, default="normal")
    parser.add_argument("--model", help="required for --policy learned")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        if args.policy == "learned" and not args.model:
            raise ValueError("--policy learned requires --model")
        parameters = json.loads(Path(args.model).read_text(encoding="utf-8")) if args.policy == "learned" else None
        base = make_clinic_scenario(Path(args.clinic_map).read_bytes(), Path(args.clinic_config).read_bytes())
        scenario = make_case(base, "clinic", args.seed, args.regime)
        result = Simulator(scenario, args.policy, args.seed, parameters).run()
        write_run(result, args.output)
        print(json.dumps(result["summary"], ensure_ascii=False))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"诊所演示失败：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
