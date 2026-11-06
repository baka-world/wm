"""Run a single scenario or compare both dispatch policies on the same seed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .domain import load_scenario
from .report import check_output_directory, write_run
from .simulation import Simulator


def main() -> int:
    parser = argparse.ArgumentParser(description="实验室样本转运仿真（示例数据、CPU）")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "compare"):
        command = commands.add_parser(name)
        command.add_argument("--scenario", default="scenarios/laboratory_demo.json")
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--output", required=True, help="输出目录；不覆盖已有结果")
        if name == "run":
            command.add_argument("--policy", choices=("nearest", "world"), default="world")
    args = parser.parse_args()
    try:
        scenario = load_scenario(args.scenario)
        output = Path(args.output)
        if args.command == "run":
            check_output_directory(output)
            result = Simulator(scenario, args.policy, args.seed).run()
            write_run(result, output)
            print(json.dumps({"policy": args.policy, **result["summary"]}, ensure_ascii=False))
            return 0 if result["summary"]["status"] == "completed" else 2

        if (output / "comparison.json").exists():
            raise FileExistsError(f"refusing to overwrite existing output: {output / 'comparison.json'}")
        for policy in ("nearest", "world"):
            check_output_directory(output / policy)
        results = {policy: Simulator(scenario, policy, args.seed).run() for policy in ("nearest", "world")}
        for policy, result in results.items():
            write_run(result, output / policy)
        comparison = {"scenario": scenario.name, "seed": args.seed,
                      "note": "单场景单种子示例，不能据此推断普遍性能。",
                      "policies": {policy: result["summary"] for policy, result in results.items()}}
        (output / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(comparison, ensure_ascii=False))
        return 0 if all(r["summary"]["status"] == "completed" for r in results.values()) else 2
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"运行失败：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
