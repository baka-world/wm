"""Export factual, observable one-step transitions from a JSON grid scenario.

The default input is synthetic. The exported rows are preparation data, not a
trained dynamics model or counterfactual outcomes for unexecuted actions.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

from .domain import load_scenario
from .simulation import Simulator
from .transition_schema import SCHEMA_VERSION, Transition


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="scenarios/laboratory_demo.json")
    parser.add_argument("--policy", choices=("nearest", "world", "rule", "learned", "learned_rule", "rollout"),
                        default="world")
    parser.add_argument("--model", help="parameter JSON for a learned policy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        learned = args.policy in {"learned", "learned_rule"}
        if learned != bool(args.model):
            raise ValueError("--model is required only for a learned policy")
        source = Path(args.scenario)
        scenario = load_scenario(source)
        parameters = json.loads(Path(args.model).read_text(encoding="utf-8")) if args.model else None
        output = Path(args.output)
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
        transitions: list[Transition] = []
        result = Simulator(scenario, args.policy, args.seed, parameters).run(on_transition=transitions.append)
        # load_scenario currently accepts grids; static topology is known before a run.
        manifest = {"schema_version": SCHEMA_VERSION,
                    "data_scope": "Scenario-provided simulation; provenance and clinical applicability require independent verification.",
                    "static_grid": {"width": scenario.grid.width, "height": scenario.grid.height,
                                    "blocked": [list(p) for p in sorted(scenario.grid.blocked)]},
                    "bookkeeping_only_not_model_input": {
                        "scenario_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "policy": args.policy, "seed": args.seed,
                        "transition_count": len(transitions), "termination": result["summary"]["status"]},
                    "limitations": ["No counterfactual labels; assignments are the executed actions.",
                                    "Other motion and service follow the simulator's built-in controller.",
                                    "Next observations may include previously unknown task arrivals or outages.",
                                    "The last end_of_run boundary is before releases at that boundary."]}
        output.mkdir(parents=True, exist_ok=True)
        (output / "transitions.jsonl").write_text(
            "".join(json.dumps(asdict(row), ensure_ascii=False) + "\n" for row in transitions), encoding="utf-8")
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"transitions": len(transitions), "termination": result["summary"]["status"],
                          "schema_version": SCHEMA_VERSION}))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"状态转移导出失败：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
