"""Exercise a trained adapter in the existing discrete simulation loop.

This is a small integration smoke test, not a validated dispatch algorithm or
clinical safety proof. The fixed horizon and score are experimental defaults.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from .domain import load_scenario
from .dynamics_experiment import check_predicted_step
from .dynamics_model import Geometry
from .hospital_case import make_scenario
from .multisite_experiment import discrete_safety_violations, make_case, prepare_tartu
from .report import write_run
from .simulation import Simulator
from .torch_training import load_adapter, load_checkpoint, resolve_device, runtime_code_hashes, source_hash


class RolloutPolicy:
    """Rank a bounded list of available single assignments using model rollout."""
    def __init__(self, dynamics, geometry, horizon: int = 3, budget: int = 8, observer=None):
        if not 1 <= horizon <= 20 or not 1 <= budget <= 100:
            raise ValueError("horizon or candidate budget is outside the smoke-test range")
        self.dynamics, self.geometry = dynamics, geometry
        self.horizon, self.budget = horizon, budget
        self.log = []
        self.model_steps = 0
        self.fallbacks = 0
        self.observer = observer

    def __call__(self, observation, candidates):
        if self.observer is not None:
            observation = self.observer.observe(observation)
        records = []
        for action in candidates[:self.budget]:
            try:
                predicted = observation
                for step in range(self.horizon):
                    next_state = self.dynamics.predict_next(predicted, (action,) if step == 0 else ())
                    self.model_steps += 1
                    check_predicted_step(predicted, next_state, self.geometry)
                    predicted = next_state
                robot = next(r for r in predicted.robots if r.id == action.robot_id)
                sample = next(s for s in predicted.samples if s.id == action.sample_id)
                stations = {s.id: s.position for s in predicted.stations}
                if sample.status == "completed":
                    score = 0
                elif sample.status == "in_transit":
                    score = self.geometry.distance(robot.position, stations[sample.dropoff])
                else:
                    score = (self.geometry.distance(robot.position, stations[sample.pickup]) +
                             self.geometry.distance(stations[sample.pickup], stations[sample.dropoff]))
            except (ValueError, RuntimeError, TypeError, KeyError, StopIteration, OverflowError) as exc:
                # Available candidates were built by the simulator and are
                # checked again there. This is a deterministic operational
                # fallback, not a guarantee of liveness or physical safety.
                self.fallbacks += 1
                self.log.append({"tick": observation.tick, "candidates_available": len(candidates),
                                 "candidates_evaluated": records, "fallback": type(exc).__name__,
                                 "selected_index": 0})
                return [candidates[0]]
            records.append({"robot_id": action.robot_id, "sample_id": action.sample_id,
                            "score": score, "predicted_status": sample.status,
                            "predicted_position": list(robot.position)})
        selected = min(range(len(records)), key=lambda index: (records[index]["score"], index))
        self.log.append({"tick": observation.tick, "candidates_available": len(candidates),
                         "candidates_evaluated": records, "selected_index": selected})
        return [candidates[selected]]


def run(checkpoint_path: Path, output: Path, scenario, *, adapter_name: str | None = None,
        device_name: str = "cpu", horizon: int = 3, budget: int = 8, seed: int = 42):
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    if adapter_name is not None and adapter_name != config["adapter"]:
        raise ValueError("requested adapter does not match checkpoint")
    adapter = load_adapter(config["adapter"])
    if (adapter.ADAPTER_VERSION != config["adapter_version"] or
            source_hash(adapter) != config["adapter_source_sha256"]):
        raise ValueError("adapter version or source does not match checkpoint")
    if runtime_code_hashes() != config["runtime_code_sha256"]:
        raise ValueError("model contract or feature code does not match checkpoint")
    device = resolve_device(device_name)
    network = adapter.build_model()
    network.load_state_dict(checkpoint["model_state"])
    geometry = Geometry(scenario.grid)
    observer = adapter.make_observer() if hasattr(adapter, "make_observer") else None
    policy = RolloutPolicy(adapter.build_dynamics(geometry, network, device), geometry, horizon, budget, observer)
    result = Simulator(scenario, "world", seed).run(dispatch_policy=policy,
                                                       on_observation=observer.observe if observer else None)
    evidence = {"scope": "Synthetic integration smoke test, not a clinical or comparative evaluation.",
                "checkpoint": checkpoint_path.name,
                "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                "dataset_manifest_sha256": config["dataset_manifest_sha256"],
                "adapter": config["adapter"], "adapter_source_sha256": config["adapter_source_sha256"],
                "device": str(device),
                "horizon": horizon, "candidate_budget_per_dispatch": budget,
                "dispatch_calls": len(policy.log), "candidate_evaluations": sum(len(x["candidates_evaluated"]) for x in policy.log),
                "model_steps": policy.model_steps, "fallbacks": policy.fallbacks,
                "model_process_cpu_seconds": getattr(policy.dynamics, "process_cpu_seconds", None),
                "discrete_safety_violations": discrete_safety_violations(result),
                "summary": result["summary"]}
    output.mkdir(parents=True, exist_ok=True)
    write_run(result, output / "playback")
    (output / "decisions.json").write_text(json.dumps(policy.log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--scenario", default="scenarios/laboratory_demo.json", help="synthetic JSON demo if no hospital map is supplied")
    parser.add_argument("--hospital-map", type=Path, help="public hospital map for the synthetic Tartu case")
    parser.add_argument("--tiago-config", type=Path, help="public Tartu robot time-model configuration")
    parser.add_argument("--download-sources", action="store_true", help="fetch and verify the pinned public sources")
    parser.add_argument("--scenario-seed", type=int, default=11000)
    parser.add_argument("--regime", choices=("normal", "short_outage"), default="short_outage")
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--budget", type=int, default=8)
    args = parser.parse_args()
    try:
        if bool(args.hospital_map) != bool(args.tiago_config):
            raise ValueError("hospital map and robot config must be provided together")
        if args.download_sources and args.hospital_map:
            raise ValueError("choose downloaded sources or local source files, not both")
        if args.download_sources or args.hospital_map:
            if args.download_sources:
                from .training_data import download_sources
                source = download_sources()
                map_bytes, config_bytes = source["hospital_map"], source["tiago_config"]
            else:
                map_bytes, config_bytes = args.hospital_map.read_bytes(), args.tiago_config.read_bytes()
            base = prepare_tartu(make_scenario(map_bytes, seconds_per_tick=10, tiago_config=config_bytes))
            scenario = make_case(base, "tartu", args.scenario_seed, args.regime)
        else:
            scenario = load_scenario(args.scenario)
        evidence = run(args.checkpoint, args.output, scenario, device_name=args.device,
                       horizon=args.horizon, budget=args.budget, seed=args.scenario_seed)
        print(json.dumps(evidence, ensure_ascii=False), flush=True)
        return 0
    except (OSError, ValueError, TypeError, KeyError, ImportError, RuntimeError) as exc:
        print(f"闭环测试失败：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
