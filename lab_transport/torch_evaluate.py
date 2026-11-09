"""Evaluate a fixed checkpoint on held-out factual trajectories.

One-step counts cover known robot positions and sample statuses only. Multi-step
counts are conditional diagnostics on windows without future assignments,
new arrivals or station changes; their coverage is reported explicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

from .dynamics_experiment import Episode, aggregate, evaluate_episode
from .torch_training import (geometry_from_map, load_adapter, load_checkpoint, observation_from_dict,
                             resolve_device, runtime_code_hashes, source_hash)
from .training_data import _path, validate_dataset
from .transition_schema import Assignment, Transition


def evaluate(dataset: Path, checkpoint_path: Path, output: Path, *, device_name: str = "cpu",
             splits: tuple[str, ...] = ("test_tartu", "test_clinic"), limit_per_split: int | None = None):
    if not splits or len(set(splits)) != len(splits) or any(split not in {"validation", "test_tartu", "test_clinic"} for split in splits):
        raise ValueError("evaluation splits must be distinct validation/test split names")
    if limit_per_split is not None and limit_per_split < 1:
        raise ValueError("transition limit must be positive")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    torch.set_num_threads(1)
    device = resolve_device(device_name)
    validate_dataset(dataset)
    dataset_hash = hashlib.sha256((dataset / "manifest.json").read_bytes()).hexdigest()
    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    if dataset_hash != config["dataset_manifest_sha256"]:
        raise ValueError("checkpoint was trained on a different dataset manifest")
    adapter = load_adapter(config["adapter"])
    if adapter.ADAPTER_VERSION != config["adapter_version"] or source_hash(adapter) != config["adapter_source_sha256"]:
        raise ValueError("adapter version or source does not match checkpoint")
    if runtime_code_hashes() != config["runtime_code_sha256"]:
        raise ValueError("model contract or feature code does not match checkpoint")
    network = adapter.build_model().to(device)
    network.load_state_dict(checkpoint["model_state"])
    network.eval()
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    geometries, cases = {}, []
    counts = {name: 0 for name in splits}
    for entry in manifest["episodes"]:
        split, layout = entry["split"], entry["layout"]
        if split not in splits or (limit_per_split is not None and counts[split] >= limit_per_split):
            continue
        if layout not in geometries:
            topology = json.loads(_path(dataset, manifest["maps"][layout]["file"]).read_text(encoding="utf-8"))
            geometries[layout] = geometry_from_map(topology)
        rows = []
        observer = adapter.make_observer() if hasattr(adapter, "make_observer") else None
        with _path(dataset, entry["file"]).open(encoding="utf-8") as handle:
            for line in handle:
                value = json.loads(line)
                current = observation_from_dict(value["observation"])
                future = observation_from_dict(value["next_observation"])
                if observer is not None:
                    current, future = observer.observe(current), observer.observe(future)
                rows.append(Transition(current,
                    tuple(Assignment(**a) for a in value["assignments"]), future,
                    value["next_boundary"], value["termination"]))
                counts[split] += 1
                if limit_per_split is not None and counts[split] >= limit_per_split:
                    break
        episode = Episode(split, layout, entry["seed"], entry["regime"], entry["behavior_policy"],
                          entry["episode_status"], entry["completed_tasks"], rows)
        def factory(geometry, parameters, mode):
            return adapter.build_dynamics(geometry, network, device)
        extra_models = adapter.build_baselines(geometries[layout]) if hasattr(adapter, 'build_baselines') else None
        cases.append(evaluate_episode(episode, geometries[layout], None, factory, extra_models))
    if any(not counts[name] for name in splits):
        raise ValueError("a requested evaluation split has no transitions")
    summaries = {split: aggregate(cases, split) for split in splits}
    if limit_per_split is not None:
        # These episode outcomes describe complete trajectories, not the
        # truncated prefix that was actually evaluated.
        for summary in summaries.values():
            summary.pop("timed_out_episodes", None)
            summary.pop("completed_tasks", None)
    result = {"scope": "Factual synthetic trajectories; no clinical or closed-loop improvement claim.",
              "checkpoint": checkpoint_path.name,
              "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
              "dataset_manifest_sha256": dataset_hash, "adapter": config["adapter"], "device": str(device),
              "limit_per_split": limit_per_split, "transitions_per_split": counts,
              "multi_step_scope": "Only windows without subsequent assignments, new arrivals or station changes; coverage is reported.",
              "known_entity_joint_scope": "Robot positions and statuses of samples already visible at the window start; excludes new arrivals and station availability.",
              "summaries": summaries, "cases": cases}
    output.mkdir(parents=True, exist_ok=True)
    with (output / "evaluation.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--splits", nargs="+", default=["test_tartu", "test_clinic"])
    parser.add_argument("--limit-per-split", type=int, help="limit trajectory transitions for a smoke test")
    args = parser.parse_args()
    try:
        result = evaluate(args.dataset, args.checkpoint, args.output, device_name=args.device,
                          splits=tuple(args.splits), limit_per_split=args.limit_per_split)
        compact = {split: {"transitions": result["transitions_per_split"][split],
                           "one_step_joint": {name: entry["rates"]["known_entity_joint_accuracy"]
                                              for name, entry in summary["one_step"].items()},
                           "multi_step_coverage": {h: value["coverage"] for h, value in summary["multi_step"].items()}}
                   for split, summary in result["summaries"].items()}
        print(json.dumps({"device": result["device"], "summaries": compact}, ensure_ascii=False), flush=True)
        return 0
    except (OSError, ValueError, TypeError, KeyError, ImportError, RuntimeError) as exc:
        print(f"评估失败：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
