"""CPU/CUDA training harness for replaceable transition-model adapters.

Only the explicitly selected adapter receives factual next observations to
construct labels. Its model forward method receives collated inputs alone.
Dataset and checkpoint files are local; no dataset upload is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import torch
from torch.utils.data import DataLoader

from .domain import DirectedGraph, Grid
from .dynamics_model import Geometry
from .training_data import _path, validate_dataset
from .transition_schema import Assignment, Observation, ObservedRobot, ObservedSample, ObservedStation

DEFAULT_ADAPTER = "lab_transport.torch_smoke_adapter"
CHECKPOINT_VERSION = 1


def observation_from_dict(value: dict[str, Any]) -> Observation:
    robots = tuple(ObservedRobot(**{**r, "position": tuple(r["position"]), "home": tuple(r["home"])})
                   for r in value["robots"])
    samples = tuple(ObservedSample(**s) for s in value["samples"])
    stations = tuple(ObservedStation(**{**s, "position": tuple(s["position"])}) for s in value["stations"])
    return Observation(value["schema_version"], value["tick"], robots, samples, stations)


def geometry_from_map(topology: dict[str, Any]) -> Geometry:
    if topology["kind"] == "directed_graph":
        coordinates = {tuple(node["position"]): tuple(node["coordinates"]) for node in topology["nodes"]}
        edges = tuple((tuple(edge[0]), tuple(edge[1])) for edge in topology["edges"])
        return Geometry(DirectedGraph(coordinates, edges))
    if topology["kind"] == "grid":
        return Geometry(Grid(topology["width"], topology["height"],
                             frozenset(tuple(p) for p in topology["blocked"])))
    raise ValueError("unsupported map type")


def load_adapter(name: str):
    adapter = importlib.import_module(name)
    required = ("ADAPTER_VERSION", "build_model", "encode_transition", "collate", "loss", "build_dynamics")
    if any(not hasattr(adapter, attr) for attr in required):
        raise ValueError("adapter is missing a required model, data, loss or dynamics hook")
    return adapter


def source_hash(module) -> str:
    """Detect an edited adapter when resuming; custom dependencies remain the adapter author's responsibility."""
    source = getattr(module, "__file__", None)
    if not source:
        raise ValueError("adapter must be loaded from a source file")
    return hashlib.sha256(Path(source).read_bytes()).hexdigest()


def runtime_code_hashes() -> dict[str, str]:
    """Fingerprint shared contracts/features used by the bundled adapter."""
    directory = Path(__file__).parent
    names = ("torch_training.py", "transition_schema.py", "simulation.py", "dynamics_model.py",
             "dynamics_revision.py", "dynamics_experiment.py", "state_v2.py")
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in names}


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; install a CUDA-enabled PyTorch and attach a local GPU")
    if name not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu, cuda or auto")
    return torch.device(name)


def configure_deterministic_cuda() -> str:
    """Configure cuBLAS before CUDA is initialized for deterministic training."""
    value = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if value not in {":4096:8", ":16:8"}:
        raise ValueError("CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8 for deterministic CUDA training")
    return value


def prepare_examples(root: Path, split: str, adapter, limit: int | None) -> tuple[list[Any], dict[str, int]]:
    """Read one split only; limits count transitions, not individual entities."""
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    geometries = {}
    examples, transitions, episodes = [], 0, 0
    for entry in manifest["episodes"]:
        if entry["split"] != split:
            continue
        layout = entry["layout"]
        if layout not in geometries:
            topology = json.loads(_path(root, manifest["maps"][layout]["file"]).read_text(encoding="utf-8"))
            geometries[layout] = geometry_from_map(topology)
        episodes += 1
        observer = adapter.make_observer() if hasattr(adapter, "make_observer") else None
        with _path(root, entry["file"]).open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                current = observation_from_dict(row["observation"])
                actions = tuple(Assignment(**a) for a in row["assignments"])
                future = observation_from_dict(row["next_observation"])
                if observer is not None:
                    current = observer.observe(current)
                examples.extend(adapter.encode_transition(geometries[layout], current, actions, future))
                if observer is not None:
                    observer.observe(future)
                transitions += 1
                if limit is not None and transitions >= limit:
                    break
        if limit is not None and transitions >= limit:
            break
    if not examples:
        raise ValueError(f"no training examples in {split} split")
    return examples, {"episodes_read": episodes, "transitions_read": transitions, "examples": len(examples)}


def _to_device(values: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in values.items()}


def _epoch(model, adapter, loader, device, optimizer=None) -> float:
    model.train(optimizer is not None)
    total, count = 0.0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for inputs, labels in loader:
            inputs, labels = _to_device(inputs, device), _to_device(labels, device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)  # Labels and next observations are never passed to forward().
            value, batch_count = adapter.loss(outputs, labels)
            if not bool(torch.isfinite(value).item()):
                raise ValueError("non-finite training or validation loss")
            if optimizer is not None:
                value.backward()
                optimizer.step()
            total += float(value.detach().item()) * batch_count
            count += batch_count
    if not count:
        raise ValueError("no examples in epoch")
    return total / count


def load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported checkpoint")
    return checkpoint


def train(dataset: Path, output: Path, *, adapter_name: str = DEFAULT_ADAPTER, device_name: str = "cpu",
          epochs: int = 2, batch_size: int = 128, learning_rate: float = 0.001, seed: int = 123,
          train_limit: int | None = None, validation_limit: int | None = None,
          resume: Path | None = None) -> dict[str, Any]:
    if epochs < 1 or batch_size < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("epochs, batch size and learning rate must be positive")
    if any(limit is not None and limit < 1 for limit in (train_limit, validation_limit)):
        raise ValueError("transition limits must be positive")
    if resume is None and output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    if resume is not None and not output.is_dir():
        raise ValueError("resume output must be the original existing run directory")
    device, adapter = resolve_device(device_name), load_adapter(adapter_name)
    cublas_workspace = configure_deterministic_cuda() if device.type == "cuda" else None
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.init()
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    torch.use_deterministic_algorithms(True)
    check = validate_dataset(dataset)
    dataset_hash = hashlib.sha256((dataset / "manifest.json").read_bytes()).hexdigest()
    train_examples, train_info = prepare_examples(dataset, "train", adapter, train_limit)
    val_examples, val_info = prepare_examples(dataset, "validation", adapter, validation_limit)
    config = {"adapter": adapter_name, "adapter_version": adapter.ADAPTER_VERSION,
              "adapter_source_sha256": source_hash(adapter),
              "runtime_code_sha256": runtime_code_hashes(),
              "dataset_manifest_sha256": dataset_hash, "device_type": device.type,
              "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
              "cuda_build": torch.version.cuda, "cublas_workspace_config": cublas_workspace,
              "batch_size": batch_size, "learning_rate": learning_rate, "seed": seed,
              "train_limit": train_limit, "validation_limit": validation_limit,
              "torch_version": str(torch.__version__), "python_version": list(sys.version_info[:3])}
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_examples, batch_size=batch_size, shuffle=True,
                              generator=generator, collate_fn=adapter.collate, num_workers=0)
    val_loader = DataLoader(val_examples, batch_size=batch_size, shuffle=False,
                            collate_fn=adapter.collate, num_workers=0)
    model = adapter.build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    start, best_loss, best_epoch = 0, float("inf"), 0
    if resume is not None:
        checkpoint = load_checkpoint(resume)
        if checkpoint.get("config") != config or resume.resolve().parent != output.resolve():
            raise ValueError("resume checkpoint does not match the dataset, adapter, device, run directory or training configuration")
        start = checkpoint["epoch"]
        if epochs <= start:
            raise ValueError("total epochs must exceed the resumed epoch")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        generator.set_state(checkpoint["loader_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(checkpoint["cuda_rng"], device)
        best_loss, best_epoch = checkpoint["best_loss"], checkpoint["best_epoch"]
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for epoch in range(start + 1, epochs + 1):
        target = output / f"epoch-{epoch:04d}.pt"
        if target.exists():
            raise FileExistsError(f"refusing to overwrite checkpoint: {target}")
        began = time.monotonic()
        train_loss = _epoch(model, adapter, train_loader, device, optimizer)
        validation_loss = _epoch(model, adapter, val_loader, device)
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
        report = {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss,
                  "seconds": time.monotonic() - began}
        state = {"checkpoint_version": CHECKPOINT_VERSION, "config": config, "epoch": epoch,
                 "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                 "loader_rng": generator.get_state(), "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                 "best_loss": best_loss, "best_epoch": best_epoch, "epoch_report": report,
                 "data": {"train": train_info, "validation": val_info}}
        # Create a complete checkpoint before publishing its final name. A
        # concurrent writer must not replace an existing checkpoint.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(prefix=".checkpoint-", dir=output, delete=False) as handle:
                temporary = Path(handle.name)
                torch.save(state, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        results.append(report)
        print(json.dumps(report), flush=True)
    result = {"device": str(device), "data_check": check, "dataset_manifest_sha256": dataset_hash,
              "adapter": adapter_name, "train": train_info, "validation": val_info,
              "device_name": config["device_name"], "cuda_build": config["cuda_build"],
              "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
              "best_epoch": best_epoch, "best_validation_loss": best_loss,
              "best_checkpoint": f"epoch-{best_epoch:04d}.pt", "epochs_this_run": results}
    report_path = output / f"report-through-{epochs:04d}.json"
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--epochs", type=int, default=2, help="total epoch count, including any resumed epochs")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train-limit", type=int, help="limit transitions for a smoke test")
    parser.add_argument("--validation-limit", type=int, help="limit validation transitions for a smoke test")
    parser.add_argument("--resume", type=Path, help="checkpoint in the same output directory")
    args = parser.parse_args()
    try:
        result = train(Path(args.dataset), Path(args.output), adapter_name=args.adapter, device_name=args.device,
                       epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, seed=args.seed,
                       train_limit=args.train_limit, validation_limit=args.validation_limit, resume=args.resume)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0
    except (OSError, ValueError, TypeError, KeyError, ImportError, RuntimeError) as exc:
        print(f"训练失败：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
