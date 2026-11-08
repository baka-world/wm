#!/usr/bin/env python3
"""Check whether a public MAPF-GPT checkpoint can initialize a MAPF-World fast branch.

Uses the upstream implementations from isolated, pinned checkouts.  It does not
train a world model, evaluate transport quality, or access a GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch


def strip_compiler_prefix(key: str) -> str:
    return key.removeprefix("_orig_mod.")


def target_name(source: str) -> str:
    mapping = (
        ("transformer.wte.", "fast_token_embedding."),
        ("transformer.wpe.", "fast_position_embedding."),
        ("transformer.h.", "shared_backbone."),
        ("transformer.ln_f.", "fast_norm."),
        ("lm_head.", "lm_head."),
    )
    for old, new in mapping:
        if source.startswith(old):
            return new + source[len(old):]
    raise ValueError(f"unmapped source parameter: {source}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapf-world", type=Path, required=True)
    parser.add_argument("--mapf-gpt", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # Local checkouts are fixed by the workflow; no source files are modified.
    sys.path.insert(0, str(args.mapf_world.resolve()))
    sys.path.insert(0, str(args.mapf_gpt.resolve()))
    from model.world import FastAndSlow_S_2
    from mapf_gpt.model import GPT, GPTConfig

    torch.set_num_threads(1)
    torch.manual_seed(7)
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    expected_hash = "1a26ed541484b096f00565dc4b654c5827ad238c137c4ad15566ceb5cede7f56"
    if checkpoint_hash != expected_hash:
        raise ValueError("downloaded MAPF-GPT checkpoint SHA-256 mismatch")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise ValueError("checkpoint does not contain a model state dictionary")

    config_values = checkpoint.get("model_args")
    if not isinstance(config_values, dict):
        raise ValueError("checkpoint does not contain model_args")
    expected_config = {"n_layer": 5, "n_head": 5, "n_embd": 160, "block_size": 256, "vocab_size": 67}
    for name, expected in expected_config.items():
        if config_values.get(name) != expected:
            raise ValueError(f"unexpected checkpoint configuration for {name}: {config_values.get(name)}")

    source = {strip_compiler_prefix(key): value for key, value in checkpoint["model"].items()}
    if len(source) != len(checkpoint["model"]):
        raise ValueError("duplicate key after compiler-prefix removal")
    reference = GPT(GPTConfig(**config_values)).eval()
    reference.load_state_dict(source, strict=True)

    target = FastAndSlow_S_2().eval()
    target_state = target.state_dict()
    mapped = {}
    for name, tensor in source.items():
        destination = target_name(name)
        if destination not in target_state:
            raise ValueError(f"destination does not exist: {destination}")
        if tensor.shape != target_state[destination].shape:
            raise ValueError(f"shape mismatch: {name} -> {destination}")
        mapped[destination] = tensor

    fast_prefixes = ("fast_token_embedding.", "fast_position_embedding.", "shared_backbone.", "fast_norm.", "lm_head.")
    required = {name for name in target_state if name.startswith(fast_prefixes)}
    if set(mapped) != required:
        raise ValueError(f"fast-branch keys do not match: missing={sorted(required - set(mapped))}, "
                         f"extra={sorted(set(mapped) - required)}")
    target.load_state_dict(mapped, strict=False)
    if target.fast_token_embedding.weight.data_ptr() != target.lm_head.weight.data_ptr():
        raise ValueError("fast input embedding and output head no longer share weights")
    for name, value in mapped.items():
        if not torch.equal(target.state_dict()[name], value):
            raise ValueError(f"mapped parameter changed while loading: {name}")

    # Identical arbitrary tokens check parameter/forward equivalence, not
    # realistic task performance, prediction quality, or model generalization.
    tokens = torch.randint(0, expected_config["vocab_size"], (2, expected_config["block_size"]), dtype=torch.long)
    with torch.inference_mode():
        expected_logits, _ = reference(tokens)
        hidden = target.shared_backbone(target._encode_fast(tokens))
        actual_logits = target.lm_head(target.fast_norm(hidden)[:, [-1], :])
    max_error = float((expected_logits - actual_logits).abs().max().item())
    if not torch.allclose(expected_logits, actual_logits, rtol=1e-5, atol=1e-6):
        raise ValueError(f"mapped fast branch differs from original: max_abs_error={max_error}")

    result = {
        "device": "cpu",
        "reference": "MAPF-GPT-2M",
        "checkpoint_sha256": checkpoint_hash,
        "config": expected_config,
        "source_tensor_count": len(source),
        "mapped_fast_tensors": len(mapped),
        "mapped_fast_tensor_values": sum(value.numel() for value in mapped.values()),
        "fast_branch_max_abs_logit_error": max_error,
        "embedding_weights_tied": True,
        "world_branch_pretrained": False,
        "transport_model_adapted": False,
        "scope": "CPU weight mapping and synthetic-token equality only; world branch and sample transport are not evaluated.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
