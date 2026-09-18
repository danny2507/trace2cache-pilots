#!/usr/bin/env python3
"""GPU parity gate for the controlled parent-aggregate typed residual.

This is deliberately a no-update check: it loads a real plain paired-runtime checkpoint, adds the
zero-initialized typed side channel, and proves that actual frozen-receiver features yield exactly
the same latent output before the residual is trained.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_pilot1 import resolve_local_model
from train_paired_trace_encoder import all_views, build_feature_map, load_oracle_codes
from trace2cache.latent import RoleAwareEventEncoder
from trace2cache.native_features import FrozenNativeFeatureExtractor, collate_views
from trace2cache.paired_evidence import read_jsonl
from trace2cache.receiver_training import vector_loss
from trace2cache.typed_values import FEATURE_DIM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/paired_runtime_v2/full_data_vector2000_seed401.pt")
    parser.add_argument("--dataset", default=".local/datasets/paired_runtime_v2/dev.jsonl")
    parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--output", default="artifacts/paired_runtime_v2/typed_parent_aggregate_gpu_parity.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("CUDA BF16 is required")
    if args.views < 1:
        raise SystemExit("--views must be positive")
    torch.cuda.reset_peak_memory_stats()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = payload["args"]
    model_path = resolve_local_model(config["model"])
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    receiver = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).to("cuda").eval()
    receiver.requires_grad_(False)
    oracle = load_oracle_codes(receiver, tokenizer, config["codebook_checkpoint"])
    plain = RoleAwareEventEncoder(
        model_width=receiver.config.hidden_size,
        output_anchor=oracle.mean(0),
        hidden_width=config["hidden_width"],
        max_events=config["max_events"],
        max_tests=config["max_tests"],
    ).to("cuda").eval()
    plain.load_state_dict(payload["encoder"], strict=True)
    typed = RoleAwareEventEncoder(
        model_width=receiver.config.hidden_size,
        output_anchor=oracle.mean(0),
        hidden_width=config["hidden_width"],
        max_events=config["max_events"],
        max_tests=config["max_tests"],
        typed_feature_dim=FEATURE_DIM,
    ).to("cuda").eval()
    missing, unexpected = typed.load_state_dict(payload["encoder"], strict=False)
    expected_missing = [
        "typed_projection.0.weight",
        "typed_projection.0.bias",
        "typed_projection.1.weight",
    ]
    if sorted(missing) != sorted(expected_missing) or unexpected:
        raise RuntimeError(f"unexpected warm-start keys: missing={missing}, unexpected={unexpected}")

    records = read_jsonl(args.dataset)
    views, labels = all_views(records)
    views, labels = views[: args.views], labels[: args.views]
    extractor = FrozenNativeFeatureExtractor(
        receiver, tokenizer, model_id=config["model"], cache_dir=config["feature_cache"]
    )
    features = build_feature_map(extractor, views, config["feature_method"], config["feature_batch_size"])
    plain_batch = collate_views(views, features, receiver.device)
    typed_batch = collate_views(views, features, receiver.device, typed_values=True)
    base_inputs_equal = all(
        torch.equal(plain_batch[name], typed_batch[name])
        for name in ("content_vectors", "role_ids", "test_ids", "event_mask")
    )
    if not base_inputs_equal:
        raise AssertionError("parent-aggregate typed collation changed base event inputs")
    with torch.inference_mode():
        plain_output = plain(**plain_batch).float()
        typed_output = typed(**typed_batch).float()
    output_difference = (plain_output - typed_output).abs()
    if output_difference.max().item() != 0.0:
        raise AssertionError(f"zero residual changed latent output: {output_difference.max().item()}")
    typed.train()
    loss = vector_loss(typed(**typed_batch), oracle[labels.to(receiver.device)])
    loss.backward()
    gradient = typed.typed_projection[1].weight.grad
    if gradient is None or gradient.abs().max().item() == 0.0:
        raise AssertionError("typed residual received no useful first-update gradient")
    result = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "views": len(views),
        "typed_value_mode": "parent_aggregate",
        "base_inputs_equal": base_inputs_equal,
        "typed_projection_max_abs": float(typed.typed_projection[1].weight.abs().max()),
        "latent_max_abs_difference": float(output_difference.max()),
        "typed_projection_gradient_max_abs": float(gradient.abs().max()),
        "gpu_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
