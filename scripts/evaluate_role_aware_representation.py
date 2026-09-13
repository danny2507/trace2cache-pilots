#!/usr/bin/env python3
"""Factor length and value-distribution transfer for a trained role-aware encoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from trace2cache.ambiguous_repair import get_ambiguous_cases
from trace2cache.latent import RoleAwareEventEncoder
from run_pilot1 import resolve_local_model
from run_repair_codebook import NativeRepairCodebook, initial_native_codes
from run_role_aware_repair import (
    make_dataset,
    native_content_table,
    numerical_examples,
    representation_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument(
        "--codebook-checkpoint",
        default="checkpoints/repair_latent/repair_codebook_seed131.pt",
    )
    parser.add_argument(
        "--encoder-checkpoint",
        default="checkpoints/repair_latent/role_aware_repair_seed151.pt",
    )
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20151)
    parser.add_argument(
        "--output", default="artifacts/repair_latent/role_aware_transfer_audit_seed151.json"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; this evaluator never falls back to CPU")
    items = get_ambiguous_cases()
    split_specs = {
        "new_short_in_range": (False, False),
        "long_in_range": (True, False),
        "short_ood_values": (False, True),
        "long_ood_values": (True, True),
    }
    raw_splits = {
        name: make_dataset(
            items,
            args.size,
            args.seed + index * 1000,
            long_bundle=long_bundle,
            heldout_values=heldout_values,
        )
        for index, (name, (long_bundle, heldout_values)) in enumerate(split_specs.items())
    }
    model_path = resolve_local_model(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.bfloat16
    ).to("cuda").eval()
    model.requires_grad_(False)
    anchors = initial_native_codes(model, tokenizer, items, slots=8)
    codebook = NativeRepairCodebook(anchors).to(model.device).eval()
    codebook_payload = torch.load(
        args.codebook_checkpoint, map_location=model.device, weights_only=True
    )
    codebook.residual.data.copy_(codebook_payload["adapter_trainable"]["residual"])
    target_codes = codebook(torch.arange(len(items), device=model.device)).detach()

    encoder_payload = torch.load(
        args.encoder_checkpoint, map_location=model.device, weights_only=True
    )
    encoder = RoleAwareEventEncoder(
        model_width=target_codes.shape[-1],
        output_anchor=encoder_payload["output_anchor"],
        hidden_width=256,
        num_roles=10,
        max_events=192,
        max_tests=8,
    ).to(model.device)
    missing, unexpected = encoder.load_state_dict(
        encoder_payload["adapter_trainable"], strict=False
    )
    if unexpected or missing != ["output_anchor"]:
        raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    encoder.eval()

    vocabulary = sorted(
        {
            content
            for split in raw_splits.values()
            for example in split
            for content in example.contents
        }
    )
    vocabulary.append("expected output unknown")
    content_to_id = {content: index for index, content in enumerate(vocabulary)}
    table = native_content_table(model, tokenizer, vocabulary)
    numerical = {
        name: numerical_examples(split, content_to_id) for name, split in raw_splits.items()
    }
    metrics = {}
    for name, split in numerical.items():
        overall = representation_metrics(encoder, split, table, target_codes)
        per_behavior = {
            items[label].case.case_id: representation_metrics(
                encoder,
                [example for example in split if example.label == label],
                table,
                target_codes,
            )["nearest_code_accuracy"]
            for label in range(len(items))
        }
        metrics[name] = {**overall, "nearest_code_accuracy_by_behavior": per_behavior}
        print(json.dumps({"split": name, **overall}), flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"size_per_split": args.size, "metrics": metrics}, indent=2) + "\n")


if __name__ == "__main__":
    main()
