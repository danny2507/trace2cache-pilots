#!/usr/bin/env python3
"""Distill receiver-readable pair codes into a full-trace resampler."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import time

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from trace2cache.latent import NativeEventResampler, NativePairEncoder, trainable_parameter_count
from run_multifact_probe import (
    TASKS,
    compositional_pair_split,
    evaluate,
    make_batch,
    one_token_ids,
    prompt_parts,
    resolve_local_model,
    trainable_state_dict,
)


DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--teacher-checkpoint",
        default="checkpoints/latent_probe/compositional_pair_encoder_balanced_1200.pt",
    )
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-size", type=int, default=256)
    parser.add_argument("--hidden-width", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--cosine-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument(
        "--output", default="artifacts/latent_probe/trace_distillation_seed71.json"
    )
    parser.add_argument(
        "--checkpoint", default="checkpoints/latent_probe/trace_distillation_seed71.pt"
    )
    return parser.parse_args()


def student_prefix(student: NativeEventResampler, batch) -> torch.Tensor:
    # No explicit sink anchor: both outcomes must be recovered from the event sequence.
    return student(batch.event_types, batch.arguments, batch.states, batch.event_mask)


@torch.no_grad()
def representation_metrics(student, teacher, batch) -> dict[str, float]:
    student.eval()
    predicted = student_prefix(student, batch)[:, 0]
    target = teacher(batch.reference_values, batch.buggy_values)[:, 0]
    difference = predicted - target
    return {
        "cosine_similarity": nn.functional.cosine_similarity(predicted, target, dim=-1).mean().item(),
        "rmse": difference.square().mean().sqrt().item(),
        "relative_l2": (difference.norm(dim=-1) / target.norm(dim=-1).clamp_min(1e-6)).mean().item(),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; this pilot will not fall back to CPU")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda"
    model_path = resolve_local_model(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    digit_ids = one_token_ids(tokenizer, [str(i) for i in range(10)])
    label_ids = one_token_ids(tokenizer, ["Yes", "No"])
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.bfloat16
    ).to(device).eval()
    model.requires_grad_(False)
    digit_token_ids = [digit_ids[str(i)] for i in range(10)]

    teacher = NativePairEncoder(
        model.get_input_embeddings(), digit_token_ids, hidden_width=256
    ).to(device).eval()
    teacher_payload = torch.load(args.teacher_checkpoint, map_location=device, weights_only=False)
    missing, unexpected = teacher.load_state_dict(
        teacher_payload["adapter_trainable"], strict=False
    )
    allowed_missing = {"token_embedding.weight", "digit_token_ids"}
    if unexpected or any(name not in allowed_missing for name in missing):
        raise RuntimeError(f"teacher checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    teacher.requires_grad_(False)

    width = model.get_input_embeddings().embedding_dim
    student = NativeEventResampler(
        model.get_input_embeddings(),
        model_width=width,
        hidden_width=args.hidden_width,
        num_latents=1,
        num_event_types=12,
        max_events=32,
        native_sink_anchor=False,
        zero_output_init=True,
        fixed_positions=True,
    ).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate, weight_decay=0.01)
    train_pairs, heldout_pairs = compositional_pair_split()
    rng = random.Random(args.seed)
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        student.train()
        batch = make_batch(
            args.batch_size,
            2,
            6,
            rng,
            digit_ids,
            allowed_pairs=train_pairs,
            mark_sinks=True,
        ).to(device)
        with torch.no_grad():
            target = teacher(batch.reference_values, batch.buggy_values)
        predicted = student_prefix(student, batch)
        mse = nn.functional.mse_loss(predicted, target)
        cosine = 1.0 - nn.functional.cosine_similarity(
            predicted[:, 0], target[:, 0], dim=-1
        ).mean()
        loss = mse + args.cosine_weight * cosine
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            print(json.dumps({
                "step": step,
                "loss": round(loss.item(), 6),
                "mse": round(mse.item(), 6),
                "cosine_similarity": round(1.0 - cosine.item(), 5),
            }), flush=True)

    seen_short = make_batch(
        args.eval_size,
        2,
        6,
        random.Random(args.seed + 10_000),
        digit_ids,
        allowed_pairs=train_pairs,
        mark_sinks=True,
    ).to(device)
    unseen_long = make_batch(
        args.eval_size,
        8,
        12,
        random.Random(args.seed + 20_000),
        digit_ids,
        allowed_pairs=heldout_pairs,
        mark_sinks=True,
    ).to(device)
    representation = {
        "seen_short": representation_metrics(student, teacher, seen_short),
        "unseen_long": representation_metrics(student, teacher, unseen_long),
    }
    prompts = {task: prompt_parts(tokenizer, device, task) for task in TASKS}
    heldout_prompts = {task: prompt_parts(tokenizer, device, task, True) for task in TASKS}
    receiver = {
        "student_seen_short": evaluate(
            model,
            student,
            prompts,
            heldout_prompts,
            seen_short,
            digit_ids,
            label_ids,
            "trace",
            minimal=True,
        ),
        "student_unseen_long": evaluate(
            model,
            student,
            prompts,
            heldout_prompts,
            unseen_long,
            digit_ids,
            label_ids,
            "trace",
            minimal=True,
        ),
        "teacher_unseen_ceiling": evaluate(
            model,
            teacher,
            prompts,
            heldout_prompts,
            unseen_long,
            digit_ids,
            label_ids,
            "pair_encoder",
            minimal=True,
        ),
    }
    result = {
        "model": args.model,
        "teacher_checkpoint": args.teacher_checkpoint,
        "seed": args.seed,
        "train_steps": args.steps,
        "train_lengths_per_trace": [2, 6],
        "eval_lengths_per_trace": [8, 12],
        "train_pair_count": len(train_pairs),
        "heldout_pair_count": len(heldout_pairs),
        "num_latents": 1,
        "student_trainable_parameters": trainable_parameter_count(student),
        "loss": "mse + cosine_weight * (1 - cosine)",
        "cosine_weight": args.cosine_weight,
        "elapsed_seconds": time.perf_counter() - started,
        "representation": representation,
        "receiver": receiver,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"adapter_trainable": trainable_state_dict(student), "result": result}, checkpoint)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
