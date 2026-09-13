#!/usr/bin/env python3
"""Causal-path gate: one latent must combine branch selectors and definitions."""

from __future__ import annotations

import argparse
from dataclasses import replace
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
    PairBatch,
    TASKS,
    compositional_pair_split,
    evaluate,
    one_token_ids,
    prompt_parts,
    resolve_local_model,
    trainable_state_dict,
)
from run_trace_distillation import representation_metrics, student_prefix


DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
# Event roles. Candidate values and selector states are stored in the event state field.
REF_FALSE, REF_TRUE, REF_BRANCH = 0, 1, 2
BUG_FALSE, BUG_TRUE, BUG_BRANCH = 3, 4, 5
REF_DISTRACTOR, BUG_DISTRACTOR = 6, 7
REF_MASKED_SINK, BUG_MASKED_SINK = 8, 9


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
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument(
        "--output", default="artifacts/latent_probe/causal_path_distillation_seed83.json"
    )
    parser.add_argument(
        "--checkpoint", default="checkpoints/latent_probe/causal_path_distillation_seed83.pt"
    )
    return parser.parse_args()


def _causal_run_events(
    *,
    target: int,
    buggy: bool,
    distractor_count: int,
    rng: random.Random,
) -> tuple[list[tuple[int, int, int]], list[str]]:
    selector = rng.randrange(2)
    other = rng.choice([value for value in range(10) if value != target])
    false_value, true_value = (other, target) if selector else (target, other)
    offset = 3 if buggy else 0
    label = "BUGGY" if buggy else "REFERENCE"
    core = [
        (REF_FALSE + offset, rng.randrange(10), false_value),
        (REF_TRUE + offset, rng.randrange(10), true_value),
        (REF_BRANCH + offset, rng.randrange(10), selector),
    ]
    distractor_type = BUG_DISTRACTOR if buggy else REF_DISTRACTOR
    distractors = [
        (distractor_type, rng.randrange(10), rng.randrange(10))
        for _ in range(distractor_count)
    ]
    # Random interleaving prevents fixed positions from identifying the causal fields.
    body = core + distractors
    rng.shuffle(body)
    sink_type = BUG_MASKED_SINK if buggy else REF_MASKED_SINK
    body.append((sink_type, rng.randrange(10), 0))  # runtime output deliberately masked
    lines = [
        f"{label}: false_candidate={false_value}, true_candidate={true_value}, "
        f"branch={selector}, masked_sink=?, target={target}"
    ]
    return body, lines


def make_causal_batch(
    *,
    batch_size: int,
    min_distractors: int,
    max_distractors: int,
    rng: random.Random,
    digit_ids: dict[str, int],
    allowed_pairs: list[tuple[int, int]],
    max_events: int = 64,
) -> PairBatch:
    event_types = torch.zeros(batch_size, max_events, dtype=torch.long)
    arguments = torch.full((batch_size, max_events), digit_ids["0"], dtype=torch.long)
    states = torch.full((batch_size, max_events), digit_ids["0"], dtype=torch.long)
    event_mask = torch.zeros(batch_size, max_events, dtype=torch.bool)
    reference_values = torch.empty(batch_size, dtype=torch.long)
    buggy_values = torch.empty(batch_size, dtype=torch.long)
    traces = []
    for row in range(batch_size):
        reference_target, buggy_target = rng.choice(allowed_pairs)
        total_distractors = rng.randint(min_distractors, max_distractors)
        reference_distractors = rng.randint(0, total_distractors)
        buggy_distractors = total_distractors - reference_distractors
        reference_events, reference_lines = _causal_run_events(
            target=reference_target,
            buggy=False,
            distractor_count=reference_distractors,
            rng=rng,
        )
        buggy_events, buggy_lines = _causal_run_events(
            target=buggy_target,
            buggy=True,
            distractor_count=buggy_distractors,
            rng=rng,
        )
        combined = reference_events + buggy_events
        if len(combined) > max_events:
            raise ValueError("causal trace exceeds max_events")
        for column, (kind, argument, state) in enumerate(combined):
            event_types[row, column] = kind
            arguments[row, column] = digit_ids[str(argument)]
            states[row, column] = digit_ids[str(state)]
            event_mask[row, column] = True
        reference_values[row] = reference_target
        buggy_values[row] = buggy_target
        traces.append("\n".join(reference_lines + buggy_lines))
    digit_lookup = torch.tensor([digit_ids[str(i)] for i in range(10)])
    return PairBatch(
        event_types,
        arguments,
        states,
        event_mask,
        digit_lookup[reference_values],
        digit_lookup[buggy_values],
        reference_values,
        buggy_values,
        traces,
    )


def flip_branch_counterfactual(batch: PairBatch, digit_ids: dict[str, int]) -> PairBatch:
    states = batch.states.clone()
    branch_mask = (batch.event_types == REF_BRANCH) | (batch.event_types == BUG_BRANCH)
    zero_id, one_id = digit_ids["0"], digit_ids["1"]
    original = states[branch_mask]
    states[branch_mask] = torch.where(
        original == zero_id,
        torch.tensor(one_id, device=states.device),
        torch.tensor(zero_id, device=states.device),
    )
    return replace(batch, states=states)


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
    payload = torch.load(args.teacher_checkpoint, map_location=device, weights_only=False)
    missing, unexpected = teacher.load_state_dict(payload["adapter_trainable"], strict=False)
    allowed_missing = {"token_embedding.weight", "digit_token_ids"}
    if unexpected or any(name not in allowed_missing for name in missing):
        raise RuntimeError(f"teacher mismatch: missing={missing}, unexpected={unexpected}")
    teacher.requires_grad_(False)

    width = model.get_input_embeddings().embedding_dim
    student = NativeEventResampler(
        model.get_input_embeddings(),
        model_width=width,
        hidden_width=args.hidden_width,
        num_latents=1,
        num_event_types=10,
        max_events=64,
        zero_output_init=True,
        fixed_positions=True,
    ).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate, weight_decay=0.01)
    train_pairs, heldout_pairs = compositional_pair_split()
    rng = random.Random(args.seed)
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        student.train()
        batch = make_causal_batch(
            batch_size=args.batch_size,
            min_distractors=0,
            max_distractors=6,
            rng=rng,
            digit_ids=digit_ids,
            allowed_pairs=train_pairs,
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

    seen_short = make_causal_batch(
        batch_size=args.eval_size,
        min_distractors=0,
        max_distractors=6,
        rng=random.Random(args.seed + 10_000),
        digit_ids=digit_ids,
        allowed_pairs=train_pairs,
    ).to(device)
    unseen_distractors = make_causal_batch(
        batch_size=args.eval_size,
        min_distractors=16,
        max_distractors=40,
        rng=random.Random(args.seed + 20_000),
        digit_ids=digit_ids,
        allowed_pairs=heldout_pairs,
    ).to(device)
    counterfactual = flip_branch_counterfactual(unseen_distractors, digit_ids)
    prompts = {task: prompt_parts(tokenizer, device, task) for task in TASKS}
    heldout_prompts = {task: prompt_parts(tokenizer, device, task, True) for task in TASKS}
    result = {
        "model": args.model,
        "teacher_checkpoint": args.teacher_checkpoint,
        "seed": args.seed,
        "train_steps": args.steps,
        "train_distractors": [0, 6],
        "eval_distractors": [16, 40],
        "train_pair_count": len(train_pairs),
        "heldout_pair_count": len(heldout_pairs),
        "num_latents": 1,
        "student_trainable_parameters": trainable_parameter_count(student),
        "elapsed_seconds": time.perf_counter() - started,
        "representation": {
            "seen_short": representation_metrics(student, teacher, seen_short),
            "unseen_many_distractors": representation_metrics(student, teacher, unseen_distractors),
        },
        "receiver": {
            "student_seen_short": evaluate(
                model, student, prompts, heldout_prompts, seen_short,
                digit_ids, label_ids, "trace", minimal=True
            ),
            "student_unseen_many_distractors": evaluate(
                model, student, prompts, heldout_prompts, unseen_distractors,
                digit_ids, label_ids, "trace", minimal=True
            ),
            "branch_flip_counterfactual": evaluate(
                model, student, prompts, heldout_prompts, counterfactual,
                digit_ids, label_ids, "trace", minimal=True
            ),
            "teacher_unseen_ceiling": evaluate(
                model, teacher, prompts, heldout_prompts, unseen_distractors,
                digit_ids, label_ids, "pair_encoder", minimal=True
            ),
        },
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
