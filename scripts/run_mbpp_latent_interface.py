#!/usr/bin/env python3
"""Align a semantic trace encoder to a frozen Qwen decoder with short labels."""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_mbpp_latent_repair import (
    COUNTERFACTUAL_KINDS,
    ROLE_FACTORIZED_SLOTS,
    NumericExample,
    _as_evidence,
    _counterfactual_evidence,
    _load_or_build_examples,
    _native_content_table,
)
from run_pilot1 import resolve_local_model
from run_repair_codebook import MARKER, splice
from trace2cache.latent import RoleAwareEventEncoder, trainable_parameter_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--dataset", default=".local/datasets/mbpp/mbpp.jsonl")
    parser.add_argument("--encoder-init", required=True)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--hidden-width", type=int, default=256)
    parser.add_argument("--latent-slots", type=int, default=8)
    parser.add_argument("--max-train-examples", type=int, default=1024)
    parser.add_argument("--max-eval-examples", type=int, default=192)
    parser.add_argument("--expanded-mutations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--candidate-mutants-per-task", type=int, default=32)
    parser.add_argument("--max-mutants-per-task", type=int, default=4)
    parser.add_argument("--fuzz-calls", type=int, default=16)
    parser.add_argument("--fuzz-bundles-per-mutant", type=int, default=1)
    parser.add_argument("--max-events-per-test", type=int, default=24)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--cache-dir", default=".local/cache/mbpp_latent_repair")
    parser.add_argument("--seed", type=int, default=307)
    parser.add_argument("--min-free-gib", type=float, default=24.0)
    parser.add_argument(
        "--output", default="artifacts/mbpp_repair/latent_interface_seed307.json"
    )
    parser.add_argument(
        "--checkpoint", default="checkpoints/mbpp_repair/latent_interface_seed307.pt"
    )
    return parser.parse_args()


def render(tokenizer) -> str:
    user = f"""A runtime execution has been supplied through the internal evidence channel.

Runtime evidence: {MARKER}

Is the evidence a coherent execution trace, with correctly bound event roles, values, and pass/fail test identity?
Answer exactly VALID or INVALID."""
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a precise runtime-evidence verifier."},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def numeric_examples(raw, vocabulary):
    content_to_id = {content: index for index, content in enumerate(vocabulary)}
    return [
        NumericExample(example, tuple(content_to_id[value] for value in example.contents))
        for example in raw
    ]


def encoder_inputs(view, table, device):
    ids = torch.tensor(view.content_ids, device=device).unsqueeze(0)
    roles = torch.tensor(view.roles, device=device).unsqueeze(0)
    tests = torch.tensor(view.test_ids, device=device).unsqueeze(0)
    return table[ids], roles, tests, torch.ones_like(roles, dtype=torch.bool)


def label_ids(tokenizer, label: str, device) -> torch.Tensor:
    ids = tokenizer(" " + label, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    eos = torch.tensor([[tokenizer.eos_token_id]], device=device)
    return torch.cat((ids, eos), dim=1)


def label_loss(model, tokenizer, encoder, rendered, view, table, label: str) -> torch.Tensor:
    contents, roles, tests, mask = encoder_inputs(view, table, model.device)
    latent = encoder(contents, roles, tests, mask).to(model.get_input_embeddings().weight.dtype)
    prompt_embeds = splice(model, tokenizer, rendered, latent)
    targets = label_ids(tokenizer, label, model.device)
    teacher = model.get_input_embeddings()(targets[:, :-1])
    inputs = torch.cat((prompt_embeds, teacher), dim=1)
    logits = model(inputs_embeds=inputs, use_cache=False).logits
    start = prompt_embeds.shape[1] - 1
    predicted = logits[:, start : start + targets.shape[1]].float()
    return nn.functional.cross_entropy(predicted.flatten(0, 1), targets.flatten())


@torch.inference_mode()
def evaluate(model, tokenizer, encoder, rendered, examples, table):
    correct = {"true": 0, **{kind: 0 for kind in COUNTERFACTUAL_KINDS}}
    total = {name: 0 for name in correct}
    true_over_corrupt = {kind: 0 for kind in COUNTERFACTUAL_KINDS}
    score_gaps = {kind: [] for kind in COUNTERFACTUAL_KINDS}
    for example in examples:
        views = {"true": _as_evidence(example)}
        views.update({kind: _counterfactual_evidence(example, kind) for kind in COUNTERFACTUAL_KINDS})
        scores = {}
        for name, view in views.items():
            valid = -label_loss(model, tokenizer, encoder, rendered, view, table, "VALID").item()
            invalid = -label_loss(model, tokenizer, encoder, rendered, view, table, "INVALID").item()
            score = valid - invalid
            scores[name] = score
            expected_valid = name == "true"
            correct[name] += int((score > 0) == expected_valid)
            total[name] += 1
        for kind in COUNTERFACTUAL_KINDS:
            gap = scores["true"] - scores[kind]
            score_gaps[kind].append(gap)
            true_over_corrupt[kind] += int(gap > 0)
    return {
        "view_accuracy": {name: correct[name] / total[name] for name in correct},
        "macro_view_accuracy": sum(correct[name] / total[name] for name in correct) / len(correct),
        "true_over_corruption_accuracy": {
            kind: true_over_corrupt[kind] / len(examples) for kind in COUNTERFACTUAL_KINDS
        },
        "mean_true_minus_corruption_score": {
            kind: sum(score_gaps[kind]) / len(score_gaps[kind]) for kind in COUNTERFACTUAL_KINDS
        },
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("A BF16 CUDA GPU is required; this pilot never falls back to CPU")
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes / 2**30 < args.min_free_gib:
        raise SystemExit(f"only {free_bytes / 2**30:.1f} GiB free")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_raw = _load_or_build_examples(args, "train")
    eval_raw = _load_or_build_examples(args, "validation")
    random.Random(args.seed).shuffle(train_raw)
    random.Random(args.seed + 1).shuffle(eval_raw)
    train_raw = train_raw[: args.max_train_examples]
    eval_raw = eval_raw[: args.max_eval_examples]
    vocabulary = sorted(
        {content for example in train_raw + eval_raw for content in example.contents}
    )

    model_path = resolve_local_model(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.bfloat16
    ).to("cuda").eval()
    model.requires_grad_(False)
    model.config.use_cache = False
    table = _native_content_table(model, tokenizer, vocabulary)
    anchor_ids = tokenizer(
        " runtime evidence observed behavior state values result details",
        add_special_tokens=False,
    ).input_ids[: args.latent_slots]
    anchor_ids += [anchor_ids[-1]] * (args.latent_slots - len(anchor_ids))
    anchors = model.get_input_embeddings()(
        torch.tensor(anchor_ids, device=model.device)
    ).detach()
    encoder = RoleAwareEventEncoder(
        model_width=model.config.hidden_size,
        output_anchor=anchors,
        hidden_width=args.hidden_width,
        num_roles=8,
        max_events=args.max_events_per_test * 2 + 4,
        max_tests=2,
        slot_roles=ROLE_FACTORIZED_SLOTS,
    ).to(model.device)
    state = torch.load(args.encoder_init, map_location="cpu", weights_only=True)
    encoder.load_state_dict(state["encoder"], strict=True)
    train = numeric_examples(train_raw, vocabulary)
    evaluation = numeric_examples(eval_raw, vocabulary)
    rendered = render(tokenizer)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.learning_rate, weight_decay=0.01)
    rng = random.Random(args.seed)
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        example = train[rng.randrange(len(train))]
        if rng.randrange(len(COUNTERFACTUAL_KINDS) + 1) == 0:
            view, label, name = _as_evidence(example), "VALID", "true"
        else:
            name = COUNTERFACTUAL_KINDS[rng.randrange(len(COUNTERFACTUAL_KINDS))]
            view, label = _counterfactual_evidence(example, name), "INVALID"
        loss = label_loss(model, tokenizer, encoder, rendered, view, table, label)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 50 == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "label": name,
                        "loss": round(loss.item(), 5),
                        "gpu_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
                    }
                ),
                flush=True,
            )
    metrics = evaluate(model, tokenizer, encoder, rendered, evaluation, table)
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "seed": args.seed,
        "steps": args.steps,
        "train_examples": len(train),
        "train_tasks": len({example.example.task_id for example in train}),
        "eval_examples": len(evaluation),
        "eval_tasks": len({example.example.task_id for example in evaluation}),
        "encoder_init": args.encoder_init,
        "trainable_parameters": trainable_parameter_count(encoder),
        "training_seconds": time.perf_counter() - started,
        "gpu_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
        "metrics": metrics,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder": {name: tensor.detach().cpu() for name, tensor in encoder.state_dict().items()},
            "config": vars(args),
            "result": result,
        },
        checkpoint,
    )
    print(json.dumps({"event": "final", **metrics, "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
