#!/usr/bin/env python3
"""Compress two execution traces into one receiver-native latent state.

The same query-independent state must support several questions, preventing the
encoder from merely emitting the answer to one fixed classification problem.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import time

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from trace2cache.latent import (
    NativeEventResampler,
    NativePairCodebook,
    NativePairEncoder,
    trainable_parameter_count,
)


DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MARKER = "__LATENT_PAIR__"
TASKS = ("reference", "buggy", "delta", "greater")
TRAIN_QUESTIONS = {
    "reference": "What is the reference execution's final digit? Answer with the integer only.",
    "buggy": "What is the buggy execution's final digit? Answer with the integer only.",
    "delta": "What is (buggy final - reference final) modulo 10? Answer with the integer only.",
    "greater": "Is the buggy final value greater than the reference final value? Answer Yes or No only.",
}
TRAIN_QUESTION_VARIANTS = {
    "reference": (
        TRAIN_QUESTIONS["reference"],
        "Give the final digit from the reference run. Output one integer only.",
        "Which digit did the correct execution finish with? Respond with only that digit.",
    ),
    "buggy": (
        TRAIN_QUESTIONS["buggy"],
        "Give the final digit from the buggy run. Output one integer only.",
        "Which digit did the failing execution finish with? Respond with only that digit.",
    ),
    "delta": (
        TRAIN_QUESTIONS["delta"],
        "Compute buggy minus reference modulo 10. Output one integer only.",
        "Give the modulo-ten difference from the correct result to the failing result. One digit only.",
    ),
    "greater": (
        TRAIN_QUESTIONS["greater"],
        "Is buggy final strictly larger than reference final? Answer Yes or No only.",
        "Did the failing execution finish with a greater value than the correct one? Reply Yes or No only.",
    ),
}
HELDOUT_QUESTIONS = {
    "reference": "Return only the final digit produced by the correct reference run.",
    "buggy": "Return only the final digit produced by the failing buggy run.",
    "delta": "Subtract the reference result from the buggy result, wrap modulo ten, and return only that digit.",
    "greater": "Does the failing run end above the reference run? Reply Yes or No only.",
}


@dataclass
class PairBatch:
    event_types: torch.Tensor
    arguments: torch.Tensor
    states: torch.Tensor
    event_mask: torch.Tensor
    reference_ids: torch.Tensor
    buggy_ids: torch.Tensor
    reference_values: torch.Tensor
    buggy_values: torch.Tensor
    text_traces: list[str]

    def to(self, device: str) -> "PairBatch":
        return PairBatch(
            self.event_types.to(device),
            self.arguments.to(device),
            self.states.to(device),
            self.event_mask.to(device),
            self.reference_ids.to(device),
            self.buggy_ids.to(device),
            self.reference_values.to(device),
            self.buggy_values.to(device),
            self.text_traces,
        )


def resolve_local_model(model: str) -> str:
    direct = Path(model)
    if direct.exists():
        return str(direct.resolve())
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    cache = hf_home / "hub" / ("models--" + model.replace("/", "--"))
    candidates: list[Path] = []
    ref = cache / "refs" / "main"
    if ref.exists():
        candidates.append(cache / "snapshots" / ref.read_text().strip())
    snapshots = cache / "snapshots"
    if snapshots.exists():
        candidates.extend(sorted(snapshots.iterdir()))
    for candidate in candidates:
        if (candidate / "config.json").exists():
            return str(candidate)
    raise FileNotFoundError(f"model unavailable locally: {model}")


def one_token_ids(tokenizer, strings: list[str]) -> dict[str, int]:
    result = {}
    for string in strings:
        encoded = tokenizer.encode(string, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"{string!r} is not one token: {encoded}")
        result[string] = encoded[0]
    return result


def _run_trace(
    rng: random.Random, steps: int, target_value: int | None = None
) -> tuple[list[tuple[int, int, int]], int, list[str]]:
    operations = [(rng.choice((1, 2)), rng.randrange(1, 10)) for _ in range(steps)]
    signed_total = sum(argument if operation == 1 else -argument for operation, argument in operations)
    value = rng.randrange(10) if target_value is None else (target_value - signed_total) % 10
    events = [(0, value, value)]
    lines = [f"init value={value}"]
    for index, (operation, argument) in enumerate(operations, start=1):
        if operation == 1:
            value = (value + argument) % 10
            name = "add"
        else:
            value = (value - argument) % 10
            name = "subtract"
        events.append((operation, argument, value))
        lines.append(f"step {index}: {name} {argument}; value={value}")
    return events, value, lines


def make_batch(
    batch_size: int,
    min_steps: int,
    max_steps: int,
    rng: random.Random,
    digit_ids: dict[str, int],
    max_events: int = 32,
    allowed_pairs: list[tuple[int, int]] | None = None,
    mark_sinks: bool = False,
) -> PairBatch:
    types = torch.zeros(batch_size, max_events, dtype=torch.long)
    arguments = torch.full((batch_size, max_events), digit_ids["0"], dtype=torch.long)
    states = torch.full((batch_size, max_events), digit_ids["0"], dtype=torch.long)
    mask = torch.zeros(batch_size, max_events, dtype=torch.bool)
    reference_values = torch.empty(batch_size, dtype=torch.long)
    buggy_values = torch.empty(batch_size, dtype=torch.long)
    traces = []
    for row in range(batch_size):
        target_pair = rng.choice(allowed_pairs) if allowed_pairs else (None, None)
        ref_events, ref_value, ref_lines = _run_trace(
            rng, rng.randint(min_steps, max_steps), target_pair[0]
        )
        bug_events, bug_value, bug_lines = _run_trace(
            rng, rng.randint(min_steps, max_steps), target_pair[1]
        )
        # Roles are encoded into event types: 0..2 reference, 3..5 buggy.
        combined = ref_events + [(kind + 3, arg, state) for kind, arg, state in bug_events]
        if mark_sinks:
            reference_sink = len(ref_events) - 1
            buggy_sink = len(combined) - 1
            for sink_index in (reference_sink, buggy_sink):
                kind, argument, state = combined[sink_index]
                combined[sink_index] = (kind + 6, argument, state)
        if len(combined) > max_events:
            raise ValueError("trace pair exceeds max_events")
        for column, (kind, argument, state) in enumerate(combined):
            types[row, column] = kind
            arguments[row, column] = digit_ids[str(argument)]
            states[row, column] = digit_ids[str(state)]
            mask[row, column] = True
        reference_values[row] = ref_value
        buggy_values[row] = bug_value
        traces.append(
            "REFERENCE RUN:\n"
            + "\n".join(ref_lines)
            + "\nBUGGY RUN:\n"
            + "\n".join(bug_lines)
        )
    reference_ids = torch.tensor([digit_ids[str(x)] for x in reference_values.tolist()])
    buggy_ids = torch.tensor([digit_ids[str(x)] for x in buggy_values.tolist()])
    return PairBatch(
        types,
        arguments,
        states,
        mask,
        reference_ids,
        buggy_ids,
        reference_values,
        buggy_values,
        traces,
    )


def targets_for(batch: PairBatch, task: str, digit_ids: dict[str, int], label_ids: dict[str, int]) -> torch.Tensor:
    if task == "reference":
        return batch.reference_ids
    if task == "buggy":
        return batch.buggy_ids
    if task == "delta":
        values = (batch.buggy_values - batch.reference_values) % 10
        mapping = torch.tensor([digit_ids[str(i)] for i in range(10)], device=values.device)
        return mapping[values]
    if task == "greater":
        return torch.where(
            batch.buggy_values > batch.reference_values,
            torch.tensor(label_ids["Yes"], device=batch.buggy_values.device),
            torch.tensor(label_ids["No"], device=batch.buggy_values.device),
        )
    raise ValueError(task)


def prompt_parts(
    tokenizer, device: str, task: str, heldout: bool = False, question: str | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    questions = HELDOUT_QUESTIONS if heldout else TRAIN_QUESTIONS
    user = (
        "One hidden runtime state jointly encodes a correct reference execution and a buggy execution.\n"
        "Compressed reference/buggy state: " + MARKER + "\n" + (question or questions[task])
    )
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You answer execution questions precisely."},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    before, after = rendered.split(MARKER)
    return (
        tokenizer(before, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device),
        tokenizer(after, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device),
    )


def splice(model, parts, prefix: torch.Tensor) -> torch.Tensor:
    embedding = model.get_input_embeddings()
    before = embedding(parts[0]).expand(prefix.shape[0], -1, -1)
    after = embedding(parts[1]).expand(prefix.shape[0], -1, -1)
    return torch.cat((before, prefix.to(before.dtype), after), dim=1)


def predict(model, parts, prefix: torch.Tensor) -> torch.Tensor:
    inputs = splice(model, parts, prefix)
    mask = torch.ones(inputs.shape[:2], dtype=torch.long, device=inputs.device)
    return model(inputs_embeds=inputs, attention_mask=mask, use_cache=False).logits[:, -1]


def latent_prefix(adapter, batch: PairBatch, compressor: str) -> torch.Tensor:
    # Reference final is the understood native anchor; the residual must pack the buggy outcome.
    if compressor in ("codebook", "pair_encoder"):
        return adapter(batch.reference_values, batch.buggy_values)
    return adapter(
        batch.event_types,
        batch.arguments,
        batch.states,
        batch.event_mask,
        anchor_token_ids=batch.reference_ids,
    )


@torch.no_grad()
def evaluate(
    model,
    adapter,
    prompts,
    heldout_prompts,
    batch: PairBatch,
    digit_ids: dict[str, int],
    label_ids: dict[str, int],
    compressor: str,
    minimal: bool = False,
) -> dict[str, object]:
    adapter.eval()
    true = latent_prefix(adapter, batch, compressor)
    shuffled = true.roll(1, 0)
    embedding = model.get_input_embeddings()
    prefixes = {"latent": true, "shuffled": shuffled}
    if not minimal:
        reference = embedding(batch.reference_ids).unsqueeze(1)
        buggy = embedding(batch.buggy_ids).unsqueeze(1)
        prefixes.update({
            "reference_anchor": reference,
            "buggy_anchor": buggy,
            "two_native_tokens": torch.cat((reference, buggy), dim=1),
        })

    def accuracy(parts, prefix: torch.Tensor, targets: torch.Tensor, chunk_size: int = 32) -> float:
        correct = 0
        for start in range(0, targets.shape[0], chunk_size):
            stop = start + chunk_size
            predictions = predict(model, parts, prefix[start:stop]).argmax(-1)
            correct += int((predictions == targets[start:stop]).sum())
        return correct / targets.shape[0]

    metrics: dict[str, object] = {}
    for heldout, prompt_map, suffix in (
        (False, prompts, ""),
        (True, heldout_prompts, "_heldout_prompt"),
    ):
        del heldout
        task_metrics = {}
        for task in TASKS:
            targets = targets_for(batch, task, digit_ids, label_ids)
            task_metrics[task] = {
                name: accuracy(prompt_map[task], prefix, targets)
                for name, prefix in prefixes.items()
            }
        metrics["tasks" + suffix] = task_metrics
        for method in prefixes:
            metrics["macro_" + method + suffix] = sum(task_metrics[t][method] for t in TASKS) / len(TASKS)
    return metrics


@torch.no_grad()
def evaluate_text(
    model,
    tokenizer,
    batch: PairBatch,
    digit_ids,
    label_ids,
    device: str,
    *,
    questions: dict[str, str],
    compact: bool,
) -> dict[str, float]:
    result = {}
    for task in TASKS:
        rendered = []
        for index, trace in enumerate(batch.text_traces):
            evidence = (
                f"REFERENCE final = {batch.reference_values[index].item()}\n"
                f"BUGGY final = {batch.buggy_values[index].item()}"
                if compact
                else trace
            )
            user = evidence + "\n\n" + questions[task]
            rendered.append(tokenizer.apply_chat_template(
                [{"role": "system", "content": "You answer execution questions precisely."}, {"role": "user", "content": user}],
                tokenize=False,
                add_generation_prompt=True,
            ))
        correct = 0
        for start in range(0, len(rendered), 32):
            old_side = tokenizer.padding_side
            tokenizer.padding_side = "left"
            encoded = tokenizer(rendered[start:start + 32], return_tensors="pt", padding=True).to(device)
            tokenizer.padding_side = old_side
            predictions = model(**encoded, use_cache=False).logits[:, -1].argmax(-1)
            targets = targets_for(batch, task, digit_ids, label_ids)[start:start + 32]
            correct += int((predictions == targets).sum())
        result[task] = correct / len(rendered)
    result["macro"] = sum(result.values()) / len(TASKS)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-size", type=int, default=256)
    parser.add_argument("--hidden-width", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--compressor", choices=("trace", "codebook", "pair_encoder"), default="trace"
    )
    parser.add_argument("--prompt-augmentation", action="store_true")
    parser.add_argument("--compositional-split", action="store_true")
    parser.add_argument("--minimal-eval", action="store_true")
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--output", default="artifacts/latent_probe/multifact_k1.json")
    parser.add_argument("--checkpoint", default="checkpoints/latent_probe/multifact_k1.pt")
    return parser.parse_args()


def trainable_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    """Avoid duplicating the frozen receiver embedding table in adapter checkpoints."""
    return {
        name: parameter.detach().cpu()
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }


def residual_scale(adapter: nn.Module, compressor: str) -> float:
    if compressor == "codebook":
        return adapter.residual.weight.norm(dim=1).mean().item()
    if compressor == "pair_encoder":
        return adapter.residual[-1].weight.norm(dim=1).mean().item()
    return adapter.residual_gate.item()


def compositional_pair_split() -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Return 80/20 pairs balanced over each input digit and modulo delta."""
    all_pairs = [(reference, buggy) for reference in range(10) for buggy in range(10)]
    heldout = [
        (reference, buggy)
        for reference, buggy in all_pairs
        if buggy in ((3 * reference) % 10, (3 * reference + 1) % 10)
    ]
    train = [pair for pair in all_pairs if pair not in heldout]
    for values in (
        [reference for reference, _ in heldout],
        [buggy for _, buggy in heldout],
        [(buggy - reference) % 10 for reference, buggy in heldout],
    ):
        if Counter(values) != Counter({digit: 2 for digit in range(10)}):
            raise AssertionError("compositional holdout is not balanced")
    return train, heldout


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
    width = model.get_input_embeddings().embedding_dim
    if args.compressor == "codebook":
        adapter = NativePairCodebook(
            model.get_input_embeddings(), [digit_ids[str(i)] for i in range(10)]
        ).to(device)
    elif args.compressor == "pair_encoder":
        adapter = NativePairEncoder(
            model.get_input_embeddings(),
            [digit_ids[str(i)] for i in range(10)],
            hidden_width=args.hidden_width,
        ).to(device)
    else:
        adapter = NativeEventResampler(
            model.get_input_embeddings(),
            model_width=width,
            hidden_width=args.hidden_width,
            num_latents=1,
            num_event_types=6,
            max_events=32,
            native_sink_anchor=True,
        ).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.learning_rate, weight_decay=0.01)
    prompts = {task: prompt_parts(tokenizer, device, task) for task in TASKS}
    augmented_prompts = {
        task: [prompt_parts(tokenizer, device, task, question=question) for question in TRAIN_QUESTION_VARIANTS[task]]
        for task in TASKS
    }
    heldout_prompts = {task: prompt_parts(tokenizer, device, task, True) for task in TASKS}
    rng = random.Random(args.seed)
    train_pairs, heldout_pairs = compositional_pair_split()
    allowed_train_pairs = train_pairs if args.compositional_split else None
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        task = TASKS[(step - 1) % len(TASKS)]
        variant = ((step - 1) // len(TASKS)) % len(TRAIN_QUESTION_VARIANTS[task])
        train_prompt = augmented_prompts[task][variant] if args.prompt_augmentation else prompts[task]
        batch = make_batch(
            args.batch_size, 2, 6, rng, digit_ids, allowed_pairs=allowed_train_pairs
        ).to(device)
        targets = targets_for(batch, task, digit_ids, label_ids)
        logits = predict(model, train_prompt, latent_prefix(adapter, batch, args.compressor))
        loss = nn.functional.cross_entropy(logits.float(), targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 25 == 0:
            accuracy = (logits.argmax(-1) == targets).float().mean().item()
            print(json.dumps({
                "step": step,
                "task": task,
                "loss": round(loss.item(), 4),
                "train_accuracy": accuracy,
                "residual_norm": round(residual_scale(adapter, args.compressor), 5),
            }), flush=True)
    evaluation = make_batch(
        args.eval_size,
        8,
        12,
        random.Random(args.seed + 10_000),
        digit_ids,
        allowed_pairs=train_pairs if args.compositional_split else None,
    ).to(device)
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    # Preserve the trained adapter even if a later baseline evaluation is interrupted.
    torch.save({"adapter_trainable": trainable_state_dict(adapter), "args": vars(args)}, checkpoint)
    metrics = evaluate(
        model,
        adapter,
        prompts,
        heldout_prompts,
        evaluation,
        digit_ids,
        label_ids,
        args.compressor,
        args.minimal_eval,
    )
    if args.compositional_split:
        unseen_evaluation = make_batch(
            args.eval_size,
            8,
            12,
            random.Random(args.seed + 20_000),
            digit_ids,
            allowed_pairs=heldout_pairs,
        ).to(device)
        metrics = {
            "seen_pairs": metrics,
            "unseen_pairs": evaluate(
                model,
                adapter,
                prompts,
                heldout_prompts,
                unseen_evaluation,
                digit_ids,
                label_ids,
                args.compressor,
                args.minimal_eval,
            ),
        }
    baseline_evaluation = unseen_evaluation if args.compositional_split else evaluation
    if not args.minimal_eval:
        metrics["compact_text"] = evaluate_text(
        model,
        tokenizer,
        baseline_evaluation,
        digit_ids,
        label_ids,
        device,
        questions=TRAIN_QUESTIONS,
        compact=True,
    )
        metrics["compact_text_heldout_prompt"] = evaluate_text(
        model,
        tokenizer,
        baseline_evaluation,
        digit_ids,
        label_ids,
        device,
        questions=HELDOUT_QUESTIONS,
        compact=True,
    )
        metrics["full_text_trace"] = evaluate_text(
        model,
        tokenizer,
        baseline_evaluation,
        digit_ids,
        label_ids,
        device,
        questions=TRAIN_QUESTIONS,
        compact=False,
    )
    result = {
        "model": args.model,
        "seed": args.seed,
        "train_steps": args.steps,
        "train_lengths_per_trace": [2, 6],
        "eval_lengths_per_trace": [8, 12],
        "eval_size": args.eval_size,
        "num_latents": 1,
        "compressor": args.compressor,
        "prompt_augmentation": args.prompt_augmentation,
        "compositional_split": args.compositional_split,
        "minimal_eval": args.minimal_eval,
        "train_pair_count": len(train_pairs) if args.compositional_split else 100,
        "heldout_pair_count": len(heldout_pairs) if args.compositional_split else 0,
        "tasks": list(TASKS),
        "trainable_parameters": trainable_parameter_count(adapter),
        "elapsed_seconds": time.perf_counter() - started,
        "residual_scale": residual_scale(adapter, args.compressor),
        **metrics,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    torch.save({"adapter_trainable": trainable_state_dict(adapter), "result": result}, checkpoint)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
