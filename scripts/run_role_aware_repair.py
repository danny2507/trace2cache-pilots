#!/usr/bin/env python3
"""Distill multi-test runtime evidence into receiver-readable repair codes."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from trace2cache.ambiguous_repair import AmbiguousRepairCase, get_ambiguous_cases
from trace2cache.latent import RoleAwareEventEncoder, trainable_parameter_count
from trace2cache.runtime import execute_with_trace
from trace2cache.sandbox import evaluate_patch, extract_function
from run_pilot1 import resolve_local_model
from run_repair_codebook import (
    MARKER,
    NativeRepairCodebook,
    greedy_generate,
    initial_native_codes,
    render,
    splice,
)


TEST_START, INPUT, CALL, LINE, BRANCH, DEFINITION, RETURN, ACTUAL, EXPECTED, STATUS = range(10)


@dataclass
class EvidenceExample:
    label: int
    contents: list[str]
    roles: list[int]
    test_ids: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument(
        "--codebook-checkpoint",
        default="checkpoints/repair_latent/repair_codebook_seed131.pt",
    )
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-width", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--cosine-weight", type=float, default=0.1)
    parser.add_argument("--train-examples", type=int, default=512)
    parser.add_argument("--eval-examples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=151)
    parser.add_argument("--output", default="artifacts/repair_latent/role_aware_repair_seed151.json")
    parser.add_argument(
        "--checkpoint", default="checkpoints/repair_latent/role_aware_repair_seed151.pt"
    )
    return parser.parse_args()


def expected_for(item: AmbiguousRepairCase, args: list[object]) -> object:
    case_id = item.case.case_id
    values = args[0]
    if case_id == "locate_first":
        return values.index(args[1])
    if case_id == "locate_last":
        return len(values) - 1 - list(reversed(values)).index(args[1])
    if case_id == "aggregate_sum":
        return sum(values)
    if case_id == "aggregate_product":
        result = 1
        for value in values:
            result *= value
        return result
    if case_id == "extreme_max":
        return max(values)
    if case_id == "extreme_min":
        return min(values)
    if case_id == "measure_even":
        return sum(value % 2 == 0 for value in values)
    if case_id == "measure_positive":
        return sum(value > 0 for value in values)
    raise ValueError(case_id)


def sample_args(
    item: AmbiguousRepairCase, rng: random.Random, *, heldout_values: bool
) -> list[object]:
    pair = item.pair_id
    while True:
        if pair == "locate":
            target = rng.randint(6, 12) if heldout_values else rng.randint(0, 5)
            length = rng.randint(6, 9) if heldout_values else rng.randint(3, 5)
            values = [
                rng.randint(-4, 14) if heldout_values else rng.randint(0, 5)
                for _ in range(length)
            ]
            first, last = sorted(rng.sample(range(length), 2))
            values[first] = target
            values[last] = target
            return [values, target]
        if pair == "aggregate":
            length = rng.randint(5, 7) if heldout_values else rng.randint(2, 4)
            low, high = (-2, 5) if heldout_values else (1, 3)
            values = [rng.randint(low, high) for _ in range(length)]
            product = 1
            for value in values:
                product *= value
            if sum(values) != product:
                return [values]
        elif pair == "extreme":
            length = rng.randint(6, 9) if heldout_values else rng.randint(3, 5)
            low, high = (-15, 15) if heldout_values else (-5, 5)
            values = [rng.randint(low, high) for _ in range(length)]
            if min(values) != max(values):
                return [values]
        elif pair == "measure":
            length = rng.randint(7, 10) if heldout_values else rng.randint(3, 6)
            low, high = (-12, 12) if heldout_values else (-4, 4)
            values = [rng.randint(low, high) for _ in range(length)]
            even = sum(value % 2 == 0 for value in values)
            positive = sum(value > 0 for value in values)
            if even != positive:
                return [values]


def input_events(args: list[object]) -> list[str]:
    events = []
    for argument_index, argument in enumerate(args):
        if isinstance(argument, (list, tuple)):
            events.append(f"argument {argument_index} length {len(argument)}")
            events.extend(
                f"argument {argument_index} position {index} value {value}"
                for index, value in enumerate(argument)
            )
        else:
            events.append(f"argument {argument_index} value {argument}")
    return events


def trace_role(event, changed: bool) -> int:
    if event.event == "call":
        return CALL
    if event.event in {"return", "exception"}:
        return RETURN
    if event.source.strip().startswith(("if ", "elif ", "for ", "while ")):
        return BRANCH
    return DEFINITION if changed else LINE


def one_test_events(item: AmbiguousRepairCase, args: list[object], test_id: int):
    expected = expected_for(item, args)
    trace = execute_with_trace(item.case.buggy_source, item.case.function_name, args)
    contents = [f"test {test_id} begins"]
    roles = [TEST_START]
    for content in input_events(args):
        contents.append(content)
        roles.append(INPUT)
    previous = {}
    candidates = []
    for event in trace.events:
        changed = event.locals != previous
        state = ", ".join(f"{name}={value}" for name, value in sorted(event.locals.items()))
        value = f" outcome {event.value}" if event.value is not None else ""
        candidates.append(
            (f"line {event.line} {event.source}; state {state}{value}", trace_role(event, changed))
        )
        previous = event.locals
    # Fixed per-test budget with temporal coverage, always including call/return endpoints.
    if len(candidates) > 6:
        indices = [round(i * (len(candidates) - 1) / 5) for i in range(6)]
        candidates = [candidates[index] for index in indices]
    for content, role in candidates:
        contents.append(content)
        roles.append(role)
    actual = trace.error or trace.output or "None"
    contents.extend((f"actual output {actual}", f"expected output {expected}", "test failed"))
    roles.extend((ACTUAL, EXPECTED, STATUS))
    return contents, roles


def make_example(
    item: AmbiguousRepairCase,
    label: int,
    rng: random.Random,
    *,
    long_bundle: bool,
    heldout_values: bool,
) -> EvidenceExample:
    test_count = rng.randint(5, 8) if long_bundle else rng.randint(2, 4)
    contents, roles, test_ids = [], [], []
    for test_id in range(test_count):
        test_contents, test_roles = one_test_events(
            item, sample_args(item, rng, heldout_values=heldout_values), test_id
        )
        contents.extend(test_contents)
        roles.extend(test_roles)
        test_ids.extend([test_id] * len(test_contents))
    return EvidenceExample(label, contents, roles, test_ids)


def make_dataset(
    items, size: int, seed: int, *, long_bundle: bool, heldout_values: bool
):
    rng = random.Random(seed)
    return [
        make_example(
            items[index % len(items)],
            index % len(items),
            rng,
            long_bundle=long_bundle,
            heldout_values=heldout_values,
        )
        for index in range(size)
    ]


def native_content_table(model, tokenizer, vocabulary: list[str]) -> torch.Tensor:
    embedding = model.get_input_embeddings()
    vectors = []
    with torch.no_grad():
        for content in vocabulary:
            ids = tokenizer(content, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
                model.device
            )
            vectors.append(embedding(ids)[0].float().mean(dim=0))
    return torch.stack(vectors)


def numerical_examples(examples, content_to_id):
    return [
        EvidenceExample(
            example.label,
            [content_to_id[content] for content in example.contents],
            example.roles,
            example.test_ids,
        )
        for example in examples
    ]


def batch_tensors(examples, table, device, *, remove_expected: bool = False):
    width = max(len(example.contents) for example in examples)
    batch = len(examples)
    content_ids = torch.zeros(batch, width, dtype=torch.long, device=device)
    roles = torch.zeros(batch, width, dtype=torch.long, device=device)
    tests = torch.zeros(batch, width, dtype=torch.long, device=device)
    mask = torch.zeros(batch, width, dtype=torch.bool, device=device)
    labels = torch.empty(batch, dtype=torch.long, device=device)
    unknown_id = table.shape[0] - 1
    for row, example in enumerate(examples):
        length = len(example.contents)
        ids = list(example.contents)
        if remove_expected:
            ids = [unknown_id if role == EXPECTED else value for value, role in zip(ids, example.roles)]
        content_ids[row, :length] = torch.tensor(ids, device=device)
        roles[row, :length] = torch.tensor(example.roles, device=device)
        tests[row, :length] = torch.tensor(example.test_ids, device=device)
        mask[row, :length] = True
        labels[row] = example.label
    return table[content_ids], roles, tests, mask, labels


@torch.no_grad()
def representation_metrics(encoder, examples, table, target_codes, batch_size: int = 32):
    predicted_all, labels_all = [], []
    for start in range(0, len(examples), batch_size):
        tensors = batch_tensors(examples[start : start + batch_size], table, target_codes.device)
        contents, roles, tests, mask, labels = tensors
        predicted_all.append(encoder(contents, roles, tests, mask))
        labels_all.append(labels)
    predicted = torch.cat(predicted_all)
    labels = torch.cat(labels_all)
    targets = target_codes[labels]
    flat_predicted = predicted.flatten(1)
    flat_targets = targets.flatten(1)
    similarities = nn.functional.cosine_similarity(
        flat_predicted.unsqueeze(1), target_codes.flatten(1).unsqueeze(0), dim=-1
    )
    return {
        "nearest_code_accuracy": (similarities.argmax(1) == labels).float().mean().item(),
        "cosine_similarity": nn.functional.cosine_similarity(
            flat_predicted, flat_targets, dim=-1
        ).mean().item(),
        "rmse": (predicted - targets).square().mean().sqrt().item(),
    }


@torch.no_grad()
def repair_evaluation(model, tokenizer, encoder, eval_examples, table, items, max_new_tokens=192):
    rows = []
    # One held-out long/multi-test evidence bundle per behavior.
    chosen = [next(example for example in eval_examples if example.label == i) for i in range(len(items))]
    for index, item in enumerate(items):
        paired = index + 1 if index % 2 == 0 else index - 1
        for condition, evidence_index, remove_expected in (
            ("true", index, False),
            ("paired_swap", paired, False),
            ("expected_removed", index, True),
        ):
            tensors = batch_tensors([chosen[evidence_index]], table, model.device, remove_expected=remove_expected)
            contents, roles, tests, mask, _ = tensors
            latent = encoder(contents, roles, tests, mask)
            rendered = render(tokenizer, item.case, MARKER, heldout=True)
            inputs = splice(model, tokenizer, rendered, latent)
            generated = greedy_generate(model, inputs, tokenizer.eos_token_id, max_new_tokens)
            response = tokenizer.decode(generated, skip_special_tokens=True)
            patch = None
            try:
                patch = extract_function(response, item.case.function_name)
                validation = evaluate_patch(patch, item.case)
            except Exception as exc:
                validation = {"passed": False, "error": f"{type(exc).__name__}: {exc}", "tests": []}
            rows.append({
                "case_id": item.case.case_id,
                "condition": condition,
                "evidence_events": len(chosen[evidence_index].contents),
                "passed": validation["passed"],
                "response": response,
                "patch": patch,
                "validation": validation,
            })
    return rows


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("A BF16 CUDA GPU is required; this pilot never falls back to CPU")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    items = get_ambiguous_cases()
    train_raw = make_dataset(
        items, args.train_examples, args.seed, long_bundle=False, heldout_values=False
    )
    eval_raw = make_dataset(
        items,
        args.eval_examples,
        args.seed + 10_000,
        long_bundle=True,
        heldout_values=True,
    )

    model_path = resolve_local_model(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.bfloat16
    ).to("cuda").eval()
    model.requires_grad_(False)
    anchors = initial_native_codes(model, tokenizer, items, slots=8)
    codebook = NativeRepairCodebook(anchors).to(model.device).eval()
    payload = torch.load(args.codebook_checkpoint, map_location=model.device, weights_only=True)
    codebook.residual.data.copy_(payload["adapter_trainable"]["residual"])
    codebook.requires_grad_(False)
    target_codes = codebook(torch.arange(len(items), device=model.device)).detach()

    vocabulary = sorted({content for example in train_raw + eval_raw for content in example.contents})
    vocabulary.append("expected output unknown")
    content_to_id = {content: index for index, content in enumerate(vocabulary)}
    table = native_content_table(model, tokenizer, vocabulary)
    train = numerical_examples(train_raw, content_to_id)
    evaluation = numerical_examples(eval_raw, content_to_id)
    encoder = RoleAwareEventEncoder(
        model_width=target_codes.shape[-1],
        output_anchor=target_codes.mean(dim=0),
        hidden_width=args.hidden_width,
        num_roles=10,
        max_events=192,
        max_tests=8,
    ).to(model.device)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.learning_rate, weight_decay=0.01)
    rng = random.Random(args.seed)
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        batch = [train[rng.randrange(len(train))] for _ in range(args.batch_size)]
        contents, roles, tests, mask, labels = batch_tensors(batch, table, model.device)
        predicted = encoder(contents, roles, tests, mask)
        target = target_codes[labels]
        mse = nn.functional.mse_loss(predicted, target)
        cosine = 1.0 - nn.functional.cosine_similarity(
            predicted.flatten(1), target.flatten(1), dim=-1
        ).mean()
        loss = mse + args.cosine_weight * cosine
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            print(json.dumps({
                "step": step,
                "loss": round(loss.item(), 6),
                "cosine": round(1.0 - cosine.item(), 5),
            }), flush=True)

    metrics = representation_metrics(encoder.eval(), evaluation, table, target_codes)
    repair_rows = repair_evaluation(model, tokenizer, encoder, evaluation, table, items)
    repair_summary = {
        condition: sum(row["passed"] for row in repair_rows if row["condition"] == condition)
        for condition in ("true", "paired_swap", "expected_removed")
    }
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "seed": args.seed,
        "steps": args.steps,
        "train_examples": args.train_examples,
        "eval_examples": args.eval_examples,
        "train_tests_per_bundle": [2, 4],
        "eval_tests_per_bundle": [5, 8],
        "vocabulary_size": len(vocabulary),
        "trainable_parameters": trainable_parameter_count(encoder),
        "training_seconds": time.perf_counter() - started,
        "representation": metrics,
        "repair_summary": repair_summary,
        "repair_rows": repair_rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "adapter_trainable": {
            name: tensor.detach().cpu()
            for name, tensor in encoder.state_dict().items()
            if name != "output_anchor"
        },
        "output_anchor": encoder.output_anchor.detach().cpu(),
        "result": result,
    }, checkpoint)
    print(json.dumps({"event": "final", "representation": metrics, "repair": repair_summary}), flush=True)


if __name__ == "__main__":
    main()
