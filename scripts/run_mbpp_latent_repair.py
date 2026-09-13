#!/usr/bin/env python3
"""Train a trace encoder for open-ended repair on task-disjoint MBPP mutants."""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
from run_pilot1 import resolve_local_model
from run_repair_codebook import MARKER, greedy_generate, splice, target_ids
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from trace2cache.latent import RoleAwareEventEncoder, trainable_parameter_count
from trace2cache.mbpp import (
    MBPPMutation,
    MBPPTask,
    generate_mutants,
    load_mbpp,
    run_mbpp_tests,
    trace_mbpp_tests,
)

TEST, CALL, BRANCH, STATE, LINE, RETURN, PASS, FAIL = range(8)
SAFE_IMPORTS = {
    "array",
    "bisect",
    "cmath",
    "collections",
    "copy",
    "datetime",
    "heapq",
    "itertools",
    "math",
    "operator",
    "re",
    "sys",
}
FORBIDDEN_CALLS = {"__import__", "compile", "eval", "exec", "input", "open"}


@dataclass(frozen=True)
class RepairExample:
    task_id: int
    mutation_kind: str
    buggy_source: str
    target_source: str
    setup_source: str
    failing_test: str
    tests: tuple[str, ...]
    contents: tuple[str, ...]
    roles: tuple[int, ...]
    test_ids: tuple[int, ...]


@dataclass(frozen=True)
class NumericExample:
    example: RepairExample
    content_ids: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--dataset", default=".local/datasets/mbpp/mbpp.jsonl")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-width", type=int, default=256)
    parser.add_argument("--latent-slots", type=int, default=8)
    parser.add_argument("--max-mutants-per-task", type=int, default=2)
    parser.add_argument("--max-events-per-test", type=int, default=24)
    parser.add_argument("--max-train-examples", type=int, default=256)
    parser.add_argument("--max-eval-examples", type=int, default=36)
    parser.add_argument("--generation-examples", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-sequence-tokens", type=int, default=768)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=173)
    parser.add_argument("--min-free-gib", type=float, default=24.0)
    parser.add_argument("--cache-dir", default=".local/cache/mbpp_latent_repair")
    parser.add_argument(
        "--output", default="artifacts/mbpp_repair/latent_repair_seed173.json"
    )
    parser.add_argument(
        "--checkpoint", default="checkpoints/mbpp_repair/latent_repair_seed173.pt"
    )
    return parser.parse_args()


def _select_events(events: list[dict], limit: int) -> list[dict]:
    selected = []
    previous_locals = None
    for event in events:
        boundary = event["event"] in {"call", "return", "exception"}
        changed = event["locals"] != previous_locals
        if boundary or changed:
            selected.append(event)
        previous_locals = event["locals"]
    if len(selected) <= limit:
        return selected
    indices = [round(index * (len(selected) - 1) / (limit - 1)) for index in range(limit)]
    return [selected[index] for index in indices]


def _event_role(event: dict) -> int:
    if event["event"] == "call":
        return CALL
    if event["event"] in {"return", "exception"}:
        return RETURN
    source = event["source"].lstrip()
    if source.startswith(("if ", "elif ", "for ", "while ")):
        return BRANCH
    return STATE if event["locals"] else LINE


def _event_text(event: dict) -> str:
    state = ", ".join(f"{name}={value}" for name, value in event["locals"].items())
    result = f" return={event['value']}" if event["value"] is not None else ""
    text = (
        f"{event['event']} {event['function']} line {event['line']} "
        f"source {event['source']} state {state}{result}"
    )
    return re.sub(r"0x[0-9a-fA-F]+", "0xADDR", text)


def _example_from_trace(
    task: MBPPTask,
    mutation: MBPPMutation,
    traced: dict,
    max_events_per_test: int,
) -> RepairExample | None:
    passing = next((index for index, value in enumerate(traced["tests"]) if value == "pass"), None)
    failing = next((index for index, value in enumerate(traced["tests"]) if value != "pass"), None)
    if passing is None or failing is None:
        return None
    contents: list[str] = []
    roles: list[int] = []
    test_ids: list[int] = []
    for local_test_id, original_index in enumerate((passing, failing)):
        contents.append(f"test assertion {task.tests[original_index]}")
        roles.append(TEST)
        test_ids.append(local_test_id)
        for event in _select_events(traced["traces"][original_index], max_events_per_test):
            contents.append(_event_text(event))
            roles.append(_event_role(event))
            test_ids.append(local_test_id)
        outcome = traced["tests"][original_index]
        contents.append(f"test outcome {outcome}")
        roles.append(PASS if outcome == "pass" else FAIL)
        test_ids.append(local_test_id)
    return RepairExample(
        task_id=task.task_id,
        mutation_kind=mutation.kind,
        buggy_source=mutation.source,
        target_source=task.canonical_source,
        setup_source=task.setup_source,
        failing_test=task.tests[failing],
        tests=task.tests,
        contents=tuple(contents),
        roles=tuple(roles),
        test_ids=tuple(test_ids),
    )


def _build_examples(
    tasks: tuple[MBPPTask, ...],
    *,
    max_mutants_per_task: int,
    max_events_per_test: int,
    workers: int,
) -> list[RepairExample]:
    candidate_jobs = [
        (task, mutation)
        for task in tasks
        for mutation in generate_mutants(task, limit=12)
    ]

    def classify(job):
        task, mutation = job
        return job, run_mbpp_tests(task, mutation.source)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        classified = list(pool.map(classify, candidate_jobs))
    per_task: dict[int, int] = {}
    mixed_jobs = []
    for job, result in classified:
        task, _ = job
        if result["status"] != "mixed" or per_task.get(task.task_id, 0) >= max_mutants_per_task:
            continue
        per_task[task.task_id] = per_task.get(task.task_id, 0) + 1
        mixed_jobs.append(job)

    def trace(job):
        task, mutation = job
        result = trace_mbpp_tests(task, mutation.source)
        return _example_from_trace(task, mutation, result, max_events_per_test)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        examples = list(pool.map(trace, mixed_jobs))
    return [example for example in examples if example is not None]


def _load_or_build_examples(args, split: str) -> list[RepairExample]:
    cache = Path(args.cache_dir) / (
        f"{split}_m{args.max_mutants_per_task}_e{args.max_events_per_test}.json"
    )
    if cache.exists():
        return [
            RepairExample(
                **{
                    **row,
                    "tests": tuple(row["tests"]),
                    "contents": tuple(row["contents"]),
                    "roles": tuple(row["roles"]),
                    "test_ids": tuple(row["test_ids"]),
                }
            )
            for row in json.loads(cache.read_text())
        ]
    examples = _build_examples(
        load_mbpp(args.dataset, split=split),
        max_mutants_per_task=args.max_mutants_per_task,
        max_events_per_test=args.max_events_per_test,
        workers=args.workers,
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps([asdict(example) for example in examples]) + "\n")
    return examples


def _render(tokenizer, example: RepairExample, evidence: str) -> str:
    user = f"""Repair this Python program using the failing test and runtime evidence.

Buggy program:
```python
{example.buggy_source.rstrip()}
```

Known failing test:
`{example.failing_test}`

Runtime evidence: {evidence}

Return only the complete corrected Python source in one code block.
"""
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a precise Python program repair assistant."},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.no_grad()
def _native_content_table(model, tokenizer, vocabulary: list[str], batch_size: int = 128):
    embedding = model.get_input_embeddings()
    vectors = []
    for start in range(0, len(vocabulary), batch_size):
        encoded = tokenizer(
            vocabulary[start : start + batch_size],
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=96,
            return_tensors="pt",
        ).to(model.device)
        token_vectors = embedding(encoded["input_ids"]).float()
        mask = encoded["attention_mask"].unsqueeze(-1)
        vectors.append((token_vectors * mask).sum(1) / mask.sum(1).clamp_min(1))
    return torch.cat(vectors)


def _batch_encoder_inputs(example, table, device):
    content_ids = torch.tensor(example.content_ids, device=device).unsqueeze(0)
    roles = torch.tensor(example.example.roles, device=device).unsqueeze(0)
    tests = torch.tensor(example.example.test_ids, device=device).unsqueeze(0)
    mask = torch.ones_like(roles, dtype=torch.bool)
    return table[content_ids], roles, tests, mask


def _repair_loss(model, tokenizer, encoder, example, table):
    contents, roles, tests, mask = _batch_encoder_inputs(example, table, model.device)
    latent = encoder(contents, roles, tests, mask).to(model.get_input_embeddings().weight.dtype)
    prompt = _render(tokenizer, example.example, MARKER)
    prompt_embeds = splice(model, tokenizer, prompt, latent)
    targets = target_ids(tokenizer, example.example.target_source, model.device)
    teacher = model.get_input_embeddings()(targets[:, :-1])
    inputs = torch.cat((prompt_embeds, teacher), dim=1)
    attention = torch.ones(inputs.shape[:2], dtype=torch.long, device=model.device)
    logits = model(inputs_embeds=inputs, attention_mask=attention, use_cache=False).logits
    start = prompt_embeds.shape[1] - 1
    predicted = logits[:, start : start + targets.shape[1]].float()
    return nn.functional.cross_entropy(predicted.flatten(0, 1), targets.flatten())


@torch.no_grad()
def _condition_loss(model, tokenizer, encoder, example, table, condition, shuffled=None):
    if condition in {"true_latent", "shuffled_latent"}:
        evidence = example if condition == "true_latent" else shuffled
        contents, roles, tests, mask = _batch_encoder_inputs(evidence, table, model.device)
        latent = encoder(contents, roles, tests, mask).to(model.get_input_embeddings().weight.dtype)
        prompt = _render(tokenizer, example.example, MARKER)
    elif condition == "trace_text":
        latent = None
        prompt = _render(tokenizer, example.example, "\n" + "\n".join(example.example.contents))
    else:
        latent = None
        prompt = _render(tokenizer, example.example, "unavailable")
    prompt_embeds = splice(model, tokenizer, prompt, latent)
    targets = target_ids(tokenizer, example.example.target_source, model.device)
    teacher = model.get_input_embeddings()(targets[:, :-1])
    inputs = torch.cat((prompt_embeds, teacher), dim=1)
    logits = model(inputs_embeds=inputs, use_cache=False).logits
    start = prompt_embeds.shape[1] - 1
    predicted = logits[:, start : start + targets.shape[1]].float()
    return nn.functional.cross_entropy(predicted.flatten(0, 1), targets.flatten()).item()


def _extract_source(response: str) -> str:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL | re.IGNORECASE)
    for candidate in blocks or [response]:
        source = candidate.strip() + "\n"
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module = (node.module or node.names[0].name).split(".")[0]
                if module not in SAFE_IMPORTS:
                    raise ValueError(f"unsafe import: {module}")
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in FORBIDDEN_CALLS
            ):
                raise ValueError(f"unsafe call: {node.func.id}")
        return source
    raise ValueError("no Python source found")


@torch.inference_mode()
def _generate(model, tokenizer, encoder, numeric, table, condition, shuffled, max_new_tokens):
    example = numeric.example
    if condition in {"true_latent", "shuffled_latent"}:
        evidence = numeric if condition == "true_latent" else shuffled
        contents, roles, tests, mask = _batch_encoder_inputs(evidence, table, model.device)
        latent = encoder(contents, roles, tests, mask)
        rendered = _render(tokenizer, example, MARKER)
    elif condition == "trace_text":
        latent = None
        rendered = _render(tokenizer, example, "\n" + "\n".join(example.contents))
    else:
        latent = None
        rendered = _render(tokenizer, example, "unavailable")
    inputs = splice(model, tokenizer, rendered, latent)
    tokens = greedy_generate(model, inputs, tokenizer.eos_token_id, max_new_tokens)
    response = tokenizer.decode(tokens, skip_special_tokens=True)
    try:
        source = _extract_source(response)
        task = MBPPTask(
            task_id=example.task_id,
            description="",
            canonical_source=source,
            tests=example.tests,
            setup_source=example.setup_source,
        )
        validation = run_mbpp_tests(task, timeout_seconds=3.0)
    except Exception as exc:  # noqa: BLE001
        source = None
        validation = {"status": "invalid", "tests": [], "error": f"{type(exc).__name__}: {exc}"}
    return {"response": response, "source": source, "validation": validation}


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("A BF16 CUDA GPU is required; this pilot never falls back to CPU")
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes / 2**30 < args.min_free_gib:
        raise SystemExit(f"only {free_bytes / 2**30:.1f} GiB free; need {args.min_free_gib:.1f} GiB")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(json.dumps({"event": "building_train_data"}), flush=True)
    train_raw = _load_or_build_examples(args, "train")[: args.max_train_examples]
    print(json.dumps({"event": "building_eval_data"}), flush=True)
    eval_raw = _load_or_build_examples(args, "validation")[: args.max_eval_examples]

    model_path = resolve_local_model(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.bfloat16
    ).to("cuda").eval()
    model.requires_grad_(False)
    model.config.use_cache = False

    def within_budget(example):
        prompt_length = len(tokenizer(_render(tokenizer, example, MARKER), add_special_tokens=False).input_ids)
        target_length = len(tokenizer(example.target_source, add_special_tokens=False).input_ids)
        return prompt_length + target_length <= args.max_sequence_tokens

    train_raw = [example for example in train_raw if within_budget(example)]
    eval_raw = [example for example in eval_raw if within_budget(example)]
    vocabulary = sorted({content for example in train_raw + eval_raw for content in example.contents})
    content_to_id = {content: index for index, content in enumerate(vocabulary)}
    table = _native_content_table(model, tokenizer, vocabulary)
    train = [NumericExample(example, tuple(content_to_id[x] for x in example.contents)) for example in train_raw]
    evaluation = [NumericExample(example, tuple(content_to_id[x] for x in example.contents)) for example in eval_raw]

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
    ).to(model.device)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.learning_rate, weight_decay=0.01)
    rng = random.Random(args.seed)
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, args.steps + 1):
        example = train[rng.randrange(len(train))]
        loss = _repair_loss(model, tokenizer, encoder, example, table)
        (loss / args.gradient_accumulation).backward()
        if step % args.gradient_accumulation == 0:
            nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % 20 == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": round(loss.item(), 5),
                        "gpu_allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
                        "gpu_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                    }
                ),
                flush=True,
            )

    encoder.eval()
    conditions = ("no_evidence", "trace_text", "true_latent", "shuffled_latent")
    losses = {condition: [] for condition in conditions}
    for index, example in enumerate(evaluation):
        shuffled = evaluation[(index + 1) % len(evaluation)]
        for condition in conditions:
            losses[condition].append(
                _condition_loss(model, tokenizer, encoder, example, table, condition, shuffled)
            )
    mean_losses = {
        condition: sum(values) / len(values) for condition, values in losses.items()
    }

    generation_rows = []
    for index, example in enumerate(evaluation[: args.generation_examples]):
        shuffled = evaluation[(index + 1) % len(evaluation)]
        for condition in conditions:
            result = _generate(
                model,
                tokenizer,
                encoder,
                example,
                table,
                condition,
                shuffled,
                args.max_new_tokens,
            )
            generation_rows.append(
                {"task_id": example.example.task_id, "condition": condition, **result}
            )
            print(
                json.dumps(
                    {
                        "event": "generation",
                        "task_id": example.example.task_id,
                        "condition": condition,
                        "status": result["validation"]["status"],
                    }
                ),
                flush=True,
            )

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "seed": args.seed,
        "steps": args.steps,
        "train_examples": len(train),
        "eval_examples": len(evaluation),
        "latent_slots": args.latent_slots,
        "trainable_parameters": trainable_parameter_count(encoder),
        "training_seconds": time.perf_counter() - started,
        "teacher_forced_patch_loss": mean_losses,
        "generation_summary": {
            condition: sum(
                row["validation"]["status"] == "all_pass"
                for row in generation_rows
                if row["condition"] == condition
            )
            for condition in conditions
        },
        "generation_rows": generation_rows,
        "gpu_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
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
            "result": {key: value for key, value in result.items() if key != "generation_rows"},
        },
        checkpoint,
    )
    print(
        json.dumps(
            {
                "event": "final",
                "losses": mean_losses,
                "generation": result["generation_summary"],
                "output": str(output),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
