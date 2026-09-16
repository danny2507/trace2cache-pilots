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
    generate_expanded_mutants,
    generate_mutants,
    load_mbpp,
    run_mbpp_calls,
    run_mbpp_tests,
    trace_mbpp_tests,
)

TEST, CALL, BRANCH, STATE, LINE, RETURN, PASS, FAIL = range(8)
# Fixed semantic jobs for the eight latent states. TEST is retained as a
# fallback in sparse traces; the remaining roles make each state specialize.
ROLE_FACTORIZED_SLOTS = (
    (TEST, FAIL, RETURN),       # failed assertion / observed return
    (TEST, BRANCH, FAIL),       # causal control decision
    (TEST, STATE, RETURN),      # last state update
    (TEST, LINE, STATE),        # source-local state context
    (TEST, PASS, RETURN),       # passing reference behavior
    (TEST, PASS, FAIL, STATE),  # pass/fail behavioral delta
    (TEST, BRANCH, STATE),      # control-to-data link
    tuple(range(8)),            # integration slot
)
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


@dataclass(frozen=True)
class NumericEvidence:
    """One trace view, possibly a causal counterfactual of an example."""

    example: RepairExample
    content_ids: tuple[int, ...]
    roles: tuple[int, ...]
    test_ids: tuple[int, ...]
    name: str = "true"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--dataset", default=".local/datasets/mbpp/mbpp.jsonl")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-width", type=int, default=256)
    parser.add_argument("--latent-slots", type=int, default=8)
    parser.add_argument("--expanded-mutations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--candidate-mutants-per-task", type=int, default=32)
    parser.add_argument("--max-mutants-per-task", type=int, default=4)
    parser.add_argument("--fuzz-calls", type=int, default=16)
    parser.add_argument("--fuzz-bundles-per-mutant", type=int, default=1)
    parser.add_argument("--max-events-per-test", type=int, default=24)
    parser.add_argument("--max-train-examples", type=int, default=768)
    parser.add_argument("--max-eval-examples", type=int, default=36)
    parser.add_argument("--generation-examples", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-sequence-tokens", type=int, default=768)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=173)
    parser.add_argument("--contrastive-weight", type=float, default=1.0)
    parser.add_argument("--contrastive-margin", type=float, default=0.1)
    parser.add_argument("--counterfactual-weight", type=float, default=1.0)
    parser.add_argument("--counterfactual-margin", type=float, default=0.1)
    parser.add_argument("--role-factorized-slots", action=argparse.BooleanOptionalAction, default=True)
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


def _literal_calls(task: MBPPTask) -> list[tuple[str, list[object]]]:
    definitions = {
        node.name
        for node in ast.parse(task.canonical_source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    calls = []
    seen = set()
    for test in task.tests:
        for node in ast.walk(ast.parse(test)):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id not in definitions or node.keywords:
                continue
            try:
                arguments = [ast.literal_eval(argument) for argument in node.args]
            except (ValueError, TypeError):
                continue
            key = (node.func.id, repr(arguments))
            if key not in seen:
                seen.add(key)
                calls.append((node.func.id, arguments))
    return calls


def _value_variants(value: object) -> list[object]:
    if type(value) is int:
        return [0, 1, -1, value - 1, value + 1, -value, value * 2]
    if type(value) is float:
        return [0.0, 1.0, -1.0, value - 1.0, value + 1.0, -value]
    if isinstance(value, str):
        return ["", value[:1], value[:-1], value[::-1], value + value[:1]]
    if isinstance(value, (list, tuple)):
        constructor = type(value)
        sequence = list(value)
        variants = [constructor(), constructor(reversed(sequence)), constructor(sequence[:-1])]
        if sequence:
            variants.append(constructor(sequence + [sequence[0]]))
            for replacement in _value_variants(sequence[0])[:3]:
                changed = list(sequence)
                changed[0] = replacement
                variants.append(constructor(changed))
        return variants
    if isinstance(value, set):
        return [set(), set(list(value)[:-1])]
    if isinstance(value, dict):
        return [{}]
    return []


def _fuzzed_calls(task: MBPPTask, *, limit: int, seed: int) -> list[str]:
    candidates = []
    for function_name, arguments in _literal_calls(task):
        candidates.append(f"{function_name}({', '.join(map(repr, arguments))})")
        for argument_index, argument in enumerate(arguments):
            for replacement in _value_variants(argument):
                changed = list(arguments)
                changed[argument_index] = replacement
                candidates.append(f"{function_name}({', '.join(map(repr, changed))})")
    candidates = list(dict.fromkeys(candidates))
    random.Random(seed).shuffle(candidates)
    return candidates[:limit]


def _examples_from_calls(
    task: MBPPTask,
    mutation: MBPPMutation,
    calls: list[str],
    oracle: dict,
    actual: dict,
    *,
    max_events_per_test: int,
    max_bundles: int,
) -> list[RepairExample]:
    if oracle.get("status") != "ok" or actual.get("status") != "ok":
        return []
    records = []
    for call, expected, observed in zip(calls, oracle["calls"], actual["calls"]):
        if expected["status"] != "ok":
            continue
        matches = observed["status"] == "ok" and observed["output"] == expected["output"]
        records.append((call, expected, observed, matches))
    passing = [record for record in records if record[3]]
    failing = [record for record in records if not record[3]]
    examples = []
    for bundle_index in range(min(max_bundles, len(passing), len(failing))):
        selected = (passing[bundle_index], failing[bundle_index])
        contents: list[str] = []
        roles: list[int] = []
        test_ids: list[int] = []
        for test_id, (call, expected, observed, matches) in enumerate(selected):
            contents.append(
                f"call {call} expected {expected['output']} observed {observed['output']}"
            )
            roles.append(TEST)
            test_ids.append(test_id)
            for event in _select_events(observed["trace"], max_events_per_test):
                contents.append(_event_text(event))
                roles.append(_event_role(event))
                test_ids.append(test_id)
            contents.append(f"test outcome {'pass' if matches else 'fail'}")
            roles.append(PASS if matches else FAIL)
            test_ids.append(test_id)
        failed_call, expected, observed, _ = selected[1]
        examples.append(
            RepairExample(
                task_id=task.task_id,
                mutation_kind=mutation.kind + ":fuzz",
                buggy_source=mutation.source,
                target_source=task.canonical_source,
                setup_source=task.setup_source,
                failing_test=(
                    f"{failed_call} expected {expected['output']} but produced {observed['output']}"
                ),
                tests=task.tests,
                contents=tuple(contents),
                roles=tuple(roles),
                test_ids=tuple(test_ids),
            )
        )
    return examples


def _build_examples(
    tasks: tuple[MBPPTask, ...],
    *,
    max_mutants_per_task: int,
    candidate_mutants_per_task: int,
    expanded_mutations: bool,
    fuzz_calls: int,
    fuzz_bundles_per_mutant: int,
    max_events_per_test: int,
    workers: int,
) -> list[RepairExample]:
    generator = generate_expanded_mutants if expanded_mutations else generate_mutants
    candidate_jobs = [
        (task, mutation)
        for task in tasks
        for mutation in generator(task, limit=candidate_mutants_per_task)
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
        base_examples = list(pool.map(trace, mixed_jobs))

    def fuzz(job):
        task, mutation = job
        calls = _fuzzed_calls(task, limit=fuzz_calls, seed=task.task_id + mutation.ordinal)
        if not calls:
            return []
        oracle = run_mbpp_calls(task, calls, collect_trace=False)
        actual = run_mbpp_calls(task, calls, mutation.source, collect_trace=True)
        return _examples_from_calls(
            task,
            mutation,
            calls,
            oracle,
            actual,
            max_events_per_test=max_events_per_test,
            max_bundles=fuzz_bundles_per_mutant,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        fuzzed = list(pool.map(fuzz, mixed_jobs))
    return [example for example in base_examples if example is not None] + [
        example for group in fuzzed for example in group
    ]


def _load_or_build_examples(args, split: str) -> list[RepairExample]:
    cache = Path(args.cache_dir) / (
        f"v2_{split}_x{int(args.expanded_mutations)}_c{args.candidate_mutants_per_task}"
        f"_m{args.max_mutants_per_task}_f{args.fuzz_calls}"
        f"_b{args.fuzz_bundles_per_mutant}_e{args.max_events_per_test}.json"
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
        candidate_mutants_per_task=args.candidate_mutants_per_task,
        expanded_mutations=args.expanded_mutations,
        fuzz_calls=args.fuzz_calls,
        fuzz_bundles_per_mutant=args.fuzz_bundles_per_mutant,
        max_events_per_test=args.max_events_per_test,
        workers=args.workers,
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps([asdict(example) for example in examples]) + "\n")
    return examples


def _one_example_per_task(examples: list[RepairExample]) -> list[RepairExample]:
    selected = {}
    for example in examples:
        selected.setdefault(example.task_id, example)
    return list(selected.values())


def _different_task_example(examples: list[NumericExample], index: int) -> NumericExample:
    task_id = examples[index].example.task_id
    for offset in range(1, len(examples)):
        candidate = examples[(index + offset) % len(examples)]
        if candidate.example.task_id != task_id:
            return candidate
    raise ValueError("shuffled control requires at least two distinct task IDs")


def _as_evidence(example: NumericExample) -> NumericEvidence:
    return NumericEvidence(
        example=example.example,
        content_ids=example.content_ids,
        roles=example.example.roles,
        test_ids=example.example.test_ids,
    )


def _counterfactual_evidence(example: NumericExample, kind: str) -> NumericEvidence:
    """Corrupt one causal field while keeping code, target, and event count fixed.

    These are deliberately much harder negatives than a different-task trace:
    every view shares the buggy source, target patch, event vocabulary, and
    trace length.  Only its typed runtime interpretation changes.
    """
    roles = list(example.example.roles)
    content_ids = list(example.content_ids)
    test_ids = list(example.example.test_ids)
    dynamic = [
        index
        for index, role in enumerate(roles)
        if role not in {TEST, PASS, FAIL}
    ]
    if kind == "role_swap":
        # Exchange control-flow and state-update identities, preserving all
        # text/value tokens and temporal positions.
        for index, role in enumerate(roles):
            if role == BRANCH:
                roles[index] = STATE
            elif role == STATE:
                roles[index] = BRANCH
    elif kind == "value_swap":
        # Move event text/value bundles between dynamic locations but retain
        # their source-role labels and test membership.
        if len(dynamic) >= 2:
            first, last = dynamic[0], dynamic[-1]
            content_ids[first], content_ids[last] = content_ids[last], content_ids[first]
    elif kind == "trace_reassigned":
        # Attribute dynamic events to the opposite pass/fail execution while
        # keeping test assertions and outcome tokens untouched.
        for index in dynamic:
            test_ids[index] = 1 - test_ids[index]
    else:
        raise ValueError(f"unknown counterfactual kind: {kind}")
    return NumericEvidence(
        example=example.example,
        content_ids=tuple(content_ids),
        roles=tuple(roles),
        test_ids=tuple(test_ids),
        name=kind,
    )


COUNTERFACTUAL_KINDS = ("role_swap", "value_swap", "trace_reassigned")


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
    roles = torch.tensor(example.roles, device=device).unsqueeze(0)
    tests = torch.tensor(example.test_ids, device=device).unsqueeze(0)
    mask = torch.ones_like(roles, dtype=torch.bool)
    return table[content_ids], roles, tests, mask


def _repair_loss(model, tokenizer, encoder, prompt_example, evidence_example, table):
    contents, roles, tests, mask = _batch_encoder_inputs(evidence_example, table, model.device)
    latent = encoder(contents, roles, tests, mask).to(model.get_input_embeddings().weight.dtype)
    prompt = _render(tokenizer, prompt_example.example, MARKER)
    prompt_embeds = splice(model, tokenizer, prompt, latent)
    targets = target_ids(tokenizer, prompt_example.example.target_source, model.device)
    teacher = model.get_input_embeddings()(targets[:, :-1])
    inputs = torch.cat((prompt_embeds, teacher), dim=1)
    attention = torch.ones(inputs.shape[:2], dtype=torch.long, device=model.device)
    logits = model(inputs_embeds=inputs, attention_mask=attention, use_cache=False).logits
    start = prompt_embeds.shape[1] - 1
    predicted = logits[:, start : start + targets.shape[1]].float()
    return nn.functional.cross_entropy(predicted.flatten(0, 1), targets.flatten())


@torch.no_grad()
def _condition_loss(model, tokenizer, encoder, example, table, condition, shuffled=None):
    if condition in {"true_latent", "shuffled_latent", *COUNTERFACTUAL_KINDS}:
        if condition == "true_latent":
            evidence = _as_evidence(example)
        elif condition == "shuffled_latent":
            evidence = _as_evidence(shuffled)
        else:
            evidence = _counterfactual_evidence(example, condition)
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
    if condition in {"true_latent", "shuffled_latent", *COUNTERFACTUAL_KINDS}:
        if condition == "true_latent":
            evidence = _as_evidence(numeric)
        elif condition == "shuffled_latent":
            evidence = _as_evidence(shuffled)
        else:
            evidence = _counterfactual_evidence(numeric, condition)
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
    train_pool = _load_or_build_examples(args, "train")
    random.Random(args.seed).shuffle(train_pool)
    train_raw = train_pool[: args.max_train_examples]
    print(json.dumps({"event": "building_eval_data"}), flush=True)
    eval_pool = _load_or_build_examples(args, "validation")
    random.Random(args.seed + 1).shuffle(eval_pool)
    eval_raw = _one_example_per_task(eval_pool)[: args.max_eval_examples]
    print(
        json.dumps(
            {
                "event": "data_ready",
                "train_examples_before_token_filter": len(train_raw),
                "train_tasks": len({example.task_id for example in train_raw}),
                "train_fuzz_examples": sum(
                    example.mutation_kind.endswith(":fuzz") for example in train_raw
                ),
                "eval_examples_before_token_filter": len(eval_raw),
                "eval_tasks": len({example.task_id for example in eval_raw}),
            }
        ),
        flush=True,
    )

    model_path = resolve_local_model(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.bfloat16
    ).to("cuda").eval()
    model.requires_grad_(False)
    model.config.use_cache = False

    def within_budget(example):
        prompt_length = len(
            tokenizer(
                _render(tokenizer, example, MARKER), add_special_tokens=False
            ).input_ids
        )
        target_length = len(tokenizer(example.target_source, add_special_tokens=False).input_ids)
        return prompt_length + target_length <= args.max_sequence_tokens

    train_raw = [example for example in train_raw if within_budget(example)]
    eval_raw = [example for example in eval_raw if within_budget(example)]
    vocabulary = sorted({content for example in train_raw + eval_raw for content in example.contents})
    content_to_id = {content: index for index, content in enumerate(vocabulary)}
    table = _native_content_table(model, tokenizer, vocabulary)
    train = [
        NumericExample(example, tuple(content_to_id[x] for x in example.contents))
        for example in train_raw
    ]
    evaluation = [
        NumericExample(example, tuple(content_to_id[x] for x in example.contents))
        for example in eval_raw
    ]

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
        slot_roles=ROLE_FACTORIZED_SLOTS if args.role_factorized_slots else None,
    ).to(model.device)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.learning_rate, weight_decay=0.01)
    rng = random.Random(args.seed)
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, args.steps + 1):
        example_index = rng.randrange(len(train))
        example = train[example_index]
        shuffled = _different_task_example(train, example_index)
        true_evidence = _as_evidence(example)
        true_loss = _repair_loss(model, tokenizer, encoder, example, true_evidence, table)
        shuffled_loss = _repair_loss(
            model, tokenizer, encoder, example, _as_evidence(shuffled), table
        )
        ranking_loss = nn.functional.relu(args.contrastive_margin + true_loss - shuffled_loss)
        counterfactual_losses = {
            kind: _repair_loss(
                model,
                tokenizer,
                encoder,
                example,
                _counterfactual_evidence(example, kind),
                table,
            )
            for kind in COUNTERFACTUAL_KINDS
        }
        counterfactual_ranking = sum(
            nn.functional.relu(args.counterfactual_margin + true_loss - negative_loss)
            for negative_loss in counterfactual_losses.values()
        ) / len(counterfactual_losses)
        loss = (
            true_loss
            + args.contrastive_weight * ranking_loss
            + args.counterfactual_weight * counterfactual_ranking
        )
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
                        "true_patch_loss": round(true_loss.item(), 5),
                        "shuffled_patch_loss": round(shuffled_loss.item(), 5),
                        "ranking_loss": round(ranking_loss.item(), 5),
                        "counterfactual_patch_loss": {
                            kind: round(value.item(), 5)
                            for kind, value in counterfactual_losses.items()
                        },
                        "counterfactual_ranking_loss": round(counterfactual_ranking.item(), 5),
                        "gpu_allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
                        "gpu_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                    }
                ),
                flush=True,
            )

    encoder.eval()
    conditions = (
        "no_evidence",
        "trace_text",
        "true_latent",
        "shuffled_latent",
        *COUNTERFACTUAL_KINDS,
    )
    losses = {condition: [] for condition in conditions}
    for index, example in enumerate(evaluation):
        shuffled = _different_task_example(evaluation, index)
        for condition in conditions:
            losses[condition].append(
                _condition_loss(model, tokenizer, encoder, example, table, condition, shuffled)
            )
    mean_losses = {
        condition: sum(values) / len(values) for condition, values in losses.items()
    }

    generation_rows = []
    for index, example in enumerate(evaluation[: args.generation_examples]):
        shuffled = _different_task_example(evaluation, index)
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
        "train_tasks": len({example.example.task_id for example in train}),
        "train_fuzz_examples": sum(
            example.example.mutation_kind.endswith(":fuzz") for example in train
        ),
        "eval_examples": len(evaluation),
        "latent_slots": args.latent_slots,
        "contrastive_weight": args.contrastive_weight,
        "contrastive_margin": args.contrastive_margin,
        "counterfactual_weight": args.counterfactual_weight,
        "counterfactual_margin": args.counterfactual_margin,
        "role_factorized_slots": args.role_factorized_slots,
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
