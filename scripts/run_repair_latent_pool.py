#!/usr/bin/env python3
"""Actual repair gate using fixed-rate, model-native pooled trace events."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from trace2cache.benchmark import get_cases
from trace2cache.refactory import load_refactory_q1
from trace2cache.repair_latent import select_temporal_events, summarize_events
from trace2cache.runtime import execute_with_trace
from trace2cache.sandbox import evaluate_patch, extract_function
from run_pilot1 import resolve_local_model


DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"
MARKER = "<TRACE2CACHE_LATENT>"
CONDITIONS = ("test_only", "selected_text", "latent_mean", "shuffled_latent_mean")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--benchmark", choices=("synthetic", "refactory-q1"), default="refactory-q1")
    parser.add_argument("--refactory-root", default="third_party/refactory")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--event-slots", type=int, default=8)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output", default="artifacts/repair_latent/refactory_q1_event_pool.jsonl")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def repair_prompt(case, actual: str, evidence: str) -> str:
    args_text = ", ".join(repr(argument) for argument in case.public_test.args)
    return f"""Fix the Python function below.

Specification: {case.description}

Buggy code:
```python
{case.buggy_source.rstrip()}
```

Failing test:
`{case.function_name}({args_text})` expected `{case.public_test.expected!r}` but produced `{actual}`.

Runtime evidence (chronological execution events):
{evidence}

Return only the complete corrected function in one Python code block. Do not include tests or explanation.
"""


def render_chat(tokenizer, case, actual: str, evidence: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a precise Python program repair assistant."},
            {"role": "user", "content": repair_prompt(case, actual, evidence)},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def mean_event_vectors(model, tokenizer, summaries: list[str]) -> torch.Tensor:
    embedding = model.get_input_embeddings()
    vectors = []
    for summary in summaries:
        token_ids = tokenizer(
            summary, return_tensors="pt", add_special_tokens=False
        )["input_ids"].to(model.device)
        vectors.append(embedding(token_ids)[0].float().mean(dim=0))
    return torch.stack(vectors).to(embedding.weight.dtype)


def prompt_embeddings(model, tokenizer, rendered: str, summaries: list[str] | None):
    embedding = model.get_input_embeddings()
    if summaries is None:
        ids = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
            model.device
        )
        return embedding(ids), int(ids.shape[1])
    before, after = rendered.split(MARKER)
    before_ids = tokenizer(before, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
        model.device
    )
    after_ids = tokenizer(after, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
        model.device
    )
    latent = mean_event_vectors(model, tokenizer, summaries).unsqueeze(0)
    inputs = torch.cat((embedding(before_ids), latent, embedding(after_ids)), dim=1)
    return inputs, int(inputs.shape[1])


@torch.inference_mode()
def greedy_generate(model, inputs_embeds: torch.Tensor, eos_token_id: int, max_new_tokens: int):
    attention = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=model.device)
    outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention, use_cache=True)
    cache = outputs.past_key_values
    next_token = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
    generated = [next_token]
    for _ in range(max_new_tokens - 1):
        if int(next_token.item()) == eos_token_id:
            break
        attention = torch.cat(
            (attention, torch.ones((1, 1), dtype=attention.dtype, device=attention.device)), dim=1
        )
        outputs = model(
            input_ids=next_token,
            attention_mask=attention,
            past_key_values=cache,
            use_cache=True,
        )
        cache = outputs.past_key_values
        next_token = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
        generated.append(next_token)
    return torch.cat(generated, dim=1)[0]


def existing_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    return {
        (row["case_id"], row["condition"])
        for row in (json.loads(line) for line in path.read_text().splitlines())
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("A BF16 CUDA GPU is required; this pilot never falls back to CPU")
    if args.event_slots < 1:
        raise SystemExit("--event-slots must be positive")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = existing_keys(output) if args.resume else set()
    cases = (
        get_cases(args.limit)
        if args.benchmark == "synthetic"
        else load_refactory_q1(args.refactory_root, args.limit)
    )
    traces = []
    for case in cases:
        trace = execute_with_trace(
            case.buggy_source, case.function_name, case.public_test.args, max_events=2_000
        )
        summaries = select_temporal_events(summarize_events(trace.events), args.event_slots)
        actual = trace.error or trace.output or "None"
        traces.append((actual, summaries, len(trace.events)))

    model_path = resolve_local_model(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.bfloat16
    ).to("cuda").eval()
    model.requires_grad_(False)
    print(json.dumps({
        "event": "model_loaded",
        "model": args.model,
        "device": str(model.device),
        "gpu_allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
    }), flush=True)

    for case_index, case in enumerate(cases):
        actual, true_summaries, raw_event_count = traces[case_index]
        shuffled_summaries = traces[(case_index + 1) % len(cases)][1]
        source_trace_tokens = len(
            tokenizer("\n".join(true_summaries), add_special_tokens=False)["input_ids"]
        )
        for condition in args.conditions:
            if (case.case_id, condition) in completed:
                print(json.dumps({"event": "skip", "case_id": case.case_id, "condition": condition}))
                continue
            if condition == "test_only":
                rendered = render_chat(tokenizer, case, actual, "No additional runtime evidence.")
                latent_summaries = None
            elif condition == "selected_text":
                rendered = render_chat(tokenizer, case, actual, "\n".join(true_summaries))
                latent_summaries = None
            elif condition == "latent_mean":
                rendered = render_chat(tokenizer, case, actual, MARKER)
                latent_summaries = true_summaries
            else:
                rendered = render_chat(tokenizer, case, actual, MARKER)
                latent_summaries = shuffled_summaries
            inputs, input_slots = prompt_embeddings(model, tokenizer, rendered, latent_summaries)
            started = time.perf_counter()
            generated = greedy_generate(
                model, inputs, tokenizer.eos_token_id, args.max_new_tokens
            )
            elapsed = time.perf_counter() - started
            response = tokenizer.decode(generated, skip_special_tokens=True)
            patch = None
            try:
                patch = extract_function(response, case.function_name)
                validation = evaluate_patch(patch, case)
            except Exception as exc:
                validation = {
                    "passed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "tests": [],
                }
            row = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": args.model,
                "case_id": case.case_id,
                "condition": condition,
                "event_slots": args.event_slots,
                "raw_trace_events": raw_event_count,
                "selected_events": len(latent_summaries or true_summaries),
                "selected_text_tokens": source_trace_tokens,
                "actual_input_slots": input_slots,
                "output_tokens": int(generated.shape[0]),
                "generation_seconds": elapsed,
                "response": response,
                "patch": patch,
                "validation": validation,
            }
            with output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps({
                "event": "result",
                "case_id": case.case_id,
                "condition": condition,
                "passed": validation["passed"],
                "input_slots": input_slots,
                "selected_text_tokens": source_trace_tokens,
                "seconds": round(elapsed, 2),
            }), flush=True)


if __name__ == "__main__":
    main()
