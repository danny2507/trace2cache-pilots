#!/usr/bin/env python3
"""Ask-events Gate 0: can a frozen 3B invent compact traces that help stdout?

Copies frozen `code_input` and `trace_text` scores. Generates only `ask_events`.
Does not train. Does not hide stdin. Does not overwrite the frozen sim panel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from trace2cache.execution_sim import (
    ASK_EVENTS,
    NATIVE_CAP,
    count_event_role_lines,
    extract_output,
    outputs_match,
    prompt_introduces_gold,
    read_examples,
    render_ask_events_chat,
    response_event_prefix,
    summarize_ask_events_gate0,
)
from trace2cache.receiver_training import splice_prompt

sys.path.insert(0, str(Path(__file__).parent))
from run_execution_sim import existing_keys, load_model
from run_repair_latent_pool import greedy_generate

FROZEN_TEST_SHA256 = "074dd6cecdd960970356907168b6f3ea5c086ae9b487ca56aecec6d0b943276c"
COPY_CONDITIONS = ("code_input", "trace_text")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument(
        "--cohort", default="artifacts/mbpp_generalization/execution_sim_runbugrun_v1"
    )
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--text-rows",
        default="artifacts/mbpp_generalization/execution_sim_text_seed1001/rows.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/mbpp_generalization/execution_sim_ask_events_seed1001",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--min-free-gib", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_baselines(text_rows: Path, wanted: set[str], handle, done: set[tuple[str, str]]) -> int:
    if not text_rows.exists():
        raise SystemExit(f"missing frozen text Gate 0 rows: {text_rows}")
    copied = 0
    for line in text_rows.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("example_id") not in wanted:
            continue
        if row.get("condition") not in COPY_CONDITIONS:
            continue
        key = (row["example_id"], row["condition"])
        if key in done:
            continue
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        done.add(key)
        copied += 1
    handle.flush()
    return copied


def generate_ask(model, tokenizer, example, args) -> dict:
    rendered = render_ask_events_chat(tokenizer, example, cap=NATIVE_CAP)
    if prompt_introduces_gold(rendered, example):
        raise SystemExit(f"gold stdout leaked into ask_events prompt for {example.example_id}")
    if "Partial execution" in rendered:
        raise SystemExit(f"gold trace leaked into ask_events prompt for {example.example_id}")
    stdin = example.stdin.strip()
    if stdin and stdin not in rendered:
        raise SystemExit(f"stdin hidden in ask_events prompt for {example.example_id}")
    if example.source.strip() and example.source.strip() not in rendered:
        raise SystemExit(f"program missing from ask_events prompt for {example.example_id}")
    inputs = splice_prompt(model, tokenizer, rendered, None)
    tokens = greedy_generate(model, inputs, tokenizer.eos_token_id, args.max_new_tokens)
    response = tokenizer.decode(tokens, skip_special_tokens=True)
    predicted = extract_output(response)
    prefix = response_event_prefix(response)
    return {
        "task_id": example.task_id,
        "example_id": example.example_id,
        "condition": ASK_EVENTS,
        "prompt_tokens": int(inputs.shape[1]),
        "event_count": example.event_count,
        "event_lines": count_event_role_lines(prefix),
        "slot_count": 0,
        "response": response,
        "predicted": predicted,
        "gold": example.gold_stdout,
        "passed": outputs_match(predicted, example.gold_stdout),
        "outcome": "pass" if outputs_match(predicted, example.gold_stdout) else "fail",
        "max_new_tokens": args.max_new_tokens,
    }


def main() -> None:
    args = parse_args()
    if args.split != "test":
        raise SystemExit("ask-events Gate 0 evaluates the frozen sim test split only")
    cohort = Path(args.cohort)
    test_path = cohort / "test.jsonl"
    panel_sha = sha256_file(test_path)
    if panel_sha != FROZEN_TEST_SHA256:
        raise SystemExit(f"frozen sim test sha mismatch: {panel_sha} != {FROZEN_TEST_SHA256}")
    examples = read_examples(test_path)
    examples.sort(key=lambda example: example.task_id)
    if args.limit:
        examples = examples[: args.limit]
    if len(examples) != 128 and not args.limit:
        raise SystemExit(f"frozen panel must have 128 examples; got {len(examples)}")
    wanted = {example.example_id for example in examples}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "rows.jsonl"
    done = existing_keys(rows_path) if args.resume else set()
    model, tokenizer, free_bytes = load_model(args)
    started = time.perf_counter()
    copied = 0
    with rows_path.open("a" if args.resume else "w") as handle:
        copied = copy_baselines(Path(args.text_rows), wanted, handle, done)
        print(json.dumps({"event": "copied_baselines", "n": copied, "panel_sha256": panel_sha}), flush=True)
        for example in examples:
            key = (example.example_id, ASK_EVENTS)
            if key in done:
                continue
            row = generate_ask(model, tokenizer, example, args)
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            done.add(key)
            print(
                json.dumps(
                    {
                        "event": "row",
                        "example_id": example.example_id,
                        "task_id": example.task_id,
                        "passed": row["passed"],
                        "event_lines": row["event_lines"],
                    }
                ),
                flush=True,
            )
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    by_condition = {}
    for row in rows:
        by_condition.setdefault(row["condition"], 0)
        by_condition[row["condition"]] += 1
    for condition in (*COPY_CONDITIONS, ASK_EVENTS):
        if by_condition.get(condition, 0) != len(examples):
            raise SystemExit(
                f"{condition} has {by_condition.get(condition, 0)} rows; expected {len(examples)}"
            )
    decision = summarize_ask_events_gate0(rows)
    summary = {
        "args": vars(args),
        "call": decision["call"],
        "call_note": decision["note"],
        "elapsed_seconds": time.perf_counter() - started,
        "eval_examples": len(examples),
        "eval_tasks": len({example.task_id for example in examples}),
        "free_gib_at_start": free_bytes / 2**30,
        "panel_sha256": panel_sha,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "summary": decision,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trainable_parameters": 0,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "done", "call": decision["call"], "summary": decision}), flush=True)


if __name__ == "__main__":
    main()
