#!/usr/bin/env python3
"""Residualize Gate 0: does a public-test lookup table predict hidden-test failure?

Frozen Qwen2.5-Coder-3B-Instruct, no training. Official MBPP train only. The
public test is the first official assertion and is shown in the prompt. Hidden
tests are the remaining assertions and never enter the prompt. Kill the method
if table residuals do not fail hidden tests more than generic ones.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from trace2cache.mbpp import load_mbpp, run_mbpp_tests
from trace2cache.residualize import (
    ANON_ENTRY,
    classify_residual,
    eligible_generation_task,
    entry_from_tests,
    evaluate_tests,
    extract_generation,
    generation_prompt,
    anonymize_tests,
    prompt_contains_hidden,
    prompt_contains_specification,
    split_public_hidden,
    summarize_gate0,
)

sys.path.insert(0, str(Path(__file__).parent))
from run_pilot1 import resolve_local_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--mbpp", default=".local/datasets/mbpp/mbpp.jsonl")
    parser.add_argument("--split", default="train", choices=("train", "validation"))
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--min-free-gib", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument(
        "--output-dir",
        default="artifacts/mbpp_generalization/residualize_gate0_seed1001",
    )
    parser.add_argument(
        "--task-ids",
        default="",
        help="comma-separated official-train task ids to freeze; default first eligible",
    )
    parser.add_argument(
        "--include-specification",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Put the English spec in the prompt. Off for the nospec Gate 0.",
    )
    parser.add_argument(
        "--anonymize-entry",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Rewrite the public/hidden entry name to f so the assert is not a spec.",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def existing_keys(path: Path) -> set[tuple[int, int]]:
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        keys.add((int(row["task_id"]), int(row["sample_i"])))
    return keys


def select_tasks(path: str, split: str, limit: int, task_ids: list[int] | None = None) -> list:
    wanted = set(task_ids or ())
    selected = []
    for task in load_mbpp(path, split=split):
        if task.split != split:
            raise SystemExit(f"split leak: task {task.task_id} is {task.split}")
        if split == "test":
            raise SystemExit("Gate 0 must not use the official MBPP test split")
        if wanted and task.task_id not in wanted:
            continue
        if not eligible_generation_task(task):
            continue
        result = run_mbpp_tests(task)
        if result.get("status") != "all_pass":
            continue
        selected.append(task)
        if not wanted and limit and len(selected) >= limit:
            break
    if wanted:
        missing = sorted(wanted - {task.task_id for task in selected})
        if missing:
            raise SystemExit(f"frozen task ids missing or ineligible: {missing[:8]}")
        order = {task_id: index for index, task_id in enumerate(task_ids or ())}
        selected.sort(key=lambda task: order[task.task_id])
    return selected


def render_chat(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a precise Python programmer."},
            {"role": "user", "content": prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.inference_mode()
def sample_generate(model, tokenizer, prompt: str, args, sample_seed: int) -> tuple[str, int]:
    torch.manual_seed(sample_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sample_seed)
    rendered = render_chat(tokenizer, prompt)
    encoded = tokenizer(rendered, return_tensors="pt").to(model.device)
    output = model.generate(
        **encoded,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    prompt_len = int(encoded["input_ids"].shape[1])
    text = tokenizer.decode(output[0, prompt_len:], skip_special_tokens=True)
    return text, prompt_len


def load_model(args):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("A BF16 CUDA GPU is required")
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes / 2**30 < args.min_free_gib:
        raise SystemExit(f"only {free_bytes / 2**30:.1f} GiB free; need {args.min_free_gib:.1f} GiB")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model_path = resolve_local_model(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, torch_dtype=torch.bfloat16
        )
        .to("cuda")
        .eval()
    )
    model.requires_grad_(False)
    model.config.use_cache = True
    return model, tokenizer, free_bytes


def split_for_run(task, anonymize: bool):
    public, hidden = split_public_hidden(task.tests)
    entry = entry_from_tests(public)
    if not anonymize:
        return public, public, hidden, entry
    prompt_public = anonymize_tests(public, entry)
    return prompt_public, prompt_public, anonymize_tests(hidden, entry), entry


def score_row(task, source: str | None, public, hidden, kind_error: str | None = None) -> dict:
    if source is None:
        report = classify_residual("not python", public)
        return {
            "kind": report.kind,
            "reason": kind_error or report.reason,
            "entry": report.entry,
            "key_uses": report.key_uses,
            "generic_uses": report.generic_uses,
            "public_pass": False,
            "hidden_pass": False,
            "public_status": "extraction_failure",
            "hidden_status": "extraction_failure",
        }
    report = classify_residual(source, public)
    public_result = evaluate_tests(task, source, public)
    hidden_result = evaluate_tests(task, source, hidden)
    return {
        "kind": report.kind,
        "reason": report.reason,
        "entry": report.entry,
        "key_uses": report.key_uses,
        "generic_uses": report.generic_uses,
        "public_pass": bool(public_result.get("passed")),
        "hidden_pass": bool(hidden_result.get("passed")),
        "public_status": public_result.get("status"),
        "hidden_status": hidden_result.get("status"),
    }


def main() -> None:
    args = parse_args()
    if args.split == "test":
        raise SystemExit("Gate 0 must not use the official MBPP test split")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "rows.jsonl"
    done = existing_keys(rows_path) if args.resume else set()
    frozen_ids = [int(part) for part in args.task_ids.split(",") if part.strip()] or None
    tasks = select_tasks(args.mbpp, args.split, args.limit, frozen_ids)
    if not tasks:
        raise SystemExit("no eligible official-train tasks")
    task_ids = [task.task_id for task in tasks]
    cohort_sha = hashlib.sha256(",".join(map(str, task_ids)).encode()).hexdigest()
    gold_kind = {}
    for task in tasks:
        original_public, original_hidden = split_public_hidden(task.tests)
        gold = classify_residual(task.canonical_source, original_public)
        gold_kind[task.task_id] = gold.kind
        prompt_public, _, eval_hidden, entry = split_for_run(task, args.anonymize_entry)
        prompt = generation_prompt(
            task.description,
            prompt_public,
            include_specification=args.include_specification,
        )
        if prompt_contains_hidden(prompt, original_hidden) or prompt_contains_hidden(
            prompt, eval_hidden
        ):
            raise SystemExit(f"hidden test leaked into prompt for task {task.task_id}")
        if not args.include_specification and prompt_contains_specification(
            prompt, task.description, prompt_public
        ):
            raise SystemExit(f"specification leaked into nospec prompt for task {task.task_id}")
        if (
            args.anonymize_entry
            and entry != ANON_ENTRY
            and re.search(rf"\b{re.escape(entry)}\b", prompt)
        ):
            raise SystemExit(f"entry name {entry} leaked into anonymized prompt for task {task.task_id}")
    print(
        json.dumps(
            {
                "event": "cohort",
                "split": args.split,
                "n_tasks": len(tasks),
                "task_ids": task_ids,
                "cohort_sha256": cohort_sha,
                "gold_table": sum(kind == "table" for kind in gold_kind.values()),
                "include_specification": args.include_specification,
                "anonymize_entry": args.anonymize_entry,
                "samples": args.samples,
            }
        ),
        flush=True,
    )
    model, tokenizer, free_bytes = load_model(args)
    started = time.perf_counter()
    with rows_path.open("a" if args.resume else "w") as handle:
        for task in tasks:
            prompt_public, eval_public, eval_hidden, entry = split_for_run(
                task, args.anonymize_entry
            )
            prompt = generation_prompt(
                task.description,
                prompt_public,
                include_specification=args.include_specification,
            )
            for sample_i in range(args.samples):
                key = (task.task_id, sample_i)
                if key in done:
                    continue
                sample_seed = args.seed + task.task_id * 100 + sample_i
                try:
                    response, prompt_tokens = sample_generate(
                        model, tokenizer, prompt, args, sample_seed
                    )
                    source = extract_generation(response)
                    scored = score_row(task, source, eval_public, eval_hidden)
                except Exception as error:  # noqa: BLE001
                    response = locals().get("response")
                    prompt_tokens = locals().get("prompt_tokens", 0)
                    source = None
                    scored = score_row(
                        task,
                        None,
                        eval_public,
                        eval_hidden,
                        kind_error=f"{type(error).__name__}: {error}",
                    )
                row = {
                    "task_id": task.task_id,
                    "sample_i": sample_i,
                    "split": task.split,
                    "seed": sample_seed,
                    "original_entry": entry,
                    "include_specification": args.include_specification,
                    "anonymize_entry": args.anonymize_entry,
                    "prompt_tokens": prompt_tokens,
                    "response": response,
                    "source": source,
                    **scored,
                }
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
                done.add(key)
                print(
                    json.dumps(
                        {
                            "event": "row",
                            "task_id": task.task_id,
                            "sample_i": sample_i,
                            "kind": row["kind"],
                            "public_pass": row["public_pass"],
                            "hidden_pass": row["hidden_pass"],
                        }
                    ),
                    flush=True,
                )
    rows = [
        json.loads(line)
        for line in rows_path.read_text().splitlines()
        if line.strip()
    ]
    summary = {
        "args": vars(args),
        "call_note": None,
        "cohort_sha256": cohort_sha,
        "free_gib_at_start": free_bytes / 2**30,
        "gold_kind": {str(task_id): kind for task_id, kind in gold_kind.items()},
        "n_tasks": len(tasks),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "seconds": time.perf_counter() - started,
        "summary": summarize_gate0(rows, gold_kind=gold_kind),
        "task_ids": task_ids,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    summary["call"] = summary["summary"]["call"]
    summary["call_note"] = summary["summary"]["note"]
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "done", "summary": summary["summary"], "call": summary["call"]}), flush=True)


if __name__ == "__main__":
    main()
