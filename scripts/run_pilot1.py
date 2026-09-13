#!/usr/bin/env python3
"""Run deterministic no-training textual-trace repair conditions."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from trace2cache.benchmark import get_cases
from trace2cache.prompts import CONDITIONS, build_prompt
from trace2cache.refactory import load_refactory_q1
from trace2cache.sandbox import evaluate_patch, extract_function


DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"


def resolve_local_model(model: str) -> str:
    """Resolve a model ID despite legacy TRANSFORMERS_CACHE/HF_HOME layouts."""
    direct = Path(model)
    if direct.exists():
        return str(direct.resolve())
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    model_cache = hf_home / "hub" / ("models--" + model.replace("/", "--"))
    main_ref = model_cache / "refs" / "main"
    if main_ref.exists():
        snapshot = model_cache / "snapshots" / main_ref.read_text().strip()
        if (snapshot / "config.json").exists():
            return str(snapshot)
    snapshots = sorted((model_cache / "snapshots").glob("*")) if model_cache.exists() else []
    complete = [path for path in snapshots if (path / "config.json").exists()]
    if complete:
        return str(complete[-1])
    raise FileNotFoundError(f"model is not available locally: {model}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--benchmark", choices=("synthetic", "refactory-q1"), default="synthetic")
    parser.add_argument("--refactory-root", default="third_party/refactory")
    parser.add_argument("--output", default="artifacts/pilot1/results.jsonl")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def existing_keys(path: Path) -> set[tuple[str, str, str, int]]:
    keys = set()
    if not path.exists():
        return keys
    for line in path.read_text().splitlines():
        row = json.loads(line)
        keys.add((row["model"], row["case_id"], row["condition"], row["seed"]))
    return keys


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    if args.device == "cuda" and not torch.cuda.is_bf16_supported():
        raise SystemExit("BF16 support is required for the CUDA pilot")
    if args.device == "cpu":
        torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = existing_keys(output_path) if args.resume else set()

    model_path = resolve_local_model(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=dtype,
    ).to(args.device).eval()
    print(
        json.dumps(
            {
                "event": "model_loaded",
                "model": args.model,
                "device": str(model.device),
                "gpu_allocated_gib": (
                    round(torch.cuda.memory_allocated() / 2**30, 2)
                    if args.device == "cuda"
                    else 0.0
                ),
            }
        ),
        flush=True,
    )

    cases = (
        get_cases(args.limit)
        if args.benchmark == "synthetic"
        else load_refactory_q1(args.refactory_root, args.limit)
    )
    for case_index, case in enumerate(cases):
        for condition in args.conditions:
            key = (args.model, case.case_id, condition, args.seed)
            if key in completed:
                print(json.dumps({"event": "skip", "key": key}), flush=True)
                continue
            evidence_case = cases[(case_index + 1) % len(cases)] if condition == "shuffled_json" else None
            prompt, trace_metadata = build_prompt(
                case, condition, evidence_case=evidence_case
            )
            messages = [
                {"role": "system", "content": "You are a precise Python program repair assistant."},
                {"role": "user", "content": prompt},
            ]
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            elapsed = time.perf_counter() - started
            new_tokens = generated[0, inputs["input_ids"].shape[1] :]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            validation: dict[str, object]
            patch = None
            try:
                patch = extract_function(response, case.function_name)
                validation = evaluate_patch(patch, case)
            except Exception as exc:
                validation = {"passed": False, "error": f"{type(exc).__name__}: {exc}", "tests": []}
            row = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": args.model,
                "seed": args.seed,
                "case_id": case.case_id,
                "condition": condition,
                "input_tokens": int(inputs["input_ids"].shape[1]),
                "output_tokens": int(new_tokens.shape[0]),
                "generation_seconds": elapsed,
                **trace_metadata,
                "response": response,
                "patch": patch,
                "validation": validation,
            }
            with output_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                json.dumps(
                    {
                        "event": "result",
                        "case_id": case.case_id,
                        "condition": condition,
                        "passed": validation["passed"],
                        "input_tokens": row["input_tokens"],
                        "seconds": round(elapsed, 2),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
