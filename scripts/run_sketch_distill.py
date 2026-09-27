#!/usr/bin/env python3
"""Sketch distillation vs direct repair LoRA on disjoint MBPP train.

Same no_evidence prompt for both LoRAs. Eval is hidden-test Repair@1 on the
frozen 128. Does not train on the frozen test. Does not overwrite RunBugRun
or sim panels. Copies no_evidence from sketchicae.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from trace2cache.mbpp_generalization import (
    RepairExample,
    allowed_calls_for_tests,
    evaluate_hidden,
    extract_patch,
    read_examples,
)
from trace2cache.receiver_training import splice_prompt
from trace2cache.sketch_distill import (
    DIRECT_SFT,
    SKETCH_DISTILL,
    TRAIN_OBJECTIVES,
    copy_text_control_rows,
    prompt_leaks_oracle,
    render_repair_chat,
    sft_target,
    sketch_prefix,
    summarize_sketch_distill,
)

sys.path.insert(0, str(Path(__file__).parent))
from run_mbpp_generalization import existing_keys
from run_pilot1 import resolve_local_model
from run_repair_latent_pool import greedy_generate

FROZEN_COHORT_SHA256 = "acc350b87778d21a6ef441ec9cab4e4ef0d6cbaca1061703bb069a476f658baa"
FROZEN_TEST_SHA256 = "6d6600ff1898d6881c9ac0f4cca30d868c6bb549d2dccd415e68761c3d96f9cb"
FROZEN_PANEL_N = 128
TEXT_CONTROL_ROWS = "artifacts/mbpp_generalization/sketchicae_seed1001/rows.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--cohort", default="artifacts/mbpp_generalization/cohort_v1")
    parser.add_argument(
        "--output-dir", default="artifacts/mbpp_generalization/sketch_distill_seed1001"
    )
    parser.add_argument("--checkpoint-dir", default="checkpoints/mbpp_generalization")
    parser.add_argument("--text-rows", default=TEXT_CONTROL_ROWS)
    parser.add_argument("--objectives", nargs="+", choices=TRAIN_OBJECTIVES, default=list(TRAIN_OBJECTIVES))
    parser.add_argument("--steps", type=int, default=1280)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--mask-p", type=float, default=0.5)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--max-sequence-tokens", type=int, default=1536)
    parser.add_argument("--min-free-gib", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--limit", type=int, default=FROZEN_PANEL_N)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_path(args, objective: str) -> Path:
    tag = "direct" if objective == DIRECT_SFT else "sketch"
    return Path(args.checkpoint_dir) / f"sketch_distill_{tag}_seed{args.seed}.pt"


def assert_train_disjoint(train: list[RepairExample], test: list[RepairExample]) -> None:
    train_tasks = {example.task_id for example in train}
    test_tasks = {example.task_id for example in test}
    overlap = train_tasks & test_tasks
    if overlap:
        raise SystemExit(f"train/test task overlap: {sorted(overlap)[:8]}")
    train_ids = {example.example_id for example in train}
    test_ids = {example.example_id for example in test}
    if train_ids & test_ids:
        raise SystemExit("train/test example_id overlap")


def attach_lora(model, rank: int, alpha: int):
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, config)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    return model


def save_lora(path: Path, model, args, objective: str, training: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "lora": {key: value.detach().cpu() for key, value in get_peft_model_state_dict(model).items()},
            "objective": objective,
            "args": vars(args),
            "training": training,
        },
        path,
    )


def load_lora(model, path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    set_peft_model_state_dict(model, payload["lora"])
    return payload


def load_base(args):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("A BF16 CUDA GPU is required")
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes / 2**30 < args.min_free_gib:
        raise SystemExit(f"only {free_bytes / 2**30:.1f} GiB free; need {args.min_free_gib:.1f} GiB")
    model_path = resolve_local_model(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.bfloat16
    ).to("cuda")
    model.config.use_cache = False
    return model, tokenizer, free_bytes


def continuation_token_count(tokenizer, prompt: str, continuation: str) -> int:
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    combined = tokenizer(prompt + continuation, add_special_tokens=False).input_ids
    return max(0, len(combined) - len(prompt_ids))


def encode_pair(tokenizer, prompt: str, target: str, device, *, mask_sketch_tokens: int = 0):
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    target_ids = tokenizer(target, add_special_tokens=False).input_ids
    if tokenizer.eos_token_id is not None:
        target_ids = target_ids + [tokenizer.eos_token_id]
    labels = [-100] * len(prompt_ids) + list(target_ids)
    if mask_sketch_tokens:
        start = len(prompt_ids)
        end = min(start + mask_sketch_tokens, start + len(target_ids) - 1)
        for index in range(start, end):
            labels[index] = -100
        if all(label == -100 for label in labels):
            labels = [-100] * len(prompt_ids) + list(target_ids)
    input_ids = torch.tensor([prompt_ids + target_ids], device=device)
    label_tensor = torch.tensor([labels], device=device)
    return input_ids, label_tensor, len(prompt_ids) + len(target_ids)


def pick_example(train, tokenizer, objective, args, rng):
    for _ in range(32):
        example = train[rng.randrange(len(train))]
        prompt = render_repair_chat(tokenizer, example)
        target = sft_target(example, objective)
        prompt_len = len(tokenizer(prompt, add_special_tokens=False).input_ids)
        target_len = len(tokenizer(target, add_special_tokens=False).input_ids) + 1
        if prompt_len + target_len <= args.max_sequence_tokens:
            return example, prompt, target
    return None, None, None


def completion_nll(model, tokenizer, prompt: str, target: str, *, mask_sketch_tokens: int = 0) -> torch.Tensor:
    input_ids, labels, _ = encode_pair(
        tokenizer, prompt, target, model.device, mask_sketch_tokens=mask_sketch_tokens
    )
    attention = torch.ones_like(input_ids)
    return model(input_ids=input_ids, attention_mask=attention, labels=labels, use_cache=False).loss


def train_objective(model, tokenizer, train, objective, args) -> dict:
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=0.01)
    rng = random.Random(args.seed)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    history = []
    skipped = 0
    masked = 0
    for step in range(1, args.steps + 1):
        example, prompt, target = pick_example(train, tokenizer, objective, args, rng)
        if example is None:
            skipped += 1
            continue
        mask_tokens = 0
        if objective == SKETCH_DISTILL and rng.random() < args.mask_p:
            n_sketch = continuation_token_count(tokenizer, prompt, sketch_prefix(example))
            if n_sketch > 0:
                mask_tokens = n_sketch
                masked += 1
        loss = completion_nll(model, tokenizer, prompt, target, mask_sketch_tokens=mask_tokens)
        (loss / args.gradient_accumulation).backward()
        if step % args.gradient_accumulation == 0 or step == args.steps:
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % 20 == 0 or step == args.steps:
            row = {
                "event": "train_step",
                "objective": objective,
                "step": step,
                "loss": round(float(loss.item()), 5),
                "masked_sketch": bool(mask_tokens),
                "gpu_allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
            }
            history.append(row)
            print(json.dumps(row), flush=True)
    model.eval()
    return {
        "steps": args.steps,
        "seconds": time.perf_counter() - started,
        "skipped": skipped,
        "masked_sketch_steps": masked,
        "history": history,
        "trainable_parameters": sum(parameter.numel() for parameter in params),
    }


@torch.inference_mode()
def generate_row(model, tokenizer, example, objective, args) -> dict:
    rendered = render_repair_chat(tokenizer, example)
    if prompt_leaks_oracle(rendered, example):
        raise SystemExit(f"oracle leaked into {objective} prompt for {example.example_id}")
    if example.failing_public_test and example.failing_public_test not in rendered:
        raise SystemExit(f"public test hidden in {objective} prompt for {example.example_id}")
    model.config.use_cache = True
    inputs = splice_prompt(model, tokenizer, rendered, None)
    tokens = greedy_generate(model, inputs, tokenizer.eos_token_id, args.max_new_tokens)
    response = tokenizer.decode(tokens, skip_special_tokens=True)
    try:
        allowed_calls = allowed_calls_for_tests(example.hidden_tests + example.public_tests)
        source = extract_patch(response, allowed_calls=allowed_calls)
        validation = evaluate_hidden(example, source)
        outcome = "pass" if validation["passed"] else validation.get("status", "fail")
    except Exception as error:  # noqa: BLE001
        source = None
        validation = {
            "passed": False,
            "status": "extraction_failure",
            "error": f"{type(error).__name__}: {error}",
            "tests": [],
        }
        outcome = "extraction_failure"
    return {
        "task_id": example.task_id,
        "example_id": example.example_id,
        "condition": objective,
        "prompt_tokens": int(inputs.shape[1]),
        "response": response,
        "source": source,
        "validation": validation,
        "passed": bool(validation.get("passed")),
        "outcome": outcome,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
    }


def eval_objective(model, tokenizer, examples, objective, args, rows_path, done) -> int:
    written = 0
    model.eval()
    with rows_path.open("a") as handle:
        for example in examples:
            key = (example.example_id, objective)
            if key in done:
                continue
            row = generate_row(model, tokenizer, example, objective, args)
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            done.add(key)
            written += 1
            print(
                json.dumps(
                    {
                        "event": "row",
                        "condition": objective,
                        "example_id": example.example_id,
                        "task_id": example.task_id,
                        "passed": row["passed"],
                        "outcome": row["outcome"],
                    }
                ),
                flush=True,
            )
    return written


def copy_controls(text_rows_path: Path, rows_path: Path, eval_ids: set[str], done: set[tuple[str, str]]) -> int:
    source = [json.loads(line) for line in text_rows_path.read_text().splitlines() if line.strip()]
    copied_rows = copy_text_control_rows(source)
    written = 0
    with rows_path.open("a") as handle:
        for row in copied_rows:
            if row["example_id"] not in eval_ids:
                continue
            key = (row["example_id"], row["condition"])
            if key in done:
                continue
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            done.add(key)
            written += 1
    return written


def unload(model) -> None:
    del model
    torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    if args.mask_p < 0 or args.mask_p > 1:
        raise SystemExit("--mask-p must be in [0, 1]")
    cohort = Path(args.cohort)
    if "runbugrun" in str(cohort) or "execution_sim" in str(cohort):
        raise SystemExit("sketch distill uses cohort_v1; refusing to touch RunBugRun or sim panels")
    train_path = cohort / "train.jsonl"
    test_path = cohort / "test.jsonl"
    manifest = json.loads((cohort / "manifest.json").read_text())
    train_sha = sha256_file(train_path)
    test_sha = sha256_file(test_path)
    if manifest.get("sha256") != FROZEN_COHORT_SHA256:
        raise SystemExit(f"frozen cohort sha mismatch: {manifest.get('sha256')} != {FROZEN_COHORT_SHA256}")
    if test_sha != FROZEN_TEST_SHA256:
        raise SystemExit(f"frozen MBPP test sha mismatch: {test_sha} != {FROZEN_TEST_SHA256}")

    train = read_examples(train_path)
    test = read_examples(test_path)
    test.sort(key=lambda example: (example.task_id, example.mutation_kind, example.mutation_ordinal))
    train.sort(key=lambda example: (example.task_id, example.mutation_kind, example.mutation_ordinal))
    if args.limit:
        test = test[: args.limit]
    if len(test) != FROZEN_PANEL_N and args.limit == FROZEN_PANEL_N:
        raise SystemExit(f"frozen panel must have {FROZEN_PANEL_N} examples; got {len(test)}")
    assert_train_disjoint(train, test)

    text_rows_path = Path(args.text_rows)
    if not text_rows_path.exists():
        raise SystemExit(f"missing text control rows {text_rows_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "rows.jsonl"
    done = existing_keys(rows_path) if args.resume else set()
    if not args.resume and rows_path.exists():
        rows_path.unlink()
        done = set()
    copied = copy_controls(text_rows_path, rows_path, {example.example_id for example in test}, done)
    print(json.dumps({"event": "control_copied", "rows": copied, "path": str(text_rows_path)}), flush=True)

    started = time.perf_counter()
    training = {}
    peak = 0.0
    for objective in args.objectives:
        ckpt = checkpoint_path(args, objective)
        need_eval = any((example.example_id, objective) not in done for example in test)
        load_existing = args.skip_train or (args.resume and ckpt.exists())
        if load_existing and not ckpt.exists():
            raise SystemExit(f"missing LoRA checkpoint {ckpt}")
        if not load_existing or need_eval:
            model, tokenizer, free_bytes = load_base(args)
            print(
                json.dumps(
                    {
                        "event": "model_loaded",
                        "model": args.model,
                        "objective": objective,
                        "eval_examples": len(test),
                        "train_examples": len(train),
                        "gpu_free_gib": round(free_bytes / 2**30, 2),
                    }
                ),
                flush=True,
            )
            model = attach_lora(model, args.lora_rank, args.lora_alpha)
            if load_existing:
                load_lora(model, ckpt)
                training[objective] = {"loaded": str(ckpt)}
            else:
                print(
                    json.dumps({"event": "train_start", "objective": objective, "n_train": len(train)}),
                    flush=True,
                )
                training[objective] = train_objective(model, tokenizer, train, objective, args)
                save_lora(ckpt, model, args, objective, training[objective])
                print(
                    json.dumps(
                        {
                            "event": "train_done",
                            "objective": objective,
                            "checkpoint": str(ckpt),
                            "seconds": training[objective]["seconds"],
                            "trainable_parameters": training[objective]["trainable_parameters"],
                        }
                    ),
                    flush=True,
                )
            if need_eval:
                eval_objective(model, tokenizer, test, objective, args, rows_path, done)
            peak = max(peak, torch.cuda.max_memory_allocated() / 2**30)
            unload(model)
        else:
            print(json.dumps({"event": "eval_skip", "objective": objective}), flush=True)

    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    by_condition = {}
    for row in rows:
        by_condition[row["condition"]] = by_condition.get(row["condition"], 0) + 1
    for objective in args.objectives:
        if by_condition.get(objective, 0) != len(test):
            raise SystemExit(
                f"{objective} has {by_condition.get(objective, 0)} rows; expected {len(test)}"
            )
    decision = summarize_sketch_distill(rows)
    summary = {
        "args": vars(args),
        "call": decision["call"],
        "call_note": decision["note"],
        "elapsed_seconds": time.perf_counter() - started,
        "eval_examples": len(test),
        "eval_tasks": len({example.task_id for example in test}),
        "panel_sha256": test_sha,
        "peak_allocated_gib": peak,
        "summary": decision,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "train_examples": len(train),
        "train_sha256": train_sha,
        "training": training,
        "trainable_parameters": {
            objective: (training.get(objective) or {}).get("trainable_parameters", 0)
            for objective in args.objectives
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "done", "call": decision["call"], "summary": decision}), flush=True)


if __name__ == "__main__":
    main()
