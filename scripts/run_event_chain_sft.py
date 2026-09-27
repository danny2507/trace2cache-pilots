#!/usr/bin/env python3
"""Event-chain SFT vs stdout-only SFT on the disjoint 320.

Same ask_events prompt for both LoRAs. Eval is last-fence stdout on the frozen
128. Does not train on the frozen test. Does not hide stdin. Does not overwrite
the sim panel.
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

from trace2cache.execution_sim import (
    NATIVE_CAP,
    SFT_EVENTS,
    SFT_STDOUT,
    assert_train_disjoint,
    count_event_role_lines,
    extract_output,
    outputs_match,
    prompt_introduces_gold,
    read_examples,
    render_ask_events_chat,
    response_event_prefix,
    sft_target,
    summarize_event_sft,
)
from trace2cache.receiver_training import splice_prompt

sys.path.insert(0, str(Path(__file__).parent))
from run_execution_sim import existing_keys, load_model
from run_repair_latent_pool import greedy_generate

FROZEN_TEST_SHA256 = "074dd6cecdd960970356907168b6f3ea5c086ae9b487ca56aecec6d0b943276c"
FROZEN_TRAIN_SHA256 = "02ddf7fe8cb110397250d30142413b0c3234e1a476fcf3d0db4fb6c15fad3390"
OBJECTIVES = (SFT_STDOUT, SFT_EVENTS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument(
        "--cohort", default="artifacts/mbpp_generalization/execution_sim_runbugrun_v1"
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/mbpp_generalization/execution_sim_event_sft_seed1001",
    )
    parser.add_argument(
        "--checkpoint-dir", default="checkpoints/mbpp_generalization"
    )
    parser.add_argument("--objectives", nargs="+", choices=OBJECTIVES, default=list(OBJECTIVES))
    parser.add_argument("--steps", type=int, default=1280)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-sequence-tokens", type=int, default=4096)
    parser.add_argument("--min-free-gib", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_path(args, objective: str) -> Path:
    tag = "stdout" if objective == SFT_STDOUT else "events"
    return Path(args.checkpoint_dir) / f"event_sft_{tag}_seed{args.seed}.pt"


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


def encode_pair(tokenizer, prompt: str, target: str, device):
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    target_ids = tokenizer(target, add_special_tokens=False).input_ids
    if tokenizer.eos_token_id is not None:
        target_ids = target_ids + [tokenizer.eos_token_id]
    input_ids = torch.tensor([prompt_ids + target_ids], device=device)
    labels = torch.tensor([[-100] * len(prompt_ids) + target_ids], device=device)
    return input_ids, labels, len(prompt_ids) + len(target_ids)


def pick_example(train, tokenizer, objective, args, rng):
    for _ in range(32):
        example = train[rng.randrange(len(train))]
        prompt = render_ask_events_chat(tokenizer, example, cap=NATIVE_CAP)
        target = sft_target(example, objective, cap=NATIVE_CAP)
        prompt_len = len(tokenizer(prompt, add_special_tokens=False).input_ids)
        target_len = len(tokenizer(target, add_special_tokens=False).input_ids) + 1
        if prompt_len + target_len <= args.max_sequence_tokens:
            return example, prompt, target
    return None, None, None


def completion_nll(model, tokenizer, prompt: str, target: str) -> torch.Tensor:
    input_ids, labels, _ = encode_pair(tokenizer, prompt, target, model.device)
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
    for step in range(1, args.steps + 1):
        example, prompt, target = pick_example(train, tokenizer, objective, args, rng)
        if example is None:
            skipped += 1
            continue
        loss = completion_nll(model, tokenizer, prompt, target)
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
                "gpu_allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
            }
            history.append(row)
            print(json.dumps(row), flush=True)
    model.eval()
    return {
        "steps": args.steps,
        "seconds": time.perf_counter() - started,
        "skipped": skipped,
        "history": history,
        "trainable_parameters": sum(parameter.numel() for parameter in params),
    }


@torch.inference_mode()
def generate_row(model, tokenizer, example, objective, args) -> dict:
    rendered = render_ask_events_chat(tokenizer, example, cap=NATIVE_CAP)
    if prompt_introduces_gold(rendered, example):
        raise SystemExit(f"gold stdout leaked into {objective} prompt for {example.example_id}")
    stdin = example.stdin.strip()
    if stdin and stdin not in rendered:
        raise SystemExit(f"stdin hidden in {objective} prompt for {example.example_id}")
    if example.source.strip() and example.source.strip() not in rendered:
        raise SystemExit(f"program missing from {objective} prompt for {example.example_id}")
    model.config.use_cache = True
    inputs = splice_prompt(model, tokenizer, rendered, None)
    tokens = greedy_generate(model, inputs, tokenizer.eos_token_id, args.max_new_tokens)
    response = tokenizer.decode(tokens, skip_special_tokens=True)
    predicted = extract_output(response)
    prefix = response_event_prefix(response)
    return {
        "task_id": example.task_id,
        "example_id": example.example_id,
        "condition": objective,
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
                        "event_lines": row["event_lines"],
                    }
                ),
                flush=True,
            )
    return written


def unload(model) -> None:
    del model
    torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    cohort = Path(args.cohort)
    train_path = cohort / "train.jsonl"
    test_path = cohort / "test.jsonl"
    train_sha = sha256_file(train_path)
    test_sha = sha256_file(test_path)
    if train_sha != FROZEN_TRAIN_SHA256:
        raise SystemExit(f"frozen sim train sha mismatch: {train_sha} != {FROZEN_TRAIN_SHA256}")
    if test_sha != FROZEN_TEST_SHA256:
        raise SystemExit(f"frozen sim test sha mismatch: {test_sha} != {FROZEN_TEST_SHA256}")
    train = read_examples(train_path)
    test = read_examples(test_path)
    assert_train_disjoint(train, test)
    if args.limit:
        test = test[: args.limit]
    if len(test) != 128 and not args.limit:
        raise SystemExit(f"frozen panel must have 128 examples; got {len(test)}")
    if len(train) != 320:
        raise SystemExit(f"disjoint train must have 320 examples; got {len(train)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "rows.jsonl"
    done = existing_keys(rows_path) if args.resume else set()
    if not args.resume and rows_path.exists():
        rows_path.unlink()
        done = set()

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
            model, tokenizer, free_bytes = load_model(args)
            model = attach_lora(model, args.lora_rank, args.lora_alpha)
            if load_existing:
                load_lora(model, ckpt)
                training[objective] = {"loaded": str(ckpt)}
            else:
                print(json.dumps({"event": "train_start", "objective": objective, "n_train": len(train)}), flush=True)
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
    decision = summarize_event_sft(rows)
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
