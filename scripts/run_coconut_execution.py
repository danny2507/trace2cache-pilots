#!/usr/bin/env python3
"""Coconut curriculum for stdout prediction.

Init from the event-chain LoRA (stage 0, already trained). mix8 replaces the
first k event lines with the LoRA 3B's own last hidden states; latent8 drops
the remaining event tokens. No tracer. No spliced event vectors. Control is
the existing sft_stdout rows — this script does not retrain it.

Does not train on the frozen test. Does not hide stdin. Does not overwrite
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

from trace2cache.coconut_execution import (
    COCONUT_K,
    coconut_language_target,
    condition_for_stage,
    copy_stdout_control_rows,
    summarize_coconut,
)
from trace2cache.execution_sim import (
    NATIVE_CAP,
    SFT_STDOUT,
    assert_train_disjoint,
    extract_output,
    outputs_match,
    prompt_introduces_gold,
    read_examples,
    render_ask_events_chat,
)

sys.path.insert(0, str(Path(__file__).parent))
from run_execution_sim import existing_keys, load_model
from run_repair_latent_pool import greedy_generate

FROZEN_TEST_SHA256 = "074dd6cecdd960970356907168b6f3ea5c086ae9b487ca56aecec6d0b943276c"
FROZEN_TRAIN_SHA256 = "02ddf7fe8cb110397250d30142413b0c3234e1a476fcf3d0db4fb6c15fad3390"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument(
        "--cohort", default="artifacts/mbpp_generalization/execution_sim_runbugrun_v1"
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/mbpp_generalization/execution_sim_coconut_seed1001",
    )
    parser.add_argument("--checkpoint-dir", default="checkpoints/mbpp_generalization")
    parser.add_argument(
        "--init-checkpoint",
        default="checkpoints/mbpp_generalization/event_sft_events_seed1001.pt",
    )
    parser.add_argument(
        "--stdout-rows",
        default="artifacts/mbpp_generalization/execution_sim_event_sft_seed1001/rows.jsonl",
    )
    parser.add_argument("--thoughts", type=int, default=COCONUT_K)
    parser.add_argument("--mix-steps", type=int, default=640)
    parser.add_argument("--latent-steps", type=int, default=640)
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


def stages_from_args(args) -> tuple[dict, ...]:
    if args.thoughts < 1:
        raise SystemExit("--thoughts must be positive")
    mix_name = "mix8" if args.thoughts == COCONUT_K else f"mix{args.thoughts}"
    latent_name = "latent8" if args.thoughts == COCONUT_K else f"latent{args.thoughts}"
    return (
        {"name": mix_name, "k": args.thoughts, "keep_events": True, "steps": args.mix_steps},
        {"name": latent_name, "k": args.thoughts, "keep_events": False, "steps": args.latent_steps},
    )


def checkpoint_path(args, stage_name: str) -> Path:
    return Path(args.checkpoint_dir) / f"coconut_{stage_name}_seed{args.seed}.pt"


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
    model.config.output_hidden_states = True
    return model


def save_lora(path: Path, model, args, stage: dict, training: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "lora": {key: value.detach().cpu() for key, value in get_peft_model_state_dict(model).items()},
            "stage": stage["name"],
            "k": stage["k"],
            "args": vars(args),
            "training": training,
        },
        path,
    )


def load_lora(model, path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    set_peft_model_state_dict(model, payload["lora"])
    return payload


def embed_ids(model, ids: torch.Tensor) -> torch.Tensor:
    return model.get_input_embeddings()(ids)


def unroll_thoughts(model, prompt_embeds: torch.Tensor, k: int, *, use_cache: bool) -> torch.Tensor:
    """k last-hidden-state thoughts, identity-fed as the next input embeddings."""
    if k < 1:
        raise ValueError("k must be positive")
    if use_cache:
        outputs = model(
            inputs_embeds=prompt_embeds,
            attention_mask=torch.ones(prompt_embeds.shape[:2], dtype=torch.long, device=prompt_embeds.device),
            output_hidden_states=True,
            use_cache=True,
        )
        thoughts = []
        hidden = outputs.hidden_states[-1][:, -1:, :]
        past = outputs.past_key_values
        attention = torch.ones(
            (prompt_embeds.shape[0], prompt_embeds.shape[1] + k),
            dtype=torch.long,
            device=prompt_embeds.device,
        )
        for index in range(k):
            thoughts.append(hidden)
            outputs = model(
                inputs_embeds=hidden,
                attention_mask=attention[:, : prompt_embeds.shape[1] + index + 1],
                past_key_values=past,
                output_hidden_states=True,
                use_cache=True,
            )
            past = outputs.past_key_values
            hidden = outputs.hidden_states[-1][:, -1:, :]
        return torch.cat((prompt_embeds, *thoughts), dim=1)
    current = prompt_embeds
    for _ in range(k):
        attention = torch.ones(current.shape[:2], dtype=torch.long, device=current.device)
        outputs = model(
            inputs_embeds=current,
            attention_mask=attention,
            output_hidden_states=True,
            use_cache=False,
        )
        current = torch.cat((current, outputs.hidden_states[-1][:, -1:, :]), dim=1)
    return current


def coconut_nll(model, tokenizer, prompt: str, target: str, k: int) -> torch.Tensor:
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    target_ids = tokenizer(target, add_special_tokens=False).input_ids
    if tokenizer.eos_token_id is not None:
        target_ids = target_ids + [tokenizer.eos_token_id]
    device = model.device
    prompt_t = torch.tensor([prompt_ids], device=device)
    target_t = torch.tensor([target_ids], device=device)
    prefix = unroll_thoughts(model, embed_ids(model, prompt_t), k, use_cache=False)
    if target_t.shape[1] == 1:
        inputs = prefix
    else:
        inputs = torch.cat((prefix, embed_ids(model, target_t[:, :-1])), dim=1)
    attention = torch.ones(inputs.shape[:2], dtype=torch.long, device=device)
    logits = model(inputs_embeds=inputs, attention_mask=attention, use_cache=False).logits
    start = prefix.shape[1] - 1
    predicted = logits[:, start : start + target_t.shape[1]].float()
    if predicted.shape[1] != target_t.shape[1]:
        raise RuntimeError(
            f"Coconut NLL shape mismatch: logits {tuple(predicted.shape)} vs targets {tuple(target_t.shape)}"
        )
    return nn.functional.cross_entropy(predicted.flatten(0, 1), target_t.flatten())


def pick_example(train, tokenizer, stage, args, rng):
    k = int(stage["k"])
    for _ in range(32):
        example = train[rng.randrange(len(train))]
        prompt = render_ask_events_chat(tokenizer, example, cap=NATIVE_CAP)
        target = coconut_language_target(
            example, k=k, keep_events=bool(stage["keep_events"]), cap=NATIVE_CAP
        )
        prompt_len = len(tokenizer(prompt, add_special_tokens=False).input_ids)
        target_len = len(tokenizer(target, add_special_tokens=False).input_ids) + 1
        if prompt_len + k + target_len <= args.max_sequence_tokens:
            return example, prompt, target
    return None, None, None


def train_stage(model, tokenizer, train, stage, args) -> dict:
    model.config.use_cache = False
    model.train()
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=0.01)
    rng = random.Random(args.seed)
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    history = []
    skipped = 0
    steps = int(stage["steps"])
    for step in range(1, steps + 1):
        example, prompt, target = pick_example(train, tokenizer, stage, args, rng)
        if example is None:
            skipped += 1
            continue
        loss = coconut_nll(model, tokenizer, prompt, target, int(stage["k"]))
        (loss / args.gradient_accumulation).backward()
        if step % args.gradient_accumulation == 0 or step == steps:
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % 20 == 0 or step == steps:
            row = {
                "event": "train_step",
                "stage": stage["name"],
                "step": step,
                "loss": round(float(loss.item()), 5),
                "gpu_allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
            }
            history.append(row)
            print(json.dumps(row), flush=True)
    model.eval()
    return {
        "steps": steps,
        "k": int(stage["k"]),
        "keep_events": bool(stage["keep_events"]),
        "seconds": time.perf_counter() - started,
        "skipped": skipped,
        "history": history,
        "trainable_parameters": sum(parameter.numel() for parameter in params),
    }


@torch.inference_mode()
def generate_row(model, tokenizer, example, condition, k, args) -> dict:
    rendered = render_ask_events_chat(tokenizer, example, cap=NATIVE_CAP)
    if prompt_introduces_gold(rendered, example):
        raise SystemExit(f"gold stdout leaked into {condition} prompt for {example.example_id}")
    stdin = example.stdin.strip()
    if stdin and stdin not in rendered:
        raise SystemExit(f"stdin hidden in {condition} prompt for {example.example_id}")
    if example.source.strip() and example.source.strip() not in rendered:
        raise SystemExit(f"program missing from {condition} prompt for {example.example_id}")
    model.config.use_cache = True
    prompt_ids = tokenizer(rendered, add_special_tokens=False).input_ids
    prompt_embeds = embed_ids(model, torch.tensor([prompt_ids], device=model.device))
    inputs = unroll_thoughts(model, prompt_embeds, k, use_cache=True)
    tokens = greedy_generate(model, inputs, tokenizer.eos_token_id, args.max_new_tokens)
    response = tokenizer.decode(tokens, skip_special_tokens=True)
    predicted = extract_output(response)
    passed = outputs_match(predicted, example.gold_stdout)
    return {
        "task_id": example.task_id,
        "example_id": example.example_id,
        "condition": condition,
        "prompt_tokens": int(inputs.shape[1]),
        "event_count": example.event_count,
        "thoughts": k,
        "slot_count": k,
        "response": response,
        "predicted": predicted,
        "gold": example.gold_stdout,
        "passed": passed,
        "outcome": "pass" if passed else "fail",
        "max_new_tokens": args.max_new_tokens,
    }


def eval_stage(model, tokenizer, examples, stage, args, rows_path, done) -> int:
    condition = condition_for_stage(stage["name"])
    written = 0
    model.eval()
    with rows_path.open("a") as handle:
        for example in examples:
            key = (example.example_id, condition)
            if key in done:
                continue
            row = generate_row(model, tokenizer, example, condition, int(stage["k"]), args)
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            done.add(key)
            written += 1
            print(
                json.dumps(
                    {
                        "event": "row",
                        "condition": condition,
                        "example_id": example.example_id,
                        "task_id": example.task_id,
                        "passed": row["passed"],
                        "thoughts": row["thoughts"],
                    }
                ),
                flush=True,
            )
    return written


def copy_control(stdout_rows_path: Path, rows_path: Path, done: set[tuple[str, str]]) -> int:
    source = [json.loads(line) for line in stdout_rows_path.read_text().splitlines() if line.strip()]
    copied_rows = copy_stdout_control_rows(source)
    written = 0
    with rows_path.open("a") as handle:
        for row in copied_rows:
            key = (row["example_id"], SFT_STDOUT)
            if key in done:
                continue
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            done.add(key)
            written += 1
    return written


def unload(model) -> None:
    del model
    torch.cuda.empty_cache()


def assert_identity_thoughts(model) -> None:
    hidden = int(model.config.hidden_size)
    embed = int(model.get_input_embeddings().weight.shape[-1])
    if hidden != embed:
        raise SystemExit(
            f"Coconut identity thoughts need hidden==embed; got hidden={hidden} embed={embed}"
        )


def main() -> None:
    args = parse_args()
    stages = stages_from_args(args)
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

    stdout_rows_path = Path(args.stdout_rows)
    if not stdout_rows_path.exists():
        raise SystemExit(f"missing sft_stdout control rows {stdout_rows_path}")
    init_ckpt = Path(args.init_checkpoint)
    if not init_ckpt.exists():
        raise SystemExit(f"missing event-chain init LoRA {init_ckpt}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "rows.jsonl"
    done = existing_keys(rows_path) if args.resume else set()
    if not args.resume and rows_path.exists():
        rows_path.unlink()
        done = set()
    copied = copy_control(stdout_rows_path, rows_path, done)
    print(json.dumps({"event": "control_copied", "rows": copied, "path": str(stdout_rows_path)}), flush=True)

    started = time.perf_counter()
    training = {}
    peak = 0.0
    model = None
    tokenizer = None

    def ensure_model(load_from: Path):
        nonlocal model, tokenizer
        if model is None:
            model, tokenizer, _ = load_model(args)
            model = attach_lora(model, args.lora_rank, args.lora_alpha)
            assert_identity_thoughts(model)
            load_lora(model, load_from)
            print(json.dumps({"event": "lora_loaded", "path": str(load_from)}), flush=True)
        return model, tokenizer

    previous_ckpt = init_ckpt
    for stage in stages:
        condition = condition_for_stage(stage["name"])
        ckpt = checkpoint_path(args, stage["name"])
        need_eval = any((example.example_id, condition) not in done for example in test)
        load_existing = args.skip_train or (args.resume and ckpt.exists())
        if load_existing and not ckpt.exists():
            raise SystemExit(f"missing LoRA checkpoint {ckpt}")
        if load_existing or not args.skip_train or need_eval:
            if load_existing:
                if model is not None:
                    unload(model)
                    model = None
                model, tokenizer = ensure_model(ckpt)
                training[stage["name"]] = {"loaded": str(ckpt)}
            else:
                model, tokenizer = ensure_model(previous_ckpt)
                print(
                    json.dumps(
                        {
                            "event": "train_start",
                            "stage": stage["name"],
                            "k": stage["k"],
                            "steps": stage["steps"],
                            "keep_events": stage["keep_events"],
                            "n_train": len(train),
                        }
                    ),
                    flush=True,
                )
                training[stage["name"]] = train_stage(model, tokenizer, train, stage, args)
                save_lora(ckpt, model, args, stage, training[stage["name"]])
                print(
                    json.dumps(
                        {
                            "event": "train_done",
                            "stage": stage["name"],
                            "checkpoint": str(ckpt),
                            "seconds": training[stage["name"]]["seconds"],
                            "trainable_parameters": training[stage["name"]]["trainable_parameters"],
                        }
                    ),
                    flush=True,
                )
            if need_eval:
                eval_stage(model, tokenizer, test, stage, args, rows_path, done)
            peak = max(peak, torch.cuda.max_memory_allocated() / 2**30)
        previous_ckpt = ckpt

    if model is not None:
        unload(model)

    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    by_condition = {}
    for row in rows:
        by_condition[row["condition"]] = by_condition.get(row["condition"], 0) + 1
    required = {SFT_STDOUT, "coconut_latent", "coconut_mix"}
    for condition in required:
        if by_condition.get(condition, 0) != len(test) and not args.limit:
            raise SystemExit(
                f"{condition} has {by_condition.get(condition, 0)} rows; expected {len(test)}"
            )
    decision = summarize_coconut(rows)
    summary = {
        "args": vars(args),
        "call": decision["call"],
        "call_note": decision["note"],
        "elapsed_seconds": time.perf_counter() - started,
        "eval_examples": len(test),
        "eval_tasks": len({example.task_id for example in test}),
        "init_checkpoint": str(init_ckpt),
        "panel_sha256": test_sha,
        "peak_allocated_gib": peak,
        "stages": list(stages),
        "summary": decision,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "train_examples": len(train),
        "train_sha256": train_sha,
        "training": training,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "done", "call": decision["call"], "summary": decision}), flush=True)


if __name__ == "__main__":
    main()
