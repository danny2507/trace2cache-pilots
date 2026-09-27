#!/usr/bin/env python3
"""LDP Gate 1: encode only the oracle REPLACE; frozen 3B applies Z.

No buggy source, public test, or trace in the encoder. No const residual.
Shuffle is another task's REPLACE. Text controls are copied, not retrained.
Does not train on the frozen 128. Does not overwrite the sim or RunBugRun panels.
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
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from trace2cache.latent import trainable_parameter_count
from trace2cache.ldp import (
    COMPACT_TEXT,
    CORRUPTED_LATENT,
    IO_TEXT,
    LDP_K,
    NO_EVIDENCE,
    SHUFFLED_LATENT,
    TEXT_CONTROLS,
    TRUE_LATENT,
    copy_text_control_rows,
    corrupt_edit_sketch,
    render_ldp_gate1_encode_text,
    summarize_ldp_gate1,
)
from trace2cache.mbpp_generalization import (
    MARKER,
    RepairExample,
    different_task_example,
    edit_sketch,
    evaluate_hidden,
    extract_patch,
    allowed_calls_for_tests,
    read_examples,
    render_chat,
)
from trace2cache.receiver_training import splice_prompt

sys.path.insert(0, str(Path(__file__).parent))
from run_mbpp_generalization import (
    clip_events,
    existing_keys,
    repair_loss_from_latent,
    sketch_loss_from_latent,
    within_budget,
)
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
        "--output-dir", default="artifacts/mbpp_generalization/ldp_gate1_seed1001"
    )
    parser.add_argument(
        "--checkpoint", default="checkpoints/mbpp_generalization/ldp_gate1_seed1001.pt"
    )
    parser.add_argument("--text-rows", default=TEXT_CONTROL_ROWS)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--contrastive-weight", type=float, default=3.0)
    parser.add_argument("--contrastive-margin", type=float, default=0.1)
    parser.add_argument("--sketch-weight", type=float, default=1.0)
    parser.add_argument("--latent-slots", type=int, default=LDP_K)
    parser.add_argument("--max-events", type=int, default=96)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-sequence-tokens", type=int, default=768)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--limit", type=int, default=FROZEN_PANEL_N)
    parser.add_argument("--min-free-gib", type=float, default=8.0)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--include-specification",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--train-hide-public-test",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def oracle_sketch(example: RepairExample) -> str:
    return edit_sketch(example.buggy_source, example.target_source)


class LDPGate1Encoder(nn.Module):
    """K gist tokens over REPLACE-only text. Z = W h_gist + b. No const residual."""

    def __init__(self, gist_init: torch.Tensor):
        super().__init__()
        if gist_init.ndim != 2:
            raise ValueError("gist_init must have shape [slots, hidden]")
        hidden = int(gist_init.shape[1])
        self.gist_embeds = nn.Parameter(gist_init.detach().float().clone())
        self.project = nn.Linear(hidden, hidden)
        nn.init.eye_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    @property
    def num_latents(self) -> int:
        return int(self.gist_embeds.shape[0])

    def encode(self, model, tokenizer, sketch: str) -> torch.Tensor:
        text = render_ldp_gate1_encode_text(sketch)
        ids = tokenizer(text, add_special_tokens=False).input_ids
        if tokenizer.eos_token_id is not None:
            ids = ids + [tokenizer.eos_token_id]
        embedding = model.get_input_embeddings()
        prompt = embedding(torch.tensor([ids], device=model.device))
        gist = self.gist_embeds.to(dtype=prompt.dtype, device=prompt.device).unsqueeze(0)
        inputs = torch.cat((prompt, gist), dim=1)
        attention = torch.ones(inputs.shape[:2], dtype=torch.long, device=model.device)
        hidden = model.model(
            inputs_embeds=inputs, attention_mask=attention, use_cache=False
        ).last_hidden_state
        return self.project(hidden[:, -self.num_latents :, :].float())


def encode_for_condition(encoder, model, tokenizer, example, shuffled, condition: str) -> torch.Tensor:
    if condition == TRUE_LATENT:
        sketch = oracle_sketch(example)
    elif condition == SHUFFLED_LATENT:
        sketch = oracle_sketch(shuffled)
    elif condition == CORRUPTED_LATENT:
        sketch = corrupt_edit_sketch(oracle_sketch(example))
    else:
        raise ValueError(f"no Gate 1 encode for {condition}")
    return encoder.encode(model, tokenizer, sketch)


def make_encoder(model, tokenizer, slots: int) -> LDPGate1Encoder:
    anchor_ids = tokenizer(
        " runtime evidence observed behavior state values result details",
        add_special_tokens=False,
    ).input_ids[:slots]
    anchor_ids += [anchor_ids[-1]] * (slots - len(anchor_ids))
    anchors = model.get_input_embeddings()(torch.tensor(anchor_ids, device=model.device)).detach()
    return LDPGate1Encoder(anchors).to(model.device)


def train_encoder(model, tokenizer, encoder, train, args) -> dict:
    args.edit_span_loss = True
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    model.requires_grad_(False)
    trainable = [parameter for parameter in encoder.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.01)
    rng = random.Random(args.seed)
    encoder.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    history = []
    train_public_test = not args.train_hide_public_test
    ema_rank = None
    ema_cos = None
    best_score = None
    best_step = None
    best_state = None
    skipped = 0
    for step in range(1, args.steps + 1):
        index = rng.randrange(len(train))
        example = train[index]
        shuffled = different_task_example(train, index)
        true_sketch = oracle_sketch(example)
        shuffled_sketch = oracle_sketch(shuffled)
        if true_sketch == shuffled_sketch:
            skipped += 1
            continue
        true_latent = encoder.encode(model, tokenizer, true_sketch)
        shuffled_latent = encoder.encode(model, tokenizer, shuffled_sketch)
        true_loss = repair_loss_from_latent(
            model,
            tokenizer,
            true_latent,
            example,
            include_specification=args.include_specification,
            include_public_test=train_public_test,
            edit_span=True,
        )
        sketch_true = sketch_loss_from_latent(model, tokenizer, true_latent, example)
        sketch_shuffled = sketch_loss_from_latent(model, tokenizer, shuffled_latent, example)
        ranking = nn.functional.relu(args.contrastive_margin + sketch_true - sketch_shuffled)
        true_flat = true_latent.float().reshape(1, -1)
        shuffled_flat = shuffled_latent.float().reshape(1, -1)
        cosine_value = nn.functional.cosine_similarity(true_flat, shuffled_flat).item()
        loss = true_loss + args.sketch_weight * sketch_true + args.contrastive_weight * ranking
        (loss / args.gradient_accumulation).backward()
        if step % args.gradient_accumulation == 0 or step == args.steps:
            nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % 20 == 0 or step == args.steps:
            rank_value = ranking.item()
            row = {
                "event": "train_step",
                "step": step,
                "loss": round(float(loss.item()), 5),
                "true_patch_loss": round(float(true_loss.item()), 5),
                "sketch_true_loss": round(float(sketch_true.item()), 5),
                "sketch_shuffled_loss": round(float(sketch_shuffled.item()), 5),
                "ranking_loss": round(rank_value, 5),
                "true_shuffled_cosine": round(cosine_value, 5),
                "gpu_allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
            }
            history.append(row)
            print(json.dumps(row), flush=True)
            if ema_rank is None:
                ema_rank = rank_value
                ema_cos = cosine_value
            else:
                ema_rank = 0.8 * ema_rank + 0.2 * rank_value
                ema_cos = 0.8 * ema_cos + 0.2 * cosine_value
            if step >= 200:
                score = ema_rank + 0.15 * ema_cos
                if best_score is None or score < best_score:
                    best_score = score
                    best_step = step
                    best_state = {
                        key: value.detach().cpu().clone() for key, value in encoder.state_dict().items()
                    }
    if best_state is not None:
        encoder.load_state_dict(best_state)
        print(
            json.dumps(
                {
                    "event": "best_checkpoint_restored",
                    "step": best_step,
                    "score": round(best_score, 5),
                    "ema_rank": round(ema_rank, 5),
                    "ema_cos": round(ema_cos, 5),
                }
            ),
            flush=True,
        )
    encoder.eval()
    model.eval()
    model.config.use_cache = False
    return {
        "steps": args.steps,
        "seconds": time.perf_counter() - started,
        "skipped": skipped,
        "history": history,
        "best_step": best_step,
        "best_score": None if best_score is None else round(best_score, 5),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
    }


@torch.inference_mode()
def generate_row(model, tokenizer, encoder, example, shuffled, condition, args) -> dict:
    rendered = render_chat(
        tokenizer,
        example,
        MARKER,
        include_specification=args.include_specification,
        include_public_test=True,
        evidence_kind="runtime",
    )
    sketch = oracle_sketch(example)
    if sketch != "NO_EDIT" and sketch in rendered:
        raise SystemExit(f"oracle REPLACE leaked into {condition} prompt for {example.example_id}")
    if example.target_source.strip() and example.target_source.strip() in rendered:
        raise SystemExit(f"canonical leaked into {condition} prompt for {example.example_id}")
    latent = encode_for_condition(encoder, model, tokenizer, example, shuffled, condition)
    latent = latent.to(model.get_input_embeddings().weight.dtype)
    inputs = splice_prompt(model, tokenizer, rendered, latent)
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
        "condition": condition,
        "prompt_tokens": int(inputs.shape[1]),
        "latent_slots": args.latent_slots,
        "encode_sketch": (
            oracle_sketch(shuffled)
            if condition == SHUFFLED_LATENT
            else corrupt_edit_sketch(sketch)
            if condition == CORRUPTED_LATENT
            else sketch
        ),
        "event_count": len(example.events),
        "response": response,
        "source": source,
        "validation": validation,
        "passed": bool(validation.get("passed")),
        "outcome": outcome,
        "max_new_tokens": args.max_new_tokens,
    }


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


def main() -> None:
    args = parse_args()
    if args.latent_slots < 1:
        raise SystemExit("--latent-slots must be positive")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("A BF16 CUDA GPU is required")
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes / 2**30 < args.min_free_gib:
        raise SystemExit(f"only {free_bytes / 2**30:.1f} GiB free; need {args.min_free_gib:.1f} GiB")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cohort = Path(args.cohort)
    train_path = cohort / "train.jsonl"
    test_path = cohort / "test.jsonl"
    manifest = json.loads((cohort / "manifest.json").read_text())
    train_sha = sha256_file(train_path)
    test_sha = sha256_file(test_path)
    if manifest.get("sha256") != FROZEN_COHORT_SHA256:
        raise SystemExit(f"frozen cohort sha mismatch: {manifest.get('sha256')} != {FROZEN_COHORT_SHA256}")
    if test_sha != FROZEN_TEST_SHA256:
        raise SystemExit(f"frozen MBPP test sha mismatch: {test_sha} != {FROZEN_TEST_SHA256}")
    if "runbugrun" in str(cohort) or "execution_sim" in str(cohort):
        raise SystemExit("LDP Gate 1 uses cohort_v1; refusing to touch RunBugRun or sim panels")

    train = [clip_events(example, args.max_events) for example in read_examples(train_path)]
    test = [clip_events(example, args.max_events) for example in read_examples(test_path)]
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

    model_path = resolve_local_model(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.bfloat16
    ).to("cuda").eval()
    model.requires_grad_(False)
    model.config.use_cache = False
    print(
        json.dumps(
            {
                "event": "model_loaded",
                "model": args.model,
                "eval_examples": len(test),
                "eval_tasks": len({example.task_id for example in test}),
                "train_examples": len(train),
                "gpu_free_gib": round(free_bytes / 2**30, 2),
            }
        ),
        flush=True,
    )

    train = [
        example
        for example in train
        if within_budget(
            tokenizer,
            example,
            args.max_sequence_tokens,
            include_specification=args.include_specification,
            include_public_test=not args.train_hide_public_test,
        )
    ]
    if len({example.task_id for example in train}) < 2:
        raise SystemExit("training requires at least two in-budget train tasks")

    encoder = make_encoder(model, tokenizer, args.latent_slots)
    checkpoint = Path(args.checkpoint)
    training = None
    started = time.perf_counter()
    load_existing = args.skip_train or (args.resume and checkpoint.exists())
    if load_existing:
        if not checkpoint.exists():
            raise SystemExit(f"missing LDP Gate 1 checkpoint {checkpoint}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        encoder.load_state_dict(payload["encoder"], strict=True)
        encoder.eval()
        training = {"loaded": str(checkpoint)}
        print(json.dumps({"event": "checkpoint_loaded", "path": str(checkpoint)}), flush=True)
    else:
        print(
            json.dumps(
                {
                    "event": "train_start",
                    "steps": args.steps,
                    "n_train": len(train),
                    "slots": args.latent_slots,
                }
            ),
            flush=True,
        )
        training = train_encoder(model, tokenizer, encoder, train, args)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "encoder": encoder.state_dict(),
                "args": vars(args),
                "cohort_sha256": FROZEN_COHORT_SHA256,
                "trainable_parameters": trainable_parameter_count(encoder),
                "train_examples": len(train),
                "train_tasks": len({example.task_id for example in train}),
                "training": training,
            },
            checkpoint,
        )
        print(json.dumps({"event": "checkpoint_written", "path": str(checkpoint)}), flush=True)

    encoder.eval()
    model.eval()
    for index, example in enumerate(test):
        shuffled = different_task_example(test, index)
        for condition in (TRUE_LATENT, SHUFFLED_LATENT, CORRUPTED_LATENT):
            key = (example.example_id, condition)
            if key in done:
                continue
            began = time.perf_counter()
            row = generate_row(model, tokenizer, encoder, example, shuffled, condition, args)
            row["seconds"] = time.perf_counter() - began
            row["seed"] = args.seed
            with rows_path.open("a") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            done.add(key)
            print(
                json.dumps(
                    {
                        "event": "generation",
                        "task_id": example.task_id,
                        "example_id": example.example_id,
                        "condition": condition,
                        "passed": row["passed"],
                        "outcome": row["outcome"],
                    }
                ),
                flush=True,
            )

    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    by_condition = {}
    for row in rows:
        by_condition[row["condition"]] = by_condition.get(row["condition"], 0) + 1
    expected = len(test)
    for condition in (NO_EVIDENCE, IO_TEXT, COMPACT_TEXT, TRUE_LATENT, SHUFFLED_LATENT, CORRUPTED_LATENT):
        if by_condition.get(condition, 0) != expected:
            raise SystemExit(
                f"{condition} has {by_condition.get(condition, 0)} rows; expected {expected}"
            )
    decision = summarize_ldp_gate1(rows)
    summary = {
        "args": vars(args),
        "call": decision["call"],
        "call_note": decision["note"],
        "cohort_sha256": FROZEN_COHORT_SHA256,
        "elapsed_seconds": time.perf_counter() - started,
        "eval_examples": len(test),
        "eval_tasks": len({example.task_id for example in test}),
        "panel_sha256": test_sha,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "summary": decision,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "train_examples": len(train),
        "train_sha256": train_sha,
        "trainable_parameters": trainable_parameter_count(encoder),
        "training": training,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "done", "call": decision["call"], "summary": decision}), flush=True)


if __name__ == "__main__":
    main()
