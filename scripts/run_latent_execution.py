#!/usr/bin/env python3
"""Distill interpreter events into a LoRA latent rollout.

`--mode full` trains a matched `code_sft` LoRA and a `latent` LoRA on the
disjoint 320, then evals on the frozen 128. Latent training teacher-forces
resampled native event embeddings and a projector that writes them. At test
`rollout` uses no tracer. `teacher` is an oracle splice of gold event vectors.
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

from trace2cache.execution_sim import (
    different_task_example,
    extract_output,
    native_event_texts,
    normalize_output,
    outputs_match,
    read_examples,
)
from trace2cache.latent_execution import (
    DISTILL_K,
    OBJECTIVES,
    distill_prompt,
    teacher_texts,
)

sys.path.insert(0, str(Path(__file__).parent))
from analyze_mbpp_generalization import mcnemar
from run_pilot1 import resolve_local_model
from run_repair_latent_pool import greedy_generate


class HiddenToEmbed(nn.Module):
    """Map a last-token hidden state into token-embedding RMS."""

    def __init__(self, width: int, embed_rms: float):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.proj = nn.Linear(width, width)
        nn.init.zeros_(self.proj.bias)
        nn.init.eye_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(float(embed_rms)))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        x = self.norm(hidden.float())
        x = self.proj(x)
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        return (self.scale * x).to(dtype=hidden.dtype)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument(
        "--cohort", default="artifacts/mbpp_generalization/execution_sim_runbugrun_v1"
    )
    parser.add_argument("--mode", choices=("train", "eval", "full"), default="full")
    parser.add_argument("--objective", choices=OBJECTIVES, default="latent")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1280)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--alignment-weight", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--latent-slots", type=int, default=DISTILL_K)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-sequence-tokens", type=int, default=1536)
    parser.add_argument("--min-free-gib", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument(
        "--checkpoint-dir", default="checkpoints/mbpp_generalization"
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/mbpp_generalization/execution_sim_distill_seed1001",
    )
    return parser.parse_args()


def existing_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    return {
        (row["example_id"], row["condition"])
        for row in (json.loads(line) for line in path.read_text().splitlines() if line.strip())
    }


@torch.no_grad()
def native_table(model, tokenizer, contents: list[str], batch_size: int = 64) -> torch.Tensor:
    embedding = model.get_input_embeddings()
    vectors = []
    for start in range(0, len(contents), batch_size):
        encoded = tokenizer(
            contents[start : start + batch_size],
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(model.device)
        token_vectors = embedding(encoded["input_ids"]).float()
        mask = encoded["attention_mask"].unsqueeze(-1)
        vectors.append((token_vectors * mask).sum(1) / mask.sum(1).clamp_min(1))
    return torch.cat(vectors)


def stack_teacher(table, content_to_id, texts, device, dtype) -> torch.Tensor:
    missing = [text for text in texts if text not in content_to_id]
    if missing:
        raise KeyError(f"event text missing from native table: {missing[0][:80]}")
    ids = torch.tensor([content_to_id[text] for text in texts], device=device)
    return table[ids].unsqueeze(0).to(dtype)


def target_ids(tokenizer, gold: str, device: torch.device) -> torch.Tensor:
    text = f"```\n{normalize_output(gold)}\n```"
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    eos = torch.tensor([[tokenizer.eos_token_id]], device=device)
    return torch.cat((ids, eos), dim=1)


def embed_ids(model, ids: torch.Tensor) -> torch.Tensor:
    return model.get_input_embeddings()(ids)


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


def make_projector(model) -> HiddenToEmbed:
    embedding = model.get_input_embeddings()
    rms = float(embedding.weight.detach().float().pow(2).mean().sqrt().clamp(min=1e-3).item())
    width = embedding.weight.shape[-1]
    return HiddenToEmbed(width, rms).to(device=model.device, dtype=torch.float32)


def trainable_params(model, projector):
    params = [p for p in model.parameters() if p.requires_grad]
    if projector is not None:
        params.extend(projector.parameters())
    return params


def forward_loss(model, projector, tokenizer, table, content_to_id, example, teacher_source, args):
    prompt = distill_prompt(tokenizer, example)
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
        model.device
    )
    prompt_embeds = embed_ids(model, prompt_ids)
    prompt_len = prompt_embeds.shape[1]
    targets = target_ids(tokenizer, example.gold_stdout, model.device)
    stdout_embeds = embed_ids(model, targets[:, :-1])
    k = 0 if args.objective == "code_sft" else args.latent_slots
    dtype = prompt_embeds.dtype
    if k:
        texts = teacher_texts(teacher_source.events, k)
        teacher = stack_teacher(table, content_to_id, texts, model.device, dtype)
        inputs = torch.cat((prompt_embeds, teacher, stdout_embeds), dim=1)
    else:
        teacher = None
        inputs = torch.cat((prompt_embeds, stdout_embeds), dim=1)
    attention = torch.ones(inputs.shape[:2], dtype=torch.long, device=model.device)
    outputs = model(
        inputs_embeds=inputs,
        attention_mask=attention,
        output_hidden_states=bool(k),
        use_cache=False,
    )
    start = prompt_len + k - 1
    predicted = outputs.logits[:, start : start + targets.shape[1]].float()
    nll = nn.functional.cross_entropy(predicted.flatten(0, 1), targets.flatten())
    align = nll.new_tensor(0.0)
    cosine = 0.0
    if k and projector is not None:
        hidden = outputs.hidden_states[-1][:, prompt_len - 1 : prompt_len - 1 + k, :]
        predicted_z = projector(hidden)
        cosine_t = nn.functional.cosine_similarity(
            predicted_z.float(), teacher.float(), dim=-1
        )
        align = (1 - cosine_t).mean()
        cosine = float(cosine_t.mean().item())
        return nll + args.alignment_weight * align, nll, align, cosine
    return nll, nll, align, cosine


def train_objective(model, projector, tokenizer, table, content_to_id, train, args) -> dict:
    params = trainable_params(model, projector)
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=0.01)
    rng = random.Random(args.seed)
    model.train()
    if projector is not None:
        projector.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    history = []
    for step in range(1, args.steps + 1):
        index = rng.randrange(len(train))
        example = train[index]
        if args.objective == "shuffled":
            teacher_source = different_task_example(train, index)
        else:
            teacher_source = example
        loss, nll, align, cosine = forward_loss(
            model, projector, tokenizer, table, content_to_id, example, teacher_source, args
        )
        (loss / args.gradient_accumulation).backward()
        if step % args.gradient_accumulation == 0 or step == args.steps:
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % 20 == 0 or step == args.steps:
            row = {
                "step": step,
                "objective": args.objective,
                "loss": round(float(loss.item()), 5),
                "nll": round(float(nll.item()), 5),
                "align": round(float(align.item()), 5),
                "cosine": round(cosine, 5),
                "gpu_allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
            }
            history.append(row)
            print(json.dumps(row), flush=True)
    model.eval()
    if projector is not None:
        projector.eval()
    return {
        "steps": args.steps,
        "seconds": time.perf_counter() - started,
        "history": history,
        "trainable_parameters": sum(p.numel() for p in params),
    }


@torch.no_grad()
def rollout_prefix(model, projector, prompt_embeds, k: int) -> torch.Tensor:
    if k <= 0:
        return prompt_embeds
    outputs = model(inputs_embeds=prompt_embeds, output_hidden_states=True, use_cache=True)
    past = outputs.past_key_values
    hidden = outputs.hidden_states[-1][:, -1, :]
    slots = []
    for _ in range(k):
        slot = projector(hidden).unsqueeze(1).to(prompt_embeds.dtype)
        slots.append(slot)
        outputs = model(
            inputs_embeds=slot,
            past_key_values=past,
            output_hidden_states=True,
            use_cache=True,
        )
        past = outputs.past_key_values
        hidden = outputs.hidden_states[-1][:, -1, :]
    return torch.cat((prompt_embeds, *slots), dim=1)


def generate_one(
    model, projector, tokenizer, table, content_to_id, example, condition, args
) -> dict:
    prompt = distill_prompt(tokenizer, example)
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
        model.device
    )
    prompt_embeds = embed_ids(model, prompt_ids)
    k = args.latent_slots
    dtype = prompt_embeds.dtype
    rollout_cosine = None
    if condition.endswith("rollout"):
        inputs = rollout_prefix(model, projector, prompt_embeds, k)
        if table is not None:
            teacher = stack_teacher(
                table,
                content_to_id,
                teacher_texts(example.events, k),
                model.device,
                dtype,
            )
            latents = inputs[:, prompt_embeds.shape[1] :, :]
            rollout_cosine = float(
                nn.functional.cosine_similarity(
                    latents.float().reshape(1, -1), teacher.float().reshape(1, -1)
                ).item()
            )
    elif condition.endswith("teacher"):
        teacher = stack_teacher(
            table,
            content_to_id,
            teacher_texts(example.events, k),
            model.device,
            dtype,
        )
        inputs = torch.cat((prompt_embeds, teacher), dim=1)
    else:
        inputs = prompt_embeds
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
        "slot_count": 0 if condition.endswith("code") or condition == "code_sft" else k,
        "rollout_cosine": rollout_cosine,
        "response": response,
        "predicted": predicted,
        "gold": example.gold_stdout,
        "passed": passed,
        "outcome": "pass" if passed else "fail",
    }


def eval_conditions(objective: str) -> tuple[str, ...]:
    if objective == "code_sft":
        return ("code_sft",)
    return (f"{objective}_code", f"{objective}_rollout", f"{objective}_teacher")


def summarize(rows: list[dict]) -> dict:
    summaries = {}
    for condition in sorted({row["condition"] for row in rows}):
        selected = [row for row in rows if row["condition"] == condition]
        summaries[condition] = {
            "correct": sum(row["passed"] for row in selected),
            "examples": len(selected),
            "tasks": len({row["task_id"] for row in selected}),
            "mean_prompt_tokens": sum(row["prompt_tokens"] for row in selected) / len(selected),
            "mean_rollout_cosine": (
                sum(row["rollout_cosine"] for row in selected if row.get("rollout_cosine") is not None)
                / max(1, sum(row.get("rollout_cosine") is not None for row in selected))
            ),
        }
    return summaries


def score_mcnemar(rows: list[dict]) -> dict:
    by_task: dict[int, dict[str, bool]] = {}
    for row in rows:
        by_task.setdefault(row["task_id"], {})[row["condition"]] = bool(row["passed"])
    present = sorted({row["condition"] for row in rows})
    pairs = []
    for left, right in (
        ("latent_rollout", "code_sft"),
        ("latent_teacher", "code_sft"),
        ("latent_rollout", "latent_teacher"),
        ("latent_code", "code_sft"),
        ("shuffled_rollout", "code_sft"),
        ("shuffled_rollout", "latent_rollout"),
    ):
        if left in present and right in present:
            pairs.append((left, right))
    comparisons = {}
    for left, right in pairs:
        left_ids = []
        right_ids = []
        for task_id, outcomes in by_task.items():
            if left not in outcomes or right not in outcomes:
                continue
            if outcomes[left] and not outcomes[right]:
                left_ids.append(task_id)
            if outcomes[right] and not outcomes[left]:
                right_ids.append(task_id)
        comparisons[f"{left} vs {right}"] = {
            "left_only": len(left_ids),
            "right_only": len(right_ids),
            "p": mcnemar(len(left_ids), len(right_ids)),
            "left_ids": left_ids,
            "right_ids": right_ids,
        }
    counts = {
        condition: sum(row["passed"] for row in rows if row["condition"] == condition)
        for condition in present
    }
    rollout = comparisons.get("latent_rollout vs code_sft", {})
    teacher = comparisons.get("latent_teacher vs code_sft", {})
    def wins(comp: dict) -> bool:
        return comp.get("left_only", 0) > comp.get("right_only", 0) and comp.get("p", 1) < 0.05

    if "latent_rollout" in present and "code_sft" in present:
        if wins(rollout):
            call = "distill_pass"
            note = (
                "Latent rollout beats code_sft. Train shuffled before claiming "
                "the tracer was distilled rather than extra compute."
            )
        elif wins(teacher):
            call = "reader_pass"
            note = (
                "LoRA reads gold event embeddings but rollout without the tracer "
                "does not beat code_sft. Writer/distillation failed."
            )
        else:
            call = "distill_fail"
            note = "LoRA latent rollout did not beat matched code_sft."
    else:
        call = "incomplete"
        note = "Need both latent_rollout and code_sft to call."
    return {
        "repair_at_1": counts,
        "comparisons": comparisons,
        "call": call,
        "note": note,
    }


def load_base(args):
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
    model.config.use_cache = False
    return model, tokenizer, free_bytes


def build_text_table(model, tokenizer, groups):
    texts = list(
        dict.fromkeys(
            text
            for group in groups
            for example in group
            for text in native_event_texts(example.events)
        )
    )
    if not texts:
        raise SystemExit("no native event texts")
    table = native_table(model, tokenizer, texts)
    return table, {text: index for index, text in enumerate(texts)}


def checkpoint_path(args, objective: str) -> Path:
    return Path(args.checkpoint_dir) / f"distill_{objective}_seed{args.seed}.pt"


def save_checkpoint(path: Path, model, projector, args, training: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "lora": {key: value.detach().cpu() for key, value in get_peft_model_state_dict(model).items()},
        "projector": None if projector is None else {
            key: value.detach().cpu() for key, value in projector.state_dict().items()
        },
        "args": vars(args),
        "training": training,
    }
    torch.save(payload, path)
    print(json.dumps({"event": "checkpoint_written", "path": str(path)}), flush=True)


def load_checkpoint(path: Path, model, projector):
    payload = torch.load(path, map_location=model.device, weights_only=False)
    set_peft_model_state_dict(model, payload["lora"])
    if projector is not None and payload.get("projector"):
        projector.load_state_dict(payload["projector"])
    return payload.get("training")


def within_budget(tokenizer, example, max_sequence_tokens: int, k: int) -> bool:
    prompt = distill_prompt(tokenizer, example)
    target = f"```\n{normalize_output(example.gold_stdout)}\n```"
    length = (
        len(tokenizer(prompt, add_special_tokens=False).input_ids)
        + k
        + len(tokenizer(target, add_special_tokens=False).input_ids)
    )
    return length <= max_sequence_tokens


def run_eval(model, projector, tokenizer, table, content_to_id, examples, objective, args, rows_path):
    conditions = eval_conditions(objective)
    done = existing_keys(rows_path)
    written = 0
    model.eval()
    if projector is not None:
        projector.eval()
    model.config.use_cache = True
    with rows_path.open("a") as handle:
        for example in examples:
            for condition in conditions:
                key = (example.example_id, condition)
                if key in done:
                    continue
                t0 = time.time()
                row = generate_one(
                    model, projector, tokenizer, table, content_to_id, example, condition, args
                )
                row["seconds"] = time.time() - t0
                row["seed"] = args.seed
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
                done.add(key)
                written += 1
                print(
                    json.dumps(
                        {
                            "event": "row",
                            "example_id": example.example_id,
                            "condition": condition,
                            "passed": row["passed"],
                            "prompt_tokens": row["prompt_tokens"],
                            "seconds": round(row["seconds"], 2),
                        }
                    ),
                    flush=True,
                )
    model.config.use_cache = False
    return written


def train_and_maybe_eval(args, objective, train, eval_examples, output_dir, rows_path):
    args.objective = objective
    model, tokenizer, free_bytes = load_base(args)
    table = content_to_id = None
    if objective != "code_sft":
        table, content_to_id = build_text_table(model, tokenizer, (train, eval_examples))
    model = attach_lora(model, args.lora_rank, args.lora_alpha)
    projector = make_projector(model) if objective != "code_sft" else None
    print(
        json.dumps(
            {
                "event": "model_loaded",
                "objective": objective,
                "eval_examples": len(eval_examples),
                "train_examples": len(train),
                "gpu_free_gib": round(free_bytes / 2**30, 2),
                "trainable_parameters": sum(p.numel() for p in trainable_params(model, projector)),
            }
        ),
        flush=True,
    )
    ckpt = checkpoint_path(args, objective)
    training = None
    if args.mode in ("train", "full"):
        training = train_objective(model, projector, tokenizer, table, content_to_id, train, args)
        save_checkpoint(ckpt, model, projector, args, training)
        if args.mode == "train":
            del model
            torch.cuda.empty_cache()
            return training, 0
    else:
        training = load_checkpoint(ckpt, model, projector)
    written = 0
    if args.mode in ("eval", "full"):
        written = run_eval(
            model, projector, tokenizer, table, content_to_id, eval_examples, objective, args, rows_path
        )
    del model
    if projector is not None:
        del projector
    torch.cuda.empty_cache()
    return training, written


def main() -> None:
    args = parse_args()
    cohort = Path(args.cohort)
    eval_examples = read_examples(cohort / f"{args.split}.jsonl")
    eval_examples.sort(key=lambda example: example.task_id)
    if args.limit:
        eval_examples = eval_examples[: args.limit]
    train_path = cohort / "train.jsonl"
    if not train_path.exists() or train_path.stat().st_size == 0:
        raise SystemExit(f"disjoint train split missing: {train_path}")
    train = [example for example in read_examples(train_path) if example.events]
    eval_task_ids = {example.task_id for example in eval_examples}
    overlap = {example.task_id for example in train} & eval_task_ids
    if overlap:
        raise SystemExit(f"train/eval task_id overlap: {sorted(overlap)[:8]}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "rows.jsonl"
    objectives = ("code_sft", "latent") if args.mode == "full" else (args.objective,)
    started = time.time()
    trainings = {}
    written = 0
    from transformers import AutoTokenizer as _T

    tokenizer = _T.from_pretrained(resolve_local_model(args.model), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train = [
        example
        for example in train
        if within_budget(tokenizer, example, args.max_sequence_tokens, args.latent_slots)
    ]
    if len(train) < 32:
        raise SystemExit(f"train split too small after filters: {len(train)}")
    del tokenizer
    for objective in objectives:
        training, n = train_and_maybe_eval(
            args, objective, train, eval_examples, output_dir, rows_path
        )
        trainings[objective] = training
        written += n
    rows = [
        json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()
    ] if rows_path.exists() else []
    test_path = cohort / f"{args.split}.jsonl"
    summary = {
        "args": vars(args),
        "eval_examples": len(eval_examples),
        "eval_tasks": len({example.task_id for example in eval_examples}),
        "train_examples": len(train),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0,
        "summary": summarize(rows) if rows else {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "training": trainings,
        "test_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
        "elapsed_seconds": time.time() - started,
        "newly_written": written,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if rows:
        scored = score_mcnemar(rows)
        scored["model"] = args.model
        scored["cohort"] = str(cohort)
        (output_dir / "mcnemar.json").write_text(json.dumps(scored, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"event": "done", "summary": summary["summary"], "call": scored["call"]}), flush=True)
    else:
        print(json.dumps({"event": "train_done", "objectives": list(trainings)}), flush=True)


if __name__ == "__main__":
    main()
