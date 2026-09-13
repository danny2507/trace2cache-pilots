#!/usr/bin/env python3
"""Pilot 2: can K native soft states communicate runtime values to a frozen SLM?"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import time

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from trace2cache.latent import NativeEventResampler, trainable_parameter_count


DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LATENT_MARKER = "__TRACE_LATENTS_GO_HERE__"
PROMPT_BEFORE = "An execution summary is supplied through hidden runtime states.\nFinal accumulator = "


def prompt_after(task: str) -> str:
    if task == "final_digit":
        return "\nReport the final accumulator modulo 10. Answer with the integer only."
    if task == "threshold":
        return "\nIs the final accumulator greater than 4? Answer Yes or No only."
    raise ValueError(f"unknown task: {task}")


@dataclass
class ProbeBatch:
    event_types: torch.Tensor
    arguments: torch.Tensor
    states: torch.Tensor
    event_mask: torch.Tensor
    targets: torch.Tensor
    text_traces: list[str]

    def to(self, device: str) -> "ProbeBatch":
        return ProbeBatch(
            event_types=self.event_types.to(device),
            arguments=self.arguments.to(device),
            states=self.states.to(device),
            event_mask=self.event_mask.to(device),
            targets=self.targets.to(device),
            text_traces=self.text_traces,
        )


def resolve_local_model(model: str) -> str:
    direct = Path(model)
    if direct.exists():
        return str(direct.resolve())
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    cache = hf_home / "hub" / ("models--" + model.replace("/", "--"))
    ref = cache / "refs" / "main"
    candidates = []
    if ref.exists():
        candidates.append(cache / "snapshots" / ref.read_text().strip())
    if (cache / "snapshots").exists():
        candidates.extend(sorted((cache / "snapshots").iterdir()))
    for candidate in candidates:
        if (candidate / "config.json").exists():
            return str(candidate)
    raise FileNotFoundError(f"model unavailable locally: {model}")


def numeric_token_ids(tokenizer) -> list[int]:
    ids = []
    for value in range(10):
        encoded = tokenizer.encode(str(value), add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"value {value} does not map to one native token: {encoded}")
        ids.append(encoded[0])
    return ids


def make_batch(
    *,
    batch_size: int,
    min_steps: int,
    max_steps: int,
    rng: random.Random,
    value_token_ids: list[int],
    max_events: int,
    task: str,
    label_token_ids: dict[str, int],
) -> ProbeBatch:
    event_types = torch.zeros(batch_size, max_events, dtype=torch.long)
    arguments = torch.full((batch_size, max_events), value_token_ids[0], dtype=torch.long)
    states = torch.full((batch_size, max_events), value_token_ids[0], dtype=torch.long)
    event_mask = torch.zeros(batch_size, max_events, dtype=torch.bool)
    targets = torch.empty(batch_size, dtype=torch.long)
    traces = []
    for batch_index in range(batch_size):
        steps = rng.randint(min_steps, max_steps)
        accumulator = rng.randrange(10)
        lines = [f"init accumulator={accumulator}"]
        event_mask[batch_index, 0] = True
        event_types[batch_index, 0] = 0
        arguments[batch_index, 0] = value_token_ids[accumulator]
        states[batch_index, 0] = value_token_ids[accumulator]
        for step in range(1, steps + 1):
            argument = rng.randrange(1, 10)
            operation = rng.choice((1, 2))  # 1=add, 2=subtract
            if operation == 1:
                accumulator = (accumulator + argument) % 10
                name = "add"
            else:
                accumulator = (accumulator - argument) % 10
                name = "subtract"
            event_mask[batch_index, step] = True
            event_types[batch_index, step] = operation
            arguments[batch_index, step] = value_token_ids[argument]
            states[batch_index, step] = value_token_ids[accumulator]
            lines.append(f"step {step}: {name} {argument}; accumulator={accumulator}")
        if task == "final_digit":
            targets[batch_index] = value_token_ids[accumulator]
        else:
            targets[batch_index] = label_token_ids["Yes" if accumulator > 4 else "No"]
        traces.append("\n".join(lines))
    return ProbeBatch(event_types, arguments, states, event_mask, targets, traces)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-size", type=int, default=256)
    parser.add_argument("--num-latents", type=int, default=4)
    parser.add_argument("--hidden-width", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--native-sink-anchor", action="store_true")
    parser.add_argument("--task", choices=("final_digit", "threshold"), default="final_digit")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", default="artifacts/latent_probe/result.json")
    parser.add_argument("--checkpoint", default="checkpoints/latent_probe/adapter.pt")
    return parser.parse_args()


def receiver_prompt_parts(tokenizer, device: str, task: str) -> tuple[torch.Tensor, torch.Tensor]:
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You answer execution questions precisely."},
            {
                "role": "user",
                "content": PROMPT_BEFORE + LATENT_MARKER + prompt_after(task),
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    before, after = rendered.split(LATENT_MARKER)
    before_ids = tokenizer(before, return_tensors="pt", add_special_tokens=False)["input_ids"]
    after_ids = tokenizer(after, return_tensors="pt", add_special_tokens=False)["input_ids"]
    return before_ids.to(device), after_ids.to(device)


def _splice_embeddings(model, before_ids, latent_prefix, after_ids):
    batch_size = latent_prefix.shape[0]
    embedding = model.get_input_embeddings()
    before = embedding(before_ids).expand(batch_size, -1, -1)
    after = embedding(after_ids).expand(batch_size, -1, -1)
    return torch.cat((before, latent_prefix.to(before.dtype), after), dim=1)


def latent_logits(model, adapter, prompt_parts, batch: ProbeBatch) -> torch.Tensor:
    prefix = adapter(batch.event_types, batch.arguments, batch.states, batch.event_mask)
    inputs = _splice_embeddings(model, prompt_parts[0], prefix, prompt_parts[1])
    attention = torch.ones(inputs.shape[:2], dtype=torch.long, device=inputs.device)
    return model(inputs_embeds=inputs, attention_mask=attention, use_cache=False).logits[:, -1, :]


@torch.no_grad()
def evaluate_latent(model, adapter, prompt_parts, batch: ProbeBatch) -> dict[str, float]:
    adapter.eval()
    true_prefix = adapter(batch.event_types, batch.arguments, batch.states, batch.event_mask)
    shuffled_prefix = true_prefix.roll(1, dims=0)

    def predict(prefix: torch.Tensor | None) -> torch.Tensor:
        if prefix is None:
            embedding = model.get_input_embeddings()
            before = embedding(prompt_parts[0]).expand(true_prefix.shape[0], -1, -1)
            after = embedding(prompt_parts[1]).expand(true_prefix.shape[0], -1, -1)
            inputs = torch.cat((before, after), dim=1)
        else:
            inputs = _splice_embeddings(model, prompt_parts[0], prefix, prompt_parts[1])
        attention = torch.ones(inputs.shape[:2], dtype=torch.long, device=inputs.device)
        return model(inputs_embeds=inputs, attention_mask=attention, use_cache=False).logits[:, -1].argmax(-1)

    true_predictions = predict(true_prefix)
    shuffled_predictions = predict(shuffled_prefix)
    no_trace_predictions = predict(None)
    last_indices = batch.event_mask.long().sum(dim=1).sub(1).clamp_min(0)
    batch_indices = torch.arange(batch.targets.shape[0], device=batch.targets.device)
    sink_token_ids = batch.states[batch_indices, last_indices]
    oracle_native_prefix = model.get_input_embeddings()(sink_token_ids).unsqueeze(1)
    oracle_native_predictions = predict(oracle_native_prefix)
    return {
        "latent_accuracy": (true_predictions == batch.targets).float().mean().item(),
        "shuffled_latent_accuracy": (
            shuffled_predictions == batch.targets
        ).float().mean().item(),
        "no_trace_accuracy": (no_trace_predictions == batch.targets).float().mean().item(),
        "oracle_native_accuracy": (
            oracle_native_predictions == batch.targets
        ).float().mean().item(),
    }


@torch.no_grad()
def evaluate_text(
    model, tokenizer, batch: ProbeBatch, device: str, task: str, chunk_size: int = 32
) -> float:
    correct = 0
    for start in range(0, len(batch.text_traces), chunk_size):
        traces = batch.text_traces[start : start + chunk_size]
        rendered = []
        for trace in traces:
            user = (
                f"Execution trace:\n{trace}\n\n{PROMPT_BEFORE}"
                f"{trace.rsplit('=', 1)[-1]}{prompt_after(task)}"
            )
            rendered.append(
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": "You answer execution questions precisely."},
                        {"role": "user", "content": user},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        old_padding = tokenizer.padding_side
        tokenizer.padding_side = "left"
        encoded = tokenizer(rendered, return_tensors="pt", padding=True).to(device)
        tokenizer.padding_side = old_padding
        predictions = model(**encoded, use_cache=False).logits[:, -1].argmax(-1).cpu()
        targets = batch.targets[start : start + len(traces)].cpu()
        correct += int((predictions == targets).sum())
    return correct / len(batch.text_traces)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(resolve_local_model(args.model), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    value_ids = numeric_token_ids(tokenizer)
    label_ids = {}
    for label in ("Yes", "No"):
        encoded = tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"label {label} is not one token: {encoded}")
        label_ids[label] = encoded[0]
    model = AutoModelForCausalLM.from_pretrained(
        resolve_local_model(args.model), local_files_only=True, torch_dtype=torch.bfloat16
    ).to(device).eval()
    model.requires_grad_(False)
    width = model.get_input_embeddings().embedding_dim
    adapter = NativeEventResampler(
        model.get_input_embeddings(),
        model_width=width,
        hidden_width=args.hidden_width,
        num_latents=args.num_latents,
        max_events=32,
        native_sink_anchor=args.native_sink_anchor,
    ).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.learning_rate, weight_decay=0.01)
    prompt_parts = receiver_prompt_parts(tokenizer, device, args.task)
    train_rng = random.Random(args.seed)
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        adapter.train()
        batch = make_batch(
            batch_size=args.batch_size,
            min_steps=2,
            max_steps=6,
            rng=train_rng,
            value_token_ids=value_ids,
            max_events=32,
            task=args.task,
            label_token_ids=label_ids,
        ).to(device)
        logits = latent_logits(model, adapter, prompt_parts, batch)
        loss = nn.functional.cross_entropy(logits.float(), batch.targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 25 == 0:
            accuracy = (logits.argmax(-1) == batch.targets).float().mean().item()
            print(
                json.dumps(
                    {"step": step, "loss": round(loss.item(), 4), "train_accuracy": accuracy}
                ),
                flush=True,
            )

    evaluation = make_batch(
        batch_size=args.eval_size,
        min_steps=8,
        max_steps=12,
        rng=random.Random(args.seed + 10_000),
        value_token_ids=value_ids,
        max_events=32,
        task=args.task,
        label_token_ids=label_ids,
    ).to(device)
    metrics = evaluate_latent(model, adapter, prompt_parts, evaluation)
    metrics["text_trace_accuracy"] = evaluate_text(
        model, tokenizer, evaluation, device, args.task
    )
    elapsed = time.perf_counter() - started
    result = {
        "model": args.model,
        "seed": args.seed,
        "train_steps": args.steps,
        "train_lengths": [2, 6],
        "eval_lengths": [8, 12],
        "eval_size": args.eval_size,
        "num_latents": args.num_latents,
        "native_sink_anchor": args.native_sink_anchor,
        "task": args.task,
        "trainable_parameters": trainable_parameter_count(adapter),
        "elapsed_seconds": elapsed,
        **metrics,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"adapter": adapter.state_dict(), "result": result}, checkpoint)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
