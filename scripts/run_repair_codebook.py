#!/usr/bin/env python3
"""Capacity gate: can a few latent states steer a frozen LM to a full repair?"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from trace2cache.ambiguous_repair import get_ambiguous_cases
from trace2cache.sandbox import evaluate_patch, extract_function
from run_pilot1 import resolve_local_model
from run_repair_latent_pool import greedy_generate


DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MARKER = "<TRACE2CACHE_REPAIR_CODE>"


class NativeRepairCodebook(nn.Module):
    def __init__(self, initial_codes: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("anchors", initial_codes.clone())
        self.residual = nn.Parameter(torch.zeros_like(initial_codes, dtype=torch.float32))

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.anchors[indices].float() + self.residual[indices]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--latent-slots", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--output", default="artifacts/repair_latent/repair_codebook_seed131.json")
    parser.add_argument("--checkpoint", default="checkpoints/repair_latent/repair_codebook_seed131.pt")
    return parser.parse_args()


def prompt(case, evidence: str, *, heldout: bool = False) -> str:
    args_text = ", ".join(repr(argument) for argument in case.public_test.args)
    request = (
        "Produce the complete repaired function only; hidden tests must match the encoded behavior."
        if heldout
        else "Return only the complete corrected function in one Python code block."
    )
    user = f"""Repair this Python function. The public failure alone is ambiguous; additional behavioral evidence identifies the intended hidden behavior.

Buggy code:
```python
{case.buggy_source.rstrip()}
```

Public failing test:
`{case.function_name}({args_text})` must return `{case.public_test.expected!r}`.

Encoded behavioral evidence: {evidence}

{request}
"""
    return user


def render(tokenizer, case, evidence: str, *, heldout: bool = False) -> str:
    return tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": "You are a precise Python program repair assistant.",
            },
            {"role": "user", "content": prompt(case, evidence, heldout=heldout)},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def initial_native_codes(model, tokenizer, items, slots: int) -> torch.Tensor:
    embedding = model.get_input_embeddings()
    pad_ids = tokenizer(" evidence", add_special_tokens=False)["input_ids"]
    pad_id = pad_ids[0]
    codes = []
    for item in items:
        ids = tokenizer(" " + item.diagnosis, add_special_tokens=False)["input_ids"][:slots]
        ids = ids + [pad_id] * (slots - len(ids))
        token_ids = torch.tensor(ids, device=model.device)
        codes.append(embedding(token_ids).detach())
    return torch.stack(codes)


def splice(model, tokenizer, rendered: str, latent: torch.Tensor | None) -> torch.Tensor:
    embedding = model.get_input_embeddings()
    if latent is None:
        ids = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
            model.device
        )
        return embedding(ids)
    before, after = rendered.split(MARKER)
    before_ids = tokenizer(before, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
        model.device
    )
    after_ids = tokenizer(after, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
        model.device
    )
    return torch.cat((embedding(before_ids), latent.to(embedding.weight.dtype), embedding(after_ids)), dim=1)


def target_ids(tokenizer, source: str, device: torch.device) -> torch.Tensor:
    text = f"```python\n{source.rstrip()}\n```"
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    eos = torch.tensor([[tokenizer.eos_token_id]], device=device)
    return torch.cat((ids, eos), dim=1)


def repair_loss(model, tokenizer, codebook, item_index: int, item) -> torch.Tensor:
    rendered = render(tokenizer, item.case, MARKER)
    index = torch.tensor([item_index], device=model.device)
    prompt_embeds = splice(model, tokenizer, rendered, codebook(index))
    targets = target_ids(tokenizer, item.correct_source, model.device)
    teacher = model.get_input_embeddings()(targets[:, :-1])
    inputs = torch.cat((prompt_embeds, teacher), dim=1)
    attention = torch.ones(inputs.shape[:2], dtype=torch.long, device=model.device)
    logits = model(inputs_embeds=inputs, attention_mask=attention, use_cache=False).logits
    start = prompt_embeds.shape[1] - 1
    predicted = logits[:, start : start + targets.shape[1]].float()
    return nn.functional.cross_entropy(predicted.reshape(-1, predicted.shape[-1]), targets.reshape(-1))


@torch.inference_mode()
def evaluate(model, tokenizer, codebook, items, max_new_tokens: int, *, heldout: bool):
    conditions = ("no_evidence", "diagnosis_text", "native_anchor", "trained", "paired_swap")
    rows = []
    for index, item in enumerate(items):
        paired = index + 1 if index % 2 == 0 else index - 1
        for condition in conditions:
            if condition == "no_evidence":
                rendered = render(tokenizer, item.case, "unavailable", heldout=heldout)
                latent = None
            elif condition == "diagnosis_text":
                rendered = render(tokenizer, item.case, item.diagnosis, heldout=heldout)
                latent = None
            else:
                rendered = render(tokenizer, item.case, MARKER, heldout=heldout)
                code_index = paired if condition == "paired_swap" else index
                latent = (
                    codebook.anchors[code_index : code_index + 1]
                    if condition == "native_anchor"
                    else codebook(torch.tensor([code_index], device=model.device))
                )
            inputs = splice(model, tokenizer, rendered, latent)
            generated = greedy_generate(
                model, inputs, tokenizer.eos_token_id, max_new_tokens
            )
            response = tokenizer.decode(generated, skip_special_tokens=True)
            patch = None
            try:
                patch = extract_function(response, item.case.function_name)
                validation = evaluate_patch(patch, item.case)
            except Exception as exc:
                validation = {"passed": False, "error": f"{type(exc).__name__}: {exc}", "tests": []}
            rows.append({
                "case_id": item.case.case_id,
                "pair_id": item.pair_id,
                "condition": condition,
                "heldout_prompt": heldout,
                "passed": validation["passed"],
                "response": response,
                "patch": patch,
                "validation": validation,
            })
    return rows


def summarize(rows) -> dict[str, dict[str, int]]:
    result = {}
    for heldout in (False, True):
        subset = [row for row in rows if row["heldout_prompt"] == heldout]
        key = "heldout_prompt" if heldout else "train_prompt"
        result[key] = {
            condition: sum(row["passed"] for row in subset if row["condition"] == condition)
            for condition in sorted({row["condition"] for row in subset})
        }
    return result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("A BF16 CUDA GPU is required; this pilot never falls back to CPU")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    items = get_ambiguous_cases()
    model_path = resolve_local_model(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.bfloat16
    ).to("cuda").eval()
    model.requires_grad_(False)
    anchors = initial_native_codes(model, tokenizer, items, args.latent_slots)
    codebook = NativeRepairCodebook(anchors).to(model.device)
    initial_rows = evaluate(
        model, tokenizer, codebook, items, args.max_new_tokens, heldout=False
    )
    print(json.dumps({"event": "initial", "summary": summarize(initial_rows + [
        {**row, "heldout_prompt": True} for row in []
    ])["train_prompt"]}), flush=True)

    optimizer = torch.optim.AdamW(codebook.parameters(), lr=args.learning_rate, weight_decay=0.0)
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        item_index = random.randrange(len(items))
        loss = repair_loss(model, tokenizer, codebook, item_index, items[item_index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(codebook.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 20 == 0:
            print(json.dumps({"step": step, "loss": round(loss.item(), 5)}), flush=True)

    final_rows = evaluate(model, tokenizer, codebook, items, args.max_new_tokens, heldout=False)
    final_rows += evaluate(model, tokenizer, codebook, items, args.max_new_tokens, heldout=True)
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "seed": args.seed,
        "steps": args.steps,
        "latent_slots": args.latent_slots,
        "trainable_parameters": codebook.residual.numel(),
        "training_seconds": time.perf_counter() - started,
        "initial_rows": initial_rows,
        "final_rows": final_rows,
        "summary": summarize(final_rows),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"adapter_trainable": {"residual": codebook.residual.detach().cpu()}, "result": result}, checkpoint)
    print(json.dumps({"event": "final", "summary": result["summary"], "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
