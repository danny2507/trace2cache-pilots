#!/usr/bin/env python3
"""Output-prediction Gate 0, failed K=8 eval, native splice, and vocab-snap.

Text-only Gate 0 trains nothing. `--mode native` splices one frozen input
embedding per intermediate event (variable K, no encoder). `--mode snap`
replaces each of those vectors with the cosine-nearest embedding-table row.
`--mode full` keeps the failed RoleAwareEventEncoder path for replay only.
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

from trace2cache.execution_sim import (
    LATENT_CONDITIONS,
    NATIVE_CAP,
    NATIVE_CONDITIONS,
    SIM_CONDITIONS,
    SIM_SLOT_ROLES,
    SNAP_CONDITIONS,
    TEXT_CONDITIONS,
    different_task_example,
    evidence_for_condition,
    extract_output,
    native_event_texts,
    normalize_output,
    outputs_match,
    read_examples,
    render_chat,
    snap_table_to_vocab,
)
from trace2cache.latent import RoleAwareEventEncoder, trainable_parameter_count
from trace2cache.mbpp_generalization import content_role_test
from trace2cache.receiver_training import splice_prompt

sys.path.insert(0, str(Path(__file__).parent))
from analyze_mbpp_generalization import mcnemar
from run_pilot1 import resolve_local_model
from run_repair_latent_pool import greedy_generate

EVAL_CONDITIONS = ("code_input", "trace_text", "true_latent", "shuffled_latent")
EVAL_NATIVE_CONDITIONS = ("code_input", "trace_text", "native_events", "shuffled_native")
EVAL_SNAP_CONDITIONS = ("code_input", "trace_text", "vocab_snap", "shuffled_snap")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument(
        "--cohort", default="artifacts/mbpp_generalization/execution_sim_runbugrun_v1"
    )
    parser.add_argument(
        "--mode",
        choices=("text", "train", "eval", "full", "native", "snap"),
        default="text",
        help="text: Gate 0; native: pooled event splice; snap: nearest vocab row; train/eval/full: K=8 replay",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--conditions", nargs="+", choices=SIM_CONDITIONS)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--contrastive-weight", type=float, default=3.0)
    parser.add_argument("--contrastive-margin", type=float, default=0.1)
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--hidden-width", type=int, default=256)
    parser.add_argument("--latent-slots", type=int, default=8)
    parser.add_argument("--max-events", type=int, default=48)
    parser.add_argument("--max-tests", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-sequence-tokens", type=int, default=1536)
    parser.add_argument("--min-free-gib", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/mbpp_generalization/execution_sim_seed1001.pt",
    )
    parser.add_argument(
        "--text-rows",
        default="artifacts/mbpp_generalization/execution_sim_text_seed1001/rows.jsonl",
        help="Copy Gate 0 text rows instead of regenerating them.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/mbpp_generalization/execution_sim_text_seed1001",
    )
    return parser.parse_args()


def existing_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    return {
        (row["example_id"], row["condition"])
        for row in (json.loads(line) for line in path.read_text().splitlines() if line.strip())
    }


def clip_events(example, max_events: int):
    if len(example.events) <= max_events:
        return example
    from dataclasses import replace

    events = example.events[:max_events]
    return replace(example, events=events, event_count=len(events))


def within_budget(tokenizer, example, max_sequence_tokens: int) -> bool:
    prompt = render_chat(tokenizer, example, evidence_for_condition(example, "true_latent"))
    target = f"```\n{normalize_output(example.gold_stdout)}\n```"
    length = len(tokenizer(prompt, add_special_tokens=False).input_ids) + len(
        tokenizer(target, add_special_tokens=False).input_ids
    )
    return length <= max_sequence_tokens


@torch.no_grad()
def native_table(
    model, tokenizer, contents: list[str], batch_size: int = 64, max_length: int = 256
) -> torch.Tensor:
    embedding = model.get_input_embeddings()
    vectors = []
    for start in range(0, len(contents), batch_size):
        encoded = tokenizer(
            contents[start : start + batch_size],
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(model.device)
        token_vectors = embedding(encoded["input_ids"]).float()
        mask = encoded["attention_mask"].unsqueeze(-1)
        vectors.append((token_vectors * mask).sum(1) / mask.sum(1).clamp_min(1))
    return torch.cat(vectors)


def stack_native_latent(table, content_to_id, texts, device, dtype) -> torch.Tensor:
    if not texts:
        raise ValueError("native splice requires at least one event")
    missing = [text for text in texts if text not in content_to_id]
    if missing:
        raise KeyError(f"event text missing from native table: {missing[0][:80]}")
    ids = torch.tensor([content_to_id[text] for text in texts], device=device)
    return table[ids].unsqueeze(0).to(dtype)


def encode_events(encoder, table, content_to_id, events, device):
    contents, roles, tests = content_role_test(events)
    missing = [content for content in contents if content not in content_to_id]
    if missing:
        raise KeyError(f"event content missing from native table: {missing[0][:80]}")
    vectors = table[torch.tensor([content_to_id[content] for content in contents], device=device)]
    role_ids = torch.tensor(roles, device=device).unsqueeze(0)
    test_ids = torch.tensor(tests, device=device).unsqueeze(0)
    mask = torch.ones_like(role_ids, dtype=torch.bool)
    return encoder(vectors.unsqueeze(0), role_ids, test_ids, mask)


def target_ids(tokenizer, gold: str, device: torch.device) -> torch.Tensor:
    text = f"```\n{normalize_output(gold)}\n```"
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    eos = torch.tensor([[tokenizer.eos_token_id]], device=device)
    return torch.cat((ids, eos), dim=1)


def teacher_forced_nll(model, tokenizer, prompt: str, latent, targets: torch.Tensor):
    latent = latent.to(model.get_input_embeddings().weight.dtype)
    prompt_embeds = splice_prompt(model, tokenizer, prompt, latent)
    teacher = model.get_input_embeddings()(targets[:, :-1])
    inputs = torch.cat((prompt_embeds, teacher), dim=1)
    attention = torch.ones(inputs.shape[:2], dtype=torch.long, device=model.device)
    logits = model(inputs_embeds=inputs, attention_mask=attention, use_cache=False).logits
    start = prompt_embeds.shape[1] - 1
    predicted = logits[:, start : start + targets.shape[1]].float()
    return nn.functional.cross_entropy(predicted.flatten(0, 1), targets.flatten())


def stdout_loss_from_latent(model, tokenizer, latent, example):
    prompt = render_chat(tokenizer, example, evidence_for_condition(example, "true_latent"))
    targets = target_ids(tokenizer, example.gold_stdout, model.device)
    return teacher_forced_nll(model, tokenizer, prompt, latent, targets)


def reconstruction_loss(latent, table, content_to_id, events, device):
    losses = []
    for slot, roles in enumerate(SIM_SLOT_ROLES):
        allowed = set(roles)
        selected = [event for event in events if event.role_id in allowed] or list(events)
        ids = torch.tensor([content_to_id[event.content] for event in selected], device=device)
        target = table[ids].mean(0)
        predicted = latent[0, slot].float()
        losses.append(1 - nn.functional.cosine_similarity(predicted, target, dim=0))
    return torch.stack(losses).mean()


def make_encoder(model, tokenizer, args):
    if args.latent_slots != len(SIM_SLOT_ROLES):
        raise SystemExit(f"latent-slots {args.latent_slots} != {len(SIM_SLOT_ROLES)}")
    anchor_ids = tokenizer(
        " runtime evidence observed behavior state values result details",
        add_special_tokens=False,
    ).input_ids[: args.latent_slots]
    anchor_ids += [anchor_ids[-1]] * (args.latent_slots - len(anchor_ids))
    anchors = model.get_input_embeddings()(torch.tensor(anchor_ids, device=model.device)).detach()
    return RoleAwareEventEncoder(
        model_width=model.config.hidden_size,
        output_anchor=anchors,
        hidden_width=args.hidden_width,
        num_roles=8,
        max_events=args.max_events,
        max_tests=args.max_tests,
        slot_roles=SIM_SLOT_ROLES,
    ).to(model.device)


def train_encoder(model, tokenizer, encoder, train, table, content_to_id, args) -> dict:
    trainable = [parameter for parameter in encoder.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.01)
    rng = random.Random(args.seed)
    encoder.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    history = []
    ema_rank = None
    ema_nll = None
    best_score = None
    best_step = None
    best_state = None
    for step in range(1, args.steps + 1):
        index = rng.randrange(len(train))
        example = clip_events(train[index], args.max_events)
        shuffled = clip_events(different_task_example(train, index), args.max_events)
        true_latent = encode_events(encoder, table, content_to_id, example.events, model.device)
        shuffled_latent = encode_events(
            encoder, table, content_to_id, shuffled.events, model.device
        )
        true_loss = stdout_loss_from_latent(model, tokenizer, true_latent, example)
        shuffled_loss = stdout_loss_from_latent(model, tokenizer, shuffled_latent, example)
        ranking = nn.functional.relu(args.contrastive_margin + true_loss - shuffled_loss)
        recon = true_loss.new_tensor(0.0)
        if args.reconstruction_weight:
            recon = reconstruction_loss(
                true_latent, table, content_to_id, example.events, model.device
            )
        loss = true_loss + args.contrastive_weight * ranking
        if args.reconstruction_weight:
            loss = loss + args.reconstruction_weight * recon
        (loss / args.gradient_accumulation).backward()
        if step % args.gradient_accumulation == 0 or step == args.steps:
            nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % 20 == 0 or step == args.steps:
            rank_value = ranking.item()
            nll_value = true_loss.item()
            cosine_value = nn.functional.cosine_similarity(
                true_latent.float().reshape(1, -1), shuffled_latent.float().reshape(1, -1)
            ).item()
            row = {
                "step": step,
                "loss": round(loss.item(), 5),
                "true_stdout_loss": round(nll_value, 5),
                "shuffled_stdout_loss": round(shuffled_loss.item(), 5),
                "ranking_loss": round(rank_value, 5),
                "reconstruction_loss": round(recon.item(), 5),
                "true_shuffled_cosine": round(cosine_value, 5),
                "gpu_allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
            }
            history.append(row)
            print(json.dumps(row), flush=True)
            if ema_rank is None:
                ema_rank = rank_value
                ema_nll = nll_value
            else:
                ema_rank = 0.8 * ema_rank + 0.2 * rank_value
                ema_nll = 0.8 * ema_nll + 0.2 * nll_value
            if step >= 80:
                score = ema_rank + 0.15 * ema_nll
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
                    "ema_nll": round(ema_nll, 5),
                }
            ),
            flush=True,
        )
    encoder.eval()
    return {
        "steps": args.steps,
        "seconds": time.perf_counter() - started,
        "history": history,
        "best_step": best_step,
        "best_score": None if best_score is None else round(best_score, 5),
    }


def generate_one(model, tokenizer, encoder, table, content_to_id, example, condition, shuffled, args):
    evidence = evidence_for_condition(example, condition)
    rendered = render_chat(tokenizer, example, evidence)
    slot_count = 0
    if condition in NATIVE_CONDITIONS or condition in SNAP_CONDITIONS:
        shuffled_name = "shuffled_native" if condition in NATIVE_CONDITIONS else "shuffled_snap"
        source = shuffled if condition == shuffled_name else example
        if source is None:
            raise ValueError(f"{shuffled_name} requires another example")
        texts = native_event_texts(clip_events(source, NATIVE_CAP).events, cap=NATIVE_CAP)
        latent = stack_native_latent(
            table,
            content_to_id,
            texts,
            model.device,
            model.get_input_embeddings().weight.dtype,
        )
        slot_count = int(latent.shape[1])
    elif condition in LATENT_CONDITIONS:
        source = shuffled if condition == "shuffled_latent" else example
        if source is None:
            raise ValueError("shuffled_latent requires another example")
        events = clip_events(source, args.max_events).events
        with torch.no_grad():
            latent = encode_events(encoder, table, content_to_id, events, model.device)
        latent = latent.to(model.get_input_embeddings().weight.dtype)
        slot_count = int(latent.shape[1])
    else:
        latent = None
    inputs = splice_prompt(model, tokenizer, rendered, latent)
    tokens = greedy_generate(model, inputs, tokenizer.eos_token_id, args.max_new_tokens)
    response = tokenizer.decode(tokens, skip_special_tokens=True)
    predicted = extract_output(response)
    passed = outputs_match(predicted, example.gold_stdout)
    spliced = (
        condition in LATENT_CONDITIONS
        or condition in NATIVE_CONDITIONS
        or condition in SNAP_CONDITIONS
    )
    prompt_tokens = int(inputs.shape[1]) if spliced else len(
        tokenizer(rendered, add_special_tokens=False).input_ids
    )
    return {
        "task_id": example.task_id,
        "example_id": example.example_id,
        "condition": condition,
        "prompt_tokens": prompt_tokens,
        "event_count": example.event_count,
        "slot_count": slot_count,
        "response": response,
        "predicted": predicted,
        "gold": example.gold_stdout,
        "passed": passed,
        "outcome": "pass" if passed else "fail",
    }


def summarize(rows: list[dict]) -> dict:
    summaries = {}
    for condition in sorted({row["condition"] for row in rows}):
        selected = [row for row in rows if row["condition"] == condition]
        summaries[condition] = {
            "correct": sum(row["passed"] for row in selected),
            "examples": len(selected),
            "tasks": len({row["task_id"] for row in selected}),
            "task_correct": sum(row["passed"] for row in selected),
            "mean_prompt_tokens": sum(row["prompt_tokens"] for row in selected) / len(selected),
        }
    return summaries


def score_mcnemar(rows: list[dict]) -> dict:
    by_task: dict[int, dict[str, bool]] = {}
    for row in rows:
        by_task.setdefault(row["task_id"], {})[row["condition"]] = bool(row["passed"])
    present = sorted({row["condition"] for row in rows})
    pairs = []
    if "trace_gist" in present and "code_input" in present:
        pairs.append(("trace_gist", "code_input"))
    if "trace_text" in present and "code_input" in present:
        pairs.append(("trace_text", "code_input"))
    if "trace_gist" in present and "trace_text" in present:
        pairs.append(("trace_gist", "trace_text"))
    if "true_latent" in present and "code_input" in present:
        pairs.append(("true_latent", "code_input"))
    if "true_latent" in present and "trace_text" in present:
        pairs.append(("true_latent", "trace_text"))
    if "true_latent" in present and "shuffled_latent" in present:
        pairs.append(("true_latent", "shuffled_latent"))
    if "native_events" in present and "code_input" in present:
        pairs.append(("native_events", "code_input"))
    if "native_events" in present and "trace_text" in present:
        pairs.append(("native_events", "trace_text"))
    if "native_events" in present and "shuffled_native" in present:
        pairs.append(("native_events", "shuffled_native"))
    if "native_events" in present and "true_latent" in present:
        pairs.append(("native_events", "true_latent"))
    if "vocab_snap" in present and "code_input" in present:
        pairs.append(("vocab_snap", "code_input"))
    if "vocab_snap" in present and "trace_text" in present:
        pairs.append(("vocab_snap", "trace_text"))
    if "vocab_snap" in present and "shuffled_snap" in present:
        pairs.append(("vocab_snap", "shuffled_snap"))
    if "vocab_snap" in present and "native_events" in present:
        pairs.append(("vocab_snap", "native_events"))
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
    if "vocab_snap" in present:
        snap_vs_code = comparisons.get("vocab_snap vs code_input", {})
        snap_vs_text = comparisons.get("vocab_snap vs trace_text", {})
        snap_vs_shuf = comparisons.get("vocab_snap vs shuffled_snap", {})
        shuffle_drops = (
            snap_vs_shuf.get("left_only", 0) > snap_vs_shuf.get("right_only", 0)
            and snap_vs_shuf.get("p", 1) < 0.05
        )
        beats_code = (
            snap_vs_code.get("left_only", 0) > snap_vs_code.get("right_only", 0)
            and snap_vs_code.get("p", 1) < 0.05
        )
        not_worse_text = not (
            snap_vs_text.get("right_only", 0) > snap_vs_text.get("left_only", 0)
            and snap_vs_text.get("p", 1) < 0.05
        )
        matches_text = counts.get("vocab_snap", 0) >= counts.get("trace_text", 0) and not_worse_text
        if shuffle_drops and (beats_code or matches_text):
            call = "snap_pass"
            note = "Vocab-snap used: shuffle drops and snap beats code_input or matches trace_text."
        elif beats_code and not shuffle_drops:
            call = "snap_prefix_fail"
            note = "vocab_snap beats code_input but shuffle does not drop. Do not claim a channel."
        else:
            call = "snap_fail"
            note = "Vocab-snap did not beat code_input with a shuffle drop."
    elif "native_events" in present:
        native_vs_code = comparisons.get("native_events vs code_input", {})
        native_vs_text = comparisons.get("native_events vs trace_text", {})
        native_vs_shuf = comparisons.get("native_events vs shuffled_native", {})
        shuffle_drops = (
            native_vs_shuf.get("left_only", 0) > native_vs_shuf.get("right_only", 0)
            and native_vs_shuf.get("p", 1) < 0.05
        )
        beats_code = (
            native_vs_code.get("left_only", 0) > native_vs_code.get("right_only", 0)
            and native_vs_code.get("p", 1) < 0.05
        )
        not_worse_text = not (
            native_vs_text.get("right_only", 0) > native_vs_text.get("left_only", 0)
            and native_vs_text.get("p", 1) < 0.05
        )
        matches_text = counts.get("native_events", 0) >= counts.get("trace_text", 0) and not_worse_text
        if shuffle_drops and (beats_code or matches_text):
            call = "channel_pass"
            note = "Native event splice used: shuffle drops and native beats code_input or matches trace_text."
        elif beats_code and not shuffle_drops:
            call = "prefix_fail"
            note = "native_events beats code_input but shuffle does not drop. Do not claim a channel."
        else:
            call = "native_fail"
            note = "Native event splice did not beat text with a shuffle drop."
    elif "true_latent" in present:
        true_vs_code = comparisons.get("true_latent vs code_input", {})
        true_vs_text = comparisons.get("true_latent vs trace_text", {})
        true_vs_shuf = comparisons.get("true_latent vs shuffled_latent", {})
        shuffle_drops = (
            true_vs_shuf.get("left_only", 0) > true_vs_shuf.get("right_only", 0)
            and true_vs_shuf.get("p", 1) < 0.05
        )
        beats_code = (
            true_vs_code.get("left_only", 0) > true_vs_code.get("right_only", 0)
            and true_vs_code.get("p", 1) < 0.05
        )
        not_worse_text = not (
            true_vs_text.get("right_only", 0) > true_vs_text.get("left_only", 0)
            and true_vs_text.get("p", 1) < 0.05
        )
        compresses = counts.get("true_latent", 0) >= counts.get("trace_text", 0) and not_worse_text
        if shuffle_drops and (beats_code or compresses):
            call = "channel_pass"
            note = "Latent channel used: shuffle drops and true beats code_input or matches trace_text."
        elif beats_code and not shuffle_drops:
            call = "prefix_fail"
            note = "true beats code_input but shuffle does not drop. Do not claim a channel."
        else:
            call = "latent_fail"
            note = "K-slot encoder did not beat text with a shuffle drop."
    else:
        gist = comparisons.get("trace_gist vs code_input", {"left_only": 0, "right_only": 1, "p": 1})
        compact = comparisons.get("trace_text vs code_input", {"left_only": 0, "right_only": 1, "p": 1})
        gist_wins = gist["left_only"] > gist["right_only"] and gist["p"] < 0.05
        text_wins = compact["left_only"] > compact["right_only"] and compact["p"] < 0.05
        call = "gate0_pass" if gist_wins or text_wins else "gate0_fail"
        note = (
            "Train a K-slot encoder only if a trace condition beats code_input by McNemar."
            if call == "gate0_fail"
            else "Gate 0 passed. A K-slot encoder is allowed on this simulation panel."
        )
    return {
        "repair_at_1": counts,
        "comparisons": comparisons,
        "call": call,
        "note": note,
    }


def copy_text_rows(text_rows_path: Path, examples, conditions, handle, done: set[tuple[str, str]]) -> int:
    wanted = {example.example_id for example in examples}
    if not text_rows_path.exists():
        return 0
    copied = 0
    for line in text_rows_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row["example_id"], row["condition"])
        if row["example_id"] not in wanted:
            continue
        if row["condition"] not in conditions:
            continue
        if (
            row["condition"] in LATENT_CONDITIONS
            or row["condition"] in NATIVE_CONDITIONS
            or row["condition"] in SNAP_CONDITIONS
        ):
            continue
        if key in done:
            continue
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        done.add(key)
        copied += 1
    if copied:
        handle.flush()
    return copied


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
    model.config.use_cache = False
    return model, tokenizer, free_bytes


def build_table(model, tokenizer, *groups):
    contents = list(
        dict.fromkeys(
            event.content for group in groups for example in group for event in example.events
        )
    )
    if not contents:
        raise SystemExit("no event contents for native table")
    table = native_table(model, tokenizer, contents)
    content_to_id = {content: index for index, content in enumerate(contents)}
    return table, content_to_id


def build_native_text_table(model, tokenizer, examples):
    texts = list(
        dict.fromkeys(
            text
            for example in examples
            for text in native_event_texts(example.events, cap=NATIVE_CAP)
        )
    )
    if not texts:
        raise SystemExit("no native event texts")
    table = native_table(model, tokenizer, texts, max_length=256)
    return table, {text: index for index, text in enumerate(texts)}


def main() -> None:
    args = parse_args()
    model, tokenizer, free_bytes = load_model(args)
    cohort = Path(args.cohort)
    eval_examples = read_examples(cohort / f"{args.split}.jsonl")
    eval_examples.sort(key=lambda example: example.task_id)
    if args.limit:
        eval_examples = eval_examples[: args.limit]
    if not eval_examples:
        raise SystemExit(f"no examples in {cohort}")
    if args.mode == "text":
        conditions = tuple(args.conditions) if args.conditions else TEXT_CONDITIONS
    elif args.mode == "native":
        conditions = tuple(args.conditions) if args.conditions else EVAL_NATIVE_CONDITIONS
    elif args.mode == "snap":
        conditions = tuple(args.conditions) if args.conditions else EVAL_SNAP_CONDITIONS
    else:
        conditions = tuple(args.conditions) if args.conditions else EVAL_CONDITIONS
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "rows.jsonl"
    done = existing_keys(rows_path)
    training = None
    encoder = None
    table = None
    content_to_id = None

    if args.mode in ("train", "full", "eval"):
        train_path = cohort / "train.jsonl"
        if args.mode in ("train", "full") and (not train_path.exists() or train_path.stat().st_size == 0):
            raise SystemExit(f"disjoint train split missing: {train_path}")
        train = read_examples(train_path) if train_path.exists() and train_path.stat().st_size else []
        train = [clip_events(example, args.max_events) for example in train]
        train = [example for example in train if example.events and within_budget(tokenizer, example, args.max_sequence_tokens)]
        eval_task_ids = {example.task_id for example in eval_examples}
        overlap = {example.task_id for example in train} & eval_task_ids
        if overlap:
            raise SystemExit(f"train/eval task_id overlap: {sorted(overlap)[:8]}")
        if args.mode in ("train", "full") and len(train) < 32:
            raise SystemExit(f"train split too small after filters: {len(train)}")
        table, content_to_id = build_table(model, tokenizer, train, eval_examples)
        encoder = make_encoder(model, tokenizer, args)
        if args.mode == "eval":
            payload = torch.load(args.checkpoint, map_location=model.device, weights_only=False)
            missing, unexpected = encoder.load_state_dict(payload["encoder"], strict=True)
            if missing or unexpected:
                raise RuntimeError(f"checkpoint mismatch missing={missing} unexpected={unexpected}")
            encoder.eval()
            training = payload.get("training")
        else:
            training = train_encoder(model, tokenizer, encoder, train, table, content_to_id, args)
            checkpoint_path = Path(args.checkpoint)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "encoder": encoder.state_dict(),
                    "args": vars(args),
                    "training": training,
                    "trainable_parameters": trainable_parameter_count(encoder),
                },
                checkpoint_path,
            )
            print(
                json.dumps(
                    {
                        "event": "checkpoint_written",
                        "path": str(checkpoint_path),
                        "trainable_parameters": trainable_parameter_count(encoder),
                    }
                ),
                flush=True,
            )
        if args.mode == "train":
            summary = {
                "args": vars(args),
                "eval_examples": len(eval_examples),
                "eval_tasks": len({example.task_id for example in eval_examples}),
                "train_examples": len(train),
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "summary": {},
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "trainable_parameters": trainable_parameter_count(encoder),
                "training": training,
            }
            (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            print(json.dumps({"event": "train_done", "training": {"steps": training["steps"], "best_step": training["best_step"]}}), flush=True)
            return

    if args.mode == "native":
        table, content_to_id = build_native_text_table(model, tokenizer, eval_examples)
        print(
            json.dumps(
                {
                    "event": "native_table",
                    "unique_event_texts": len(content_to_id),
                    "cap": NATIVE_CAP,
                    "trainable_parameters": 0,
                }
            ),
            flush=True,
        )
    if args.mode == "snap":
        table, content_to_id = build_native_text_table(model, tokenizer, eval_examples)
        snapped, vocab_ids, snap_cosine = snap_table_to_vocab(
            table, model.get_input_embeddings().weight.detach()
        )
        table = snapped
        print(
            json.dumps(
                {
                    "event": "snap_table",
                    "unique_event_texts": len(content_to_id),
                    "unique_vocab_ids": int(vocab_ids.unique().numel()),
                    "mean_snap_cosine": round(float(snap_cosine.mean()), 5),
                    "min_snap_cosine": round(float(snap_cosine.min()), 5),
                    "cap": NATIVE_CAP,
                    "trainable_parameters": 0,
                }
            ),
            flush=True,
        )

    print(
        json.dumps(
            {
                "event": "model_loaded",
                "model": args.model,
                "eval_examples": len(eval_examples),
                "gpu_free_gib": round(free_bytes / 2**30, 2),
                "conditions": list(conditions),
                "mode": args.mode,
            }
        ),
        flush=True,
    )

    started = time.time()
    written = 0
    with rows_path.open("a") as handle:
        if args.mode not in ("text",):
            copied = copy_text_rows(Path(args.text_rows), eval_examples, conditions, handle, done)
            written += copied
            if copied:
                print(json.dumps({"event": "copied_text_rows", "n": copied}), flush=True)
        for index, example in enumerate(eval_examples):
            shuffled = different_task_example(eval_examples, index)
            for condition in conditions:
                key = (example.example_id, condition)
                if key in done:
                    continue
                t0 = time.time()
                row = generate_one(
                    model,
                    tokenizer,
                    encoder,
                    table,
                    content_to_id,
                    example,
                    condition,
                    shuffled,
                    args,
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
                            "slot_count": row.get("slot_count", 0),
                            "seconds": round(row["seconds"], 2),
                        }
                    ),
                    flush=True,
                )
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    test_path = cohort / f"{args.split}.jsonl"
    summary = {
        "args": vars(args),
        "eval_examples": len(eval_examples),
        "eval_tasks": len({example.task_id for example in eval_examples}),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "summary": summarize(rows),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trainable_parameters": 0 if encoder is None else trainable_parameter_count(encoder),
        "training": training,
        "test_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
        "elapsed_seconds": time.time() - started,
        "newly_written": written,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    scored = score_mcnemar(rows)
    scored["model"] = args.model
    scored["cohort"] = str(cohort)
    (output_dir / "mcnemar.json").write_text(json.dumps(scored, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "done", "summary": summary["summary"], "call": scored["call"]}), flush=True)


if __name__ == "__main__":
    main()
