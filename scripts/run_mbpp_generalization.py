#!/usr/bin/env python3
"""Matched MBPP repair: text-only prompting versus eight latent states.

Text conditions need no training. Latent conditions train a frozen-receiver encoder
on train-split canonical patches, then evaluate greedy Repair@1 on hidden tests.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from trace2cache.discrepancy_residual import DiscrepancyResidual
from trace2cache.latent import RoleAwareEventEncoder, trainable_parameter_count
from trace2cache.mbpp_generalization import (
    MARKER,
    RepairExample,
    allowed_calls_for_tests,
    content_role_test,
    different_task_example,
    discrepancy_fields,
    discrepancy_text,
    evidence_for_condition,
    events_for_condition,
    evaluate_hidden,
    extract_patch,
    edit_sketch,
    gist_events,
    gist_text,
    is_wrong_value_failure,
    read_examples,
    render_chat,
    render_gist_chat,
    render_icae_encode_chat,
    render_sketch_chat,
    shared_affix_edit_mask,
)
from trace2cache.receiver_training import splice_prompt

sys.path.insert(0, str(Path(__file__).parent))
from run_pilot1 import resolve_local_model
from run_repair_latent_pool import greedy_generate

TEXT_CONDITIONS = ("no_evidence", "io_text", "compact_text")
LATENT_CONDITIONS = ("true_latent", "shuffled_latent", "corrupted_latent")
DIAGNOSTIC_CONDITIONS = ("oracle_sketch",)
HAQUE_TEXT_CONDITIONS = ("gist_text", "collated_text", "line_text", "discrepancy_text")
EMBED_CONDITIONS = ("discrepancy_embed", "shuffled_embed", "corrupted_embed")
PROTOCOL_CONDITIONS = TEXT_CONDITIONS + LATENT_CONDITIONS
ALL_CONDITIONS = PROTOCOL_CONDITIONS + DIAGNOSTIC_CONDITIONS + HAQUE_TEXT_CONDITIONS + EMBED_CONDITIONS
TEST, CALL, BRANCH, STATE, LINE, RETURN, PASS, FAIL = range(8)
ROLE_FACTORIZED_SLOTS = (
    (TEST, FAIL, RETURN),
    (TEST, BRANCH, FAIL),
    (TEST, STATE, RETURN),
    (TEST, LINE, STATE),
    (TEST, PASS, RETURN),
    (TEST, PASS, FAIL, STATE),
    (TEST, BRANCH, STATE),
    tuple(range(8)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--cohort", default="artifacts/mbpp_generalization/cohort_v1")
    parser.add_argument(
        "--mode",
        choices=("text", "train", "eval", "full"),
        default="full",
        help="text: frozen prompting only; train: adapter only; eval: latent from checkpoint; full: train then all conditions",
    )
    parser.add_argument("--conditions", nargs="+", choices=ALL_CONDITIONS)
    parser.add_argument("--split", default="test", choices=("train", "validation", "test"))
    parser.add_argument("--limit", type=int, default=0, help="0 = every eligible example in the split")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--contrastive-weight", type=float, default=3.0)
    parser.add_argument("--contrastive-margin", type=float, default=0.1)
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--separation-weight", type=float, default=0.5)
    parser.add_argument("--separation-margin", type=float, default=0.3)
    parser.add_argument(
        "--include-specification",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Put the English spec in every condition. Off by default so evidence can matter.",
    )
    parser.add_argument(
        "--train-hide-public-test",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Withhold the public failing test during encoder training so latents must carry I/O.",
    )
    parser.add_argument(
        "--role-skip",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add each slot's role-pooled native event vectors to the encoder output.",
    )
    parser.add_argument(
        "--constant-latent",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train eight event-independent embeddings (soft-prompt control). Ignores the trace.",
    )
    parser.add_argument(
        "--gist-latent",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Encode a short gist (I/O plus last LINE/STATE/RETURN) and reconstruct it through the frozen LM.",
    )
    parser.add_argument("--gist-weight", type=float, default=1.0)
    parser.add_argument(
        "--edit-span-loss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Repair NLL only on target tokens outside the shared prefix/suffix with the buggy program.",
    )
    parser.add_argument(
        "--train-value-failures",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train only on public wrong-value failures, not immediate exceptions.",
    )
    parser.add_argument(
        "--edit-sketch-icae",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="ICAE encoder: frozen-LM gist tokens as a residual on a frozen const prefix; reconstruct the oracle REPLACE.",
    )
    parser.add_argument("--sketch-weight", type=float, default=1.0)
    parser.add_argument(
        "--const-prefix-checkpoint",
        default="checkpoints/mbpp_generalization/latent_seed1001_const.pt",
        help="Frozen 8-slot const prefix used as the ICAE residual base.",
    )
    parser.add_argument(
        "--discrepancy-residual",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="K unit-RMS slots from (expected, got), scaled to token RMS. Not a trace encoder.",
    )
    parser.add_argument(
        "--io-probe-weight",
        type=float,
        default=0.1,
        help="MSE of a linear probe from mean slot to pooled (expected, got). Used with --discrepancy-residual.",
    )
    parser.add_argument(
        "--train-misses-rows",
        default="",
        help="Keep train examples whose task_id failed no_evidence in this rows.jsonl.",
    )
    parser.add_argument("--hidden-width", type=int, default=256)
    parser.add_argument("--latent-slots", type=int, default=8)
    parser.add_argument("--max-events", type=int, default=96)
    parser.add_argument("--max-tests", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-sequence-tokens", type=int, default=768)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--min-free-gib", type=float, default=12.0)
    parser.add_argument("--checkpoint", default="checkpoints/mbpp_generalization/latent_seed1001.pt")
    parser.add_argument("--output-dir", default="artifacts/mbpp_generalization/full_seed1001")
    return parser.parse_args()


def target_ids(tokenizer, source: str, device: torch.device) -> torch.Tensor:
    text = f"```python\n{source.rstrip()}\n```"
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    eos = torch.tensor([[tokenizer.eos_token_id]], device=device)
    return torch.cat((ids, eos), dim=1)


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
            max_length=96,
            return_tensors="pt",
        ).to(model.device)
        token_vectors = embedding(encoded["input_ids"]).float()
        mask = encoded["attention_mask"].unsqueeze(-1)
        vectors.append((token_vectors * mask).sum(1) / mask.sum(1).clamp_min(1))
    return torch.cat(vectors)


def clip_events(example: RepairExample, max_events: int) -> RepairExample:
    if len(example.events) <= max_events:
        return example
    from dataclasses import replace

    return replace(example, events=example.events[:max_events])


def missed_no_evidence_tasks(path: Path) -> set[int]:
    missed: set[int] = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("condition") != "no_evidence":
            continue
        if not row.get("passed"):
            missed.add(int(row["task_id"]))
    if not missed:
        raise SystemExit(f"no no_evidence misses in {path}")
    return missed


def load_split(cohort: Path, split: str, limit: int) -> list[RepairExample]:
    examples = read_examples(cohort / f"{split}.jsonl")
    examples.sort(key=lambda example: (example.task_id, example.mutation_kind, example.mutation_ordinal))
    if limit:
        examples = examples[:limit]
    return examples


def within_budget(
    tokenizer,
    example: RepairExample,
    max_sequence_tokens: int,
    *,
    include_specification: bool,
    include_public_test: bool,
) -> bool:
    prompt_length = len(
        tokenizer(
            render_chat(
                tokenizer,
                example,
                MARKER,
                include_specification=include_specification,
                include_public_test=include_public_test,
            ),
            add_special_tokens=False,
        ).input_ids
    )
    target_length = len(tokenizer(example.target_source, add_special_tokens=False).input_ids)
    return prompt_length + target_length <= max_sequence_tokens


class ConstantLatent(nn.Module):
    """Eight learned embeddings with no dependence on the runtime trace."""

    def __init__(self, anchors: torch.Tensor):
        super().__init__()
        self.slots = nn.Parameter(anchors.detach().float().clone())

    def forward(self, *args, **kwargs):
        return self.slots.unsqueeze(0)


class EditSketchICAE(nn.Module):
    """Frozen const prefix plus a zero-init residual from frozen-LM gist tokens."""

    def __init__(self, const_slots: torch.Tensor, hidden_width: int):
        super().__init__()
        if const_slots.ndim != 2:
            raise ValueError("const_slots must have shape [slots, hidden]")
        self.register_buffer("const", const_slots.detach().float().clone())
        self.gist_embeds = nn.Parameter(const_slots.detach().float().clone())
        self.residual = nn.Linear(hidden_width, hidden_width)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    @property
    def num_latents(self) -> int:
        return int(self.const.shape[0])

    def encode(self, model, tokenizer, example, events):
        text = render_icae_encode_chat(tokenizer, example, events)
        embedding = model.get_input_embeddings()
        encoded = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=512,
        )
        prompt = embedding(encoded["input_ids"].to(model.device))
        gist = self.gist_embeds.to(dtype=prompt.dtype, device=prompt.device).unsqueeze(0)
        inputs = torch.cat((prompt, gist), dim=1)
        attention = torch.ones(inputs.shape[:2], dtype=torch.long, device=model.device)
        hidden = model.model(
            inputs_embeds=inputs, attention_mask=attention, use_cache=False
        ).last_hidden_state
        residual = self.residual(hidden[:, -self.num_latents :, :].float())
        return self.const.unsqueeze(0) + residual


def select_events(events, args):
    if args.gist_latent:
        return gist_events(events)
    return events


def encode_example_latent(
    encoder,
    model,
    tokenizer,
    table,
    content_to_id,
    example,
    events,
    args,
    *,
    include_public_test: bool | None = None,
    corrupt: bool = False,
):
    if isinstance(encoder, DiscrepancyResidual):
        public = True if include_public_test is None else include_public_test
        return encoder.encode(
            model,
            tokenizer,
            example,
            events,
            include_specification=args.include_specification,
            include_public_test=public,
            corrupt=corrupt,
            max_length=args.max_sequence_tokens,
        )
    if isinstance(encoder, EditSketchICAE):
        return encoder.encode(model, tokenizer, example, gist_events(events))
    return encode_events(
        encoder,
        table,
        content_to_id,
        select_events(events, args),
        model.device,
        role_skip=args.role_skip,
    )


def encode_events(encoder, table, content_to_id, events, device, *, role_skip: bool = False):
    if isinstance(encoder, ConstantLatent):
        return encoder()
    contents, roles, tests = content_role_test(events)
    missing = [content for content in contents if content not in content_to_id]
    if missing:
        raise KeyError(f"event content missing from native table: {missing[0][:80]}")
    vectors = table[torch.tensor([content_to_id[content] for content in contents], device=device)]
    role_ids = torch.tensor(roles, device=device).unsqueeze(0)
    test_ids = torch.tensor(tests, device=device).unsqueeze(0)
    mask = torch.ones_like(role_ids, dtype=torch.bool)
    latent = encoder(vectors.unsqueeze(0), role_ids, test_ids, mask)
    if not role_skip:
        return latent
    skip = torch.zeros_like(latent)
    role_tensor = torch.tensor(roles, device=device)
    for slot, allowed in enumerate(ROLE_FACTORIZED_SLOTS):
        chosen = torch.isin(role_tensor, torch.tensor(allowed, device=device))
        if not bool(chosen.any()):
            chosen = torch.ones(len(roles), dtype=torch.bool, device=device)
        skip[0, slot] = vectors[chosen].mean(0)
    return latent + skip


def teacher_forced_nll(model, tokenizer, prompt: str, latent, targets: torch.Tensor, token_mask=None):
    latent = latent.to(model.get_input_embeddings().weight.dtype)
    prompt_embeds = splice_prompt(model, tokenizer, prompt, latent)
    teacher = model.get_input_embeddings()(targets[:, :-1])
    inputs = torch.cat((prompt_embeds, teacher), dim=1)
    attention = torch.ones(inputs.shape[:2], dtype=torch.long, device=model.device)
    logits = model(inputs_embeds=inputs, attention_mask=attention, use_cache=False).logits
    start = prompt_embeds.shape[1] - 1
    predicted = logits[:, start : start + targets.shape[1]].float()
    labels = targets
    if token_mask is not None:
        labels = targets.clone()
        labels[:, ~token_mask] = -100
        if int((labels != -100).sum()) == 0:
            labels = targets
    return nn.functional.cross_entropy(predicted.flatten(0, 1), labels.flatten(), ignore_index=-100)


def edit_span_mask(tokenizer, example, targets: torch.Tensor) -> torch.Tensor:
    buggy = tokenizer(
        f"```python\n{example.buggy_source.rstrip()}\n```",
        add_special_tokens=False,
    )["input_ids"]
    target_list = targets[0, :-1].tolist()
    middle = shared_affix_edit_mask(buggy, target_list) + [True]
    return torch.tensor(middle, dtype=torch.bool, device=targets.device)


def repair_loss_from_latent(
    model,
    tokenizer,
    latent,
    example,
    *,
    include_specification: bool,
    include_public_test: bool,
    edit_span: bool = False,
):
    prompt = render_chat(
        tokenizer,
        example,
        MARKER,
        include_specification=include_specification,
        include_public_test=include_public_test,
    )
    targets = target_ids(tokenizer, example.target_source, model.device)
    mask = edit_span_mask(tokenizer, example, targets) if edit_span else None
    return teacher_forced_nll(model, tokenizer, prompt, latent, targets, mask)


def gist_loss_from_latent(model, tokenizer, latent, example):
    prompt = render_gist_chat(tokenizer, example)
    text = gist_text(example.events)
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
    eos = torch.tensor([[tokenizer.eos_token_id]], device=model.device)
    targets = torch.cat((ids, eos), dim=1)
    return teacher_forced_nll(model, tokenizer, prompt, latent, targets)


def sketch_loss_from_latent(model, tokenizer, latent, example):
    prompt = render_sketch_chat(tokenizer, example)
    text = edit_sketch(example.buggy_source, example.target_source)
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
    eos = torch.tensor([[tokenizer.eos_token_id]], device=model.device)
    targets = torch.cat((ids, eos), dim=1)
    return teacher_forced_nll(model, tokenizer, prompt, latent, targets)


def reconstruction_loss(latent, table, content_to_id, events, device):
    """Ground each slot in the native embeddings of the roles it is allowed to read."""
    losses = []
    for slot, roles in enumerate(ROLE_FACTORIZED_SLOTS):
        allowed = set(roles)
        selected = [event for event in events if event.role_id in allowed] or list(events)
        ids = torch.tensor([content_to_id[event.content] for event in selected], device=device)
        target = table[ids].mean(0)
        predicted = latent[0, slot].float()
        losses.append(1 - nn.functional.cosine_similarity(predicted, target, dim=0))
    return torch.stack(losses).mean()


def make_encoder(model, tokenizer, args):
    if args.discrepancy_residual:
        encoder = DiscrepancyResidual(
            model.config.hidden_size,
            num_slots=args.latent_slots,
            mlp_width=args.hidden_width,
        ).to(model.device)
        encoder.set_scale_from_embeddings(model.get_input_embeddings())
        return encoder
    if args.edit_sketch_icae:
        path = Path(args.const_prefix_checkpoint)
        if not path.exists():
            raise SystemExit(f"const prefix checkpoint missing: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        slots = payload["encoder"]["slots"].to(model.device)
        if tuple(slots.shape) != (args.latent_slots, model.config.hidden_size):
            raise SystemExit(f"const prefix shape {tuple(slots.shape)} does not match slots/hidden")
        return EditSketchICAE(slots, model.config.hidden_size).to(model.device)
    anchor_ids = tokenizer(
        " runtime evidence observed behavior state values result details",
        add_special_tokens=False,
    ).input_ids[: args.latent_slots]
    anchor_ids += [anchor_ids[-1]] * (args.latent_slots - len(anchor_ids))
    anchors = model.get_input_embeddings()(torch.tensor(anchor_ids, device=model.device)).detach()
    if args.constant_latent:
        return ConstantLatent(anchors).to(model.device)
    return RoleAwareEventEncoder(
        model_width=model.config.hidden_size,
        output_anchor=anchors,
        hidden_width=args.hidden_width,
        num_roles=8,
        max_events=args.max_events,
        max_tests=args.max_tests,
        slot_roles=ROLE_FACTORIZED_SLOTS,
    ).to(model.device)


def train_encoder(model, tokenizer, encoder, train, table, content_to_id, args) -> dict:
    if args.discrepancy_residual:
        args.reconstruction_weight = 0.0
        args.separation_weight = 0.0
    if args.edit_sketch_icae:
        args.edit_span_loss = True
        args.reconstruction_weight = 0.0
        args.separation_weight = 0.0
        model.config.use_cache = False
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
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
    for step in range(1, args.steps + 1):
        index = rng.randrange(len(train))
        example = train[index]
        shuffled = different_task_example(train, index)
        io_expected = io_got = io_delta = None
        if isinstance(encoder, DiscrepancyResidual):
            embedding = model.get_input_embeddings()

            def residual_from(events):
                expected_text, got_text = discrepancy_fields(events)
                expected = encoder.pool_text(tokenizer, embedding, expected_text)
                got = encoder.pool_text(tokenizer, embedding, got_text)
                delta = encoder.delta_vector(expected, got)
                return encoder.compose(delta), expected, got, delta

            true_latent, io_expected, io_got, io_delta = residual_from(example.events)
            shuffled_latent, _shuffled_e, _shuffled_g, _shuffled_delta = residual_from(shuffled.events)
        else:
            true_latent = encode_example_latent(
                encoder,
                model,
                tokenizer,
                table,
                content_to_id,
                example,
                example.events,
                args,
                include_public_test=train_public_test,
            )
            shuffled_latent = encode_example_latent(
                encoder,
                model,
                tokenizer,
                table,
                content_to_id,
                shuffled,
                shuffled.events,
                args,
                include_public_test=train_public_test,
            )
        true_loss = repair_loss_from_latent(
            model,
            tokenizer,
            true_latent,
            example,
            include_specification=args.include_specification,
            include_public_test=train_public_test,
            edit_span=args.edit_span_loss,
        )
        gist_true = true_loss.new_tensor(0.0)
        gist_shuffled = true_loss.new_tensor(0.0)
        sketch_true = true_loss.new_tensor(0.0)
        sketch_shuffled = true_loss.new_tensor(0.0)
        recon = true_loss.new_tensor(0.0)
        io_probe = true_loss.new_tensor(0.0)
        if io_expected is not None:
            io_probe = encoder.probe_loss(true_latent, io_expected, io_got)
        if args.constant_latent:
            shuffled_loss = true_loss.detach()
            ranking = true_loss.new_tensor(args.contrastive_margin)
        elif args.edit_sketch_icae:
            sketch_true = sketch_loss_from_latent(model, tokenizer, true_latent, example)
            sketch_shuffled = sketch_loss_from_latent(model, tokenizer, shuffled_latent, example)
            ranking = nn.functional.relu(args.contrastive_margin + sketch_true - sketch_shuffled)
            shuffled_loss = sketch_shuffled
        elif args.gist_latent:
            gist_true = gist_loss_from_latent(model, tokenizer, true_latent, example)
            gist_shuffled = gist_loss_from_latent(model, tokenizer, shuffled_latent, example)
            ranking = nn.functional.relu(args.contrastive_margin + gist_true - gist_shuffled)
            shuffled_loss = gist_shuffled
        else:
            shuffled_loss = repair_loss_from_latent(
                model,
                tokenizer,
                shuffled_latent,
                example,
                include_specification=args.include_specification,
                include_public_test=train_public_test,
                edit_span=args.edit_span_loss,
            )
            ranking = nn.functional.relu(args.contrastive_margin + true_loss - shuffled_loss)
            if args.reconstruction_weight:
                recon = reconstruction_loss(
                    true_latent, table, content_to_id, example.events, model.device
                )
        true_flat = true_latent.float().reshape(1, -1)
        shuffled_flat = shuffled_latent.float().reshape(1, -1)
        cosine = 1 - nn.functional.cosine_similarity(true_flat, shuffled_flat)
        separation = nn.functional.relu(args.separation_margin - cosine)
        loss = true_loss + args.contrastive_weight * ranking
        if args.gist_latent:
            loss = loss + args.gist_weight * gist_true
        if args.edit_sketch_icae:
            loss = loss + args.sketch_weight * sketch_true
        if args.reconstruction_weight:
            loss = loss + args.reconstruction_weight * recon
        if args.separation_weight:
            loss = loss + args.separation_weight * separation
        if args.io_probe_weight and io_expected is not None:
            loss = loss + args.io_probe_weight * io_probe
        (loss / args.gradient_accumulation).backward()
        if step % args.gradient_accumulation == 0 or step == args.steps:
            nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % 20 == 0 or step == args.steps:
            rank_value = ranking.item()
            cosine_value = nn.functional.cosine_similarity(true_flat, shuffled_flat).item()
            row = {
                "step": step,
                "loss": round(loss.item(), 5),
                "true_patch_loss": round(true_loss.item(), 5),
                "shuffled_patch_loss": round(shuffled_loss.item(), 5),
                "gist_true_loss": round(gist_true.item(), 5),
                "gist_shuffled_loss": round(gist_shuffled.item(), 5),
                "sketch_true_loss": round(sketch_true.item(), 5),
                "sketch_shuffled_loss": round(sketch_shuffled.item(), 5),
                "ranking_loss": round(rank_value, 5),
                "reconstruction_loss": round(recon.item(), 5),
                "io_probe_loss": round(io_probe.item(), 5),
                "scale": round(float(encoder.scale.detach()) if isinstance(encoder, DiscrepancyResidual) else 0.0, 5),
                "rms_delta": round(
                    float(io_delta.float().pow(2).mean().sqrt()) if io_delta is not None else 0.0, 5
                ),
                "separation_loss": round(separation.item(), 5),
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
    if best_state is not None and not args.constant_latent:
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
    elif args.constant_latent:
        print(
            json.dumps({"event": "constant_latent_keeps_final_weights", "steps": args.steps}),
            flush=True,
        )
    encoder.eval()
    if args.edit_sketch_icae:
        model.eval()
        model.config.use_cache = False
    return {
        "steps": args.steps,
        "seconds": time.perf_counter() - started,
        "history": history,
        "best_step": best_step,
        "best_score": None if best_score is None else round(best_score, 5),
    }


def embed_text(model, tokenizer, text: str) -> torch.Tensor:
    """Frozen token embeddings of a short string. No trained encoder."""
    embedding = model.get_input_embeddings()
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    if ids.numel() == 0:
        ids = tokenizer("unavailable", return_tensors="pt", add_special_tokens=False)["input_ids"]
    return embedding(ids.to(embedding.weight.device))


@torch.inference_mode()
def generate_one(model, tokenizer, encoder, table, content_to_id, example, condition, shuffled, args):
    evidence = evidence_for_condition(example, condition)
    rendered = render_chat(
        tokenizer,
        example,
        evidence,
        include_specification=args.include_specification,
        evidence_kind="oracle" if condition == "oracle_sketch" else "runtime",
    )
    encoded_events = example.events
    if condition in LATENT_CONDITIONS:
        encoded_events = events_for_condition(example, condition, shuffled)
        encode_example = (
            shuffled
            if isinstance(encoder, EditSketchICAE) and condition == "shuffled_latent"
            else example
        )
        latent = encode_example_latent(
            encoder,
            model,
            tokenizer,
            table,
            content_to_id,
            encode_example,
            encoded_events,
            args,
            include_public_test=True,
            corrupt=isinstance(encoder, DiscrepancyResidual) and condition == "corrupted_latent",
        )
        latent = latent.to(model.get_input_embeddings().weight.dtype)
        encoded_events = (
            gist_events(encoded_events)
            if isinstance(encoder, EditSketchICAE)
            else select_events(encoded_events, args)
        )
    elif condition in EMBED_CONDITIONS:
        source = shuffled if condition == "shuffled_embed" else example
        if source is None:
            raise ValueError("shuffled_embed requires another example")
        line = discrepancy_text(source.events, swap=condition == "corrupted_embed")
        latent = embed_text(model, tokenizer, line)
        encoded_events = source.events
    else:
        latent = None
    inputs = splice_prompt(model, tokenizer, rendered, latent)
    tokens = greedy_generate(model, inputs, tokenizer.eos_token_id, args.max_new_tokens)
    response = tokenizer.decode(tokens, skip_special_tokens=True)
    prompt_tokens = int(inputs.shape[1]) if condition in EMBED_CONDITIONS else len(
        tokenizer(rendered, add_special_tokens=False).input_ids
    )
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
        "prompt_tokens": prompt_tokens,
        "latent_slots": (
            int(latent.shape[1])
            if condition in EMBED_CONDITIONS and latent is not None
            else args.latent_slots if condition in LATENT_CONDITIONS
            else 0
        ),
        "event_count": len(example.events),
        "encoded_event_count": len(encoded_events),
        "response": response,
        "source": source,
        "validation": validation,
        "passed": bool(validation.get("passed")),
        "outcome": outcome,
    }


def summarize(rows: list[dict]) -> dict:
    summaries = {}
    for condition in sorted({row["condition"] for row in rows}):
        selected = [row for row in rows if row["condition"] == condition]
        by_task = {}
        for row in selected:
            by_task.setdefault(row["task_id"], []).append(row)
        summaries[condition] = {
            "correct": sum(row["passed"] for row in selected),
            "examples": len(selected),
            "tasks": len(by_task),
            "task_correct": sum(any(row["passed"] for row in group) for group in by_task.values()),
            "extraction_failure": sum(row["outcome"] == "extraction_failure" for row in selected),
            "mean_prompt_tokens": sum(row["prompt_tokens"] for row in selected) / len(selected),
        }
    return summaries


def existing_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    return {
        (row["example_id"], row["condition"])
        for row in (json.loads(line) for line in path.read_text().splitlines() if line.strip())
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("A BF16 CUDA GPU is required")
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes / 2**30 < args.min_free_gib:
        raise SystemExit(f"only {free_bytes / 2**30:.1f} GiB free; need {args.min_free_gib:.1f} GiB")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cohort = Path(args.cohort)
    manifest = json.loads((cohort / "manifest.json").read_text())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    conditions = tuple(args.conditions) if args.conditions else (
        TEXT_CONDITIONS if args.mode == "text" else PROTOCOL_CONDITIONS if args.mode == "full" else LATENT_CONDITIONS
    )
    if args.mode == "text":
        allowed = (
            set(TEXT_CONDITIONS)
            | set(DIAGNOSTIC_CONDITIONS)
            | set(HAQUE_TEXT_CONDITIONS)
            | set(EMBED_CONDITIONS)
        )
        conditions = tuple(condition for condition in conditions if condition in allowed)
    needs_encoder = args.mode in {"train", "eval", "full"} or any(
        condition in LATENT_CONDITIONS for condition in conditions
    )

    eval_examples = [
        clip_events(example, args.max_events) for example in load_split(cohort, args.split, args.limit)
    ]
    train_examples = [
        clip_events(example, args.max_events) for example in load_split(cohort, "train", 0)
    ]
    if not eval_examples:
        raise SystemExit(f"no examples in {args.split}")
    if needs_encoder and args.mode in {"train", "full"} and len({example.task_id for example in train_examples}) < 2:
        raise SystemExit("training requires at least two train-split tasks")

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
                "eval_examples": len(eval_examples),
                "eval_tasks": len({example.task_id for example in eval_examples}),
                "train_examples": len(train_examples),
                "gpu_free_gib": round(free_bytes / 2**30, 2),
            }
        ),
        flush=True,
    )

    encoder = None
    table = None
    content_to_id = {}
    train_report = None
    if needs_encoder:
        train_examples = [
            example
            for example in train_examples
            if within_budget(
                tokenizer,
                example,
                args.max_sequence_tokens,
                include_specification=args.include_specification,
                include_public_test=not args.train_hide_public_test,
            )
        ]
        if args.train_value_failures:
            before = len(train_examples)
            train_examples = [example for example in train_examples if is_wrong_value_failure(example)]
            print(
                json.dumps(
                    {
                        "event": "train_value_failures",
                        "kept": len(train_examples),
                        "dropped": before - len(train_examples),
                        "tasks": len({example.task_id for example in train_examples}),
                    }
                ),
                flush=True,
            )
            if len({example.task_id for example in train_examples}) < 2:
                raise SystemExit("train-value-failures left fewer than two tasks")
        if args.train_misses_rows:
            before = len(train_examples)
            missed = missed_no_evidence_tasks(Path(args.train_misses_rows))
            train_examples = [example for example in train_examples if example.task_id in missed]
            print(
                json.dumps(
                    {
                        "event": "train_no_evidence_misses",
                        "kept": len(train_examples),
                        "dropped": before - len(train_examples),
                        "tasks": len({example.task_id for example in train_examples}),
                        "miss_file": args.train_misses_rows,
                    }
                ),
                flush=True,
            )
            if len({example.task_id for example in train_examples}) < 2:
                raise SystemExit("no-evidence misses left fewer than two train tasks")
        if args.edit_sketch_icae or args.discrepancy_residual:
            table = None
            content_to_id = {}
        else:
            vocabulary = sorted(
                {
                    event.content
                    for example in train_examples + eval_examples
                    for event in example.events
                }
                | {"CORRUPTED_RUNTIME"}
            )
            content_to_id = {content: index for index, content in enumerate(vocabulary)}
            table = native_table(model, tokenizer, vocabulary)
        encoder = make_encoder(model, tokenizer, args)
        if args.mode in {"train", "full"}:
            train_report = train_encoder(model, tokenizer, encoder, train_examples, table, content_to_id, args)
            checkpoint = Path(args.checkpoint)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "encoder": encoder.state_dict(),
                    "args": vars(args),
                    "cohort_sha256": manifest.get("sha256"),
                    "trainable_parameters": trainable_parameter_count(encoder),
                    "train_examples": len(train_examples),
                    "train_tasks": len({example.task_id for example in train_examples}),
                },
                checkpoint,
            )
            print(json.dumps({"event": "checkpoint_written", "path": str(checkpoint)}), flush=True)
        elif args.mode == "eval":
            payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            encoder.load_state_dict(payload["encoder"], strict=True)
            encoder.eval()
            print(json.dumps({"event": "checkpoint_loaded", "path": args.checkpoint}), flush=True)

    if args.mode == "train":
        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "args": vars(args),
            "cohort_sha256": manifest.get("sha256"),
            "trainable_parameters": trainable_parameter_count(encoder) if encoder is not None else 0,
            "training": train_report,
        }
        (output_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"event": "train_only_done"}), flush=True)
        return

    row_path = output_dir / "rows.jsonl"
    done = existing_keys(row_path)
    rows = [
        json.loads(line) for line in row_path.read_text().splitlines() if line.strip()
    ] if row_path.exists() else []
    for index, example in enumerate(eval_examples):
        shuffled = different_task_example(eval_examples, index)
        for condition in conditions:
            key = (example.example_id, condition)
            if key in done:
                continue
            began = time.perf_counter()
            row = generate_one(
                model, tokenizer, encoder, table, content_to_id, example, condition, shuffled, args
            )
            row["seconds"] = time.perf_counter() - began
            row["seed"] = args.seed
            with row_path.open("a") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            rows.append(row)
            done.add(key)
            print(
                json.dumps(
                    {
                        "event": "generation",
                        "task_id": example.task_id,
                        "condition": condition,
                        "passed": row["passed"],
                        "outcome": row["outcome"],
                    }
                ),
                flush=True,
            )

    summary = summarize(rows)
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "cohort_sha256": manifest.get("sha256"),
        "eval_examples": len(eval_examples),
        "eval_tasks": len({example.task_id for example in eval_examples}),
        "trainable_parameters": trainable_parameter_count(encoder) if encoder is not None else 0,
        "training": train_report,
        "summary": summary,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "done", "summary": summary}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
