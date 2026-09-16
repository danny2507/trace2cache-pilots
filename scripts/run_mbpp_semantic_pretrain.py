#!/usr/bin/env python3
"""Pretrain the MBPP trace encoder on explicit runtime-semantic controls."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_mbpp_latent_repair import (
    COUNTERFACTUAL_KINDS,
    ROLE_FACTORIZED_SLOTS,
    NumericEvidence,
    NumericExample,
    _as_evidence,
    _counterfactual_evidence,
    _load_or_build_examples,
    _native_content_table,
)
from run_pilot1 import resolve_local_model
from trace2cache.latent import RoleAwareEventEncoder, trainable_parameter_count


VIEW_NAMES = ("true", *COUNTERFACTUAL_KINDS)


@dataclass(frozen=True)
class Batch:
    contents: torch.Tensor
    roles: torch.Tensor
    tests: torch.Tensor
    mask: torch.Tensor


class SemanticPretrainer(nn.Module):
    def __init__(self, encoder: RoleAwareEventEncoder, hidden_width: int, mutation_classes: int):
        super().__init__()
        self.encoder = encoder
        model_width = encoder.output_anchor.shape[1]
        self.validity_head = nn.Sequential(nn.LayerNorm(model_width), nn.Linear(model_width, 1))
        self.corruption_head = nn.Sequential(
            nn.LayerNorm(model_width), nn.Linear(model_width, len(VIEW_NAMES))
        )
        self.mutation_head = nn.Sequential(
            nn.LayerNorm(model_width), nn.Linear(model_width, mutation_classes)
        )
        self.role_head = nn.Linear(hidden_width, encoder.role_embedding.num_embeddings)

    def encode(self, batch: Batch) -> torch.Tensor:
        latents = self.encoder(batch.contents, batch.roles, batch.tests, batch.mask)
        return latents.mean(dim=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--dataset", default=".local/datasets/mbpp/mbpp.jsonl")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-width", type=int, default=256)
    parser.add_argument("--latent-slots", type=int, default=8)
    parser.add_argument("--ranking-margin", type=float, default=0.5)
    parser.add_argument("--validity-weight", type=float, default=1.0)
    parser.add_argument("--corruption-weight", type=float, default=1.0)
    parser.add_argument("--mutation-weight", type=float, default=0.5)
    parser.add_argument("--role-weight", type=float, default=0.5)
    parser.add_argument("--max-train-examples", type=int, default=1024)
    parser.add_argument("--max-eval-examples", type=int, default=192)
    parser.add_argument("--expanded-mutations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--candidate-mutants-per-task", type=int, default=32)
    parser.add_argument("--max-mutants-per-task", type=int, default=4)
    parser.add_argument("--fuzz-calls", type=int, default=16)
    parser.add_argument("--fuzz-bundles-per-mutant", type=int, default=1)
    parser.add_argument("--max-events-per-test", type=int, default=24)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--cache-dir", default=".local/cache/mbpp_latent_repair")
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--min-free-gib", type=float, default=12.0)
    parser.add_argument(
        "--output", default="artifacts/mbpp_repair/semantic_pretrain_seed211.json"
    )
    parser.add_argument(
        "--checkpoint", default="checkpoints/mbpp_repair/semantic_pretrain_seed211.pt"
    )
    return parser.parse_args()


def mutation_family(kind: str) -> str:
    return kind.removesuffix(":fuzz").split(":", 1)[0]


def make_batch(views: list[NumericEvidence], table: torch.Tensor, device) -> Batch:
    maximum = max(len(view.content_ids) for view in views)
    width = table.shape[1]
    contents = torch.zeros(len(views), maximum, width, device=device)
    roles = torch.zeros(len(views), maximum, dtype=torch.long, device=device)
    tests = torch.zeros_like(roles)
    mask = torch.zeros(len(views), maximum, dtype=torch.bool, device=device)
    for row, view in enumerate(views):
        length = len(view.content_ids)
        ids = torch.tensor(view.content_ids, dtype=torch.long, device=device)
        contents[row, :length] = table[ids]
        roles[row, :length] = torch.tensor(view.roles, dtype=torch.long, device=device)
        tests[row, :length] = torch.tensor(view.test_ids, dtype=torch.long, device=device)
        mask[row, :length] = True
    return Batch(contents, roles, tests, mask)


def numeric_examples(raw, vocabulary):
    content_to_id = {content: index for index, content in enumerate(vocabulary)}
    return [
        NumericExample(example, tuple(content_to_id[value] for value in example.contents))
        for example in raw
    ]


@torch.inference_mode()
def evaluate(model, examples, table, mutation_to_id, batch_size):
    model.eval()
    device = table.device
    validity_correct = {name: 0 for name in COUNTERFACTUAL_KINDS}
    validity_total = {name: 0 for name in COUNTERFACTUAL_KINDS}
    corruption_correct = {name: 0 for name in VIEW_NAMES}
    corruption_total = {name: 0 for name in VIEW_NAMES}
    mutation_correct = 0
    mutation_total = 0
    role_correct = 0
    role_total = 0
    score_gaps = {name: [] for name in COUNTERFACTUAL_KINDS}

    for start in range(0, len(examples), batch_size):
        selected = examples[start : start + batch_size]
        views = []
        labels = []
        for example in selected:
            views.append(_as_evidence(example))
            labels.append(0)
            for class_id, kind in enumerate(COUNTERFACTUAL_KINDS, start=1):
                views.append(_counterfactual_evidence(example, kind))
                labels.append(class_id)
        batch = make_batch(views, table, device)
        representation = model.encode(batch)
        validity = model.validity_head(representation).squeeze(-1)
        corruption = model.corruption_head(representation).argmax(-1)
        labels_tensor = torch.tensor(labels, device=device)
        for index, label in enumerate(labels):
            name = VIEW_NAMES[label]
            corruption_correct[name] += int(corruption[index].item() == label)
            corruption_total[name] += 1
        for row in range(len(selected)):
            true_score = validity[row * len(VIEW_NAMES)]
            for offset, kind in enumerate(COUNTERFACTUAL_KINDS, start=1):
                negative = validity[row * len(VIEW_NAMES) + offset]
                gap = (true_score - negative).item()
                score_gaps[kind].append(gap)
                validity_correct[kind] += int(gap > 0)
                validity_total[kind] += 1

        true_indices = torch.arange(0, len(views), len(VIEW_NAMES), device=device)
        true_repr = representation[true_indices]
        mutation_labels = torch.tensor(
            [
                mutation_to_id.get(
                    mutation_family(example.example.mutation_kind), mutation_to_id["<other>"]
                )
                for example in selected
            ],
            device=device,
        )
        mutation_prediction = model.mutation_head(true_repr).argmax(-1)
        mutation_correct += (mutation_prediction == mutation_labels).sum().item()
        mutation_total += len(selected)

        true_batch = make_batch([_as_evidence(example) for example in selected], table, device)
        projected = model.encoder.content_projection(true_batch.contents.float())
        role_prediction = model.role_head(projected).argmax(-1)
        role_correct += ((role_prediction == true_batch.roles) & true_batch.mask).sum().item()
        role_total += true_batch.mask.sum().item()

    return {
        "pairwise_validity_accuracy": {
            kind: validity_correct[kind] / validity_total[kind]
            for kind in COUNTERFACTUAL_KINDS
        },
        "mean_validity_score_gap": {
            kind: sum(score_gaps[kind]) / len(score_gaps[kind])
            for kind in COUNTERFACTUAL_KINDS
        },
        "corruption_accuracy": {
            name: corruption_correct[name] / corruption_total[name] for name in VIEW_NAMES
        },
        "corruption_macro_accuracy": sum(
            corruption_correct[name] / corruption_total[name] for name in VIEW_NAMES
        ) / len(VIEW_NAMES),
        "mutation_accuracy": mutation_correct / mutation_total,
        "event_role_accuracy": role_correct / role_total,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("A BF16 CUDA GPU is required; this pilot never falls back to CPU")
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes / 2**30 < args.min_free_gib:
        raise SystemExit(f"only {free_bytes / 2**30:.1f} GiB free")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_raw = _load_or_build_examples(args, "train")
    eval_raw = _load_or_build_examples(args, "validation")
    random.Random(args.seed).shuffle(train_raw)
    random.Random(args.seed + 1).shuffle(eval_raw)
    train_raw = train_raw[: args.max_train_examples]
    eval_raw = eval_raw[: args.max_eval_examples]
    vocabulary = sorted(
        {content for example in train_raw + eval_raw for content in example.contents}
    )
    mutation_names = [
        "<other>",
        *sorted({mutation_family(example.mutation_kind) for example in train_raw}),
    ]
    mutation_to_id = {name: index for index, name in enumerate(mutation_names)}

    model_path = resolve_local_model(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    receiver = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.bfloat16
    ).to("cuda").eval()
    receiver.requires_grad_(False)
    table = _native_content_table(receiver, tokenizer, vocabulary)
    anchor_ids = tokenizer(
        " runtime evidence observed behavior state values result details",
        add_special_tokens=False,
    ).input_ids[: args.latent_slots]
    anchor_ids += [anchor_ids[-1]] * (args.latent_slots - len(anchor_ids))
    anchors = receiver.get_input_embeddings()(
        torch.tensor(anchor_ids, device=receiver.device)
    ).detach()
    model_width = receiver.config.hidden_size
    del receiver
    torch.cuda.empty_cache()

    train = numeric_examples(train_raw, vocabulary)
    evaluation = numeric_examples(eval_raw, vocabulary)
    encoder = RoleAwareEventEncoder(
        model_width=model_width,
        output_anchor=anchors,
        hidden_width=args.hidden_width,
        num_roles=8,
        max_events=args.max_events_per_test * 2 + 4,
        max_tests=2,
        slot_roles=ROLE_FACTORIZED_SLOTS,
    ).to("cuda")
    model = SemanticPretrainer(encoder, args.hidden_width, len(mutation_names)).to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    rng = random.Random(args.seed)
    started = time.perf_counter()

    for step in range(1, args.steps + 1):
        selected = [train[rng.randrange(len(train))] for _ in range(args.batch_size)]
        views = []
        corruption_labels = []
        for example in selected:
            views.append(_as_evidence(example))
            corruption_labels.append(0)
            for class_id, kind in enumerate(COUNTERFACTUAL_KINDS, start=1):
                views.append(_counterfactual_evidence(example, kind))
                corruption_labels.append(class_id)
        batch = make_batch(views, table, "cuda")
        representation = model.encode(batch)
        validity = model.validity_head(representation).squeeze(-1)
        corruption_logits = model.corruption_head(representation)
        labels = torch.tensor(corruption_labels, device="cuda")
        validity_targets = (labels == 0).float()
        validity_loss = nn.functional.binary_cross_entropy_with_logits(validity, validity_targets)
        corruption_loss = nn.functional.cross_entropy(corruption_logits, labels)
        true_scores = validity[:: len(VIEW_NAMES)]
        ranking_loss = sum(
            nn.functional.relu(
                args.ranking_margin
                + validity[offset :: len(VIEW_NAMES)]
                - true_scores
            ).mean()
            for offset in range(1, len(VIEW_NAMES))
        ) / (len(VIEW_NAMES) - 1)

        true_repr = representation[:: len(VIEW_NAMES)]
        mutation_labels = torch.tensor(
            [
                mutation_to_id.get(
                    mutation_family(example.example.mutation_kind), mutation_to_id["<other>"]
                )
                for example in selected
            ],
            device="cuda",
        )
        mutation_loss = nn.functional.cross_entropy(
            model.mutation_head(true_repr), mutation_labels
        )
        true_batch = make_batch([_as_evidence(example) for example in selected], table, "cuda")
        projected = model.encoder.content_projection(true_batch.contents.float())
        role_logits = model.role_head(projected)
        role_loss = nn.functional.cross_entropy(
            role_logits[true_batch.mask], true_batch.roles[true_batch.mask]
        )
        loss = (
            ranking_loss
            + args.validity_weight * validity_loss
            + args.corruption_weight * corruption_loss
            + args.mutation_weight * mutation_loss
            + args.role_weight * role_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": round(loss.item(), 5),
                        "ranking": round(ranking_loss.item(), 5),
                        "validity": round(validity_loss.item(), 5),
                        "corruption": round(corruption_loss.item(), 5),
                        "mutation": round(mutation_loss.item(), 5),
                        "role": round(role_loss.item(), 5),
                        "gpu_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
                    }
                ),
                flush=True,
            )

    metrics = evaluate(model, evaluation, table, mutation_to_id, args.batch_size)
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "seed": args.seed,
        "steps": args.steps,
        "train_examples": len(train),
        "train_tasks": len({example.example.task_id for example in train}),
        "eval_examples": len(evaluation),
        "eval_tasks": len({example.example.task_id for example in evaluation}),
        "mutation_families": mutation_names,
        "trainable_parameters": trainable_parameter_count(model),
        "training_seconds": time.perf_counter() - started,
        "gpu_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
        "metrics": metrics,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder": {
                name: tensor.detach().cpu() for name, tensor in encoder.state_dict().items()
            },
            "heads": {
                name: tensor.detach().cpu()
                for name, tensor in model.state_dict().items()
                if not name.startswith("encoder.")
            },
            "config": vars(args),
            "result": result,
        },
        checkpoint,
    )
    print(json.dumps({"event": "final", **metrics, "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
