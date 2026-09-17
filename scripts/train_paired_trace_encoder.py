#!/usr/bin/env python3
"""Train the v2 runtime adapter against fixed oracle repair codes.

This intentionally starts with the Phase-D0 vector objective. Decoder ranking is kept in
``receiver_training.py`` and will be enabled only after this data/feature/gradient gate can
overfit its fixed 48-view panel.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from trace2cache.ambiguous_repair import get_ambiguous_cases
from trace2cache.latent import RoleAwareEventEncoder, trainable_parameter_count
from trace2cache.native_features import FrozenNativeFeatureExtractor, collate_views
from trace2cache.paired_evidence import read_jsonl
from trace2cache.receiver_training import vector_loss, oracle_identity_loss
from trace2cache.receiver_training import paired_patch_loss, splice_prompt

sys.path.insert(0, str(Path(__file__).parent))
from run_pilot1 import resolve_local_model
from run_repair_codebook import NativeRepairCodebook, initial_native_codes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--dataset", default=".local/datasets/paired_runtime_v2_smoke/train.jsonl")
    parser.add_argument("--codebook-checkpoint", default="checkpoints/repair_latent/expanded_codebook_seed313.pt")
    parser.add_argument("--feature-method", choices=("mean0", "context2"), default="context2")
    parser.add_argument("--objective", choices=("vector", "vector_identity", "decoder"), default="vector")
    parser.add_argument("--identity-weight", type=float, default=0.1)
    parser.add_argument("--identity-temperature", type=float, default=0.1)
    parser.add_argument("--feature-cache", default=".local/cache/paired_runtime_v2/features")
    parser.add_argument("--feature-batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=8, help="matched pairs per decoder update; processed in one same-family batch")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--init-checkpoint", help="v2 encoder checkpoint used to fork a post-warmup objective")
    parser.add_argument("--hidden-width", type=int, default=256)
    parser.add_argument("--max-events", type=int, default=512)
    parser.add_argument("--max-tests", type=int, default=8)
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument("--checkpoint", default="checkpoints/paired_runtime_v2/overfit_context2_vector_seed401.pt")
    parser.add_argument("--output", default="artifacts/paired_runtime_v2/overfit_context2_vector_seed401.json")
    return parser.parse_args()


def load_oracle_codes(model: object, tokenizer: object, checkpoint: str) -> torch.Tensor:
    items = get_ambiguous_cases()
    codebook = NativeRepairCodebook(initial_native_codes(model, tokenizer, items, 8)).to(model.device)
    payload = torch.load(checkpoint, map_location=model.device, weights_only=False)
    missing, unexpected = codebook.load_state_dict(payload["adapter_trainable"], strict=False)
    if missing != ["anchors"] or unexpected: raise RuntimeError(f"oracle checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    with torch.no_grad():
        indices = torch.arange(len(items), device=model.device)
        codes = codebook(indices).float()
    if tuple(codes.shape) != (24, 8, model.config.hidden_size): raise RuntimeError(f"unexpected oracle shape: {tuple(codes.shape)}")
    return codes


def all_views(records: list[object]) -> tuple[list[object], torch.Tensor]:
    views, labels = [], []
    for record in records:
        views.extend((record.evidence_a, record.evidence_b)); labels.extend((record.label_a, record.label_b))
    return views, torch.tensor(labels, dtype=torch.long)


def build_feature_map(extractor: FrozenNativeFeatureExtractor, views: list[object], method: str, batch_size: int) -> dict[str, torch.Tensor]:
    contents = list(dict.fromkeys(event.content for view in views for event in view.events))
    # The workspace is on network storage. One validated pack avoids hundreds of tiny
    # per-payload reads on every resumed training/evaluation job. Individual cache keys
    # still encode all extractor settings and exact payloads.
    keys = [extractor.cache_key(content, method).digest() for content in contents]
    digest = hashlib.sha256(json.dumps(keys).encode()).hexdigest()
    packed_path = extractor.cache_dir / "packs" / f"{digest}.pt"
    if packed_path.exists():
        packed = torch.load(packed_path, map_location="cpu", weights_only=True)
        if packed["keys"] != keys: raise RuntimeError("packed feature keys mismatch")
        vectors = packed["vectors"].to(extractor.device)
        if vectors.shape != (len(contents), extractor.width) or not torch.isfinite(vectors).all(): raise RuntimeError("invalid packed features")
    else:
        vectors = extractor.extract_contents(contents, method, batch_size=batch_size)
        packed_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = packed_path.with_suffix(".tmp")
        torch.save({"keys": keys, "vectors": vectors.detach().cpu()}, temporary)
        temporary.replace(packed_path)
    return {content: vector.detach() for content, vector in zip(contents, vectors)}


@torch.inference_mode()
def evaluate(encoder: RoleAwareEventEncoder, views: list[object], labels: torch.Tensor, feature_map: dict[str, torch.Tensor], oracle: torch.Tensor, *, batch_size: int) -> dict[str, float]:
    predictions = []
    for start in range(0, len(views), batch_size):
        batch = collate_views(views[start:start + batch_size], feature_map, oracle.device)
        predictions.append(encoder(**batch))
    latent = torch.cat(predictions).float(); target = oracle[labels.to(oracle.device)]
    similarities = torch.nn.functional.cosine_similarity(latent.flatten(1).unsqueeze(1), oracle.flatten(1).unsqueeze(0), dim=-1)
    return {"nearest_code_accuracy": (similarities.argmax(1) == labels.to(oracle.device)).float().mean().item(), "mean_cosine": torch.nn.functional.cosine_similarity(latent.flatten(1), target.flatten(1), dim=-1).mean().item()}


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise SystemExit("CUDA BF16 is required")
    torch.manual_seed(args.seed); random.seed(args.seed); torch.cuda.reset_peak_memory_stats()
    records = read_jsonl(args.dataset); views, labels = all_views(records)
    if max(len(view.events) for view in views) > args.max_events: raise RuntimeError("dataset exceeds max-events")
    model_path = resolve_local_model(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    receiver = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="eager").to("cuda").eval()
    for parameter in receiver.parameters(): parameter.requires_grad_(False)
    oracle = load_oracle_codes(receiver, tokenizer, args.codebook_checkpoint)
    extractor = FrozenNativeFeatureExtractor(receiver, tokenizer, model_id=args.model, cache_dir=args.feature_cache)
    started_features = time.perf_counter(); feature_map = build_feature_map(extractor, views, args.feature_method, args.feature_batch_size); feature_seconds = time.perf_counter() - started_features
    encoder = RoleAwareEventEncoder(model_width=receiver.config.hidden_size, output_anchor=oracle.mean(0), hidden_width=args.hidden_width, max_events=args.max_events, max_tests=args.max_tests).to("cuda")
    if args.init_checkpoint:
        payload = torch.load(args.init_checkpoint, map_location=receiver.device, weights_only=False)
        missing, unexpected = encoder.load_state_dict(payload["encoder"], strict=False)
        if missing or unexpected: raise RuntimeError(f"encoder warm-start mismatch: missing={missing}, unexpected={unexpected}")
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.learning_rate, weight_decay=0.01)
    order_rng = random.Random(args.seed + 17); history = []; started = time.perf_counter()
    rendered_prompts = []
    targets = []
    if args.objective == "decoder":
        from run_repair_codebook import MARKER, render, target_ids
        for item in get_ambiguous_cases():
            rendered_prompts.append(render(tokenizer, item.case, MARKER))
            targets.append(target_ids(tokenizer, item.correct_source, receiver.device))
    encoder.train()
    for update in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        if args.objective in ("vector", "vector_identity"):
            selected = [order_rng.randrange(len(views)) for _ in range(args.batch_size)]
            batch = collate_views([views[index] for index in selected], feature_map, receiver.device)
            target = oracle[labels[selected].to(receiver.device)]
            latent = encoder(**batch)
            loss = vector_loss(latent, target)
            diagnostics = {"vector": float(loss.detach())}
            if args.objective == "vector_identity":
                identity = oracle_identity_loss(latent, oracle, labels[selected].to(receiver.device), temperature=args.identity_temperature)
                loss = loss + args.identity_weight * identity
                diagnostics["identity"] = float(identity.detach())
            loss.backward()
        else:
            # Samples from one family share a visible prompt, so batching preserves the
            # per-pair mean objective while avoiding 16 serial frozen-receiver forwards.
            # A and B remain separate, and each has positive + paired-negative candidates.
            component_sums = {"nll": 0.0, "rank": 0.0, "vector": 0.0, "margin": 0.0}
            total_loss = 0.0
            family = order_rng.choice(tuple(sorted({record.family_id for record in records})))
            candidates = [record for record in records if record.family_id == family]
            chosen = [candidates[order_rng.randrange(len(candidates))] for _ in range(args.gradient_accumulation)]
            for side in ("a", "b"):
                side_views = [getattr(record, f"evidence_{side}") for record in chosen]
                label = getattr(chosen[0], f"label_{side}"); other_label = getattr(chosen[0], f"label_{'b' if side == 'a' else 'a'}")
                if any(getattr(record, f"label_{side}") != label for record in chosen): raise RuntimeError("family label mismatch")
                batch = collate_views(side_views, feature_map, receiver.device)
                latent = encoder(**batch)
                prompt = splice_prompt(receiver, tokenizer, rendered_prompts[label], latent)
                positive = targets[label].expand(len(chosen), -1)
                negative = targets[other_label].expand(len(chosen), -1)
                code = oracle[label].unsqueeze(0).expand(len(chosen), -1, -1)
                components = paired_patch_loss(receiver, prompt, positive, negative, code, latent)
                (components["loss"] / 2).backward()
                total_loss += float(components["loss"].detach()) / 2
                for name in component_sums: component_sums[name] += float(components[name].detach()) / 2
            loss = torch.tensor(total_loss, device=receiver.device)
            diagnostics = component_sums
        grad_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0); optimizer.step()
        if update == 1 or update % 50 == 0 or update == args.steps:
            encoder.eval(); metrics = evaluate(encoder, views, labels, feature_map, oracle, batch_size=args.batch_size); encoder.train()
            history.append({"update": update, "loss": float(loss.detach()), "grad_norm": float(grad_norm), **diagnostics, **metrics})
            print(json.dumps(history[-1], sort_keys=True), flush=True)
    encoder.eval(); final = evaluate(encoder, views, labels, feature_map, oracle, batch_size=args.batch_size)
    payload = {"schema_version": 2, "timestamp": datetime.now(timezone.utc).isoformat(), "args": vars(args), "behavior_ids": [item.case.case_id for item in get_ambiguous_cases()], "encoder": encoder.state_dict(), "oracle_code_shape": list(oracle.shape), "final": final, "history": history}
    checkpoint = Path(args.checkpoint); checkpoint.parent.mkdir(parents=True, exist_ok=True); temporary = checkpoint.with_suffix(".tmp"); torch.save(payload, temporary); temporary.replace(checkpoint)
    result = {"timestamp": payload["timestamp"], "args": vars(args), "dataset": args.dataset, "views": len(views), "unique_payloads": len(feature_map), "feature_seconds": feature_seconds, "train_seconds": time.perf_counter() - started, "trainable_parameters": trainable_parameter_count(encoder), "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3), "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3), "history": history, "final": final}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"final": final, "output": str(output)}, sort_keys=True))


if __name__ == "__main__": main()
