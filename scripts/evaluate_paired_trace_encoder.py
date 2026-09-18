#!/usr/bin/env python3
"""Execute repair and matched-evidence interventions for a saved v2 encoder.

Both sides of every selected pair are evaluated. Training-panel evaluation is explicitly
marked in the report and never counted as unseen-program generalization.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from trace2cache.ambiguous_repair import get_ambiguous_cases
from trace2cache.latent import RoleAwareEventEncoder
from trace2cache.native_features import FrozenNativeFeatureExtractor, collate_views
from trace2cache.paired_evidence import EXPECTED, STATUS, EvidenceView, read_jsonl
from trace2cache.evidence_controls import corrupt_runtime_keep_io, io_only
from trace2cache.receiver_training import splice_prompt
from trace2cache.sandbox import EVALUATOR_REVISION, evaluate_patch, extract_function
from run_pilot1 import resolve_local_model
from run_repair_codebook import MARKER, render
from run_repair_latent_pool import greedy_generate
from train_paired_trace_encoder import build_feature_map, load_oracle_codes

CONDITIONS = ("true_latent", "paired_swap", "nearest_oracle_latent", "paired_swap_nearest", "no_evidence", "oracle_latent", "expected_and_status_removed", "no_roles")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default=".local/datasets/paired_runtime_v2_smoke/train.jsonl")
    parser.add_argument("--pairs-per-family", type=int, default=1)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=["true_latent", "paired_swap"])
    parser.add_argument("--heldout-wording", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--cache-fixed-codes", action="store_true", help="reuse decoding only for exactly equal prompt embeddings and fixed oracle/projected codes; never continuous latents")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def controlled_view(view: EvidenceView, condition: str) -> EvidenceView:
    if condition == "expected_and_status_removed":
        return replace(view, events=tuple(replace(event, content="UNKNOWN") if event.role_id in (EXPECTED, STATUS) else event for event in view.events))
    if condition == "no_roles":
        return replace(view, events=tuple(replace(event, role_id=0) for event in view.events))
    return view


def selected_evidence_view(view: EvidenceView, name: str) -> EvidenceView:
    if name == "full_runtime": return view
    if name == "io_only": return io_only(view)
    if name == "runtime_corrupted_keep_io": return corrupt_runtime_keep_io(view)
    raise ValueError(f"unknown checkpoint evidence view: {name}")


def validate(response, case):
    try:
        source = extract_function(response, case.function_name)
    except Exception as error:
        return None, {
            "passed": False,
            "outcome": "extraction_failure",
            "error": f"{type(error).__name__}: {error}",
            "tests": [],
            "evaluator_revision": EVALUATOR_REVISION,
        }
    # Keep an extracted candidate even if policy validation or execution later fails.
    return source, evaluate_patch(source, case)


def summarize(rows):
    summaries = {}
    for condition in sorted({row["condition"] for row in rows}):
        selected = [row for row in rows if row["condition"] == condition]
        pairs = {}
        for row in selected: pairs.setdefault(row["pair_uid"], []).append(row)
        summaries[condition] = {
            "correct": sum(row["intended"]["passed"] for row in selected), "views": len(selected),
            "both_correct": sum(len(pair) == 2 and all(row["intended"]["passed"] for row in pair) for pair in pairs.values()),
            "pairs": len(pairs), "opposite_correct": sum(row["opposite"]["passed"] for row in selected),
            "cap_hits": sum(row["cap_hit"] for row in selected),
        }
    return summaries


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def _snapshot_metadata(model_path: str | Path) -> dict[str, object]:
    root = Path(model_path)
    names = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
    return {
        "path": str(root.resolve()),
        "files": {name: _sha256_file(root / name) for name in names if (root / name).is_file()},
    }


def _evaluation_manifest(args, config, records, checkpoint_hash, dataset_hash, model_path, tokenizer, items) -> dict:
    test_suite = [asdict(item.case) for item in items]
    manifest = {
        "schema_version": 1,
        "checkpoint_sha256": checkpoint_hash,
        "dataset_sha256": dataset_hash,
        "checkpoint_config": config,
        "model": _snapshot_metadata(model_path),
        "tokenizer": {
            "vocab_size": tokenizer.vocab_size,
            "special_token_ids": {
                "bos": tokenizer.bos_token_id,
                "eos": tokenizer.eos_token_id,
                "pad": tokenizer.pad_token_id,
            },
        },
        "codebook": {
            "path": str(Path(config["codebook_checkpoint"]).resolve()),
            "sha256": _sha256_file(Path(config["codebook_checkpoint"])),
        },
        "test_suite_sha256": hashlib.sha256(
            json.dumps(test_suite, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "panel_pair_uids": [record.pair_uid for record in records],
        "conditions": list(args.conditions),
        "generation": {
            "heldout_wording": args.heldout_wording,
            "max_new_tokens": args.max_new_tokens,
            "cache_fixed_codes": args.cache_fixed_codes,
        },
        "evaluator_revision": EVALUATOR_REVISION,
        "code_revision": _git_revision(),
    }
    manifest["sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return manifest


@torch.inference_mode()
def audit_encoder_padding(encoder, views, feature_map, oracle, *, typed_values: bool, binding_relations: bool, typed_value_mode: str):
    """Check that repair-time singleton encoding matches the padded training evaluation."""
    singleton = torch.cat([encoder(**collate_views([view], feature_map, oracle.device, typed_values=typed_values, binding_relations=binding_relations, typed_value_mode=typed_value_mode)) for view in views])
    grouped = torch.cat([encoder(**collate_views(views[start:start + 16], feature_map, oracle.device, typed_values=typed_values, binding_relations=binding_relations, typed_value_mode=typed_value_mode)) for start in range(0, len(views), 16)])
    difference = singleton.float() - grouped.float()
    singleton_codes = torch.nn.functional.cosine_similarity(singleton.flatten(1).unsqueeze(1), oracle.flatten(1).unsqueeze(0), dim=-1).argmax(1)
    grouped_codes = torch.nn.functional.cosine_similarity(grouped.flatten(1).unsqueeze(1), oracle.flatten(1).unsqueeze(0), dim=-1).argmax(1)
    return {"max_abs": float(difference.abs().max()), "rmse": float(difference.square().mean().sqrt()), "code_disagreements": int((singleton_codes != grouped_codes).sum()), "views": len(views)}


def main():
    args = parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise SystemExit("CUDA BF16 required")
    torch.cuda.reset_peak_memory_stats()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = payload["args"]
    model_path = resolve_local_model(config["model"])
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    receiver = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="eager").to("cuda").eval()
    receiver.requires_grad_(False)
    oracle = load_oracle_codes(receiver, tokenizer, config["codebook_checkpoint"])
    from trace2cache.typed_values import FEATURE_DIM
    from trace2cache.binding_encoder import NUM_RELATIONS
    typed_values = bool(config.get("typed_values", False))
    typed_value_mode = config.get("typed_value_mode", "legacy_virtual_nodes")
    binding_relations = bool(config.get("binding_relations", False))
    evidence_view = config.get("evidence_view", "full_runtime")
    encoder = RoleAwareEventEncoder(model_width=receiver.config.hidden_size, output_anchor=oracle.mean(0), hidden_width=config["hidden_width"], max_events=config["max_events"], max_tests=config["max_tests"], typed_feature_dim=FEATURE_DIM if typed_values else 0, relation_types=NUM_RELATIONS if binding_relations else 0).to("cuda").eval()
    encoder.load_state_dict(payload["encoder"], strict=True)
    counts = {}; records = []
    for record in read_jsonl(args.dataset):
        count = counts.get(record.family_id, 0)
        if count < args.pairs_per_family: records.append(record); counts[record.family_id] = count + 1
    if not records: raise ValueError("empty evaluation panel")
    views = [controlled_view(selected_evidence_view(view, evidence_view), condition) for record in records for view in (record.evidence_a, record.evidence_b) for condition in args.conditions]
    extractor = FrozenNativeFeatureExtractor(receiver, tokenizer, model_id=config["model"], cache_dir=config["feature_cache"])
    feature_map = build_feature_map(extractor, views, config["feature_method"], config["feature_batch_size"])
    padding_audit = audit_encoder_padding(encoder, [selected_evidence_view(view, evidence_view) for record in records for view in (record.evidence_a, record.evidence_b)], feature_map, oracle, typed_values=typed_values, binding_relations=binding_relations, typed_value_mode=typed_value_mode)
    print(json.dumps({"encoder_padding_audit": padding_audit}), flush=True)
    items = get_ambiguous_cases()
    checkpoint_hash = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    dataset_hash = hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest()
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _evaluation_manifest(
        args, config, records, checkpoint_hash, dataset_hash, model_path, tokenizer, items
    )
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("resume manifest differs from checkpoint, panel, conditions, or evaluation dependencies")
    elif (output_dir / "rows.jsonl").exists():
        raise ValueError("legacy rows lack an immutable manifest; select a new output directory")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    row_path = output_dir / "rows.jsonl"
    rows = []
    if row_path.exists():
        rows = [json.loads(line) for line in row_path.read_text().splitlines() if line]
        if any(row.get("evaluation_manifest_sha256") != manifest["sha256"] for row in rows):
            raise ValueError("resume rows do not belong to the immutable evaluation manifest")
    done = {(row["pair_uid"], row["side"], row["condition"]) for row in rows}
    fixed_code_cache = {}
    with torch.inference_mode():
        for record in records:
            for side in ("a", "b"):
                other = "b" if side == "a" else "a"
                label = getattr(record, f"label_{side}"); opposite_label = getattr(record, f"label_{other}")
                for condition in args.conditions:
                    if (record.pair_uid, side, condition) in done: continue
                    began = time.perf_counter()
                    evidence_side = other if condition in ("paired_swap", "paired_swap_nearest") else side
                    source_view = selected_evidence_view(
                        getattr(record, f"evidence_{evidence_side}"), evidence_view
                    )
                    view = controlled_view(source_view, condition)
                    latent = encoder(**collate_views([view], feature_map, receiver.device, typed_values=typed_values, binding_relations=binding_relations, typed_value_mode=typed_value_mode))
                    similarity = torch.nn.functional.cosine_similarity(latent.flatten(1), oracle.flatten(1), dim=-1)
                    nearest_label = int(similarity.argmax())
                    fixed_label = None
                    if condition in ("nearest_oracle_latent", "paired_swap_nearest"):
                        latent = oracle[nearest_label].unsqueeze(0); fixed_label = nearest_label
                    if condition == "oracle_latent":
                        latent = oracle[label].unsqueeze(0); fixed_label = label
                    if condition == "no_evidence": latent = None
                    rendered = render(tokenizer, items[label].case, "unavailable" if latent is None else MARKER, heldout=args.heldout_wording)
                    inputs = splice_prompt(receiver, tokenizer, rendered, latent)
                    cache_key = (rendered, fixed_label) if args.cache_fixed_codes and fixed_label is not None else None
                    cached = fixed_code_cache.get(cache_key) if cache_key is not None else None
                    if cached is None:
                        generated = greedy_generate(receiver, inputs, tokenizer.eos_token_id, args.max_new_tokens)
                        if cache_key is not None:
                            fixed_code_cache[cache_key] = (inputs.detach().cpu().clone(), generated)
                    else:
                        cached_inputs, generated = cached
                        if not torch.equal(inputs.detach().cpu(), cached_inputs):
                            raise RuntimeError("fixed-code decoding cache inputs differ; refusing approximate reuse")
                    response = tokenizer.decode(generated, skip_special_tokens=True)
                    patch, intended = validate(response, items[label].case)
                    _, opposite = validate(response, items[opposite_label].case)
                    row = {"evaluation_manifest_sha256": manifest["sha256"], "checkpoint_sha256": checkpoint_hash, "dataset_sha256": dataset_hash, "pair_uid": record.pair_uid, "family": record.family_id, "split": record.split, "side": side, "condition": condition, "heldout_wording": args.heldout_wording, "max_new_tokens": args.max_new_tokens, "nearest_label": nearest_label, "label": label, "response": response, "patch": patch, "intended": intended, "opposite": opposite, "generation_cache_hit": cached is not None, "cap_hit": len(generated) == args.max_new_tokens and int(generated[-1]) != tokenizer.eos_token_id, "seconds": time.perf_counter() - began}
                    with row_path.open("a") as handle: handle.write(json.dumps(row, sort_keys=True) + "\n")
                    rows.append(row)
                    print(json.dumps({"family": record.family_id, "side": side, "condition": condition, "correct": intended["passed"], "opposite": opposite["passed"]}), flush=True)
    result = {"args": vars(args), "evaluation_manifest_sha256": manifest["sha256"], "checkpoint_sha256": checkpoint_hash, "dataset_sha256": dataset_hash, "training_panel": str(Path(args.dataset).resolve()) == str(Path(config["dataset"]).resolve()), "encoder_padding_audit": padding_audit, "summary": summarize(rows), "generation_cache_hits": sum(row.get("generation_cache_hit", False) for row in rows), "fixed_code_cache_entries_this_run": len(fixed_code_cache), "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], sort_keys=True), flush=True)


if __name__ == "__main__": main()
