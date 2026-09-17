#!/usr/bin/env python3
"""Generate the audited, matched paired-runtime v2 datasets (CPU trace collection only)."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from trace2cache.paired_evidence import PRIMARY_DISTRIBUTIONS, audit_records, generate_split, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=".local/datasets/paired_runtime_v2")
    parser.add_argument("--manifest", default="artifacts/paired_runtime_v2/data_manifest.json")
    parser.add_argument("--audit", default="artifacts/paired_runtime_v2/data_audit.json")
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument("--train-pairs", type=int, default=64)
    parser.add_argument("--eval-pairs", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    used_hashes: set[str] = set(); all_records = []; split_stats = {}
    split_counts = {"train": args.train_pairs, **{name: args.eval_pairs for name in PRIMARY_DISTRIBUTIONS if name != "train"}}
    for index, (split, distribution) in enumerate(PRIMARY_DISTRIBUTIONS.items()):
        records, rejected = generate_split(split, split_counts[split], seed=args.seed + index * 1009, distribution=distribution, used_hashes=used_hashes)
        destination = output_dir / f"{split}.jsonl"; write_jsonl(records, destination)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        split_stats[split] = {"records": len(records), "rejected": rejected, "sha256": digest, "distribution": asdict(distribution)}
        all_records.extend(records)
    audit = audit_records(all_records)
    audit["cross_split_input_hashes"] = len(used_hashes)
    Path(args.audit).parent.mkdir(parents=True, exist_ok=True)
    Path(args.audit).write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    manifest = {"schema_version": 2, "seed": args.seed, "splits": split_stats, "audit": str(args.audit), "passed": audit["passed"]}
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if not audit["passed"]: raise SystemExit("paired evidence audit failed")
    print(json.dumps({"manifest": str(args.manifest), "audit": str(args.audit), "records": len(all_records)}, sort_keys=True))


if __name__ == "__main__": main()
