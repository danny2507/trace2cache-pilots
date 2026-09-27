#!/usr/bin/env python3
"""Freeze the MBPP public/hidden repair cohort before any adapter training."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from trace2cache.mbpp import load_mbpp
from trace2cache.mbpp_generalization import (
    SCHEMA_VERSION,
    collect_examples,
    example_to_json,
    load_mbppplus_assertions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=".local/datasets/mbpp/mbpp.jsonl")
    parser.add_argument("--mbppplus", default=".local/datasets/mbppplus/MBPPPlus-v0.2.0.jsonl.gz")
    parser.add_argument("--max-mutants", type=int, default=12)
    parser.add_argument("--max-events-per-test", type=int, default=24)
    parser.add_argument("--train-per-task", type=int, default=4)
    parser.add_argument("--eval-per-task", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--output-dir", default="artifacts/mbpp_generalization/cohort_v1")
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plus_path = Path(args.mbppplus)
    plus_by_id = load_mbppplus_assertions(plus_path) if plus_path.exists() else {}
    split_counts = {}
    all_examples = []
    all_audits = []
    for split, per_task in (
        ("train", args.train_per_task),
        ("validation", args.eval_per_task),
        ("test", args.eval_per_task),
    ):
        tasks = load_mbpp(args.dataset, split=split)
        examples, audits = collect_examples(
            tasks,
            plus_by_id=plus_by_id,
            max_mutants=args.max_mutants,
            max_selected_per_task=per_task,
            max_events_per_test=args.max_events_per_test,
            timeout_seconds=args.timeout,
            workers=args.workers,
        )
        _write_jsonl(output_dir / f"{split}.jsonl", [example_to_json(example) for example in examples])
        _write_jsonl(output_dir / f"{split}_audit.jsonl", audits)
        split_counts[split] = {
            "tasks_loaded": len(tasks),
            "eligible_tasks": len({example.task_id for example in examples}),
            "eligible_examples": len(examples),
            "audit_rows": len(audits),
            "exclusion_reasons": dict(Counter(row.get("reason", "unknown") for row in audits if not row.get("selected"))),
        }
        all_examples.extend(examples)
        all_audits.extend(audits)
        print(json.dumps({"event": "split_ready", "split": split, **split_counts[split]}), flush=True)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "dataset_sha256": _sha256_file(Path(args.dataset)),
        "mbppplus": str(plus_path) if plus_path.exists() else None,
        "mbppplus_sha256": _sha256_file(plus_path) if plus_path.exists() else None,
        "max_mutants": args.max_mutants,
        "max_events_per_test": args.max_events_per_test,
        "train_per_task": args.train_per_task,
        "eval_per_task": args.eval_per_task,
        "timeout": args.timeout,
        "splits": split_counts,
        "public_hidden_overlap": sum(
            bool(set(example.public_hashes) & set(example.hidden_hashes)) for example in all_examples
        ),
        "test_task_ids": [example.task_id for example in all_examples if example.split == "test"],
    }
    manifest["sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "cohort_ready", "output_dir": str(output_dir), "splits": split_counts}), flush=True)


if __name__ == "__main__":
    main()
