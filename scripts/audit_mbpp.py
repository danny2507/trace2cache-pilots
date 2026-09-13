#!/usr/bin/env python3
"""Measure canonical-test health and deterministic mutation yield on MBPP."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from trace2cache.mbpp import generate_mutants, load_mbpp, run_mbpp_tests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=".local/datasets/mbpp/mbpp.jsonl")
    parser.add_argument("--split", choices=("prompt", "test", "validation", "train"))
    parser.add_argument("--max-mutants", type=int, default=12)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = load_mbpp(args.dataset, split=args.split)
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        canonical = list(
            pool.map(lambda task: run_mbpp_tests(task, timeout_seconds=args.timeout), tasks)
        )
    jobs = [
        (task, mutation)
        for task in tasks
        for mutation in generate_mutants(task, limit=args.max_mutants)
    ]

    def evaluate(job):
        task, mutation = job
        result = run_mbpp_tests(task, mutation.source, timeout_seconds=args.timeout)
        return task.task_id, mutation.kind, result["status"], result["tests"]

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        mutants = list(pool.map(evaluate, jobs))
    by_task = defaultdict(list)
    for result in mutants:
        by_task[result[0]].append(result)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "split": args.split,
        "task_count": len(tasks),
        "canonical_status": dict(Counter(result["status"] for result in canonical)),
        "mutation_count": len(mutants),
        "mutation_status": dict(Counter(result[2] for result in mutants)),
        "mutation_kind_status": {
            kind: dict(Counter(result[2] for result in mutants if result[1] == kind))
            for kind in sorted({result[1] for result in mutants})
        },
        "tasks_with_mutations": len(by_task),
        "tasks_with_failing_mutant": sum(
            any(result[2] in {"mixed", "all_fail"} for result in results)
            for results in by_task.values()
        ),
        "tasks_with_mixed_mutant": sum(
            any(result[2] == "mixed" for result in results) for results in by_task.values()
        ),
        "tasks_with_two_mixed_mutants": sum(
            sum(result[2] == "mixed" for result in results) >= 2
            for results in by_task.values()
        ),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
