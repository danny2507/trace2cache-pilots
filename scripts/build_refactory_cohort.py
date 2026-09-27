#!/usr/bin/env python3
"""Freeze a Refactory student-repair cohort with disjoint public/hidden tests.

Canonical/reference source is stored as the training target only. It never belongs
in a test-split prompt. Gate 0 reads test.jsonl, which is the predeclared panel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from trace2cache.mbpp import MBPPTask, run_mbpp_tests
from trace2cache.mbpp_generalization import SCHEMA_VERSION, example_to_json
from trace2cache.refactory import (
    QUESTION_IDS,
    example_from_refactory,
    load_assert_tests,
    load_reference_source,
    load_setup_source,
    wrong_source_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="third_party/refactory")
    parser.add_argument("--max-events-per-test", type=int, default=24)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--panel-per-question", type=int, default=25)
    parser.add_argument("--output-dir", default="artifacts/mbpp_generalization/refactory_cohort_v1")
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_rev(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _assign_split(index: int, n: int) -> str:
    if n <= 0:
        raise ValueError("empty question")
    test_cut = max(1, int(n * 0.8))
    val_cut = max(1, int(n * 0.7))
    if val_cut >= test_cut:
        val_cut = max(0, test_cut - 1)
    if index >= test_cut:
        return "test"
    if index >= val_cut:
        return "validation"
    return "train"


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    preflight = {}
    for question_id in QUESTION_IDS:
        probe = MBPPTask(
            task_id=question_id,
            description=f"refactory question {question_id}",
            canonical_source=load_reference_source(root, question_id),
            tests=load_assert_tests(root, question_id),
            setup_source=load_setup_source(root, question_id),
        )
        result = run_mbpp_tests(probe, timeout_seconds=args.timeout)
        preflight[str(question_id)] = result.get("status")
        if result.get("status") != "all_pass":
            raise SystemExit(f"reference fails instructor tests for question {question_id}: {result}")
    print(json.dumps({"event": "reference_preflight", "status": preflight}), flush=True)

    jobs = []
    for question_id in QUESTION_IDS:
        for path in wrong_source_paths(root, question_id):
            jobs.append((question_id, path))

    def process(job: tuple[int, Path]) -> tuple[dict | None, dict]:
        question_id, path = job
        try:
            example, audit = example_from_refactory(
                root,
                question_id,
                path,
                split="train",
                max_events_per_test=args.max_events_per_test,
                timeout_seconds=args.timeout,
            )
        except Exception as exc:  # noqa: BLE001
            return None, {
                "question_id": question_id,
                "path": str(path),
                "selected": False,
                "reason": f"exception:{type(exc).__name__}",
                "error": str(exc)[:200],
            }
        return (example_to_json(example) if example is not None else None, audit)

    raw_examples: list[dict] = []
    audits: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for example, audit in pool.map(process, jobs):
            audits.append(audit)
            if example is not None:
                raw_examples.append(example)

    by_question: dict[int, list[dict]] = defaultdict(list)
    for example in raw_examples:
        by_question[example["task_id"] // 10000].append(example)
    examples: list[dict] = []
    for question_id in QUESTION_IDS:
        rows = sorted(by_question[question_id], key=lambda row: (row["task_id"], row["mutation_ordinal"]))
        for index, row in enumerate(rows):
            row["split"] = _assign_split(index, len(rows))
            examples.append(row)

    by_split: dict[str, list[dict]] = defaultdict(list)
    by_question_test: dict[int, list[dict]] = defaultdict(list)
    for example in examples:
        by_split[example["split"]].append(example)
        if example["split"] == "test":
            by_question_test[example["task_id"] // 10000].append(example)

    panel: list[dict] = []
    for question_id in QUESTION_IDS:
        selected = sorted(
            by_question_test[question_id],
            key=lambda row: (row["task_id"], row["mutation_ordinal"]),
        )[: args.panel_per_question]
        panel.extend(selected)

    for split in ("train", "validation", "test"):
        rows = sorted(by_split[split], key=lambda row: (row["task_id"], row["mutation_ordinal"]))
        _write_jsonl(output_dir / f"{split}.jsonl", rows)
    _write_jsonl(output_dir / "panel.jsonl", panel)
    _write_jsonl(output_dir / "test.jsonl", panel)
    _write_jsonl(output_dir / "test_full.jsonl", sorted(by_split["test"], key=lambda row: row["task_id"]))
    _write_jsonl(output_dir / "audit.jsonl", audits)

    reference_hashes = {
        question_id: _sha256_file(root / "data" / f"question_{question_id}" / "code" / "reference" / "reference.py")
        for question_id in QUESTION_IDS
    }
    test_counts = {question_id: len(load_assert_tests(root, question_id)) for question_id in QUESTION_IDS}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(root),
        "dataset_git": _git_rev(root),
        "max_events_per_test": args.max_events_per_test,
        "timeout": args.timeout,
        "panel_per_question": args.panel_per_question,
        "panel_n": len(panel),
        "panel_task_ids": [row["task_id"] for row in panel],
        "panel_questions": dict(Counter(row["task_id"] // 10000 for row in panel)),
        "reference_sha256": reference_hashes,
        "instructor_tests": test_counts,
        "splits": {
            split: {
                "eligible_examples": len(by_split[split]),
                "eligible_questions": len({row["task_id"] // 10000 for row in by_split[split]}),
            }
            for split in ("train", "validation", "test")
        },
        "eligible_examples": len(examples),
        "audit_rows": len(audits),
        "exclusion_reasons": dict(Counter(row.get("reason", "unknown") for row in audits if not row.get("selected"))),
        "reference_preflight": preflight,
    }
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest["sha256"] = digest
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output_dir / "panel.json").write_text(
        json.dumps(
            {
                "predeclared": True,
                "selection": f"first {args.panel_per_question} eligible test-split examples per question",
                "n": len(panel),
                "task_ids": [row["task_id"] for row in panel],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "event": "refactory_cohort_written",
                "output_dir": str(output_dir),
                "panel_n": manifest["panel_n"],
                "eligible_examples": manifest["eligible_examples"],
                "exclusion_reasons": manifest["exclusion_reasons"],
                "splits": manifest["splits"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
