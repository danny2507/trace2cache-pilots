#!/usr/bin/env python3
"""Build a frozen output-prediction panel from RunBugRun gold programs.

Does not modify the repair cohort. Public stdin is the query; gold stdout is
the answer. Events are intermediate runtime of the *fixed* program.

The test split is frozen once written. Train is pulled from the official
RunBugRun train split, excluding frozen test task_ids and gold sources.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from trace2cache.execution_sim import example_from_repair, example_from_stdio, example_to_json, read_examples
from trace2cache.mbpp_generalization import read_examples as read_repair_examples
from trace2cache.mbpp_generalization import source_hash
from trace2cache.runbugrun import DEFAULT_DB, TASK_ID_BASE, iter_python_bugs

FROZEN_TEST_SHA256 = "074dd6cecdd960970356907168b6f3ea5c086ae9b487ca56aecec6d0b943276c"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repair-cohort",
        default="artifacts/mbpp_generalization/runbugrun_cohort_v1",
    )
    parser.add_argument("--split", default="test", choices=("train", "test"))
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--max-events-per-test", type=int, default=24)
    parser.add_argument("--timeout", type=float, default=4.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cap", type=int, default=320, help="max train examples to keep")
    parser.add_argument("--max-candidates", type=int, default=2500)
    parser.add_argument("--max-test-chars", type=int, default=800)
    parser.add_argument(
        "--overwrite-test",
        action="store_true",
        help="Allow rewriting the frozen sim test.jsonl. Off by default.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/mbpp_generalization/execution_sim_runbugrun_v1",
    )
    return parser.parse_args()


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(example_to_json(row), sort_keys=True) + "\n" for row in rows))


def _load_manifest(output_dir: Path) -> dict:
    path = output_dir / "manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _save_manifest(output_dir: Path, manifest: dict) -> None:
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True), flush=True)


def build_test_from_repair(args, output_dir: Path) -> None:
    out_path = output_dir / "test.jsonl"
    if out_path.exists() and not args.overwrite_test:
        current = _sha256_file(out_path)
        raise SystemExit(
            f"refusing to overwrite frozen sim test.jsonl (sha {current}). "
            "Pass --overwrite-test only if you intend to replace the panel."
        )
    repair_path = Path(args.repair_cohort) / "test.jsonl"
    examples = read_repair_examples(repair_path)

    def convert(example):
        return example_from_repair(
            example,
            max_events_per_test=args.max_events_per_test,
            timeout_seconds=args.timeout,
        )

    rows = []
    missed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for sim in pool.map(convert, examples):
            if sim is None:
                missed += 1
                continue
            rows.append(sim)
    rows.sort(key=lambda item: item.task_id)
    _write_jsonl(out_path, rows)
    manifest = _load_manifest(output_dir)
    manifest.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "repair_cohort": str(args.repair_cohort),
            "repair_split_sha256": _sha256_file(repair_path),
            "split": "test",
            "n": len(rows),
            "missed": missed,
            "max_events_per_test": args.max_events_per_test,
            "mean_events": (sum(row.event_count for row in rows) / len(rows)) if rows else 0,
            "panel_sha256": _sha256_file(out_path),
        }
    )
    _save_manifest(output_dir, manifest)


def build_train_from_db(args, output_dir: Path) -> None:
    test_path = output_dir / "test.jsonl"
    if not test_path.exists():
        raise SystemExit(f"frozen sim test.jsonl missing: {test_path}")
    test_sha = _sha256_file(test_path)
    if test_sha != FROZEN_TEST_SHA256:
        raise SystemExit(f"frozen sim test sha {test_sha} != {FROZEN_TEST_SHA256}")
    test_examples = read_examples(test_path)
    excluded_task_ids = {example.task_id for example in test_examples}
    excluded_hashes = {example.source_hash for example in test_examples}
    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"RunBugRun sqlite database missing: {db_path}")
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    bugs = []
    for bug in iter_python_bugs(conn, min_tests=1, splits=("train",)):
        bugs.append(bug)
        if args.max_candidates and len(bugs) >= args.max_candidates:
            break
    conn.close()
    print(json.dumps({"event": "train_bugs_loaded", "n": len(bugs)}), flush=True)

    def convert(bug: dict):
        task_id = TASK_ID_BASE + int(bug["bug_id"])
        if task_id in excluded_task_ids:
            return None, "test_task_id"
        fixed = str(bug["fixed_code"] or "")
        hashed = source_hash(fixed)
        if hashed in excluded_hashes:
            return None, "test_source_hash"
        tests = list(bug.get("tests") or [])
        if not tests:
            return None, "no_tests"
        stdin, gold = tests[0]
        if len(stdin) > args.max_test_chars or len(gold) > args.max_test_chars:
            return None, "io_too_long"
        try:
            sim = example_from_stdio(
                fixed,
                stdin,
                gold,
                task_id=task_id,
                example_id=f"sim_train_{task_id}",
                source_hash=hashed,
                max_events_per_test=args.max_events_per_test,
                timeout_seconds=args.timeout,
            )
        except Exception as exc:  # noqa: BLE001
            return None, f"exception:{type(exc).__name__}"
        if sim is None:
            return None, "trace_failed"
        return sim, "ok"

    rows = []
    reasons: dict[str, int] = {}
    batch = max(args.workers * 2, 16)
    index = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while index < len(bugs) and len(rows) < args.cap:
            chunk = bugs[index : index + batch]
            index += len(chunk)
            for sim, reason in pool.map(convert, chunk):
                reasons[reason] = reasons.get(reason, 0) + 1
                if sim is None:
                    continue
                rows.append(sim)
                if len(rows) >= args.cap:
                    break
            print(
                json.dumps(
                    {
                        "event": "train_progress",
                        "scanned": min(index, len(bugs)),
                        "kept": len(rows),
                        "reasons": reasons,
                    }
                ),
                flush=True,
            )
    rows.sort(key=lambda item: item.task_id)
    overlap = {row.task_id for row in rows} & excluded_task_ids
    if overlap:
        raise SystemExit(f"train/test task_id overlap: {sorted(overlap)[:8]}")
    hash_overlap = {row.source_hash for row in rows} & excluded_hashes
    if hash_overlap:
        raise SystemExit(f"train/test source_hash overlap: {len(hash_overlap)}")
    out_path = output_dir / "train.jsonl"
    _write_jsonl(out_path, rows)
    if test_sha != _sha256_file(test_path):
        raise SystemExit("test.jsonl changed while writing train; aborting")
    manifest = _load_manifest(output_dir)
    manifest.update(
        {
            "train_created_at": datetime.now(timezone.utc).isoformat(),
            "train_n": len(rows),
            "train_sha256": _sha256_file(out_path),
            "train_mean_events": (sum(row.event_count for row in rows) / len(rows)) if rows else 0,
            "train_reasons": reasons,
            "train_scanned": min(index, len(bugs)),
            "train_cap": args.cap,
            "panel_sha256": test_sha,
            "max_events_per_test": args.max_events_per_test,
        }
    )
    _save_manifest(output_dir, manifest)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.split == "test":
        build_test_from_repair(args, output_dir)
        return
    build_train_from_db(args, output_dir)


if __name__ == "__main__":
    main()
