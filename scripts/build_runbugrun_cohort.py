#!/usr/bin/env python3
"""Freeze a RunBugRun Python cohort with disjoint public/hidden stdin tests.

Fixed programs are stored as training targets only. Gate 0 reads test.jsonl,
the predeclared panel of eligible official-test-split examples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from trace2cache.mbpp_generalization import SCHEMA_VERSION, example_to_json
from trace2cache.runbugrun import (
    DEFAULT_DB,
    DEFAULT_SQL,
    example_from_runbugrun,
    iter_python_bugs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--sql", default=str(DEFAULT_SQL))
    parser.add_argument("--min-tests", type=int, default=5)
    parser.add_argument("--max-events-per-test", type=int, default=24)
    parser.add_argument("--timeout", type=float, default=4.0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--panel-n", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0, help="0 = every loaded candidate")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test"],
        choices=("train", "validation", "test"),
        help="official RunBugRun splits to scan (Gate 0 uses test)",
    )
    parser.add_argument(
        "--max-candidates-per-split",
        type=int,
        default=1500,
        help="cap executed candidates per split; 0 = no cap",
    )
    parser.add_argument("--output-dir", default="artifacts/mbpp_generalization/runbugrun_cohort_v1")
    return parser.parse_args()


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def ensure_database(db_path: Path, sql_path: Path) -> None:
    if db_path.exists() and db_path.stat().st_size > 10_000:
        return
    if not sql_path.exists():
        raise SystemExit(f"RunBugRun dump missing: {sql_path}")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = db_path.with_suffix(".importing.db")
    if tmp.exists():
        tmp.unlink()
    print(json.dumps({"event": "import_sql", "sql": str(sql_path), "db": str(db_path)}), flush=True)
    with sql_path.open("rb") as handle:
        subprocess.run(["sqlite3", str(tmp)], stdin=handle, check=True)
    tmp.replace(db_path)


def main() -> None:
    args = parse_args()
    db_path = Path(args.db)
    sql_path = Path(args.sql)
    ensure_database(db_path, sql_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    bugs = []
    per_split: dict[str, int] = defaultdict(int)
    for bug in iter_python_bugs(conn, min_tests=args.min_tests, splits=tuple(args.splits)):
        split = bug["split"]
        if args.max_candidates_per_split and per_split[split] >= args.max_candidates_per_split:
            if all(per_split[name] >= args.max_candidates_per_split for name in args.splits):
                break
            continue
        per_split[split] += 1
        bugs.append(bug)
        if args.limit and len(bugs) >= args.limit:
            break
    conn.close()
    print(
        json.dumps({"event": "python_bugs_loaded", "n": len(bugs), "per_split": dict(per_split)}),
        flush=True,
    )

    def process(bug: dict) -> tuple[dict | None, dict]:
        try:
            example, audit = example_from_runbugrun(
                bug,
                split=bug.get("split"),
                max_events_per_test=args.max_events_per_test,
                timeout_seconds=args.timeout,
            )
        except Exception as exc:  # noqa: BLE001
            return None, {
                "bug_id": bug.get("bug_id"),
                "selected": False,
                "reason": f"exception:{type(exc).__name__}",
                "error": str(exc)[:200],
            }
        return (example_to_json(example) if example is not None else None, audit)

    examples: list[dict] = []
    audits: list[dict] = []
    batch = max(args.workers * 2, 32)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        index = 0
        while index < len(bugs):
            chunk = bugs[index : index + batch]
            index += len(chunk)
            for example, audit in pool.map(process, chunk):
                audits.append(audit)
                if example is not None:
                    examples.append(example)
            print(
                json.dumps(
                    {
                        "event": "cohort_progress",
                        "scanned": min(index, len(bugs)),
                        "eligible": len(examples),
                    }
                ),
                flush=True,
            )
            if sum(1 for row in examples if row["split"] == "test") >= args.panel_n:
                break

    by_split: dict[str, list[dict]] = defaultdict(list)
    for example in examples:
        by_split[example["split"]].append(example)

    test_rows = sorted(by_split.get("test", []), key=lambda row: row["task_id"])
    panel = test_rows[: args.panel_n]

    for split in ("train", "validation", "test"):
        rows = sorted(by_split.get(split, []), key=lambda row: row["task_id"])
        _write_jsonl(output_dir / f"{split}.jsonl", rows)
    _write_jsonl(output_dir / "panel.jsonl", panel)
    _write_jsonl(output_dir / "test.jsonl", panel)
    _write_jsonl(output_dir / "test_full.jsonl", test_rows)
    _write_jsonl(output_dir / "audit.jsonl", audits)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "RunBugRun v2 Python",
        "db": str(db_path),
        "sql_sha256": _sha256_file(sql_path),
        "db_sha256": _sha256_file(db_path),
        "min_tests": args.min_tests,
        "max_events_per_test": args.max_events_per_test,
        "timeout": args.timeout,
        "panel_n": len(panel),
        "panel_task_ids": [row["task_id"] for row in panel],
        "splits": {
            split: {
                "eligible_examples": len(by_split.get(split, [])),
                "eligible_problems": len({row.get("description") for row in by_split.get(split, [])}),
            }
            for split in ("train", "validation", "test")
        },
        "eligible_examples": len(examples),
        "audit_rows": len(audits),
        "exclusion_reasons": dict(Counter(row.get("reason", "unknown") for row in audits if not row.get("selected"))),
        "candidate_python_bugs": len(bugs),
    }
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest["sha256"] = digest
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output_dir / "panel.json").write_text(
        json.dumps(
            {
                "predeclared": True,
                "selection": "first eligible official test-split Python bugs by id",
                "n": len(panel),
                "task_ids": [row["task_id"] for row in panel],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (output_dir / "README.md").write_text(
        "# RunBugRun Python cohort v1\n\n"
        "Official RunBugRun v2 Python bugs with at least five stdin/stdout tests. "
        "Public evidence is the first failing I/O pair; hidden tests are the rest. "
        "Fixed submissions are `target_source` only.\n"
    )
    print(
        json.dumps(
            {
                "event": "runbugrun_cohort_written",
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
