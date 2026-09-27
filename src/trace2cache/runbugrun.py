"""Adapter for RunBugRun Python contest bugs (stdin/stdout tests).

Haque et al. (arXiv:2505.04441) sampled 1k Python bugs with at least five
tests. Public evidence is the first failing I/O pair; hidden tests are the
rest. Fixed code is `target_source` only and never belongs in a test prompt.
"""

from __future__ import annotations

import ast
import sqlite3
from dataclasses import replace
from pathlib import Path

from .mbpp import MBPPMutation, MBPPTask, run_mbpp_tests
from .mbpp_generalization import (
    RepairExample,
    build_events,
    encode_stdio_test,
    hash_key,
    literal_calls,
    partition_tests,
    source_hash,
)

LANGUAGE_PYTHON = 5
SPLIT_NAMES = {0: "train", 1: "validation", 2: "test"}
TASK_ID_BASE = 2_000_000
DEFAULT_DB = Path("third_party/run_bug_run/runbugrun.db")
DEFAULT_SQL = Path("third_party/run_bug_run/runbugrun.sql")


def connect(path: str | Path = DEFAULT_DB) -> sqlite3.Connection:
    db = Path(path)
    if not db.exists():
        raise FileNotFoundError(f"RunBugRun sqlite database missing: {db}")
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    return conn


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def load_problem_tests(conn: sqlite3.Connection, problem_id) -> tuple[tuple[str, str], ...]:
    columns = _columns(conn, "tests")
    clauses = ["problem_id = ?"]
    if "active" in columns:
        clauses.append("active = 1")
    order = "test_id" if "test_id" in columns else "id"
    rows = conn.execute(
        f"SELECT input, output FROM tests WHERE {' AND '.join(clauses)} ORDER BY {order}",
        (problem_id,),
    ).fetchall()
    return tuple((str(row["input"]), str(row["output"])) for row in rows)


def load_problem_text(conn: sqlite3.Connection, problem_id) -> str:
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "problems" not in tables:
        return ""
    columns = _columns(conn, "problems")
    text_col = next((name for name in ("text", "description", "title") if name in columns), None)
    if text_col is None:
        return ""
    row = conn.execute(
        f"SELECT {text_col} FROM problems WHERE problem_id = ? LIMIT 1",
        (problem_id,),
    ).fetchone()
    return "" if row is None or row[0] is None else str(row[0]).strip()


def iter_python_bugs(
    conn: sqlite3.Connection,
    *,
    min_tests: int = 5,
    splits: tuple[str, ...] | None = None,
    max_source_chars: int = 3000,
):
    columns = _columns(conn, "bugs")
    where = ["language = ?"]
    params: list[object] = [LANGUAGE_PYTHON]
    if "active" in columns:
        where.append("active = 1")
    if splits:
        wanted = [key for key, name in SPLIT_NAMES.items() if name in splits]
        if not wanted:
            return
        where.append(f"split IN ({','.join('?' for _ in wanted)})")
        params.extend(wanted)
    split_expr = "split" if "split" in columns else "NULL"
    query = (
        "SELECT id, problem_id, buggy_code, fixed_code, "
        f"{split_expr} AS split FROM bugs WHERE {' AND '.join(where)} ORDER BY id"
    )
    tests_cache: dict = {}
    text_cache: dict = {}
    for row in conn.execute(query, params):
        buggy = row["buggy_code"] or ""
        fixed = row["fixed_code"] or ""
        if len(buggy) > max_source_chars or len(fixed) > max_source_chars:
            continue
        problem_id = row["problem_id"]
        if problem_id not in tests_cache:
            tests_cache[problem_id] = load_problem_tests(conn, problem_id)
        tests = tests_cache[problem_id]
        if len(tests) < min_tests:
            continue
        if problem_id not in text_cache:
            text_cache[problem_id] = load_problem_text(conn, problem_id)
        yield {
            "bug_id": int(row["id"]),
            "problem_id": problem_id,
            "buggy_code": buggy,
            "fixed_code": fixed,
            "split": SPLIT_NAMES.get(row["split"], "train"),
            "tests": tests,
            "description": text_cache[problem_id],
        }


def example_from_runbugrun(
    bug: dict,
    *,
    split: str | None = None,
    max_events_per_test: int = 24,
    timeout_seconds: float = 4.0,
    max_source_chars: int = 3000,
    max_test_chars: int = 800,
) -> tuple[RepairExample | None, dict]:
    bug_id = int(bug["bug_id"])
    task_id = TASK_ID_BASE + bug_id
    assigned_split = split or str(bug.get("split") or "train")
    buggy = str(bug["buggy_code"])
    fixed = str(bug["fixed_code"])
    pairs = [
        (stdin, stdout)
        for stdin, stdout in bug["tests"]
        if len(stdin) <= max_test_chars and len(stdout) <= max_test_chars
    ]
    audit = {
        "task_id": task_id,
        "bug_id": bug_id,
        "problem_id": bug.get("problem_id"),
        "split": assigned_split,
        "mutation_kind": "student",
        "mutation_ordinal": bug_id,
        "source_hash": source_hash(buggy),
        "selected": False,
        "reason": "unclassified",
    }
    if len(pairs) < 2:
        audit["reason"] = "too_few_short_tests"
        return None, audit
    if len(buggy) > max_source_chars or len(fixed) > max_source_chars:
        audit["reason"] = "source_too_long"
        return None, audit
    try:
        ast.parse(buggy)
        ast.parse(fixed)
    except SyntaxError:
        audit["reason"] = "buggy_syntax"
        return None, audit
    tests = tuple(encode_stdio_test(stdin, stdout) for stdin, stdout in pairs)
    task = MBPPTask(
        task_id=task_id,
        description=str(bug.get("description") or f"RunBugRun problem {bug.get('problem_id')}"),
        canonical_source=fixed,
        tests=tests,
    )
    mutation = MBPPMutation("student", bug_id, buggy)
    canonical = run_mbpp_tests(task, timeout_seconds=timeout_seconds)
    audit["canonical_status"] = canonical.get("status")
    if canonical.get("status") != "all_pass":
        audit["reason"] = f"canonical_{canonical.get('status')}"
        return None, audit
    mutant = run_mbpp_tests(task, mutation.source, timeout_seconds=timeout_seconds)
    audit["mutant_status"] = mutant.get("status")
    partition = partition_tests(
        task.tests,
        canonical.get("tests") or [],
        mutant.get("tests") or [],
        source=task.canonical_source,
    )
    if partition is None:
        audit["reason"] = "no_disjoint_hidden_failure"
        return None, audit
    events = build_events(
        task, mutation, partition.public_tests, max_events_per_test=max_events_per_test
    )
    if not events:
        audit["reason"] = "public_trace_failed"
        return None, audit
    public_result = run_mbpp_tests(
        replace(task, tests=partition.public_tests), mutation.source, timeout_seconds=timeout_seconds
    )
    hidden_result = run_mbpp_tests(
        replace(task, tests=partition.hidden_tests), mutation.source, timeout_seconds=timeout_seconds
    )
    example = RepairExample(
        example_id=f"runbugrun_{bug_id}",
        task_id=task_id,
        split=assigned_split,
        mutation_kind="student",
        mutation_ordinal=bug_id,
        description=task.description,
        buggy_source=buggy,
        target_source=fixed,
        setup_source="",
        failing_public_test=partition.public_tests[0],
        public_tests=partition.public_tests,
        hidden_tests=partition.hidden_tests,
        public_hashes=partition.public_hashes,
        hidden_hashes=tuple(hash_key(key) for key in literal_calls(task.canonical_source, partition.hidden_tests)),
        events=events,
        source_hash=source_hash(buggy),
        mutant_public_status=public_result.get("status", "unknown"),
        mutant_hidden_status=hidden_result.get("status", "unknown"),
    )
    audit.update(
        {
            "selected": True,
            "reason": "ok",
            "public_tests": list(example.public_tests),
            "hidden_tests": list(example.hidden_tests),
            "event_count": len(example.events),
            "mutant_public_status": example.mutant_public_status,
            "mutant_hidden_status": example.mutant_hidden_status,
        }
    )
    return example, audit
