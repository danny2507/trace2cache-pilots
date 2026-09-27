"""Adapter for the published Refactory student-program benchmark."""

from __future__ import annotations

import ast
import re
from dataclasses import replace
from pathlib import Path

from .benchmark import RepairCase, TestSpec
from .mbpp import MBPPMutation, MBPPTask, run_mbpp_tests
from .mbpp_generalization import (
    RepairExample,
    build_events,
    hash_key,
    literal_calls,
    partition_tests,
    source_hash,
)
from .runtime import execute_with_trace


QUESTION_1_SPEC = (
    "Given a value x and a sorted sequence seq, return the first index i where "
    "x <= seq[i]. Return len(seq) if no such index exists. Empty sequences return 0."
)
QUESTION_IDS = (1, 2, 3, 4, 5)
_WRONG_NAME = re.compile(r"wrong_(\d+)_(\d+)\.py$")


def _literal_output(rendered: str | None):
    if rendered is None:
        return None
    try:
        return ast.literal_eval(rendered)
    except (SyntaxError, ValueError):
        return rendered


def _load_tests(question: Path) -> list[TestSpec]:
    tests = []
    for input_path in sorted((question / "ans").glob("input_*.txt")):
        test_number = input_path.stem.split("_")[-1]
        call = ast.parse(input_path.read_text().strip(), mode="eval").body
        if not isinstance(call, ast.Call):
            raise ValueError(f"not a call expression: {input_path}")
        args = [ast.literal_eval(arg) for arg in call.args]
        output_path = question / "ans" / f"output_{test_number}.txt"
        expected = ast.literal_eval(output_path.read_text().strip())
        tests.append(TestSpec(args=args, expected=expected))
    return tests


def load_refactory_q1(root: str | Path, limit: int | None = None) -> tuple[RepairCase, ...]:
    """Load deterministic failing submissions from Refactory question 1.

    We select in filename order and expose the first failing test. All other instructor tests are
    held out. The source programs are dataset artifacts and are traced as trusted benchmark code.
    """
    question = Path(root) / "data" / "question_1"
    if not question.exists():
        raise FileNotFoundError(f"Refactory question_1 was not found under {root}")
    tests = _load_tests(question)
    cases = []
    for source_path in sorted((question / "code" / "wrong").glob("wrong_1_*.py")):
        source = source_path.read_text()
        failing_index = None
        for index, test in enumerate(tests):
            result = execute_with_trace(source, "search", test.args, max_events=2_000)
            actual = _literal_output(result.output)
            if result.error is not None or actual != test.expected:
                failing_index = index
                break
        if failing_index is None:
            continue
        public_test = tests[failing_index]
        hidden = tuple(test for index, test in enumerate(tests) if index != failing_index)
        cases.append(
            RepairCase(
                case_id=f"refactory_q1_{source_path.stem}",
                description=QUESTION_1_SPEC,
                function_name="search",
                buggy_source=source,
                public_test=public_test,
                hidden_tests=hidden,
            )
        )
        if limit is not None and len(cases) >= limit:
            break
    return tuple(cases)


def question_dir(root: str | Path, question_id: int) -> Path:
    return Path(root) / "data" / f"question_{question_id}"


def load_question_description(root: str | Path, question_id: int) -> str:
    if question_id == 1:
        return QUESTION_1_SPEC
    path = question_dir(root, question_id) / "description.txt"
    return path.read_text().strip()


def load_setup_source(root: str | Path, question_id: int) -> str:
    path = question_dir(root, question_id) / "code" / "global.py"
    return path.read_text() if path.exists() else ""


def load_reference_source(root: str | Path, question_id: int) -> str:
    path = question_dir(root, question_id) / "code" / "reference" / "reference.py"
    if not path.exists():
        raise FileNotFoundError(f"missing reference for question {question_id}")
    return path.read_text()


def load_assert_tests(root: str | Path, question_id: int) -> tuple[str, ...]:
    question = question_dir(root, question_id)
    tests = []
    for input_path in sorted((question / "ans").glob("input_*.txt")):
        output_path = question / "ans" / f"output_{input_path.stem.split('_', 1)[-1]}.txt"
        call = input_path.read_text().strip()
        expected = output_path.read_text().strip()
        tests.append(f"assert {call} == {expected}")
    if not tests:
        raise FileNotFoundError(f"no instructor tests for question {question_id}")
    return tuple(tests)


def wrong_source_paths(root: str | Path, question_id: int) -> tuple[Path, ...]:
    folder = question_dir(root, question_id) / "code" / "wrong"
    return tuple(sorted(folder.glob(f"wrong_{question_id}_*.py")))


def submission_ordinal(path: Path) -> int:
    match = _WRONG_NAME.match(path.name)
    if match is None:
        raise ValueError(f"unexpected Refactory filename: {path.name}")
    return int(match.group(2))


def example_from_refactory(
    root: str | Path,
    question_id: int,
    source_path: Path,
    *,
    split: str,
    max_events_per_test: int = 24,
    timeout_seconds: float = 2.0,
) -> tuple[RepairExample | None, dict]:
    """Wrap one student submission as a RepairExample. Canonical is reference-only."""
    ordinal = submission_ordinal(source_path)
    task_id = question_id * 10000 + ordinal
    buggy = source_path.read_text()
    reference = load_reference_source(root, question_id)
    tests = load_assert_tests(root, question_id)
    task = MBPPTask(
        task_id=task_id,
        description=load_question_description(root, question_id),
        canonical_source=reference,
        tests=tests,
        setup_source=load_setup_source(root, question_id),
    )
    mutation = MBPPMutation("student", ordinal, buggy)
    audit = {
        "task_id": task_id,
        "question_id": question_id,
        "split": split,
        "mutation_kind": "student",
        "mutation_ordinal": ordinal,
        "source_hash": source_hash(buggy),
        "path": str(source_path),
        "selected": False,
        "reason": "unclassified",
    }
    try:
        ast.parse(buggy)
    except SyntaxError:
        audit["reason"] = "buggy_syntax"
        return None, audit
    canonical = run_mbpp_tests(task, timeout_seconds=timeout_seconds)
    audit["canonical_status"] = canonical.get("status")
    if canonical.get("status") != "all_pass":
        audit["reason"] = f"canonical_{canonical.get('status')}"
        return None, audit
    mutant = run_mbpp_tests(task, mutation.source, timeout_seconds=timeout_seconds)
    audit["mutant_status"] = mutant.get("status")
    partition = partition_tests(
        task.tests, canonical.get("tests") or [], mutant.get("tests") or [], source=task.canonical_source
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
        example_id=f"refactory_q{question_id}_{source_path.stem}",
        task_id=task_id,
        split=split,
        mutation_kind="student",
        mutation_ordinal=ordinal,
        description=task.description,
        buggy_source=buggy,
        target_source=reference,
        setup_source=task.setup_source,
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

