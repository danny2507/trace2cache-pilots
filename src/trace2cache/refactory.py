"""Adapter for the published Refactory student-program benchmark."""

from __future__ import annotations

import ast
from pathlib import Path

from .benchmark import RepairCase, TestSpec
from .runtime import execute_with_trace


QUESTION_1_SPEC = (
    "Given a value x and a sorted sequence seq, return the first index i where "
    "x <= seq[i]. Return len(seq) if no such index exists. Empty sequences return 0."
)


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

