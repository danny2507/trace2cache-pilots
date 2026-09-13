"""Small diagnostic benchmark. It validates the harness; it is not a paper dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TestSpec:
    args: list[Any]
    expected: Any


@dataclass(frozen=True)
class RepairCase:
    case_id: str
    description: str
    function_name: str
    buggy_source: str
    public_test: TestSpec
    hidden_tests: tuple[TestSpec, ...]


CASES: tuple[RepairCase, ...] = (
    RepairCase(
        case_id="negative_max",
        description="Return the largest number in a non-empty list.",
        function_name="find_max",
        buggy_source="""def find_max(a):
    m = 0
    for x in a:
        if x > m:
            m = x
    return m
""",
        public_test=TestSpec([[-5, -2, -8, -11, -3]], -2),
        hidden_tests=(
            TestSpec([[-9]], -9),
            TestSpec([[1, 7, 2]], 7),
            TestSpec([[-10, -1, -4]], -1),
        ),
    ),
    RepairCase(
        case_id="factorial_endpoint",
        description="Return n factorial for a non-negative integer n.",
        function_name="factorial",
        buggy_source="""def factorial(n):
    result = 1
    for value in range(1, n):
        result *= value
    return result
""",
        public_test=TestSpec([6], 720),
        hidden_tests=(TestSpec([0], 1), TestSpec([1], 1), TestSpec([4], 24)),
    ),
    RepairCase(
        case_id="first_index",
        description="Return the index of the first target value, or -1 when absent.",
        function_name="first_index",
        buggy_source="""def first_index(values, target):
    for index, value in enumerate(values):
        if value == target:
            return value
    return -1
""",
        public_test=TestSpec([[8, 4, 9, 4, 3], 4], 1),
        hidden_tests=(
            TestSpec([[5, 6, 5], 5], 0),
            TestSpec([[2, 3], 9], -1),
            TestSpec([[], 1], -1),
        ),
    ),
    RepairCase(
        case_id="count_even",
        description="Count how many integers in the list are even.",
        function_name="count_even",
        buggy_source="""def count_even(values):
    count = 0
    for value in values:
        if value % 2 == 1:
            count += 1
    return count
""",
        public_test=TestSpec([[2, 7, 4, 9, 6, 11, 8]], 4),
        hidden_tests=(
            TestSpec([[1, 3, 5]], 0),
            TestSpec([[0, -2, -3]], 2),
            TestSpec([[]], 0),
        ),
    ),
    RepairCase(
        case_id="clamp_upper",
        description="Clamp value so it lies in the inclusive interval [low, high].",
        function_name="clamp",
        buggy_source="""def clamp(value, low, high):
    if value < low:
        return low
    if value > high:
        return value
    return value
""",
        public_test=TestSpec([19, 0, 10], 10),
        hidden_tests=(
            TestSpec([-2, 0, 10], 0),
            TestSpec([5, 0, 10], 5),
            TestSpec([10, 0, 10], 10),
        ),
    ),
    RepairCase(
        case_id="average_denominator",
        description="Return the arithmetic mean of a non-empty list of numbers.",
        function_name="average",
        buggy_source="""def average(values):
    total = 0
    for value in values:
        total += value
    return total / (len(values) - 1)
""",
        public_test=TestSpec([[2, 4, 6, 8, 10]], 6.0),
        hidden_tests=(
            TestSpec([[7]], 7.0),
            TestSpec([[-2, 2]], 0.0),
            TestSpec([[1.5, 2.5, 5.0]], 3.0),
        ),
    ),
)


def get_cases(limit: int | None = None) -> tuple[RepairCase, ...]:
    return CASES if limit is None else CASES[:limit]

