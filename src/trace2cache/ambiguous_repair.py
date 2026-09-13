"""Paired repair tasks whose public context is intentionally underdetermined."""

from __future__ import annotations

from dataclasses import dataclass

from .benchmark import RepairCase, TestSpec


@dataclass(frozen=True)
class AmbiguousRepairCase:
    case: RepairCase
    pair_id: str
    diagnosis: str
    correct_source: str


def get_ambiguous_cases() -> tuple[AmbiguousRepairCase, ...]:
    locate_bug = """def locate(values, target):
    for index, value in enumerate(values):
        if value == target:
            return value
    return -1
"""
    aggregate_bug = """def aggregate(values):
    return len(values)
"""
    extreme_bug = """def extreme(values):
    return 0
"""
    measure_bug = """def measure(values):
    return len(values)
"""
    return (
        AmbiguousRepairCase(
            RepairCase(
                "locate_first", "Behavior is supplied separately.", "locate", locate_bug,
                TestSpec([[8, 4, 9], 4], 1),
                (
                    TestSpec([[4, 1, 4], 4], 0),
                    TestSpec([[1, 4, 4], 4], 1),
                    TestSpec([[2], 9], -1),
                ),
            ),
            "locate", "return the first matching index",
            """def locate(values, target):
    for index, value in enumerate(values):
        if value == target:
            return index
    return -1
""",
        ),
        AmbiguousRepairCase(
            RepairCase(
                "locate_last", "Behavior is supplied separately.", "locate", locate_bug,
                TestSpec([[8, 4, 9], 4], 1),
                (TestSpec([[4, 1, 4], 4], 2), TestSpec([[1, 4, 4], 4], 2), TestSpec([[2], 9], -1)),
            ),
            "locate", "return the last matching index",
            """def locate(values, target):
    result = -1
    for index, value in enumerate(values):
        if value == target:
            result = index
    return result
""",
        ),
        AmbiguousRepairCase(
            RepairCase(
                "aggregate_sum", "Behavior is supplied separately.", "aggregate", aggregate_bug,
                TestSpec([[2, 2]], 4),
                (TestSpec([[2, 3]], 5), TestSpec([[-1, 3]], 2), TestSpec([[5]], 5)),
            ),
            "aggregate", "sum all values",
            """def aggregate(values):
    total = 0
    for value in values:
        total += value
    return total
""",
        ),
        AmbiguousRepairCase(
            RepairCase(
                "aggregate_product", "Behavior is supplied separately.", "aggregate", aggregate_bug,
                TestSpec([[2, 2]], 4),
                (TestSpec([[2, 3]], 6), TestSpec([[-1, 3]], -3), TestSpec([[5]], 5)),
            ),
            "aggregate", "multiply all values",
            """def aggregate(values):
    total = 1
    for value in values:
        total *= value
    return total
""",
        ),
        AmbiguousRepairCase(
            RepairCase(
                "extreme_max", "Behavior is supplied separately.", "extreme", extreme_bug,
                TestSpec([[-2, -2]], -2),
                (TestSpec([[3, 1, 2]], 3), TestSpec([[-4, -1]], -1), TestSpec([[7]], 7)),
            ),
            "extreme", "return the largest value",
            """def extreme(values):
    result = values[0]
    for value in values[1:]:
        if value > result:
            result = value
    return result
""",
        ),
        AmbiguousRepairCase(
            RepairCase(
                "extreme_min", "Behavior is supplied separately.", "extreme", extreme_bug,
                TestSpec([[-2, -2]], -2),
                (TestSpec([[3, 1, 2]], 1), TestSpec([[-4, -1]], -4), TestSpec([[7]], 7)),
            ),
            "extreme", "return the smallest value",
            """def extreme(values):
    result = values[0]
    for value in values[1:]:
        if value < result:
            result = value
    return result
""",
        ),
        AmbiguousRepairCase(
            RepairCase(
                "measure_even", "Behavior is supplied separately.", "measure", measure_bug,
                TestSpec([[2, -1, 4]], 2),
                (TestSpec([[1, 3]], 0), TestSpec([[0, -2, -3]], 2), TestSpec([[]], 0)),
            ),
            "measure", "count the even values",
            """def measure(values):
    count = 0
    for value in values:
        if value % 2 == 0:
            count += 1
    return count
""",
        ),
        AmbiguousRepairCase(
            RepairCase(
                "measure_positive", "Behavior is supplied separately.", "measure", measure_bug,
                TestSpec([[2, -1, 4]], 2),
                (TestSpec([[1, 3]], 2), TestSpec([[0, -2, -3]], 0), TestSpec([[]], 0)),
            ),
            "measure", "count the positive values",
            """def measure(values):
    count = 0
    for value in values:
        if value > 0:
            count += 1
    return count
""",
        ),
    )
