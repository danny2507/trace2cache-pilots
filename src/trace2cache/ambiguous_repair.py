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
        AmbiguousRepairCase(
            RepairCase(
                "arrange_ascending", "Behavior is supplied separately.", "arrange",
                "def arrange(values):\n    return []\n",
                TestSpec([[3]], [3]),
                (TestSpec([[3, 1, 2]], [1, 2, 3]), TestSpec([[-1, 2, 0]], [-1, 0, 2])),
            ),
            "arrange", "sort values from smallest to largest",
            """def arrange(values):
    return sorted(values)
""",
        ),
        AmbiguousRepairCase(
            RepairCase(
                "arrange_descending", "Behavior is supplied separately.", "arrange",
                "def arrange(values):\n    return []\n",
                TestSpec([[3]], [3]),
                (TestSpec([[3, 1, 2]], [3, 2, 1]), TestSpec([[-1, 2, 0]], [2, 0, -1])),
            ),
            "arrange", "sort values from largest to smallest",
            """def arrange(values):
    return sorted(values, reverse=True)
""",
        ),
        AmbiguousRepairCase(
            RepairCase(
                "choose_shortest", "Behavior is supplied separately.", "choose",
                "def choose(values):\n    return ''\n",
                TestSpec([["aa"]], "aa"),
                (TestSpec([["pear", "x", "tea"]], "x"), TestSpec([["four", "bb"]], "bb")),
            ),
            "choose", "return the shortest string",
            """def choose(values):
    return min(values, key=len)
""",
        ),
        AmbiguousRepairCase(
            RepairCase(
                "choose_longest", "Behavior is supplied separately.", "choose",
                "def choose(values):\n    return ''\n",
                TestSpec([["aa"]], "aa"),
                (TestSpec([["pear", "x", "tea"]], "pear"), TestSpec([["four", "bb"]], "four")),
            ),
            "choose", "return the longest string",
            """def choose(values):
    return max(values, key=len)
""",
        ),
        AmbiguousRepairCase(
            RepairCase(
                "truth_any_positive", "Behavior is supplied separately.", "truth",
                "def truth(values):\n    return False\n",
                TestSpec([[1]], True),
                (TestSpec([[-2, 3]], True), TestSpec([[-2, 0]], False)),
            ),
            "truth", "whether any value is positive",
            """def truth(values):
    return any(value > 0 for value in values)
""",
        ),
        AmbiguousRepairCase(
            RepairCase(
                "truth_all_positive", "Behavior is supplied separately.", "truth",
                "def truth(values):\n    return False\n",
                TestSpec([[1]], True),
                (TestSpec([[-2, 3]], False), TestSpec([[2, 1]], True)),
            ),
            "truth", "whether every value is positive",
            """def truth(values):
    return all(value > 0 for value in values)
""",
        ),
        AmbiguousRepairCase(
            RepairCase(
                "unique_first", "Behavior is supplied separately.", "unique",
                "def unique(values):\n    return []\n",
                TestSpec([[1]], [1]),
                (TestSpec([[2, 1, 2, 3, 1]], [2, 1, 3]),),
            ),
            "unique", "remove duplicates keeping first occurrences",
            """def unique(values):
    if not values:
        return []
    first = values[0]
    return [first] + unique([value for value in values[1:] if value != first])
""",
        ),
        AmbiguousRepairCase(
            RepairCase(
                "unique_last", "Behavior is supplied separately.", "unique",
                "def unique(values):\n    return []\n",
                TestSpec([[1]], [1]),
                (TestSpec([[2, 1, 2, 3, 1]], [2, 3, 1]),),
            ),
            "unique", "remove duplicates keeping last occurrences",
            """def unique(values):
    if not values:
        return []
    first = values[0]
    rest = unique(values[1:])
    if first in rest:
        return rest
    return [first] + rest
""",
        ),
        AmbiguousRepairCase(
            RepairCase(
                "median_lower", "Behavior is supplied separately.", "median",
                "def median(values):\n    return 0\n",
                TestSpec([[5]], 5),
                (TestSpec([[4, 1, 3, 2]], 2), TestSpec([[7, 1, 4]], 4)),
            ),
            "median", "return the lower middle sorted value",
            """def median(values):
    ordered = sorted(values)
    return ordered[(len(ordered) - 1) // 2]
""",
        ),
        AmbiguousRepairCase(
            RepairCase(
                "median_upper", "Behavior is supplied separately.", "median",
                "def median(values):\n    return 0\n",
                TestSpec([[5]], 5),
                (TestSpec([[4, 1, 3, 2]], 3), TestSpec([[7, 1, 4]], 4)),
            ),
            "median", "return the upper middle sorted value",
            """def median(values):
    ordered = sorted(values)
    return ordered[len(ordered) // 2]
""",
        ),
        AmbiguousRepairCase(
            RepairCase(
                "rotate_left", "Behavior is supplied separately.", "rotate",
                "def rotate(values):\n    return []\n",
                TestSpec([[1]], [1]),
                (TestSpec([[1, 2, 3]], [2, 3, 1]),),
            ),
            "rotate", "rotate the list one position left",
            """def rotate(values):
    return values[1:] + values[:1]
""",
        ),
        AmbiguousRepairCase(
            RepairCase(
                "rotate_right", "Behavior is supplied separately.", "rotate",
                "def rotate(values):\n    return []\n",
                TestSpec([[1]], [1]),
                (TestSpec([[1, 2, 3]], [3, 1, 2]),),
            ),
            "rotate", "rotate the list one position right",
            """def rotate(values):
    return values[-1:] + values[:-1]
""",
        ),
        AmbiguousRepairCase(
            RepairCase(
                "transform_absolute", "Behavior is supplied separately.", "transform",
                "def transform(values):\n    return 1\n",
                TestSpec([[0]], 0),
                (TestSpec([[-2, 3]], 5), TestSpec([[-4, -1]], 5)),
            ),
            "transform", "sum absolute values",
            """def transform(values):
    return sum(abs(value) for value in values)
""",
        ),
        AmbiguousRepairCase(
            RepairCase(
                "transform_square", "Behavior is supplied separately.", "transform",
                "def transform(values):\n    return 1\n",
                TestSpec([[0]], 0),
                (TestSpec([[-2, 3]], 13), TestSpec([[-4, -1]], 17)),
            ),
            "transform", "sum squared values",
            """def transform(values):
    return sum(value * value for value in values)
""",
        ),
        AmbiguousRepairCase(
            RepairCase(
                "filter_even", "Behavior is supplied separately.", "filter_values",
                "def filter_values(values):\n    return []\n",
                TestSpec([[2]], [2]),
                (TestSpec([[-2, -1, 3, 4]], [-2, 4]),),
            ),
            "filter", "keep only even values",
            """def filter_values(values):
    return [value for value in values if value % 2 == 0]
""",
        ),
        AmbiguousRepairCase(
            RepairCase(
                "filter_positive", "Behavior is supplied separately.", "filter_values",
                "def filter_values(values):\n    return []\n",
                TestSpec([[2]], [2]),
                (TestSpec([[-2, -1, 3, 4]], [3, 4]),),
            ),
            "filter", "keep only positive values",
            """def filter_values(values):
    return [value for value in values if value > 0]
""",
        ),
    )
