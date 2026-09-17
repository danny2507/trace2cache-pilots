from __future__ import annotations

import unittest

from trace2cache.structured_runtime import execute_structured_trace, tagged_value, values_equal


class StructuredRuntimeTest(unittest.TestCase):
    def test_typed_values_distinguish_bool_and_int(self) -> None:
        self.assertFalse(values_equal(tagged_value(True), tagged_value(1)))
        self.assertEqual(tagged_value([1, None, "x"])["type"], "list")

    def test_snapshots_are_captured_before_in_place_mutation(self) -> None:
        source = """def mutate(values):\n    values.append(9)\n    return values\n"""
        result = execute_structured_trace(source, "mutate", [[1]])
        self.assertEqual(result.output, tagged_value([1, 9]))
        states = [event.locals.get("values") for event in result.events if "values" in event.locals]
        self.assertIn(tagged_value([1]), states)
        self.assertIn(tagged_value([1, 9]), states)

    def test_return_captured_in_same_trace(self) -> None:
        result = execute_structured_trace("def f(x):\n    return x + 1\n", "f", [3])
        self.assertIsNone(result.error)
        self.assertEqual(result.output, tagged_value(4))
        self.assertEqual(result.events[-1].value, tagged_value(4))
