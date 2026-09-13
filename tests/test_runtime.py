from __future__ import annotations

import unittest

from trace2cache.runtime import execute_with_trace, to_json_trace, to_text_trace


class RuntimeTests(unittest.TestCase):
    def test_trace_captures_output_and_loop_state(self):
        source = """def total(values):
    result = 0
    for value in values:
        result += value
    return result
"""
        result = execute_with_trace(source, "total", [[1, 2, 3]])
        self.assertEqual(result.output, "6")
        self.assertIsNone(result.error)
        self.assertGreater(len(result.events), 4)
        self.assertIn("result='6'", to_text_trace(result).replace('result=6', "result='6'"))

    def test_serializations_are_deterministic(self):
        source = "def identity(x):\n    return x\n"
        result = execute_with_trace(source, "identity", [3])
        self.assertEqual(to_json_trace(result), to_json_trace(result))
        self.assertLessEqual(len(to_text_trace(result, compact=True)), len(to_text_trace(result)))


if __name__ == "__main__":
    unittest.main()

