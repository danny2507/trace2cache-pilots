from __future__ import annotations

import unittest

from trace2cache.benchmark import CASES
from trace2cache.sandbox import evaluate_patch, extract_function, validate_function


class SandboxTests(unittest.TestCase):
    def test_extract_and_validate(self):
        response = "```python\ndef find_max(a):\n    return max(a)\n```"
        source = extract_function(response, "find_max")
        validate_function(source, "find_max")

    def test_reject_import(self):
        with self.assertRaises(ValueError):
            validate_function("import os\ndef find_max(a): return 0\n", "find_max")

    def test_allows_defensive_value_error(self):
        source = "def find_max(a):\n    if not a:\n        raise ValueError('empty')\n    return max(a)\n"
        validate_function(source, "find_max")

    def test_known_patch_passes_hidden_tests(self):
        result = evaluate_patch("def find_max(a):\n    return max(a)\n", CASES[0])
        self.assertTrue(result["passed"], result)


if __name__ == "__main__":
    unittest.main()
