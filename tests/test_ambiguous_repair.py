import unittest

from trace2cache.ambiguous_repair import get_ambiguous_cases
from trace2cache.runtime import execute_with_trace
from trace2cache.sandbox import evaluate_patch


class AmbiguousRepairTests(unittest.TestCase):
    def test_bug_fails_public_and_reference_passes_all(self):
        for item in get_ambiguous_cases():
            trace = execute_with_trace(
                item.case.buggy_source, item.case.function_name, item.case.public_test.args
            )
            self.assertNotEqual(trace.output, repr(item.case.public_test.expected), item.case.case_id)
            self.assertTrue(evaluate_patch(item.correct_source, item.case)["passed"], item.case.case_id)

    def test_pair_has_identical_observed_context(self):
        cases = get_ambiguous_cases()
        for left, right in zip(cases[::2], cases[1::2]):
            self.assertEqual(left.pair_id, right.pair_id)
            self.assertEqual(left.case.buggy_source, right.case.buggy_source)
            self.assertEqual(left.case.public_test, right.case.public_test)
            self.assertNotEqual(left.correct_source, right.correct_source)


if __name__ == "__main__":
    unittest.main()
