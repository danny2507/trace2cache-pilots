from __future__ import annotations

from pathlib import Path
import unittest

from trace2cache.mbpp import MBPPTask, run_mbpp_tests
from trace2cache.mbpp_generalization import evidence_for_condition, gist_text, render_user_prompt
from trace2cache.refactory import (
    QUESTION_IDS,
    example_from_refactory,
    load_assert_tests,
    load_refactory_q1,
    load_reference_source,
    wrong_source_paths,
)


class RefactoryTests(unittest.TestCase):
    def test_loads_real_failing_submission(self):
        root = Path("third_party/refactory")
        if not root.exists():
            self.skipTest("optional Refactory checkout is unavailable")
        case = load_refactory_q1(root, limit=1)[0]
        self.assertEqual(case.case_id, "refactory_q1_wrong_1_001")
        self.assertEqual(case.function_name, "search")
        self.assertEqual(len(case.hidden_tests), 10)

    def test_all_questions_have_reference_and_asserts(self):
        root = Path("third_party/refactory")
        if not root.exists():
            self.skipTest("optional Refactory checkout is unavailable")
        for question_id in QUESTION_IDS:
            reference = load_reference_source(root, question_id)
            tests = load_assert_tests(root, question_id)
            self.assertTrue(reference.strip())
            self.assertGreaterEqual(len(tests), 2)
            self.assertTrue(all(test.startswith("assert ") for test in tests))
            self.assertNotIn("REPLACE", reference)

    def test_example_from_refactory_hides_canonical_and_keeps_hidden_tests(self):
        root = Path("third_party/refactory")
        if not root.exists():
            self.skipTest("optional Refactory checkout is unavailable")
        path = wrong_source_paths(root, 1)[0]
        example, audit = example_from_refactory(root, 1, path, split="test")
        self.assertTrue(audit["selected"], audit)
        self.assertIsNotNone(example)
        assert example is not None
        self.assertTrue(example.hidden_tests)
        self.assertTrue(example.events)
        self.assertTrue(example.failing_public_test.startswith("assert "))
        self.assertEqual(example.mutation_kind, "student")
        prompt = render_user_prompt(example, evidence_for_condition(example, "gist_text"))
        self.assertIn(example.buggy_source.strip(), prompt)
        self.assertIn(example.failing_public_test, prompt)
        self.assertNotIn(example.target_source.strip(), prompt)
        gist = evidence_for_condition(example, "gist_text")
        self.assertEqual(gist, gist_text(example.events))
        self.assertIn("TEST:", gist)
        self.assertLess(len(gist), 2000)

    def test_worker_ignores_student_prints(self):
        task = MBPPTask(
            task_id=11,
            description="printer",
            canonical_source="def f():\n    print('noise')\n    return 1\n",
            tests=("assert f() == 1",),
        )
        result = run_mbpp_tests(task)
        self.assertEqual(result.get("status"), "all_pass")
        self.assertEqual(result.get("tests"), ["pass"])
        buggy = MBPPTask(
            task_id=11,
            description="printer",
            canonical_source="def f():\n    print(1)\n    return 0\n",
            tests=("assert f() == 1",),
        )
        failed = run_mbpp_tests(buggy)
        self.assertIn(failed.get("status"), {"all_fail", "mixed"})
        self.assertNotEqual(failed.get("status"), "worker_error")


if __name__ == "__main__":
    unittest.main()
