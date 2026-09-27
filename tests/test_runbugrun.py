from __future__ import annotations

import unittest

from trace2cache.mbpp import MBPPTask, run_mbpp_tests, trace_mbpp_tests
from trace2cache.mbpp_generalization import (
    allowed_calls_for_tests,
    encode_stdio_test,
    evidence_for_condition,
    extract_patch,
    format_public_test,
    line_text,
    parse_stdio_test,
    render_user_prompt,
)
from trace2cache.runbugrun import example_from_runbugrun


class RunBugRunTests(unittest.TestCase):
    def test_stdio_roundtrip_and_trace(self):
        tests = (
            encode_stdio_test("2\n", "4\n"),
            encode_stdio_test("5\n", "10\n"),
            encode_stdio_test("0\n", "0\n"),
        )
        self.assertEqual(parse_stdio_test(tests[0]), ("2\n", "4\n"))
        source = "n = int(input())\nprint(n * 2)\n"
        task = MBPPTask(task_id=11, description="double", canonical_source=source, tests=tests)
        result = run_mbpp_tests(task)
        self.assertEqual(result.get("status"), "all_pass")
        self.assertEqual(result.get("tests"), ["pass", "pass", "pass"])
        traced = trace_mbpp_tests(task, max_events=64)
        self.assertEqual(traced.get("status"), "all_pass")
        self.assertTrue(traced.get("traces"))
        self.assertTrue(any(event.get("event") == "line" for event in traced["traces"][0]))

    def test_stdio_detects_wrong_output(self):
        tests = (
            encode_stdio_test("2\n", "4\n"),
            encode_stdio_test("5\n", "10\n"),
        )
        task = MBPPTask(
            task_id=11,
            description="double",
            canonical_source="n = int(input())\nprint(n * 3)\n",
            tests=tests,
        )
        result = run_mbpp_tests(task)
        self.assertEqual(result.get("status"), "all_fail")
        self.assertEqual(result.get("tests"), ["fail", "fail"])
        observed = result.get("observed") or []
        self.assertEqual(len(observed), 2)
        self.assertEqual(str(observed[0].get("expected", "")).strip(), "4")
        self.assertEqual(str(observed[0].get("got", "")).strip(), "6")

    def test_example_hides_canonical_and_keeps_hidden_io(self):
        bug = {
            "bug_id": 7,
            "problem_id": "toy",
            "description": "double the integer",
            "buggy_code": "n = int(input())\nprint(n + 2)\n",
            "fixed_code": "n = int(input())\nprint(n * 2)\n",
            "split": "test",
            "tests": [("1\n", "2\n"), ("3\n", "6\n"), ("4\n", "8\n"), ("5\n", "10\n"), ("0\n", "0\n")],
        }
        example, audit = example_from_runbugrun(bug, split="test")
        self.assertTrue(audit["selected"], audit)
        self.assertIsNotNone(example)
        assert example is not None
        self.assertTrue(example.hidden_tests)
        self.assertNotEqual(example.buggy_source, example.target_source)
        prompt = render_user_prompt(example, evidence_for_condition(example, "gist_text"))
        self.assertIn(example.buggy_source.strip(), prompt)
        self.assertNotIn(example.target_source.strip(), prompt)
        self.assertIn("Input:", format_public_test(example.failing_public_test))
        self.assertIn("Input:", prompt)
        gist = evidence_for_condition(example, "gist_text")
        self.assertIn("TEST:", gist)
        self.assertIn("DISCREPANCY:", gist)
        self.assertIn("expected", gist)
        self.assertIn("got", gist)
        fail_text = " ".join(event.content for event in example.events if event.role_id == 7)
        self.assertIn("expected", fail_text)
        self.assertIn("got", fail_text)
        locator = evidence_for_condition(example, "line_text")
        self.assertEqual(locator, line_text(example.events, example.buggy_source))
        self.assertIn("print(n + 2)", locator)
        self.assertNotIn("print(n * 2)", locator)
        self.assertNotIn("DISCREPANCY", locator)
        self.assertTrue(locator.startswith("failing statement"))
        one_liner = evidence_for_condition(example, "discrepancy_text")
        self.assertTrue(one_liner.startswith("DISCREPANCY:"))
        self.assertIn("expected", one_liner)
        self.assertIn("got", one_liner)
        self.assertNotIn("LAST RUNTIME", one_liner)
        self.assertNotIn("print(n * 2)", one_liner)
        self.assertEqual(evidence_for_condition(example, "discrepancy_embed"), "<TRACE2CACHE_REPAIR_CODE>")

    def test_extract_patch_allows_input_only_when_requested(self):
        payload = "```python\nn = int(input())\nprint(n)\n```"
        with self.assertRaises(ValueError):
            extract_patch(payload)
        source = extract_patch(payload, allowed_calls={"input"})
        self.assertIn("input()", source)
        sys_patch = extract_patch("```python\nimport sys\nprint(sys.stdin.read())\n```")
        self.assertIn("import sys", sys_patch)
        eval_patch = extract_patch(
            "```python\nn = eval(input())\nprint(n)\n```",
            allowed_calls={"input", "eval"},
        )
        self.assertIn("eval(input())", eval_patch)
        random_patch = extract_patch("```python\nimport random\nprint(random.random())\n```")
        self.assertIn("import random", random_patch)
        with self.assertRaises(ValueError):
            extract_patch("```python\nn = eval(input())\nprint(n)\n```", allowed_calls={"input"})
        with self.assertRaises(ValueError):
            extract_patch("```python\nimport os\nprint(os.getcwd())\n```")
        self.assertEqual(
            allowed_calls_for_tests((encode_stdio_test("1\n", "1\n"),)),
            {"input", "eval"},
        )
        self.assertEqual(allowed_calls_for_tests(("assert f(1) == 1",)), set())


if __name__ == "__main__":
    unittest.main()
