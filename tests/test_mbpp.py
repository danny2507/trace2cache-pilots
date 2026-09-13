from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from trace2cache.mbpp import (
    MBPPTask,
    generate_expanded_mutants,
    generate_mutants,
    load_mbpp,
    run_mbpp_calls,
    run_mbpp_tests,
    trace_mbpp_tests,
)


class MBPPTests(unittest.TestCase):
    def test_loads_official_split(self):
        row = {
            "task_id": 601,
            "text": "Increment x.",
            "code": "def inc(x):\n    return x + 1\n",
            "test_list": ["assert inc(2) == 3"],
            "test_setup_code": "",
            "challenge_test_list": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mbpp.jsonl"
            path.write_text(json.dumps(row) + "\n")
            task = load_mbpp(path, split="train")[0]
        self.assertEqual(task.task_id, 601)
        self.assertEqual(task.split, "train")

    def test_mutations_are_deterministic_and_executable(self):
        task = MBPPTask(
            task_id=601,
            description="Increment x.",
            canonical_source="def inc(x):\n    return x + 1\n",
            tests=("assert inc(0) == 1", "assert inc(2) == 3"),
        )
        first = generate_mutants(task, limit=8)
        second = generate_mutants(task, limit=8)
        self.assertEqual(first, second)
        self.assertGreater(len(first), 0)
        self.assertEqual(run_mbpp_tests(task)["status"], "all_pass")
        self.assertTrue(
            any(run_mbpp_tests(task, mutation.source)["status"] == "all_fail" for mutation in first)
        )

    def test_collects_separate_test_traces(self):
        task = MBPPTask(
            task_id=601,
            description="Increment x.",
            canonical_source="def inc(x):\n    y = x + 1\n    return y\n",
            tests=("assert inc(0) == 1", "assert inc(2) == 3"),
        )
        result = trace_mbpp_tests(task)
        self.assertEqual(result["status"], "all_pass")
        self.assertEqual(len(result["traces"]), 2)
        self.assertTrue(all(trace[0]["event"] == "call" for trace in result["traces"]))
        self.assertEqual(result["traces"][0][-1]["value"], "1")

    def test_expanded_mutants_include_structural_changes(self):
        task = MBPPTask(
            task_id=602,
            description="Accumulate positive values.",
            canonical_source=(
                "def total(xs):\n"
                "    result = 0\n"
                "    for x in xs:\n"
                "        if x > 0:\n"
                "            result += x\n"
                "    return result\n"
            ),
            tests=("assert total([-1, 2]) == 2",),
        )
        mutations = generate_expanded_mutants(task, limit=64)
        kinds = {mutation.kind.split(":")[0] for mutation in mutations}
        self.assertIn("negate_condition", kinds)
        self.assertIn("delete_assignment", kinds)
        self.assertIn("augassign", kinds)

    def test_evaluates_and_traces_literal_calls(self):
        task = MBPPTask(
            task_id=603,
            description="Double x.",
            canonical_source="def double(x):\n    return x * 2\n",
            tests=(),
        )
        result = run_mbpp_calls(task, ["double(3)", "double(-2)"])
        self.assertEqual([row["output"] for row in result["calls"]], ["6", "-4"])
        self.assertTrue(all(row["trace"] for row in result["calls"]))


if __name__ == "__main__":
    unittest.main()
