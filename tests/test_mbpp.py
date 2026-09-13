from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from trace2cache.mbpp import (
    MBPPTask,
    generate_mutants,
    load_mbpp,
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


if __name__ == "__main__":
    unittest.main()
