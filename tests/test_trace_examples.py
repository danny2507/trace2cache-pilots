from __future__ import annotations

import unittest

from scripts.generate_python_trace_examples import build_examples


class TraceExampleTests(unittest.TestCase):
    def test_examples_are_executed_failures_with_structured_roles(self):
        examples = build_examples(limit=2)
        self.assertEqual(len(examples), 2)
        for example in examples:
            self.assertFalse(example["test_passed"])
            self.assertGreater(example["trace_event_count"], 0)
            roles = {event["semantic_role"] for event in example["structured_events"]}
            self.assertIn("call_input", roles)
            self.assertIn("return_sink", roles)
            self.assertTrue(any(event["on_compact_slice"] for event in example["structured_events"]))


if __name__ == "__main__":
    unittest.main()
