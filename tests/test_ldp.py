from __future__ import annotations

import unittest

from trace2cache.ldp import (
    COMPACT_TEXT,
    CORRUPTED_LATENT,
    IO_TEXT,
    NO_EVIDENCE,
    SHUFFLED_LATENT,
    TRUE_LATENT,
    copy_text_control_rows,
    corrupt_edit_sketch,
    render_ldp_gate1_encode_text,
    summarize_ldp_gate1,
)
from trace2cache.mbpp_generalization import RepairExample, edit_sketch


class LDPGate1Tests(unittest.TestCase):
    def _example(self) -> RepairExample:
        return RepairExample(
            example_id="x",
            task_id=1,
            split="train",
            mutation_kind="binop",
            mutation_ordinal=0,
            description="Increment x.",
            buggy_source="def inc(x):\n    return x - 1\n",
            target_source="def inc(x):\n    return x + 1\n",
            setup_source="",
            failing_public_test="assert inc(0) == 1",
            public_tests=("assert inc(0) == 1",),
            hidden_tests=("assert inc(2) == 3",),
            public_hashes=("a",),
            hidden_hashes=("b",),
            events=(),
            source_hash="h",
            mutant_public_status="fail",
            mutant_hidden_status="fail",
        )

    def test_encode_text_is_replace_only(self):
        example = self._example()
        sketch = edit_sketch(example.buggy_source, example.target_source)
        text = render_ldp_gate1_encode_text(sketch)
        self.assertEqual(text, sketch)
        self.assertIn("REPLACE", text)
        self.assertNotIn("BUGGY PROGRAM", text)
        self.assertNotIn("PUBLIC FAILING TEST", text)
        self.assertNotIn(example.failing_public_test, text)
        self.assertNotIn("def inc", text)
        self.assertNotIn(example.description, text)
        with self.assertRaises(ValueError):
            render_ldp_gate1_encode_text("  ")

    def test_corrupt_swaps_replace_operands(self):
        sketch = "REPLACE `return x - 1` WITH `return x + 1`"
        corrupted = corrupt_edit_sketch(sketch)
        self.assertEqual(corrupted, "REPLACE `return x + 1` WITH `return x - 1`")
        self.assertNotEqual(corrupted, sketch)
        self.assertEqual(corrupt_edit_sketch("NO_EDIT"), "CORRUPTED_SKETCH")
        self.assertTrue(corrupt_edit_sketch("DELETE `x`").startswith("CORRUPTED "))

    def test_copy_text_controls_and_kill_rule(self):
        rows = [
            {"task_id": 11, "condition": NO_EVIDENCE, "passed": True},
            {"task_id": 12, "condition": NO_EVIDENCE, "passed": False},
            {"task_id": 11, "condition": IO_TEXT, "passed": True},
            {"task_id": 12, "condition": IO_TEXT, "passed": False},
            {"task_id": 11, "condition": COMPACT_TEXT, "passed": False},
            {"task_id": 12, "condition": COMPACT_TEXT, "passed": False},
        ]
        copied = copy_text_control_rows(rows)
        self.assertEqual(len(copied), 6)
        with self.assertRaises(ValueError):
            copy_text_control_rows([{"condition": TRUE_LATENT, "passed": True, "task_id": 11}])

        fail_rows = rows + [
            {"task_id": 11, "condition": TRUE_LATENT, "passed": True},
            {"task_id": 12, "condition": TRUE_LATENT, "passed": False},
            {"task_id": 11, "condition": SHUFFLED_LATENT, "passed": True},
            {"task_id": 12, "condition": SHUFFLED_LATENT, "passed": False},
            {"task_id": 11, "condition": CORRUPTED_LATENT, "passed": True},
            {"task_id": 12, "condition": CORRUPTED_LATENT, "passed": False},
        ]
        fail = summarize_ldp_gate1(fail_rows)
        self.assertEqual(fail["call"], "ldp_g1_fail")
        self.assertFalse(fail["beats_no_evidence"])
        self.assertFalse(fail["shuffle_drop"])

        pass_rows = [
            {"task_id": i, "condition": NO_EVIDENCE, "passed": i < 2}
            for i in range(8)
        ] + [
            {"task_id": i, "condition": TRUE_LATENT, "passed": True}
            for i in range(8)
        ] + [
            {"task_id": i, "condition": SHUFFLED_LATENT, "passed": i < 2}
            for i in range(8)
        ]
        passed = summarize_ldp_gate1(pass_rows)
        self.assertEqual(passed["call"], "ldp_g1_pass")
        self.assertTrue(passed["beats_no_evidence"])
        self.assertTrue(passed["shuffle_drop"])
        self.assertGreater(passed["true_vs_no_evidence"]["left_only"], passed["true_vs_no_evidence"]["right_only"])
        self.assertLess(passed["true_vs_no_evidence"]["p"], 0.05)
