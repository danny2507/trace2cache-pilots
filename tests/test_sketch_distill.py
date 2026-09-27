from __future__ import annotations

import unittest

from trace2cache.mbpp_generalization import RepairExample, edit_sketch
from trace2cache.sketch_distill import (
    COMPACT_TEXT,
    DIRECT_SFT,
    IO_TEXT,
    NO_EVIDENCE,
    SKETCH_DISTILL,
    copy_text_control_rows,
    fenced_patch,
    prompt_leaks_oracle,
    render_repair_chat,
    sft_target,
    sketch_prefix,
    summarize_sketch_distill,
)


class _Tok:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        text = "\n".join(message["content"] for message in messages)
        if add_generation_prompt:
            text += "\nASSISTANT:"
        return text


class SketchDistillTests(unittest.TestCase):
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

    def test_same_prompt_both_objectives_hides_oracle(self):
        example = self._example()
        prompt = render_repair_chat(_Tok(), example)
        again = render_repair_chat(_Tok(), example)
        self.assertEqual(prompt, again)
        self.assertIn("unavailable", prompt)
        self.assertIn("assert inc(0) == 1", prompt)
        self.assertIn("return x - 1", prompt)
        self.assertNotIn("Increment x.", prompt)
        self.assertFalse(prompt_leaks_oracle(prompt, example))
        sketch = edit_sketch(example.buggy_source, example.target_source)
        self.assertIn("REPLACE", sketch)
        self.assertNotIn(sketch, prompt)
        self.assertNotIn(example.target_source.strip(), prompt)

    def test_targets_put_sketch_only_on_distill(self):
        example = self._example()
        sketch = edit_sketch(example.buggy_source, example.target_source)
        fence = fenced_patch(example.target_source)
        direct = sft_target(example, DIRECT_SFT)
        distill = sft_target(example, SKETCH_DISTILL)
        self.assertEqual(direct, fence)
        self.assertNotIn("REPLACE", direct)
        self.assertTrue(distill.startswith(sketch))
        self.assertEqual(distill, sketch_prefix(example) + fence)
        self.assertIn(fence, distill)
        with self.assertRaises(ValueError):
            sft_target(example, "no_evidence")

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
            copy_text_control_rows(
                [{"condition": SKETCH_DISTILL, "passed": True, "task_id": 11}]
            )

        fail_rows = rows + [
            {"task_id": 11, "condition": DIRECT_SFT, "passed": True},
            {"task_id": 12, "condition": DIRECT_SFT, "passed": False},
            {"task_id": 11, "condition": SKETCH_DISTILL, "passed": True},
            {"task_id": 12, "condition": SKETCH_DISTILL, "passed": False},
        ]
        fail = summarize_sketch_distill(fail_rows)
        self.assertEqual(fail["call"], "sketch_distill_fail")
        self.assertFalse(fail["beats_direct"])
        self.assertFalse(fail["beats_no_evidence"])

        pass_rows = (
            [{"task_id": i, "condition": NO_EVIDENCE, "passed": i < 2} for i in range(8)]
            + [{"task_id": i, "condition": DIRECT_SFT, "passed": i < 2} for i in range(8)]
            + [{"task_id": i, "condition": SKETCH_DISTILL, "passed": True} for i in range(8)]
        )
        passed = summarize_sketch_distill(pass_rows)
        self.assertEqual(passed["call"], "sketch_distill_pass")
        self.assertTrue(passed["beats_direct"])
        self.assertTrue(passed["beats_no_evidence"])
        self.assertGreater(
            passed["sketch_vs_direct"]["left_only"], passed["sketch_vs_direct"]["right_only"]
        )
        self.assertLess(passed["sketch_vs_direct"]["p"], 0.05)
        self.assertLess(passed["sketch_vs_no_evidence"]["p"], 0.05)
