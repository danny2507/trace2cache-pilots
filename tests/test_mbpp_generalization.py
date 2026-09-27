from __future__ import annotations

import unittest
from dataclasses import replace

from trace2cache.mbpp import MBPPMutation, MBPPTask
from trace2cache.mbpp_generalization import (
    IO_ROLES,
    MARKER,
    FAIL,
    LINE,
    RETURN,
    STATE,
    TEST,
    EvidenceEvent,
    RepairExample,
    collated_trace_text,
    compact_trace_text,
    corrupt_runtime_events,
    discrepancy_text,
    edit_sketch,
    evidence_for_condition,
    example_from_mutation,
    extract_patch,
    failing_span,
    gist_events,
    gist_text,
    line_text,
    hash_key,
    io_only_events,
    is_wrong_value_failure,
    literal_calls,
    partition_tests,
    render_gist_user_prompt,
    render_icae_encode_text,
    render_sketch_user_prompt,
    render_user_prompt,
    shared_affix_edit_mask,
)
from trace2cache.receiver_training import MARKER as RECEIVER_MARKER


class MBPPGeneralizationTests(unittest.TestCase):
    def _task(self) -> MBPPTask:
        return MBPPTask(
            task_id=601,
            description="Increment x.",
            canonical_source="def inc(x):\n    return x + 1\n",
            tests=("assert inc(0) == 1", "assert inc(2) == 3", "assert inc(10) == 11"),
        )

    def test_partition_requires_a_hidden_failure(self):
        tests = ("assert inc(0) == 1", "assert inc(2) == 3", "assert inc(10) == 11")
        source = "def inc(x):\n    return x + 1\n"
        mixed_one_fail = partition_tests(
            tests, ["pass", "pass", "pass"], ["fail", "pass", "pass"], source=source
        )
        self.assertIsNone(mixed_one_fail)
        eligible = partition_tests(
            tests, ["pass", "pass", "pass"], ["fail", "fail", "pass"], source=source
        )
        self.assertIsNotNone(eligible)
        self.assertEqual(eligible.public_tests, ("assert inc(0) == 1",))
        self.assertEqual(eligible.hidden_tests, ("assert inc(2) == 3", "assert inc(10) == 11"))
        self.assertFalse(set(eligible.public_hashes) & set(eligible.hidden_hashes))

    def test_literal_call_hashes_are_stable_and_input_specific(self):
        source = "def inc(x):\n    return x + 1\n"
        first = literal_calls(source, ("assert inc(2) == 3",))
        second = literal_calls(source, ("assert inc(2) == 99",))
        third = literal_calls(source, ("assert inc(3) == 4",))
        self.assertEqual(first, ["inc(2)"])
        self.assertEqual(first, second)
        self.assertNotEqual(hash_key(first[0]), hash_key(third[0]))

    def test_example_hides_canonical_source_from_prompts_and_keeps_hidden_tests_out(self):
        task = self._task()
        mutation = MBPPMutation("binop", 0, "def inc(x):\n    return x - 1\n")
        example, audit = example_from_mutation(task, mutation)
        self.assertTrue(audit["selected"], audit)
        self.assertIsInstance(example, RepairExample)
        self.assertEqual(example.public_tests, ("assert inc(0) == 1",))
        self.assertIn("assert inc(2) == 3", example.hidden_tests)
        self.assertTrue(set(example.public_hashes).isdisjoint(example.hidden_hashes))
        prompt = render_user_prompt(example, evidence_for_condition(example, "compact_text"))
        self.assertIn(example.buggy_source.strip(), prompt)
        self.assertIn(example.failing_public_test, prompt)
        self.assertNotIn(example.target_source.strip(), prompt)
        for hidden in example.hidden_tests:
            self.assertNotIn(hidden, prompt)
        latent_prompt = render_user_prompt(example, evidence_for_condition(example, "true_latent"))
        self.assertIn(MARKER, latent_prompt)
        self.assertEqual(MARKER, RECEIVER_MARKER)
        self.assertIn("TEST 0", compact_trace_text(example.events))
        self.assertIn("FAIL:", compact_trace_text(example.events))
        self.assertNotIn("Specification:", prompt)
        self.assertNotIn(example.description, prompt)
        with_spec = render_user_prompt(
            example, evidence_for_condition(example, "compact_text"), include_specification=True
        )
        self.assertIn(example.description, with_spec)
        without_public = render_user_prompt(example, MARKER, include_public_test=False)
        self.assertNotIn(example.failing_public_test, without_public)

    def test_io_and_corruption_keep_assertions_and_drop_runtime_payloads(self):
        task = self._task()
        mutation = MBPPMutation("binop", 0, "def inc(x):\n    return x - 1\n")
        example, audit = example_from_mutation(task, mutation)
        self.assertTrue(audit["selected"], audit)
        io_events = io_only_events(example.events)
        self.assertTrue(io_events)
        self.assertTrue(all(event.role_id in IO_ROLES for event in io_events))
        self.assertLess(len(io_events), len(example.events))
        io_text = evidence_for_condition(example, "io_text")
        compact = evidence_for_condition(example, "compact_text")
        self.assertIn("test assertion", io_text)
        self.assertGreater(len(compact), len(io_text))
        corrupted = corrupt_runtime_events(example.events)
        self.assertEqual(len(corrupted), len(example.events))
        for original, changed in zip(example.events, corrupted):
            if original.role_id in {0, 6, 7}:
                self.assertEqual(original, changed)
            else:
                self.assertEqual(changed.content, "CORRUPTED_RUNTIME")

    def test_extract_patch_requires_a_code_block_and_rejects_exec(self):
        source = extract_patch("Sure.\n```python\ndef inc(x):\n    return x + 1\n```\n")
        self.assertEqual(source, "def inc(x):\n    return x + 1\n")
        with self.assertRaises(ValueError):
            extract_patch("```python\ndef inc(x):\n    return eval('x+1')\n```")

    def test_gist_keeps_io_and_last_runtime_and_drops_unrolled_lines(self):
        task = self._task()
        mutation = MBPPMutation("binop", 0, "def inc(x):\n    return x - 1\n")
        example, audit = example_from_mutation(task, mutation)
        self.assertTrue(audit["selected"], audit)
        chosen = gist_events(example.events)
        self.assertTrue(chosen)
        self.assertLessEqual(len(chosen), len(example.events))
        self.assertTrue(any(event.role_id in IO_ROLES for event in chosen))
        line_events = [event for event in example.events if event.role_id == LINE]
        if line_events:
            self.assertEqual(
                [event for event in chosen if event.role_id == LINE][-1],
                line_events[-1],
            )
        note = gist_text(example.events)
        self.assertIn("TEST:", note)
        self.assertIn("expected", note.lower())
        self.assertIn("got", note.lower())
        self.assertLess(len(note), len(compact_trace_text(example.events)))
        fail_events = [event for event in example.events if event.role_id == FAIL]
        self.assertTrue(fail_events)
        self.assertIn("expected", fail_events[-1].content)
        self.assertIn("got", fail_events[-1].content)
        self.assertIn("DISCREPANCY:", note)
        frozen_style = (
            EvidenceEvent(0, TEST, "test assertion assert inc(0) == 1"),
            EvidenceEvent(0, RETURN, "return inc line 2 source return x - 1 state x=0 return=-1"),
            EvidenceEvent(0, FAIL, "test outcome fail"),
        )
        frozen_note = gist_text(frozen_style)
        self.assertIn("DISCREPANCY:", frozen_note)
        self.assertIn("expected 1", frozen_note)
        self.assertIn("got -1", frozen_note)
        self.assertIn(MARKER, render_gist_user_prompt(example))
        self.assertNotIn(example.failing_public_test, render_gist_user_prompt(example))
        self.assertTrue(is_wrong_value_failure(example) or any(event.role_id == FAIL for event in example.events))

    def test_wrong_value_filter_rejects_exceptions(self):
        task = self._task()
        mutation = MBPPMutation("binop", 0, "def inc(x):\n    return x - 1\n")
        example, audit = example_from_mutation(task, mutation)
        self.assertTrue(audit["selected"], audit)
        if any("error" in event.content.lower() for event in example.events if event.role_id == FAIL):
            self.assertFalse(is_wrong_value_failure(example))
        else:
            self.assertTrue(is_wrong_value_failure(example))

    def test_shared_affix_mask_marks_the_changed_middle(self):
        buggy = [1, 2, 3, 4, 5]
        target = [1, 2, 9, 4, 5]
        self.assertEqual(shared_affix_edit_mask(buggy, target), [False, False, True, False, False])
        self.assertEqual(shared_affix_edit_mask([1, 2], [1, 2]), [True, True])

    def test_gist_events_drops_unrolled_middle_runtime(self):
        events = (
            EvidenceEvent(0, TEST, "assert inc(0) == 1"),
            EvidenceEvent(0, LINE, "line 1"),
            EvidenceEvent(0, LINE, "line 2"),
            EvidenceEvent(0, LINE, "line 3"),
            EvidenceEvent(0, STATE, "x=-1"),
            EvidenceEvent(0, STATE, "x=-2"),
            EvidenceEvent(0, RETURN, "-1"),
            EvidenceEvent(0, FAIL, "AssertionError"),
        )
        chosen = gist_events(events)
        self.assertEqual(
            [event.content for event in chosen],
            ["assert inc(0) == 1", "line 3", "x=-2", "-1", "AssertionError"],
        )
        exception_example = RepairExample(
            example_id="x",
            task_id=1,
            split="train",
            mutation_kind="binop",
            mutation_ordinal=0,
            description="",
            buggy_source="x",
            target_source="y",
            setup_source="",
            failing_public_test="assert inc(0) == 1",
            public_tests=("assert inc(0) == 1",),
            hidden_tests=("assert inc(2) == 3",),
            public_hashes=("a",),
            hidden_hashes=("b",),
            events=events,
            source_hash="h",
            mutant_public_status="fail",
            mutant_hidden_status="fail",
        )
        self.assertFalse(is_wrong_value_failure(exception_example))
        value_events = events[:-1] + (EvidenceEvent(0, FAIL, "test outcome fail: expected 1 got -1"),)
        self.assertTrue(is_wrong_value_failure(replace(exception_example, events=value_events)))

    def test_edit_sketch_is_a_short_replace_and_not_the_full_canonical(self):
        buggy = "def inc(x):\n    return x - 1\n"
        target = "def inc(x):\n    return x + 1\n"
        sketch = edit_sketch(buggy, target)
        self.assertIn("REPLACE", sketch)
        self.assertIn("x - 1", sketch)
        self.assertIn("x + 1", sketch)
        self.assertNotIn("def inc", sketch)
        self.assertEqual(edit_sketch(target, target), "NO_EDIT")
        prompt = render_user_prompt(
            RepairExample(
                example_id="x",
                task_id=1,
                split="test",
                mutation_kind="binop",
                mutation_ordinal=0,
                description="Increment x.",
                buggy_source=buggy,
                target_source=target,
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
            ),
            sketch,
            evidence_kind="oracle",
        )
        self.assertIn("ORACLE EDIT SKETCH", prompt)
        self.assertIn(sketch, prompt)
        self.assertNotIn(target.strip(), prompt)

    def test_sketch_prompt_has_marker_and_omits_public_test(self):
        example = RepairExample(
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
        prompt = render_sketch_user_prompt(example)
        self.assertIn(MARKER, prompt)
        self.assertIn(example.buggy_source.strip(), prompt)
        self.assertNotIn(example.failing_public_test, prompt)
        self.assertNotIn("PUBLIC FAILING TEST", prompt)
        self.assertNotIn("WITH `return x + 1`", prompt)

    def test_icae_encode_text_has_buggy_and_public_test_not_canonical(self):
        events = (
            EvidenceEvent(0, TEST, "assert inc(0) == 1"),
            EvidenceEvent(0, FAIL, "test outcome fail: expected 1 got -1"),
        )
        example = RepairExample(
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
            events=events,
            source_hash="h",
            mutant_public_status="fail",
            mutant_hidden_status="fail",
        )
        text = render_icae_encode_text(example, events)
        self.assertIn(example.buggy_source.strip(), text)
        self.assertIn(example.failing_public_test, text)
        self.assertIn("TEST:", text)
        self.assertNotIn("REPLACE", text)
        self.assertNotIn("WITH `return x + 1`", text)
        self.assertNotIn(example.target_source.strip(), text)
        sketch = edit_sketch(example.buggy_source, example.target_source)
        self.assertNotIn(sketch, text)

    def test_gist_text_condition_is_shorter_than_compact_and_keeps_io(self):
        events = (
            EvidenceEvent(0, TEST, "assert inc(0) == 1"),
            EvidenceEvent(0, LINE, "line inc line 2 source return x - 1"),
            EvidenceEvent(0, LINE, "line inc line 2 source return x - 1 again"),
            EvidenceEvent(0, STATE, "line inc line 2 source return x - 1 state x=0"),
            EvidenceEvent(0, RETURN, "return inc line 2 source return x - 1 return=-1"),
            EvidenceEvent(0, FAIL, "test outcome fail: expected 1 got -1"),
        )
        example = RepairExample(
            example_id="x",
            task_id=1,
            split="test",
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
            events=events,
            source_hash="h",
            mutant_public_status="fail",
            mutant_hidden_status="fail",
        )
        gist = evidence_for_condition(example, "gist_text")
        compact = evidence_for_condition(example, "compact_text")
        self.assertEqual(gist, gist_text(events))
        self.assertIn("TEST:", gist)
        self.assertIn("FAIL:", gist)
        self.assertIn("RETURN:", gist)
        self.assertIn("again", gist)
        self.assertNotIn("LINE: line inc line 2 source return x - 1\n", gist)
        self.assertLess(len(gist), len(compact))
        self.assertNotIn(example.target_source.strip(), gist)

    def test_collated_text_comments_source_lines_and_omits_canonical(self):
        events = (
            EvidenceEvent(0, TEST, "assert inc(0) == 1"),
            EvidenceEvent(0, LINE, "line inc line 2 source return x - 1"),
            EvidenceEvent(0, STATE, "line inc line 2 source return x - 1 state x=0"),
            EvidenceEvent(0, FAIL, "test outcome fail: expected 1 got -1"),
        )
        example = RepairExample(
            example_id="x",
            task_id=1,
            split="test",
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
            events=events,
            source_hash="h",
            mutant_public_status="fail",
            mutant_hidden_status="fail",
        )
        collated = evidence_for_condition(example, "collated_text")
        self.assertEqual(collated, collated_trace_text(example.buggy_source, events))
        self.assertIn("TEST:", collated)
        self.assertIn("FAIL:", collated)
        self.assertIn("return x - 1  # TRACE:", collated)
        self.assertIn("state x=0", collated)
        self.assertNotIn("return x + 1", collated)
        self.assertNotIn("def inc(x):  # TRACE:", collated)

    def test_discrepancy_text_is_expected_vs_got_without_runtime_or_canonical(self):
        events = (
            EvidenceEvent(0, TEST, "assert inc(0) == 1"),
            EvidenceEvent(0, LINE, "line inc line 2 source return x - 1"),
            EvidenceEvent(0, RETURN, "return inc line 2 source return x - 1 return=-1"),
            EvidenceEvent(0, FAIL, "test outcome fail: expected 1 got -1"),
        )
        example = RepairExample(
            example_id="x",
            task_id=1,
            split="test",
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
            events=events,
            source_hash="h",
            mutant_public_status="fail",
            mutant_hidden_status="fail",
        )
        note = evidence_for_condition(example, "discrepancy_text")
        self.assertEqual(note, discrepancy_text(events))
        self.assertEqual(note, "DISCREPANCY: expected 1 got -1")
        self.assertTrue(gist_text(events).startswith(note))
        self.assertNotIn("LAST RUNTIME", note)
        self.assertNotIn("return x - 1", note)
        self.assertNotIn("return x + 1", note)
        self.assertEqual(
            discrepancy_text(events, swap=True),
            "DISCREPANCY: expected -1 got 1",
        )
        self.assertEqual(evidence_for_condition(example, "discrepancy_embed"), MARKER)
        self.assertEqual(evidence_for_condition(example, "shuffled_embed"), MARKER)
        self.assertEqual(evidence_for_condition(example, "corrupted_embed"), MARKER)
        self.assertEqual(discrepancy_text(events[:1]), "DISCREPANCY: expected 1 got ?")
        self.assertEqual(discrepancy_text(()), "unavailable")

    def test_line_text_is_the_last_executed_source_not_io_or_canonical(self):
        events = (
            EvidenceEvent(0, TEST, "assert inc(0) == 1"),
            EvidenceEvent(0, LINE, "line inc line 1 source def inc(x): state "),
            EvidenceEvent(0, LINE, "line inc line 2 source return x - 1 state x=0"),
            EvidenceEvent(0, RETURN, "return inc line 2 source return x - 1 state x=0 return=-1"),
            EvidenceEvent(0, FAIL, "test outcome fail: expected 1 got -1"),
        )
        example = RepairExample(
            example_id="x",
            task_id=1,
            split="test",
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
            events=events,
            source_hash="h",
            mutant_public_status="fail",
            mutant_hidden_status="fail",
        )
        note = evidence_for_condition(example, "line_text")
        self.assertEqual(note, line_text(events, example.buggy_source))
        self.assertEqual(failing_span(events, example.buggy_source), (2, "return x - 1"))
        self.assertEqual(note, "failing statement (line 2): return x - 1")
        self.assertNotIn("expected", note)
        self.assertNotIn("got", note)
        self.assertNotIn("DISCREPANCY", note)
        self.assertNotIn("return x + 1", note)
        self.assertNotIn(example.target_source.strip(), note)
        io_only = events[:1] + events[-1:]
        self.assertEqual(line_text(io_only, example.buggy_source), "unavailable")
        crashed = events[:-1] + (
            EvidenceEvent(0, RETURN, "exception inc line 2 source return x - 1 state x=0"),
            events[-1],
        )
        self.assertEqual(failing_span(crashed, example.buggy_source)[1], "return x - 1")
        numbered = (
            EvidenceEvent(0, TEST, "assert inc(0) == 1"),
            EvidenceEvent(0, LINE, "line inc line 2"),
            EvidenceEvent(0, FAIL, "test outcome fail: expected 1 got -1"),
        )
        self.assertEqual(failing_span(numbered, example.buggy_source), (2, "return x - 1"))


if __name__ == "__main__":
    unittest.main()
