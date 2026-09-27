from __future__ import annotations

import unittest

from trace2cache.execution_sim import (
    MARKER,
    EvidenceEvent,
    SimExample,
    different_task_example,
    prompt_contains_gold,
    render_user_prompt,
    evidence_for_condition,
)
from trace2cache.latent_execution import (
    DISTILL_K,
    resample_events,
    resample_indices,
    teacher_texts,
)
from trace2cache.mbpp_generalization import CALL, LINE, STATE


class LatentExecutionTests(unittest.TestCase):
    def _example(self, events, task_id=11) -> SimExample:
        return SimExample(
            example_id=f"sim_{task_id}",
            task_id=task_id,
            source="n = int(input())\nprint(n * 2)\n",
            stdin="3\n",
            gold_stdout="6\n",
            events=tuple(events),
            event_count=len(events),
            source_hash="abc",
        )

    def test_resample_identity_and_endpoints(self):
        self.assertEqual(resample_indices(16, 16), tuple(range(16)))
        self.assertEqual(resample_indices(1, 8), (0,) * 8)
        self.assertEqual(resample_indices(5, 1), (0,))
        idx = resample_indices(19, 16)
        self.assertEqual(len(idx), 16)
        self.assertEqual(idx[0], 0)
        self.assertEqual(idx[-1], 18)

    def test_teacher_texts_are_role_prefixed_and_length_k(self):
        events = (
            EvidenceEvent(0, CALL, "call double(3)"),
            EvidenceEvent(0, LINE, "line source print(n * 2) state n=3"),
            EvidenceEvent(0, STATE, "state n=3"),
        )
        texts = teacher_texts(events, 8)
        self.assertEqual(len(texts), 8)
        self.assertTrue(all(text.startswith(("CALL:", "LINE:", "STATE:")) for text in texts))
        self.assertEqual(texts[0], "CALL: call double(3)")
        self.assertEqual(texts[-1], "STATE: state n=3")

    def test_distill_prompt_has_code_stdin_not_trace_or_gold(self):
        example = self._example(
            (EvidenceEvent(0, STATE, "state n=3 out=hello-output"),)
        )
        prompt = render_user_prompt(example, evidence_for_condition(example, "code_input"))
        self.assertIn("n = int(input())", prompt)
        self.assertIn("3", prompt)
        self.assertNotIn(MARKER, prompt)
        self.assertNotIn("Partial execution", prompt)
        self.assertFalse(prompt_contains_gold(prompt, example.gold_stdout))

    def test_shuffled_teacher_uses_other_task_events(self):
        a = self._example((EvidenceEvent(0, STATE, "state-a"),), task_id=11)
        b = self._example((EvidenceEvent(0, STATE, "state-b"),), task_id=12)
        picked = different_task_example([a, b], 0)
        self.assertEqual(picked.task_id, 12)
        self.assertNotEqual(teacher_texts(a.events, DISTILL_K), teacher_texts(picked.events, DISTILL_K))
        with self.assertRaises(ValueError):
            resample_events((), 8)
        with self.assertRaises(ValueError):
            resample_indices(0, 8)
