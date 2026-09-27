from __future__ import annotations

import unittest

from trace2cache.coconut_execution import (
    COCONUT_K,
    COCONUT_LATENT,
    COCONUT_MIX,
    DEFAULT_STAGES,
    coconut_language_target,
    condition_for_stage,
    copy_stdout_control_rows,
    gold_event_lines,
    stage_for_step,
    summarize_coconut,
    total_steps,
)
from trace2cache.execution_sim import (
    SFT_STDOUT,
    EvidenceEvent,
    SimExample,
    sft_events_target,
    sft_stdout_target,
)
from trace2cache.mbpp_generalization import LINE, STATE


class CoconutExecutionTests(unittest.TestCase):
    def _example(self) -> SimExample:
        events = (
            EvidenceEvent(
                0,
                LINE,
                "line <module> line 1 source n = int(input()) state n=3",
            ),
            EvidenceEvent(
                0,
                STATE,
                "line <module> line 2 source print(n * 2) state n=3, out=6",
            ),
            EvidenceEvent(
                0,
                LINE,
                "line <module> line 2 source print(n * 2) state n=3",
            ),
        )
        return SimExample(
            example_id="sim_toy",
            task_id=11,
            source="n = int(input())\nprint(n * 2)\n",
            stdin="3\n",
            gold_stdout="6\n",
            events=events,
            event_count=len(events),
            source_hash="abc",
        )

    def test_language_target_keeps_tail_events_then_stdout(self):
        example = self._example()
        lines = gold_event_lines(example)
        self.assertGreaterEqual(len(lines), 2)
        self.assertTrue(all(line.startswith(("LINE:", "STATE:")) for line in lines))
        mixed = coconut_language_target(example, k=1, keep_events=True)
        self.assertIn(lines[1], mixed)
        self.assertNotIn(lines[0], mixed)
        self.assertTrue(mixed.endswith(sft_stdout_target(example)))
        self.assertEqual(coconut_language_target(example, k=8, keep_events=True), sft_stdout_target(example))
        self.assertEqual(
            coconut_language_target(example, k=COCONUT_K, keep_events=False),
            sft_stdout_target(example),
        )
        with self.assertRaises(ValueError):
            coconut_language_target(example, k=0, keep_events=True)

    def test_mix_is_not_full_event_chain_and_not_tracer_text(self):
        example = self._example()
        mixed = coconut_language_target(example, k=1, keep_events=True)
        full = sft_events_target(example)
        self.assertNotEqual(mixed, full)
        self.assertLess(mixed.count("LINE:"), full.count("LINE:"))
        self.assertNotIn("__builtins__", mixed)

    def test_stage_schedule(self):
        self.assertEqual(total_steps(), 1280)
        self.assertEqual(stage_for_step(1)["name"], "mix8")
        self.assertTrue(stage_for_step(640)["keep_events"])
        self.assertEqual(stage_for_step(641)["name"], "latent8")
        self.assertFalse(stage_for_step(1280)["keep_events"])
        self.assertEqual(stage_for_step(9999)["name"], DEFAULT_STAGES[-1]["name"])
        self.assertEqual(condition_for_stage("mix8"), COCONUT_MIX)
        self.assertEqual(condition_for_stage("latent8"), COCONUT_LATENT)
        with self.assertRaises(ValueError):
            stage_for_step(0)
        with self.assertRaises(ValueError):
            condition_for_stage("events")

    def test_copy_stdout_control_does_not_keep_event_chain(self):
        rows = [
            {"task_id": 1, "example_id": "a", "condition": SFT_STDOUT, "passed": True},
            {"task_id": 1, "example_id": "a", "condition": "sft_events", "passed": False},
        ]
        copied = copy_stdout_control_rows(rows)
        self.assertEqual(len(copied), 1)
        self.assertEqual(copied[0]["condition"], SFT_STDOUT)
        self.assertEqual(copied[0]["thoughts"], 0)
        self.assertTrue(copied[0]["passed"])
        with self.assertRaises(ValueError):
            copy_stdout_control_rows([{"condition": "sft_events", "passed": True}])

    def test_kill_rule(self):
        def rows(coconut_pass, stdout_pass, task_id=1):
            return [
                {
                    "task_id": task_id,
                    "condition": COCONUT_LATENT,
                    "passed": coconut_pass,
                    "thoughts": 8,
                },
                {"task_id": task_id, "condition": SFT_STDOUT, "passed": stdout_pass, "thoughts": 0},
            ]

        win = []
        for i in range(12):
            win.extend(rows(True, False, i))
        for i in range(12, 20):
            win.extend(rows(True, True, i))
        summary = summarize_coconut(win)
        self.assertEqual(summary["call"], "coconut_pass")
        self.assertEqual(summary["correct"][COCONUT_LATENT], 20)
        self.assertEqual(summary["latent_vs_stdout"]["left_only"], 12)

        fail = []
        for i in range(20):
            fail.extend(rows(False, True, i))
        self.assertEqual(summarize_coconut(fail)["call"], "coconut_fail")
        with self.assertRaises(ValueError):
            summarize_coconut(
                [{"task_id": 1, "condition": COCONUT_MIX, "passed": True, "thoughts": 8}]
            )


if __name__ == "__main__":
    unittest.main()
