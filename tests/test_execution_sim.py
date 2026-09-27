from __future__ import annotations

import unittest

from trace2cache.execution_sim import (
    ASK_EVENTS,
    LATENT_CONDITIONS,
    MARKER,
    NATIVE_CAP,
    NATIVE_CONDITIONS,
    RUNTIME_KEEP,
    SFT_EVENTS,
    SFT_STDOUT,
    SIM_SLOT_ROLES,
    SNAP_CONDITIONS,
    EvidenceEvent,
    SimExample,
    assert_train_disjoint,
    compact_ask_event_line,
    count_event_role_lines,
    different_task_example,
    evidence_for_condition,
    example_from_repair,
    example_from_stdio,
    extract_output,
    filter_intermediate,
    gist_intermediate,
    gold_ask_events_text,
    native_event_texts,
    normalize_output,
    outputs_match,
    prompt_contains_gold,
    redact_answer,
    render_ask_events_prompt,
    render_user_prompt,
    response_event_prefix,
    sft_events_target,
    sft_stdout_target,
    sft_target,
    snap_table_to_vocab,
    summarize_ask_events_gate0,
    summarize_event_sft,
)
from trace2cache.mbpp_generalization import (
    BRANCH,
    CALL,
    FAIL,
    LINE,
    RETURN,
    STATE,
    TEST,
    compact_trace_text,
    encode_stdio_test,
)
from trace2cache.runbugrun import example_from_runbugrun


class ExecutionSimTests(unittest.TestCase):
    def _example(self, events=()) -> SimExample:
        return SimExample(
            example_id="sim_toy",
            task_id=11,
            source="n = int(input())\nprint(n * 2)\n",
            stdin="3\n",
            gold_stdout="6\n",
            events=tuple(events),
            event_count=len(events),
            source_hash="abc",
        )

    def test_normalize_and_match_ignore_trailing_newline(self):
        self.assertEqual(normalize_output("6\r\n"), "6")
        self.assertTrue(outputs_match("6", "6\n"))
        self.assertFalse(outputs_match("7", "6\n"))

    def test_redact_removes_gold_and_long_lines_not_single_digits(self):
        gold = "hello world\n"
        self.assertIn("[REDACTED]", redact_answer("got hello world from print", gold))
        self.assertEqual(redact_answer("cur=4", "4\n2\n3\n"), "cur=4")

    def test_filter_drops_io_and_return_and_redacts_gold(self):
        events = (
            EvidenceEvent(0, TEST, "test stdin='3\\n' expected='6\\n'"),
            EvidenceEvent(0, LINE, "line <module> line 1 source n = int(input()) state n=3"),
            EvidenceEvent(0, STATE, "line <module> line 2 source print(n * 2) state n=3"),
            EvidenceEvent(0, RETURN, "return <module> line 2 source print(n * 2) return=None"),
            EvidenceEvent(0, FAIL, "test outcome pass: expected 6 got 6"),
        )
        kept = filter_intermediate(events, "hello-output")
        roles = [event.role_id for event in kept]
        self.assertEqual(roles, [LINE, STATE])
        leaked = filter_intermediate(
            (EvidenceEvent(0, LINE, "printed hello-output here"),),
            "hello-output",
        )
        self.assertIn("[REDACTED]", leaked[0].content)

    def test_code_input_prompt_has_stdin_not_gold(self):
        example = self._example()
        prompt = render_user_prompt(example, evidence_for_condition(example, "code_input"))
        self.assertIn("n = int(input())", prompt)
        self.assertIn("3", prompt)
        self.assertNotIn("expected", prompt.lower())
        self.assertFalse(prompt_contains_gold(prompt, example.gold_stdout))
        self.assertNotIn("Partial execution", prompt)

    def test_trace_prompt_includes_redacted_runtime_not_gold(self):
        events = (
            EvidenceEvent(0, LINE, "line source print(n * 2) state n=3"),
            EvidenceEvent(0, STATE, "line source print(n * 2) state n=3 out=hello-output"),
        )
        example = self._example(events)
        # gold is "6\n"; "hello-output" stays. Build a leaking event with gold "6" of len 1 — not redacted.
        long_gold = SimExample(
            example_id="sim_toy",
            task_id=11,
            source=example.source,
            stdin="3\n",
            gold_stdout="hello-output\n",
            events=filter_intermediate(events, "hello-output\n"),
            event_count=2,
            source_hash="abc",
        )
        prompt = render_user_prompt(long_gold, evidence_for_condition(long_gold, "trace_text"))
        self.assertIn("Partial execution", prompt)
        self.assertIn("[REDACTED]", prompt)
        self.assertFalse(prompt_contains_gold(prompt, long_gold.gold_stdout))
        gist = evidence_for_condition(long_gold, "trace_gist")
        self.assertIn("LINE:", gist)
        self.assertNotIn("TEST:", gist)

    def test_extract_output_prefers_last_fence(self):
        response = "Sure.\n```\nwrong\n```\n```\n6\n```\n"
        self.assertEqual(extract_output(response).strip(), "6")
        self.assertEqual(extract_output("6\n").strip(), "6")

    def test_ask_events_prompt_has_stdin_format_not_gold_trace(self):
        events = (
            EvidenceEvent(0, LINE, "line source print(n * 2) state n=3"),
            EvidenceEvent(0, STATE, "secret-trace-token n=3"),
        )
        example = self._example(events)
        prompt = render_ask_events_prompt(example)
        self.assertIn("n = int(input())", prompt)
        self.assertIn("3", prompt)
        self.assertIn("CALL:", prompt)
        self.assertIn("BRANCH:", prompt)
        self.assertIn("LINE:", prompt)
        self.assertIn("STATE:", prompt)
        self.assertIn(str(NATIVE_CAP), prompt)
        self.assertNotIn("Partial execution", prompt)
        self.assertNotIn("secret-trace-token", prompt)
        self.assertNotIn("Emit only the program's stdout", prompt)
        self.assertFalse(prompt_contains_gold(prompt, example.gold_stdout))
        self.assertNotIn("expected", prompt.lower())
        from trace2cache.execution_sim import prompt_introduces_gold

        self.assertFalse(prompt_introduces_gold(prompt, example))
        overlapping = SimExample(
            example_id="sim_overlap",
            task_id=11,
            source=example.source,
            stdin="hello-output and more\n",
            gold_stdout="hello-output\n",
            events=example.events,
            event_count=example.event_count,
            source_hash="abc",
        )
        overlap_prompt = render_ask_events_prompt(overlapping)
        self.assertTrue(prompt_contains_gold(overlap_prompt, overlapping.gold_stdout))
        self.assertFalse(prompt_introduces_gold(overlap_prompt, overlapping))
        injected = overlap_prompt + "\nsecret-gold-stdout\n"
        leaked = SimExample(
            example_id="sim_leak",
            task_id=11,
            source=example.source,
            stdin="3\n",
            gold_stdout="secret-gold-stdout\n",
            events=example.events,
            event_count=example.event_count,
            source_hash="abc",
        )
        self.assertTrue(prompt_introduces_gold(injected, leaked))

    def test_ask_events_scores_last_fence_and_counts_prefix_roles(self):
        response = "LINE: n=3\nSTATE: n=3\n```\n6\n```\n"
        self.assertEqual(extract_output(response).strip(), "6")
        prefix = response_event_prefix(response)
        self.assertEqual(count_event_role_lines(prefix), 2)
        self.assertNotIn("6", prefix)

    def test_ask_events_kill_rule(self):
        def rows(ask_pass, code_pass, gold_pass, task_id=1):
            return [
                {"task_id": task_id, "condition": "code_input", "passed": code_pass},
                {"task_id": task_id, "condition": "trace_text", "passed": gold_pass},
                {
                    "task_id": task_id,
                    "condition": ASK_EVENTS,
                    "passed": ask_pass,
                    "event_lines": 3,
                },
            ]

        inference = []
        for i in range(12):
            inference.extend(rows(True, False, True, i))
        for i in range(12, 20):
            inference.extend(rows(True, True, True, i))
        summary = summarize_ask_events_gate0(inference)
        self.assertEqual(summary["call"], "gate0_pass_inference")
        self.assertEqual(summary["correct"][ASK_EVENTS], 20)

        sft = []
        for i in range(12):
            sft.extend(rows(False, False, True, i))
        for i in range(12, 20):
            sft.extend(rows(True, True, True, i))
        self.assertEqual(summarize_ask_events_gate0(sft)["call"], "gate0_sft_justified")

        fail = []
        for i in range(20):
            fail.extend(rows(False, True, False, i))
        self.assertEqual(summarize_ask_events_gate0(fail)["call"], "gate0_fail")

    def test_gist_keeps_last_runtime_roles_only(self):
        events = (
            EvidenceEvent(0, CALL, "call first"),
            EvidenceEvent(0, LINE, "line first"),
            EvidenceEvent(0, LINE, "line last"),
            EvidenceEvent(0, STATE, "state last"),
        )
        gist = gist_intermediate(events)
        self.assertEqual([event.content for event in gist], ["call first", "line last", "state last"])

    def test_example_from_repair_traces_gold_and_hides_stdout(self):
        bug = {
            "bug_id": 1,
            "problem_id": "toy",
            "buggy_code": "n = int(input())\nprint(n + 2)\n",
            "fixed_code": "n = int(input())\nprint(n * 2)\n",
            "split": "test",
            "tests": [("2\n", "4\n"), ("5\n", "10\n"), ("0\n", "0\n")],
            "description": "double",
        }
        repair, audit = example_from_runbugrun(bug, max_events_per_test=16)
        self.assertTrue(audit.get("selected"), audit)
        sim = example_from_repair(repair, max_events_per_test=16)
        self.assertIsNotNone(sim)
        assert sim is not None
        self.assertEqual(normalize_output(sim.gold_stdout), "10")
        self.assertEqual(sim.source, bug["fixed_code"])
        prompt = render_user_prompt(sim, evidence_for_condition(sim, "trace_text"))
        self.assertFalse(prompt_contains_gold(prompt, sim.gold_stdout))
        self.assertTrue(all(event.role_id not in {TEST, FAIL, RETURN} for event in sim.events))
        self.assertIn("n = int(input())", prompt)

    def test_example_from_stdio_keeps_runtime_and_latent_uses_marker(self):
        sim = example_from_stdio(
            "value = int(input())\nprint(value * 2)\n",
            "7\n",
            "14\n",
            task_id=99,
            example_id="sim_stdio_toy",
            max_events_per_test=16,
        )
        self.assertIsNotNone(sim)
        assert sim is not None
        self.assertEqual(normalize_output(sim.gold_stdout), "14")
        self.assertTrue(sim.events)
        self.assertTrue(all(event.role_id in RUNTIME_KEEP for event in sim.events))
        self.assertEqual(len(SIM_SLOT_ROLES), 8)
        self.assertTrue(all(tuple(sorted(RUNTIME_KEEP)) == roles for roles in SIM_SLOT_ROLES))
        latent_prompt = render_user_prompt(sim, evidence_for_condition(sim, "true_latent"))
        self.assertEqual(LATENT_CONDITIONS, ("true_latent", "shuffled_latent"))
        self.assertIn(MARKER, latent_prompt)
        self.assertNotIn(f"```\n{MARKER}", latent_prompt)
        self.assertFalse(prompt_contains_gold(latent_prompt, sim.gold_stdout))
        text_prompt = render_user_prompt(sim, evidence_for_condition(sim, "trace_text"))
        self.assertIn("Partial execution", text_prompt)
        self.assertNotIn(MARKER, text_prompt)

    def test_native_event_texts_match_compact_lines_and_use_marker(self):
        events = (
            EvidenceEvent(0, CALL, "call double(3)"),
            EvidenceEvent(0, LINE, "line source print(n * 2) state n=3"),
            EvidenceEvent(0, STATE, "line source print(n * 2) state n=3 out=6"),
        )
        example = self._example(events)
        compact_lines = [
            line
            for line in compact_trace_text(events).splitlines()
            if not line.startswith("TEST ") and line != "END_TEST"
        ]
        self.assertEqual(list(native_event_texts(events)), compact_lines)
        self.assertEqual(len(native_event_texts(events, cap=2)), 2)
        with self.assertRaises(ValueError):
            native_event_texts(events, cap=0)
        self.assertEqual(NATIVE_CONDITIONS, ("native_events", "shuffled_native"))
        self.assertEqual(NATIVE_CAP, 24)
        prompt = render_user_prompt(example, evidence_for_condition(example, "native_events"))
        self.assertIn(MARKER, prompt)
        self.assertNotIn(f"```\n{MARKER}", prompt)
        self.assertIn("n = int(input())", prompt)
        self.assertIn("3", prompt)
        shuffled_prompt = render_user_prompt(
            example, evidence_for_condition(example, "shuffled_native")
        )
        self.assertEqual(prompt, shuffled_prompt)
        other = SimExample(
            example_id="sim_other",
            task_id=12,
            source="print(1)\n",
            stdin="",
            gold_stdout="1\n",
            events=(EvidenceEvent(0, STATE, "other-state"),),
            event_count=1,
            source_hash="def",
        )
        picked = different_task_example([example, other], 0)
        self.assertEqual(picked.task_id, 12)
        self.assertNotEqual(native_event_texts(example.events), native_event_texts(picked.events))

    def test_vocab_snap_picks_nearest_row_and_uses_marker(self):
        import torch

        vocab = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]],
            dtype=torch.float32,
        )
        table = torch.tensor([[0.95, 0.05], [0.1, 0.9]], dtype=torch.float32)
        snapped, index, cosine = snap_table_to_vocab(table, vocab)
        self.assertEqual(index.tolist(), [0, 1])
        torch.testing.assert_close(snapped, vocab[index])
        self.assertTrue(torch.all(cosine > 0.9))
        identity, identity_index, identity_cosine = snap_table_to_vocab(vocab, vocab)
        self.assertEqual(identity_index.tolist(), [0, 1, 2])
        torch.testing.assert_close(identity, vocab)
        self.assertTrue(torch.all(identity_cosine > 0.999))
        with self.assertRaises(ValueError):
            snap_table_to_vocab(table, vocab, batch_size=0)
        example = self._example((EvidenceEvent(0, STATE, "state n=3"),))
        self.assertEqual(SNAP_CONDITIONS, ("vocab_snap", "shuffled_snap"))
        prompt = render_user_prompt(example, evidence_for_condition(example, "vocab_snap"))
        self.assertIn(MARKER, prompt)
        self.assertNotIn(f"```\n{MARKER}", prompt)
        self.assertIn("n = int(input())", prompt)
        self.assertIn("3", prompt)
        self.assertEqual(
            prompt,
            render_user_prompt(example, evidence_for_condition(example, "shuffled_snap")),
        )

    def test_compact_ask_events_drop_builtins_and_keep_user_state(self):
        call = EvidenceEvent(
            0,
            CALL,
            "call <module> line 0 source  state __builtins__={'__name__': 'builtins'}, __name__='__main__'",
        )
        state = EvidenceEvent(
            0,
            STATE,
            'line <module> line 3 source print(i,"x",j) state __builtins__={"x": 1}, __name__=\'__main__\', i=1, j=3',
        )
        branch = EvidenceEvent(
            0,
            BRANCH,
            "line <module> line 2 source for i in range(2): state __name__='__main__', i=0",
        )
        hidden = EvidenceEvent(0, RETURN, "return <module> line 4 source print(1) return=None")
        self.assertEqual(compact_ask_event_line(call), "CALL: <module>")
        self.assertEqual(compact_ask_event_line(state), "STATE: i=1, j=3")
        self.assertEqual(compact_ask_event_line(branch), "BRANCH: for i in range(2):")
        truncated = EvidenceEvent(
            0,
            STATE,
            "line <module> line 3 source print(i, j) state __builtins__={'__doc__': \"Built-in functions, types... provide..., __name__='__main__', i=1, j=3",
        )
        self.assertEqual(compact_ask_event_line(truncated), "STATE: i=1, j=3")
        self.assertNotIn("__builtins__", compact_ask_event_line(truncated) or "")
        self.assertIsNone(compact_ask_event_line(hidden))
        example = self._example((call, state, branch, hidden, state))
        text = gold_ask_events_text(example, cap=2)
        self.assertEqual(text, "CALL: <module>\nSTATE: i=1, j=3")
        self.assertNotIn("__builtins__", text)
        self.assertNotIn("__name__", text)
        with self.assertRaises(ValueError):
            gold_ask_events_text(example, cap=0)

    def test_sft_targets_match_ask_events_format(self):
        events = (
            EvidenceEvent(
                0,
                LINE,
                "line <module> line 2 source print(n * 2) state n=3",
            ),
            EvidenceEvent(
                0,
                STATE,
                "line <module> line 2 source print(n * 2) state n=3, out=6",
            ),
        )
        example = self._example(events)
        stdout = sft_stdout_target(example)
        chain = sft_events_target(example)
        self.assertEqual(stdout, "```\n6\n```")
        self.assertNotIn("CALL:", stdout)
        self.assertIn("LINE: print(n * 2)", chain)
        self.assertIn("STATE: n=3, out=6", chain)
        self.assertTrue(chain.endswith("```\n6\n```"))
        self.assertGreaterEqual(chain.count("```"), 4)
        self.assertEqual(extract_output(chain).strip(), "6")
        prefix = response_event_prefix(chain)
        self.assertEqual(count_event_role_lines(prefix), 2)
        self.assertEqual(sft_target(example, SFT_STDOUT), stdout)
        self.assertEqual(sft_target(example, SFT_EVENTS), chain)
        with self.assertRaises(ValueError):
            sft_target(example, "nope")
        long_gold = SimExample(
            example_id="sim_toy",
            task_id=11,
            source=example.source,
            stdin=example.stdin,
            gold_stdout="hello world\n",
            events=(
                EvidenceEvent(
                    0,
                    STATE,
                    "line <module> line 1 source print(msg) state msg=hello world",
                ),
            ),
            event_count=1,
            source_hash="abc",
        )
        redacted = sft_events_target(long_gold)
        self.assertIn("[REDACTED]", redacted)
        self.assertNotIn("hello world", response_event_prefix(redacted))

    def test_event_sft_kill_rule_and_disjoint_train(self):
        def rows(events_pass, stdout_pass, task_id=1):
            return [
                {"task_id": task_id, "condition": SFT_EVENTS, "passed": events_pass, "event_lines": 4},
                {"task_id": task_id, "condition": SFT_STDOUT, "passed": stdout_pass, "event_lines": 0},
            ]

        win = []
        for i in range(12):
            win.extend(rows(True, False, i))
        for i in range(12, 20):
            win.extend(rows(True, True, i))
        summary = summarize_event_sft(win)
        self.assertEqual(summary["call"], "sft_pass")
        self.assertEqual(summary["correct"][SFT_EVENTS], 20)
        self.assertEqual(summary["events_vs_stdout"]["left_only"], 12)

        fail = []
        for i in range(20):
            fail.extend(rows(False, True, i))
        self.assertEqual(summarize_event_sft(fail)["call"], "sft_fail")
        with self.assertRaises(ValueError):
            summarize_event_sft([{"task_id": 1, "condition": SFT_EVENTS, "passed": True}])

        train = [self._example()]
        other = SimExample(
            example_id="sim_other",
            task_id=12,
            source="print(1)\n",
            stdin="",
            gold_stdout="1\n",
            events=(),
            event_count=0,
            source_hash="def",
        )
        assert_train_disjoint(train, [other])
        with self.assertRaises(ValueError):
            assert_train_disjoint(train, train)
