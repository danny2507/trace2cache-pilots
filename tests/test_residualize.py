from __future__ import annotations

import re
import unittest

from trace2cache.mbpp import MBPPTask
from trace2cache.residualize import (
    GENERIC,
    TABLE,
    UNPARSED,
    classify_residual,
    extract_generation,
    fisher_exact_two_sided,
    generation_prompt,
    prompt_contains_hidden,
    split_public_hidden,
    summarize_gate0,
)


class ResidualizeTests(unittest.TestCase):
    def test_public_is_the_first_official_assertion(self):
        tests = ("assert inc(0) == 1", "assert inc(2) == 3", "assert inc(10) == 11")
        public, hidden = split_public_hidden(tests)
        self.assertEqual(public, ("assert inc(0) == 1",))
        self.assertEqual(hidden, ("assert inc(2) == 3", "assert inc(10) == 11"))
        with self.assertRaises(ValueError):
            split_public_hidden(("assert inc(0) == 1",))

    def test_generation_prompt_hides_hidden_tests_and_canonical(self):
        public, hidden = split_public_hidden(
            ("assert inc(0) == 1", "assert inc(2) == 3", "assert inc(10) == 11")
        )
        prompt = generation_prompt("Increment x.", public)
        self.assertIn("assert inc(0) == 1", prompt)
        self.assertIn("Specification:", prompt)
        self.assertFalse(prompt_contains_hidden(prompt, hidden))
        self.assertNotIn("return x + 1", prompt)

    def test_anonymize_strips_descriptive_entry_names(self):
        from trace2cache.residualize import ANON_ENTRY, anonymize_tests, entry_from_tests

        tests = ("assert inc(0) == 1", "assert inc(2) == 3")
        self.assertEqual(entry_from_tests(tests), "inc")
        anon = anonymize_tests(tests, "inc")
        self.assertEqual(anon, ("assert f(0) == 1", "assert f(2) == 3"))
        prompt = generation_prompt("Increment x.", (anon[0],), include_specification=False)
        self.assertIn("assert f(0) == 1", prompt)
        self.assertIsNone(re.search(r"\binc\b", prompt))
        self.assertNotIn("Increment", prompt)
        self.assertEqual(ANON_ENTRY, "f")

    def test_nospec_prompt_omits_the_specification(self):
        from trace2cache.residualize import prompt_contains_specification

        public, hidden = split_public_hidden(
            ("assert inc(0) == 1", "assert inc(2) == 3", "assert inc(10) == 11")
        )
        spec = "Write a function to increment a number by one."
        prompt = generation_prompt(spec, public, include_specification=False)
        self.assertIn("assert inc(0) == 1", prompt)
        self.assertNotIn("Specification:", prompt)
        self.assertFalse(prompt_contains_hidden(prompt, hidden))
        self.assertFalse(prompt_contains_specification(prompt, spec, public))
        self.assertTrue(
            prompt_contains_specification(
                generation_prompt(spec, public, include_specification=True), spec, public
            )
        )

    def test_extracts_fenced_python(self):
        source = extract_generation("Sure.\n```python\ndef inc(x):\n    return x + 1\n```\n")
        self.assertIn("def inc", source)
        self.assertIn("return x + 1", source)

    def test_n_plus_one_is_generic(self):
        report = classify_residual("def inc(x):\n    return x + 1\n", ("assert inc(0) == 1",))
        self.assertEqual(report.kind, GENERIC)
        self.assertGreater(report.generic_uses, 0)

    def test_computed_equality_is_generic_not_a_table(self):
        parity = classify_residual(
            "def f(n):\n    return n % 2 == 1\n",
            ("assert f(1) == True",),
        )
        even = classify_residual(
            "def f(x):\n    if x % 2 == 0:\n        return 2\n    return -1\n",
            ("assert f(2) == 2",),
        )
        self.assertEqual(parity.kind, GENERIC, parity)
        self.assertEqual(even.kind, GENERIC, even)

    def test_factorial_and_loop_are_generic(self):
        factorial = classify_residual(
            "def fact(n):\n    return 1 if n <= 1 else n * fact(n - 1)\n",
            ("assert fact(3) == 6",),
        )
        loop = classify_residual(
            "def fact(n):\n    p = 1\n    for i in range(1, n + 1):\n        p *= i\n    return p\n",
            ("assert fact(3) == 6",),
        )
        self.assertEqual(factorial.kind, GENERIC)
        self.assertEqual(loop.kind, GENERIC)

    def test_if_chain_and_dict_lookup_are_tables(self):
        chain = classify_residual(
            "def f(n):\n    if n == 1:\n        return 2\n    return 0\n",
            ("assert f(1) == 2",),
        )
        lookup = classify_residual(
            "def f(n):\n    return {1: 2, 3: 4}[n]\n",
            ("assert f(1) == 2",),
        )
        constant = classify_residual(
            "def f(n):\n    return 2\n",
            ("assert f(1) == 2",),
        )
        membership = classify_residual(
            "def f(n):\n    if n in (1, 3):\n        return 2\n    return 0\n",
            ("assert f(1) == 2",),
        )
        self.assertEqual(chain.kind, TABLE, chain)
        self.assertEqual(lookup.kind, TABLE, lookup)
        self.assertEqual(constant.kind, TABLE, constant)
        self.assertEqual(membership.kind, TABLE, membership)
        self.assertGreater(chain.key_uses, 0)
        self.assertGreater(lookup.key_uses, 0)

    def test_aliased_key_use_stays_a_table(self):
        report = classify_residual(
            "def f(n):\n    x = n\n    if x == 1:\n        return 2\n    return 0\n",
            ("assert f(1) == 2",),
        )
        self.assertEqual(report.kind, TABLE, report)

    def test_identity_and_index_reads_are_generic(self):
        identity = classify_residual("def f(n):\n    return n\n", ("assert f(1) == 1",))
        index = classify_residual(
            "def f(xs, i):\n    return xs[i]\n",
            ("assert f([1, 2], 0) == 1",),
        )
        self.assertEqual(identity.kind, GENERIC, identity)
        self.assertEqual(index.kind, GENERIC, index)

    def test_unparsed_source_is_unparsed(self):
        report = classify_residual("def f(n)\n    return n\n")
        self.assertEqual(report.kind, UNPARSED)

    def test_fisher_detects_association(self):
        self.assertLess(fisher_exact_two_sided(2, 18, 15, 5), 0.05)
        self.assertGreater(fisher_exact_two_sided(5, 5, 5, 5), 0.9)

    def test_gate0_kill_rule(self):
        tables = [
            {"public_pass": True, "hidden_pass": False, "kind": TABLE} for _ in range(12)
        ] + [{"public_pass": True, "hidden_pass": True, "kind": TABLE} for _ in range(2)]
        generics = [
            {"public_pass": True, "hidden_pass": True, "kind": GENERIC} for _ in range(12)
        ] + [{"public_pass": True, "hidden_pass": False, "kind": GENERIC} for _ in range(2)]
        passed = summarize_gate0(tables + generics)
        self.assertEqual(passed["call"], "gate0_pass")
        equal = summarize_gate0(
            [{"public_pass": True, "hidden_pass": True, "kind": TABLE} for _ in range(12)]
            + [{"public_pass": True, "hidden_pass": True, "kind": GENERIC} for _ in range(12)]
        )
        self.assertEqual(equal["call"], "gate0_fail")
        small = summarize_gate0(
            [{"public_pass": True, "hidden_pass": False, "kind": TABLE} for _ in range(3)]
            + [{"public_pass": True, "hidden_pass": True, "kind": GENERIC} for _ in range(12)]
        )
        self.assertEqual(small["call"], "gate0_insufficient")

    def test_task_dataclass_still_round_trips(self):
        task = MBPPTask(
            task_id=601,
            description="Increment x.",
            canonical_source="def inc(x):\n    return x + 1\n",
            tests=("assert inc(0) == 1", "assert inc(2) == 3"),
        )
        self.assertTrue(len(task.tests) >= 2)
        self.assertEqual(classify_residual(task.canonical_source, task.tests[:1]).kind, GENERIC)


if __name__ == "__main__":
    unittest.main()
