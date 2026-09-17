from __future__ import annotations

import unittest

from trace2cache.paired_evidence import PRIMARY_DISTRIBUTIONS, audit_records, generate_split


class PairedEvidenceTest(unittest.TestCase):
    def test_deterministic_matched_pairs_pass_audit(self) -> None:
        first, _ = generate_split("train", 2, seed=401, distribution=PRIMARY_DISTRIBUTIONS["train"])
        second, _ = generate_split("train", 2, seed=401, distribution=PRIMARY_DISTRIBUTIONS["train"])
        self.assertEqual(first, second)
        report = audit_records(first)
        self.assertTrue(report["passed"], report["errors"])
        self.assertEqual(len(first), 24)

    def test_splits_do_not_reuse_input_bundles(self) -> None:
        used: set[str] = set()
        train, _ = generate_split("train", 2, seed=401, used_hashes=used)
        dev, _ = generate_split("dev", 2, seed=1410, used_hashes=used)
        self.assertTrue({row.input_hash for row in train}.isdisjoint({row.input_hash for row in dev}))

    def test_views_only_differ_at_expected_or_status(self) -> None:
        rows, _ = generate_split("train", 1, seed=401)
        for row in rows:
            self.assertNotEqual(row.label_a, row.label_b)
            for left, right in zip(row.evidence_a.events, row.evidence_b.events):
                if left.role_id not in (8, 9): self.assertEqual(left, right)
